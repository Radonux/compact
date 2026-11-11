# Data, training and scoring

The commands that produce step data, train net1 and net2, and run and
score a policy in the closed loop. Terms are in
[methodology.md](methodology.md#terms). Training needs
`uv sync --extra train`. Large runs are experiment files in GitHub Actions
([architecture.md](architecture.md#experiments-in-github-actions)).

## Step data

A campaign records steps when it sets `"steps": true`; an episode that
runs a policy always records them. Each shard then has `steps.parquet`:
per 10 ms step, the 78-value message (`obs`), the labels at that instant
(`label_*`), the action taken and whether a policy set it.

    {"name": "teachers", "duration_ms": 20000, "seed": 9, "steps": true,
     "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr3"}],
     "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
               "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
     "conditions": [{}, {"loss": {"downlink": 0.01}}]}

    uv run expliccit campaign plan teachers.json out/teachers
    uv run expliccit campaign run out/teachers --build build/uml --workers 4 --tables steps

`--tables steps` drops the large link tables. `--steps-stride K` keeps
every K-th step: enough to supervise an estimator, not enough for a
learner that needs consecutive steps. A sweep `{"kind": "paths", "paths":
[...]}` takes any paths, and `"ccs_per_point": k` runs k of the
controllers per point: many links with a few controllers each teach an
estimator more than the reverse. Any of the 17 Linux controllers can
generate data (`StockName` in `spec.py`).

| Link set | Made by | Notes |
|---|---|---|
| 120 paths, BBRv1, Cubic and Vegas, 30 s | `examples/campaign-expliccit-teachers.json` | expliCCit's own setup |
| 180 links with every controller, 1,050 with three each | `examples/net1_varied.py` | three conditions; its plan also defines validation and test links |
| 3,600 broad links under 64 conditions, and four held-out families | `examples/net1_broad.py` | needs `--models DIR` holding `rule-u0.25.onnx`, a fixed-rule policy some episodes run (export one with `experiments/rule_export.py`) |

The Phase 0 step data is the 30 s training set of the broad links with
every 6th step kept. `experiments/phase0-steps.json` publishes it as a
release of this repository.

## net1: `train run`

    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`, the
reference for every field); only `data` is required.
`experiments/specs/net1v4.json` is a worked set of settings (a
distribution net1, 64 bins, 3 members), and `experiments/net1_train.py`
trains such a net1 and scores it on held-out families.

| Field | Default | Meaning |
|---|---|---|
| `data` | | campaign directories, searched for `steps.parquet` |
| `split_by` | `spec` | `link` keeps a link's episodes in one split under every condition; use it. `row` scatters rows and overstates accuracy |
| `val_fraction`, `test_fraction`, `seed` | 0.1, 0.1, 0 | the split and the initialization |
| `controllers` | all | train only on these controllers' episodes |
| `step_stride` | 1 | keep every n-th step and the step after it |
| `net1_from` | none | reuse a trained `net1.pt` |
| `net1.lcfs` | expliCCit's 7 | the labels net1 predicts, from the names in `Lcf` |
| `net1.obs_cols` | expliCCit's 70 | the message columns net1 reads |
| `net1.output` | `{"kind": "point"}` | what net1 states per label: one value (`point`), a centre and an interval holding `coverage` (`interval`), or a distribution over `bins` fixed bins, trained with the log score and averaged over `members` net1s (`bins`) |
| `net1.width`, `dropout`, `lr`, `epochs`, `patience` | 512, 0.3, 1e-3, 50, 5 | the network and its training |

Output: `net1.pt`, `net2.pt`, `policy.onnx`, `report.json` and the spec.
Per label on the test split, `report.json` gives R^2 and the mean absolute
error of the belief's mean, CRPS skill and log-score bits against a
forecast that ignores the measurements, coverage at 50% to 95%, and the
90% width, also on busy and idle steps.

## net2 from step data: `train run`

`train run` also trains an offline net2 on net1's output: a memoryless
actor asked at every step, on any kind of net1. The defaults are untuned
(TD3+BC on Sage's reward collapsed the window), so tune it before any
claim.

| `net2` field | Default | Meaning |
|---|---|---|
| `algo` | `td3bc` | `iql` (implicit Q-learning), `bc` (cloning), `expliccit-v1` (expliCCit's objective, kept for comparison) |
| `inputs` | `net1` | `truth`: train on the labels; the exported policy reads the live truth (an oracle). `raw`: the measurements net1 reads, without net1 (a black box). `net1+raw`: both |
| `view`, `input_bins` | `bins`, none | `bins` gives net2 all of net1's output; `means` only each belief's mean; `input_bins` pools a distribution's bins |
| `action` | `multiplier` | `bdp`: net2 outputs u, the window as a log2 multiple of the believed BDP (for `raw`, of BBR's estimate) |
| `reward` | `sage` | `utility` or `log_utility` (with `delay_weight` 0.5, `loss_weight` 4) |
| `discount`, `n_step` | 0.95, 1 | multi-step returns let a target's effect over an RTT reach the critic |
| `action_max`, `action_decimals` | 1.0, none | a = action_max tanh(...), rounded in the exported graph |

Every learner update reports diagnostics (`report.json`, `history.json`).
Read them before any closed-loop result:

| Key | Meaning | Warning sign |
|---|---|---|
| `q`, `q_pull`, `bc_pull` | TD3+BC: mean Q; per row, the critic's and the data's pull on the action | `q_pull` far above `bc_pull`: the actor follows the critic, not the data |
| `ood` | TD3+BC: share of target actions outside the batch's logged range | high: the critic's targets are extrapolation |
| `pi_mean`, `pi_std`, `sat` | the actor's output, and its share near the tanh limits | a mean far from the data's, or saturation |
| `adv_std`, `w_mean`, `w_capped` | IQL: advantage spread, mean weight, share at the cap | weights near 1: IQL is cloning |

## net2 in the closed loop: `train ppo`

    uv run python experiments/net2_ppo_setup.py out/setup --net1 NET1.pt --models MODELS --u 0.25
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

The setup writes what every run shares: a fixed rule (the start and the
baseline), the training paths (600 broad links, or 9,600 with `--paths
net2-train-xl`; MODELS is a directory holding `rule-u0.25.onnx`), a
validation set scored during training, and `ppo-base.json`, which each run
edits. `train ppo` takes a `PPOSpec` (`train/spec.py`, the reference for
every field). The actor is a GRU that decides once per min RTT (clamped to
10 to 50 ms) and holds u between decisions. The fields that matter here:

| Field | Default | Meaning |
|---|---|---|
| `algo` | `ppo` | `td3bc` (from a replay of recent iterations), `dagger` (imitation: the student acts, the teacher labels each decision) |
| `inputs` | `net1` | what the actor reads: net1's beliefs, the `truth` (an oracle), or `raw` measurements through net1's input scaler, with the window aimed at a measured BDP (the black box) |
| `student` | `{"kind": "u"}` | what the actor learns; `u` learns u itself. The other kinds in `train/spec.py` learn beliefs about the other flows on the link |
| `teacher_from` | none | `dagger`, student `u`: a run on the true LCFs (`inputs` "truth") as the teacher, in place of a rule |
| `init_actor`, `init_critic`, `init_from` | none | start from saved networks, or continue an earlier run's directory |
| `anchor` | 0 | `ppo`: weight of the KL divergence to the policy it started from |
| `alone`, `compete`, `copies`, `competition` | 0, 0.5, (2, 3, 4), `at-start` | the mixture of training links: the flow alone, with Linux competitors, or with copies of itself, and when competitors start |
| `iterations`, `links_per_iteration`, `group` | 40, 8, 4 | rollouts per iteration are links times group |
| `noise_sigma`, `noise_hold_steps`, `tau_s`, `lr` | 0.1, 1, 3.0, 3e-4 | exploration, the reward's time constant, the actor's learning rate |
| `time_budget_s` | none | stop before an iteration would cross it |

Output: `policy.onnx`, `actor.pt`, `critic.pt` and `history.json` (each
iteration's rollouts and diagnostics, and every `eval_every` iterations
the policy against the rule).

## Rules

    uv run python experiments/rule_export.py out/rules NET1.pt --u 0.25 [--oracle]

A fixed rule sets the window to 2^u times net1's believed BDP; `--oracle`
exports the same rule reading the live truth. The Phase 0 rule teacher
(`teachers/rule-bdp.onnx`) is such a rule with u = 0.25.

## Running and scoring a policy

Any exported graph runs as an episode's controller, observed or as a
cross flow:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

In `hold` mode the window and pacing stay as the policy set them; `reno`
lets Reno grow the window between actions.

    experiments/rule_eval.sh out/val val 11 11 bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/rule_report.py out/val

`rule_eval.sh` plans families of evaluation conditions on the validation
or test links (`experiments/heldout.py` writes any one); run each `c-*`
campaign with `campaign run`. On any campaign with step tables,
`expliccit score DIRS --reference bbr3` prints log utility per controller,
link and condition, paired by link; it refuses incomplete shards.

## Oracles

A policy can read the true values of the seven LCFs at each step through
the contract's `truth` input, equal to the labels stored with that step.
An oracle reads all of them (`net2.inputs: "truth"`, or `--oracle`). A
patched policy takes some from the truth and the rest from net1
(`fit.policy_graph(..., truth=frozenset({"busy_fraction"}))`), which
isolates what one estimate costs.

## expliCCit's released models

expliCCit's released net1 and actor are in `third_party/expliccit-models`;
its code stays out of this repository until expliCCit is published.

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit \
        --as-deployed [--truth busy_fraction] [--data out/teachers]

`--as-deployed` repeats its deployed controller: beliefs clamped at 0, the
first six rounded, and the action rounded to three decimals. `--data`
reports its net1's error on step data. The pickled scalers are never
loaded, only their parameters.

## GPU

Training runs on CPU by default. On a GPU node, replace the wheel after
syncing and run without syncing again (UNRUN here: no GPU):

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model
