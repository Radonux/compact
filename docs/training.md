# Data, training and scoring

The commands that make step data, train an estimator, train a controller
in the closed loop, and run and score a policy. Terms are in
[methodology.md](methodology.md#terms). Training needs
`uv sync --extra train`. Large runs are experiment files in GitHub
Actions ([architecture.md](architecture.md#experiments-in-github-actions)).

Which command serves which step of the [plan](methodology.md#plan):

| Step | Command |
|---|---|
| The discovery controller, and the retrained net2 | `train ppo` |
| The retrained net1 | `train run` |
| Grounding | `experiments/ground.py` on step data from the controllers' own rollouts; its library holds every stored label and composite |

## Step data

A campaign records steps when it sets `"steps": true`. An episode that
runs a policy always records them. Each shard then has `steps.parquet`:
per 10 ms step, the 78-value message (`obs`), the labels at that instant
(`label_*`), the action taken and whether a policy set it.

    {"name": "steps", "duration_ms": 20000, "seed": 9, "steps": true,
     "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr3"}],
     "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
               "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
     "conditions": [{}, {"loss": {"downlink": 0.01}}]}

    uv run expliccit campaign plan steps.json out/steps
    uv run expliccit campaign run out/steps --build build/uml --workers 4 --tables steps

`--tables steps` drops the large link tables. `--steps-stride K` keeps
every K-th step, which is enough to supervise an estimator and not enough
for a learner that needs consecutive steps. A sweep
`{"kind": "paths", "paths": [...]}` takes any paths, and
`"ccs_per_point": k` runs k of the controllers per point. Many links with
a few controllers each teach an estimator more than the reverse. Any of
the 17 Linux controllers can generate data (`StockName` in `spec.py`).

| Link set | Made by | Notes |
|---|---|---|
| 120 paths, BBRv1, Cubic and Vegas, 30 s | `examples/campaign-expliccit-teachers.json` | expliCCit's own setup |
| 180 links with every controller, 1,050 with three each | `examples/net1_varied.py` | three conditions; its plan also defines validation and test links |
| 3,600 broad links under 64 conditions, and four held-out families | `examples/net1_broad.py` | needs `--models DIR` holding `rule-u0.25.onnx`, a fixed-rule policy some episodes run (export one with `experiments/rule_export.py`) |

## The estimator: `train run`

    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`, the
reference for every field). Only `data` is required.
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
| `net1.lcfs` | expliCCit's 7 | what net1 predicts: any stored label or composite, in either window (`lcfs.LCF_UNITS`; [architecture.md](architecture.md#ground-truth-labels)) |
| `net1.obs_cols` | expliCCit's 70 | the message columns net1 reads |
| `net1.output` | `{"kind": "point"}` | what net1 states per label: one value (`point`), a centre and an interval holding `coverage` (`interval`), or a distribution over `bins` fixed bins, trained with the log score and averaged over `members` net1s (`bins`) |
| `net1.width`, `dropout`, `lr`, `epochs`, `patience` | 512, 0.3, 1e-3, 50, 5 | the network and its training |

Output: `net1.pt`, `net2.pt`, `policy.onnx`, `report.json` and the spec.
Per label on the test split, `report.json` gives R^2 and the mean
absolute error of the belief's mean, CRPS skill and log-score bits
against a forecast that ignores the measurements, coverage at 50% to 95%,
and the 90% width, also on busy and idle steps.

`train run` also fits a memoryless net2 to the logged steps (the `net2`
fields of `TrainSpec`: `algo`, `inputs`, `action`, `reward`). That net2
is fitted to a log and is not judged in the closed loop, so it is no
evidence about a controller. compact's controllers come from `train ppo`.

## A controller in the closed loop: `train ppo`

    uv run python experiments/net2_ppo_setup.py out/setup --net1 NET1.pt --models MODELS --u 0.25
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

The setup writes what every run shares: a fixed rule (a start and a
baseline), the training paths (600 broad links, or 9,600 with
`--paths net2-train-xl`; MODELS is a directory holding
`rule-u0.25.onnx`), a validation set scored during training, and
`ppo-base.json`, which each run edits.

`train ppo` takes a `PPOSpec` (`train/spec.py`, the reference for every
field). Each iteration runs the current controller in the emulator and
updates it from those rollouts. The actor is a GRU that decides once per
min RTT (clamped to 10 to 50 ms) and holds its decision u in between.

| Field | Default | Meaning |
|---|---|---|
| `algo` | `ppo` | `ppo` learns from the reward. `clone` is the bootstrap: a Linux controller drives the rollouts and the controller learns the window that sends as it did. `dagger` is imitation in which the controller acts and a fixed rule labels each decision. `td3bc` learns from a replay of recent iterations |
| `teacher_cc` | none | `clone` only: the Linux controller to clone, for example `cubic` or `bbr3` |
| `inputs` | `net1` | what the actor reads: net1's beliefs, the `truth` (an oracle), or `raw` measurements through net1's input scaler |
| `latents` | none | `raw` only. The discovery controller: an encoder maps each step's scaled measurements to this many latents, and the actor reads the latents and its previous u alone |
| `encoder_hidden` | 256 | the width of the encoder's two hidden layers |
| `action_max` | 3 | the bound on u; 8 for the discovery controller |
| `init_actor`, `init_critic`, `init_from` | none | start from saved networks, or continue an earlier run's directory |
| `anchor` | 0 | `ppo`: weight of the KL divergence to the policy it started from |
| `alone`, `compete`, `copies`, `competition` | 0, 0.5, (2, 3, 4), `at-start` | the mixture of training links: the flow alone, with Linux competitors, or with copies of itself, and when competitors start |
| `iterations`, `links_per_iteration`, `group` | 40, 8, 4 | rollouts per iteration are links times group |
| `noise_sigma`, `noise_hold_steps`, `tau_s`, `lr` | 0.1, 1, 3.0, 3e-4 | exploration, the reward's time constant, the actor's learning rate |
| `eval_every` | 5 | score the noise-free controller on the validation set every this many iterations |
| `time_budget_s` | none | stop before an iteration would cross it |

What u means depends on `latents`. Without it, u is log2 of the window
over a bandwidth-delay product: net1's believed one, or for `raw` a
measured one (the maximum delivery rate over 2 s times the min RTT). With
it, u is log2 of the window over a fixed 2^7 packets, so no measurement
reaches the window except through the latents
([methodology.md](methodology.md#the-discovery-controller)). A width of
64 stands for no constraint.

Output: `policy.onnx`, `actor.pt`, `critic.pt` and `history.json`, which
holds each iteration's rollouts and diagnostics and each evaluation
against the rule. Read the diagnostics before any closed-loop result.

| Key | Meaning | Warning sign |
|---|---|---|
| `pi_mean`, `pi_std`, `sat` | the actor's output, and its share near the tanh limits | a mean far from the data's, or saturation |
| `q`, `q_pull`, `bc_pull` | `td3bc`: mean Q, and per row the critic's and the data's pull on the action | `q_pull` far above `bc_pull`: the actor follows the critic and not the data |
| `ood` | `td3bc`: share of target actions outside the batch's logged range | high: the critic's targets are extrapolation |

`experiments/net2_ppo_report.py` collects several runs into one summary:
log utility against the rule, per run and evaluation.
`experiments/controller-teachers.json` is a full worked example of the
bootstrap.

A `clone` run's training loss says how well the controller fits its
teacher's rollouts, in which it never drove. Only the evaluations, where
it drives, say how the start behaves. The label is the teacher's window,
or its pacing rate times its smoothed RTT where it paces below its window
as BBR does (`trajectories.paced_window`), because the `agent` controller
paces at cwnd mss / srtt.

## Rules

    uv run python experiments/rule_export.py out/rules NET1.pt --u 0.25 [--oracle]

A fixed rule sets the window to 2^u times net1's believed
bandwidth-delay product. `--oracle` exports the same rule reading the
live truth.

## Running and scoring a policy

Any exported graph runs as an episode's controller, observed or as a
cross flow:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

In `hold` mode the window and pacing stay as the policy set them. `reno`
lets Reno grow the window between actions.

    experiments/rule_eval.sh out/val val 11 11 bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/rule_report.py out/val

`rule_eval.sh` plans families of evaluation conditions on the validation
or test links (`experiments/heldout.py` writes any one). Run each `c-*`
campaign with `campaign run`. On any campaign with step tables,
`expliccit score DIRS --reference bbr3` prints log utility per
controller, link and condition, paired by link. It refuses incomplete
shards.

## Oracles

A policy can read the path's true state at each step through the
contract's `truth` input: every stored label but `queue_delay_ns`, in
both windows, and any composite of them. The values equal the labels
stored with that step. An oracle reads all of its LCFs from it (`inputs: "truth"`, or `--oracle`). A
patched policy takes some from the truth and the rest from net1
(`fit.policy_graph(..., truth=frozenset({"busy_fraction"}))`), which
isolates what one estimate costs.

## expliCCit's released models

expliCCit's released net1 and actor are in
`third_party/expliccit-models`. Its code stays out of this repository
until expliCCit is published.

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit \
        --as-deployed [--truth busy_fraction] [--data out/steps]

`--as-deployed` repeats its deployed controller: beliefs clamped at 0,
the first six rounded, and the action rounded to three decimals. `--data`
reports its net1's error on step data. The pickled scalers are never
loaded, only their parameters.

## GPU

Training runs on CPU by default. On a GPU node, replace the wheel after
syncing and run without syncing again (UNRUN here: no GPU):

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model
