# Training net1 and net2

How to produce training data, train net1 and net2, and run and score a
policy; terms are in [methodology.md](methodology.md#terms), results in
[results.md](results.md). Training needs `uv sync --extra train`. Large
runs are experiment files in GitHub Actions
([architecture.md](architecture.md#experiments-in-github-actions)); those
under `experiments/` are worked examples of every command below.

## Data: teacher campaigns and step data

A campaign records Sage's steps when it sets `"steps": true`; an episode
that runs a policy always records them. Each shard then has
`steps.parquet`: per 10 ms step, the 78-value message (`obs`), the labels
at that instant (`label_*`), Sage's reward from the exact capacity, the
action taken (the next step's column 76) and whether a policy set it.

    {"name": "teachers", "duration_ms": 20000, "seed": 9, "steps": true,
     "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr3"}],
     "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
               "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
     "conditions": [{}, {"loss": {"downlink": 0.01}}]}

    uv run expliccit campaign plan teachers.json out/teachers
    uv run expliccit campaign run out/teachers --build build/uml --workers 4 --tables steps

`--tables steps` drops the large link tables; `--steps-stride K` keeps
every K-th step (enough for net1, not for net2, which needs consecutive
steps). A sweep `{"kind": "paths", "paths": [...]}` takes any paths, and
`"ccs_per_point": k` runs k of the controllers per point: net1 learns
more from many links with a few controllers than the reverse. Any of the
17 Linux controllers can teach (`StockName` in `spec.py`). The datasets:

| Data | Made by | Used for |
|---|---|---|
| expliCCit-like: BBRv1, Cubic and Vegas on 120 paths, 30 s | `examples/campaign-expliccit-teachers.json` | the reproduction, the row split |
| 180 links with every controller, 1,050 with three each | `examples/net1_varied.py` | net1 v3; the evaluation links |
| 3,600 broad links under 64 conditions, and four held-out families | `examples/net1_broad.py`; releases `exp-data-net1-broad-30s-73`, `-60s-74`, `-ood-75` | net1 v4 and v5; net2's training paths |

## net1: `train run`

    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`); only
`data` is required. net1 v5's settings are `experiments/specs/net1v4.json`,
and `experiments/net1_train.py` trains such a net1 and scores it on
held-out families (as `experiments/net1v4-recheck.json` does).

| Field | Default | Meaning |
|---|---|---|
| `data` | | campaign directories, searched for `steps.parquet` |
| `split_by` | `spec` | `link` keeps a link's episodes in one split under every condition; use it |
| `val_fraction`, `test_fraction`, `seed` | 0.1, 0.1, 0 | the split and the initialization |
| `controllers` | all | train only on these controllers' episodes |
| `step_stride` | 1 | keep every n-th step and the step after it |
| `net1_from` | none | reuse a trained `net1.pt` |
| `net1.bins` | none | a distribution over this many bins per LCF, trained with the log score |
| `net1.members` | 1 | an ensemble of this many net1s, their distributions averaged |
| `net1.intervals` | false | instead, a centre and a conformal interval per LCF |
| `net1.lcfs`, `net1.obs_cols` | expliCCit's 7 and 70 | what net1 predicts and reads |
| `net1.width`, `dropout`, `lr`, `epochs`, `patience` | 512, 0.3, 1e-3, 50, 5 | the network and its training |

Output: `net1.pt`, `net2.pt`, `policy.onnx`, `report.json` and the spec.
Per LCF on the test split, `report.json` gives R^2 of the belief's mean,
CRPS skill and log-score bits against a forecast that ignores the
measurements, coverage at 50% to 95%, and the 90% width, also on busy and
idle steps. `fit.quantiles` reads a median and intervals off a belief.

## net2 from step data: `train run` and `train online`

`train run` also trains an offline net2 on net1's beliefs, and `train
online` one in closed loop from a replay buffer. Neither beat the fixed
rule ([results](results.md#learned-net2-has-not-passed-its-gate)), and the
defaults are untuned (TD3+BC on Sage's reward collapsed the window).

| `net2` field | Default | Meaning |
|---|---|---|
| `algo` | `td3bc` | `iql` (implicit Q-learning), `bc` (cloning), `expliccit-v1` (expliCCit's objective, kept for comparison) |
| `inputs` | `net1` | `true_lcfs`: train on the labels; the exported policy reads the live truth (an oracle) |
| `view`, `input_bins` | `bins`, none | `means` gives net2 only each belief's mean (arm A2); `input_bins` pools bins |
| `action` | `multiplier` | `bdp`: net2 outputs u, the window as a log2 multiple of the believed BDP |
| `reward` | `sage` | `utility` or `log_utility` (with `delay_weight` 0.5, `loss_weight` 4) |
| `discount`, `n_step` | 0.95, 1 | multi-step returns let a target's effect over an RTT reach the critic |
| `action_max`, `action_decimals` | 1.0, none | a = action_max tanh(...), rounded in the exported graph |

`train online online.json out/online --build build/uml` takes an
`OnlineSpec` (`train/online.py`): campaigns whose sweeps give the training
and evaluation paths, a frozen `net1_from`, a starting `init_from`, and
per iteration `episodes_per_iteration` rollouts with noise `noise_sigma`
held for `noise_hold_steps` steps, then `updates_per_iteration` updates.

Every learner update reports diagnostics (`report.json`, `history.json`).
Read them before any closed-loop result:

| Key | Meaning | Warning sign |
|---|---|---|
| `q`, `q_pull`, `bc_pull` | TD3+BC: mean Q; per row, the critic's and the data's pull on the action | `q_pull` far above `bc_pull`: the actor follows the critic, not the data |
| `ood` | TD3+BC: share of target actions outside the batch's logged range | high: the critic's targets are extrapolation |
| `pi_mean`, `pi_std`, `sat` | the actor's output, and its share near the tanh limits | a mean far from the data's, or saturation |
| `adv_std`, `w_mean`, `w_capped` | IQL: advantage spread, mean weight, share at the cap | weights near 1: IQL is cloning |

## The rules

    uv run python experiments/rule_export.py out/rules NET1.pt --u 0.25 [--oracle]
    uv run python experiments/share_rule_export.py out/rules NET1.pt --u0 0.25 --k 0.5 [--oracle]

`--oracle` exports the same rule reading the guest's live truth. The share
rule's exporter also takes `--flows N` (and `--used`) for the oracle rules
told the number of flows; `experiments/mode_rule_export.py` exports the
mode rule. Release `model-net1v5` holds net1 v5 (`net1v5.pt`), the fixed
rule (`rule-u0.25.onnx`), its oracle (`rule-u0.25-oracle.onnx`) and the
share rule (`share-u0.25-k0.5.onnx`).

## Learned net2: `train ppo`

    uv run python experiments/net2_ppo_setup.py out/setup --net1 NET1.pt --models MODELS --u 0.25
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

The setup writes what every run shares: the fixed rule (the start and
the baseline), the training paths (600 broad links of their own; MODELS
is release `model-net1v3`, whose rule some of their episodes run), the
validation set scored during training, and `ppo-base.json`, which each
run edits. `train ppo` takes a `PPOSpec` (`train/ppo.py`). The actor is a
GRU that sees net1's pooled beliefs and its own last u, decides once per
min RTT (clamped to 10 to 50 ms) and holds u between decisions while the
window follows each step's believed BDP. The critic also sees the true
LCFs.

| Field | Default | Meaning |
|---|---|---|
| `algo` | `ppo` | `td3bc` (from a replay of recent iterations, weight `alpha`), `dagger` (imitation of a privileged teacher) |
| `student` | `u` | with `dagger`: `u` learns the teacher's u; `gate` learns p, the probability that a stock TCP flow shares the link, and sets u = u_rule + p (share rule's u - u_rule); `gate3` also learns q for copies and cooperates with them |
| `base` | `fixed` | `share`: the actor learns a residual on the share rule (gain `share_k`) |
| `u_rule`, `share_k` | 0.125, 0.5 | the fixed rule's u, and the share rule's gain (the teacher's gain for `dagger`) |
| `gate_threshold`, `gate_threshold_q` | 0, none | act on a belief only past t: p' = clamp((p - t) / (1 - t), 0, 1); `gate3`'s q uses `gate_threshold_q` when set |
| `gate3_copies` | `coop` | `gate3` where it believes copies: `coop` moves toward the cooperative window by q; `rule` lets q veto the push, u = u_rule + p (1 - q) (share - u_rule) |
| `init_actor` | none | start from a saved `actor.pt` |
| `compete`, `copies` | 0.5, (2, 3, 4) | share of training links with competition, and the copies drawn there |
| `copies_reward`, `critic_flows` | `own`, false | `team`: copies learn from their mean utility; the critic also counts the flows on the link |
| `iterations`, `links_per_iteration`, `group` | 40, 8, 4 | rollouts per iteration are links times group (K rollouts per link) |
| `noise_sigma`, `tau_s`, `lr` | 0.1, 3.0, 3e-4 | exploration (log2 of the window), the reward's time constant, the actor's learning rate |
| `updates_per_iteration`, `replay_iterations` | 200, 8 | `td3bc` and `dagger` batches per iteration, and the replay's length |
| `time_budget_s` | none | stop before an iteration would cross it |

Output: `policy.onnx`, `actor.pt`, `critic.pt`, and `history.json` (each
iteration's rollouts and diagnostics, and every `eval_every` iterations
the policy against the rule). The gate experiments
(`experiments/net2-gate-*.json`) apply `experiments/net2_ppo_gate_report.py`;
`gate_threshold_export.py` and `gate_ensemble_export.py` there re-export
gate students at a threshold or as one policy on their mean belief.

## Running and scoring a policy

Any exported graph runs as an episode's controller, observed or as a
cross flow:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

In `hold` mode the window and pacing stay as the policy set them, as in
Sage; `reno` lets Reno grow the window between actions.

    experiments/rule_eval.sh out/val val 11 11 bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/copies.py out/copies --split val --duration-ms 30000 --seed 11 --n 2 4 \
        --stock bbr3 --policy rule=out/rules/rule-u0.25.onnx
    uv run python experiments/rule_report.py out/val

`rule_eval.sh` plans the five families (`experiments/heldout.py` writes
any one, `held-competitors` included); run each `c-*` campaign with
`campaign run`, then `rule_report.py` scores them against BBRv3. On any
campaign with step tables, `expliccit score DIRS --reference bbr3` prints
log utility per controller, link and condition, paired by link; it
refuses incomplete shards. `campaign summary` gives Mahak's metrics.

## Oracle and patched policies

A policy can read the path's true LCFs at each step through the
contract's `truth` input, equal to the labels stored with that step. An
oracle reads all of them (`net2.inputs: "true_lcfs"`, or `--oracle`). A
patch takes some from the truth and the rest from net1, to test whether a
wrong belief explains a fault; a patched belief puts all its probability
on the true value's bin (net1's own where the truth is NaN):

    from pathlib import Path
    import torch
    from expliccit_env.train import fit, export, models, spec as ts

    run = Path("out/model")
    spec = ts.TrainSpec.parse((run / "train_spec.json").read_text())
    net1, _ = fit.load_net1(run / "net1.pt")
    actor = models.NET2[spec.net2.arch](net1.width, spec.net2.hidden)
    actor.load_state_dict(torch.load(run / "net2.pt", map_location="cpu", weights_only=True))
    graph = fit.policy_graph(net1, actor.eval(), spec.net2.action_max, spec.net2.action_decimals,
                             truth=frozenset({"busy_fraction"}))
    (run / "policy-busy-truth.onnx").write_bytes(export.to_onnx(graph))

## expliCCit's released models

expliCCit's released net1 and actor are in `third_party/expliccit-models`;
its code stays out of this repository until expliCCit is published.

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit \
        --as-deployed [--truth busy_fraction] [--data out/teachers]

`--as-deployed` repeats its deployed controller: beliefs clamped at 0, the
first six rounded, and the action rounded to three decimals. `--truth`
patches the named LCFs, and `--data` reports its net1's error on step
data. The pickled scalers are never loaded, only their parameters.

## GPU

Training runs on CPU by default. On a GPU node, replace the wheel after
syncing and run without syncing again (UNRUN here: no GPU):

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model
