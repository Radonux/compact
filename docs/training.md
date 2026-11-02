# Training net1 and net2

This page covers the workflow from episodes to a running policy, and
the results so far. Sections 1 to 6 are the workflow, section 7 has the
results, and section 8 says where the plan goes next. The research goal,
the plan and the current status are in `docs/methodology.md`. The policy
contract and the data layout are in `docs/architecture.md` ("Sage's
step").

## 1. Data: teacher campaigns with steps

A campaign records Sage's steps when it sets `"steps": true`; an agent's
episodes always record them. Each shard then has `steps.parquet`, with
one row per step: the 78-value message (`obs`), the labels at that instant
(`label_*`), Sage's reward, the action taken (the next step's column 76)
and whether the policy set it.

`examples/campaign-expliccit-teachers.json` is the expliCCit-like dataset
behind the results in section 7. It runs BBRv1, Cubic and Vegas (the
teachers expliCCit names) over 120 paths, 30 s each. Capacity alternates
between b and 2b every 7 or 15 s for b = 12, 24 or 48 Mbit/s, or stays
constant at 12 to 96 Mbit/s. One-way delay is 5 to 40 ms and the droptail
buffer holds 50 to 1000 packets.

    uv run expliccit campaign plan examples/campaign-expliccit-teachers.json out/teachers
    uv run expliccit campaign run out/teachers --build build/uml --workers 4 --tables steps

Its 360 episodes took 100 s on 4 cores and give 1.07M steps. With
`--tables steps` the shard is 216 MB.

Any Linux 7.0 congestion control can teach: Cubic, BBRv1 (`bbr`), Reno
and BBRv3 (`bbr3`), plus BIC, CDG, HighSpeed, H-TCP, Hybla, Illinois, LP,
NV, Scalable, Vegas, Veno, Westwood and YeAH. These are built as modules
and loaded only by the episodes that use them. A campaign's `sweep` can be
`{"kind": "paths", "paths": [...]}` for any list of paths, including
`periodic_levels` traces. This one is a grid of Mahak's two-level traces
under two loss conditions:

    {
      "name": "teachers",
      "ccs": [{"kind": "stock", "name": "cubic"}, {"kind": "stock", "name": "bbr"},
              {"kind": "stock", "name": "bbr3"}, {"kind": "stock", "name": "reno"}],
      "sweep": {"kind": "mahak_grid", "bw_mbps": [12, 24, 48], "step": [0.5, 1.0, 2.0],
                "mahak_compat_mrtt_ms": [10, 40], "buffer_pkts": [50, 300]},
      "conditions": [{}, {"loss": {"downlink": 0.01}}],
      "duration_ms": 20000, "seed": 9, "steps": true
    }

It plans 288 episodes. Without the loss condition, its 144 twenty-second
episodes take 30 s on 4 cores and give 283k steps.

`"ccs_per_point": k` runs only k of the controllers on each sweep point
(the same k under every condition), drawn with the campaign's seed. An
estimator learns more from many links with a few controllers each than
from a few links with every controller: with too few links, net1 can
recognize a link (its min RTT nearly names it) and recall its capacity
and buffer. `examples/net1_varied.py` writes net1's data this way: every
controller on 180 links, and 3 of them on each of 1,050 more.

## 2. Train

    uv sync --extra train
    uv run expliccit train run train.json out/model

`train.json` is a `TrainSpec` (`src/expliccit_env/train/spec.py`). Only
`data` is required. In this example the net2 settings are the ones
recorded for the tuned TD3+BC winner (section 7, "Tuned comparison"):

    {"data": ["out/teachers"], "seed": 1, "controllers": ["bbr", "vegas"],
     "net1": {"epochs": 12, "patience": 3},
     "net2": {"algo": "td3bc", "reward": "utility", "discount": 0.99, "action_max": 0.1,
              "alpha": 0.5, "policy_noise": 0.2, "noise_clip": 0.5,
              "batch_size": 256, "steps": 60000}}

The defaults are not the tuned settings. In an untuned run, TD3+BC on
Sage's reward at discount 0.95 (the default algorithm, reward and
discount, with alpha 2.5) collapsed the window: 0.025 utilization on
the 18 test paths of section 7, with a mean action of -0.021 per step
against the teachers' +0.0014. Sage's reward credits lower delay at
once, but credits delivered rate only over a 1 s window, and a discount
of 0.95 (about 200 ms) barely sees that window.

| Field | Default | Meaning |
|---|---|---|
| `seed` | 0 | the split, initialization and batches |
| `split_by` | `spec` | what a path is for the split: everything but controller and seed, or `link` (below) |
| `val_fraction`, `test_fraction` | 0.1, 0.1 | share of paths held out for validation and for test |
| `step_stride` | 1 | keep every n-th step of each episode and the step after it, since net2 learns from consecutive steps |
| `controllers` | all | train only on these controllers' episodes (`["bbr", "vegas"]`) |
| `net1_from` | none | a trained net1 to reuse (a run's `net1.pt`); its saved spec replaces `net1` |
| `net1.arch` | `expliccit-v1` | expliCCit's encoder (1,265,927 parameters at width 512) |
| `net1.obs_cols` | expliCCit's 70 | message columns net1 reads |
| `net1.scaling` | `robust` | `robust` clips to training quantiles, then standardizes; `minmax` is expliCCit's |
| `net1.lcfs` | expliCCit's 7 | `queue_delay_mean_ns, queue_pkts, capacity_bps, free_buffer_pkts, busy_fraction, one_way_delay_ns, receive_rate_bps` |
| `net1.intervals` | false | an interval per LCF (section 5) |
| `net1.bins` | none | a probability distribution over this many bins per LCF, trained with the log score (section 5); excludes `intervals` |
| `net2.algo` | `td3bc` | TD3+BC; `iql` is implicit Q-learning (`expectile`, `beta`); `bc` clones; `expliccit-v1` is expliCCit's objective, kept for comparison |
| `net2.inputs` | `net1` | `true_lcfs` trains on the labels, and the exported policy reads the live truth (an oracle, section 3) |
| `net2.view` | `bins` | distribution net1 only: `bins` gives net2 each LCF's distribution; `means` gives only each distribution's mean, standardized (the single-value arm, same estimator) |
| `net2.input_bins` | none | with `bins`: each LCF's distribution summed into this many groups of adjacent bins |
| `net2.action` | `multiplier` | `multiplier`: net2 outputs a; `bdp`: net2 outputs the next window as a log2 multiple of the believed bandwidth-delay product (below) |
| `net2.reward` | `sage` | Sage's reward; `utility` and `log_utility` are the ground-truth utilities below |
| `net2.discount` | 0.95 | per step |
| `net2.n_step` | 1 | sum the rewards of this many consecutive steps before bootstrapping, so a target's effect over an RTT reaches the critic directly |
| `net2.reward_alignment` | `next` | the reward step t+1 reports; `same` is step t's, as expliCCit and Sage's offline data pair them |
| `net2.action_max` | 1.0 | a = action_max * tanh(...); the critic and the noise settings work in units of it |
| `net2.action_decimals` | none | round a to this many decimals in the exported graph; expliCCit's deployed controller uses 3, a dead band of 0.0005 around no change |
| `net2.steps`, `net2.batch_size` | 20000, 1024 | gradient updates, and rows per update |

The other fields (learning rates, TD3's noise, IQL's expectile, beta and
weight cap, the utility's weights) are in `spec.py`.

`"reward": "utility"` is computed from the ground truth at each step. It
is the receive rate over the capacity, minus `delay_weight` (default 0.5)
times the mean queueing delay over the base RTT, minus `loss_weight`
(default 4) times the flow's dropped fraction. A step with no departures
costs no delay. Every utility on this page uses the default weights.
`"log_utility"` takes the log of the receive rate over the capacity
(floored at 0.01) instead, so flows that share a link settle at
proportionally fair rates (Kelly, 1998), and starving costs far more than
a small queue.

With `"action": "bdp"`, net2's output u sets the next window to
2^u times the believed bandwidth-delay product: net1's mean capacity times
twice its mean one-way delay, in 1500-byte packets. The exported graph
reads the sender's window (the contract's `cwnd` input) and answers
a = log2(BDP) + u - log2(cwnd), so the window lands on the target in one
step whatever it was. For training, the logged u of any step, from any
controller, is log2 of the next step's window over that step's believed
BDP. The precise scale thus comes from net1's full distributions, and net2
learns only the correction, which turns the control problem into one of
choosing a target from beliefs rather than tracking a window through time.

Splits are by path. Episodes that differ only in congestion control or
seed land in the same split, so no path is both trained and tested on.
With `"split_by": "link"`, a path is the bottleneck link alone (capacity
trace, delay, buffer): its episodes under every cross-traffic and loss
condition share one split, so a test link is new to net1. `step_stride`
keeps every n-th step of each episode and the step after it (net2 learns
from consecutive pairs), which bounds memory on large campaigns; steps
10 ms apart are strongly correlated anyway.

Output: `net1.pt`, `net2.pt`, `policy.onnx` (contract `sage-v1`),
`report.json` and the spec. The report has net1's error per LCF on the
test split in physical units, next to classical estimators, and net2's
losses and diagnostics.

On GPU nodes, replace the CPU wheel after syncing, then run without
syncing again:

    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
    uv run --no-sync expliccit train run train.json out/model

The CUDA index above is UNRUN here (no GPU in this environment).

### Tuning

Every learner update returns diagnostics next to its losses (in
`report.json` for `train run`, in `history.json` for `train online`):

| Key | Meaning | Read it as |
|---|---|---|
| `q`, `q_pull`, `bc_pull` | TD3+BC: mean Q of the policy's action; per row, the critic's and the data's pull on that action | `q_pull` far above `bc_pull`: the actor follows the critic, not the data |
| `ood` | TD3+BC: share of target actions outside the batch's logged actions (0.5% to 99.5%) | high: the critic's targets are extrapolation |
| `pi_mean`, `pi_std`, `sat` | the actor's output on the batch, and the share near the tanh limits | a mean far from the data's, or saturation, before any closed-loop run |
| `adv_std`, `w_mean`, `w_capped` | IQL: spread of advantages, mean weight, share at `max_weight` | `w_capped` high: beta too large for the reward scale |

The teachers' actions are 0 at 80% of steps and between -0.002 and +0.013
at 90%. TD3's default noise (0.05 exploration, 0.2 target smoothing
clipped at 0.5, all in units of `action_max`) is therefore many times the
data's spread unless `action_max` is small. A bias of 0.001 per step in
the policy's mean action moves the window by 7% a second, so tune on
closed-loop results, not on the regression loss. IQL weights each logged
action by exp(beta x advantage). At the default reward scale (1/25) the
advantages spread over only 0.01 to 0.04, so beta of 10 or less leaves
every weight near 1 and IQL clones. Section 7, "Tuned comparison", has
the search, its ranges and what each knob did.

## 3. Run and evaluate a policy

Use the exported graph as an episode's controller:

    {"kind": "agent", "mode": "hold",
     "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}

`sha256sum out/model/policy.onnx` gives the hash. In `"hold"` mode the
congestion window and pacing rate stay as the policy set them, as in
Sage. Put the controller in an episode spec or in a campaign's `ccs`,
and compare it with stock controllers on paths held out from training.
As an example of the format, this campaign runs BBRv1 and a policy on
expliCCit's case-study link (section 7):

    {"name": "case-study", "duration_ms": 45000, "seed": 7,
     "ccs": [{"kind": "stock", "name": "bbr"},
             {"kind": "agent", "mode": "hold",
              "policy": {"path": "out/model/policy.onnx", "sha256": "<sha256 of the file>"}}],
     "sweep": {"kind": "paths", "paths": [
       {"trace": {"kind": "periodic_levels", "levels_mbps": [24, 48], "level_ms": 7000},
        "delay": {"mahak_compat_mrtt_ms": 40}, "buffer": {"packets": 1000}}]}}

    uv run expliccit campaign plan case-study.json out/case-study
    uv run expliccit campaign run out/case-study --build build/uml --workers 4
    uv run expliccit campaign summary out/case-study

`campaign summary` prints, per controller, how many episodes succeeded and
failed. For each metric it prints the mean over paths with a 95%
bootstrap interval over paths: utilization, mean queueing delay
(`mean_delay_ms`), drop fraction, power (`first_power`) and goodput. It
averages only the episodes that succeeded. If any failed, the controllers
are averaged over different paths, so rerun before comparing.

Drop fraction is the share of the flow's packets reaching the bottleneck
that the full queue dropped. Read utilization next to drops: a policy that
holds its window above what the path holds keeps the link full while the
queue drops most of what it sends. `campaign summary` does not report
utility. The utilities in section 7 are the mean over steps of the
per-step utility (section 2, unscaled), computed from the step tables by a
harness kept outside the repository.

### Oracle and patched policies

A policy can read the path's true state at each step through the
contract's `truth` input. It contains the seven LCFs, computed live in the
guest from the bottleneck's records and equal to the labels stored with
the step (`docs/validation.md`, M13). There are two uses.

An oracle (arm A1 in `docs/methodology.md`, section 5) is a net2 trained
with `"net2": {"inputs": "true_lcfs"}`. It trains on the labels, and the
exported `policy.onnx` takes its LCFs from `truth`. Comparing it with the
same net2 on net1's beliefs splits a policy's loss into what the LCF set
costs and what net1's errors cost. For a trained net2 that comparison is
UNRUN; section 7 makes it for a fixed rule, on the true BDP against
net1's ("net2 on the fixed net1").

A patch takes some LCFs from `truth` and the rest from net1, to test
whether a wrong belief explains a fault (expliCCit's repair experiment).
`train import-expliccit ... --truth capacity_bps` patches expliCCit's
released policy (section 4). For a policy trained here, build the graph
in Python (run from the repository with the train extra):

    from pathlib import Path

    import torch

    from expliccit_env.train import fit
    from expliccit_env.train.export import to_onnx
    from expliccit_env.train.models import NET2
    from expliccit_env.train.spec import TrainSpec

    run = Path("out/model")
    spec = TrainSpec.parse((run / "train_spec.json").read_text())
    net1, _ = fit.load_net1(run / "net1.pt")
    actor = NET2[spec.net2.arch](net1.width, spec.net2.hidden)
    actor.load_state_dict(torch.load(run / "net2.pt", map_location="cpu", weights_only=True))
    graph = fit.policy_graph(
        net1, actor.eval(), spec.net2.action_max, spec.net2.action_decimals,
        truth=frozenset({"busy_fraction"}),
    )
    (run / "policy-busy-truth.onnx").write_bytes(to_onnx(graph))

The truth passes through net1's LCF scaler exactly as net2's training
targets did, so a patched belief is what a perfect net1 would have said.
For an interval net1, the centre and both bounds become the truth.

## 4. expliCCit's released models

expliCCit's code is not in this repository and must not be. With a local
copy, its released net1 and actor (`sage_rl/offline/models`) load into our
modules unchanged. The loader reads `best_model.pth`, `actor.pth`, and the
scalers' parameters from `x_scaler_params.json` and
`y_scaler_params.json`. The pickled scikit-learn scalers are never
loaded.

    uv run expliccit train import-expliccit <expliCCit>/sage_rl/offline/models out/expliccit \
        --as-deployed --data out/teachers

`--as-deployed` repeats what its deployed controller does between and
after the networks: LCF beliefs clamped at 0, the first six rounded, and a
rounded to three decimals. `--data` reports its net1's error on our data.
`--truth` takes the named LCFs from the live truth instead of its net1,
for example:

    uv run expliccit train import-expliccit <expliCCit>/sage_rl/offline/models out/expliccit-busy \
        --as-deployed --truth busy_fraction

## 5. Beliefs instead of points

### Distributions

With `"net1": {"bins": 32}`, net1 reports for each LCF a probability
distribution over 32 bins of that quantity, and net2 sees every bin's
probability. The bins are fitted once to the training targets, in the
report units of `LCF_UNITS`. A quantity within [0, 1] gets equal bins.
Any other gets one bin for zero and near-zero values, then log-spaced
bins up to its 99.9th percentile, then a last bin that also takes
everything above. net1 is trained with the log score: the negative log
of the probability it gave the bin that holds the true value, which is
lowest only when the stated probabilities are right.

For each LCF, `report.json` gives:

- accuracy: R^2 and RMSE of the distribution's mean, the estimate squared
  error rewards (comparable with a single-value net1), and the absolute
  error of its median;
- the continuous ranked probability score (CRPS) and its skill against a
  forecast that ignores the measurements. CRPS works on the distribution's
  CDF, so it compares distributions over different bins, and it equals the
  absolute error for a single value; it is what chooses the number of
  bins;
- the log score, next to that of a forecast that ignores the measurements
  (the training rows' bin frequencies), and the difference in bits: how
  much the measurements tell net1;
- coverage of central sets at 50, 80, 90 and 95%. It is tested on bins,
  with the randomized probability integral transform, because net1 states
  probabilities of bins; a value's position inside its bin is not part of
  the claim;
- sharpness: the entropy, and the width of the 90% interval a human reads,
  which is accurate to one bin;
- the log score, 90% coverage and width when the link is kept busy and
  when it is not.

`fit.quantiles` turns a distribution into the readings a human uses
(median, intervals), spreading each bin's mass evenly within it. A truth
patch on a distribution net1 puts all probability on the true value's bin,
and keeps net1's own belief where the truth is NaN.

### Intervals

With `"net1": {"intervals": true, "coverage": 0.9}`, net1 reports a centre
and an interval for each LCF, meant to hold the truth 90% of the time. It
is trained with the pinball loss at the 5th, 50th and 95th percentiles,
the bounds cannot cross, and a conformal correction fitted on the
validation paths sets the coverage. net2 then sees all three numbers, so
it can tell "capacity is about 24 Mbit/s" from "capacity is at least 24
Mbit/s". `report.json` gives each LCF's held-out coverage and mean width.
It also gives the width when the link is kept busy (busy fraction at
least 0.95) and when it is not (below 0.8): the identifiability check of
`docs/methodology.md`, section 2. Results are in section 7. The plan
replaces intervals with distributions (`docs/methodology.md`, section 3).

## 6. Online training

    uv run expliccit train online online.json out/online --build build/uml --workers 32

`online.json` is an `OnlineSpec` (`src/expliccit_env/train/online.py`):

    {"paths_from": "examples/campaign-expliccit-teachers.json",
     "eval_from": "eval.json",
     "net1_from": "out/model/net1.pt", "init_from": "out/bc/net2.pt",
     "net2": {"algo": "iql", "beta": 10, "expectile": 0.9, "reward": "utility",
              "discount": 0.99, "action_max": 0.1, "batch_size": 256},
     "iterations": 8, "episodes_per_iteration": 16, "duration_ms": 20000,
     "noise_sigma": 0.005, "updates_per_iteration": 2000, "eval_every": 4}

These are the tuned settings (section 7, "Tuned comparison").
`init_from` must come from a run with the same `action_max`. `paths_from`
and `eval_from` are campaigns whose `sweep` and `conditions` give the
paths; their `ccs` are ignored. Each rollout draws a path and a condition
(cross traffic, random loss), and the evaluation runs every path under
every condition.

Each iteration exports the policy with net1 frozen and runs
`episodes_per_iteration` episodes on paths drawn from `paths_from`, with
Gaussian noise of `noise_sigma` on each action (seeded per episode), each
draw kept for `noise_hold_steps` steps. With a window target, held noise
holds the window at the perturbed target long enough for the effect to
show. It
then adds their transitions to a replay buffer and runs
`updates_per_iteration` off-policy updates. `history.json` records each
iteration's rollouts and losses and, every `eval_every` iterations, the
noise-free policy on the `eval_from` paths. `offline_data` puts teacher
step data in the replay buffer before the first iteration
(`offline_controllers` filters it), and `critic_warmup` trains the critic
on it first. Online training always feeds net2 net1's beliefs and exports
the policy without the truth input, so an `OnlineSpec` whose
`net2.inputs` is `true_lcfs` is rejected.

The defaults (TD3+BC with alpha 2.5, noise 0.05) are the first, untuned
settings, and under them the policy collapses. In four untuned runs from
cloned policies, utilization on the test paths fell to 0.072 to 0.280.
Noise of 0.05 per 10 ms step on the log window is a random walk of a
factor of about 1.4 over a second, so the rollouts mostly showed what the
noise does. In the search, the collapse stopped at 0.005, and at 0.02 the
policy still backed off.

## 7. Results so far

All results use the expliCCit-like data of section 1 unless stated
otherwise. Utility uses delay weight 0.5 and loss weight 4 (section 2).
Intervals in brackets are 95% bootstraps.

### net1 on held-out paths

Three kinds of net1, each tuned on the validation paths, then trained with
three seeds and scored once on the test paths: a single value per LCF
(squared error), an interval per LCF (section 5), and a distribution per
LCF (section 5).

What was searched (expliCCit's encoder; up to 12 epochs, stopping after 3
without improvement):

- distribution: 16, 32 and 64 bins, each with width 256 or 512, dropout
  0.1 or 0.3, and learning rate 1e-3 or 3e-4; 128 bins with width 256 or
  512, dropout 0.1 and learning rate 1e-3;
- single value and interval: the same widths, dropouts and learning
  rates.

Each kind's settings were chosen by its own validation loss. The number of
bins was chosen by the mean CRPS skill on the validation paths: 0.726 for
16 bins, 0.777 for 32, 0.811 for 64 and 0.822 for 128. The gain per
doubling fell from 0.051 to 0.011; 256 bins are UNRUN. The winners all
use width 512 and learning rate 1e-3, with dropout 0.3 for the single
value and 0.1 for the others; the distribution has 128 bins.

Test paths, means over three seeds:

| LCF | R^2: value, interval, distribution | value's MAE; distribution's CRPS | 90% coverage: interval, distribution | distribution's 90% width, link busy / not busy | information (bits) |
|---|---|---|---|---|---|
| mean queueing delay (ms) | 0.987, 0.997, 0.997 | 6.07; 1.13 | 0.84, 0.89 | 8.4 / 2.9 | 4.8 |
| queue (packets) | 0.987, 0.996, 0.997 | 15.3; 3.40 | 0.89, 0.89 | 21.1 / 11.2 | 3.7 |
| capacity (Mbit/s) | 0.956, 0.967, 0.975 | 2.15; 0.62 | 0.89, 0.90 | 1.4 / 14.9 | 2.0 |
| free buffer (packets) | 0.06, -0.06, -0.00 | 235; 175 | 0.80, 0.73 | 368 / 483 | 1.8 |
| busy fraction | 0.53, 0.67, 0.77 | 0.015; 0.007 | 0.81, 0.90 | 0.017 / 0.34 | 0.56 |
| one-way delay (ms) | 0.993, 0.999, 0.999 | 0.78; 0.12 | 0.89, 0.90 | 0.34 / 1.08 | 2.0 |
| receive rate (Mbit/s) | 0.995, 0.998, 0.998 | 1.38; 0.41 | 0.92, 0.90 | 1.5 / 12.6 | 2.6 |

The distribution's R^2 is that of its mean, the estimate squared error
rewards; for a single value, CRPS equals the absolute error, so the two
error columns are on one scale. "Link busy" means busy fraction at least
0.95, "not busy" below 0.8. Information is the log score gained over a
forecast that ignores the measurements. Across seeds, the standard
deviation of R^2 is at most 0.01, except busy fraction (up to 0.04) and
free buffer (up to 0.10); that of coverage is at most 0.016, except the
interval net1's on mean queueing delay, busy fraction and one-way delay
(0.03 to 0.04).

Reading:

- The distribution is at least as accurate as both other kinds on every
  LCF but free buffer, and its CRPS is 3 to 6 times lower than the single
  value's absolute error.
- It is honest on six LCFs: its 90% statements held 88.7% to 90.2% of
  the time on the test paths, on busy and idle links alike (on idle links
  it is slightly under-confident about busy fraction and receive rate,
  0.95 to 0.96). The interval net1's correction, fitted on the validation
  paths, did not carry over for mean queueing delay (0.84) and busy
  fraction (0.81).
- Free buffer is the exception: no kind predicts it (R^2 near 0), and the
  distribution is over-confident about it (0.73 at 90%). Free buffer
  depends on the buffer size, which these measurements reveal only when
  the queue fills. All three buffer sizes of the grid (50, 300 and 1000
  packets) occur in the training, validation and test paths alike, so
  the cause is not an unseen size. A plausible reading is that each
  training path has one buffer size, so net1 can learn to recognize a
  training path and its buffer, and on a new combination of trace, delay
  and buffer it guesses with confidence it has not earned. The log score
  teaches the uncertainty the measurements leave, not uncertainty about
  unfamiliar paths. "net1 on unseen links" below tests that reading
  with links held out, many more links and an ensemble.
- The widths show the identifiability limit: capacity is known to within
  1.4 Mbit/s (90%) when the link is kept busy and to within 14.9 Mbit/s
  when it is not.

These runs used a harness kept outside the repository; `train run` with
`"net1": {"bins": 128, "width": 512, "dropout": 0.1}` trains the
winner.

expliCCit's released net1 was scored on all 283k steps of the
144-episode campaign of section 1 (without the loss condition), none of
which it saw in training. It had R^2 1.000 on one-way delay, which
confirms that its "mRTT" is the one-way delay, 0.899 on receive rate,
0.847 on mean queueing delay, 0.594 on queue, 0.361 on capacity, -0.743
on busy fraction and -59.0 on free buffer. Its
ground-truth definitions are not in its code, so part of that gap may be
definitional.

### net1 on unseen links

The comparison above split episodes by path, so a test path's link
(capacity trace, delay, buffer) could appear in training under another
condition. Here the split is by link (`"split_by": "link"`; seed 1, 15% of
links each for validation and test), and three controllers never appear in
training: BBRv3, Westwood and the threshold-0.3 policy. Their episodes on
the test links test honesty under controllers net1 has not seen.

Data: the two campaigns of `examples/net1_varied.py`, every 12th step:
every controller on 180 links, and 3 controllers on each of 1,050 more
(20,790 episodes of 20 s; 1.84M training rows).

**Link breadth matters more than tuning.** On the 180-link campaign alone
(132 training links), the validation loss rose from the first epoch on,
and net1 was over-confident about the quantities fixed by the link: its
90% statements held 71% of the time for capacity, 79% for one-way delay
and 69% for free buffer. Dropout 0.3 did not help. With few links, the
min RTT nearly names a link, and net1 can recall that link's capacity and
buffer. Adding the 1,050 links raised those coverages to 0.89, 0.82 and
0.86 with the same network settings (128 bins, dropout 0.1).

What was searched on the validation links (width 512, learning rate 1e-3,
up to 12 epochs, stopping after 3 without improvement): the seven LCFs or
a minimal set (capacity, queue, buffer size, one-way delay, cross-traffic
share), 64 or 128 bins, dropout 0.1 or 0.3; then dropout 0.05 and 0 for
the winner, which sat at 0.1. The seven LCFs won: their mean CRPS skill
was 0.81 against 0.74 for the minimal set, and the minimal set's
validation loss rose from the first epoch, since buffer size and
cross-traffic share are constant over a link or condition and reward
recognizing it (its capacity R^2 was 0.84 against 0.86; buffer size 0.70,
covered 0.81 at 90%). Among the seven-LCF runs, the validation loss chose
64 bins and dropout 0.05. The final net1 is an ensemble of three such
members (seeds 0, 1, 2), whose validation log loss is 1.601 against 1.687
to 1.708 for its members alone.

Test links, scored once:

| LCF | R^2 | CRPS | 90% coverage | 50% coverage | 90% width, link busy / not busy | information (bits) |
|---|---|---|---|---|---|---|
| mean queueing delay (ms) | 0.989 | 1.00 | 0.971 | 0.669 | 14.3 / 2.0 | 3.9 |
| queue (packets) | 0.946 | 7.0 | 0.939 | 0.599 | 69 / 13 | 2.7 |
| capacity (Mbit/s) | 0.882 | 3.47 | 0.888 | 0.515 | 12.3 / 41.2 | 1.9 |
| free buffer (packets) | 0.564 | 39 | 0.922 | 0.581 | 141 / 440 | 1.5 |
| busy fraction | 0.972 | 0.020 | 0.922 | 0.547 | 0.037 / 0.29 | 1.6 |
| one-way delay (ms) | 0.998 | 0.29 | 0.962 | 0.619 | 2.8 / 2.4 | 4.3 |
| receive rate (Mbit/s) | 0.991 | 0.99 | 0.940 | 0.562 | 9.9 / 6.0 | 3.6 |

Held-out controllers on the test links, 90% coverage:

| controller | queueing delay | queue | capacity | free buffer | busy fraction | one-way delay | receive rate |
|---|---|---|---|---|---|---|---|
| BBRv3 | 0.799 | 0.862 | 0.840 | 0.891 | 0.867 | 0.960 | 0.925 |
| Westwood | 0.928 | 0.918 | 0.840 | 0.897 | 0.897 | 0.961 | 0.930 |
| threshold 0.3 | 0.969 | 0.927 | 0.912 | 0.920 | 0.919 | 0.961 | 0.941 |

Reading:

- On new links, net1 is accurate except about free buffer (R^2 0.56),
  and honest or cautious: every LCF's 90% statements held at least 88.8%
  of the time, on busy and idle links alike (0.88 to 0.99).
- The ensemble trades over-confidence for under-confidence. Each member
  alone covered capacity 0.82 to 0.86 and one-way delay 0.90 to 0.93 at
  90%; the ensemble covers 0.89 and 0.96, and its 50% sets hold 52% to
  67%. Its mean coverage error is therefore larger (0.052 against 0.029
  to 0.034), but its CRPS skill is higher (0.833 against 0.818 to 0.824)
  and its log score, which weighs both, is better. Erring wide is the safe
  direction for net2. A calibration step fitted on validation links could
  narrow it; that is UNRUN.
- Unseen controllers cost honesty where their behaviour is new. BBRv3's
  queueing delay is covered 80% of the time and its busy fraction has R^2
  0.69 (0.97 on the training controllers); the threshold policy, which
  holds a full queue, is covered as well as the training controllers.
- Capacity remains the identifiability limit: known to within 12 Mbit/s
  (90%) on a busy link and 41 Mbit/s on an idle one, across links of 4 to
  200 Mbit/s.

### net2 on the fixed net1

net2 acts on the ensemble net1 of "net1 on unseen links", kept fixed. Its
action is the window target (`"action": "bdp"`): the next window is 2^u
times the believed bandwidth-delay product, net1's mean capacity times
twice its mean one-way delay. It sees each LCF's distribution pooled to
32 bins, and is rewarded with the log utility (section 2). Each policy
runs in closed loop on the 21 validation links of the 180-link campaign
under the three conditions (63 episodes of 20 s); the kernel's
controllers are scored on the same links from the data campaign.
Differences are paired by link, with bootstrap 95% intervals over links.

**Learning from other controllers' logs fails.** Cloning all 21
controllers scored -1.61 on the validation links, and IQL -0.97 to -1.60
(beta 3 or 10, expectile 0.7 or 0.9), against -0.48 for BBRv3. The
policies drop 6% to 25% of their packets. In the logs, the state shows
which controller produced it: a full queue comes from a controller that
keeps its window large, so a policy that copies the logs keeps the window
large where the queue is full. net1's belief was not the cause: its
believed BDP had a median error of 0 (log2), the middle 80% of steps
within -0.9 and +0.3.

**A constant multiple of the believed BDP beats every kernel controller on
average.**
Holding u fixed gives on the validation links -0.65 (u = -0.25), -0.45
(0), -0.37 (0.125), -0.37 (0.25), -0.41 (0.375) and -0.49 (0.5). At
u = 0.25 it scores 0.112 (0.054 to 0.172) above BBRv3. It loses almost
only under cross traffic (-0.89, against -0.11 on a clean link and -0.11
under random loss): a Cubic flow fills the buffer, and a window fixed at
1.19 times the BDP gets a small share.

**Online training needs multi-step returns.** From the u = 0.25 policy,
with exploration of 0.3 or 0.5 (log2 of the window) held for 10 or 20
steps, TD3+BC with one-step returns drifted below its start (0.058 to
0.097 lower after 10 iterations), and IQL dropped 17% of its packets
after 5. With 20- or 40-step returns (`net2.n_step`), the policy stays
at the constant's level and learns to raise its target under cross
traffic. After 20 iterations of 32 episodes it scored -0.369 (20 steps)
and -0.360 (40 steps) on the validation links, 0.001 and 0.009 (-0.015 to
+0.034) above the constant: on validation, a tie.

What was searched (validation links): offline cloning and IQL on all
controllers' logs; the constant u above; online TD3+BC (alpha 2.5,
discount 0.97, 512 rows per update, 2,000 updates per iteration) with
one-step returns and exploration 0.3 held 10 steps or 0.5 held 20, then
20- and 40-step returns; online IQL (beta 10, expectile 0.9). The learned
policy with 40-step returns and the constant at u = 0.25 went to test,
with the same rule on the true BDP (arm A1, the guest's live truth) and
on a classical estimate, without net1: the largest delivery rate over
Sage's long window (10 s) times the min RTT, at least 10 packets, the
estimate BBR builds. Its multiple was tuned the same way: on the
validation links it scored -2.59 (u = 0, which never probes), -1.33
(0.125), -0.59 (0.25), -0.60 (0.375), -0.66 (0.5) and -1.12 (1). After
the test, the 40-step run was repeated for 60 iterations, on the
distributions (arm A3) and on their means (arm A2), on the validation
links only (below).

Test links (27 links, three conditions, 81 episodes each, scored once):

| Policy | log utility | utilization | queueing delay (ms) | drops |
|---|---|---|---|---|
| u = 0.25 on the true BDP (oracle) | -0.328 (-0.380 to -0.279) | 0.980 | 14.7 | 0.39% |
| learned net2, 40-step returns (three seeds) | -0.348, -0.349, -0.353 | 0.962 to 0.969 | 11.1 to 12.3 | 0.34% to 0.62% |
| u = 0.25 on net1's BDP | -0.393 (-0.461 to -0.328) | 0.960 | 14.5 | 0.48% |
| u = 0.25 on a classical BDP estimate (no net1) | -0.548 (-0.606 to -0.488) | 0.929 | 18.5 | 2.0% |
| BBRv3 | -0.472 (-0.530 to -0.418) | 0.880 | 15.5 | 0.78% |
| BBRv1 | -0.490 (-0.566 to -0.420) | 0.945 | 17.8 | 3.2% |
| Illinois, the next kernel controller | -0.691 | 0.846 | 23.0 | 0.61% |
| Cubic | -1.212 | 0.699 | 23.3 | 0.09% |
| TD3+BC of "Tuned comparison" (old design) | -1.002 | 0.752 | 17.0 | 5.2% |
| expliCCit's released policy | -1.826 | 0.652 | 18.5 | 31% |

Paired differences: the constant rule beats BBRv3 by 0.079 (0.032 to
0.123) and BBRv1 by 0.097 (0.023 to 0.172); the oracle beats the constant
by 0.065 (0.031 to 0.105). The three seeds of the learned net2 beat BBRv3
by 0.119 to 0.123 and the constant by 0.040 to 0.044 (each interval
above 0), and lie within 0.004 of each other. By condition (seed 0,
re-run with the same seed, which reproduced its score, -0.348), the
learned net2 gains on the constant mostly under cross traffic: -0.812
against -0.911, +0.099 (0.014 to 0.186). On clean links it scores -0.124
against -0.133, +0.010 (-0.027 to +0.042), and under random loss -0.109
against -0.134, +0.025 (-0.017 to +0.069). Under cross traffic it raises
its target: over the steps after the first 2 s, u runs from -0.31 to 0.88
(10th to 90th percentile), where the constant holds 0.25.

On the validation links the same three seeds tied the constant (+0.009,
+0.002, -0.023), so the policies were also scored once on a second
held-out set, never used for any choice: the 158 test links of the
1,050-link campaign, under the three conditions (474 episodes per
policy; kernel controllers are not scored there, since each link has
only 3 of them):

| Policy | log utility |
|---|---|
| u = 0.25 on the true BDP (oracle) | -0.333 (-0.356 to -0.314) |
| learned net2 (seeds 1, 0, 2) | -0.360, -0.367, -0.373 |
| u = 0.25 on net1's BDP | -0.374 (-0.399 to -0.352) |
| u = 0.25 on the classical BDP | -0.602 (-0.635 to -0.571) |

There, the oracle beats the constant by 0.041 (0.033 to 0.049), the
constant beats the classical rule by 0.228 (0.196 to 0.259), and the
learned net2 beats the constant by 0.001 to 0.014 (one seed's interval
above 0, two include 0). On the 27 test links the constant beats the
classical rule by 0.155 (0.075 to 0.238).

Reading:

- With an honest net1, even the simplest decision on its output, a fixed
  multiple of the believed BDP, beats every controller in the kernel on
  links neither network has seen, on average over the three conditions
  (it wins two and loses one, below), and the old design by 0.6 in log
  utility.
- net1 is worth having: under the same rule, its belief beats the
  classical estimate by 0.155 and 0.228 on the two held-out sets, and
  falls short of the truth by 0.065 and 0.041. Between the classical
  estimate and the truth, net1 closes about 85% of the gap (158 links).
- The learned net2 does not reliably beat the constant: by about 0.04 on
  the 27 test links, 0.01 on the 158 others, and nothing on the
  validation links, consistently over three seeds; 60 iterations instead
  of 20 did not change that on validation (below). What it learned, to
  compete harder with a buffer-filling flow, is the right direction, but
  its size depends on how many links have a large buffer for Cubic to
  fill. The fixed multiple is the net2 to beat.
- Learning from other controllers' logs is confounded here; online
  training with multi-step returns is what worked.

**By condition, the constant wins two conditions and loses one.** The
constant's test run was repeated with the same policy and seed, and
reproduced its score exactly (-0.3927). Log utility by condition on the
27 test links:

| Condition | constant | BBRv3 | BBRv1 | Cubic | Vegas | constant minus BBRv3 |
|---|---|---|---|---|---|---|
| clean | -0.133 | -0.262 | -0.337 | -0.511 | -0.208 | +0.129 (0.079 to 0.179) |
| 1% random loss | -0.134 | -0.354 | -0.309 | -2.252 | -2.314 | +0.220 (0.172 to 0.267) |
| cross traffic | -0.911 | -0.799 | -0.824 | -0.874 | -1.028 | -0.112 (-0.202 to -0.026) |
| mean (the score above) | -0.393 | -0.472 | -0.490 | -1.212 | -1.183 | +0.079 |

Cross traffic is 3 Mbit/s of UDP from 5 to 15 s and one Cubic flow from
8 s. Against all 17 kernel controllers, the constant beats every one on
clean links and under random loss, each interval above 0; the closest are
Vegas on clean links, +0.075 (0.027 to 0.127), and BBRv1 under loss,
+0.175 (0.086 to 0.259). Under cross traffic it trails BBRv3, Illinois
(-0.073, -0.149 to -0.001), Scalable (-0.073, -0.148 to -0.004) and BIC
(-0.076, -0.149 to -0.010). Against BBRv1 there the difference is -0.087
(-0.193 to +0.014), and every other controller's interval includes 0
except CDG's: the constant beats CDG by 0.576 (0.336 to 0.821).

The three terms of the score (log throughput share / delay term / loss
term):

| Condition | constant | BBRv3 |
|---|---|---|
| clean | -0.044 / -0.078 / -0.011 | -0.069 / -0.166 / -0.027 |
| 1% random loss | -0.062 / -0.067 / -0.005 | -0.290 / -0.058 / -0.007 |
| cross traffic | -0.521 / -0.347 / -0.043 | -0.383 / -0.368 / -0.048 |

Reading:

- On clean links the constant keeps a smaller standing queue. A window
  of 1.19 BDP leaves about 0.19 BDP queued, which predicts a delay term
  near -0.095; it measured -0.078. Cubic gets the most throughput on
  clean links (log share -0.033) but fills the buffer (delay term
  -0.474).
- Under random loss the constant scores as on clean links: it does not
  react to loss, since net1's believed BDP does not move. Cubic, Reno and
  Vegas fall to 16% to 17% utilization, and BBRv3 to 75%. A likely cause
  for BBRv3, not checked: it bounds inflight when a round's loss rate
  exceeds its 2% loss threshold, which 1% random loss on small windows
  often does. BBRv1 also ignores loss, but drops 3.1% of its packets on
  shallow buffers.
- Under cross traffic the constant's delay term matches the others' (the
  Cubic flow fills the queue), and it loses throughput share: a fixed
  window takes a small share against a flow that keeps growing and
  inflating the RTT. This is where the learned net2 gained, by raising u
  under cross traffic.

So the headline, +0.079 over BBRv3, is a mean over three equally
weighted conditions, of which the constant wins two and loses one. What
would change the conclusion:

- the condition mix: under cross traffic alone, BBRv3, Illinois,
  Scalable and BIC rank above the constant (intervals below 0), and BBRv1
  is ahead by a similar margin (0.087) whose interval includes 0;
- the penalty weights: scored on log throughput alone (no delay or loss
  terms), BBRv1 beats the constant by 0.067, while the constant still
  beats BBRv3 by 0.038 and every other controller by 0.138 or more;
  tuning u for that score is UNRUN;
- tuning: only the constant was tuned for this score; the kernel
  controllers ran with their defaults;
- one flow per bottleneck: net1 estimates the link's capacity, not a
  fair share, so two flows of the constant would likely each aim at 1.19
  BDP, together 2.38 BDP in flight and a standing queue of 1.38 BDP, with
  steady drops on smaller buffers (UNRUN);
- the pairing: the kernel controllers' episodes come from the data
  campaign (seed 2026) and the constant's from seed 13, so pairs share
  links and conditions but not random draws.

**Longer online training does not beat the constant on validation, and
means do as well as distributions.** After the test, the 40-step run was
repeated for 60 iterations in two views of the same net1. Arm A3 sees
each LCF's distribution pooled to 32 bins (`net2.view: "bins"`,
`net2.input_bins: 32`); arm A2 sees only the 7 means of those distributions
(`net2.view: "means"`). Each starts from the constant (u = 0.25) in its
own view and trains online with TD3+BC (alpha 2.5, discount 0.97, 40-step
returns, 512 rows per update, 2,000 updates per iteration, `policy_noise`
0.05 clipped at 0.1), exploration 0.5 held for 20 steps, and 32 episodes
of 20 s per iteration; one seed each. Every 10 iterations each was scored
on the 21 validation links under the three conditions, paired with the
constant run again on the same links (-0.369):

| Iteration | A3 minus constant | A2 minus constant | A3 minus A2 |
|---|---|---|---|
| 10 | +0.001 (-0.031 to +0.030) | +0.031 (+0.013 to +0.050) | -0.030 (-0.060 to -0.002) |
| 20 | +0.001 (-0.017 to +0.019) | +0.014 (-0.013 to +0.042) | -0.013 (-0.036 to +0.011) |
| 30 | +0.004 (-0.021 to +0.029) | +0.007 (-0.021 to +0.035) | -0.003 (-0.029 to +0.023) |
| 40 | +0.000 (-0.024 to +0.028) | -0.001 (-0.028 to +0.027) | +0.001 (-0.025 to +0.027) |
| 50 | -0.014 (-0.035 to +0.008) | -0.014 (-0.042 to +0.015) | +0.000 (-0.026 to +0.027) |
| 60 | -0.014 (-0.037 to +0.014) | +0.005 (-0.019 to +0.031) | -0.019 (-0.039 to +0.003) |

At iteration 60, A3 scores -0.383 and A2 -0.364.

Reading:

- Longer training does not separate a learned net2 from the constant on
  validation: A3 stays level, and drifts slightly below after iteration
  40.
- Means do at least as well as distributions. The one interval that
  excludes 0 (iteration 10) favours the means, and it is the best of six
  checkpoints chosen on validation, so it overstates the gain. Neither
  run went to test, since neither beat the constant on validation.
- So far this net2 gets no measurable value from net1's stated
  uncertainty. A likely reason, not tested: a window target built from
  the believed BDP needs mostly the means, and these runs keep the link
  busy, where capacity is identifiable; uncertainty should matter when
  the link is idle and capacity is bounded only from below. net1's
  honesty (H3) still serves the human reading.

**What these results cover.** The net1 of "net1 on unseen links" and
every net2 run on it use one trace family and three conditions
(`examples/net1_varied.py`). Capacity is constant, or steps 2x up or down
every 3, 7 or 15 s, from a base of 8 to 100 Mbit/s; one-way delay is 5 to
50 ms; the buffer holds 0.25 to 4 BDP (20 to 2000 packets); episodes last
20 s, with one observed flow. The conditions are a clean link, 3 Mbit/s
of UDP from 5 to 15 s plus one Cubic flow from 8 s, and 1% random loss on
the downlink. Test links are new combinations of these, but the
conditions themselves are not held out, so generalization to new
conditions (other competing controllers, several flows, other loss
rates, other traces) was untested when these runs were made. The kernel
controllers ran on the same links and conditions, so the comparison is
matched, but its ranking holds for this mix only. The next two checks
test both limits.

**Longer episodes: at 60 s the constant ties BBRv3 and BBRv1.** A 20 s
episode might charge slow-starting controllers a startup cost they would
amortize over a longer transfer, and gives the Cubic competitor only
12 s. The 27 test links were run again under the same three conditions
with 60 s episodes (the Cubic flow then runs from 8 to 60 s), once, with
the same policies:

| Condition | constant, 20 s | constant, 60 s | minus BBRv3, 20 s | minus BBRv3, 60 s | minus BBRv1, 60 s | oracle minus constant, 60 s |
|---|---|---|---|---|---|---|
| clean | -0.133 | -0.121 | +0.129 (0.079 to 0.179) | +0.145 (0.097 to 0.195) | +0.220 | +0.022 |
| 1% random loss | -0.134 | -0.134 | +0.220 (0.172 to 0.267) | +0.253 (0.197 to 0.309) | +0.161 | +0.039 |
| cross traffic | -0.911 | -1.433 | -0.112 (-0.202 to -0.026) | -0.422 (-0.659 to -0.207) | -0.468 | +0.373 |
| mean | -0.393 | -0.563 | +0.079 (0.029 to 0.127) | -0.008 (-0.114 to +0.087) | -0.029 (-0.141 to +0.070) | +0.145 |

- Startup is not what the constant gains from. On clean links and under
  random loss its lead over the kernel controllers stays or grows at
  60 s, and it still beats Illinois (+0.205), Reno (+0.758), Cubic
  (+0.760) and Vegas (+0.911) on average.
- Longer competition is what it loses to. Under cross traffic the gap to
  BBRv3 grows from 0.112 to 0.422, and the oracle, the same rule on the
  true BDP, is 0.373 better: net1's belief, not the rule, fails there
  (next paragraph). The learned net2 recovers 0.117 of it.
- So the headline holds for 20 s episodes only. At 60 s the constant
  ties BBRv3 and BBRv1 on average and beats the other kernel controllers
  run (Illinois, Cubic, Reno, Vegas; the other 11 were not run at 60 s).

**New conditions: net1's capacity belief fails under competition it has
not seen.** The same policies, BBRv3, BBRv1 and Cubic ran once on the 27
test links under six conditions no campaign contained, on a deep-buffer,
long-RTT version of the same links (4 to 16 BDP, RTT up to 160 ms, under
the three usual conditions), and under 60 s of competition with a Cubic
flow present from the start:

| Condition | constant minus BBRv3 | constant minus BBRv1 | oracle minus constant | net1's 90% capacity statements held |
|---|---|---|---|---|
| random loss 0.1% | +0.116 | +0.189 | +0.041 | 86% |
| random loss 3% | +0.859 | +0.182 | +0.026 | 84% |
| 1% loss on the ACK path | +0.139 | +0.210 | +0.041 | 88% |
| deep buffers and long RTTs, clean | +0.268 | +0.221 | +0.006 | 78% |
| a BBRv1 flow from 4 s | +0.290 | -0.181 | +0.153 | 57% |
| Cubic flows from 2 s and 10 s | -0.301 | -0.316 | +0.255 | 57% |
| a Cubic flow from 0 s | -0.387 | -0.547 | +0.476 | 26% |
| a Cubic flow from 0 s, 60 s | -0.606 | -0.743 | +0.614 | 29% |

- Non-congestive loss and new paths leave the constant ahead. It ignores
  random loss, and BBRv3 collapses at 3% loss (-0.995).
- Competition net1 has not seen breaks its honesty. With a Cubic flow
  from the start, net1's median capacity belief is 0.59 of the truth
  (10th to 90th percentile 0.30 to 0.85), and its 90% statements hold
  26% of the time. In training the Cubic flow always arrived at 8 s,
  after the observed flow had measured the link alone, and Sage's
  longest window (10 s) still remembered that. Believing a smaller link,
  the constant sets a smaller window and takes a smaller share (0.37),
  which keeps the belief low. On the true BDP the same rule competes as
  well as the kernel controllers (-1.19 against -1.12 to -1.28).
- Outside the training ranges net1 is not honest either. On deep
  buffers its free-buffer statements hold 6% to 16% of the time at 90%,
  and its one-way delay has R^2 0.43 at RTTs above the 100 ms it was
  trained on, though the constant still matches the oracle there.
- The learned net2 stays within 0.05 of the constant everywhere: it does
  not repair net1's errors.

These two checks are why the next net1 needs a broader set of
conditions: its honesty, the claim the method rests on, fails where the
training data had no variety.

These runs used a harness kept outside the repository; the specs above
reproduce them with `train run` and `train online`.

### Tuned comparison

The search below ran on the expliCCit-like data. Its harness (grid
files, a parallel trainer, a closed-loop scorer) was fitted to this
dataset and is not in the repository. What carries over to other data is
in the trainer: IQL, the teacher filter (`controllers`) and the
diagnostics of section 2.

**Protocol.** Configurations were chosen on 12 validation paths that are
in neither the training grid nor the test set: capacity alternating 20
and 40 Mbit/s every 5 s, 40 and 80 every 12 s, constant 16 and constant
60, each with one-way delay and buffer of (8 ms, 150 packets), (25 ms,
400) and (35 ms, 80); 30 s closed-loop episodes. The score is the
utility. Winners were retrained over 4 or 5 seeds. Then every one was run
once on 18 test paths for 45 s: expliCCit's two case-study links and 16
held-out paths with rates, delays and buffers outside the training grid.
Intervals are over seeds and paths. All runs use the point net1: offline,
point and interval net1 tied (32 paired runs, mean difference -0.002).

**What was searched** (batch 256, learning rate 3e-4 and network sizes
fixed):

| Round | net2 | Settings |
|---|---|---|
| offline 1: 134 runs, 1 seed | cloning | net1 point or interval; `action_max` 0.05, 0.1, 0.25, 1; teachers all, BBRv1 and Vegas, BBRv1, Vegas; 10k and 30k updates |
| | expliCCit's objective | `action_max` 0.1, 0.25, 1; CQL weight 0.1, 1, 5; discount 0.95, 0.99 |
| | TD3+BC, utility reward | `action_max` 0.1, 0.25, 1; alpha 0.1, 0.5, 2.5; target noise 0.02, 0.2; discount 0.95, 0.99 |
| | IQL, utility reward | `action_max` 0.1, 0.25, 1; beta 3, 10, 30, 100; expectile 0.7, 0.9; discount 0.95, 0.99 |
| offline 2: 80 runs, 4 more seeds | cloning | `action_max` 0.05, 0.1, 0.25, 1; teachers all, BBRv1 and Vegas, Vegas; 30k and 100k updates; a rounded to 3 decimals or not |
| | TD3+BC, IQL | `action_max` 0.1, discount 0.99; alpha 0.5, 2.5 or beta 10, 100; teachers all, BBRv1 and Vegas; 30k and 60k updates |
| online 1: 32 runs | from the median of 5 cloned seeds | TD3+BC alpha 0.5, 2.5 or IQL beta 10, 100; noise 0.005, 0.02; 500 or 2000 updates per iteration; replay of own episodes, or with BBRv1 and Vegas logs and 5000 critic warm-up updates; 12 iterations of 16 episodes |
| online 2: 12 runs | IQL, own episodes, 2000 updates | beta 10, 30; noise 0.0025, 0.005, 0.01; 2 more seeds; 24 iterations |
| online 3: 4 runs | the online winner | from each of the other 4 cloned seeds; 8 iterations |

**What the diagnostics showed.**

- Noise between runs: two cloning runs that differ only in net1 kind
  differ by 0.33 (standard deviation over 32 pairs) in validation
  utility. A single run's rank means little, so every result below is
  over seeds.
- The regression loss on held-out steps does not predict closed-loop
  utility (Spearman 0.02 over 64 cloned checkpoints). The policy's mean
  action on held-out states does, negatively (-0.53 for cloning, -0.67
  for TD3+BC): a small positive bias integrates through the window into
  overdriving.
- `action_max` sets where the policy sits. At 0.05 to 0.1, cloned
  policies under-send (utilization 0.25 to 0.58 over seeds, almost no
  drops). At 1 they overdrive (0.73 to 0.87, 3 to 23% drops), and
  training longer lowers utilization further. TD3+BC at `action_max` 1
  and alpha 2.5 drives the mean action to +0.06 to +0.11 per step; with
  target noise 0.2, 56 to 60% of its target actions fall outside the
  data. Only `action_max` 0.1 keeps TD3+BC sane.
- IQL is cloning for beta 3 and 10 (mean weight 0.88 to 1.01). Beta 30
  and 100 reweight (mean weight up to 1.5 and 4.6) and do worse (round 1
  means +0.24 and +0.07, against +0.40 at beta 3).
- expliCCit's objective: its Q grows to 10^2 to 10^3, the critic's pull
  on the action is 14 to 5900 times the data's, and every setting stays
  far below cloning (mean -0.58 over 36 checkpoints, best +0.31). In an
  earlier untuned run at discount 0.99, it held a window far above what
  the path holds: the full queue dropped 73% of its packets, and its
  critic loss fell to about -4900, because its penalty on random actions
  has no floor.
- Online: with noise of 0.02 the policy's mean action turns negative
  and utilization falls (a plausible reading: it learns to back off from
  its own noise). TD3+BC can overdrive at once (one run's first rollouts
  scored -0.96). The teachers' logs in the replay buffer do not help.
  IQL at beta 10 and noise 0.005 is the one setting that does not fall
  below its start: with weights near 1 it is a slow advantage-weighted
  regression on its own episodes. Its rollouts' reward stays flat over 24
  iterations, and more iterations do not help.

**Validation** (mean over seeds): TD3+BC (alpha 0.5, `action_max` 0.1,
BBRv1 and Vegas logs, 60k updates) +0.60 [+0.46, +0.72], 4 seeds;
cloning (`action_max` 0.1, BBRv1 and Vegas logs, 30k updates) +0.55
[+0.40, +0.69], 5 seeds; IQL beta 10 (all teachers) +0.52, 5 seeds; Vegas
+0.79, BBRv1 +0.77. Online IQL from each of the 5 cloned seeds changes
utility by +0.019 [-0.070, +0.103] against its start.

**Test, once** (18 paths, 45 s, episode seed 7; one campaign):

| controller | seeds | utility [95%] | seeds' range | utilization | mean queueing delay ms | drops |
|---|---|---|---|---|---|---|
| BBRv1 | | +0.82 [+0.75, +0.88] | | 0.970 | 10.9 | 0.005 |
| Vegas | | +0.74 [+0.64, +0.82] | | 0.951 | 18.6 | 0.001 |
| BBRv3 | | +0.70 [+0.57, +0.82] | | 0.942 | 17.8 | 0.002 |
| Cubic | | -0.02 [-0.41, +0.35] | | 0.981 | 76.6 | 0.001 |
| expliCCit's released policy, as deployed | | +0.29 [+0.07, +0.48] | | 0.824 | 21.7 | 0.127 |
| TD3+BC | 4 | +0.64 [+0.54, +0.73] | +0.57 to +0.71 | 0.725 | 8.2 | 0.004 |
| online IQL, from each cloned seed | 5 | +0.59 [+0.41, +0.72] | +0.41 to +0.69 | 0.730 | 16.2 | 0.009 |
| cloning | 5 | +0.55 [+0.44, +0.67] | +0.46 to +0.66 | 0.617 | 7.8 | 0.002 |
| IQL beta 10 | 5 | +0.50 [+0.39, +0.61] | +0.38 to +0.61 | 0.595 | 9.6 | 0.001 |

Differences, resampling seeds and paths (paths paired): online minus its
cloned start +0.03 [-0.13, +0.16]; TD3+BC minus cloning +0.09 [-0.02,
+0.20]; TD3+BC minus BBRv1 -0.18 [-0.30, -0.07]; TD3+BC minus Vegas
-0.09 [-0.24, +0.04]; cloning minus Vegas -0.18 [-0.34, -0.03]; cloning
minus expliCCit's released policy +0.27 [+0.10, +0.45].

expliCCit's released policy drops 12.7% on these paths. Its net1 is far
off on our data ("net1 on held-out paths"), and our pacing is refreshed
once per step, so this need not be its loss on its own testbed.

The first run of this campaign filled the disk, and 304 of its 468
episodes failed. It was discarded, the disk freed and the same campaign
run again. In the rerun the teachers repeat an earlier campaign on the
same paths (BBRv1 +0.817 against +0.82, Vegas +0.735 against +0.74).

**Reading.**

- Every learned net2 acting on seven inferred LCFs stays below BBRv1,
  by 0.18 or more. TD3+BC is level with Vegas and BBRv3 within the
  intervals; cloning is below Vegas. expliCCit's draft makes no
  comparison with its teachers, so this is our finding, not a
  reproduction of its result.
- net1 is least accurate where the learned policies operate. On the
  validation paths, net1's capacity estimate has R^2 0.78 on the
  teachers' episodes, 0.45 on a cloned policy's and -0.01 on the online
  policies' (4 policies, 12 paths each). These policies leave the link
  idle more often, and an idle link does not reveal its capacity
  (`docs/methodology.md`, section 2). Whether the decisions would improve
  with the true LCFs needs net2 trained and run on them (arm A1). The
  environment now supports that (section 3), but it is UNRUN; the case
  study below only patches beliefs, on one path.
- For cloning, tuning moved the operating point more than the score.
  Against an untuned one-seed run on the same test paths, it scores
  +0.55 against +0.54, but drops 0.2% of its packets instead of 4.8%, at
  lower utilization (0.62 against 0.84). TD3+BC, which overdrove untuned,
  gained most: -0.31 to +0.64.
- No method is shown to beat cloning: TD3+BC's +0.09 has an interval
  that includes 0, IQL is cloning, and online training adds +0.03 with
  an interval from -0.13 to +0.16.
- Within these settings, online training no longer degrades the cloned
  policy, and it does not improve it either. The claim that closed-loop
  training on intervals learns when to probe stays untested.
- Not searched, all UNRUN: reward scale 1 for TD3+BC (after the 1/25
  scale a step is worth about 0.04, and TD3+BC normalizes the critic term
  by its mean size); discount 0.995 or 0.999 (0.99 per 10 ms step is a
  1 s horizon, while a probe on an 80 ms path pays off over several
  RTTs); noise held for an RTT (the guest then drew noise per step;
  independent noise of 0.005 per 10 ms step moves the window by about 1%
  over 100 ms, so it barely explores at the window's own time scale); and
  one decision per RTT.

### Case study: expliCCit's 24 and 48 Mbit/s link

expliCCit's draft traces its policy's under-utilization on this link to
capacity underestimated by 25%. It reports that replacing that belief
with the truth restores utilization. We repeated the experiment on
2026-10-31 with the patched policies of sections 3 and 4.

The link: capacity alternating 24 and 48 Mbit/s every 7 s, one-way delay
40 ms, a 1000-packet buffer, 45 s. It is one deterministic path: episode
seeds 1, 2 and 3 gave identical runs, so each row below is one run. The
link is also one of the 120 paths of the teacher campaign (section 1), so
our models may have trained on it.

expliCCit's released policy, as deployed, runs at 0.49 utilization with a
mean queueing delay of 6.6 ms. With some of its beliefs replaced by the
live truth:

| LCFs from the truth | utilization |
|---|---|
| none | 0.49 |
| capacity | 0.43 |
| busy fraction | 0.72 |
| queue | 0.59 |
| mean queueing delay | 0.04 |
| free buffer | 0.03 |
| busy fraction, queue, mean queueing delay and free buffer | 0.71 |
| all seven | 0.62 |

Its beliefs along its own run, against the truth (medians over the run;
capacity in the first two 7 s segments):

| LCF | released net1 | truth |
|---|---|---|
| queue | 18 packets | 0 packets |
| mean queueing delay | 11.6 ms | 0.76 ms |
| free buffer | 124 packets | 1000 packets |
| busy fraction | 1.00 | 0.53 |
| capacity | 24.7 and 47 Mbit/s | 24 and 48 Mbit/s |

The released net1's value range for each LCF is: queueing delay 0 to
5123 ms, queue 0 to 10240 packets, capacity 2.4 to 385 Mbit/s, free
buffer 0 to 10240 packets, busy fraction 0.2 to 1, one-way delay 5 to 40
ms, receive rate 0 to 385 Mbit/s. A free buffer of 1000 packets is inside
its trained range, so the collapse when free buffer comes from the truth
is a problem of the LCFs' joint range, not of any one LCF's range.

We also ran our own models, trained in UML, on the same link. In the
table, the queue group is mean queueing delay, queue and free buffer.

| policy | LCFs from the truth | utilization | mean queueing delay ms | drop fraction |
|---|---|---|---|---|
| TD3+BC (the tuned winner, training seed 1) | none | 0.987 | 63.7 | 0 |
| | all seven | 0.971 | 43.0 | 0 |
| | busy fraction | 0.924 | 9.6 | 0 |
| | queue group | 0.951 | 181.6 | 0.385 |
| cloning (median seed) | none | 0.770 | 71.0 | 0 |
| | all seven | 0.979 | 116.0 | 0.105 |
| | busy fraction | 0.144 | 1.0 | 0 |
| | queue group | 0.968 | 188.4 | 0.255 |

**Reading.** This is one path, with one run per row.

- The draft's capacity diagnosis does not reproduce with the released
  model here. Its capacity belief is about right, and taking capacity
  from the truth lowers utilization (0.43 against 0.49).
- Comparing its beliefs with the truth localized a different fault
  within minutes: busy fraction (1.00 against 0.53) and queue (18
  packets against 0). Taking busy fraction from the truth gives 0.72.
- The released net1 was trained on expliCCit's own data, so this result
  mixes transfer between emulators with the method.
- The LCFs are coupled, so replacing one belief can describe a path that
  cannot exist. A net2 trained on net1's beliefs has adapted to them, so
  the truth is not automatically a repair. On our own models some
  patches hurt: busy fraction from the truth drops the cloned policy to
  0.144 utilization, and with the queue group from the truth TD3+BC's
  drop fraction rises from 0 to 0.385.

## 8. Next

"net2 on the fixed net1" is the first run of the design in
`docs/methodology.md`: an honest net1, fixed, and net2 setting a window
target from its beliefs. Three seeds of the learned net2, 60 iterations
of online training and arm A2 (the means) have run; on validation none
beat the constant. What it leaves open, in the order the plan needs it:

- First, a net1 trained on a broad set of conditions, with whole families
  held out for testing: other competing controllers and more flows,
  competitors that arrive first (the observed flow cannot start late
  yet), other loss rates and directions, larger steps, deeper buffers and
  longer RTTs, and episodes long enough for competition to settle. The
  checks above show net1's honesty failing outside its three conditions.
- The human reading (a few percentiles per LCF) as net2's input, against
  the means and the distributions.
- Arm A1 for the learned net2 (trained and run on the truth), and A4
  (raw measurements), to measure the costs of estimation and of the
  split for a learned decision, not only for the constant rule.
- Several learned flows on one bottleneck, for fairness between copies:
  the constant aims at the link's BDP, not a fair share.

`docs/methodology.md` has the claims (section 1), the plan (section 3),
the arms (section 5) and the status (section 6).
