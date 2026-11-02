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
| `net2.input_bins` | none | distribution net1 only: net2 sees each LCF's distribution summed into this many groups of adjacent bins |
| `net2.action` | `multiplier` | `multiplier`: net2 outputs a; `bdp`: net2 outputs the next window as a log2 multiple of the believed bandwidth-delay product (below) |
| `net2.reward` | `sage` | Sage's reward; `utility` and `log_utility` are the ground-truth utilities below |
| `net2.discount` | 0.95 | per step |
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
costs and what net1's errors cost. That comparison is UNRUN.

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
  unfamiliar paths. The plan replaces free buffer by buffer size and
  queue, varies the paths and controllers more, and adds an ensemble
  (`docs/methodology.md`, section 3).
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
  RTTs); noise held for an RTT (the guest draws noise per step;
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

None of the policies above uses the design in `docs/methodology.md`;
they set the baseline it must beat. That page has the claims to test
(section 1), the training plan for net1 and net2 (section 3), the
experiment arms (section 5) and the status of the reproduction
(section 6). The first changes the results point to are one decision per
RTT on an absolute target, exploration held over an RTT, and teacher
logs used only to start (section 3 there).
