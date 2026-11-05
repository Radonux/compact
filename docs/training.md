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

expliCCit's code stays out of this repository until expliCCit is
published (`AGENTS.md`); its released net1 and actor (`sage_rl/offline/models`)
are in `third_party/expliccit-models` and load into our modules unchanged. The loader reads `best_model.pth`, `actor.pth`, and the
scalers' parameters from `x_scaler_params.json` and
`y_scaler_params.json`. The pickled scikit-learn scalers are never
loaded.

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit \
        --as-deployed --data out/teachers

`--as-deployed` repeats what its deployed controller does between and
after the networks: LCF beliefs clamped at 0, the first six rounded, and a
rounded to three decimals. `--data` reports its net1's error on our data.
`--truth` takes the named LCFs from the live truth instead of its net1,
for example:

    uv run expliccit train import-expliccit third_party/expliccit-models out/expliccit-busy \
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

Until 2026-11-04 the guest had Linux's own TCP buffer limits for 1 GiB (a
4 MB send buffer). That changes nothing on the train, validation and test
links of the training, new and cubic-first families, but on about half of
the deep-long family's links buffer-filling controllers queued less than
with the original setups' 64 MB (`docs/validation.md`, "TCP buffer
limits"). The deep-long family of every run below that is still used was
rerun on the same episodes with the fix; the tables give the rerun's
deep-long rows and means, marked with a dagger (†), and "Deep-long with the
original setups' TCP limits" below lists the changes. The runs not rerun
(the learned net2 on net1 v4, the mode rule and the share rule on the
truth: negative results decided elsewhere) keep their old deep-long rows.
net1 v4's data (about 7% of its links) is being rerun.

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
  ties BBRv3 and BBRv1 on average and beats the other kernel controllers.

Release
[`exp-test-links-duration-2`](https://github.com/Radonux/palaestra/releases/tag/exp-test-links-duration-2)
(`experiments/test-links-duration.json`) repeats both lengths with all 17
kernel controllers and the policies in one campaign (seed 13), so every
controller ran the same links and conditions; each episode still draws its
own kernel seed, so the pairing is by link and condition, not by random
stream. No episode failed. At
20 s the constant leads BBRv3 by 0.079 (0.037 to 0.119), the learned net2
by 0.126 and the oracle by 0.145. At 60 s the constant ties BBRv3
(-0.004, -0.087 to +0.070), as do BBRv1 (+0.029) and the learned net2
(+0.047, -0.020 to +0.112); the oracle still leads by 0.140, and every
other kernel controller trails BBRv3 by 0.20 (Illinois) to 1.02 (CDG).

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

### net1 on a broad set of conditions (net1 v4)

The checks above found net1 v3 honest only within its three conditions.
net1 v4 keeps its design (the seven LCFs, 64 bins, an ensemble of three
members) and is trained on a much broader dataset, generated on GitHub
Actions by `examples/net1_broad.py` and published as releases:

- [`exp-data-net1-broad-30s-5`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-30s-5)
  and [`exp-data-net1-broad-60s-8`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-60s-8):
  2,400 links with 30 s episodes and 1,200 more with 60 s episodes
  (21,600 episodes, none failed). Capacity starts at 4 to 150 Mbit/s and
  is constant or steps by 0.25x to 4x every 2 to 20 s; one-way delay is 5
  to 60 ms; buffers hold 0.25 to 8 BDP. Each link runs under two of 64
  conditions: up to three competing TCP flows (any kernel controller but
  BBRv3 and Westwood, arriving at the start in a third of cases, some
  leaving early), UDP at 5% to 50% of capacity, and random loss of 0.01%
  to 2% on either path; 70% of episodes have a competing flow. The
  observed flow is one of the 17 kernel controllers, the four
  delay-threshold policies or the fixed rule on net1 v3 (with and without
  exploration), three per link.
- [`exp-data-net1-broad-ood-9`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-ood-9):
  four families held out whole, 900 episodes each on new links: competing
  BBRv3 or Westwood flows, random loss of 3% to 6%, one-way delay of 61
  to 90 ms, and buffers of 8 to 16 BDP.

Training follows net1 v3: every 12th step, split by link (seed 1, 15% of
links each for validation and test), with BBRv3, Westwood and
threshold-0.3 held out of the training rows; 3.61M training rows, twice
net1 v3's.

**What was searched.** Dropout 0, 0.05 and 0.1 by width 512 and 1024,
with net1 v3's other settings (learning rate 1e-3, up to 12 epochs,
stopping after 3 without improvement), in release
[`exp-net1v4-grid-10`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-grid-10).
Ranked by mean CRPS skill over the seven LCFs on the validation rows, the
six runs span 0.768 to 0.780; dropout 0.05 at width 512 won, and width
1024 took three times as long for no gain. Every run stopped after 4 to 7
epochs: each member's validation loss was lowest after 1 to 4 epochs and
rose after, fastest at dropout 0.1. A lower learning rate did not change
that: release
[`exp-net1v4-lr-12`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-lr-12)
tried 3e-4 and 1e-4 at dropout 0 and 0.05 (width 512), and its four runs
reached 0.762 to 0.771, below every run at 1e-3 but one; their validation
loss still turned after 2 to 5 epochs, at a worse minimum (1.69 to 1.76
against 1.62). The loss turns because the network starts to fit the
training links, as net1 v3's did on too few links, not because steps are
too large. The winner stands, and release
[`exp-net1v4-seeds-13`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-seeds-13)
confirms it over two more ensemble seeds (10 and 20, whose members' batch
orders stay apart from seed 0's): dropout 0.05 scores 0.780, 0.777 and
0.779, dropout 0 scores 0.777, 0.776 and 0.774. The gap, 0.003 on
average, is as large as the spread over seeds but lies the same way on
every seed. The numbers below are seed 0's, the best of the three by at
most 0.003; over the three seeds, capacity's 90% coverage is 0.896 to
0.910 on the test links, 0.868 to 0.874 under held-out controllers and
0.894 to 0.909 with a BBRv3 or Westwood competitor, and free buffer's on
deep buffers is 0.425 to 0.447.

**Against net1 v3**, 90% coverage / R^2 on the same rows of the broad
dataset's test links and held-out families. net1 v3 never saw these
conditions, so most of the gap is the data, not the network.

| Rows | net1 | capacity | free buffer | queue | busy fraction | one-way delay | lowest coverage, 7 LCFs |
|---|---|---|---|---|---|---|---|
| test links | v4 | 0.90 / 0.82 | 0.90 / 0.53 | 0.92 / 0.90 | 0.91 / 0.88 | 0.97 / 1.00 | 0.90 |
| | v3 | 0.63 / 0.68 | 0.67 / 0.19 | 0.78 / 0.75 | 0.86 / 0.80 | 0.90 / 0.97 | 0.63 |
| held-out controllers, test links | v4 | 0.87 / 0.79 | 0.88 / 0.51 | 0.88 / 0.79 | 0.90 / 0.81 | 0.97 / 1.00 | 0.87 |
| | v3 | 0.59 / 0.63 | 0.65 / 0.21 | 0.75 / 0.58 | 0.86 / 0.73 | 0.91 / 0.97 | 0.59 |
| BBRv3 or Westwood competing | v4 | 0.90 / 0.85 | 0.90 / 0.48 | 0.90 / 0.87 | 0.91 / 0.79 | 0.97 / 1.00 | 0.90 |
| | v3 | 0.56 / 0.72 | 0.72 / 0.23 | 0.74 / 0.72 | 0.87 / 0.60 | 0.89 / 0.97 | 0.56 |
| random loss 3% to 6% | v4 | 0.87 / 0.59 | 0.90 / 0.26 | 0.92 / 0.59 | 0.85 / 0.82 | 0.97 / 1.00 | 0.85 |
| | v3 | 0.64 / 0.50 | 0.63 / 0.18 | 0.90 / 0.57 | 0.80 / 0.91 | 0.94 / 0.98 | 0.63 |
| one-way delay 61 to 90 ms | v4 | 0.88 / 0.69 | 0.91 / 0.33 | 0.90 / 0.80 | 0.91 / 0.78 | 0.92 / -2.40 | 0.87 |
| | v3 | 0.61 / 0.54 | 0.62 / -0.00 | 0.75 / 0.47 | 0.86 / 0.60 | 0.89 / -7.05 | 0.61 |
| buffers 8 to 16 BDP | v4 | 0.89 / 0.85 | 0.42 / 0.06 | 0.90 / 0.85 | 0.91 / 0.88 | 0.95 / 1.00 | 0.42 |
| | v3 | 0.69 / 0.73 | 0.19 / -0.22 | 0.77 / 0.59 | 0.87 / 0.79 | 0.77 / 0.90 | 0.19 |

- On new links under the broad conditions, net1 v4's 90% statements hold
  90% to 97% of the time per LCF. Competition from the start, which broke
  net1 v3, is now in the training data, and capacity is covered 90% of
  the time.
- Controllers it has not seen cost little. With BBRv3, Westwood or
  threshold-0.3 as the observed flow, capacity is covered 87% of the time
  (82% for BBRv3 alone); with BBRv3 or Westwood as the competitor, which
  no training condition had, every LCF is covered at least 90% of the
  time.
- Loss above the training range costs accuracy more than honesty:
  coverage stays at 0.85 to 0.97, but capacity R^2 falls to 0.59.
- Two families remain outside what net1 v4 can state. Its bins end just
  above the training range, and the last bin takes everything beyond it,
  so one-way delays above 60 ms count as covered (0.92) while the median
  cannot leave the training range (R^2 -2.40). On buffers of 8 to 16 BDP
  its free-buffer statements hold 42% of the time. Both are range limits
  of the data; whether they matter for control is for the experiments
  below.

**The fixed rule on net1 v4, validation links only.** Release
[`exp-rule-net1v4-val-11`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-val-11)
(`experiments/rule-net1v4-val.json`) ran the rule for u in {-0.25, 0,
0.125, 0.25, 0.375, 0.5} on net1 v4, on net1 v3 and on the true BDP,
with BBRv3, on net1 v3's 21 validation links: the training conditions at
20 s and 60 s, and the three families of "New conditions" above. No
episode failed. Each group's u is the one with the best mean difference
from BBRv3 over the five campaigns. Log utility minus BBRv3, paired by
link:

| Campaign | net1 v4, u = 0.125 | net1 v3, u = 0.375 | net1 v3, u = 0.25 (its earlier choice) | true BDP, u = 0.375 |
|---|---|---|---|---|
| training conditions, 20 s | +0.154 | +0.084 | +0.123 | +0.130 |
| training conditions, 60 s | +0.131 | -0.015 | +0.003 | +0.130 |
| new conditions, 20 s | +0.271 | +0.163 | +0.148 | +0.292 |
| deep buffers and long RTTs, 20 s † | +0.213 | +0.106 | +0.097 | +0.162 |
| a Cubic flow from 0 s, 60 s | -0.082 (-0.253 to +0.082) | -0.526 (-0.911 to -0.185) | -0.668 | +0.066 (-0.059 to +0.182) |
| mean of the five † | +0.137 | -0.038 | -0.059 | +0.156 |

- On these links the rule on net1 v4 leads BBRv3 by 0.137 on average
  (0.150 before the deep-long rerun), within 0.02 of the same rule on the
  true BDP; on net1 v3 it trails by 0.038 at best. Most of the difference is the competition net1 v3 had
  not seen: with a Cubic flow from the start for 60 s, the rule on net1
  v4 ties BBRv3, and on net1 v3 it loses by 0.53.
- It also does better under net1 v3's own training conditions (+0.154
  against +0.123 at 20 s, +0.131 against +0.003 at 60 s).
- For net1 v4 the choice of u is flat: 0.125 and 0.25 differ by 0.002
  (0.001 after the deep-long rerun, which keeps every group's u).
  On deep, long paths the rule on net1 v4 beats the rule on the true BDP
  (+0.213 against +0.162); with one u for every campaign, the true BDP is
  not an upper bound for the rule.
- Limits of this result: these are the links u was chosen on, and each
  number is the best of six, so it leans optimistic; there are 21 links;
  and most of these families lie inside net1 v4's training conditions
  (only 3% loss and part of the deep, long paths are outside), so it
  shows that the broader data repairs net1 v3's failure, not that net1 v4
  generalizes beyond its data. net1 v4's estimates under net1 v3's three
  narrow conditions are UNRUN as an estimation check.

**The fixed rule on net1 v4, test links, once.** Release
[`exp-rule-net1v4-test-14`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-14)
(`experiments/rule-net1v4-test.json`) ran each rule at its validation u
(net1 v4 0.125, net1 v3 0.375 and its released 0.25, the true BDP 0.375)
and the learned net2 on net1 v3, with all 17 kernel controllers, on the 27
test links: the five validation families (campaign seed 13 for the
training conditions, 31 for the rest) and, on the test links only, a BBRv3
or a Westwood flow competing from 0 s for 60 s, the competitors neither
net1 trained on. No episode failed. Log utility differences, paired by
link, with 95% intervals over the 27 links:

| Campaign | v4 minus BBRv3 | v4 minus v3 (u = 0.375) | v4 minus true BDP | best other kernel controller minus BBRv3 |
|---|---|---|---|---|
| training conditions, 20 s | +0.112 (+0.076 to +0.146) | +0.042 (+0.014 to +0.070) | -0.006 (-0.038 to +0.023) | BBRv1 -0.022 |
| training conditions, 60 s | +0.118 (+0.070 to +0.163) | +0.115 (+0.064 to +0.170) | -0.001 (-0.034 to +0.028) | BBRv1 +0.022 |
| new conditions, 20 s | +0.250 (+0.194 to +0.302) | +0.078 (+0.040 to +0.117) | -0.045 (-0.072 to -0.018) | BBRv1 +0.217 |
| deep buffers and long RTTs, 20 s † | +0.175 | +0.081 | +0.032 | BBRv1 -0.027 |
| a Cubic flow from 0 s, 60 s | -0.039 (-0.191 to +0.103) | +0.428 (+0.277 to +0.580) | -0.092 (-0.176 to -0.020) | Scalable +0.395 |
| mean of the five † | +0.123 | +0.149 | -0.022 | BBRv1 +0.062 |
| BBRv3 or Westwood from 0 s, 60 s (test only) | +0.094 (-0.016 to +0.202) | +0.099 (+0.029 to +0.169) | -0.050 (-0.097 to -0.007) | BIC +0.226 |

- On the test links the rule on net1 v4 leads BBRv3 by 0.123 on average
  over the five families (validation said 0.137, the best of six; 0.150
  before the deep-long rerun), and
  BBRv1, the best kernel controller on average, by 0.061. It beats the
  same rule on net1 v3 in every campaign, the interval excluding zero each
  time: by 0.04 under net1 v3's own conditions at 20 s, and by 0.43 with a
  Cubic flow from the start.
- It stays within 0.05 of the same rule on the true BDP everywhere but
  with a Cubic flow from the start (0.09): the cost of estimation left.
- With competitors neither net1 saw, it keeps its lead over net1 v3
  (+0.10) and stays within 0.05 of the true BDP, so its estimates carry to
  them for control, not only in coverage.
- Against loss-based flows present from the start, the fixed multiple
  itself is the limit, not net1: with a Cubic flow from 0 s, Scalable and
  BIC lead BBRv3 by 0.36 to 0.40 and the rule on the true BDP by only
  0.05; with BBRv3 or Westwood from 0 s, BIC and Scalable lead by 0.23,
  the rule on the true BDP by 0.14. A window of 1.09 BDP takes a small
  share from a flow that fills the buffer. A net2 that recognizes
  competition and takes more there is the next step; the fixed rule is
  its starting point.
- The learned net2 on net1 v3 ties the rule on net1 v4 under the
  training conditions at 20 s (the rule minus it: -0.011, -0.045 to
  +0.025) and trails it everywhere else, by 0.30 with the held-out
  competitors.

### net1 v5: net1 v4's settings, retrained under the frozen settings

The guest's TCP settings changed twice after net1 v4 was trained (the
setups' 64 MB limits, then the per-path send-buffer limit; both in
`docs/validation.md`), and are frozen at `1365f0e`. net1 v4's three
datasets were regenerated with the same specs under them
([`exp-data-net1-broad-30s-73`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-30s-73),
[`-60s-74`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-60s-74),
[`-ood-75`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-broad-ood-75)),
and release
[`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79)
trained a net1 on them with net1 v4's chosen settings (three members,
dropout 0.05, width 512, `experiments/specs/net1v4.json`) and scored net1
v4 itself on the same rows. The new net1 is as good or better on every
LCF's 90% coverage, and its CRPS skill on the validation rows is higher on
six of the seven LCFs (mean 0.774 against 0.753; one-way delay 0.001
lower), so it replaces net1 v4 as **net1 v5**, with no new search (the
settings were searched for net1 v4, and nothing got worse). 90% coverage /
R^2:

| Rows | net1 | capacity | free buffer | queue | busy fraction | one-way delay | receive rate |
|---|---|---|---|---|---|---|---|
| validation links | v5 | 0.92 / 0.80 | 0.89 / 0.41 | 0.92 / 0.91 | 0.91 / 0.87 | 0.97 / 1.00 | 0.94 / 0.99 |
| | v4 | 0.90 / 0.79 | 0.86 / 0.40 | 0.90 / 0.76 | 0.90 / 0.87 | 0.96 / 1.00 | 0.93 / 0.95 |
| test links | v5 | 0.92 / 0.81 | 0.91 / 0.47 | 0.92 / 0.90 | 0.91 / 0.88 | 0.97 / 1.00 | 0.94 / 0.99 |
| | v4 | 0.89 / 0.80 | 0.88 / 0.45 | 0.90 / 0.60 | 0.91 / 0.88 | 0.96 / 1.00 | 0.92 / 0.94 |
| BBRv3 or Westwood competing | v5 | 0.92 / 0.83 | 0.92 / 0.41 | 0.92 / 0.85 | 0.91 / 0.78 | 0.98 / 1.00 | 0.94 / 0.99 |
| | v4 | 0.88 / 0.81 | 0.88 / 0.37 | 0.89 / 0.71 | 0.90 / 0.77 | 0.97 / 1.00 | 0.92 / 0.95 |
| random loss 3% to 6% | v5 | 0.88 / 0.56 | 0.88 / 0.23 | 0.91 / 0.85 | 0.81 / 0.81 | 0.97 / 1.00 | 0.92 / 0.99 |
| | v4 | 0.86 / 0.57 | 0.88 / 0.22 | 0.91 / 0.43 | 0.84 / 0.83 | 0.97 / 1.00 | 0.93 / 0.96 |
| one-way delay 61 to 90 ms | v5 | 0.84 / 0.70 | 0.89 / 0.30 | 0.89 / 0.74 | 0.89 / 0.77 | 0.90 / -2.54 | 0.92 / 0.95 |
| | v4 | 0.85 / 0.64 | 0.88 / 0.27 | 0.86 / 0.46 | 0.91 / 0.78 | 0.91 / -2.44 | 0.89 / 0.84 |
| buffers 8 to 16 BDP | v5 | 0.90 / 0.83 | 0.44 / 0.04 | 0.90 / 0.75 | 0.90 / 0.87 | 0.96 / 1.00 | 0.94 / 0.97 |
| | v4 | 0.85 / 0.78 | 0.41 / 0.04 | 0.87 / 0.47 | 0.90 / 0.87 | 0.94 / 1.00 | 0.88 / 0.87 |

The largest gain is the queue (R^2 0.90 against 0.60 on the test links):
v4 never saw the deeper queues the 64 MB limits let buffer-filling flows
build. Two cells are a little worse: queueing delay's coverage at 61 to 90
ms (0.80 against 0.86) and busy fraction's under 3% to 6% loss (0.81
against 0.84). The range limits stay as they were (one-way delay above the
training range, free buffer on deep buffers).

**The rules on net1 v5, validation links over three campaign seeds.** The
fixed rule's u (release
[`exp-rule-net1v5-val-82`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v5-val-82),
u in -0.25 to 0.5 on the five families, seeds 11, 21 and 31): u = 0.25 and
u = 0.125 tie (means over the five families 0.1448 and 0.1456 on seed 11,
0.1494 and 0.1490 on seed 21, 0.1472 and 0.1454 on seed 31), and the rule
stated before the run, the higher mean over seeds, picks **u = 0.25**
(0.1471 against 0.1467). The two trade places by family on every seed:
u = 0.25 is better with a Cubic flow from the start by 0.065 to 0.073, u =
0.125 on the training conditions by about 0.03. The share rule's k (release
[`exp-share-net1v5-val-85`](https://github.com/Radonux/palaestra/releases/tag/exp-share-net1v5-val-85),
u0 = 0.25, k in 0.5 to 2, the five families and 2 and 4 copies of each
controller, the same seeds): **k = 0.5** on every seed, by the mean over
the six campaigns. Every share rule is below the fixed rule once copies
count (k = 0.5: +0.045, +0.061, +0.051 against the rule's +0.104, +0.123,
+0.109), since copies of it cost it about -0.55 against the rule's -0.09;
over the five families alone k = 0.5 is ahead (about +0.16 against +0.14),
most with a Cubic flow from the start (+0.12 to +0.17 against -0.001 to
+0.048). net1 v5 and both rules are release
[`model-net1v5`](https://github.com/Radonux/palaestra/releases/tag/model-net1v5).

### net2 on net1 v4

The fixed rule leaves share on the table against loss-based flows present
from the start. Two ways of taking it were tried, each starting from the
rule and acting on net1 v4's output alone.

**Online TD3+BC does not beat the rule.** Release
[`exp-net2v4-online-15`](https://github.com/Radonux/palaestra/releases/tag/exp-net2v4-online-15)
(`experiments/net2v4-online.json`) trained net2 online from the rule at
u = 0.125 with TD3+BC and 40-step returns, as for net1 v3: 40 iterations
of 32 rollouts of 30 s over 600 broad training links of their own
(`examples/net1_broad.py`, set `net2-train`), searching the BC weight
(alpha 1 or 2.5) by the exploration width (0.5 or 1.0, log2 of the
window, held 20 steps). In every run the critic's loss grew (to 1.5 to
3.2, against values of -2.8 to -5) while the spread of the actions grew
(standard deviation 0.09 to 0.21, and 0.15 to 0.43), and the noise-free
policy fell below its start on the in-training evaluation. On the five
validation families (release
[`exp-net2v4-val-16`](https://github.com/Radonux/palaestra/releases/tag/exp-net2v4-val-16)),
all 16 checkpoints (iterations 10 to 40) trail the rule: the best, alpha
1 and exploration 0.5 at iteration 40, by 0.036 on average, and by 0.17
on deep, long paths. With broad conditions the effect of u on log utility
is too small against the episodes' noise for the critic to resolve, so
the actor follows a noisy gradient.

**A share rule beats it on validation.** The rule's window raised when
net1 believes the link is busy and this flow gets a small share of it:

    u = 0.125 + k * busy * max(0, 1 - receive_rate / capacity)

from net1's means (`experiments/share_rule_export.py`; k = 0 is the
fixed rule, bit for bit). It has one number to choose, so it is chosen
in closed loop on validation like u. In the same release, paired with the
fixed rule by link (95% intervals over the 21 validation links):

| Family | share k = 0.5 | share k = 1 | share k = 2 |
|---|---|---|---|
| training conditions, 20 s | -0.003 | -0.019 (-0.052 to +0.016) | -0.077 (-0.148 to -0.007) |
| training conditions, 60 s | +0.020 | +0.026 (-0.006 to +0.057) | +0.010 |
| new conditions, 20 s | +0.011 | +0.025 (-0.015 to +0.063) | +0.004 |
| deep buffers and long RTTs, 20 s | -0.012 | -0.015 (-0.052 to +0.024) | -0.096 (-0.181 to -0.018) |
| a Cubic flow from 0 s, 60 s | +0.120 | +0.238 (+0.141 to +0.337) | +0.361 |
| mean of the five, minus BBRv3 | +0.169 | +0.193 | +0.182 |

The fixed rule's mean is +0.142, and k = 3 falls to +0.117. At k = 1 the
share rule matches the fixed rule wherever no competitor is there from
the start and gains 0.24 when one is; larger k gains more there and loses
elsewhere. Its risk is copies of itself: each believes it gets a small
share, so all may push.

**Copies of the share rule hurt each other.** Release
[`exp-share-copies-val-17`](https://github.com/Radonux/palaestra/releases/tag/exp-share-copies-val-17)
(`experiments/share-copies-val.json`, `experiments/copies.py`) started 2
or 4 copies of one controller together on each validation link for 30 s;
no episode failed. Log utility of the copies minus BBRv3's copies, paired
by link, and the copies' fairness and queueing delay:

| Controller | log utility minus BBRv3 copies | queueing delay, 2 copies (ms) | queueing delay, 4 copies (ms) | Jain's index, 4 copies |
|---|---|---|---|---|
| fixed rule on net1 v4 | +0.078 (-0.104 to +0.266) | 22.7 | 35.6 | 0.921 |
| share, k = 0.5 | -0.267 (-0.536 to -0.006) | 25.4 | 45.2 | 0.917 |
| share, k = 1 | -0.564 (-0.889 to -0.241) | 31.1 | 48.1 | 0.908 |
| share, k = 1.5 | -1.035 (-1.354 to -0.712) | 42.4 | 55.0 | 0.934 |
| share, k = 2 | -1.473 (-1.767 to -1.129) | 46.6 | 59.4 | 0.947 |
| Cubic | -0.014 (-0.196 to +0.149) | 60.0 | 64.1 | 0.958 |
| BBRv1 | -0.317 (-0.449 to -0.203) | 46.2 | 51.5 | 0.919 |

- Copies of the fixed rule share fairly, keep the link busier than any
  kernel controller's copies (utilization 0.986 to 0.989) and the queue
  shortest.
- Copies of the share rule stay as fair and as busy, but each believes
  its share is small and pushes, so the queue grows with k and their own
  log utility falls: by 0.35 at k = 0.5 and 0.64 at k = 1 against the
  fixed rule's copies. The share signal cannot tell a copy of itself
  from a Cubic flow.
- k = 1.5 on the five families: +0.198 against BBRv3 on average, about
  level with k = 1 (+0.193).

No k met the bar set before this run (copies' queueing delay no worse
than the fixed rule's), so the share rule's gain against loss-based
competitors comes with a cost against copies, and which to prefer is a
choice of weights, not a validation result. The single test therefore
ran three candidates fixed in advance (the fixed rule, and the share rule
at k = 0.5 and 1) with copies on the test links.

**Test links, once.** Release
[`exp-share-test-18`](https://github.com/Radonux/palaestra/releases/tag/exp-share-test-18)
(`experiments/share-test.json`) ran them with the true-BDP rule and all
17 kernel controllers on the 27 test links: the five families, BBRv3 or
Westwood from 0 s, and 2 and 4 copies for 30 s. No episode failed.
Paired with the fixed rule on net1 v4 by link (95% intervals over the 27
links):

| Campaign | share k = 0.5 minus fixed rule | share k = 1 minus fixed rule |
|---|---|---|
| training conditions, 20 s | +0.014 (-0.004 to +0.030) | +0.010 (-0.025 to +0.043) |
| training conditions, 60 s | +0.039 (+0.019 to +0.061) | +0.046 (+0.012 to +0.080) |
| new conditions, 20 s | +0.049 (+0.029 to +0.067) | +0.070 (+0.045 to +0.096) |
| deep buffers and long RTTs, 20 s † | +0.009 | +0.016 |
| a Cubic flow from 0 s, 60 s | +0.165 (+0.110 to +0.229) | +0.258 (+0.161 to +0.364) |
| mean of the five † | +0.055 | +0.080 |
| BBRv3 or Westwood from 0 s (test only) | +0.087 (+0.065 to +0.108) | +0.126 (+0.082 to +0.172) |
| 2 or 4 copies of itself | -0.337 (-0.453 to -0.230) | -0.794 (-0.975 to -0.613) |

- Against BBRv3 the share rule at k = 1 leads by 0.207 on average over
  the five families (the fixed rule by 0.126, the fixed rule on the true
  BDP by 0.149), and with BBRv3 or Westwood competing, which neither
  net1 saw, it matches BIC and Scalable (+0.206 against +0.226 and
  +0.214), which lose 0.4 to 1.7 elsewhere.
- It is never worse than the fixed rule outside copies, and better under
  any competition: by 0.26 with a Cubic flow from the start, and 0.13
  with the held-out competitors.
- Its copies pay for it: 0.79 below the fixed rule's copies, whose queue
  stays shortest (19 and 29 ms for 2 and 4 copies, against 31 and 37 ms)
  with the same fairness (Jain 0.97 and 0.93 to 0.95).
- So the test confirms the trade-off validation found, on new links.
  Weighting copies as one family of six, the fixed rule is ahead; leaving
  them out, the share rule is. A net2 that tells a copy of itself from a
  buffer-filling competitor would keep both; the share signal alone
  cannot.

**A net2 with memory does not resolve it, on validation.** Copies of the
fixed rule fill the buffer about as much as a Cubic flow does (in
`exp-share-copies-val-17` the median queue is 0.38 and 0.57 base RTTs for
2 and 4 copies, against 0.55 with Cubic), so the queue alone cannot tell
them apart. The mode rule (`experiments/mode_rule_export.py`, exported
with a recurrent state) therefore cooperates with a window that follows
its own share,

    share = (s + 0.3 (1 - busy) + 0.05) / 1.05, at most 1;  u = 0.125 + log2(share)

(s is receive rate over capacity), which is the fixed rule when alone. It
competes as the share rule (k = 1) for a period of 10 or 20 units (the
base RTT, at least 100 ms) after net1 believes the buffer more than 0.7
full or the queue over one base RTT, and again whenever the mean believed
fill stays above 0.35 just after. Local runs on four validation links,
not published, chose this design over six others; its other numbers are
not tuned. Release
[`exp-mode-val-19`](https://github.com/Radonux/palaestra/releases/tag/exp-mode-val-19)
(`experiments/mode-val.json`) ran it, its cooperative mode alone, the
fixed rule and the share rule on the five validation families and as 2
and 4 copies; no episode failed. Log utility minus BBRv3, paired by link:

| Family | fixed rule | share k = 1 | mode, period 20 | mode, period 10 | cooperative mode alone |
|---|---|---|---|---|---|
| training conditions, 20 s | +0.141 | +0.121 | +0.089 | +0.081 | -0.041 |
| training conditions, 60 s | +0.136 | +0.163 | +0.120 | +0.108 | -0.232 |
| new conditions, 20 s | +0.265 | +0.296 | +0.199 | +0.192 | -0.266 |
| deep buffers and long RTTs, 20 s | +0.228 | +0.209 | +0.122 | +0.126 | +0.205 |
| a Cubic flow from 0 s, 60 s | -0.087 | +0.156 | -0.056 | -0.143 | -1.822 |
| 2 or 4 copies of itself, 30 s | +0.066 | -0.583 | -0.058 | +0.019 | +0.510 |
| mean of the six | +0.124 | +0.060 | +0.070 | +0.064 | -0.274 |

- The cooperative mode does what it was built for: its copies keep the
  queue at 5 and 8 ms (2 and 4 copies, against 23 and 36 ms for the fixed
  rule) at the same fairness (Jain 0.97 to 0.99), and lead the fixed
  rule's copies by 0.44. Against any competitor it yields and starves, by
  1.7 below the fixed rule with a Cubic flow from the start.
- The switch keeps neither gain. It trails the fixed rule in every family
  but Cubic from the start, where it gains only 0.03 (the share rule
  gains 0.24), and its copies do no better than the fixed rule's. In the
  local runs such copies often stayed in the competing mode, which their
  own full buffer keeps calling for; that is not checked on these links.
- So no candidate beats the fixed rule with copies weighted as one family
  of six, and the mode rule was not run on test. In the local runs, every
  switch that held up against Cubic left its copies competing, and every
  one that let copies cooperate yielded to Cubic: probes of whether the
  others yield failed on noise in single-step beliefs, on Cubic's slow
  growth, and on copies falling out of step.

**On the true LCFs (arm A1), the share rule gains alone and fails with
copies.** Release
[`exp-share-oracle-val-20`](https://github.com/Radonux/palaestra/releases/tag/exp-share-oracle-val-20)
(`experiments/share-oracle-val.json`) ran the share rule read from the
guest's live truth (`share_rule_export.py --oracle`; each belief becomes
the true value's bin centre in net1 v4's bins) for u0 = 0.125 or 0.375 and
k = 0, 0.5, 1 or 2, next to the same rule on net1 v4, on the five
validation families and as copies; no episode failed. Log utility minus
BBRv3, paired by link (differences between columns are of these means,
without their own intervals):

| Family | fixed rule, net1 v4 | share k = 1, net1 v4 | fixed rule (u0 = 0.125), truth | share k = 1, truth | share k = 2, truth |
|---|---|---|---|---|---|
| training conditions, 20 s | +0.139 | +0.123 | +0.175 | +0.186 | +0.171 |
| training conditions, 60 s | +0.130 | +0.157 | +0.153 | +0.195 | +0.214 |
| new conditions, 20 s | +0.246 | +0.277 | +0.270 | +0.348 | +0.389 |
| deep buffers and long RTTs, 20 s | +0.231 | +0.220 | +0.225 | +0.270 | +0.274 |
| a Cubic flow from 0 s, 60 s | -0.057 | +0.185 | -0.054 | +0.120 | +0.228 |
| mean of the five | +0.138 | +0.192 | +0.154 | +0.224 | +0.255 |
| 2 or 4 copies of itself, 30 s | +0.069 | -0.592 | -0.991 | -1.797 | -2.351 |
| mean of the six | +0.127 | +0.061 | -0.037 | -0.113 | -0.179 |

- Without copies of itself, the truth is worth 0.03 to the share rule at
  k = 1 and 0.06 between the best of each (k = 2 on the truth, k = 1 on
  net1 v4); on the truth u0 = 0.375 is better only at k = 0 (+0.168
  against +0.154).
- With copies, every rule on the truth fails, the fixed one included:
  each copy sets its window from the link's whole BDP, so 2 and 4 copies
  build queues of 41 and 66 ms (the fixed rule on net1 v4: 23 and 35 ms)
  and lose 1.06 against the same rule on net1 v4. net1 v4 believes a
  shared link has less capacity than it has (0.4 to 0.9 of it in local
  runs, not published), which for this window target works like an estimate of the
  capacity left to this flow.
- So with copies weighted as one family of six, the fixed rule on net1 v4
  is ahead of every rule on the truth. The link's capacity is the wrong
  quantity for a window target once the link is shared; what the target
  needs, the capacity this flow can take, is not among the LCFs (arm A0
  against A1). net1's bias there helps by accident and is not a
  property to count on.

**Told the number of flows, a rule on the truth cooperates with its copies
and competes with Cubic.** The candidate missing quantity is the number
of flows N on the bottleneck. `experiments/share_rule_export.py --flows N`
gives it to the share rule, and `experiments/fairshare.py` runs conditions
where N is constant and known: alone for 20 s (N = 1), a Cubic flow from
0 s for 60 s (N = 2), and 2 or 4 copies starting together for 30 s, on
the 21 validation links. Each episode's policy is told its N, so these
are upper bounds for a net1 that would have to estimate it. Three
releases, each with the fixed rule on net1 v4 and BBRv3 (log utility
minus BBRv3, paired by link):

| Rule | release | alone | 2 or 4 copies | Cubic from 0 s | mean |
|---|---|---|---|---|---|
| fixed rule, net1 v4 | 21, 22, 23 | +0.183, +0.183, +0.180 | +0.096, +0.056, +0.111 | -0.031, -0.059, -0.029 | +0.083, +0.060, +0.087 |
| aims at 1/N, pushes below 1/N of the capacity, truth, k = 1 | 21 | +0.211 | +0.449 | -0.395 | +0.089 |
| the same, k = 3 | 21 | +0.167 | +0.439 | -0.194 | +0.137 |
| the same, k = 4 | 22, 23 | +0.111, +0.108 | +0.269, +0.326 | +0.040, +0.068 | +0.140, +0.167 |
| the same, k = 6 | 22 | -0.210 | -1.217 | +0.224 | -0.401 |
| the same on net1 v4's beliefs, k = 2 | 21 | +0.076 | +0.173 | -0.465 | -0.072 |
| aims at 1/N, pushes below 1/N of the capacity in use, truth, k = 2 | 23 | +0.199 | +0.460 | -0.286 | +0.124 |
| the same, k = 4 | 23 | +0.187 | +0.398 | +0.065 | +0.216 |
| the same, k = 6 | 23 | +0.166 | -0.605 | +0.249 | -0.064 |

(Releases
[`exp-fairshare-val-21`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-val-21),
[`exp-fairshare-k-val-22`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-k-val-22)
and
[`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23);
`experiments/fairshare-val.json`, `fairshare-k-val.json` and
`fairshare-used-val.json`. Episode seeds are drawn in order, so the runs
differ in seeds; the fixed rule's rows show that spread.)

- Knowing N fixes copies: aiming at 1/N of the true BDP, 2 and 4 copies
  keep the queue at 3 to 6 ms for k up to 3 (11 ms at k = 4 in its second
  form) with utilization 0.96 to 0.98 and Jain's index 1.00 (the fixed
  rule's copies: 23 and 36 ms, Jain 0.92 with 4).
- The first form of the push, k busy max(0, 1 - N s), also fires when a
  flow alone is not yet using the link, so the gain that holds a Cubic
  flow off (k of 4 and more) overshoots alone and among copies.
- Measured against 1/N of the capacity in use, k max(0, 1 - N s / busy),
  the push is zero for a flow alone and for a copy at its fair share. At
  k = 4 this rule matches the fixed rule alone (+0.187 against +0.180)
  and leads it with copies (by 0.29) and with a Cubic flow from the start
  (by 0.09); larger k gains more against Cubic and loses with copies.
- On net1 v4's beliefs the same rules leave copies' links underused
  (utilization 0.63 to 0.82): net1 already believes less capacity on a
  shared link, so dividing by N counts the sharing twice.
- What this shows: with the truth and N, one rule does well in all three
  conditions, so what the window target lacks is information about the
  other flows, not a mode switch. What it does not show: k = 4 was chosen
  on these links; N is constant here and handed to the rule; conditions
  where N changes (the training families' flows from 8 s, UDP) are not
  run.
- N is not added to the LCFs. It describes the other endpoints and their
  arrivals rather than the path, changes as flows come and go (like the
  free buffer, which net1 v4 already covers worst), and the measurements
  of one sender fit different N (one Cubic flow or two, a copy or a
  constant-rate flow), so an honest net1 would state it with little
  confidence. This oracle is an upper bound on what knowing the other
  flows is worth, and it leaves that inference to net2.

### Sage's released policy

expliCCit reports that its split costs about 5% utilization and 9%
queueing delay against Sage. Sage's released policy, rewritten in torch
and exported under `sage-v1`
([`exp-sage-model-24`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-model-24)),
ran on the validation links
([`exp-sage-val-25`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-val-25))
and then once on the 27 test links
([`exp-sage-test-26`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-test-26);
`experiments/sage-val.json`, `sage-test.json`). Next to it ran the fixed
rule (u = 0.125) and the share rule (k = 1) on net1 v4, both fixed before
either run, and all 17 kernel controllers. No episode failed. On the test
links with the flow alone, as expliCCit compares (means over links;
ratios to Sage with 95% bootstrap intervals over links):

| Campaign, condition | Sage: utilization, delay (ms) | Fixed rule | Fixed rule against Sage: utilization, delay |
|---|---|---|---|
| training-20, clean | 0.911, 9.9 | 0.948, 5.4 | +4.1% (-1.9% to +11.5%), -46.0% (-56.8% to -31.0%) |
| training-60, clean | 0.927, 10.6 | 0.958, 5.7 | +3.4% (-1.1% to +8.4%), -46.1% (-61.1% to -26.1%) |
| deep-long-20, clean † | 0.820, 94.5 | 0.933, 33.4 | +13.7% (+4.8% to +26.2%), -64.7% (-75.2% to -50.2%) |
| new-20, ACK loss 1% | 0.911, 9.8 | 0.946, 5.2 | +3.9% (-1.6% to +10.3%), -47.0% (-58.5% to -31.1%) |
| training-20, loss 1% | 0.489, 3.0 | 0.944, 5.3 | +93.1% (+59.8% to +145.4%), +75.2% (+43.3% to +141.6%) |
| new-20, loss 3% | 0.324, 2.4 | 0.928, 4.9 | +185.9% (+119.7% to +304.0%), +102.3% (+66.7% to +166.0%) |

- The comparison goes the other way from expliCCit's: on clean test links
  the fixed rule has 3 to 14% more utilization than Sage and 46 to 65%
  less queueing delay, as on validation (+8 to +13% and -38 to -44% in the
  training families, +17% and -64% on deep, long paths). The
  share rule sits between them on delay (-23 to -32%) with 5 to 17% more
  utilization.
- Under random loss Sage's utilization falls to 0.32 to 0.62, while both
  rules keep 0.93 to 0.96; their higher delay there comes with that
  throughput.
- Log utility against BBRv3, over the five families †: Sage -0.505, the
  share rule +0.198, the fixed rule +0.119, BBRv1 +0.062, Cubic -0.720.
  Against held-out competitors for 60 s: Sage -0.872, the fixed rule
  +0.101.
- Copies of Sage share less evenly than copies of the fixed rule (Jain's
  index 0.942 and 0.911 with 2 and 4 copies, against 0.969 and 0.934) and
  queue less (16 and 21 ms, against 19 and 29 ms).
- What this does not show: that Sage loses to a split in Sage's own
  setting. Here it runs as a drop-in controller outside its home: on Linux
  7.0, not Sage's patched 4.19 kernel; with pacing refreshed at each step,
  not each ACK; with no inference delay; on links it was not trained on
  (its training paths are not known here). And the split here is net1 v4
  with a fixed rule, not expliCCit's net1 and net2.

**With expliCCit's own split, under the frozen settings.** Release
[`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94)
(`experiments/sage-v5-val.json`) runs expliCCit's released net1 and actor
as deployed (`third_party/expliccit-models`) next to Sage, the fixed rule
(u = 0.25) and the share rule (k = 0.5) on net1 v5, and BBRv3, on the
validation links (seed 11) and as 2 and 4 copies. No episode failed.
Nothing is chosen from it. Its plan draws other episode seeds than
`exp-sage-val-72` on the same links and conditions, since the planner's
random stream runs over the controllers, so Sage's rows are a second
draw: Sage's five-family mean against BBRv3 is -0.598 here and -0.589
there. With the flow alone (means over the 21 links; ratios to Sage with
95% bootstrap intervals over links):

| Campaign, condition | Sage: utilization, delay (ms) | expliCCit's policy | against Sage: utilization, delay | Fixed rule, net1 v5 | against Sage: utilization, delay |
|---|---|---|---|---|---|
| training-20, clean | 0.866, 11.1 | 0.657, 15.1 | -24.1% (-35.8% to -13.7%), +36% (-39% to +199%) | 0.982, 12.2 | +13.4% (+4.9% to +25.9%), +10% (-32% to +102%) |
| training-60, clean | 0.902, 9.4 | 0.673, 18.5 | -25.4% (-39.5% to -12.4%), +97% (-14% to +297%) | 0.990, 12.7 | +9.7% (+3.4% to +18.7%), +35% (-3% to +104%) |
| deep-long-20, clean | 0.792, 111.9 | 0.138, 2.1 | -82.6% (-92.3% to -69.9%), -98% (-99% to -95%) | 0.931, 35.7 | +17.5% (+4.3% to +35.6%), -68% (-77% to -45%) |
| new-20, ACK loss 1% | 0.871, 11.6 | 0.593, 15.5 | -31.9% (-45.9% to -19.4%), +33% (-42% to +211%) | 0.982, 12.2 | +12.7% (+4.7% to +24.8%), +5% (-37% to +101%) |
| training-20, loss 1% | 0.475, 2.4 | 0.678, 28.6 | +42.8% (+13.2% to +84.9%), +1112% (+538% to +1819%) | 0.979, 11.2 | +106.2% (+65.9% to +175.1%), +377% (+273% to +546%) |

- expliCCit's claim does not reproduce with its own released policy:
  on clean validation links it has 24 to 25% less utilization than Sage
  (expliCCit: about 5%), and on deep, long paths it nearly stops (0.14).
  Its delay against Sage is higher, but the intervals are too wide to
  size it. In log utility against BBRv3 it is the lowest controller in
  every family (-1.38 to -2.25; Sage -0.42 to -0.98). Its copies share
  least evenly (Jain's index 0.937 and 0.844 with 2 and 4 copies).
- On expliCCit's own case-study link it reaches 0.50 utilization, against
  0.92 for Sage (`exp-crosscheck-93`, link a).
- The fixed rule on net1 v5 keeps 10 to 18% more utilization than Sage on
  clean links. Its queueing delay is no longer below Sage's (+10% and +35%
  in the training families, both intervals including 0). A likely reason,
  not tested: u = 0.25 asks for 2^0.25 = 1.19 times the believed BDP,
  where net1 v4's rule asked for 2^0.125 = 1.09.
- Whether the gap belongs to this environment or to the released
  checkpoint is open: the cross-check (`tools/crosscheck`) runs the same
  policy on the case-study link on the original stack, UNRUN there.

### Deep-long with the original setups' TCP limits

The deep-long family (2.5 times the delay, buffers of 4 to 16 BDP) is the
only held-out family whose links needed more than the old 4 MB in flight.
Its campaign in each run above was rerun on the same episodes (the same
ids) with Sage's and ccBench's 64 MB limits:
[`exp-rule-net1v4-val-deeplong-32`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-val-deeplong-32),
[`exp-share-copies-val-deeplong-33`](https://github.com/Radonux/palaestra/releases/tag/exp-share-copies-val-deeplong-33),
[`exp-sage-val-deeplong-34`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-val-deeplong-34),
[`exp-rule-net1v4-test-deeplong-35`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-deeplong-35),
[`exp-share-test-deeplong-36`](https://github.com/Radonux/palaestra/releases/tag/exp-share-test-deeplong-36) and
[`exp-sage-test-deeplong-37`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-test-deeplong-37). No episode
failed. Log utility minus BBRv3 on deep-long, before and after, on the test
links:

| Controller | before | after | mean of the five families, before and after |
|---|---|---|---|
| fixed rule on net1 v4 | +0.173 | +0.175 | +0.123, +0.123 |
| share rule, k = 1 | +0.209 | +0.200 | +0.208, +0.207 |
| fixed rule on the true BDP | +0.148 | +0.143 | +0.147, +0.146 |
| BBRv1 | +0.013 | -0.027 | +0.070, +0.062 |
| Sage | -0.358 | -0.449 | -0.487, -0.505 |
| Cubic | -1.418 | -1.619 | -0.679, -0.720 |
| BIC | -1.739 | -2.512 | -0.539, -0.693 |
| Scalable | -1.681 | -2.537 | -0.467, -0.639 |

(Rows are from the rule test's runs, except the share rule's, from its own
test, and Sage's and Cubic's, from Sage's; where the three tests overlap
they agree within 0.05.)

- The rules barely move, since their own windows stay near the BDP; what
  changes is the flows that fill the buffer: loss-based controllers fall
  by 0.2 to 0.9 on deep-long, and with them the competitors the rules
  face there.
- Every choice made on validation stands: the fixed rule's u = 0.125
  (+0.137 against +0.136 for u = 0.25, the five-family means), the true
  BDP's and net1 v3's u = 0.375, and the share rule's ordering.
- Against Sage alone on clean deep-long links, Sage now queues more (94.5
  against 72.4 ms) and the fixed rule's lead widens to 13.7% more
  utilization and 64.7% less queueing delay.

### ccperf's experiments

The 59 of ccperf's 121 experiments this emulator expresses, plus 2
(`experiments/ccperf.json`; `docs/architecture.md`), for every kernel
controller, Sage's released policy and the two rules on net1 v5 (u =
0.25; share rule k = 0.5), five times each with fresh seeds, under the
frozen settings: release
[`exp-ccperf-77`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-77)
ran the controllers and
[`exp-ccperf-v5-87`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-v5-87)
the rules on net1 v5 on the same episodes (planned with `--only`; all 61
of BBRv3's rows, run in both, are identical). Nothing was chosen on these
links. Means over each experiment's variants and repeats:

| Experiment | Measure | Fixed rule (v5) | Share rule (v5) | BBRv1 | BBRv3 | Cubic | Sage |
|---|---|---|---|---|---|---|---|
| single flow, 5 paths | utilization | 0.917 | 0.911 | 0.863 | 0.861 | 0.621 | 0.454 |
|  | queueing delay (ms) | 25.0 | 32.8 | 52.2 | 43.5 | 9.9 | 2.0 |
| deep buffers, 1 to 16 BDP | utilization | 0.998 | 0.998 | 0.986 | 0.984 | 0.999 | 0.998 |
|  | queueing delay (ms) | 2.6 | 2.4 | 19.3 | 8.7 | 74.6 | 4.8 |
| random loss, 0.1 to 5% | utilization | 0.972 | 0.977 | 0.978 | 0.684 | 0.415 | 0.528 |
| shallow buffers | utilization | 0.926 | 0.927 | 0.797 | 0.726 | 0.636 | 0.762 |
| rate steps, pulses, random walks | utilization | 0.987, 0.986, 0.972 | 0.989, 0.989, 0.981 | 0.979, 0.968, 0.978 | 0.970, 0.957, 0.975 | 0.962, 0.940, 0.998 | 0.979, 0.933, 0.894 |
| Poisson rate traces | utilization | 0.876 | 0.813 | 0.943 | 0.914 | 0.875 | 0.946 |
|  | queueing delay (ms) | 314 | 365 | 244 | 211 | 523 | 540 |
| synchronous, start-time, RTT fairness | Jain's index | 0.897, 0.824, 0.832 | 0.899, 0.858, 0.880 | 0.855, 0.873, 0.855 | 0.969, 0.965, 0.839 | 0.938, 0.953, 0.668 | 0.791, 0.796, 0.823 |
| against Reno, Cubic, Vegas | share over fair share | 0.763, 0.847, 1.495 | 1.061, 1.088, 1.534 | 1.245, 0.976, 1.739 | 0.823, 0.746, 1.694 | 1.056, 1.076, 1.679 | 0.252, 0.303, 1.186 |

- The rules keep utilization at 0.97 to 0.99 under random loss, on deep
  and shallow buffers and through rate steps, pulses and walks, where
  BBRv3 falls to 0.68 under loss and Cubic and Sage further. On deep
  buffers their queueing delay stays at 2 to 3 ms, against 9 to 75 ms
  for the kernel controllers.
- On the Poisson traces, whose capacity jumps, they queue 314 and 365 ms:
  more than BBRv1 and BBRv3 (211 to 244 ms), less than Cubic and Sage.
  Alone, their mean queueing delay (25 and 33 ms) is mostly the 800 ms
  WINDS satellite path's.
- On ccperf's default path (16 Mbit/s, 15 ms) the fixed rule now uses
  0.998; the 0.85 on net1 v4 came from u = 0.125's margin, not from net1
  ("A small path" below).
- Against Reno and Cubic the fixed rule takes a little less than its fair
  share (0.76 and 0.85), the share rule about its fair share (1.06 and
  1.09), and Sage a quarter to a third. Among copies with different RTTs,
  Jain's index is 0.83 to 0.88, against Cubic's 0.67.
- On net1 v4 (u = 0.125, k = 1; `exp-ccperf-31`, before the send-buffer
  limit) the fixed rule used 0.851 alone, 0.914 on deep buffers and 0.829
  under loss; the kernel controllers' rows there equal these.

### ccBench's suites

ccBench's solo and friendliness suites and three more (random loss,
copies, BBRv1 or BBRv3 as the competitor), for every kernel controller,
Sage's released policy and the two rules on net1 v5, under the frozen
settings: release
[`exp-ccbench-78`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-78)
ran the controllers and
[`exp-ccbench-v5-89`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-v5-89)
the rules on net1 v5 on the same episodes (`--only`; BBRv3, run in both,
has identical metrics), and
[`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92)
ranks them together, the rules on net1 v4 left out. Nothing was chosen on
these links. Games are ccBench's leagues among these controllers
(`docs/architecture.md`, "Benchmark"); utilization and queueing delay are
means over runs and windows:

| Suite | Measure | Fixed rule | Share rule | BBRv1 | BBRv3 | Cubic | Sage |
|---|---|---|---|---|---|---|---|
| solo (2,760 games) | games won | 73.5% | 66.0% | 42.4% | 25.0% | 3.4% | 25.2% |
| | utilization, queueing delay (ms) | 0.945, 15.6 | 0.948, 18.9 | 0.936, 35.7 | 0.923, 42.5 | 0.946, 153.9 | 0.742, 21.5 |
| random loss (600) | games won | 80.8% | 78.2% | 61.0% | 8.3% | 1.3% | 17.7% |
| | utilization | 0.943 | 0.946 | 0.929 | 0.676 | 0.217 | 0.453 |
| friendly, a Cubic flow first (500) | games won | 10.2% | 14.4% | 49.6% | 34.6% | 31.0% | 8.0% |
| | share of capacity | 0.398 | 0.426 | 0.497 | 0.399 | 0.369 | 0.206 |
| BBRv1 or BBRv3 first (400) | games won | 32.5% | 32.2% | 52.5% | 44.5% | 35.5% | 2.2% |
| | share of capacity | 0.514 | 0.584 | 0.541 | 0.382 | 0.441 | 0.143 |
| 2, 4 or 8 copies | Jain's index, queueing delay (ms) | 0.924, 41.6 | 0.925, 50.6 | 0.967, 68.8 | 0.985, 57.8 | 0.934, 126.1 | 0.899, 49.9 |

- Alone and under random loss the fixed rule wins the most games, by its
  low delay at full utilization: ccBench's score is throughput squared
  over delay.
- Against a Cubic flow that started 5 s earlier, with buffers of 1 to 16
  BDP, both rules take less than half (0.40 and 0.43 of the capacity),
  and BBRv1 comes closest to half: a window near the BDP takes a small
  share from a flow that fills the buffer, the gap the validation
  families show with a Cubic flow from the start.
- Against BBRv1 or BBRv3 first, the share rule takes more than half
  (0.58) and the fixed rule about half (0.51).
- On net1 v4 (`exp-ccbench-30`, before the send-buffer limit) the rules
  ranked the same, with less utilization alone (0.932) and under loss
  (0.923).

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

### net1 on a random row split

expliCCit's net1 trainer splits rows at random: 20% for test, then 20% of
the rest for validation. It reports R^2 above 0.9 for every LCF on the
held-out rows. A row split puts a training row of the same episode 10 ms
from each test row.

Release
[`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97)
(`experiments/net1_row_split.py`) trains net1 with expliCCit's settings
on the expliCCit-like data of section 1: its inputs and LCFs, min-max
scaling, batch 256, and up to 500 epochs with patience 20. The data is
regenerated in each cell under the frozen settings, and all six cells
loaded identical data. The two arms use the same fractions and differ
only in the split: once a random row split, once a link split, for seeds
0, 1 and 2. Test rows of each split, mean over seeds (range):

| LCF | R^2, row split | R^2, link split | mean absolute error, row split | link split |
|---|---|---|---|---|
| queueing delay (ms) | 0.971 (0.968 to 0.978) | 0.980 (0.970 to 0.986) | 9.1 | 11.0 |
| queue (packets) | 0.963 (0.960 to 0.967) | 0.985 (0.981 to 0.987) | 21.1 | 17.9 |
| capacity (Mbit/s) | 0.998 (0.997 to 0.998) | 0.976 (0.964 to 0.982) | 0.43 | 1.68 |
| free buffer (packets) | 0.651 (0.644 to 0.655) | -0.215 (-0.264 to -0.127) | 136 | 280 |
| busy fraction | 0.963 (0.952 to 0.976) | 0.754 (0.712 to 0.813) | 0.007 | 0.021 |
| one-way delay (ms) | 0.999 (0.999 to 0.999) | 0.992 (0.985 to 0.997) | 0.32 | 0.66 |
| receive rate (Mbit/s) | 0.998 (0.997 to 0.999) | 0.994 (0.990 to 0.997) | 0.66 | 1.22 |

- **What a row split overstates.** On links net1 has not seen, it does
  worse than the row split shows:
  - free buffer falls from 0.65 to below 0, worse than predicting the
    mean;
  - busy fraction falls from 0.96 to 0.75;
  - capacity's mean absolute error is 4.0 times as large, one-way
    delay's 2.1 times and receive rate's 1.9 times.
  Queue and queueing delay are the exception: there the row split is no
  better.
- **Training length.** The row split's validation rows sit next to its
  training rows, so its validation loss kept improving: it stopped after
  57 to 88 epochs, against 21 to 24 for the link split.
- **The claim itself.** Even on the row split, free buffer reaches only
  0.65, so the R^2 above 0.9 does not reproduce here for every LCF.
- **Limits.** This is our expliCCit-like data, not expliCCit's. Its LCFs
  are computed by our labeller, so the comparison is with the method, not
  with expliCCit's numbers.

### Case study: expliCCit's 24 and 48 Mbit/s link

expliCCit's draft traces its policy's under-utilization on this link to
capacity underestimated by 25%. It reports that replacing that belief
with the truth restores utilization. Release
[`exp-expliccit-case-42`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-42)
(`experiments/expliccit-case.json`) repeats the experiment with its
released net1 and actor (`third_party/expliccit-models`), as deployed.

The link: capacity alternating 24 and 48 Mbit/s every 7 s, one-way delay
40 ms, a 1000-packet buffer, 45 s. It has no randomness, so episode seeds
give the same run. The policy is fragile on it: in a local check its
utilization fell from 0.47 to 0.04 when a few TCP settings changed, and a
patch that gave 0.04 in a local run on 2026-10-31 (mean queueing delay
from the truth) gives 0.57 here, which the TCP settings alone do not
explain. One run on this link is one sample of a policy that collapses or
holds about 0.5. So each policy also ran on the eight links around it
(one-way delay 39 to 41 ms by buffer 980 to 1020 packets). The link is
also one of the 120 paths of the teacher campaign (section 1), so our
models may have trained on it.

| LCFs from the truth | utilization, expliCCit's link | median over the 9 links | range | mean queueing delay (ms), expliCCit's link |
|---|---|---|---|---|
| none | 0.483 | 0.483 | 0.340 to 0.534 | 4.44 |
| capacity | 0.425 | 0.431 | 0.425 to 0.443 | 1.21 |
| busy fraction | 0.707 | 0.707 | 0.703 to 0.707 | 4.63 |
| queue | 0.521 | 0.527 | 0.521 to 0.575 | 4.95 |
| mean queueing delay | 0.573 | 0.573 | 0.566 to 0.581 | 5.21 |
| free buffer | 0.062 | 0.019 | 0.018 to 0.064 | 6.92 |
| busy fraction, queue, mean queueing delay and free buffer | 0.707 | 0.708 | 0.694 to 0.722 | 2.94 |
| all seven | 0.621 | 0.621 | 0.615 to 0.639 | 1.20 |

Its beliefs along its own run on expliCCit's link, against the truth
(medians over the run; capacity in the first two 7 s segments):

| LCF | released net1 | truth |
|---|---|---|
| queue | 19.4 packets | 0 packets |
| mean queueing delay | 15.2 ms | 0.71 ms |
| free buffer | 128 packets | 1000 packets |
| busy fraction | 1.00 | 0.51 |
| capacity | 27.3 and 41.6 Mbit/s | 24.1 and 48.1 Mbit/s |
| one-way delay | 39.9 ms | 40 ms |
| receive rate | 21.5 Mbit/s | 15.6 Mbit/s |

The released net1's value range for each LCF is: queueing delay 0 to
5123 ms, queue 0 to 10240 packets, capacity 2.4 to 385 Mbit/s, free
buffer 0 to 10240 packets, busy fraction 0.2 to 1, one-way delay 5 to 40
ms, receive rate 0 to 385 Mbit/s. A free buffer of 1000 packets is inside
its trained range, so the collapse when free buffer comes from the truth
is a problem of the LCFs' joint range, not of any one LCF's range.

We also ran our own models of the time (M12), trained in UML, on the same
link, in a local run on 2026-10-31 that is not in a release; those models
are superseded. In the table, the queue group is mean queueing delay,
queue and free buffer.

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

**Reading.**

- The draft's capacity diagnosis does not reproduce with the released
  model here. Its capacity belief is 13% high at 24 Mbit/s and 14% low at
  48 Mbit/s, not 25% low throughout, and taking capacity from the truth lowers
  utilization (0.43 against 0.48, on all nine links).
- Comparing its beliefs with the truth localized a different fault:
  busy fraction (1.00 against 0.51) and queue (19 packets against 0).
  Taking busy fraction from the truth gives 0.71, on all nine links
  (0.70 to 0.71), the steadiest repair of any.
- The released net1 was trained on expliCCit's own data, so this result
  mixes transfer between emulators with the method.
- The LCFs are coupled, so replacing one belief can describe a path that
  cannot exist. A net2 trained on net1's beliefs has adapted to them, so
  the truth is not automatically a repair. On our own models some
  patches hurt: busy fraction from the truth drops the cloned policy to
  0.144 utilization, and with the queue group from the truth TD3+BC's
  drop fraction rises from 0 to 0.385.
- The deployed rounding is not the fault. expliCCit's deployed
  controller rounds net1's first six LCFs to integers, busy fraction
  among them, a share it so turns into 0 or 1. Release
  [`exp-expliccit-case-91`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-91)
  runs the released policy without that rounding: with no LCF rounded it
  uses 0.416 of expliCCit's link (0.299 to 0.498 on the nine), with every
  rounding but busy fraction's 0.300 (median 0.507 over the nine, 0.300
  to 0.540), against 0.483 as deployed. Taking busy fraction from the
  truth, still rounded, is what repairs it. The same release repeats
  release 42's rows but two (busy fraction from the truth 0.708 against
  0.707, the group of four 0.713 against 0.707): a true busy fraction of
  0.505 sits at the rounding's edge, where the rebuilt graph's float
  order can flip it.

### Alarms that need no ground truth (R2)

Can a deployed net1 tell when its beliefs are wrong? Release
[`exp-r2-alarms-81`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-alarms-81)
(`experiments/r2_alarms.py`) scores alarms computed only from net1's output
and the sender's own measurements: physical invariants among the beliefs
(receive rate at most capacity times busy fraction; mean queueing delay
equal to queue over capacity; queue plus free buffer constant over an
episode), each distribution's entropy, and delayed checks against the
delivery rate and min RTT measured three smoothed RTTs later. Each alarm's
threshold is the 95th percentile of its episode mean on net1 v4's
validation links. A belief about an LCF is wrong in an episode when the
truth lies outside net1's central 90% set in more than half of its steps.
Detection is counted among the episodes wrong about the LCF, false alarms
among the in-distribution test links' episodes right about it.

- False alarms stay near the nominal 5% per alarm on the test links, and
  0.07 to 0.17 per LCF when any alarm counts.
- The alarms catch net1 v3 where its belief contradicts what the flow
  measures: one-way delay on buffers of 8 to 16 BDP (0.83 of 168 wrong
  episodes, AUC 0.91, mostly the min RTT check), queueing delay at 61 to
  90 ms (0.89 of 90, AUC 0.92, the queue-over-capacity invariant), and
  capacity under 3% to 6% loss (0.63 of 287, AUC 0.79).
- They miss net1 v3's capacity under competition, the fault that
  retraining net1 alone repaired (net1 v3 to v4): with a Cubic flow from 0 s, 0.12 of 43 wrong episodes
  (AUC 0.52); with a BBRv3 or Westwood competitor, 0.11 of 388 (AUC 0.46).
  A capacity believed at 0.59 of the truth is still above the flow's own
  share, so nothing it measures contradicts it, and net1 v3 is confident
  there, so its entropy does not rise either (the explanation is
  inferred, not tested).

So alarms of this kind cover beliefs that the flow's own measurements can
refute, and not beliefs that those measurements are consistent with; the
latter would need the flow to probe, which is outside this work.

### A small path: ccperf's default link

The fixed rule on net1 v4 (u = 0.125) used 0.854 of ccperf's default link,
16 Mbit/s with a 15 ms base RTT and a 20-packet buffer, where BBRv3 uses
0.98 (`exp-ccperf-77`). net1 does not underestimate that path's BDP. Run
locally (2026-11-05, seed 3, 30 s, the guest frozen at `1365f0e`), net1's
median beliefs after the first half second, against the truth:

| Rule | believed capacity (16 Mbit/s) | believed one-way delay (7.5 ms) | believed BDP (20 packets) | window | utilization |
|---|---|---|---|---|---|
| net1 v4, u = 0.125 | 15.4 | 7.67 | 19.7 | 22 | 0.854 |
| net1 v5, u = 0.125 | 16.0 | 7.67 | 20.5 | 22 | 0.914 |
| net1 v5, u = 0.25 | 16.5 | 7.75 | 21.3 | 25 | 0.998 |

The shortfall is the rule's margin: a window of 1.09 times the BDP leaves
two packets of slack on a 20-packet path, too little for whole-packet
windows paced at the window over the smoothed RTT, and u = 0.25, the
choice on net1 v5, fills it (0.998 in the release run too,
`exp-ccperf-v5-87`).

### Repairing a fault in net1 alone (R3)

net1 v3's fault and its repair, as a case study of fixing the estimator
without touching net2:

- **Symptom.** The fixed rule on net1 v3 lost to BBRv3 with a Cubic flow
  from the start (-0.387 in log utility, "net2 on the fixed net1" above).
- **Localization.** net1 v3's capacity belief there was 0.59 of the truth
  (10th to 90th percentile 0.30 to 0.85), its 90% statements holding 26%
  of the time: in its training the Cubic flow always came after the
  observed flow had measured the link alone.
- **Verification.** The same rule on the true BDP gained +0.476 there, so
  the rule was sound and the belief was the fault.
- **Repair.** net1 retrained on the broad data (net1 v4), net2 unchanged:
  +0.43 on the test links with a Cubic flow from the start, +0.04 under
  net1 v3's own conditions ("net1 on a broad set of conditions").

Under the frozen settings, release
[`exp-r3-capacity-head-val-86`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-val-86)
repeats it on the validation links (seed 11, each net1 at its best u):
the rule on net1 v3 has -0.555 against BBRv3 with a Cubic flow from the
start and -0.040 over the five families; on net1 v4, -0.037 and +0.138;
on the true BDP, +0.058 and +0.165.

**A cheaper repair: the capacity head only.** Release
[`exp-r3-capacity-head-83`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-83)
(`experiments/r3_capacity_head.py`) retrains only the capacity head of
each of net1 v3's members (one layer of 64 by 64 bins), on the same broad
data, every other weight frozen and the other six LCFs checked
bit-identical. It took 439 s of wall time on 2 CPU threads (268 s of it
computing the frozen features once), against 3,571 s for net1 v4's whole
training. Capacity's calibration is repaired, its sharpness less so, and
its mean now runs high (validation rows; net1 v4 in parentheses):

| net1 | 90% coverage | R^2 | CRPS (Mbit/s) | median belief / truth |
|---|---|---|---|---|
| v3 | 0.62 | 0.67 | 12.6 | 1.01 |
| v3, capacity head retrained (bins refitted) | 0.93 (0.90) | 0.73 (0.79) | 10.3 (8.3) | 1.15 (1.04) |

In closed loop it recovers little of what retraining net1 whole did: the
rule on it has -0.392 with a Cubic flow from the start and -0.009 over the
five families (bins refitted, u = 0.25; with net1 v3's bins kept, -0.465
and +0.005 at u = 0.125), a gain of about +0.16 and +0.04 where net1 v4
gained +0.52 and +0.18. A new head on net1 v3's frozen features cannot
supply what those features never learned, the cues of competition from
the start (the explanation is inferred, not tested): the repair has to
reach the representation, which the full retrain does in an hour of CPU.

**Would an alarm have flagged the fault first?** No. The alarms that need
no ground truth (R2 above) detect net1 v3's wrong capacity with a Cubic
flow from the start in 12% of the wrong episodes (AUC 0.52), chance level:
a capacity believed too low is still above the flow's own share, so
nothing the flow measures contradicts it. The fault was found with ground
truth (the oracle rule and net1's coverage), which a deployment would not
have.

### net2 by recurrent PPO (did not pass its gate)

The design (`src/expliccit_env/train/ppo.py`, `recurrent.py`):

- **Actor.** A GRU actor that sees net1's binned beliefs and its own
  previous action, never raw step messages or the true LCFs. It sets u,
  the next window as 2^u times the believed BDP, once per min RTT clamped
  to 10 to 50 ms, and holds it in between inside the exported graph.
- **Critic.** It also sees the true LCFs (an asymmetric critic).
- **Reward.** Log utility per step from the ground truth, discounted by
  exp(-dt / tau).
- **Rollouts.** K rollouts per link on one seed with different noise, and
  a group-mean baseline.
- **Start.** A clone of the fixed rule.

The cloned start matches the rule within noise over 126 validation
pairs: +0.010 on net1 v4 (`exp-net2-ppo-debug-80`), and +0.007 and -0.003
on v5 (`exp-net2-ppo-debug-v5-84`, from u = 0.125; each stage-1 cell). The
start holds each decision for a min RTT, where the rule decides every
step.

**Stage 1** ([`exp-net2-ppo-search1-90`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-ppo-search1-90)):
tau in 1, 3 and 10 s by exploration sigma in 0.05, 0.1 and 0.2, from
u = 0.25 on net1 v5, 40 iterations of 32 rollouts. The evaluation is
noise-free, on the validation links' new-conditions family, paired with
the rule. No cell moved far from the rule: last evaluations ranged from
-0.027 to +0.014. By the rule stated in advance, the best last
evaluation, stage 1 kept tau = 1 s and sigma = 0.05. Every cell shows the
same pattern, at the last evaluation (minus the rule):

| Cell | alone, ACK loss 1% | alone, loss 0.1% | BBRv1 from 4 s | Cubic from 0 s | two Cubic flows | mean u |
|---|---|---|---|---|---|---|
| tau 1 s, sigma 0.05 | +0.033 | +0.029 | +0.071 | -0.033 | -0.029 | 0.22 |
| tau 1 s, sigma 0.2 | +0.076 | +0.058 | +0.018 | -0.187 | -0.141 | 0.07 |
| tau 10 s, sigma 0.1 | +0.046 | +0.037 | +0.036 | -0.024 | -0.045 | 0.19 |

Every cell drifted toward a lower u that barely depends on the state
(spread across states 0.015 to 0.037), faster for larger sigma. That is
what a lower constant u does: it gains alone and loses against Cubic.

The training mix explains it. Half the links have competition, split
evenly over Cubic, BBR and 2, 3 or 4 copies. Cubic is therefore a tenth
of the links, and on the rest a lower u is better. PPO's clipped step in
the mean grows with sigma, which is why larger sigma drifted further.

A local check (not a release) ruled out a bias from the exploration
noise itself. With the guest's per-step noise, the best constant u on the
validation links' training family stays 0.125 for sigma up to 0.2, and
the noise hurts a low u most.

No cell learned to tell competition apart, which is what the gate asks
for.

**Stage 2** ([`exp-net2-ppo-search2-96`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-ppo-search2-96)),
at tau = 1 s and sigma = 0.05, crosses three axes:

- learning rate 1e-4, 3e-4 or 1e-3;
- K = 2 or 4, with 32 rollouts per iteration either way;
- the training mix: stage 1's, or competition on 80% of the links
  (Cubic, BBR and two copies, evenly).

Last evaluations ran from +0.006 to +0.017. The best three are within
0.001 of each other, far inside the evaluation's noise, and all use the
competition mix:

| Cell | last evaluation | Cubic from 0 s | two Cubic flows | BBRv1 from 4 s | alone, ACK loss 1% | mean u |
|---|---|---|---|---|---|---|
| learning rate 1e-4, K = 4 | +0.017 | -0.061 | -0.079 | +0.096 | +0.062 | 0.15 |
| learning rate 1e-3, K = 2 | +0.016 | +0.001 | -0.031 | +0.073 | +0.024 | 0.23 |
| learning rate 1e-3, K = 4 | +0.016 | +0.012 | -0.002 | +0.077 | +0.004 | 0.26 |

The first still drifts to a low u. The last is the first cell to gain
against Cubic from the start without drifting down.

**Seed stage and gate.** A 0.001 gap should not decide the gate, so all
three candidates go to the seed stage (`experiments/net2-ppo-seeds.json`):
each trains again on seeds 1, 2 and 3, which stage 2 did not use. The gate
(`experiments/net2-ppo-gate.json`, rule in
`experiments/net2_ppo_gate_report.py`, both written before those results)
scores the nine policies against the rule on every validation family and
as copies, with campaign seed 21 (the search used 11).

- **Kept:** the candidate with the best five-family mean over its seeds.
- **Passes** only if it beats the rule in cubic-first and in the new
  family's competition conditions, each with its 95% interval over links
  above 0. It must also be at or above the rule on the five-family mean,
  and not worse with copies beyond its interval.
- **Otherwise** net2 stays the fixed rule and the share rule.

**The gate does not pass.** The seed stage
([`exp-net2-ppo-seeds-98`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-ppo-seeds-98);
policies in release `model-net2-ppo-candidates`) and the gate
([`exp-net2-ppo-gate-99`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-ppo-gate-99))
ran with no failed episode. The rule's own lead over BBRv3 there (+0.159
over the five families) matches its earlier seed-21 run (+0.149). Per
link, mean over three seeds, minus the rule, with 95% intervals over the
21 validation links:

| Candidate | five-family mean | training-60 | cubic-first | new, competition | copies |
|---|---|---|---|---|---|
| learning rate 1e-4, K = 4 | -0.000 | +0.027 (+0.016 to +0.040) | -0.068 (-0.099 to -0.036) | -0.030 (-0.074 to +0.007) | +0.041 (-0.098 to +0.163) |
| learning rate 1e-3, K = 2 (kept) | +0.004 | +0.011 (+0.006 to +0.017) | -0.007 (-0.032 to +0.015) | +0.005 (-0.029 to +0.037) | -0.127 (-0.235 to -0.031) |
| learning rate 1e-3, K = 4 | -0.001 | +0.011 (+0.005 to +0.018) | -0.020 (-0.040 to -0.000) | +0.002 (-0.032 to +0.036) | -0.121 (-0.230 to -0.028) |

- **Verdict.** The rule kept the candidate with the best five-family mean
  (+0.004). It does not beat the rule against competition: both
  competition intervals include 0. Its copies are worse: -0.07 with 2
  copies and -0.18 with 4, a count the competition mix never trained on.
  net2 stays the fixed rule and the share rule, and the final test runs
  without a learned net2.
- **Pattern.** It is stage 1's, on a smaller scale. Every candidate
  gains alone and ties or loses against Cubic.
  - The candidate that drifted to a low u (learning rate 1e-4) gains most
    alone and loses most with Cubic first. It is the only one not worse
    with copies.
  - The two that stayed near u = 0.25 tie the rule against competition
    and lose with copies.
- **Stage 2's gain against Cubic did not repeat.** Learning rate 1e-3
  with K = 4 had +0.012 with Cubic from the start in stage 2. Its new
  seeds had -0.016 to -0.033 at training time and -0.020 at the gate:
  seed noise.
- **What it says.** PPO from the rule, with this actor and reward, finds
  the training mix's best offset of u and nothing that depends on the
  state. It did not learn, in 40 iterations of 32 rollouts, to tell a
  copy of itself from a buffer-filling competitor, which is the gain the
  gate asks for. Whether longer training or another signal would is
  UNRUN.
- **What came next.** Two changes follow from this diagnosis
  (`experiments/net2-ppo-team.json`, `net2-ppo-flows.json`; UNRUN at the
  time of writing).
  - **A team reward for copies.** Copies are scored with every flow
    running the policy. A copy that learns from its own utility gains by
    pushing harder than the others, which lowers what they all get. With
    the team reward, a copy learns from the copies' mean utility, while a
    flow facing Cubic or BBR keeps its own.
  - **A value network that also counts the other flows on the link.**
    It sees how many copies, stock TCP flows and UDP flows are active at
    each decision; the actor never sees this.
  - Online TD3+BC from the rule already lost to it on net1 v4
    (`exp-net2v4-val-16`), and stays the next fallback.

### Inference cost per decision

Measured outside the guest with the guest's own runtime and loading steps
(`tools/policy-bench`: tract 0.22.5, inputs by name, the graph optimized
once and then run once per decision with its state carried), on one core
of the development container (an Intel Xeon, 4 vCPUs, shared, so times
are noisy; not a CI runner), 20,000 calls after 1,000 warm-up calls, 2026-11-05:

| Policy | median | 90th percentile | 99th percentile |
|---|---|---|---|
| fixed rule on net1 v5 (`model-net1v5`, `rule-u0.25.onnx`) | 1.07 ms | 1.18 ms | 1.29 ms |
| share rule on net1 v5 (`share-u0.25-k0.5.onnx`) | 1.23 ms | 1.70 ms | 2.68 ms |
| Sage's released policy (`exp-sage-model-24`) | 0.84 ms | 0.92 ms | 0.97 ms |
| recurrent net2 holding its decision, on net1 v4 (GRU of 64) | 1.15 ms | 1.23 ms | 1.33 ms |

net1's three members (width 512) dominate: the split costs about 0.2 ms
more than Sage per decision, about 1.1 ms in all, a tenth of a core at one
decision per 10 ms step. In the guest, under `time-travel=inf-cpu`,
inference costs no virtual time, so none of this shows in the episodes'
results (`docs/validation.md`). Reproduce with `cargo run --release
--manifest-path tools/policy-bench/Cargo.toml -- POLICY.onnx`.

## 8. Next

"net2 on the fixed net1" is the first run of the design in
`docs/methodology.md`: an honest net1, fixed, and net2 setting a window
target from its beliefs. Three seeds of the learned net2, 60 iterations
of online training and arm A2 (the means) have run; on validation none
beat the constant. What it leaves open, in the order the plan needs it:

- A net2 that tells a copy of itself from a buffer-filling competitor
  (section 7, "net2 on net1 v4"): the share rule gains under competition
  and loses against its own copies, on validation and test alike, and a
  hand-made switch between a cooperative mode and the share rule kept
  neither gain (`exp-mode-val-19`). Told the number of flows N, a rule on
  the truth keeps both (`exp-fairshare-used-val-23`). N is not added to
  the LCFs (it describes the other endpoints, not the path, and changes
  with their arrivals), so this is net2's to infer over time: a learned
  net2 with memory, which sees how the others respond to its own actions. Online TD3+BC
  from the rule lost on validation; a learned net2 needs a lower-variance
  signal than broad 30 s rollouts give its critic (paired rollouts, or a
  policy search over a few parameters).
  net1 v4's data still lacks competitors that arrive before the observed
  flow, and its bins end at the training ranges of delay and buffer.
- The human reading (a few percentiles per LCF) as net2's input, against
  the means and the distributions.
- Arm A1 for the learned net2 (trained and run on the truth), and A4
  (raw measurements), to measure the costs of estimation and of the
  split for a learned decision, not only for the constant rule.
- Several learned flows on one bottleneck, for fairness between copies:
  the constant aims at the link's BDP, not a fair share.

`docs/methodology.md` has the claims (section 1), the plan (section 3),
the arms (section 5) and the status (section 6).
