# Methodology

The idea under test is expliCCit's: split a learned congestion controller
into an estimator (net1) and a decision maker (net2). net1 turns standard
sender-side measurements into beliefs about a few physical properties of
the path, the latent congestion factors (LCFs), each of which has a ground
truth in emulation. net2 sets the congestion window from those beliefs
alone. The goal of this project is to show that this split works: that it
costs little against an unsplit controller, and that it makes a
controller's decisions explainable and its faults repairable.

The work has two phases with different rules:

1. **Reproduction.** Run expliCCit's artifacts and setup in this
   environment and name every divergence. The point is to show that UML
   reproduces its setup and roughly its results, so that results obtained
   here can be compared with it. Here we stay faithful to the original.
2. **Research.** After that, any design choice that serves the goal is
   open: training algorithm, network size, reward, action space, and
   offline or online training. Only the split itself is fixed.

`docs/training.md` is the workflow and the results so far; this page is
the reasoning and the plan.

## 1. What "works" means

| # | Claim | How it can fail |
|---|---|---|
| H1 | Sufficiency: a policy acting on the true LCFs performs about as well as the best policy we can train on raw measurements | the LCF set omits something the decision depends on |
| H2 | Estimability: net1 recovers the LCFs from sender-side measurements, where the measurements allow it | some LCFs are not identifiable at some operating points (section 2) |
| H3 | Honesty: net1's stated confidence matches its actual error | over-confidence, above all where the link is not kept busy |
| H4 | Faithfulness without leakage: net1's output carries only its beliefs, and net2 acts on nothing else | soft outputs can pass on information beyond the LCFs |
| H5 | Repairability: a fault is traced to one LCF and fixed there, without retraining net2 from scratch | errors only show in combination |
| H6 | Affordability: small loss against a black box with the same inputs, data and training | |

## 2. Why net1 reports beliefs, not values

A sender learns the link's capacity only while it keeps the link busy.
Below capacity with an empty queue, the delivery rate equals the sending
rate, so the measurements bound capacity from below and say nothing more.
The minimum RTT has the opposite limit: it cannot be measured while a
queue persists. No operating point reveals every LCF, so a controller has
to probe, and probing costs delay and loss.

So a single value per LCF cannot be the interface. When capacity is
unidentifiable, "24 Mbit/s" misleads; "at least 18 Mbit/s" is what the
measurements support. net1 must report a belief, and net2 must be able to
learn when reducing its uncertainty is worth the cost of probing.

The data already show the limit. The current interval net1 reports a
capacity interval 3.6 Mbit/s wide on steps where the link is kept busy and
40.3 Mbit/s wide where it is not. On the validation paths, net1's
capacity estimate has R^2 0.78 on the teachers' episodes but 0.45 on a
cloned policy's and -0.01 on online-trained policies', which leave the
link idle more often.

## 3. Training, in order

net1 and net2 are trained one after the other. net1's training never
involves net2. net2 is trained afterwards on top of a fixed net1, on
net1's actual outputs, so it learns to act on net1's beliefs, errors
included. This is the sequential regime of concept bottleneck models (Koh
et al., 2020). Training net2 on true LCFs instead would leave it to meet
net1's errors for the first time at test time.

Adapting net2 to net1 is safe only because net1 is honest. If net1's
belief is the true conditional distribution of the LCFs given what the
sender has seen, it is a sufficient statistic for the decision, and what
net2 learns about net1's errors is how to act under stated uncertainty,
not net1's quirks. Honesty is what keeps the interface meaningful.

### net1: accurate and honest

net1 has two goals and no others: be accurate, and state its confidence
truthfully.

- **Output.** For each LCF, a probability distribution over fixed bins of
  that quantity (a softmax head; log-spaced bins for rates and delays,
  about 32 per LCF). A histogram can say what a mean and a spread cannot:
  "at least X" when the link is idle, a skewed belief, or two candidate
  values.
- **Loss.** The log score (cross-entropy against the bin holding the true
  value). Its expected value is the entropy of the true conditional
  distribution plus the KL divergence from it to net1's belief, so it is
  minimized only by the true distribution: overstating or understating
  confidence always costs more (Gneiting and Raftery, 2007). One loss
  covers both goals: the entropy term is the uncertainty the measurements
  cannot remove, and the KL term is net1's own error.
- **Independent quantities.** Predict a minimal set of physical
  quantities (capacity, queue, buffer size, propagation delay,
  cross-traffic share) and derive the rest (free buffer is buffer size
  minus queue; mean queueing delay is close to queue over capacity).
  Separate distributions over coupled quantities lose their dependence and
  can describe paths that cannot exist.
- **Data.** Logs of existing controllers, collected on traces: the
  kernel's controllers, BBRv3, and deliberately varied controllers that
  probe, back off and hold, so the data cover the states a reasonable net2
  may reach. net1 needs no online training, but it is honest only on the
  kinds of states it was trained on. Held-out paths and held-out
  controllers test that.
- **Unfamiliar states.** The log score teaches the uncertainty the
  measurements leave, not uncertainty about inputs never seen. A small
  ensemble of net1s widens the belief where they disagree. If net2 later
  drives the path into states the data do not cover, adding those episodes
  to net1's data changes its data, not its objective.
- **Evaluation.** Accuracy: error and R^2 of the median per LCF. Honesty:
  held-out log loss, coverage of central intervals at several levels, and
  the histogram of probability integral transform values. Sharpness:
  entropy or interval width. Each reported separately for steps where the
  link is busy and where it is not, because average calibration can hide
  over-confidence exactly where identifiability fails.

Status: the current net1 predicts the 5th, 50th and 95th percentiles
(pinball loss, then a conformal correction on validation paths). The
histogram head, the minimal set of quantities, the varied controllers and
the ensemble are UNRUN.

### net2: what a good congestion controller optimizes

| Property | Why it matters | Measure |
|---|---|---|
| Throughput | the purpose | utilization, log throughput |
| Latency | a standing queue delays every packet | mean and 95th-percentile queueing delay |
| Loss | lost packets waste capacity and harm other flows | drop fraction, retransmissions |
| Fairness and friendliness | the link is shared: with copies of itself, with Cubic and BBR, without starving or being starved | Jain's index, share against other controllers |
| Convergence | capacity and competition change | time to a new capacity or a fair share |
| Stability | oscillation hurts every flow | variance of the sending rate |
| Robustness | random loss, jitter, ACK aggregation, unfamiliar paths | performance under each; never collapse, never flood |

The reward is one scalar; evaluation covers every row. The current reward
is linear in utilization, minus penalties on queueing delay relative to
the base RTT and on the flow's drop fraction. A linear throughput term
gives competing flows no reason to share. If each flow maximizes the log
of its throughput minus a price for congestion, competition settles on
proportionally fair shares (Kelly, 1998); Copa and PCC Vivace build their
objectives on the same idea. The plan is therefore log throughput minus
penalties on relative queueing delay and loss. A single-flow reward cannot
teach fairness, so training includes competing flows: stock controllers
now, copies of the policy once the environment runs more than one agent
per kernel.

Training is online, in closed loop: the tuning in `docs/training.md` ran
3,216 closed-loop episodes of 30 s in about 35 minutes on 4 cores, so
offline training is not forced on us as it was on Sage, which emulated in
real time. The results so far
(`docs/training.md`) point to the design changes to make first:

- Act once per RTT on an absolute target. The current action
  multiplies the window every 10 ms, so a bias of 0.001 per step moves the
  window by 7% a second. A target such as a multiple of the estimated
  bandwidth-delay product plus a queue target does not accumulate errors.
- Explore over an RTT, not per step. Independent noise of 0.005 every
  10 ms moves the window by about 1% over 100 ms, so it barely explores at
  the window's own time scale.
- Use teacher logs only to start. Cloning and offline RL from them
  reached +0.55 to +0.64 utility on held-out paths, against +0.82 for
  BBRv1.

### What net2 sees

net2 takes the full distributions (7 LCFs times about 32 bins). Whether
it needs them is an experiment, not an assumption: train net2 once on the
full distributions and once on the human reading of section 4 (a few
percentiles per LCF, in physical units). If they perform the same, net2
takes the reading, and what an engineer reads is exactly what net2 acts
on. If the full distributions do better, the difference measures what the
reading leaves out, and we report what that is.

## 4. Interpretability

**What an engineer reads.** Each row of net1's output is a probability
distribution over one physical quantity, so every number has a stated
meaning ("the probability that capacity lies in [22, 24) Mbit/s is 0.31")
and can be checked against the truth. The human reading is a statement
taken from it with equal-tailed intervals:

- "Capacity is about 24 Mbit/s (90% interval: 22 to 26)."
- "Capacity is at least 18 Mbit/s (95%). The link was not kept busy, so
  nothing bounds it from above."
- "Capacity is either about 24 or about 48 Mbit/s (60% and 40%)."

Each statement comes with net1's measured honesty on held-out paths, for
example "net1's 90% statements held 89% of the time". The interpretability
claim rests on that measurement, not on the wording.

**Leakage.** A soft output can carry information beyond the concept it
names, and a downstream network can use it. With the log score, net1's
output tends to the true conditional distribution of the LCFs, so two
histories with the same belief produce the same input to net2. The width
of a belief does reveal how informative the measurements were, for
example whether the link was kept busy; that is information about the
LCF, and net2 needs it to decide when to probe. Leakage is still tested:
train probes to recover quantities that are not LCFs (the controller that
generated the episode, raw measurement columns) from net1's output, and
compare with what the human reading alone allows.

**Explaining a decision.** net2 is an exact function of what it sees, so a
counterfactual is exact: "had the capacity belief been 24 plus or minus 1
Mbit/s instead of at least 18, the window would have grown 12% instead of
3%" is net2 evaluated on the changed belief, with no surrogate model. Exact
Shapley values over the seven LCFs, each LCF's bins treated as one player,
take 2^7 = 128 evaluations of net2 per decision.

**Repair.** A fault traced to one LCF is fixed by retraining that part of
net1, then fine-tuning net2 lightly to the corrected beliefs. Replacing a
belief with the truth at run time can confirm a diagnosis, with two
cautions found in the case study (section 6): LCFs are physically
coupled, so a patch must change a consistent group (one patched LCF can
describe a path that cannot exist); and a net2 trained on net1's beliefs
has adapted to them, so the truth is not automatically a repair.

## 5. Experiments

**Arms** (same paths, seeds and training budget):

| Arm | net2 sees | Isolates |
|---|---|---|
| A1 | the true LCFs (an oracle, trained and run on the truth) | upper bound for the LCF set (H1) |
| A2 | net1's single-value estimates | |
| A3 | net1's distributions | A3 against A2: what honest uncertainty is worth; A1 against A3: the cost of estimation (H2) |
| A4 | raw measurements (a black box, same training) | A4 against A3: the cost of the split (H6) |
| A0 | the full true state of the path, beyond the LCFs | A0 against A1: what the LCF set omits |

**Other experiments:**

- Reading ablation: net2 on the full distributions against net2 on
  the human reading (section 3).
- Leakage probes: section 4.
- Repair: inject bias, noise or lag into one LCF at run time; measure
  the damage, whether it is detected, and recovery after retraining that
  part of net1.
- Identifiability pairs: paths the sender cannot tell apart that
  differ in one LCF, for example a 48 Mbit/s link carrying 24 Mbit/s of
  constant-rate UDP against a 24 Mbit/s link alone. They show which
  quantities can be LCFs at all.

**Statistics.** The unit is a path and a seed, never a step. Arms are
paired on the same paths and seeds. Every learned model is tuned on
validation paths before it is compared (AGENTS.md), and reported once on
held-out test paths, with at least four training seeds and bootstrap
intervals over seeds and paths. Every result is stated with its reward and
path distribution.

## 6. Where things stand

**Reproduction**

| expliCCit's result | Here |
|---|---|
| net1 estimates every LCF with R^2 above 0.9 | 5 of 7 (0.95 to 0.99); busy fraction 0.45 and free buffer 0.09 |
| The split costs about 5% utilization and 9% queueing delay against Sage | UNRUN: Sage's released checkpoint is available locally, but not yet exported to ONNX and run here |
| On a 24/48 Mbit/s link (one-way delay 40 ms, 1000-packet buffer), under-utilization is traced to capacity underestimated by 25%, and replacing that belief with the truth restores utilization | Not reproduced. The released policy runs at 0.49 utilization, and its capacity belief is about right; replacing it with the truth gives 0.43. The wrong beliefs are busy fraction (1.0 against 0.53 true), queue (18 packets against 0), free buffer and queueing delay; replacing busy fraction alone gives 0.72 |
| Sage's state, as computed here, equals `sage.cc`'s | UNRUN (`tests/test_sage.py`, needs a Sage checkout and g++) |

The released net1 was trained on expliCCit's own data, so the case study
tests transfer from their emulation to this one as much as the method.
Why that net1 misreads busy fraction and queue here is the main open item
of the reproduction.

**Results so far** (`docs/training.md`): tuned and checked over seeds,
learned policies reach +0.55 (cloning) to +0.64 (TD3+BC) utility on
held-out paths, against +0.82 for BBRv1 and +0.29 for expliCCit's released
policy; online fine-tuning neither helps nor hurts the cloned policy
(+0.03, 95% interval -0.13 to +0.16). None of these is the design above;
they set the baseline it must beat.

**What the environment supports**

| Need | Status |
|---|---|
| Exact, time-indexed ground truth of the bottleneck | yes: every arrival, drop and departure, both directions, by flow and drop cause |
| Paths: capacity traces, delay, buffer, random loss, cross traffic | yes; no AQM or ECN, delay jitter, bursty loss or multiple bottlenecks |
| Policies of any network design, with recurrent state | yes (contract `sage-v1`) |
| The true LCFs as a policy input (arm A1, patches) | yes, equal to the labels at every step; the full state beyond the LCFs (A0) not yet |
| Offline and online training, evaluation with bootstrap intervals over paths | yes |
| net1 as a distribution over bins, trained with the log score | not yet (it predicts three percentiles) |
| Varied controllers for net1's data | not yet (the kernel's controllers and BBRv3 are available) |
| One decision per RTT, exploration held over an RTT | not yet |
| Log-throughput reward; several learned flows on one bottleneck | not yet (one agent per kernel) |
| Raw-measurement net2 (arm A4) | not yet |
| net1's beliefs logged during closed-loop runs | not yet (they can be recomputed from the stored measurements) |
| Labels recomputed from stored shards | not yet (no reader for stored link tables) |
| Inference latency | not modelled: free in virtual time; to be measured outside and added as a delay |
