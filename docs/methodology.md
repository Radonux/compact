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

The data already show the limit. The tuned distribution net1 states
capacity to within 1.4 Mbit/s (90%) on steps where the link is kept busy
and to within 14.9 Mbit/s where it is not (`docs/training.md`, section 7).
And on the validation paths, the earlier single-value net1's capacity
estimate has R^2 0.78 on the teachers' episodes but 0.45 on a cloned
policy's and -0.01 on online-trained policies', which leave the link idle
more often.

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

Status (`docs/training.md`, section 7): the histogram head beats a
single-value and an interval net1 on held-out paths. On links held out
entirely, with three controllers held out of training, the net1 that net2
now builds on is an ensemble of three, trained on 1,230 sampled links and
21 controllers. Its 90% statements hold 88.8% to 97.1% of the time per
LCF, on busy and idle links alike, so it errs on the cautious side; on
held-out controllers they hold 80% to 97%, BBRv3's queueing delay being
the worst. Link breadth turned out to matter more than any setting: on
132 training links, net1 recognized links and was over-confident about
capacity (0.71 at 90%). The minimal set of quantities was tried and lost
to the seven LCFs (mean CRPS skill 0.74 against 0.81): buffer size and
cross-traffic share are constant over a link, so predicting them rewards
recognizing the link.

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

The reward is one scalar; evaluation covers every row. The first reward
was linear in utilization, minus penalties on queueing delay relative to
the base RTT and on the flow's drop fraction. A linear throughput term
gives competing flows no reason to share. If each flow maximizes the log
of its throughput minus a price for congestion, competition settles on
proportionally fair shares (Kelly, 1998); Copa and PCC Vivace build their
objectives on the same idea. The reward is therefore now log throughput
minus penalties on relative queueing delay and loss
(`net2.reward: "log_utility"`). A single-flow reward cannot
teach fairness, so training includes competing flows: stock controllers,
and copies of the policy, which the environment now runs on one
bottleneck (any number of agents per kernel); training against copies is
UNRUN.

Training is online, in closed loop: the tuning in `docs/training.md` ran
3,216 closed-loop episodes of 30 s in about 35 minutes on 4 cores, so
offline training is not forced on us as it was on Sage, which emulated in
real time. The results so far
(`docs/training.md`) pointed to three design changes, now implemented
(`docs/training.md`, section 2):

- Act on an absolute target. The old action multiplies the window every
  10 ms, so a bias of 0.001 per step moves the window by 7% a second.
  net2 now sets the next window as a multiple of the believed
  bandwidth-delay product (net1's mean capacity times twice its mean
  one-way delay), so errors do not accumulate, and net2 learns only the
  multiple.
- Explore over several steps. Independent noise of 0.005 every 10 ms
  moves the window by about 1% over 100 ms, so it barely explores at the
  window's own time scale. Each noise draw is now held for a set number
  of steps, which holds the perturbed target long enough to show.
- Use teacher logs only to start. Cloning and offline RL from them
  reached +0.55 to +0.64 utility on held-out paths, against +0.82 for
  BBRv1. Online training now draws its rollouts under every condition
  of its campaign, cross traffic and random loss included.

### What net2 sees

net2 takes the full distributions (7 LCFs times about 32 bins). Whether
it needs them is an experiment, not an assumption: train net2 once on the
full distributions and once on the human reading of section 4 (a few
percentiles per LCF, in physical units). If they perform the same, net2
takes the reading, and what an engineer reads is exactly what net2 acts
on. If the full distributions do better, the difference measures what the
reading leaves out, and we report what that is.

So far only the means have been tried, on validation links: net2 on the
means of the distributions (arm A2) did as well as net2 on the
distributions (section 6). The human reading as net2's input is UNRUN.

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
| A2 | net1's single values: the means of the same net1's distributions (`net2.view: "means"`) | |
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
path distribution. The results so far use fewer seeds than this: three
for the learned net2, one for each 60-iteration run (section 6).

## 6. Where things stand

**Reproduction**

| expliCCit's result | Here |
|---|---|
| net1 estimates every LCF with R^2 above 0.9 | 5 of 7 (0.95 to 0.99); busy fraction 0.45 and free buffer 0.09 |
| The split costs about 5% utilization and 9% queueing delay against Sage | UNRUN: Sage's released policy is exported under `sage-v1` and matches TensorFlow at every step (`docs/validation.md`); no comparison is published yet |
| On a 24/48 Mbit/s link (one-way delay 40 ms, 1000-packet buffer), under-utilization is traced to capacity underestimated by 25%, and replacing that belief with the truth restores utilization | Not reproduced. The released policy runs at 0.49 utilization, and its capacity belief is about right; replacing it with the truth gives 0.43. The wrong beliefs are busy fraction (1.0 against 0.53 true), queue (18 packets against 0), free buffer and queueing delay; replacing busy fraction alone gives 0.72 |
| Sage's state, as computed here, equals `sage.cc`'s | yes: every compared column of every step, on a Cubic and a BBRv1-with-loss episode (`tests/test_sage.py`, run locally with a Sage checkout; CI skips it) |

The released net1 was trained on expliCCit's own data, so the case study
tests transfer from their emulation to this one as much as the method.
Why that net1 misreads busy fraction and queue here is the main open item
of the reproduction.

**Results so far** (`docs/training.md`, section 7). The first run of the
design above, on links neither network has seen, in log utility:

- net1 is honest on new links under the training conditions: its 90%
  statements hold 88.8% to 97.1% of the time per LCF. Under competition
  it has not seen it is not: with a Cubic flow present from the start,
  its 90% capacity statements hold 26% of the time, and its median
  belief is 0.59 of the true capacity.
- net1 v4, the same design trained on a broad set of conditions
  (`docs/training.md`, "net1 on a broad set of conditions"; release
  `exp-net1v4-grid-10`), is honest where net1 v3 was not: on new links
  under those conditions its 90% statements hold 90% to 97% of the time
  per LCF, capacity's 90% statements hold 90% of the time with a BBRv3 or
  Westwood competitor it never saw, and 87% and 88% under loss and
  delays beyond its training range. On buffers deeper than any in training its
  free-buffer statements hold 42% of the time. Lower learning rates did
  not improve it (`exp-net1v4-lr-12`), and two more seeds confirm the
  choice and these coverages to within 0.02 (`exp-net1v4-seeds-13`). On
  net1 v3's validation links the
  fixed rule on it leads BBRv3 by 0.150 on average over five campaigns
  (release `exp-rule-net1v4-val-11`). Once on the 27 test links
  (`exp-rule-net1v4-test-14`), it leads BBRv3 by 0.123 and BBRv1, the best
  kernel controller there, by 0.053 on average, beats the rule on net1 v3
  in every family (by 0.04 to 0.43, every interval above zero), and stays
  within 0.05 of the rule on the true BDP except with a Cubic flow from
  the start (0.09). With competitors neither net1 trained on (BBRv3 or
  Westwood from 0 s) it still leads net1 v3 by 0.10. There Scalable and
  BIC beat every fixed rule, the true-BDP one included: the remaining gap
  is the policy's, not net1's.
- net2 on net1 v4 (`docs/training.md`, "net2 on net1 v4"). Online TD3+BC
  from the fixed rule lost to it on validation (`exp-net2v4-val-16`). A
  share rule, the fixed rule's window raised when net1 believes the link
  busy and this flow's share small, wins once on the test links
  (`exp-share-test-18`): at k = 1 it is never worse than the fixed rule
  outside copies, leads it by 0.26 with a Cubic flow from the start and
  by 0.13 with BBRv3 or Westwood, and leads BBRv3 by 0.208 on average.
  Its own copies lose 0.79 against the fixed rule's copies: it cannot
  tell a copy of itself from a buffer-filling competitor. A net2 with
  memory that cooperates with its copies and switches to the share rule
  when net1 believes the buffer nearly full kept neither gain on
  validation (`exp-mode-val-19`), so the fixed rule remains ahead with
  copies weighted as one family of six. On the true LCFs (arm A1,
  `exp-share-oracle-val-20`) the share rule gains 0.03 to 0.06 without
  copies, but every rule on the truth fails with copies (the fixed one
  by 1.06 against itself on net1 v4), since the link's capacity is not
  the capacity a shared flow can take: the LCF set lacks that quantity.
- net2 setting the window to a fixed multiple of net1's believed BDP
  beats every kernel controller on average: by 0.079 (95% interval 0.032
  to 0.123) over BBRv3, the best of them. The average is over three
  equally weighted conditions. It beats all 17 controllers on clean links
  and under 1% random loss, but under cross traffic it trails BBRv3 (by
  0.112, 0.026 to 0.202), Illinois, Scalable and BIC, since a fixed
  window takes a small share against a growing Cubic flow. Scored on log
  throughput alone, BBRv1 would beat it by 0.067. All of this is with
  20 s episodes: at 60 s, where the Cubic flow competes for 52 s instead
  of 12 s, it ties BBRv3 (-0.008, -0.114 to +0.087) and BBRv1 on
  average, and the same rule on the true BDP would gain 0.373 under
  cross traffic, so the loss is net1's.
- The same rule on the true BDP (arm A1) does 0.041 to 0.065 better on
  two held-out sets (27 and 158 links): the cost of estimation for that
  rule (H2). On a classical estimate (largest delivery rate times min
  RTT, as BBR builds it) it does 0.155 to 0.228 worse: net1 closes about
  85% of the gap between that estimate and the truth.
- A net2 trained online on net1's distributions (arm A3, three seeds)
  beats BBRv3 by 0.12, but the fixed multiple by only 0.04, 0.01 and 0
  on the two held-out sets and the validation links. It learned to
  compete harder with a buffer-filling Cubic flow. The same training run
  for 60 iterations instead of 20 (one seed) did not beat the fixed
  multiple on validation either; the fixed multiple remains the net2 to
  beat.
- On the validation links, a net2 on the means of the same distributions
  (arm A2) did at least as well as one on the distributions (A3) over 60
  iterations (A3 minus A2 between -0.030 and +0.001; one seed each, not
  run on test). So far net2 gets no measurable value from net1's stated
  uncertainty. A likely reason, not tested: this window target needs
  mostly the means, and these runs keep the link busy, where capacity is
  identifiable. Honesty (H3) still serves the human reading.
- Learning from other controllers' logs failed (cloning and IQL, -0.97
  to -1.61 against -0.48 for BBRv3 on validation links), as did the
  earlier design (multiplicative action, single-value net1): its tuned
  TD3+BC policy scores 0.6 below the fixed multiple.

Arm A2 has run on the validation links only (above). Arms A4 (raw
measurements), A0, A1 for a learned net2, and the human reading as
net2's input are UNRUN.

All of these results were measured on one family of links and three
conditions
(`docs/training.md`, section 7, "What these results cover"): constant or
stepping capacity of 8 to 100 Mbit/s, one-way delay of 5 to 50 ms,
buffers of 0.25 to 4 BDP, and 20 s episodes with one observed flow, each
link clean, with UDP and a Cubic flow, or with 1% random loss. Test links
are new combinations of these, but the conditions are not held out. The
kernel controllers ran on the same links and conditions, so their
ranking holds for this mix only. A probe outside it (`docs/training.md`,
"Longer episodes" and "New conditions") found net1 honest under other
loss rates and on deeper, longer paths, and badly over-confident under
competition it had not seen, which also costs the fixed multiple its
lead over BBRv3 and BBRv1 once competition lasts 52 s. net1 v4's data
widens the conditions for net1; the control results on it are pending.

**What the environment supports**

| Need | Status |
|---|---|
| Exact, time-indexed ground truth of the bottleneck | yes: every arrival, drop and departure, both directions, by flow and drop cause |
| Paths: capacity traces, delay, buffer, random loss, cross traffic | yes; no AQM or ECN, delay jitter, bursty loss or multiple bottlenecks |
| Policies of any network design, with recurrent state | yes (contract `sage-v1`) |
| The true LCFs as a policy input (arm A1, patches) | yes, equal to the labels at every step; the full state beyond the LCFs (A0) not yet |
| Offline and online training, evaluation with bootstrap intervals over paths | yes |
| net1 as a distribution over bins, trained with the log score | yes (`net1.bins`), tuned and checked over three seeds |
| Varied controllers for net1's data | yes: 17 kernel controllers and four delay-threshold policies on 1,230 sampled links (`examples/net1_varied.py`) |
| An absolute window target, exploration held over several steps | yes (`net2.action: "bdp"`, `noise_hold_steps`); decisions are still made every step |
| Log-throughput reward | yes (`net2.reward: "log_utility"`) |
| Several learned flows on one bottleneck | yes: any TCP cross flow can be an agent with its own policy (up to 64 cross flows), each with its own steps (`cross_steps`) and labels; decisions take no virtual time |
| Raw-measurement net2 (arm A4) | not yet |
| net1's beliefs logged during closed-loop runs | not yet (they can be recomputed from the stored measurements) |
| Labels recomputed from stored shards | not yet (no reader for stored link tables) |
| Inference latency | not modelled: free in virtual time; to be measured outside and added as a delay |
