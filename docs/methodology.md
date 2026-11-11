# Methodology

compact asks one question: can the concepts of a concept-bottleneck
congestion controller be chosen from what a trained controller actually
uses, instead of being picked from a literature review? This page states
the idea, the claims, the terms, how results are measured, the plan and
the status of each step.

The controller design is expliCCit's. net1, the estimator, maps end-host
measurements to a small set of latent congestion factors (LCFs): physical
path state with exact ground truth in emulation. net2, the decision maker,
sets the window from net1's estimates alone. Reviewers criticized
expliCCit because its LCF set is ad hoc: nothing says which LCFs to use or
how many. compact is a fork of
[Radonux/palaestra](https://github.com/Radonux/palaestra), which rebuilt
expliCCit in User-Mode Linux and tested the split itself. palaestra's
claims and their status at the fork are in [palaestra.md](palaestra.md),
its results in [results.md](results.md). compact keeps palaestra's
environment, trainer and evaluation protocol, and uses the name palaestra
for that environment.

## The idea

Discovery is data driven. Deployment stays by design.

1. **Discover.** Train a controller (the student) whose decision passes
   through a narrow layer of k latents. The width at which performance
   stops improving estimates how many concepts the task needs.
2. **Ground.** Fit the latents to short closed-form formulas over named
   variables (emulator ground truth plus basic sender metrics) under a
   harsh complexity penalty. Keep the formulas that are stable across
   seeds. A person names each one.
3. **Retrain.** Train a normal net1 and net2 with the fitted formulas as
   the LCF labels, through the existing pipeline unchanged. Only this
   controller ships, and the student of step 1 is thrown away. Every
   decision of the shipped controller passes through exact, checkable
   quantities, so it keeps expliCCit's faithfulness property.

The main metric is retained performance: the performance of net1 + net2 on
the fitted concepts over the performance of the unconstrained teacher,
measured in closed-loop rollouts, and compared with other ways of choosing
concepts at the same budget ([below](#retained-performance)).

## Claims

| | Claim | How it can fail |
|---|---|---|
| C1 | Count: a student's closed-loop performance stops improving at a small width k, the same across seeds | no knee: the controller does not compress into a few factors, which bears on the two-net design itself |
| C2 | Grounding: at that width each latent is a short closed-form formula over named variables (held-out R^2 of 0.9 or more at the knee of size against fit), and the same formulas recur in at least 3 of 5 seeds | the latents fit no short formula, or differ between seeds |
| C3 | Retention: net1 + net2 trained on the fitted concepts retain more of the teacher's closed-loop performance than the same number of concepts chosen any other way (the hand-picked LCFs, greedy subset search, SHAP-ranked inputs, a random subset) | a baseline ties or wins at the same budget |
| C4 | The vocabulary is not the choice: with one base variable or operator family removed from the library, the selected concepts change only when the removed item was one of them | the selection follows whatever the library holds, which is the objection that a person still chooses the candidates |
| C5 | Teacher independence: teachers trained differently yield the same concepts | the concepts describe one teacher, not the task |

No threshold for "retains" in C3 is fixed yet. The choice rule is written
down before the run it decides, as for every other choice here.

## What is not claimed

- A bottleneck does not make a network explainable. Narrow layers tend to
  pack several features into each unit (superposition), and unsupervised
  latents are defined only up to mixing (Locatello et al., ICML 2019). The
  claim is narrower: the bottleneck counts the concepts the task needs, and
  grounding against emulator ground truth says which ones they are.
- A closed-form concept is exact, which does not make it readable. Expect
  a mix: some concepts match familiar quantities, others are short but
  unfamiliar formulas. Unfamiliar ones are ranked by how much performance
  drops without them.
- The student is never shipped and nothing is claimed about it. Claims
  about the shipped controller rest on the retrained net1 and net2.

## Related work

Encoding and then interpreting is a known pattern in physics ML (SciNet,
Iten et al., PRL 2020; Miles et al., PRB 2021; Cranmer et al., NeurIPS
2020) and in vision concept-bottleneck work (Discover-then-Name CBM, ECCV
2024). The new parts are the networking instance with emulator ground
truth, the use of the grounding to choose concepts for a by-design
controller, and the retained-performance metric.

The closest networking work: Agua (SIGCOMM 2025) maps a controller's
embeddings to text concepts written by an LLM, after the fact. Niedermayer
et al. (IFIP Networking 2026) measure the completeness of a given concept
set. Mutant (NSDI 2025) and UNUM (NSDI 2026) learn latents but do not
ground them. [lcf-concepts.md](lcf-concepts.md) is the literature review of
candidate LCFs made before the fork: it is the alternative this project
replaces, the source of the hand-picked baseline, and a reference for
naming a fitted formula.

## Terms

| Term | Meaning |
|---|---|
| teacher | The controller whose behaviour is distilled. An unconstrained teacher reads raw measurements and does not pass through net1 (the black-box teacher of step 3). The two Phase 0 teachers act through net1's seven LCF beliefs. |
| student | A network trained to imitate the teacher, whose decision passes through the latents. It is a measuring device and is never deployed. |
| latents, z, k | The student's narrow layer z, of k dimensions. Everything after it reads z alone. |
| head | An output computed from z. The action head predicts the teacher's action; the next-state head predicts the next step's log sRTT and log delivery rate; the value head predicts the teacher's Q-values on a fixed grid of actions. |
| knee | The smallest k beyond which performance stops improving. |
| grounding | Fitting each latent to a closed-form formula over the library, on held-out links. |
| library | The candidate terms for grounding: base variables (hidden state from the `label_*` columns, and sender metrics) crossed with operators (windowed mean, min and max, first difference, pairwise ratio and product, log). |
| concept | A grounded formula that is stable across seeds and has been named. The concepts become the LCF labels of the retrained net1. |
| retained performance | Performance of net1 + net2 on the fitted concepts over the teacher's, in closed loop ([below](#retained-performance)). |
| completeness curve | Retained performance against the number of concepts m. |
| oracle net2 | net2 trained on and fed the true concept values. A diagnostic only: teacher minus oracle is what the concept set lacks, and oracle minus net1 + net2 is what net1's estimation costs. |
| action collapse | z encoding the teacher's action itself and not the state behind it. Checked in every run by regressing the teacher's action on z. |
| linear CKA | Centered kernel alignment with a linear kernel: a similarity of two sets of latents that ignores rotation and scale (Kornblith et al., ICML 2019). |
| DAgger | Imitation in which the student acts in every rollout and each of its decisions is labelled with the teacher's action (Ross, Gordon and Bagnell, 2011). |

## Inherited terms

palaestra's terms, unchanged. Its privileged teacher (one that sees the
other flows on the link) is described in [rounds.md](rounds.md).

| Term | Meaning |
|---|---|
| LCF | Latent congestion factor, expliCCit's term: a physical property of the path at one instant, with an exact value in emulation. Candidates from the literature: [lcf-concepts.md](lcf-concepts.md). |
| the seven LCFs | expliCCit's set: mean queueing delay, queue (packets), capacity, free buffer (packets), busy fraction (bytes sent over bytes the link offered), one-way delay, and the flow's receive rate, each over the trailing 100 ms. Definitions: [architecture.md](architecture.md#ground-truth-labels). |
| net1 | The estimator. It reads Sage's per-step state message (70 of its 78 values) and outputs one belief per LCF. It is trained against ground truth and never with net2. |
| belief | net1's probability distribution for one LCF over 64 fixed bins (equal bins for a fraction; for a rate or delay a bin for zero, then log-spaced bins), averaged over an ensemble of three networks. |
| net1 v3, v4, v5 | v3: trained on 1,230 sampled links under three conditions (`examples/net1_varied.py`). v4: the same design on a broad dataset of 3,600 links under 64 conditions (`examples/net1_broad.py`). v5: v4's settings retrained on that data regenerated under the frozen guest ([`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79); weights in `model-net1v5`). Current results use v5. |
| net2 | The decision maker: it sets the congestion window from net1's output alone, never from raw measurements or the truth. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| u | net2's action: the next window is 2^u times the believed BDP. The policy graph reads the current window and lands on the target in one step. |
| fixed rule | u held constant. On net1 v5, u = 0.25 (a window of 1.19 believed BDPs), chosen on validation (`exp-rule-net1v5-val-82`). |
| share rule | u = 0.25 + k busy max(0, 1 - receive rate / capacity), on net1's means, k = 0.5 chosen on validation (`exp-share-net1v5-val-85`; `train/share.py`): it pushes when the link is believed busy and this flow's share small. |
| oracle | The same policy fed the true LCFs, computed live in the guest, in place of net1's beliefs ("the rule on the true BDP"). |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps; a controller's score on a link is the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |
| copies | 2 or 4 flows of one controller starting together on one link. |
| families | The sets of conditions controllers are scored under (table below). The five-family mean averages the first five. |
| the gate | The rule a learned net2 must pass on validation before its one test run (below). |
| test once | Each candidate runs on the test links once, after every choice was fixed on validation. |

## Why net1 reports beliefs

A sender learns the link's capacity only while it keeps the link busy.
Below capacity with an empty queue, the delivery rate equals the sending
rate, so the measurements bound capacity from below and say nothing more.
The minimum RTT has the opposite limit: it cannot be measured while a
queue persists. No operating point reveals every LCF, so a single value
per LCF misleads ("24 Mbit/s" where the data support "at least 18"). net1
therefore reports a distribution per LCF, and net2 can learn when
reducing uncertainty is worth the cost of probing.

net1 is trained with the log score (cross-entropy against the bin that
holds the true value). Its expected value is the entropy of the true
conditional distribution plus net1's KL divergence from it, so it is
lowest only for the true distribution: overstating or understating
confidence always costs (Gneiting and Raftery, 2007). The log score
teaches the uncertainty the measurements leave, not uncertainty about
inputs never seen; the ensemble widens the belief where its members
disagree. Accuracy is the R^2 of the belief's mean; honesty is the
coverage of central intervals (90% unless stated) on held-out links,
reported separately for busy steps (busy fraction at least 0.95) and idle
ones (below 0.8), because average calibration can hide over-confidence
exactly where identifiability fails.

net2 is trained after net1, on a fixed net1's outputs, so it learns to act
on net1's beliefs, errors included: the sequential regime of concept
bottleneck models (Koh et al., 2020). This is sound only while net1 is
honest: then its belief is a sufficient statistic for the decision, and
what net2 learns about net1's errors is how to act under stated
uncertainty.

## Evaluation protocol

Links. Data and evaluation are split by link (capacity trace, delay,
buffer), never by row or episode, so no test link was seen under any
condition. The evaluation links are the 21 validation and 27 test links of
`examples/campaign-net1-varied.json` (split seed 1, 15% each): capacity
constant or stepping 2x up or down every 3, 7 or 15 s from a base of 8 to
100 Mbit/s, one-way delay 5 to 50 ms, buffers of 0.25 to 4 BDP.

Families (`experiments/heldout.py`, `experiments/copies.py`):

| Family | Conditions | Length |
|---|---|---|
| training-20 | net1 v3's three: clean; 1% random loss; 3 Mbit/s UDP from 5 to 15 s plus a Cubic flow from 8 s | 20 s |
| training-60 | the same | 60 s |
| new-20 | conditions net1 v3 never saw: a Cubic flow from 0 s, a BBRv1 flow from 4 s, Cubic flows from 2 s and 10 s (the three competition conditions); 0.1% and 3% random loss; 1% loss on the ACK path | 20 s |
| deep-long-20 | the training conditions with 2.5 times the one-way delay (at most 80 ms) and buffers of 4 to 16 BDP | 20 s |
| cubic-first-60 | a Cubic flow from 0 s | 60 s |
| held-competitors-60 | a BBRv3 or a Westwood flow from 0 s, competitors held out of net1 v4 and v5's data | 60 s |
| copies-30s | 2 or 4 copies of the controller | 30 s |
| dynamic-120 | competitors arriving and leaving mid-episode: Cubic from 30 to 90 s; BBRv1 from 0 to 60 s; Reno from 20 s; Cubic from 0 s and 60 s; Vegas from 15 s; BBRv1 from 40 s with Cubic from 40 to 100 s. Reported beside the gate, not part of it | 120 s |

Scoring. Log utility, differences paired by link, 95% bootstrap intervals
over links. Most tables give a controller's score minus BBRv3's (or minus
the fixed rule's) on the same links.

Choosing. Every learned model is tuned on validation links before any
comparison: training diagnostics first, then a grid over the settings
that matter, then the winners over several seeds (three training seeds
for a learned net2; three campaign seeds, 11, 21 and 31, for a rule's
constant). A choice rule is written down before the run it decides.

The gate (`experiments/net2_ppo_gate_report.py`). Each candidate is
trained on three seeds and scored against the fixed rule on every
validation family. The candidate with the best five-family mean is kept.
It passes only if it beats the fixed rule in cubic-first-60 and in
new-20's three competition conditions, each with its 95% interval above
zero, is at or above the rule on the five-family mean, and is not worse
as copies beyond its interval (the interval's upper end at or above
zero). Otherwise net2 stays the fixed rule and the share rule.

Testing. The test links run once, after every choice is fixed:
[`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)
(campaign seeds 13 and 31). A learned net2 that passes a later gate runs
on them once, on its own (round 7e:
[`exp-net2-test-r7e-k0.5-134`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r7e-k0.5-134);
round 18 on net1 v6:
[`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)).


### Retained performance

Retained performance is the closed-loop performance of net1 + net2 on the
fitted concepts over the unconstrained teacher's, on the same links,
conditions and seeds. It is reported with the oracle net2 beside it, so
the gap splits into what the concept set lacks (teacher minus oracle) and
what net1's estimation costs (oracle minus net1 + net2), and with every
baseline at the same number of concepts.

The scale is an open choice, to be fixed before step 5 runs. Log utility
is at most about zero (the log of a share of capacity, minus penalties),
so a plain ratio of two log utilities is not a fraction retained: a larger
ratio would mean a worse controller. One candidate is the fraction of the
teacher's lead over a fixed reference controller, (score - reference) /
(teacher - reference), reported with the paired difference in log utility
and its interval. Undecided, and UNRUN.

## Plan

The plan is a starting point. A step changes order, is added or is dropped
when the evidence points elsewhere, and the entry for it says why.

1. **Setup.** This page, `AGENTS.md` and the README rewritten for compact,
   keeping palaestra's engineering rules.
2. **Phase 0 as a release.** Port the prototype (below) to an experiment
   file and reproduce its findings. Add a value head (the teacher's
   Q-values on a fixed grid of actions) as a third head, and report the
   action-only, next-state and value variants side by side.
3. **A black-box teacher.** Train a teacher on raw measurements that does
   not pass through net1, with net1's observation window and net2's action
   space, and tune it as the rules require before trusting it. This
   teacher is the real test: the Phase 0 teachers already act through
   seven LCFs. Then distill students with DAgger in palaestra (roll out
   the student, label the visited states with the teacher, retrain). Sweep
   k = 1 to 8 and the weight beta of a variational information bottleneck,
   with 5 seeds.
4. **Grounding.** Build the library mechanically: base variables (hidden
   state and sender metrics) crossed with operators (windowed mean, min
   and max over 2 to 3 horizons, first difference, pairwise ratio and
   product, log).
   - Fit all k latents jointly with a multi-output group lasso, then take
     a sparse basis of the fitted subspace.
   - Run PySR, with a parsimony penalty, only on the residual the linear
     fit leaves.
   - For each latent, report formula size against held-out R^2 and pick
     the knee. The target is R^2 of 0.9 or more.
   - Keep the concepts that recur in at least 3 of 5 seeds, matching
     formulas by a correlation above 0.9 on shared held-out traces.
   - Vocabulary sensitivity (C4): refit with each base variable or
     operator family removed, and report whether the selected concepts
     change only when a removed item was one of them.
5. **Retrain and compare.** Compute labels from the frozen formulas, train
   net1 and net2 as usual, and measure retained performance in closed
   loop.
   - Train the oracle net2 on the true concept values, as a diagnostic.
   - Plot the completeness curve for m = 1, 2, 3, 5 and the full set.
   - Baselines at the same budget: the current hand-picked LCFs, a greedy
     forward subset search over the base variables, the teacher's inputs
     ranked by SHAP, and a random subset.

## Risks

| Risk | What to do |
|---|---|
| No knee | Report it and work out why: it bears on the two-net design itself |
| The latents fit no short formula | Report it, and look at what the latents do encode before choosing what to try next |
| Action collapse | Check z against the teacher's action directly, in every run |
| Teacher dependence | Compare the concepts across teachers (C5) |
| Leakage | Compare net2 on net1's soft outputs with net2 on the exact concept values |

## Definition of done, for each step

- A release with the code, the configuration, the seeds and the raw
  outputs.
- A short entry here or in a results page: what ran, what was searched,
  the held-out numbers, and what remains unrun.
- No performance claim without the tuning the rules in `AGENTS.md`
  require. Anything not run is labelled UNRUN.

## Status

| Step | State |
|---|---|
| 1 Setup | Done: the documents are rewritten for compact |
| 2 Phase 0 as a release | UNRUN. A local prototype ran (below). The value head is not built |
| 3 Black-box teacher, DAgger students | UNRUN. palaestra trained a black box on raw measurements with PPO (round 19b, [results.md](results.md#the-split-costs-little)); whether it is good enough to serve as the teacher is not assessed |
| 4 Grounding | UNRUN |
| 5 Retrain and compare | UNRUN |
| C1 to C5 | No evidence from a release yet |

### Phase 0: the local prototype

A prototype ran on 2 CPUs before this repository was set up. It is not a
release, it was not tuned and it has no closed loop, so its numbers are
feasibility signals only and support no claim. They are recorded here as
reported, and nothing below was rerun in this repository. Its scripts
(`build.py`, `student.py`, `ground.py`) are not in the repository; step 2
replaces them.

Setup:

- Data: 300,000 steps from 8 of the 20 shards of release
  `exp-data-net1-broad-30s-73` (1,440 links), split by link 70/15/15 with
  `expliccit_env.train.data.link_key`. The data keeps every 6th step, so
  the next state is 60 ms later.
- Teachers, applied to the logged states: `rule-u0.25.onnx` from release
  `model-net1v5` (a window of 2^0.25 times net1's believed BDP) and
  `learned-td3bc-n40-s0.onnx` from `model-net1v3`. The graphs have a
  fixed batch dimension of 1; setting dimension 0 of their inputs and
  outputs to a symbolic `N` with the `onnx` package gave identical
  outputs on batches. The target is the action plus log2 of the window,
  which is log2 of the window the teacher sets.
- Student: net1's observation columns (`EXPLICCIT_OBS_COLS`) and log2 of
  the window, an MLP of two 256-unit layers, then z. Heads: the action,
  and the next state (log sRTT, observation index 2, and log delivery
  rate, index 7) with weight 0.5. k in {1, 2, 3, 4, 6, 32}, 3 seeds, 20
  epochs of behavioural cloning.
- Library: 12 hidden-state variables from the `label_*` columns and 7
  sender observables, mostly as logs, so a linear fit is a power-law
  product. Fitted on the validation links and scored on the test links.

Reported findings:

1. Positive control. With the action head only, one latent matched the
   unconstrained student for the rule teacher (held-out action R^2 0.969
   against 0.968). Greedy selection over the hidden state picked log
   capacity and then log base RTT (R^2 0.86), which together are log BDP:
   the rule's own concept, recovered without being given.
2. With the next-state head, both teachers level off at k = 3. Without
   it, the rule levels off at k = 1 and TD3+BC at k = 2.
3. The next-state head shapes z. It lowers the action R^2 at k = 1 from
   0.969 to 0.893 and makes the two teachers' latents nearly one subspace
   (linear CKA 0.96). Without it, z is close to the teacher's action
   (linear R^2 0.92 to 0.94), which is the action-collapse risk.
4. With the next-state head and k of 2 or more, the first three greedy
   picks are mostly log receive rate, log base RTT and log1p queueing
   delay (R^2 0.62 to 0.84).
5. The full library fits z linearly with a joint R^2 of 0.73 to 0.93; an
   MLP on the same library reaches 0.92 to 0.99. So the latents are mostly
   nonlinear functions of known variables.
6. Linear CKA between seeds is 0.92 to 0.99; a random 3-dimensional
   projection of the inputs scores 0.42.

The main limitation: both teachers already act through net1's seven LCF
beliefs, so this tests the recovery of a known bottleneck, not discovery.
The knee here is in held-out action R^2 on logged states, while C1 is
about closed-loop performance.

### Open items

- **Inherited releases are not in this repository.** compact has no
  releases yet (checked 2026-11-11); every release cited before the fork
  is in `Radonux/palaestra`. An experiment's `"inputs": {"NAME":
  "release:TAG"}` is fetched from the repository the workflow runs in
  (`tools/ci/experiment.py`), so step 2 cannot read
  `exp-data-net1-broad-30s-73`, `model-net1v5` or `model-net1v3` until
  inputs can name a repository or those releases are copied here.
  `scripts/release-pull.sh` defaults to `Radonux/palaestra` outside
  Actions and to the running repository inside it.
- **The scale of retained performance** ([above](#retained-performance)).
- **The threshold for C3**, and the choice rule for the knee in C1.
