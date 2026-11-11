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
how many. A learned method meets a standing objection: a person still
chooses the vocabulary of candidate concepts. Claim C4 exists to answer
it.

Everything runs in the emulator this repository holds: the Linux TCP
stack in User-Mode Linux in virtual time, with the true path state
recorded at every instant ([architecture.md](architecture.md)).

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
ground them. [lcf-concepts.md](lcf-concepts.md) reviews the candidate LCFs
the literature defines. Choosing from such a review is the alternative
this project tests against; the review also supplies the hand-picked
baseline and helps name a fitted formula.

## Terms

| Term | Meaning |
|---|---|
| teacher | The controller whose behaviour is distilled. An unconstrained teacher reads raw measurements and does not pass through net1 (the black-box teacher of step 3). The two Phase 0 teachers are not unconstrained: each acts through an estimator of the seven LCFs. |
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
| LCF | Latent congestion factor, expliCCit's term: a physical property of the path at one instant, with an exact value in emulation. |
| the seven LCFs | expliCCit's hand-picked set: mean queueing delay, queue (packets), capacity, free buffer (packets), busy fraction (bytes sent over bytes the link offered), one-way delay, and the flow's receive rate, each over the trailing 100 ms. Definitions: [architecture.md](architecture.md#ground-truth-labels). |
| net1 | The estimator. It reads the per-step state message (70 of its 78 values, `EXPLICCIT_OBS_COLS`) and outputs one belief per LCF. It is trained against ground truth and never with net2. |
| belief | net1's statement about one LCF: a single value, an interval, or a distribution over fixed bins ([below](#net1s-output)). |
| net2 | The decision maker: it sets the congestion window from net1's output alone, never from raw measurements or the truth. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| u | An action in window-target form: the next window is 2^u times the believed BDP. |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps; a controller's score on a link is the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |

## net1's output

A sender learns the link's capacity only while it keeps the link busy.
Below capacity with an empty queue, the delivery rate equals the sending
rate, so the measurements bound capacity from below and say nothing more.
The minimum RTT has the opposite limit: it cannot be measured while a
queue persists. No operating point reveals every quantity, so a single
value per concept can mislead. net1 can therefore state a distribution
over fixed bins per concept, trained with the log score (cross-entropy
against the bin that holds the true value). The log score's expected
value is lowest only for the true conditional distribution, so
overstating or understating confidence always costs (Gneiting and
Raftery, 2007). Accuracy is the R^2 of the belief's mean; honesty is the
coverage of central intervals on held-out links.

net2 is trained after net1, on a fixed net1's outputs: the sequential
regime of concept bottleneck models (Koh et al., 2020). Soft outputs can
carry more than the concepts, which is the leakage risk below.

Which kind of output the retrained net1 uses in step 5 is not chosen yet.

## Evaluation protocol

These rules hold for every step. The concrete link set, conditions and
episode lengths are fixed in the experiment file of the step that first
needs them, before its run.

- **Split by link.** Data and evaluation are split by link (capacity
  trace, delay, buffer; `expliccit_env.train.data.link_key`), never by row
  or episode, so no test link is seen in training under any condition. A
  row split puts one path's steps in every split and overstates accuracy.
- **Closed loop.** A claim about a controller rests on rollouts in the
  emulator, not on fit to logged states. Fit to logged states (Phase 0)
  is a feasibility signal.
- **Score.** Log utility, with differences paired by link and 95%
  bootstrap intervals over links.
- **Tune first.** No claim about a learned model before it is tuned:
  read its training diagnostics, search the settings that matter on
  validation links, and confirm the winners over several seeds.
- **Decide before running.** A choice rule (a knee, a threshold, a
  winner) is written down before the run it decides.
- **Test once.** The test links run once per candidate, after every
  choice is fixed on validation, and the entry says what was searched.
- **Failed episodes count for nothing.** A failed or invalid episode
  gives no metrics and no labels.

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

1. **Setup.** The documents state the idea, the claims and the plan; the
   Phase 0 inputs are releases of this repository.
2. **Phase 0 as a release.** Port the prototype (below) to an experiment
   file and reproduce its findings. Add a value head (the teacher's
   Q-values on a fixed grid of actions) as a third head, and report the
   action-only, next-state and value variants side by side.
3. **A black-box teacher.** Train a teacher on raw measurements that does
   not pass through net1, with net1's observation window and net2's action
   space, and tune it as the rules require before trusting it. This
   teacher is the real test: the Phase 0 teachers already act through
   seven LCFs. Then distill students with DAgger (roll out
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
| 1 Setup | Done. The Phase 0 teachers are release [`exp-phase0-teachers-1`](https://github.com/Radonux/compact/releases/tag/exp-phase0-teachers-1). The step data is copied by `experiments/phase0-steps.json`; its release is not published yet |
| 2 Phase 0 as a release | UNRUN. A local prototype ran (below). The value head is not built |
| 3 Black-box teacher, DAgger students | UNRUN. No black-box teacher has been trained or tuned in this project |
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

- Data: 300,000 steps from 8 of the 20 shards of the logged step data
  (1,440 links), split by link 70/15/15 with
  `expliccit_env.train.data.link_key`. The data keeps every 6th step, so
  the next state is 60 ms later.
- Teachers, applied to the logged states: the rule teacher
  (`teachers/rule-bdp.onnx`, a window of 2^0.25 times its estimator's
  believed BDP) and the TD3+BC teacher (`teachers/td3bc.onnx`). The graphs
  have a fixed batch dimension of 1; setting dimension 0 of their inputs
  and outputs to a symbolic `N` with the `onnx` package gave identical
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

The main limitation: both teachers already act through an estimator of
the seven LCFs, so this tests the recovery of a known bottleneck, not discovery.
The knee here is in held-out action R^2 on logged states, while C1 is
about closed-loop performance.

### Open items

- **The scale of retained performance** ([above](#retained-performance)).
- **The threshold for C3**, and the choice rule for the knee in C1.
- **The links, conditions and episode lengths** of the closed-loop
  evaluation, to be fixed in step 3's experiment file.
- **The kind of output** the retrained net1 states (value, interval or
  distribution).
- **Live truth for new concepts.** The guest gives a policy the true
  values of the seven LCFs only, and labels are computed when an episode
  is processed ([architecture.md](architecture.md#known-limits)). The
  oracle net2 of step 5 and labels for fitted formulas need both
  extended.
