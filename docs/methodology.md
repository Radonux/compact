# Methodology

compact asks whether the concepts of a concept-bottleneck congestion
controller can be derived from a trained controller, where today they are
picked by hand. This page gives the design, the ways it can fail, the
claims, the rules of evidence and the plan. What has run so far is in
[status.md](status.md).

## The problem

expliCCit splits a learned congestion controller in two. net1, the
estimator, maps the sender's measurements to a few latent congestion
factors (LCFs): physical state of the path, such as capacity or queueing
delay, with an exact value in emulation. net2, the decision maker, sets
the window from net1's estimates alone. Every decision therefore passes
through quantities that can be checked against ground truth.

Its seven LCFs were chosen by hand. Nothing says which LCFs a controller
needs, or how many, and reviewers said so. compact tries to derive the
set.

## The design

Read expliCCit's controller as an encoder and a decoder with a hand-made
middle: net1 encodes measurements into LCFs, net2 decodes LCFs into a
window. compact trains the same shape with the middle left free, then
names what the middle learned.

1. Discover. Train a controller whose decision passes through k latents,
   in the closed loop, to control well. Its closed-loop performance
   against k says how many latents it needs.
2. Ground. Fit the latents to short closed-form formulas over named
   variables: emulator ground truth and basic sender metrics. Keep the
   formulas that recur across seeds. A person names each one.
3. Retrain. Train a normal net1 and net2 with the fitted formulas as the
   LCF labels, through the existing pipeline. Only this controller ships.
   The discovery controller is thrown away.

Discovery is data driven. Deployment stays by design: the shipped
controller still passes every decision through exact quantities.

### The discovery controller

It is `PPOSpec.latents` in `src/expliccit_env/train/recurrent.py`.

| Part | In place of | What it is |
|---|---|---|
| Encoder | net1 | An MLP from one step's message (net1's 70 inputs, net1's scaler) to k latents. No memory, as net1 has none: the message already holds windowed statistics. |
| Decoder | net2 | A recurrent network (GRU) that reads the latents and its own previous decision. It decides once per minimum RTT, clamped to 10 to 50 ms, and holds the decision in between. |
| Output | | log2 of the window over a fixed 2^7 packets. |

Nothing passes around the latents. A window aimed at a measured
bandwidth-delay product would hand the decoder capacity and delay for
free, so the base is a constant.

Training has two stages, both in the emulator.

1. Bootstrap. The controller is cloned from a traditional congestion
   controller, the teacher: Cubic or BBRv3. The teacher drives the
   rollouts, and the controller learns the window that sends as the
   teacher did. There is one start per teacher and never a mixture,
   because the two set contradictory windows in the same state.
2. Reinforcement learning on the log utility, with no teacher term. This
   is where the controller can develop behaviour its teacher does not
   have, and the claims rest on this stage.

The bootstrap only makes reinforcement learning start from a controller
that works. Two details keep it honest. While it learns, the controller
reads its own previous decisions and never the teacher's, or it would fit
the teacher by repeating the last window and ignore the measurements. And
its fit to the teacher's rollouts is not evidence: a clone drifts once it
drives, so only its closed-loop score says how good the start is.

The same controller with 64 latents, too wide to constrain anything, is
the wide controller against which performance is measured.

## How it can fail

These follow from the design, before any data. Each has a check.

| Failure | Why it can happen | Check |
|---|---|---|
| Action collapse | A controller's output is one number. An encoder trained end to end can compute that number itself and send it through one latent. The width then counts nothing. expliCCit cannot collapse this way because net1 is trained on labels, never with net2. | Regress the decision on the latents in every run. Keep widths 1 and 2 in every curve. If one latent matches the wide controller, change the design before reading any count. |
| The concepts follow the teacher | Reinforcement learning may stay near the controller it started from. | Start from two teachers that work differently, Cubic (loss based) and BBRv3 (model based), and compare the concepts (C5). Measure how far each controller moved from its start. |
| A weak learner looks like a need for width | Reinforcement learning may train a narrow controller worse for reasons of optimization alone. | Tune before any claim. Compare with the wide controller trained the same way. |
| No knee | The controller may not compress into a few factors. | Report it. It bears on the two-net design itself. The comparison at seven concepts still stands. |
| The latents fit no short formula | Latents are defined only up to mixing, and narrow layers pack several features into each unit (Locatello et al., ICML 2019). | Report it, and look at what the latents do encode. |
| The vocabulary decides the answer | A person chooses the grounding library. | Refit with each base variable or operator family removed (C4). |
| Leakage | net1's soft outputs can carry more than the concepts. | Compare net2 on net1's soft outputs with net2 on exact concept values. |

## Claims

| | Claim | How it fails |
|---|---|---|
| C1 | Count. Closed-loop performance of the discovery controller stops improving at a small width k, the same across seeds, and at that width it matches the wide controller. | No width short of the wide one keeps the performance, or one latent already does (action collapse). |
| C2 | Grounding. At that width each latent is a short closed-form formula over named variables (held-out R^2 of 0.9 or more at the knee of size against fit), and the same formulas recur in at least 3 of 5 seeds. | The latents fit no short formula, or differ between seeds. |
| C3 | Retention. net1 + net2 trained on the fitted concepts retain more of the wide controller's closed-loop performance than the same number of concepts chosen another way: the hand-picked LCFs, greedy subset search, SHAP-ranked inputs, a random subset. | A baseline ties or wins at the same budget. |
| C4 | Vocabulary. With one base variable or operator family removed from the library, the selected concepts change only when the removed item was one of them. | The selection follows whatever the library holds. |
| C5 | Start. The discovery controller yields the same concepts from different teachers. | The concepts describe one teacher. |

Every threshold and choice rule a claim depends on is written down before
the run that decides it.

Two things are not claimed. A bottleneck does not make a network
explainable: it counts the concepts, and grounding against ground truth
names them. And a closed-form concept is exact, which does not make it
readable. Some will match familiar quantities and some will be short,
unfamiliar formulas, ranked by how much performance drops without them.

## Rules of evidence

They hold for every step. The links, conditions and episode lengths of a
run are fixed in its experiment file before it runs.

| Rule | Meaning |
|---|---|
| Closed loop | A claim about a controller rests on rollouts in the emulator. Fit to logged steps is never evidence about a controller. |
| Split by link | Data and evaluation are split by link (capacity trace, delay, buffer; `expliccit_env.train.data.link_key`), never by row or episode. A row split puts one path's steps in every split and overstates accuracy. |
| Score | Log utility, with differences paired by link and 95% bootstrap intervals over links. |
| Tune first | No claim about a learned model before it is tuned: read its training diagnostics, search the settings that matter on validation links, confirm the winners over several seeds. |
| Decide before running | A knee, a threshold or a winner is chosen by a rule written before the run. |
| Test once | The test links run once per candidate, after every choice is fixed on validation. The report says what was searched. |
| Failed episodes count for nothing | A failed or invalid episode gives no metrics and no labels. |

### Retained performance

It is the closed-loop performance of net1 + net2 on the fitted concepts
over the wide controller's, on the same links, conditions and
seeds. It is reported with every baseline at the same number of concepts,
and with the oracle net2 beside it. The oracle splits the gap in two:
wide controller minus oracle is what the concept set lacks, and oracle minus
net1 + net2 is what net1's estimation costs.

Log utility is at most about zero, so a plain ratio of two log utilities
is not a fraction retained: a larger ratio would mean a worse controller.
The scale must be fixed before step 4 runs. One candidate is the fraction
of the wide controller's lead over a fixed reference controller,
(score - reference) / (wide - reference), reported with the paired
difference and its interval.

## Plan

The plan is a starting point. A step moves, is added or is dropped when
the evidence says so.

1. Setup. Every input is a release of this repository.
2. The controller in the closed loop.
   - Bootstrap each width from each teacher, Cubic and BBRv3
     (`experiments/controller-teachers.json`). A teacher's window is one
     number, so this stage is the easiest case for action collapse and
     its width curve is not read as a count.
   - Continue each by reinforcement learning on the log utility. Tune it
     and confirm over seeds.
   - Read the width curve: closed-loop log utility on validation links
     against k, beside the wide controller, the teachers themselves and
     the fixed rule on the seven hand-picked LCFs. Width 7 is always
     present, so the comparison at expliCCit's budget exists whatever the
     curve looks like.
   - Check for action collapse at every width.
   - Compare the two teachers' controllers (C5), and report how far
     reinforcement learning moved each from its start.
3. Grounding, on the trained controllers' latents and the steps of their
   own rollouts ([below](#grounding)).
4. Retrain and compare. Compute labels from the frozen formulas, train
   net1 and net2 as usual, and measure retained performance.
   - Train the oracle net2 on the true concept values.
   - Plot retained performance against the number of concepts m, for
     m = 1, 2, 3, 5 and the full set.
   - Baselines at the same budget: the hand-picked LCFs, a greedy forward
     subset search over the base variables, the wide controller's inputs ranked
     by SHAP, and a random subset.

A step is done when it has a release with the code, the configuration,
the seeds and the raw outputs, and [status.md](status.md) says what ran,
what was searched, the held-out numbers and what remains UNRUN.

## Grounding

Grounding is meaningful only for latents that agree across seeds. The
library is built mechanically: base variables crossed with operators. The
base variables are the hidden state of the path and the sender's
metrics. The hidden state is every label the step data stores, in each
window it is stored in, and every composite of them: a fixed formula over
a few labels that the literature defines, such as the load factor or the
fair share ([architecture.md](architecture.md#ground-truth-labels)). The
library reads both lists from the code, so a label or composite added
there takes part in the fitting with no change here. The operators are a windowed mean, minimum and
maximum over 2 to 3 horizons, a first difference, pairwise ratios and
products, and a log.

1. Probe (`experiments/ground.py probe`). Linear fits between the latents
   and the base variables, in both directions, and between the latents of
   two seeds. It shows which variables the latents carry, how much of a
   latent the vocabulary can express at all, and whether seeds agree.
   Each is a lower bound.
2. Select (`experiments/ground.py select`). Fit all k latents jointly and
   linearly on the library, adding one feature at a time, the one that
   most raises the fit. The variables are on log scales, so a linear fit
   is a fit by products and ratios of powers, and the selected features
   are the candidate concepts.
3. Report the path twice: on the whole library, and on ground truth
   alone. The latents are computed from the sender's message, so on the
   whole library the sender's own readings can stand in for the path
   quantities they measure. Only a feature of ground truth is a concept
   for net1 to estimate.
4. Run PySR with a parsimony penalty on the residual the linear fit
   leaves. For each latent, report formula size against held-out R^2 and
   pick the knee. The target is R^2 of 0.9 or more.
5. Keep the concepts that recur in at least 3 of 5 seeds, matching
   formulas by a correlation above 0.9 on shared held-out traces.
6. Vocabulary sensitivity (C4). Refit with each base variable or operator
   family removed. Every grounding result is reported with this check.

How much of the latents a feature spans is not its importance for
control. Importance is read in step 4, from closed-loop performance with
and without the concept.

## net1's output

A sender learns the link's capacity only while it keeps the link busy.
Below capacity with an empty queue, the delivery rate equals the sending
rate, so the measurements bound capacity from below and say nothing more.
The minimum RTT has the opposite limit: it cannot be measured while a
queue persists. No operating point reveals every quantity, so a single
value per concept can mislead.

net1 can therefore state a distribution over fixed bins per concept,
trained with the log score. The log score's expected value is lowest only
for the true conditional distribution, so overstating or understating
confidence always costs (Gneiting and Raftery, 2007). Accuracy is the R^2
of the belief's mean. Honesty is the coverage of central intervals on
held-out links.

net2 is trained after net1, on a fixed net1's outputs: the sequential
regime of concept bottleneck models (Koh et al., 2020).

## Related work

Encoding and then interpreting is a known pattern in physics ML (SciNet,
Iten et al., PRL 2020; Miles et al., PRB 2021; Cranmer et al., NeurIPS
2020) and in vision concept-bottleneck work (Discover-then-Name CBM, ECCV
2024). The new parts here are the networking instance with emulator
ground truth, the use of grounding to choose concepts for a by-design
controller, and the retained-performance metric.

The closest networking work: Agua (SIGCOMM 2025) maps a controller's
embeddings to text concepts written by an LLM, after the fact. Niedermayer
et al. (IFIP Networking 2026) measure the completeness of a given concept
set. Mutant (NSDI 2025) and UNUM (NSDI 2026) learn latents but do not
ground them. [concepts.md](concepts.md) reviews the concepts the
literature defines. Choosing from such a review is the alternative
compact tests against.

## Terms

| Term | Meaning |
|---|---|
| LCF | Latent congestion factor, expliCCit's term: a physical property of the path at one instant, with an exact value in emulation. |
| the seven LCFs | expliCCit's hand-picked set: mean queueing delay, queue (packets), capacity, free buffer (packets), busy fraction, one-way delay and the flow's receive rate, each over the trailing 100 ms ([architecture.md](architecture.md#ground-truth-labels)). |
| net1 | The estimator. It reads 70 of the 78 values of the step message (`EXPLICCIT_OBS_COLS`) and outputs one belief per LCF. It is trained against ground truth and never with net2. |
| belief | net1's statement about one LCF: a value, an interval, or a distribution over fixed bins. |
| net2 | The decision maker. It sets the window from net1's output alone. |
| discovery controller | The encoder, k latents and decoder of [the design](#the-discovery-controller). Never shipped. |
| latents, z, k | The narrow layer z, of k dimensions, between encoder and decoder. |
| wide controller | The discovery controller with 64 latents: the unconstrained reference of retained performance. |
| teacher | The traditional congestion controller a discovery controller is bootstrapped from: Cubic or BBRv3, as Linux runs them. |
| bootstrap | Cloning a teacher to give reinforcement learning a start: the teacher drives the rollouts and the controller learns the window that sends as it did (`PPOSpec.algo` `clone`). |
| fixed rule | A window of 2^0.25 times the believed BDP of the estimator of the seven LCFs. A common reference in evaluations. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| knee | The smallest width beyond which closed-loop performance stops improving, by a rule fixed before the run. |
| action collapse | The latents carrying the decision itself and not the state behind it. |
| grounding | Fitting each latent to a closed-form formula over the library, on held-out links. |
| library | The candidate terms for grounding: base variables crossed with operators. |
| concept | A grounded formula that recurs across seeds and has been named. Concepts become the LCF labels of the retrained net1. |
| retained performance | [Above](#retained-performance). |
| oracle net2 | net2 trained on and fed the true concept values. A diagnostic only. |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps, and a controller scores on a link the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |
