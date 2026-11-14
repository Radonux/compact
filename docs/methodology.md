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

1. Discover. Train a controller whose decision passes through latents,
   in the closed loop, to control well.
2. Ground. Fit the latents and the decision to short closed-form
   formulas over named variables: emulator ground truth and basic sender
   metrics. Keep the formulas that recur across seeds, in the order in
   which the decision needs them. A person names each one.
3. Retrain. Train a normal net1 and net2 with the first m formulas as
   the LCF labels, through the existing pipeline. Closed-loop performance
   against m says how many concepts a controller needs. Only this
   controller ships. The discovery controller is thrown away.

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

The controller that is grounded has 64 latents, too wide to constrain
anything. It is the wide controller, and performance is measured against
it.

### What the width cannot count

A controller's output is one number. An encoder trained end to end can
compute that number itself, as far as it follows from one step's
measurements, and pass it through one latent. So one latent is always
enough for the decision, whatever the path's state is made of, and the
width at which performance stops improving counts nothing. This is
action collapse. It is a property of the objective and not of the
training: the least a bottleneck must carry to produce an output is the
output. Giving the decoder several tasks, or a preference the encoder
does not see, moves the collapse to as many numbers as there are tasks,
and what is then counted is the tasks. A penalty on the information in
the latents favours the collapse, since the decision is the cheapest
thing to send.

expliCCit cannot collapse this way, because net1 is trained on labels
and never with net2: its outputs are facts about the path. The count
therefore belongs where that holds, in step 3. The retrained net1 can
output only the fitted formulas, so the number of them at which net1 +
net2 keep the wide controller's performance is a count of concepts.

The narrow middle keeps two uses. Its latents are what the decoder
reads, so formulas that span them are enough for a decoder. And widths 1
and 2 stay in every run as the measurement of the collapse: the decision
regressed on the latents, and the score beside the wide controller's.

## How it can fail

These follow from the design, before any data. Each has a check.

| Failure | Why it can happen | Check |
|---|---|---|
| The wide latents hold only the decision | Nothing makes a wide encoder keep the path's state apart from the decision ([above](#what-the-width-cannot-count)). Formulas that span such latents are formulas for the window, not concepts behind it. | Regress the decision on the latents in every run, and read each base variable from the latents. Ground the decision itself beside the latents, and compare the two paths. |
| The order is not the order of need | Grounding orders formulas by how much of the decision they explain on logged steps, which is not their worth in the closed loop. | The count is read from closed-loop performance against m. Compare the order with a random one and with the baselines at each m (C3). |
| The concepts follow the teacher | Reinforcement learning may stay near the controller it started from. | Start from two teachers that work differently, Cubic (loss based) and BBRv3 (model based), and compare the concepts (C5). Measure how far each controller moved from its start. |
| A weak learner looks like a need for concepts | Reinforcement learning may train a net2 on few concepts worse for reasons of optimization alone. | Tune before any claim. Compare with the oracle net2 trained the same way on the full set. |
| No count | The decision may need more concepts than a controller can be retrained on. | Report it. It bears on the two-net design itself. The comparison at seven concepts still stands. |
| The latents fit no short formula | Latents are defined only up to mixing, and narrow layers pack several features into each unit (Locatello et al., ICML 2019). | Report it, and look at what the latents do encode. |
| The vocabulary decides the answer | A person chooses the grounding library. | Refit with each base variable or operator family removed (C4). |
| Leakage | net1's soft outputs can carry more than the concepts. | Compare net2 on net1's soft outputs with net2 on exact concept values. |

## Claims

| | Claim | How it fails |
|---|---|---|
| C1 | Count. Closed-loop performance of net1 + net2 on the first m fitted concepts stops improving at a small m, the same across seeds of the wide controller, and at that m it matches the oracle net2 on the full set. | No m short of the full set keeps the performance, or the curve differs between seeds. |
| C2 | Grounding. The wide controller's latents and its decision are spanned by a few short closed-form formulas over named variables (a flexible fit on them reaches a flexible fit on the whole library), and the same formulas recur in at least 3 of 5 seeds. | They fit no short formula, or the formulas differ between seeds. |
| C3 | Retention. net1 + net2 trained on the fitted concepts retain more of the wide controller's closed-loop performance than the same number of concepts chosen another way: the hand-picked LCFs, greedy subset search, SHAP-ranked inputs, a random subset. | A baseline ties or wins at the same budget. |
| C4 | Vocabulary. With one base variable or operator family removed from the library, the selected concepts change only when the removed item was one of them. | The selection follows whatever the library holds. |
| C5 | Start. The discovery controller yields the same concepts from different teachers. | The concepts describe one teacher. |

Every threshold and choice rule a claim depends on is written down before
the run that decides it.

Two things are not claimed. A bottleneck does not make a network
explainable, and its width does not count concepts: grounding against
ground truth names them, and retraining on them counts them. And a closed-form concept is exact, which does not make it
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
| Decide before running | A count, a threshold or a winner is chosen by a rule written before the run. |
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
   - Continue each by reinforcement learning on the log utility
     (`experiments/controller-rl.json`): PPO from the clone's actor and
     value network. Tune it and confirm over seeds.
   - The widths are 64, the wide controller, and 1 and 2, which measure
     action collapse. A reference without an encoder runs beside them.
     Report each against the teachers themselves and the fixed rule on
     the seven hand-picked LCFs (`experiments/width_curve.py`). A run's
     score is that of the policy it leaves, not of its best iterate.
   - Check for action collapse at every width: the decision regressed on
     the latents, on the controller's own rollouts.
   - Compare the two teachers' controllers (C5), and report how far
     reinforcement learning moved each from its start: the RMS difference
     in log2 of the window between the two policies on the same rollouts.
3. Grounding, on the wide controllers' latents and decisions and the
   steps of their own rollouts ([below](#grounding)).
4. Retrain and compare. Compute labels from the frozen formulas, train
   net1 and net2 as usual, and measure retained performance.
   - Train the oracle net2 on the true concept values.
   - Plot retained performance against the number of concepts m, for
     m = 1, 2, 3, 5, 7 and the full set, taking concepts in the order of
     the decision's path. The count (C1) is the smallest m whose mean
     over seeds is within one standard error of the best m's mean, the
     standard error pooled over m. m = 7 is expliCCit's budget.
   - Baselines at the same budget: the hand-picked LCFs, a greedy forward
     subset search over the base variables, the wide controller's inputs ranked
     by SHAP, and a random subset.

A step is done when it has a release with the code, the configuration,
the seeds and the raw outputs, and [status.md](status.md) says what ran,
what was searched, the held-out numbers and what remains UNRUN.

## Grounding

Grounding is meaningful only for latents that agree across seeds. Two
things are grounded for each controller, in the same way: its latents,
which are all its decoder reads, and its decision, log2 of the window it
set. The latents' path says what the controller represents. The
decision's path says what the window depends on, and gives the order in
which concepts are tried in retraining. Below, "the latents" stands for
either.

The data is the controller's own. Each controller drives the emulator
without exploration noise, with every step kept
(`experiments/ground.py rollouts`): on a sample of the training links,
where the fits are made, and on the validation links, where they are
scored. Every controller gets the same links and seeds. A row is one
decision. Its latents are computed from the stored measurements through
the controller's own encoder, so they are exactly what the decoder read.
Results for one controller use its own rollouts. Comparisons between
controllers use the rollouts of all of them.

The library is built mechanically: base variables crossed with operators. The
base variables are the hidden state of the path and the sender's
metrics. The hidden state is every label the step data stores, in each
window it is stored in, and every composite of them: a fixed formula over
a few labels that the literature defines, such as the load factor or the
fair share ([architecture.md](architecture.md#ground-truth-labels)). The
library reads both lists from the code, so a label or composite added
there takes part in the fitting with no change here. The operators are a windowed mean, minimum and
maximum over two horizons and a first difference.

1. Probe (`experiments/ground.py probe`). The decision regressed on the
   latents, which is the action-collapse check. Then linear fits between
   the latents and the base variables, in both directions, and between
   the latents of two seeds. They show which variables the latents carry,
   how much of a latent the vocabulary can express at all, and whether
   seeds agree. Each is a lower bound.
2. Select (`experiments/ground.py select`). Every fit is made on places:
   a latent and a feature are each replaced by their rank among the
   rows. The encoder may hold a variable in any monotone form, because
   the decoder can undo it, and on places no such form changes a fit.
   Fit all k latents jointly and linearly on the library, adding one
   feature at a time, the one that most raises the fit. Then order the
   chosen features by taking them away again, each time the one the fit
   loses least without: a feature chosen early because it stood in for a
   mix of later ones falls to the end. The selected features are the
   candidate concepts. The joint fit needs no rotation of the latents:
   it asks which features span them, and any invertible mix of the
   latents has the same span.
3. Report the path twice: on the whole library, and on ground truth
   alone. The latents are computed from the sender's message, so on the
   whole library the sender's own readings can stand in for the path
   quantities they measure. Only a feature of ground truth is a concept
   for net1 to estimate. Beside each linear number, report a flexible
   fit (a small network) on the same features and on the whole library.
   A mix of concepts through a nonlinear form is not linear even on
   places, so a linear R^2 is a lower bound and no fixed bar applies to
   it. The selected features are enough when the flexible fit on them
   reaches the flexible fit on the whole library. What the flexible fit
   on the whole library misses is no function of the library: the
   sender's estimation error, or a variable nobody listed.
4. Search for formulas over the stored labels behind the selected
   features, so that a concept nobody listed can still be written:
   products of two or three labels or their inverses, and sums and
   differences of two such terms. Two terms are added only when they
   have the same dimension in SI units, so no coefficient is fitted.
   This is the form of the composites in the literature. Formulas of
   two factors are tried over every stored label, because a difference
   of two nearly equal variables shows in no linear fit on either, so
   the selected features need not lead to it. Longer formulas are tried
   over the labels of the best of those and of the selected features. A
   formula is scored by how well the latents predict it, since what the
   decoder can read from the latents is what they hold. Of formulas that order
   the rows alike, the simplest is kept. The kept formulas join the
   library and the path is selected again. A formula that is chosen
   becomes a composite in the code, and from then on is in the library
   under every operator. PySR on the selected labels is the next step
   only where the flexible fit shows a form beyond these.
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
| latents, z, k | The layer z, of k dimensions, between encoder and decoder. Its width is not a count of concepts. |
| wide controller | The discovery controller with 64 latents: the unconstrained reference of retained performance. |
| teacher | The traditional congestion controller a discovery controller is bootstrapped from: Cubic or BBRv3, as Linux runs them. |
| bootstrap | Cloning a teacher to give reinforcement learning a start: the teacher drives the rollouts and the controller learns the window that sends as it did (`PPOSpec.algo` `clone`). |
| fixed rule | A window of 2^0.25 times the believed BDP of the estimator of the seven LCFs. A common reference in evaluations. |
| believed BDP | net1's mean capacity times twice its mean one-way delay, in 1500-byte packets (`fit.window_target`). |
| count | The smallest number m of fitted concepts with which net1 + net2 keep their performance, by a rule fixed before the run. |
| action collapse | The latents carrying the decision itself and not the state behind it. One latent is always enough for that. |
| grounding | Fitting each latent to a closed-form formula over the library, on held-out links. |
| library | The candidate terms for grounding: base variables crossed with operators. |
| concept | A grounded formula that recurs across seeds and has been named. Concepts become the LCF labels of the retrained net1. |
| retained performance | [Above](#retained-performance). |
| oracle net2 | net2 trained on and fed the true concept values. A diagnostic only. |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`). An episode scores its mean over steps, and a controller scores on a link the mean over the link's episodes. The log term makes competing flows settle at proportionally fair shares (Kelly, 1998). |
