# Methodology

compact asks one question: can the concepts of a concept-bottleneck
congestion controller be chosen from what a trained controller actually
uses, instead of being picked from a literature review? This page states
the idea, the claims, the terms, how results are measured and the plan.
What has run, what it showed and what is still open is in
[log.md](log.md).

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

1. **Discover.** Train a model whose output passes through a narrow layer
   of k latents. The width at which its performance stops improving
   estimates how many concepts the task needs. The model is the codec
   ([below](#the-codec)): it predicts the flow's own future from the
   latents, so no teacher defines what it learns.
2. **Ground.** Fit the latents to short closed-form formulas over named
   variables (emulator ground truth plus basic sender metrics) under a
   harsh complexity penalty. Keep the formulas that are stable across
   seeds. A person names each one.
3. **Retrain.** Train a normal net1 and net2 with the fitted formulas as
   the LCF labels, through the existing pipeline unchanged. Only this
   controller ships, and the model of step 1 is thrown away. Every
   decision of the shipped controller passes through exact, checkable
   quantities, so it keeps expliCCit's faithfulness property.

The main metric is retained performance: the performance of net1 + net2 on
the fitted concepts over the performance of the unconstrained teacher,
measured in closed-loop rollouts, and compared with other ways of choosing
concepts at the same budget ([below](#retained-performance)).

## The codec

What must a controller know at a step? It acts on a path it sees only
through its own measurements. The state it needs is whatever, together
with the actions it then takes, decides what happens to its flow next. A
representation from which the future of the flow's measurements can be
predicted under any actions is therefore sufficient for controlling
anything defined on those measurements (predictive representations of
state; Littman, Sutton and Singh, NIPS 2001). The codec
(`experiments/codec.py`) is built on that:

- **Encoder.** One step's message (net1's observation columns) to k
  latents z, each normalized to unit variance.
- **Decoder.** From z and the window's log2 change up to each horizon, it
  predicts what the flow measures 60 ms to 960 ms later: smoothed RTT,
  delivery rate and loss. It sees nothing of the step but z, and horizon
  h sees only the actions up to h.
- **Ordered latents.** Each training example keeps a random prefix of z
  (nested dropout; Rippel, Gelbart and Adams, ICML 2014). One model then
  gives a score at every width from 0 to k, and latent i matters more
  than latent i + 1. Ordering costs skill, most at small widths, so the
  count is read from models trained at one fixed width with every latent
  kept; the ordered model ranks the latents.
- **Scores.** R^2 on held-out links, and skill over persistence: 1 minus
  the codec's squared error over that of predicting that nothing changes.
  Smoothed RTT and delivery rate change slowly, so R^2 alone rewards a
  codec that only copies the present; skill is 0 for such a codec.

Three reasons to prefer this to imitating a teacher through the same
layer. A student's latents can collapse to the teacher's action, and the
codec's cannot, because the decoder is given the actions. The concepts of
a student describe its teacher, and the codec has none. And a teacher
worth imitating must first be trained and tuned without concepts, which
is a project of its own. The student remains the second route, and
agreement between the two is evidence for C5.

What the codec does not cover:

- **Utility.** The utility depends on the capacity, which no measurement
  states. An optional group of targets adds the utility's terms; it uses
  labels, so latents shaped by it are no longer label-free.
- **Logged actions.** The actions in the data are those of the logging
  controllers and follow from the path's state: a halved window reveals a
  loss. The decoder can read state from the actions it is given, which
  lowers what z must hold.
- **Control.** Predictive skill is not closed-loop performance. That a
  controller needs no more than the first k latents is shown only by a
  controller that reads them, in the closed loop.

## Claims

| | Claim | How it can fail |
|---|---|---|
| C1 | Count: the codec's held-out skill stops improving at a small width k, the same across seeds, and a controller that reads only the first k latents does as well in the closed loop as one that reads all | no knee: the controller does not compress into a few factors, which bears on the two-net design itself |
| C2 | Grounding: at that width each latent is a short closed-form formula over named variables (held-out R^2 of 0.9 or more at the knee of size against fit), and the same formulas recur in at least 3 of 5 seeds | the latents fit no short formula, or differ between seeds |
| C3 | Retention: net1 + net2 trained on the fitted concepts retain more of the teacher's closed-loop performance than the same number of concepts chosen any other way (the hand-picked LCFs, greedy subset search, SHAP-ranked inputs, a random subset) | a baseline ties or wins at the same budget |
| C4 | The vocabulary is not the choice: with one base variable or operator family removed from the library, the selected concepts change only when the removed item was one of them | the selection follows whatever the library holds, which is the objection that a person still chooses the candidates |
| C5 | Independence of the route: the codec and students of differently trained teachers yield the same concepts | the concepts describe one objective or one teacher, not the task |

Every threshold and choice rule a claim depends on is written down
before the run it decides.

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
- The codec and the students are never shipped and nothing is claimed
  about them as controllers. Claims about the shipped controller rest on
  the retrained net1 and net2.

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
ground them. [concepts.md](concepts.md) reviews the concepts the
literature defines. Choosing from such a review is the alternative this
project tests against; the review also supplies a baseline, names for
fitted formulas, and the base variables of the library.

## Terms

| Term | Meaning |
|---|---|
| codec | The discovery model: an encoder to k latents and a decoder of the flow's future from the latents and its actions ([above](#the-codec)). |
| anchor, horizon | An anchor is a step the encoder reads; a horizon is how far ahead of it the decoder predicts. |
| persistence | The forecast that every target keeps its current value. |
| skill | 1 minus the codec's mean squared error over persistence's, averaged over horizons and targets: 0 for copying the present, 1 for a perfect forecast. |
| teacher | The controller whose behaviour is distilled. An unconstrained teacher reads raw measurements and does not pass through net1 (the black-box controller of step 5). The two Phase 0 teachers are not unconstrained: each acts through an estimator of the seven LCFs. |
| student | A network trained to imitate the teacher, whose decision passes through the latents. It is a measuring device and is never deployed. |
| latents, z, k | The narrow layer z, of k dimensions, of the codec or of a student. Everything after it reads z alone. In the codec the latents are ordered, and "width" means the number of leading latents kept. |
| head | An output computed from z. The action head predicts the teacher's action; the next-state head predicts the next step's log sRTT and log delivery rate; the value head predicts the teacher's Q-values on a fixed grid of actions. |
| knee | The smallest width beyond which performance stops improving. For the codec: the smallest fixed width whose skill on validation links is within 5% of the skill of the model with no narrow layer (64 latents), both at the tuned settings. |
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

Log utility is at most about zero (the log of a share of capacity, minus
penalties), so a plain ratio of two log utilities is not a fraction
retained: a larger ratio would mean a worse controller. The scale must
therefore be fixed before step 6 runs. One candidate is the fraction of
the teacher's lead over a fixed reference controller, (score - reference)
/ (teacher - reference), reported with the paired difference in log
utility and its interval.

## Plan

The plan is a starting point. A step changes order, is added or is dropped
when the evidence points elsewhere, and the log entry for it says why.

The order follows from what each step can show. The codec needs only
logged steps, so it comes first and is trained as well as we can train
it. Grounding is meaningful only for latents that are stable across
seeds. Control needs the closed loop and a tuned controller, so its
closed-loop part comes late; its question (does a controller need more
than these concepts?) is asked as soon as grounding has named concepts,
and before a formula search refines them, because the answer decides
whether they are worth refining.

1. **Setup.** The documents state the idea, the claims and the plan, and
   the logged step data is a release of this repository.
2. **The codec on logged steps.** Search its settings on validation
   links, confirm the winner over 5 seeds, and report the width curve
   beside two references: models trained at one fixed width, and a model
   with no narrow layer.
3. **Strengthen the codec.** Try what could raise its skill and keep what
   does on validation links: a history of messages as input, the
   utility's terms as targets, an information bottleneck in place of a
   hard width. Then run the test links once.
4. **Grounding.** Build the library mechanically: base variables (hidden
   state and sender metrics) crossed with operators (windowed mean, min
   and max over 2 to 3 horizons, first difference, pairwise ratio and
   product, log).
   - First look, before any formula (`experiments/ground.py probe`):
     linear fits between the latents and the base variables, in both
     directions, and between the latents of two seeds. It says which
     variables the latents carry, how much of a latent the vocabulary
     can express at all, and whether seeds agree, each as a lower bound.
   - Select (`experiments/ground.py select`): fit all k latents jointly
     and linearly on the library, adding one feature at a time, the one
     that most raises the fit. The variables are on logarithmic scales,
     so a linear fit is a fit by products and ratios of powers, and the
     selected features are themselves the candidate concepts. A greedy
     path replaces the group lasso first planned: it gives fit against
     size directly, and taking out a variable that was never picked
     provably leaves it unchanged. The group lasso stays the fallback if
     the picks differ between seeds.
   - Report the path twice: on the whole library, and on ground truth
     alone. The latents are computed from the sender's message, so on
     the whole library they can be fitted by what the sender measured,
     which names no state of the path. Only a feature of ground truth
     is a candidate concept for net1 to estimate.
   - Run PySR, with a parsimony penalty, only on the residual the linear
     fit leaves, and only after the offline check of step 5 has shown
     the selected concepts to be worth refining.
   - For each latent, report formula size against held-out R^2 and pick
     the knee. The target is R^2 of 0.9 or more.
   - Keep the concepts that recur in at least 3 of 5 seeds, matching
     formulas by a correlation above 0.9 on shared held-out traces.
   - Vocabulary sensitivity (C4): refit with each base variable or
     operator family removed, and report whether the selected concepts
     change only when a removed item was one of them.
5. **Control.** First, offline and on logged steps: predict each logging
   controller's next action on held-out links from the codec's latents,
   from the selected ground-truth features, from the hand-picked LCFs
   and from the raw message, at the same number of inputs. It is a
   proxy: it shows whether a set of concepts holds what known
   controllers act on, and it supports no claim about control. Next,
   still on logged steps: a student that predicts the logged actions
   through k latents, whose width curve counts the concepts the actions
   need and whose latents are grounded as the codec's are, with the
   selection scored on the action and not only on the fit of the
   latents. Then check the count in the closed loop: controllers that read
   the first k latents, or the grounded concepts, against one that reads
   raw measurements. This needs a tuned black-box controller, which also
   serves as a teacher: distill students from it with DAgger through a
   k-latent layer, and from the two logged teachers
   ([log.md](log.md#before-2026-11-11-the-phase-0-prototype)), and
   compare their concepts with the codec's (C5).
6. **Retrain and compare.** Compute labels from the frozen formulas, train
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
| The codec copies the present | Report skill over persistence beside R^2, at every width |
| Predictive is not decision-relevant | The closed-loop check of step 5; until then no claim about control |
| Action collapse (students) | Check z against the teacher's action directly, in every run |
| Dependence on the route | Compare the concepts of the codec and of students of different teachers (C5) |
| Leakage | Compare net2 on net1's soft outputs with net2 on the exact concept values |

## Definition of done, for each step

- A release with the code, the configuration, the seeds and the raw
  outputs.
- An entry in [log.md](log.md): what ran, what was searched, the held-out
  numbers, and what remains unrun.
- No performance claim without the tuning the rules in `AGENTS.md`
  require. Anything not run is labelled UNRUN.
