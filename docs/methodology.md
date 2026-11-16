# Methodology

The question and the idea are in the [README](../README.md#the-question).
What has run is in [status.md](status.md).

## The design

expliCCit's controller is an encoder and a decoder with a hand-made
middle: net1 encodes measurements into LCFs, net2 decodes LCFs into a
window. compact trains the same shape with the middle left free, then
names what the middle learned.

1. Discover. Train a controller whose decision passes through latents,
   in the closed loop, to control well.
2. Ground. Recover the latents of the best-trained controllers as short
   closed-form formulas over named variables by symbolic regression. A
   person names each concept.
3. Retrain. Train a normal net1 and net2 with the first m formulas as
   the LCF labels. Closed-loop performance against m says how many
   concepts are needed. Only this controller ships.

### The discovery controller

`PPOSpec.latents` in `src/expliccit_env/train/recurrent.py`.

| Part | In place of | What it is |
|---|---|---|
| Encoder | net1 | An MLP from one step's message (net1's 70 inputs and scaler) to k latents. |
| Decoder | net2 | A GRU that reads the latents and its own previous decision. It decides once per minimum RTT, clamped to 10 to 50 ms, and holds the decision in between. |
| Output | | log2 of the window over a fixed 2^7 packets, so nothing measured passes around the latents. |

It is cloned from a teacher (Cubic or BBRv3: the teacher drives, the
controller learns the window that sends as it did), then trained by
reinforcement learning on the log utility. While it learns from the
teacher it reads its own previous decisions, never the teacher's. The
grounded widths are 7, 10 and 16. There is no budget on the number of
concepts; past about 16 they are hard to explain.

### What the width cannot count

An encoder trained end to end can compute the decision itself and pass
it through one latent, so the width at which performance stops improving
counts nothing (action collapse). The count is read in retraining, where
net1 can output only the fitted formulas.

## Rules of evidence

| Rule | Meaning |
|---|---|
| Closed loop | A result about a controller rests on rollouts in the emulator. |
| Split by link | Data and evaluation are split by link (`expliccit_env.train.data.link_key`), never by row or episode. |
| Score | Log utility, with differences paired by link and 95% bootstrap intervals over links. |
| Tune first | No result about a learned model before it is tuned on validation links. |
| Decide before running | A threshold or a winner is chosen by a rule written in the experiment file before it runs. |
| Test once | The test links run once per candidate, after every choice is fixed. |

### Comparing controllers

- BBRv3 is the reference for every comparison. Each controller is
  reported as its log utility minus BBRv3's on the same episodes, paired
  by link, without and with a competing flow, with a 95% bootstrap
  interval over links.
- A controller is trained well when it is within the margin of BBRv3
  both alone and with a competing flow: in each, the whole interval of
  its difference lies above -0.1 (about 10% of throughput at equal delay
  and loss). Below that, the target is the log utility over all
  conditions, a larger share of a competing flow included.
- Fairness is left until everything else is done (ccperf,
  `experiments/ccperf.py`).
- A run keeps its best evaluated policy on the validation links and
  scores as that policy. Since it is chosen on those links, the test
  links give the unbiased number.

## Plan

A step moves, is added or is dropped when the evidence says so.

1. Setup. Every input is a release of this repository.
2. Train the discovery controllers: bootstrap from each teacher, then
   reinforcement learning, tuned on validation links, beside a reference
   without an encoder, until they are trained well against BBRv3.
3. Ground the best-trained controllers ([below](#grounding)).
4. Retrain net1 and net2 on the concepts, m = 1, 2, 3, 5, 7, 10, 16 and
   all of them, with an oracle net2 on the true values, and read the
   count. Compare with the hand-picked LCFs, a greedy subset of the base
   variables, SHAP-ranked inputs and a random subset at the same m.

A step is done when it has a release and [status.md](status.md) says
what ran, what was searched, the held-out numbers and what remains
UNRUN.

## Grounding

Many sets of concepts can explain the same latents: the library's
variables overlap, and a network can hold its concepts in any rotation
or mix. One set that works is enough, so seeds are not compared. The
concepts come from the best-trained networks by validation log utility:
the single best, or the best few pooled. Retraining shows whether the set
works.

Each controller drives the emulator without exploration noise, every
step kept (`experiments/ground.py rollouts`), on a sample of the
training links (fits) and on the validation links (scores). A row is one
decision, and its latents are computed through the controller's own
encoder.

The library is every stored label and composite
([architecture.md](architecture.md#ground-truth-labels)) under a
windowed mean, minimum and maximum over two horizons and a first
difference. The sender's own window is held in every fit: it predicts
the latents and the next decision without saying anything about the
path, and so does everything it sets.

1. `probe`: the decision and its change regressed on the latents, and
   linear fits between the latents and the base variables.
2. `select`: fits on ranks. Features are added one at a time to the
   window's own, the one that most raises the joint fit of the latents,
   then ordered by taking them away again. This path, on ground truth
   only, gives PySR its inputs. A search over products and sums of a few
   labels adds composites to the library.
3. `concepts`: the sender's own window is the first concept, since the
   latents carry the decision and the window in force is the previous
   one; the rest are what the latents hold beyond it. A small network on
   the window's features takes the window out of each latent. PySR fits
   the rest over the window in force, in packets, and the path's first 8
   features in ms, Mbit/s or kB, with + - * / and log and at most 20
   nodes; the formula kept is the simplest within 0.01 of the best R^2
   on the validation links. The ceiling is the same network on the
   window's and the path's features. A formula is recovered when it
   reads the path, the ceiling is at least 0.1 and PySR reaches 0.8 of
   it. Pooled over networks, of formulas whose values correlate above
   0.9 the simplest is kept.

## net1's output

A sender cannot observe every quantity at every operating point (no
capacity while the link is idle, no base RTT while a queue persists), so
net1 can state a distribution over fixed bins per concept, trained with
the log score. net2 is trained after net1, on a fixed net1.

## Related work

Encoding then interpreting is known in physics ML (SciNet, Iten et al.,
PRL 2020; Cranmer et al., NeurIPS 2020) and vision (Discover-then-Name
CBM, ECCV 2024). In networking, Agua (SIGCOMM 2025) labels a
controller's embeddings with LLM-written concepts after the fact, and
Mutant (NSDI 2025) and UNUM (NSDI 2026) learn latents without grounding
them. [concepts.md](concepts.md) reviews the literature's concepts.

## Terms

| Term | Meaning |
|---|---|
| LCF | Latent congestion factor, expliCCit's term: a physical property of the path at one instant, exact in emulation. |
| the seven LCFs | expliCCit's hand-picked set: mean queueing delay, queue, capacity, free buffer, busy fraction, one-way delay and receive rate, over the trailing 100 ms. |
| net1, net2 | The estimator (70 of the 78 message values to one belief per LCF, trained against ground truth) and the decision maker (window from net1's output alone). |
| discovery controller | [Above](#the-discovery-controller). Never shipped. |
| fixed rule | A window of 2^0.25 times the believed BDP (net1's capacity times twice its one-way delay) of the estimator of the seven LCFs. The earlier reference, before BBRv3. |
| concept | A formula recovered from a best-trained network's latents, once named. |
| count | The number of concepts at which the log utility of net1 + net2 retrained on them stops rising. |
| log utility | Per 10 ms step, from ground truth: ln(max(receive rate / capacity, 0.01)) - 0.5 queueing delay / base RTT - 4 drop fraction (`train/score.py`), averaged over steps and episodes. |
