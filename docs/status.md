# Status

What has run, what it showed and what is open. Rewritten in place; the
history is in git. The steps are those of the
[plan](methodology.md#plan).

## Where the project stands

Against BBRv3, Cubic's width 10 is trained well over three seeds, as are
both references without an encoder and the fixed rule; the other widths
fall short with a competing flow ([below](#against-bbrv3)).

Step 3 has its result. The latents carry the decision itself, a linear
function of them at every width, and the window in force is the previous
decision, so the window explains most of every latent. Hiding the
packets in flight from the encoder does not change that and leaves
BBRv3's controllers no worse; Cubic's widths 10 and 16 do worse
([below](#the-window-hidden-from-the-encoder)).
Beyond the window, five formulas are recovered. Without their constants,
which stand for no path variable, three read as the mean queueing delay
over 240 ms and the flow's arrival rate, alone or over the base RTT; the
other two are weak or an emulator effect ([grounding](#grounding)).
These formulas come from several networks. By the owner's choice the
concepts come from one network, found by a search over seeds: none of
the 36 earlier controllers decomposes, BBRv3's come closest, and the
first two batches, 64 new draws, have no hit; the third is training
([below](#the-seed-search)). Longer
training with a falling rate did not pay: the best policies come early
and most runs end below their start; at a tenth of the rate Cubic's
drift less and gain no more ([below](#longer-training)).

| Step | State |
|---|---|
| 1 Setup | done: the estimator of the seven LCFs and the rule graph as releases |
| 2 Train | bootstrapped from Cubic and BBRv3; reinforcement learning tuned at width 10 and confirmed at widths 7, 10 and 16 over three seeds; against BBRv3 Cubic's width 10 is trained well; longer training did not pay, at either rate |
| 3 Ground | the latents carry the decision, whether or not the encoder sees the window; beyond it, the queueing delay and the arrival rate; the seed search for one network that decomposes is running |
| 4 Retrain | UNRUN. Built: `train run` for net1 on any stored label or composite, `train ppo` for net2, live truth for oracles. Not built: labels from fitted formulas beyond the composites. |

## Training the discovery controller

Bootstrap (`exp-controller-teachers-22`): widths 1 to 64 cloned from
Cubic and BBRv3, three seeds. Every start ended 0.55 to 2.38 below the
fixed rule, and a low imitation loss did not give a stable controller. A
start taught from the fixed rule (`exp-controller-clone-21`) ended 0.03
to 0.05 below the rule at every width; it leans on the seven LCFs and is
not used further.

Reinforcement learning at one untuned point (`exp-controller-rl-24`,
120 iterations) brought the clones to 0.13 to 0.83 below the rule. At
widths 1 and 2 the decision is read from the latents at R^2 0.80 to
0.93, and width 1 scored as well as width 64: action collapse, so the
width counts nothing. Width 64 trained no better and was dropped.

A settings search at width 10 (`exp-controller-rl-search-29`: learning
rate, clip and exploration noise, two seeds) and its confirmation at
widths 7, 10 and 16 with the reference without an encoder
(`exp-controller-rl-confirm-31`, three seeds) chose, by rules fixed in
the files, learning rate 0.0003, clip 0.2 and noise 0.2 for Cubic, and
learning rate 0.00003, clip 0.2 and noise 0.1 for BBRv3. Learning rate
0.0003 with noise 0.1 diverged for both.

| Log utility minus the fixed rule, validation links, mean of 3 seeds | none | 7 | 10 | 16 |
|---|---|---|---|---|
| Cubic | -0.16 | -0.59 | -0.17 | -0.31 |
| BBRv3 | -0.11 | -0.20 | -0.28 | -0.28 |

- Scores still rose by 0.02 to 0.03 per 20 iterations at 120, so 240
  more iterations from each run were started
  (`experiments/controller-rl-longer.json`) and cancelled once the
  first Cubic runs fell: width 16, seed 1, from -0.40 to -2.99 by iteration 60,
  width 7, seed 2, from -0.13 to -2.99 after iteration 100, both to the
  smallest window, and the reference, seed 3, from -0.23 to -0.32. The
  first update after the restart is a large step (a fresh optimizer:
  up to 71% of decisions clipped), and at learning rate 0.0003 the
  policy then drifts into the smallest window, where the reward's floor
  of 1% of capacity leaves no gradient back. The trainer now ramps the
  rate up after a restart, can anneal it to a final rate, and keeps the
  best evaluated policy (`lr_warmup`, `lr_end`, `policy-best.onnx`).
- Cubic's width 7, seed 1, fails at both confirmed settings: from the
  clone with the smallest window it settles on a small window, using 25%
  to 46% of the link. Why that clone does is UNRUN.
- More than three seeds need more clones from the bootstrap.

### Longer training

The same continuation, 240 iterations with the rate falling to a tenth
(`exp-controller-rl-scheduled-39`), scored against BBRv3 on the
validation links. Each run keeps its best evaluation, chosen on those
links; its number on the test links is UNRUN.

| Log utility minus BBRv3's, all conditions, mean of 3 seeds | Start | Best | Last | Seeds trained well at their best |
|---|---|---|---|---|
| Cubic, reference | +0.13 | +0.16 | -0.77 | 2 |
| Cubic, width 7 | -0.29 | +0.02 | -0.29 | 1 |
| Cubic, width 10 | +0.12 | +0.19 | +0.04 | 3 |
| Cubic, width 16 | -0.01 | -0.00 | -0.71 | 1 |
| BBRv3, reference | +0.18 | +0.22 | +0.20 | 3 |
| BBRv3, width 7 | +0.09 | +0.12 | +0.01 | 2 |
| BBRv3, width 10 | +0.01 | +0.09 | -0.01 | 0 |
| BBRv3, width 16 | +0.01 | +0.06 | -0.05 | 0 |

- Keeping the best policy gains 0.01 to 0.08 over the start at each
  width (0.31 at Cubic's width 7, from seed 1's failed start). The best
  comes by iteration 59 in 13 of the 18 runs with an encoder, and 14 of
  them end below their start. BBRv3's drift too, at a rate ten times
  lower than Cubic's, while BBRv3's references hold.
- Cubic's reference, seed 3, fell to -2.67 by the end.
- Where a width falls short, it is with a competing flow, as
  [below](#against-bbrv3).
- Cubic's continuation from these best policies at BBRv3's rate,
  0.00003 falling to 0.000003 over 240 iterations
  (`exp-controller-rl-cubic-slow-45`): the best gains 0.00 to 0.04 over
  the start (none, 0.03, 0.03, 0.04 for the reference and widths 7, 10,
  16), and the last ends at +0.10, -0.07, +0.17, -0.02, against -0.77,
  -0.29, +0.04, -0.71 at the higher rate. 10 of 12 still end below their
  start, by less. Seeds trained well at their best: 3, 2, 3 and 1.

### The window hidden from the encoder

The controllers trained again from the teachers with the packets in
flight, the delivered BDP over the window and the previous decision
hidden from the encoder, 11 of its 70 inputs
(`exp-controller-hidden-window-44`: 40 iterations of cloning, 120 of
reinforcement learning, the best policy kept). Log utility minus
BBRv3's on the validation links, the mean of three seeds per link, as in
the table [below](#against-bbrv3):

| Controller | Alone | With a competing flow | Trained well | Share |
|---|---|---|---|---|
| Cubic, width 7 | +0.16 (+0.09 to +0.23) | -0.24 (-0.37 to -0.10) | no | 0.50 |
| Cubic, width 10 | -0.33 (-0.47 to -0.19) | -0.62 (-0.82 to -0.42) | no | 0.42 |
| Cubic, width 16 | -0.27 (-0.36 to -0.17) | -0.14 (-0.27 to -0.02) | no | 0.49 |
| BBRv3, width 7 | +0.20 (+0.09 to +0.30) | +0.02 (-0.14 to +0.17) | no | 0.59 |
| BBRv3, width 10 | +0.25 (+0.16 to +0.34) | -0.01 (-0.18 to +0.14) | no | 0.57 |
| BBRv3, width 16 | +0.24 (+0.15 to +0.33) | -0.06 (-0.23 to +0.11) | no | 0.57 |

- BBRv3's are no worse than with the window seen: closer to BBRv3 with a
  competing flow, a little lower alone. Per seed, 3 of 9 are within 0.1
  of BBRv3 both ways at their best evaluation.
- Cubic's width 10 is worse both ways and width 16 worse alone; width 7
  is better, without the failed start of seed 1. None is trained well.
- These are best policies, chosen on the same links, against the last
  policies of the confirmation below; the test links are UNRUN.

### Against BBRv3

The confirmed controllers, their references without an encoder, the
fixed rule and the teachers on the same validation episodes
(`exp-versus-bbr3-41`). Log utility minus BBRv3's, the mean of three seeds per link,
with a 95% interval over 21 links; trained well is within 0.1 both alone
and with a competing flow. The share is the observed flow's part of what
all flows took, with a competing flow.

| Controller | Alone | With a competing flow | Trained well | Share |
|---|---|---|---|---|
| Cubic, width 7 | -0.20 (-0.42 to +0.01) | -0.38 (-0.58 to -0.19) | no | 0.50 |
| Cubic, width 10 | +0.20 (+0.07 to +0.32) | +0.04 (-0.08 to +0.17) | yes | 0.60 |
| Cubic, width 16 | +0.23 (+0.13 to +0.34) | -0.26 (-0.47 to -0.06) | no | 0.51 |
| Cubic, reference | +0.17 (+0.08 to +0.27) | +0.09 (-0.03 to +0.20) | yes | 0.65 |
| BBRv3, width 7 | +0.28 (+0.17 to +0.38) | -0.10 (-0.25 to +0.04) | no | 0.53 |
| BBRv3, width 10 | +0.31 (+0.21 to +0.40) | -0.28 (-0.49 to -0.09) | no | 0.49 |
| BBRv3, width 16 | +0.28 (+0.18 to +0.37) | -0.26 (-0.44 to -0.08) | no | 0.50 |
| BBRv3, reference | +0.31 (+0.25 to +0.37) | +0.06 (-0.04 to +0.15) | yes | 0.56 |
| The fixed rule | +0.45 (+0.40 to +0.52) | +0.13 (+0.00 to +0.24) | yes | 0.62 |
| Cubic itself | -1.01 (-1.21 to -0.81) | +0.03 (-0.04 to +0.10) | no | 0.53 |
| BBRv3 itself | | | | 0.53 |

- Alone, every controller but Cubic's width 7 is above BBRv3.
- With a competing flow, of the widths only Cubic's 10 is within 0.1;
  the others take a smaller share than their references.

## Grounding

The library is every stored label (37 columns over 100 ms and 10 ms
windows) and 12 composites under eight operators, with four sender
measurements. Every fit holds the sender's own window; the earlier
groundings that did not (`exp-ground-teachers-26`, `exp-ground-rl-27`)
picked the window first and are superseded.

The confirmed controllers (`exp-ground-rl-confirm-32`), latents pooled
per width, R^2 of a small network on the validation links:

| | Cubic 7 | Cubic 10 | Cubic 16 | BBRv3 7 | BBRv3 10 | BBRv3 16 |
|---|---|---|---|---|---|---|
| The window alone | 0.52 | 0.53 | 0.52 | 0.67 | 0.65 | 0.73 |
| The window and the first 8 features of ground truth | 0.80 | 0.82 | 0.83 | 0.85 | 0.86 | 0.86 |
| The whole library | 0.86 | 0.86 | 0.86 | 0.85 | 0.87 | 0.87 |
| The decision from the latents, linear | 0.87 | 0.93 | 0.95 | 0.92 | 0.94 | 0.96 |
| With the window hidden from the encoder (`exp-controller-hidden-window-44`): | | | | | | |
| The window alone | 0.56 | 0.52 | 0.50 | 0.81 | 0.73 | 0.73 |
| The window and the first 8 features of ground truth | 0.83 | 0.78 | 0.80 | 0.92 | 0.88 | 0.85 |
| The whole library | 0.87 | 0.83 | 0.85 | 0.92 | 0.89 | 0.83 |
| The decision from the latents, linear | 0.91 | 0.92 | 0.95 | 0.92 | 0.95 | 0.97 |

- The latents carry the decision at every width (action collapse), and
  the window in force is the previous decision, so the window explains
  the latents whether or not the encoder sees it.
- From Cubic a drop fraction comes first at every width; from BBRv3 the
  flow's arrival rate (with the window held, the RTT), then the base RTT
  and the queueing delay on the ACK path.
- The change a decision makes is mostly unexplained by ground truth of
  one step (0.17 to 0.37), and comes mostly from the decoder's memory:
  the step's latents and window explain more than 0.5 of it in 2 of 9
  runs from Cubic and none from BBRv3 (`exp-probe-memory-34`; the
  quadratic fit of `exp-probe-memory-33` could not be read).
- The window in force at a decision is always the one the controller set
  at its previous decision, in all 18 controllers.
- The two-term formula search adds 0.01 to 0.02 over single variables.

PySR on the three best confirmed networks per teacher, run on the
session's machine and published through the runner
(`exp-concepts-confirm-42`, `exp-concepts-beyond-window-43`):

- The latents are mostly the sender's own window. A network on the
  window's features explains 0.16 to 0.93 of a latent, most above 0.8. With the window as
  an input, PySR recovers 10 of 21 latents from BBRv3 and 4 of 33 from
  Cubic at 0.8; pooled, they are the log of the window and, from BBRv3,
  one mix of the window and the flow's arrival rate.
- Beyond the window, little carries over from the training links to the
  validation links: a network on the window's and the path's features
  explains at most 0.63 of what remains of a latent, and on many latents
  less than a constant. One formula is recovered, from BBRv3: the log of
  the flow's arrival rate over the base RTT (0.27, of a ceiling of 0.32).
  The remainder from Cubic reads loss and delay, which PySR fits at 0.28
  or less against ceilings up to 0.60, per latent or along the
  directions a network explains best.

The same on the three best controllers per teacher with the window
hidden from the encoder (`exp-controller-hidden-window-44`):

- Beyond the window, a network on the window's and the path's features
  explains at most 0.71 of what remains of a latent.
- Recovered from BBRv3: a latent the window does not explain (0.02),
  fitted at 0.59 of a ceiling of 0.71 by the mean queueing delay over
  240 ms and the inverse of the minimum RTT ratio over 960 ms times the
  ACK path's queueing delay; and two weaker forms of the queueing delay
  with the window (0.28 of 0.32, and 0.28 above a ceiling of 0.12).
- Recovered from Cubic: a function of the ACK path's queueing delay over
  10 ms (0.31 of 0.36). The rest reads loss again, at 0.31 or less.

The constants of the eight formulas above, and the same fits without
units and with them (`exp-concepts-units-46`):

- No constant stands for a path variable: capacity, cross traffic, fair
  share and the base RTT in their place fit worse. The 3.98 in
  log(W - 3.98) is the guest's minimum window of 4 packets; the 1.002 ms
  is the ACK path's floor of about 1 ms; the rest are offsets, weights
  between terms, or not identifiable.
- With any smooth function of a formula's value as the readout, as net2
  can apply, dropping every constant costs at most 0.02 of R^2; with a
  linear readout it costs up to 0.27.
- Without constants the formulas read as the sender's window, the mean
  queueing delay over 240 ms, and the flow's arrival rate, alone or over
  the base RTT. The weak form of the queueing delay with the window
  (ceiling 0.12) and the ACK path's delay at its floor are not concepts.
- With every input in SI and the formula and its constants
  dimensionless (`ground.py concepts --units`), PySR recovers no formula.
  The mean share of the ceiling falls from 0.28 to 0.16 and from 0.54 to
  0.37 on BBRv3's controllers (window seen, hidden), and from 0.29 to
  0.21 and stays at 0.27 on Cubic's; two plain runs differ by 0.03 or
  less.

A stress test of the fitter (`exp-ground-stress-23`) plants latents with
a known answer. From the stored labels alone the formula search names
all 12 literature composites without noise and 28 of 36 with noise; a
mix of seven concepts comes back as a sufficient set, not as the planted
basis; nothing is found in noise.

### The seed search

The concepts come from one network that is trained well and whose
latents decompose ([methodology](methodology.md#grounding)). Two batches
have run, five draws at a time, and every draw finished. No draw is a
hit.

- The first batch (`exp-controller-seed-search-47`) trained seeds 101 to
  108 at each teacher's better recipe, widths 7 and 10: 32 draws, 2.5 to
  4.8 h each. Each draw was set against BBRv3 on its own (the batch's
  collect averages the seeds of each teacher and width) and scored by
  `ground.py decompose` on the session's machine at commit `9eddfc365`.
- The second batch (`exp-controller-seed-search-48`) dropped Cubic and
  trained BBRv3's seeds 109 to 124 at widths 7 and 10: 32 draws in about
  21 h. Each cell scored its own draw with the same code, and the collect
  set each draw against BBRv3 on its own.

| Batch | Teacher | Trained well | Recovered along the axes | Recovered after varimax | Most recovered in one network | Decomposed |
|---|---|---|---|---|---|---|
| 1 | BBRv3, window hidden | 1 of 16 | 11 of 108 | 14 of 105 | 2 | 0 of 16 |
| 1 | Cubic, window seen | 0 of 16 | 5 of 100 | 2 of 107 | 1 | 0 of 16 |
| 2 | BBRv3, window hidden | 7 of 32 | 42 of 210 | 38 of 211 | 6 | 1 of 32 |

- Trained well: in the first batch, BBRv3's width 7 seed 104 (+0.07,
  -0.08 to +0.20, with a competing flow). In the second, width 10 seeds
  111, 115, 116 and 122 and width 7 seeds 115, 116 and 121.
- The draws that come closest to a hit:
  - Width 10 seed 122 is trained well (+0.08, -0.06 to +0.22, with a
    competing flow). Along its axes it recovers 6 of 7 directions, with
    3 distinct formulas; the seventh reaches an R^2 of 0.12 against a
    ceiling of 0.27, where 0.22 is needed.
  - Width 7 seed 115 is trained well (+0.08, -0.06 to +0.21). Along its
    axes it recovers both of its 2 directions, so it has 2 formulas, not
    3.
  - Width 7 seed 109 decomposes along its axes: 3 of 3 directions, 3
    distinct formulas, over the flow's queue, the queueing delay, the
    flow's arrival rate and, in one, the window. It is not trained well:
    with a competing flow it is -0.09 (-0.28 to +0.10), unresolved.
- Alone, every draw is within the margin of BBRv3 but Cubic's width 7
  seed 106 and width 10 seed 108, which are unresolved. With a competing
  flow, 4 of Cubic's draws and BBRv3's width 10 seed 109 fall below the
  margin. The draws not named here as trained well are unresolved.
- Over BBRv3's 57 networks with the window hidden (9 earlier, 48 of the
  two batches), 125 of 793 directions are recovered (0.16), with a median
  of 7 to recover per network and basis.

The owner keeps the search as it is. The third batch
(`exp-controller-seed-search-49`, `experiments/controller-seed-search.json`
as it now stands) trains BBRv3's seeds 125 to 140 at widths 7 and 10, in
the same way as the second. UNRUN: the third batch, which is training,
and a release of the first batch's scores.

`ground.py decompose` on the 18 confirmed controllers
(`exp-ground-rl-confirm-32`, the window seen) and the 18 with the window
hidden (`exp-controller-hidden-window-44`), run on the session's machine
at commit `9eddfc365` on 65,000 to 93,000 decisions of 126 validation
episodes per network. Its release is UNRUN: the outputs are too large to
carry inline, so it waits for a rerun in Actions within the search's five
runners.

| Controllers | Directions with a ceiling of 0.1 or more, per network (axes, varimax) | Recovered along the axes | Recovered after varimax | With 3 distinct formulas or more (axes, varimax) |
|---|---|---|---|---|
| BBRv3, window seen | 2.4, 7.4 | 7 of 22 | 11 of 67 | 1, 1 |
| BBRv3, window hidden | 9.0, 8.7 | 12 of 81 | 8 of 78 | 1, 1 |
| Cubic, window seen | 7.8, 8.8 | 3 of 70 | 4 of 79 | 0, 0 |
| Cubic, window hidden | 9.1, 8.6 | 4 of 82 | 1 of 77 | 0, 0 |

- No network decomposes, along either basis.
- BBRv3's networks recover 10% to 32% of their directions, Cubic's 1% to
  5%. Every network with 3 distinct formulas or more is BBRv3's.
- The closest are BBRv3's width 10 seed 1 with the window seen, along its
  axes (4 of 8 directions recovered, 4 distinct formulas), and BBRv3's
  width 16 seed 2 with the window hidden (3 of 9 along its axes, 3 of 11
  after varimax, 3 distinct formulas each).
- The strong directions are recovered less often than the weak: of
  BBRv3's directions with a ceiling of 0.4 or more, 9 of 81; of 0.1 to
  0.2, 17 of 57. With a floor of 0.2 or 0.3 in place of 0.1, still no
  network decomposes.
- Varimax recovers a smaller share than the axes in 3 of the 4 groups.
- PySR's search is not the limit. On BBRv3's width 16 seed 2 with the
  window hidden, along its axes, five times the iterations (300) recovers
  the same 3 directions; the others gain 0.00 to 0.12 and stay below
  0.8 of the ceiling.
  Nor are PySR's inputs: a network on the window in force and the path
  explains each direction about as well as the ceiling, which also sees
  the window's history. The unrecovered directions are weak (most have a
  ceiling of 0.12 to 0.20) and have no formula of 20 nodes or fewer.

The decoder fed rebuilt latents is no test. On BBRv3's width 10 seed 3
with the window hidden, the decision's change has a standard deviation
of 0.129. Latents rebuilt by the window's network give an error in u of
0.191 (R^2 of the change -1.21), and by a network on the whole library
0.253 (-2.90). The decoder on its own latents gives 1.0, and against the
recorded decisions 0.97, so the code is right: the frozen decoder turns
the rebuilt latents' small errors into errors larger than the change.
Over the 36 networks, no rebuild of BBRv3's scores above 0, and none
with the window seen above 0.001. In Cubic's with the window hidden, 25
of 54 score above 0, at most 0.41, and the window's network alone does
as well as the formulas.

## Releases

| Release | What it holds |
|---|---|
| `exp-estimator-seven-20` | The estimator of the seven LCFs, as copied and converted to the current spec. Input of the bootstrap. |
| `exp-steps-behaviour-rule-8` | The fixed rule as an ONNX graph. Input of the bootstrap. |
| `exp-controller-clone-21` | The start taught from the fixed rule. |
| `exp-controller-teachers-22` | The bootstrap from Cubic and BBRv3. |
| `exp-ground-stress-23` | The stress test of the fitter. |
| `exp-controller-rl-24` | Reinforcement learning at one untuned point. |
| `exp-ground-teachers-26`, `exp-ground-rl-27` | Grounding without the window held; superseded. |
| `exp-controller-rl-search-29` | The settings search at width 10. |
| `exp-ground-rl-30` | Grounding of `exp-controller-rl-24` with the window held. |
| `exp-controller-rl-confirm-31` | The confirmation of the settings. |
| `exp-ground-rl-confirm-32` | Grounding of the confirmed controllers, the references' rollouts and the comparison with them. |
| `exp-probe-memory-33`, `exp-probe-memory-34` | The decision's change from the step's latents and window. |
| `exp-teachers-val-35` | Cubic and BBRv3 on the grounding's validation episodes. |
| `exp-versus-teachers-36` | Each controller against its teacher. |
| `exp-controller-rl-scheduled-39` | The confirmed controllers trained 240 iterations more with a falling rate, keeping the best policy. |
| `exp-versus-bbr3-41` | Every controller, the fixed rule and the teachers against BBRv3. |
| `exp-concepts-confirm-42` | PySR on the best confirmed controllers' latents, the window an input. |
| `exp-concepts-beyond-window-43` | PySR on what those latents hold beyond the window. |
| `exp-controller-hidden-window-44` | The controllers trained with the window hidden from the encoder, against BBRv3, their grounding and PySR beyond the window. |
| `exp-controller-rl-cubic-slow-45` | Cubic's controllers continued from their best policies at a tenth of the rate. |
| `exp-concepts-units-46` | The concepts fitted with and without units, and the test of their constants. |
| `exp-controller-seed-search-47` | The first batch of the seed search: seeds 101 to 108, both teachers, widths 7 and 10. Its `versus-bbr3` averages the seeds of each teacher and width. |
| `exp-controller-seed-search-48` | The second batch: BBRv3's seeds 109 to 124, widths 7 and 10. Each draw's `decompose-*` scores and its own row in `versus-bbr3`. |
| `exp-controller-seed-search-49` | The third batch: BBRv3's seeds 125 to 140, widths 7 and 10, as the second. Running. |
| `exp-steps-broad-30s-9`, `exp-phase0-teachers-1`, `exp-phase0-steps-2` | Old guest; not used. |

## A fault in earlier episodes

The guest let a stopped TCP cross flow keep sending and could start a
cross flow late, so the active-flow count was not always right. Commit
`a26b711a1` fixes it (result version 6). Releases made before it carry
the fault in episodes with cross flows.

## Withdrawn work

- Predictors fitted to logged steps (`exp-codec-*`, `exp-student-*`,
  `exp-ground-probe-10`, `exp-ground-select-12`, `exp-ground-knee-13`,
  `exp-actions-knee-14`): none is a controller. Their code is removed;
  the split by link and the selection method remain.
- Counting concepts by the width of the latents: one latent can carry the
  decision itself, so the count is read in retraining.
- Claims, failure predictions and cross-seed checks: dropped by the
  owner. One set of concepts that works is enough.
- Pooling formulas over networks: dropped by the owner. A set of
  concepts is not expected to recur across seeds; it comes from one
  network.
- The frozen decoder as the test of a hit: the first batch's file asked
  for R^2 of 0.8 on the decisions' change from latents rebuilt from the
  formulas. It was dropped before any draw was scored, since no rebuild
  comes near it: the best is 0.41 ([above](#the-seed-search)).

## Open questions

| Question | Note |
|---|---|
| Which path orders the concepts in retraining | The decision's change comes mostly from the decoder's memory, while the latents are explained to 0.80 to 0.86. The latents' path, or ground truth over a longer history, are the candidates. |
| Labels for fitted formulas | A formula over stored labels becomes a composite with no new data ([architecture.md](architecture.md#composites)). One with a windowed operator over past steps has no such path yet. |
| The kind of output the retrained net1 states | A value, an interval or a distribution. |

## Checks

ruff and pyright pass. The last full `uv run pytest -q` passed (231
passed, 61 skipped); later changes ran the tests they touch. CI runs the
tests that boot the guest.
