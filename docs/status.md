# Status

What has run, what it showed and what is open. Rewritten in place; the
history is in git. The steps are those of the
[plan](methodology.md#plan).

## Where the project stands

Against BBRv3, Cubic's width 10 is trained well over three seeds, as are
both references without an encoder and the fixed rule; the other widths
fall short with a competing flow ([below](#against-bbrv3)).

Step 3 is open. The latents of the best confirmed controllers are mostly
the sender's own window, which the encoder reads from the packets in
flight, with little of the path beyond it ([grounding](#grounding)).
Whether path concepts can be recovered at all is decided by the same
controllers trained with those columns hidden from the encoder
(`experiments/controller-hidden-window.json`), running, UNRUN. Longer
training with a falling rate did not pay: the best policies come early
and most runs end below their start ([below](#longer-training)). Cubic's
continuation at BBRv3's lower rate is running, UNRUN.

| Step | State |
|---|---|
| 1 Setup | done: the estimator of the seven LCFs and the rule graph as releases |
| 2 Train | bootstrapped from Cubic and BBRv3; reinforcement learning tuned at width 10 and confirmed at widths 7, 10 and 16 over three seeds; against BBRv3 Cubic's width 10 is trained well; longer training did not pay; a slower continuation for Cubic is running |
| 3 Ground | open: the best confirmed controllers' latents are mostly the window; the controllers with the window hidden from the encoder are training |
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
- Cubic's continuation from these best policies at BBRv3's rate
  (0.00003 falling to 0.000003, 240 iterations,
  `experiments/controller-rl-cubic-slow.json`) is running, UNRUN.

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

- The latents are mostly the sender's own window, which the encoder
  reads from the packets in flight. A network on the window's features
  explains 0.16 to 0.93 of a latent, most above 0.8. With the window as
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

A stress test of the fitter (`exp-ground-stress-23`) plants latents with
a known answer. From the stored labels alone the formula search names
all 12 literature composites without noise and 28 of 36 with noise; a
mix of seven concepts comes back as a sufficient set, not as the planted
basis; nothing is found in noise.

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
