# Log

What has run, what it showed, and what is open. Everything that changes
with time lives here; the other documents state only what stays true.
Entries are dated, newest first, and an entry is not edited after it is
superseded: a later entry says what changed. Each entry names its
release, what was searched, the held-out numbers, and what remains UNRUN.

## State of the plan

Updated with each entry. The steps are those of the
[plan](methodology.md#plan).

| Step | State | Can reuse | Not built |
|---|---|---|---|
| 1 Setup | done (2026-11-11) | | |
| 2 The codec on logged steps | done (2026-11-12): search `exp-codec-search-3`, confirmation `exp-codec-confirm-4` | `experiments/codec.py`, its positive control (`tests/test_codec.py`), the link split; the trained weights in `exp-codec-confirm-4` | |
| 3 Strengthen the codec | dropout 0.3 chosen and the width curve rerun with it (2026-11-12, `exp-codec-dropout-6`); knee rule fixed, 6 latents; longer training and a deeper encoder tried and not adopted (2026-11-13, `exp-codec-capacity-7`) | the `dropout` option of the codec; the trained weights in `exp-codec-dropout-6` | the information-bottleneck term; the run on the test links |
| 4 Grounding | linear grounding done on the knee codecs (2026-11-13, `exp-ground-knee-13`), after a first look (`exp-ground-probe-10`, `exp-ground-select-12`); the latents mostly repeat three sender readings, so the formula search waits for step 5 | the labels and the step message in `steps.parquet`; `experiments/ground.py probe` and `select`; the trained weights in `exp-codec-knee-11` | formulas beyond a linear fit (PySR on the residual); the operator-family sensitivity |
| 5 The controller in the closed loop | the controller with a narrow middle is built and unit-tested (2026-11-13); its first closed-loop stage (`experiments/controller-clone.json`) is UNRUN. Done earlier on logged steps, as proxies: the offline action check (`exp-actions-knee-14`) and a student of the logged actions (`exp-student-search-16`, `exp-student-confirm-17`, grounded in `exp-student-ground-18`, not yet read) | `PPOSpec.latents` (`train/recurrent.py`); `train ppo` by DAgger, PPO or TD3+BC; closed-loop rollouts and scoring; `experiments/net2_ppo_setup.py` and `net2_ppo_report.py` | reinforcement learning after the taught start; the width curve in the closed loop; grounding on a controller's own rollouts; a start that does not come from the rule |
| 6 Retrain and compare | UNRUN | `train run` for net1 and an offline net2; `train ppo` for a closed-loop net2; oracles on the seven LCFs | labels from fitted formulas; net1 on them; the live truth for them |

C1 is now stated on the controller with a narrow middle (entry of
2026-11-13, "the discovery model is a controller") and has no evidence
yet: the width curves so far are of the codec and of a student of logged
actions. C2 is not met by linear fits on the codec's latents (entries of
2026-11-13). C4 has its first check for base variables. C3 and C5 have
none.

## Open items

- **Targets for a value head** of a distilled student (step 5). The
  teacher graphs are policies and come without a critic, and a rule has
  none at all. Q-values on a grid of actions need a critic fitted to
  each teacher first.
- **The scale of retained performance** ([methodology.md](methodology.md#retained-performance)).
- **The threshold for C3.**
- **The links, conditions and episode lengths** of the closed-loop
  evaluation, to be fixed in step 5's experiment file.
- **The kind of output** the retrained net1 states (value, interval or
  distribution).
- **Live truth for new concepts.** The guest gives a policy the true
  values of the seven LCFs only, and labels are computed when an episode
  is processed ([architecture.md](architecture.md#known-limits)). The
  oracle net2 of step 6 and labels for fitted formulas need both
  extended.

## 2026-11-13: the guest let a stopped cross flow keep sending

Not a result: a fault in the emulator, found before the first
closed-loop run and fixed first.

- **The fault.** A TCP cross flow was told to stop at its planned time,
  but the bytes its sender had already queued kept going out, and a
  connection whose handshake waited behind a queue started later than
  planned. So the number of flows that the labels and the scores call
  active was not the number really sending. The fix (commit
  "fix(guest): keep a cross flow inside its planned window") closes both
  ends at the stop, refuses a connection that arrives after it, and
  reports when each cross flow really connected
  (`cross_connected_ms`); the result version is now 6. The commit
  message has the measurements that found it.
- **What it touches here.** Every episode run before this commit:
  `exp-steps-broad-30s-9` and `exp-phase0-steps-2`, and so every codec,
  student and grounding release above, in the episodes that have cross
  flows. Those releases are proxies on logged steps since the entry
  below; none of them is rerun for this alone. Any of their numbers that
  is used again is used with this caveat, and data for a claim is
  generated again with the fixed guest.
- **The start of the closed-loop run.** The seven-LCF estimator and the
  rule built on it were fitted on episodes of the old guest. They serve
  only as the teacher of the taught start (`experiments/controller-clone.json`);
  every rollout and every score of that run comes from the fixed guest,
  the rule included, so the comparison with the rule is fair, but the
  rule may be weaker than one fitted on clean episodes.
- **Checked.** Format, lint, types, the unit tests (213 passed, 58
  skipped) and the guest's clippy. UNRUN here: the tests that boot the
  guest; the experiment workflow runs the guest itself.

## 2026-11-13: the plan changed: the discovery model is a controller

- **What changed.** Discovery (step 1 of the idea, step 5 of the plan)
  is now a controller of expliCCit's shape with the middle left free: an
  encoder in net1's place to k latents, a recurrent decoder in net2's
  place to the window, trained in the closed loop
  ([methodology.md](methodology.md#the-controller)). The codec stays as
  a second route. C1 and C5 are restated on the controller.
- **Why.** The goal is the concepts a trained controller uses, judged by
  closed-loop performance. Every run so far was offline: the codec
  predicts the flow's future and has no controller in it, and the
  student imitates 22 logging controllers, 17 of them kernel heuristics,
  none of which is the controller whose performance is to be retained.
  The entry of 2026-11-11 that took the teacher out of discovery gave
  true reasons (action collapse, the cost of a good teacher) and the
  wrong conclusion: it removed the controller from the question. The
  project's owner pointed out that the encoder and decoder asked for
  were net1 and net2 themselves.
- **What the earlier results still say.** They are about logged steps.
  The student's curve with no sharp knee (entry below) describes
  imitation of the logging controllers from one step, scored by squared
  error, and says nothing yet about a controller's need for width. The
  grounding tools, the link split and the offline baselines carry over.
- **Local diagnostics, not a release** (200,000 validation anchors,
  seed 0 students of `exp-student-confirm-17`; a script outside the
  repository, so nothing here is citable). They are why the student is
  not pursued further: at 60 ms, 5% of the anchors hold 83% of the
  target's variance, so the score is mostly about rare large window
  changes; the widest student explains 0.54 of the variance and BBR's
  actions stay near 0.3 at any width; controllers differ in the width
  they need (Reno and Cubic level off at 2 latents, BBR gains at the
  third).
- **Built.** `PPOSpec.latents` and `encoder_hidden`: the encoder (two
  hidden layers) inside `RecurrentActor`, and `graph.FixedWindow`, a
  fixed 2^7 packets as the window's base so that no measured
  bandwidth-delay product passes around the latents. DAgger's labels,
  the learners, the exported graph and the handoff between stages use
  the existing code paths. Unit tests (`tests/test_ppo.py`): the window
  depends on the measurements only through the latents, one exported
  step equals the trained sequence, the label is the rule's window over
  the fixed base. Ran: ruff, pyright, `uv run pytest -q` (212 passed,
  57 skipped). The UML tests and the exported graph in the guest are
  UNRUN here; the first experiment exercises them.
- **Decisions, made before any run.**
  - Widths 4, 7, 10, 16 and 64. 7 is expliCCit's budget, 10 the owner's
    first proposal, 64 stands for no constraint and is the unconstrained
    teacher. No count is read from logged steps.
  - The first stage teaches each controller a fixed rule's window by
    DAgger, as a start only. The rule reads the estimator of the seven
    hand-picked LCFs, so the start leans toward capacity and delay. The
    claim will rest on the reinforcement-learning stage, and a run from
    another start is the check (C5).
  - A reference without an encoder (k 0 in the experiment file): the
    recurrent network on the measurements with its window on a measured
    bandwidth-delay product.
- **Inputs.** `experiments/estimator-seven.json` copies the estimator of
  the seven hand-picked LCFs into a release of this repository; the
  rule graph of `exp-steps-behaviour-rule-8` supplies the training
  paths' logging controller.
- **UNRUN.** Everything in the closed loop: the taught start
  (`experiments/controller-clone.json`), reinforcement learning, the
  width curve, grounding on a controller's rollouts, the start that does
  not come from the rule. Reading `exp-student-ground-18`.

## 2026-11-13: the student of the actions over seeds, and a curve with no sharp knee

- **Release.**
  [`exp-student-confirm-17`](https://github.com/Radonux/compact/releases/tag/exp-student-confirm-17)
  (`experiments/student-confirm.json`). The student of the entry below
  with the settings chosen there (dropout 0.3, learning rate 3e-4), seeds
  0 to 4 at 2, 3, 4, 6, 8 and 64 latents. All 30 cells succeeded and ran
  their 10 epochs; the validation loss was best in the last epoch in
  every run, and no latent collapsed (standard deviations from 0.99).
  The test links were not read.
- **Searched.** Nothing new; this confirms the choice of
  `exp-student-search-16`.
- **Validation R^2 and skill** (mean and standard deviation over the 5
  seeds; R^2 over all validation anchors in the scaled units, mean of
  the three horizons; skill against a window that does not change):

  | Latents | R^2 | 60 ms | 240 ms | 960 ms | Skill | Share of the skill at 64 | Gain in R^2 over the row above |
  |---|---|---|---|---|---|---|---|
  | 2 | 0.5101 ± 0.0010 | 0.527 | 0.547 | 0.456 | 0.5193 | 94.0% | |
  | 3 | 0.5262 ± 0.0013 | 0.546 | 0.562 | 0.470 | 0.5351 | 96.8% | 0.0160 ± 0.0020 |
  | 4 | 0.5324 ± 0.0007 | 0.552 | 0.569 | 0.476 | 0.5412 | 98.0% | 0.0063 ± 0.0015 |
  | 6 | 0.5399 ± 0.0009 | 0.562 | 0.576 | 0.482 | 0.5486 | 99.3% | 0.0075 ± 0.0010 |
  | 8 | 0.5426 ± 0.0005 | 0.566 | 0.578 | 0.484 | 0.5512 | 99.8% | 0.0027 ± 0.0010 |
  | 64 | 0.5439 ± 0.0003 | 0.573 | 0.580 | 0.479 | 0.5525 | 100% | 0.0013 ± 0.0003 |

  The gains are differences within a seed. One latent was not rerun; on
  one seed it had 83% of the skill (entry below).
- **What it shows.**
  1. Seeds agree to about 0.001 in R^2, so the curve of the search was
     not noise. Every step of the table is positive in every seed, down
     to the 0.0013 from 8 to 64 latents.
  2. By the rule fixed for the codec (the smallest width within 5% of
     the skill at 64 latents) the knee is 3 in every seed: 3 latents
     hold 96.6% to 97.0%, 2 hold 93.8% to 94.2%.
  3. The curve has no sharp knee, so the count depends on the
     threshold and the rule decides little. Within 5%: 3 latents.
     Within 2%: 6 (4 latents reach 97.96% on average and miss it by a
     hair). Within 1%: 6. Within 0.5%: 8. The codec's curve was steeper
     at the same widths (6 latents held about 95% of its skill and 8 about
     97%), so the actions do need less width than the sender's future
     at any of these thresholds.
  4. What a threshold on this score means for control is not known.
     The score is offline, and 1% of skill here may or may not be 1% of
     anything in the closed loop. That is the reason to carry several
     widths (3, 4, 6 and 8) through grounding and into the closed loop,
     and not to fix the count here.
- **It is offline.** One step of input, logged actions, no claim about
  control.
- **Next.** `experiments/student-ground.json` grounds the students of 3,
  4, 6 and 8 latents with `experiments/ground.py` (`probe` and
  `select`, with the vocabulary sensitivity check). Then
  `experiments/actions.py` puts the students' latents and the selected
  features on the per-controller scale of `exp-actions-knee-14`, which
  scores the selection on the action.
- **UNRUN.** The grounding and its sensitivity check; the per-controller
  scores and action collapse per controller; one latent over seeds; more
  than 10 epochs; a wider or deeper student; a student with history; the
  closed loop; anything on the test links.

## 2026-11-13: a student of the logged actions, its search and a first width curve

- **Release.**
  [`exp-student-search-16`](https://github.com/Radonux/compact/releases/tag/exp-student-search-16)
  (`experiments/student-search.json`; `experiments/codec.py` with
  `targets=action`). The codec's network with a different loss: the
  encoder reads one step's message and gives k latents; the decoder
  reads the latents and the identity of the controller that logged the
  step, and predicts the change of that controller's window (log2,
  clamped to 8 either way) over the next 60, 240 and 960 ms. Each
  controller's changes are scaled to unit variance on the training
  links, so every controller weighs the same per anchor. Steps of
  `exp-steps-broad-30s-9`: 4,078,781 training anchors and 768,684
  validation anchors, links split 70/15/15 with split seed 0. All 21
  cells succeeded, each ran its 10 epochs, and no latent collapsed
  (standard deviations 0.95 to 1.10). The test links were not read.
- **An earlier run did not publish.** Run 15 of the workflow was
  cancelled: its cell with one latent crashed after training, where the
  report correlates the latents with each other (one latent gives a
  scalar, not a matrix). Fixed with a unit test, and the whole search
  was run again as this release. Nothing from run 15 is used.
- **Searched.** One seed (0). Dropout 0, 0.1 and 0.3 with learning rate
  3e-4 and 1e-3, and dropout 0.5 with learning rate 1e-3, at 6 and at 64
  latents. Fixed and not searched: width 512, encoder depth 4, decoder
  width 512, weight decay 1e-4, 10 epochs with a cosine schedule, batch
  2048, one step of input. Validation R^2 (mean of the three horizons,
  over all validation anchors in the scaled units):

  | Dropout | Learning rate | 6 latents | 64 latents | Best epoch (6 / 64) |
  |---|---|---|---|---|
  | 0 | 3e-4 | 0.5276 | 0.5287 | 7 / 5 |
  | 0 | 1e-3 | 0.5202 | 0.5288 | 7 / 6 |
  | 0.1 | 3e-4 | 0.5378 | 0.5364 | 10 / 9 |
  | 0.1 | 1e-3 | 0.5369 | 0.5346 | 10 / 10 |
  | 0.3 | 3e-4 | **0.5401** | **0.5441** | 10 / 10 |
  | 0.3 | 1e-3 | 0.5373 | 0.5413 | 10 / 10 |
  | 0.5 | 1e-3 | 0.5311 | 0.5376 | 10 / 9 |

  Without dropout the student overfits at both widths (validation loss
  is best at epoch 5 to 7 and rises after). Dropout 0.3 is best at both
  widths, 0.1 and 0.5 are both worse, so the search brackets it. The
  lower learning rate is ahead by 0.003 at both widths, which one seed
  cannot tell from noise. Chosen for the confirmation: dropout 0.3 and
  learning rate 3e-4. At dropout 0.3 the validation loss was still
  falling in the last epoch, by 0.0001 to 0.0004 under a learning rate
  that the schedule had taken near zero.
- **Width curve** (dropout 0.3, learning rate 1e-3, seed 0; skill is
  against predicting that the window does not change, whose R^2 is
  -0.019):

  | Latents | R^2 | 60 ms | 240 ms | 960 ms | Skill | Share of the skill at 64 |
  |---|---|---|---|---|---|---|
  | 1 | 0.4462 | 0.475 | 0.489 | 0.375 | 0.4567 | 83.1% |
  | 2 | 0.5062 | 0.523 | 0.543 | 0.453 | 0.5154 | 93.7% |
  | 3 | 0.5241 | 0.544 | 0.560 | 0.469 | 0.5330 | 96.9% |
  | 4 | 0.5303 | 0.551 | 0.568 | 0.472 | 0.5391 | 98.0% |
  | 6 | 0.5373 | 0.559 | 0.575 | 0.478 | 0.5460 | 99.3% |
  | 8 | 0.5408 | 0.564 | 0.576 | 0.482 | 0.5494 | 99.9% |
  | 12 | 0.5415 | 0.568 | 0.577 | 0.480 | 0.5501 | 100.0% |
  | 16 | 0.5408 | 0.568 | 0.575 | 0.479 | 0.5495 | 99.9% |
  | 64 | 0.5413 | 0.569 | 0.578 | 0.477 | 0.5499 | 100% |

- **What it shows, provisionally.**
  1. The actions need fewer latents than the sender's future. By the
     rule fixed for the codec (the smallest width within 5% of the
     skill at 64 latents) the knee is 3, against the codec's 6. One
     latent already holds 83% of the skill and two hold 94%.
  2. The curve is flat from 8 latents: 8, 12, 16 and 64 agree within
     0.001. Between 3 and 8 the R^2 rises by 0.017, in steps of 0.006,
     0.007 and 0.004. Whether those steps are real is for the seeds to
     say; the differences between settings at one width (0.003) give a
     first idea of the noise.
  3. The rule is lenient here, because the floor is high: with one
     latent the decoder still knows which controller it predicts, and
     much of a window's change over one step may follow from few
     quantities. A count of 3 by the rule and a count of 6 to 8 by where
     the curve stops rising are both defensible readings of this one
     curve, and this entry does not choose between them.
  4. These scores are not on the scale of the offline action check
     (`exp-actions-knee-14`). That check averages R^2 per controller
     and fits a small network on 300,000 anchors; this score pools all
     validation anchors and the student trains on 4 million. The
     student's latents are put on the check's scale by running
     `experiments/actions.py` on them, which is UNRUN.
- **It is offline.** The student sees one step and the controllers have
  memory; predicting logged actions is not control, and this entry
  makes no claim about control. The count is of what one step's message
  holds about the logged actions.
- **Next.** `experiments/student-confirm.json`: dropout 0.3 and learning
  rate 3e-4 over seeds 0 to 4 at 2, 3, 4, 6, 8 and 64 latents. Then the
  students' latents go through `experiments/actions.py` and
  `experiments/ground.py` (`probe` and `select`), with the selection
  scored on the action.
- **UNRUN.** The confirmation over seeds; the grounding of the student's
  latents and its vocabulary sensitivity check; the student's latents
  on the scale of the action check; action collapse per controller (z
  against each controller's action); more than 10 epochs; a wider or
  deeper student; a student with history; the closed loop; anything on
  the test links.

## 2026-11-13: an offline check of what the logging controllers act on

- **Release.**
  [`exp-actions-knee-14`](https://github.com/Radonux/compact/releases/tag/exp-actions-knee-14)
  (`experiments/actions-knee.json`, `experiments/actions.py`). One small
  network per set of inputs predicts, from the inputs at one step and
  the identity of the controller that logged it, the change of that
  controller's window (log2) over the next 60, 240 and 960 ms. Steps and
  labels of `exp-steps-broad-30s-9`, codecs of `exp-codec-knee-11`,
  ground-truth paths of `exp-ground-knee-13`. Fits on 300,000 anchors of
  the training links, scores on 200,000 anchors of the validation links,
  as R^2 per controller around its own mean change, then averaged over
  the 22 controllers (17 kernel algorithms and 5 learned policies). No
  cell failed. The test links were not read.
- **Searched.** Nothing. One network (2 hidden layers of 256, 20 epochs,
  the epoch chosen on a tenth of the training episodes) serves every
  set, with 3 model seeds per set and one fit per codec run (5 seeds).
  No set was tuned, so differences of a few hundredths between sets are
  not a ranking.
- **It is a proxy.** The controllers have memory and the predictor sees
  one step; the whole message at one step reaches only 0.50. Predicting
  logged actions is not control, and this entry makes no claim about
  control.
- **Mean over controllers** (mean and standard deviation over the fits;
  "columns" counts the inputs, where a feature that is sometimes
  undefined has a second column saying so):

  | Inputs | Columns | 60 ms | 240 ms | 960 ms |
  |---|---|---|---|---|
  | controller's identity (floor) | 0 | 0.000 | 0.000 | 0.000 |
  | sender's srtt, delivery rate, loss rate | 3 | 0.225 ± 0.003 | 0.200 ± 0.004 | 0.147 ± 0.001 |
  | 4 latents | 4 | 0.314 ± 0.004 | 0.312 ± 0.003 | 0.234 ± 0.003 |
  | 6 latents (the knee) | 6 | 0.383 ± 0.005 | 0.404 ± 0.004 | 0.314 ± 0.003 |
  | 8 latents | 8 | 0.417 ± 0.004 | 0.441 ± 0.005 | 0.353 ± 0.005 |
  | first 4 ground-truth picks | 6 | 0.295 ± 0.002 | 0.287 ± 0.001 | 0.229 ± 0.003 |
  | first 6 ground-truth picks | 9 | 0.304 ± 0.005 | 0.305 ± 0.004 | 0.252 ± 0.004 |
  | first 8 ground-truth picks | 12 | 0.389 ± 0.006 | 0.406 ± 0.003 | 0.317 ± 0.005 |
  | expliCCit's seven concepts | 7 | 0.407 ± 0.004 | 0.425 ± 0.003 | 0.326 ± 0.004 |
  | whole step message (ceiling for one step) | 70 | 0.501 ± 0.007 | 0.488 ± 0.003 | 0.395 ± 0.005 |

  With the sender's own window added as one more column: seven concepts
  0.479 / 0.510 / 0.408, first 8 picks 0.473 / 0.504 / 0.403, 8 latents
  0.437 / 0.464 / 0.374, 6 latents 0.410 / 0.430 / 0.336, sender
  readings 0.354 / 0.331 / 0.244. The seven concepts with the window
  match the whole message.
- **Kernel algorithms and learned policies apart** (60 ms; means of the
  per-controller table of the release, computed here):

  | Inputs | 17 kernel algorithms | 5 learned policies |
  |---|---|---|
  | sender readings | 0.231 | 0.204 |
  | 6 latents | 0.381 | 0.390 |
  | 8 latents | 0.405 | 0.456 |
  | first 6 ground-truth picks | 0.252 | 0.484 |
  | first 8 ground-truth picks | 0.353 | 0.513 |
  | seven concepts | 0.359 | 0.571 |
  | whole message | 0.462 | 0.633 |

- **What it shows.**
  1. The latents hold more of the actions than the three sender
     readings that span 85% of them (0.383 against 0.225 at the knee).
     The part of the latents that the linear grounding left unnamed is
     not noise for this purpose.
  2. Grounding by how well features span the latents picks the wrong
     features for actions. The first 6 picks score 0.304, below the 6
     latents they were picked to span (0.383) and below the seven
     hand-picked concepts (0.407). The seventh and eighth picks,
     `change_60ms(queue_delay_mean_ns)` and `capacity_bps`, each add
     only 0.002 to the fit of the latents and together add 0.085 here.
     Explained variance of the latents is therefore a poor order of
     importance for control.
  3. At the same number of inputs the codec's latents and the seven
     hand-picked concepts are close on average (6 latents 0.383, seven
     concepts 0.407, 8 latents 0.417), and they differ by controller.
     For the kernel algorithms the latents are at least as good as the
     seven concepts. For the five learned policies the seven concepts
     are far ahead (0.571 against 0.390). What those policies read is
     not recorded in compact's documents, so the gap may only say that
     they were given such quantities.
  4. The local dry run on the older codecs gave the same picture (seven
     concepts 0.41, 6 latents 0.39, first 6 picks 0.31, sender 0.23).
- **Consequence for the plan.** The count and the names should come
  from a model whose loss is the action, not the sender's future. The
  next part of step 5 is a student of the logged actions with k latents
  between its encoder and its action head: its width curve on this same
  score gives the count for actions (to set beside the codec's 6), and
  its latents are grounded with `ground.py` as the codec's were. The
  selection should then be scored on the action and not on the fit of
  the latents. The formula search still waits.
- **UNRUN.** The bottlenecked student of the actions; a predictor with
  history; any tuning per set; what the five learned policies read; the
  closed loop; anything on the test links.

## 2026-11-13: grounding the knee codecs, and what it changes

- **Release.**
  [`exp-ground-knee-13`](https://github.com/Radonux/compact/releases/tag/exp-ground-knee-13)
  (`experiments/ground-knee.json`): `ground.py probe` and `select` over
  the codecs of `exp-codec-knee-11` (4, 6 and 8 latents, seeds 0 to 4),
  with the messages and labels they were trained on
  (`exp-steps-broad-30s-9`). Fits on 300,000 anchors of the training
  links, scores on 200,000 anchors of the validation links, all linear
  R^2 unless said. Nothing was tuned; the library is that of the first
  look. No cell failed. The test links were not read. The probe's table
  still carries an empty `buffer_bytes` row.
- **It repeats the first look**, to within 0.01 in every number both
  hold:

  | | 4 latents | 6 | 8 |
  |---|---|---|---|
  | One seed's latents from another's (quadratic) | 0.993 (0.996) | 0.981 (0.988) | 0.964 (0.980) |
  | The sender's srtt, delivery rate and loss rate | 0.925 | 0.855 | 0.792 |
  | First 8 picks, whole library | 0.947 | 0.894 | 0.856 |
  | Whole library | 0.959 | 0.929 | 0.902 |
  | Ground truth only: first 4 picks | 0.875 | 0.830 | 0.788 |
  | Ground truth only: all of it | 0.913 | 0.886 | 0.855 |
  | expliCCit's seven concepts | 0.703 | 0.691 | 0.637 |

  - The three sender readings are the first three picks of all 15 runs.
  - The ground-truth path is the same at the three widths for its first
    five picks: `receive_rate_bps`, `base_rtt_ns`,
    `mean_240ms(queue_delay_ns)`, `max_960ms(drop_fraction)`,
    `buffer_pkts`.
  - Read back from 6 latents: receive rate 0.977, the flow's arrival
    rate 0.880, base RTT, queue delay and total arrival rate 0.70,
    capacity 0.680, drop fraction 0.631, buffer size 0.593, busy
    fraction 0.197, cross-traffic share 0.126, random loss rate 0.072,
    active flows 0.043.
- **Stability over five seeds** (6 latents, each seed's first 8 picks).
  In all five: the three sender readings, `base_rtt_ns` and
  `max_240ms(drop_fraction)`. In three: `change_60ms(sender_cwnd)`. In
  two each: `capacity_bps`, `queue_delay_ns`, `sender_cwnd`,
  `min_960ms(sender_cwnd)` and a window of `arrival_rate_bps`. One
  seed's 8 picks fitted to another seed's latents give 0.893 on average
  (lowest 0.885), the same as on its own (0.888 to 0.902), so the later
  picks stand in for each other.
- **Vocabulary sensitivity** (6 latents, first 8 picks, 0.894 with
  nothing taken out). Without `sender_srtt`: 0.875, with `base_rtt_ns`
  and `mean_240ms(queue_delay_ns)` in its place. Without
  `sender_loss_rate`: 0.884, with two windows of `drop_fraction`.
  Without `sender_delivery_rate`: 0.893, with `receive_rate_bps`. Any
  other picked variable: within 0.005, with the three sender readings
  still first and the later picks reshuffled. This is the first check
  for C4, for base variables: the fit and the leading picks change only
  when a leading pick is removed, and the later picks are not stable
  under removal, as they are not between seeds. Operator families are
  UNRUN.
- **Against the claims.**
  - C1: the knee is at 6 latents on both copies of the data. Its
    closed-loop half is UNRUN.
  - C2 is not met at the knee by linear fits. Seeds agree on the space
    and on five features, but the joint fit is 0.894 at 8 features and
    0.929 with the whole library, and ground truth alone stops at 0.886,
    below the 0.9 the claim asks for. It is met at 4 latents on the
    whole library (0.947 at 8 features) and by ground truth alone only
    with all of it (0.913). Fits beyond linear are UNRUN.
- **What it shows.** The codec's concepts are what the sender already
  measures: the three sender readings, or on the path's side the flow's
  rate, the base RTT, the queue delay and the drop fraction, then the
  buffer size and the capacity weakly. Sharing (flows, cross traffic)
  and random loss are absent. The encoder reads one message, so it can
  only hold what one message determines, and its task is to predict
  those same readings one second ahead; that this needs little beyond
  the readings is the plain reading of the result. It says the task of
  self-prediction is easy to summarize. It does not say a controller
  needs no more.
- **The plan changes** ([methodology.md](methodology.md#plan)). The
  formula search of step 4 (PySR on the residual) is put after step 5.
  Reasons: the linear picks are base variables under a window, so there
  is no formula to find for most of the latents; the 7% the library
  leaves at the knee may well be message fields the library does not
  hold (it has 4 of them), which a formula over path state would not
  name either; and whether these concepts are the ones control needs is
  now the question that decides everything downstream. Step 5 starts
  with what needs no closed loop: how well each logging controller's
  next action is predicted on held-out links from the codec's latents,
  from the picked ground-truth features, from expliCCit's seven
  concepts, and from the raw message, at the same budget. That is a
  proxy and no claim about control; the closed-loop comparison follows
  it.
- **UNRUN.** The offline action check and the closed-loop comparison of
  step 5; formulas beyond a linear fit; the operator-family sensitivity;
  a codec with message history on matched anchors; the message's own
  fields as library variables; anything on the test links.

## 2026-11-13: the knee codecs on the new copy of the data

- **Release.**
  [`exp-codec-knee-11`](https://github.com/Radonux/compact/releases/tag/exp-codec-knee-11)
  (`experiments/codec-knee.json`), with the trained weights of its 18
  models: fixed widths 4, 6 and 8 over seeds 0 to 4 and 64 latents over
  seeds 0 to 2, dropout 0.3, the settled settings, trained on
  `exp-steps-broad-30s-9`. No cell failed and no run stopped early.
  Nothing was searched.
- **Validation links** (skill over persistence, mean and standard
  deviation over seeds; the first copy is `exp-codec-dropout-6` and
  `exp-codec-variants-5`, 3 seeds):

  | Latents | First copy | New copy | Share of 64 latents, new copy |
  |---|---|---|---|
  | 4 | 0.607 ± 0.001 | 0.604 ± 0.002 | 89.7% |
  | 6 | 0.643 ± 0.001 | 0.641 ± 0.002 | 95.2% |
  | 8 | 0.657 ± 0.001 | 0.655 ± 0.001 | 97.4% |
  | 64 | 0.674 ± 0.001 | 0.673 ± 0.001 | 100% |

- **What it shows.** The two copies of the campaign train the same
  codec: every width is within 0.003 of its first-copy skill. The knee
  rule gives 6 latents again, but narrowly: the mean is 95.2% of the
  64-latent skill against a threshold of 95%, and one of the five seeds
  (0.638) falls below it. The count is therefore 6 with 4 and 8 as its
  sensitivity, as planned, and not a sharp number.
- **Dispatched.** `experiments/ground-knee.json`: the probe and the
  selection of the first look, over these codecs at 4, 6 and 8 latents
  with five seeds each. No result yet.
- **UNRUN.** Grounding these codecs (dispatched); anything on the test
  links.

## 2026-11-13: grounding, a first look

Both runs use the codecs of `exp-codec-dropout-6` (dropout 0.3, seeds 0 to
2), which were trained on the first copy of the step data, with the
messages and labels of `exp-steps-broad-30s-9`. Fits are made on 300,000
anchors of the training links and scored on 200,000 anchors of the
validation links. Every number is an R^2 of a linear fit. The test links
were not read. It is a first look: the same two runs are to be repeated on
the codecs trained on the new data, over five seeds.

- **Releases.**
  [`exp-ground-probe-10`](https://github.com/Radonux/compact/releases/tag/exp-ground-probe-10)
  (`experiments/ground-probe.json`; widths 2, 3, 4, 6, 12 and 16) and
  [`exp-ground-select-12`](https://github.com/Radonux/compact/releases/tag/exp-ground-select-12)
  (`experiments/ground-select.json`; widths 3, 4, 6 and 12). No cell
  failed. The probe ran from the commit before never-defined labels were
  dropped, so its tables carry an empty `buffer_bytes` row.
- **Searched.** Nothing was tuned. The library is the one in
  `experiments/ground.py`: 18 ground-truth labels and 4 sender readings
  on a log scale, each as it is now, its change over 60 ms, and its mean,
  minimum and maximum over 240 ms and 960 ms; 32 features were left out
  as repeats of another (correlation above 0.999), among them
  `one_way_delay_ns` for `base_rtt_ns` and `queue_pkts` for `queue_bytes`.
- **Seeds find the same latent space** (probe; one seed's latents from
  another's, mean over pairs):

  | Latents | 2 | 3 | 4 | 6 | 12 | 16 |
  |---|---|---|---|---|---|---|
  | linear | 0.574 | 0.994 | 0.992 | 0.980 | 0.950 | 0.955 |
  | quadratic | 0.724 | 0.995 | 0.996 | 0.989 | 0.977 | 0.977 |

  From 3 latents on, the seeds differ by little more than a linear map.
  This is the first evidence for C2, and only at the level of the space:
  C2 is about the fitted concepts.
- **What a linear read of the latents gives back** (probe; a selection of
  the labels, the release has all of them):

  | Label | 3 | 4 | 6 | 12 | 16 |
  |---|---|---|---|---|---|
  | `receive_rate_bps` | 0.966 | 0.979 | 0.978 | 0.974 | 0.975 |
  | `flow_arrival_rate_bps` | 0.878 | 0.885 | 0.881 | 0.897 | 0.900 |
  | `arrival_rate_bps` | 0.673 | 0.677 | 0.706 | 0.702 | 0.717 |
  | `base_rtt_ns` | 0.532 | 0.570 | 0.694 | 0.879 | 0.929 |
  | `queue_delay_mean_ns` | 0.557 | 0.583 | 0.694 | 0.871 | 0.909 |
  | `capacity_bps` | 0.597 | 0.611 | 0.683 | 0.742 | 0.771 |
  | `drop_fraction` | 0.695 | 0.630 | 0.637 | 0.726 | 0.739 |
  | `buffer_pkts` | 0.498 | 0.530 | 0.591 | 0.638 | 0.679 |
  | `busy_fraction` | 0.084 | 0.111 | 0.202 | 0.409 | 0.465 |
  | `cross_traffic_share` | 0.116 | 0.121 | 0.125 | 0.202 | 0.284 |
  | `active_flows` | 0.034 | 0.037 | 0.043 | 0.105 | 0.171 |
  | `random_loss_rate` | 0.055 | 0.060 | 0.072 | 0.123 | 0.144 |

  A quadratic read raises most rows by 0.03 to 0.10 at 6 latents and
  leaves the same labels at the bottom. The flow's own rate is there from 3 latents.
  Base RTT and queue delay are told apart only as the layer widens. How
  many flows share the link, how much of the traffic is theirs, and the
  random loss rate are barely there at any width.
- **What spans the latents** (selection; pooled over the 3 seeds of a
  width; the path adds, one at a time, the feature that most raises the
  joint fit of all latents):

  | | 3 latents | 4 | 6 | 12 |
  |---|---|---|---|---|
  | The sender's srtt, delivery rate and loss rate (the first three picks at every width) | 0.944 | 0.926 | 0.854 | 0.708 |
  | First 6 picks, whole library | 0.964 | 0.945 | 0.884 | 0.778 |
  | Whole library | 0.977 | 0.959 | 0.928 | 0.860 |
  | Ground truth only: first 4 picks | 0.913 | 0.874 | 0.829 | 0.728 |
  | Ground truth only: all of it | 0.944 | 0.912 | 0.885 | 0.814 |
  | expliCCit's seven concepts | 0.734 | 0.700 | 0.690 | 0.590 |

  - At 6 latents the whole-library path is sender delivery rate, sender
    srtt, sender loss rate, `base_rtt_ns`, `max_240ms(drop_fraction)`,
    `capacity_bps`. After the three sender readings each pick adds 0.013
    or less.
  - The ground-truth path opens the same way at every width:
    `receive_rate_bps`, `base_rtt_ns`, `mean_240ms(queue_delay_ns)`, then
    a windowed `drop_fraction`. These are the twins of the three sender
    readings, with srtt split into its two parts. At 6 latents
    `buffer_pkts` is fifth and `capacity_bps` tenth, and after the fourth
    pick each adds 0.009 or less.
  - The seven hand-picked concepts span the latents worse than the first
    three picks of the ground-truth path (0.771 at 6 latents). They hold
    no loss, and their values are of one moment.
- **Vocabulary sensitivity** (selection; one base variable taken out, the
  first 8 picks again; 6 latents, 0.893 with nothing taken out). Taking
  out a sender reading costs the most and its twin takes its place:
  without `sender_srtt`, 0.874, with `base_rtt_ns` and
  `mean_240ms(queue_delay_ns)` picked; without `sender_loss_rate`, 0.883,
  with two windows of `drop_fraction`; without `sender_delivery_rate`,
  0.892, with `receive_rate_bps`. Taking out any other variable costs
  0.003 or less. The fit therefore does not hang on one name in the
  library, and the names it returns do: the same latent reads as a sender
  measurement or as a path quantity depending on which the library holds.
- **Stability between seeds** (selection; 6 latents). Each seed alone
  picks the same first six features, in slightly different orders.
  One seed's first 8 features fitted to another seed's latents give 0.893
  on average and 0.887 at the lowest, against 0.887 to 0.900 on its own.
- **What it shows.**
  1. A codec that reads one step's message and predicts the sender's own
     next second mostly passes on three of the message's fields. Its
     ground-truth concepts are the path quantities those fields measure:
     the flow's rate, the base RTT, the queue delay and the drop
     fraction. That list has loss, which expliCCit's seven lack, and has
     capacity, the buffer and the busy fraction only weakly.
  2. This is what the codec's task asks for, so it is weak evidence
     about which concepts a controller needs. Nothing in the task
     rewards holding what the sender has not measured yet (the flows it
     shares with, the capacity before it fills the link). The count at
     the knee counts what in one message helps predict the sender's
     future, and that is not yet shown to be the count of concepts a
     controller uses.
  3. The latents are far from fully named: at the knee a linear fit on
     the whole library leaves 7%, and on ground truth alone 11.5%.
- **Consequence for the plan.** No step changes on a first look on
  codecs from the other copy of the data. Step 5 gains weight: the
  question of point 2 is settled only by a model whose loss is control
  (a bottlenecked student of a teacher's action), grounded the same way
  and compared with this list. If the knee codecs repeat this result,
  step 5 is moved ahead of the formula search of step 4, because a
  formula for a latent that repeats a sender reading names nothing new.
- **UNRUN.** Probe and selection on the codecs trained on
  `exp-steps-broad-30s-9`, over five seeds; formulas beyond a linear fit;
  the operator-family sensitivity; a codec with message history; a model
  bottlenecked on a teacher's action; anything on the test links.

## 2026-11-13: longer training and a deeper encoder do not pay

- **Release.**
  [`exp-codec-capacity-7`](https://github.com/Radonux/compact/releases/tag/exp-codec-capacity-7)
  (`experiments/codec-capacity.json`), 12 models: 13 epochs in place of
  10, and encoder depth 6 in place of 4, each at 6 and 64 latents,
  dropout 0.3, seeds 0 to 2, on the first copy of the step data. No cell
  failed.
- **Validation links** (skill over persistence, mean and standard
  deviation over 3 seeds; the baselines are `drop0.3-fixed6` of
  `exp-codec-dropout-6` and `drop0.3-wide64` of `exp-codec-variants-5`):

  | | 6 latents | 64 latents | Share |
  |---|---|---|---|
  | 10 epochs, depth 4 (baseline) | 0.643 ± 0.001 | 0.674 ± 0.001 | 95.4% |
  | 13 epochs | 0.647 ± 0.002 | 0.676 ± 0.001 | 95.7% |
  | depth 6 | 0.643 ± 0.001 | 0.674 ± 0.001 | 95.5% |

- **What it shows.** Three more epochs raise the skill by 0.004 at 6
  latents and 0.002 at 64, for 30% more training. Two more encoder
  blocks change nothing. The settings stay at 10 epochs and depth 4, and
  the knee rule gives 6 latents under either variant. Within what a
  runner's 6 hours allows, the codec's skill is now limited by something
  other than its training length, depth or regularization.
- **Searched in step 3 so far.** Dropout 0, 0.1, 0.3 and 0.5; ordered
  latents; message history; 13 epochs; depth 6.
- **UNRUN.** A wider network; the information bottleneck; anything on
  the test links.

## 2026-11-12: three more labels, and the step data run again

Grounding can only return a concept its vocabulary holds, and three base
variables were missing ([concepts.md](concepts.md#what-this-means-for-compact)).

- **Code.** `labels_at` now gives `arrival_rate_bps`,
  `flow_arrival_rate_bps` and `active_flows` over the same trailing 100 ms
  as the other windowed labels
  ([architecture.md](architecture.md#ground-truth-labels)). A unit test
  covers the flow count, and the UML test with UDP and TCP cross traffic
  checks all three against the episode's known traffic.
- **Data.** Labels are computed from link events, which the first copy of
  the step data (`exp-phase0-steps-2`) does not hold, so the campaign was
  run again in compact: release `exp-steps-broad-30s-9`
  (`experiments/steps-broad-30s.json`), 14,400 episodes, all valid. The
  rule controller that drives some of its episodes is release
  `exp-steps-behaviour-rule-8`; it is not the rule of
  `exp-phase0-teachers-1`, which sits on a later estimator.
- **It is the same campaign but not the same bytes.** Checked locally on
  the two releases, not by an experiment:
  - Every episode spec of one copy is in the other, once the directory of
    each policy file is dropped. Links and their split are therefore the
    same.
  - In 2 of 20 shards (1,440 episodes, 591,331 steps), 468 episodes are
    equal in every column both copies have, and 972 are not.
  - In 1 shard (720 episodes), no BBR, BBR3 or CDG episode has the same
    number of steps, mean window and mean receive rate in both copies;
    every other controller has some that do. The ratio of an episode's
    mean receive rate, new over old, has median 1.0000 and its 10th and
    90th percentiles are 0.950 and 1.062.
  - Why the copies differ is not known. The first copy was made by a
    different build of the emulator, outside compact.
- **The new labels**, in 1 shard (294,830 steps): never undefined; the
  flow's own arrival rate never exceeds the total; the flow count runs
  from 0 to 5 with median 2.
- **Consequence.** A codec is grounded on the data it was trained on, so
  the knee codecs are trained again on the new release, with five seeds
  as C2 needs: `experiments/codec-knee.json` (4, 6 and 8 latents, and 64
  latents over 3 seeds to compare skills between the two copies).
  Dispatched, no result yet.
- **Probe.** `experiments/ground.py probe` is the first, linear look
  ([methodology.md](methodology.md#plan)). `experiments/ground-probe.json`
  runs it on the codecs of `exp-codec-dropout-6` with the new release's
  messages and labels. Dispatched, no result yet.
- **UNRUN.** The cause of the difference between the copies; the
  comparison over all 20 shards; the probe on codecs trained on the new
  release.

## 2026-11-12: the width curve with dropout, and the knee rule

- **Release.**
  [`exp-codec-dropout-6`](https://github.com/Radonux/compact/releases/tag/exp-codec-dropout-6)
  (`experiments/codec-dropout.json` at commit `e84c9abcc`), with the
  trained weights of its 24 models: fixed widths 2, 3, 4, 6, 12 and 16
  at dropout 0.3, and dropout 0.5 at 8 and 64 latents, seeds 0 to 2.
  Widths 8 and 64 at dropout 0.3 are in `exp-codec-variants-5`. No cell
  failed and no run stopped early. Searched so far for dropout: 0, 0.1,
  0.3 and 0.5, at 8 and 64 latents.
- **Validation links** (skill over persistence, mean and standard
  deviation over 3 seeds, 5 for 64 latents without dropout; the share is
  of the 64-latent skill in the same column):

  | Latents, all kept | No dropout | Share | Dropout 0.3 | Share | 60 ms, dropout 0.3 |
  |---|---|---|---|---|---|
  | 2 | 0.091 ± 0.044 | 14% | 0.086 ± 0.037 | 13% | -0.69 |
  | 3 | 0.519 ± 0.002 | 80% | 0.520 ± 0.003 | 77% | 0.23 |
  | 4 | 0.589 ± 0.004 | 90% | 0.607 ± 0.001 | 90% | 0.44 |
  | 6 | 0.624 ± 0.003 | 96% | 0.643 ± 0.001 | 95.4% | 0.52 |
  | 8 | 0.636 ± 0.002 | 97% | 0.657 ± 0.001 | 97% | 0.55 |
  | 12 | 0.647 ± 0.002 | 99% | 0.666 ± 0.000 | 99% | 0.57 |
  | 16 | 0.649 ± 0.002 | 100% | 0.670 ± 0.000 | 99% | 0.57 |
  | 64 | 0.652 ± 0.001 | 100% | 0.674 ± 0.001 | 100% | 0.59 |

  Dropout at 8 latents: 0: 0.636, 0.1: 0.652, 0.3: 0.657, 0.5: 0.647.
  At 64 latents: 0: 0.652, 0.1: 0.669, 0.3: 0.674, 0.5: 0.669.
- **What it shows.**
  1. Dropout 0.3 is the best of the four values at both widths, with
     lower skill on either side, so the search has an interior optimum.
     Dropout 0.3 is now the codec's setting.
  2. The shape of the width curve does not depend on regularization.
     With and without dropout, 2 latents fail, 3 give about 80% of the
     64-latent skill, 4 give 90%, 6 give 95 to 96%, and the curve is flat
     from 12. Dropout does not help at 2 and 3 latents, where the model
     is limited by its width and not by overfitting, and helps by about
     0.02 from 4 latents on.
  3. The spread over seeds with dropout is at most 0.003 from 3 latents
     on, far below the steps between widths up to 8.
  4. As before, the extra latents mostly buy the 60 ms horizon.
- **The knee rule is fixed** ([methodology.md](methodology.md#terms)):
  the smallest fixed width whose validation skill is within 5% of the
  64-latent skill at the tuned settings. That is **6 latents** (95.4%
  with dropout 0.3, 96% without). Reasons for 5% and not 10%: the 10%
  rule gives 4 latents, which sit exactly on its threshold (90%) and
  reach only 0.44 skill at 60 ms against 0.59; and the steps from 4 to 6
  are 20 or more standard deviations, so they are real. The rule was
  chosen after seeing validation links and before any test link is
  read. Grounding will also be run at 4 and 8 latents, as the
  sensitivity of everything downstream to this choice. The number is
  close to the seven hand-picked LCFs, which says nothing yet about
  whether the concepts are the same.
- **Dispatched.** `experiments/codec-capacity.json`: 13 epochs in place
  of 10, and encoder depth 6 in place of 4, each at 6 and 64 latents,
  dropout 0.3, 3 seeds. The validation loss was still falling in the
  last epoch of the dropout models, and the search's winner was the
  largest network on its grid. A wider network does not fit a runner's
  6 hours and is UNRUN. No result yet.
- **UNRUN.** Longer training and a deeper encoder (dispatched); a wider
  network; dropout with ordered latents; message history on matched
  anchors; the information bottleneck; widths 1, 5, 7 and 10; anything
  on the test links; grounding.

## 2026-11-12: codec variants, dropout helps and the other two do not

- **Release.**
  [`exp-codec-variants-5`](https://github.com/Radonux/compact/releases/tag/exp-codec-variants-5)
  (`experiments/codec-variants.json` at commit `fdf49cf22`): 24 models,
  each change alone on the confirmed settings, at 8 latents and at 64,
  all kept, seeds 0 to 2. Searched: dropout 0.1 and 0.3, the last 4
  messages as input, the utility's terms as extra targets. No cell
  failed and no run stopped early. The baselines are the 8-latent and
  64-latent models of `exp-codec-confirm-4`.
- **Validation links** (mean and standard deviation over seeds; skill
  and R^2 are over the three observation targets and five horizons;
  "60 ms" and "960 ms" are the skill at that horizon):

  | Model | Seeds | R^2 | Skill | 60 ms | 960 ms | Training loss | Validation loss |
  |---|---|---|---|---|---|---|---|
  | 8 latents, baseline | 3 | 0.9619 | 0.636 ± 0.002 | 0.527 | 0.661 | 0.0096 | 0.0146 |
  | 8 latents, dropout 0.1 | 3 | 0.9637 | 0.652 ± 0.001 | 0.545 | 0.684 | 0.0110 | 0.0140 |
  | 8 latents, dropout 0.3 | 3 | 0.9647 | 0.657 ± 0.001 | 0.546 | 0.697 | 0.0129 | 0.0137 |
  | 64 latents, baseline | 5 | 0.9629 | 0.652 ± 0.001 | 0.568 | 0.667 | 0.0088 | 0.0142 |
  | 64 latents, dropout 0.1 | 3 | 0.9647 | 0.669 ± 0.001 | 0.586 | 0.690 | 0.0099 | 0.0135 |
  | 64 latents, dropout 0.3 | 3 | 0.9658 | 0.674 ± 0.001 | 0.586 | 0.704 | 0.0116 | 0.0132 |
  | 8 latents, utility targets | 3 | 0.9586 | 0.583 ± 0.004 | 0.405 | 0.637 | | |
  | 64 latents, utility targets | 3 | 0.9603 | 0.613 ± 0.002 | 0.488 | 0.651 | | |
  | 8 latents, 4 messages | 3 | 0.9629 | 0.606 ± 0.002 (other anchors) | 0.490 | 0.638 | 0.0092 | 0.0141 |
  | 64 latents, 4 messages | 3 | 0.9640 | 0.625 ± 0.001 (other anchors) | 0.538 | 0.645 | 0.0082 | 0.0136 |

  The training loss of a dropout model is measured with dropout on, so
  it is not a clean measure of fit. The losses of the utility models
  cover six targets and are left out.
- **What it shows.**
  1. Dropout helps at both widths, by 10 or more standard deviations:
     +0.021 skill at 8 latents and +0.022 at 64 for dropout 0.3. Most of
     the gain is at the long horizons. 0.3 beat 0.1 and is the largest
     value tried, so the best value may be higher.
  2. Dropout does not close the gap between 8 and 64 latents (0.016
     without, 0.018 with dropout 0.3). The width curve of the
     confirmation was read from models that overfit, so it is rerun with
     dropout before the count is read.
  3. The utility's terms as extra targets lower the skill on the
     observation targets (0.636 to 0.583 at 8 latents, 0.652 to 0.613 at
     64), most at 60 ms. The codec also predicts those terms badly at
     short horizons: for the flow's share the skill is -0.88 at 60 ms
     and 0.19 at 960 ms (64 latents, seed 0), so persistence is better
     there. Dropped.
  4. With the last 4 messages as input the anchors differ (an anchor
     needs 3 earlier steps; persistence R^2 0.875 against 0.869), so the
     skill cannot be set beside the baseline's. R^2 rises by 0.001, a
     third of what dropout 0.3 gives, and by target the skill is lower
     for the delivery rate (0.63 against 0.73 at 64 latents) and higher
     for loss (0.70 against 0.68), which may be the anchors and not the
     model. It also fits the training links hardest (0.0082). Dropped:
     no clear gain, and the one-message encoder keeps a concept a
     function of one step. A comparison on the same anchors is UNRUN.
  These agree with the local probes of 2026-11-11.
- **Dispatched.** `experiments/codec-dropout.json`: the width curve at
  dropout 0.3 (2, 3, 4, 6, 12 and 16 latents; 8 and 64 are in this
  release), and dropout 0.5 at 8 and 64 latents, 3 seeds each. No result
  yet.
- **UNRUN.** The width curve with dropout; dropout above 0.3; dropout
  with ordered latents; message history on matched anchors; the
  information bottleneck; widths 1, 5, 7 and 10; a network larger than
  512x4; the knee rule; anything on the test links; grounding.

## 2026-11-12: the codec confirmed over seeds

- **Release.**
  [`exp-codec-confirm-4`](https://github.com/Radonux/compact/releases/tag/exp-codec-confirm-4)
  (`experiments/codec-confirm.json` at commit `88db3f48d`), with the
  trained weights of all 31 models. Settings: the search's winner
  (learning rate 1e-3, width 512, encoder depth 4, weight decay 1e-4),
  all 4,079,476 training anchors, 10 epochs; nothing else was searched.
  One cell (16 ordered latents, seed 4) lost its runner and was rerun in
  the same run. No run stopped early.
- **Validation links** (768,620 anchors; mean and standard deviation over
  seeds; persistence has R^2 0.869 and skill 0 by definition):

  | Model | Seeds | R^2 | Skill | Share of the 64-latent skill |
  |---|---|---|---|---|
  | 2 latents, all kept | 3 | 0.920 | 0.091 ± 0.044 | 14% |
  | 3 latents | 3 | 0.954 | 0.519 ± 0.002 | 80% |
  | 4 latents | 3 | 0.959 | 0.589 ± 0.004 | 90% |
  | 6 latents | 3 | 0.961 | 0.624 ± 0.003 | 96% |
  | 8 latents | 3 | 0.962 | 0.636 ± 0.002 | 97% |
  | 12 latents | 3 | 0.963 | 0.647 ± 0.002 | 99% |
  | 16 latents | 3 | 0.963 | 0.649 ± 0.002 | 100% |
  | 64 latents (no narrow layer) | 5 | 0.963 | 0.652 ± 0.001 | 100% |
  | 16 ordered latents, all kept | 5 | 0.962 | 0.626 ± 0.003 | 96% |

  The ordered model's skill with its first n latents kept: 1: -2.49,
  2: -0.14, 3: 0.33, 4: 0.48, 6: 0.56, 8: 0.59, 10: 0.61, 12: 0.62,
  16: 0.63 (standard deviation over 5 seeds at most 0.01 from 3 latents
  on).
- **What it shows.**
  1. Two latents are not enough and three are: skill jumps from 0.09 to
     0.52 between 2 and 3 latents, then rises slowly (0.59 at 4, 0.62 at
     6, 0.64 at 8) and is flat from 12 on. The spread over seeds is far
     smaller than the steps between widths up to 8.
  2. 16 is wide enough: 16 latents reach the skill of 64 (0.649 against
     0.652).
  3. Ordering has a cost that is largest at small widths: the first 3
     ordered latents reach 0.33 where a 3-latent model reaches 0.52, and
     all 16 ordered reach 0.626 where 16 unordered reach 0.649. So the
     ordered model's curve understates what a narrow layer can do, and
     the count must be read from the fixed-width models. The ordered
     model is still useful as one model whose latents come in order of
     importance.
  4. The gain of more latents is mostly at the shortest horizon: at
     60 ms the skill is 0.21 with 3 latents, 0.40 with 4, 0.53 with 8 and
     0.57 with 64, while at 240 ms and beyond 3 latents already reach
     0.61 to 0.62 of 64 latents' 0.67 to 0.69.
  5. The fixed-width models fit the training links better than the
     validation links (64 latents: training loss 0.0088, validation loss
     0.0142, validation flat over the last two epochs). The search did
     not see this because its models were ordered, and dropping latents
     regularizes. So regularization is worth a try (dispatched below).
  6. The ordered latents are correlated with each other (largest absolute
     correlation of a latent with another: 0.17 to 0.66 in seed 0), so
     grounding should fit the subspace, as planned, and not each latent
     alone.
- **The knee is not chosen yet.** The curve has a sharp step at 3 and a
  long shallow tail. Candidates for the rule: the smallest width within
  10% of the 64-latent skill (4 latents) or within 5% (6 latents). The
  rule is an open item; it should be fixed before the test links are
  read.
- **Dispatched.** `experiments/codec-variants.json`, step 3: each change
  alone at 8 and at 64 latents, 3 seeds: the last 4 messages as input,
  the utility's terms as extra targets, and dropout 0.1 and 0.3. No
  result yet.
- **UNRUN.** The variants' results; the information bottleneck; widths 1,
  5, 7 and 10 as fixed-width models; a network larger than 512x4;
  anything on the test links; grounding.

## 2026-11-11: the codec, its search and local probes

- **The plan changed.** Discovery no longer distills a teacher. The
  discovery model is the codec ([methodology.md](methodology.md#the-codec)):
  it predicts the flow's own future (smoothed RTT, delivery rate, loss,
  60 ms to 960 ms ahead) through k latents, given the window changes in
  between. Reason: a student of a teacher can only return what that
  teacher uses, and its latents collapse toward the action (finding 3 of
  the prototype below); a predictive state depends on the path alone.
  Teachers and the closed loop moved to step 5.
- **Width.** 16 ordered latents (nested dropout), so one model gives the
  score at every width from 1 to 16. The prototype's knee was near 3 and
  the hand-picked set has 7, so 16 leaves room above both.
- **Code.** `experiments/codec.py`, `experiments/codec_report.py`,
  `tests/test_codec.py`. The positive control passes: on synthetic data
  with 3 hidden factors the score levels off at 3 latents.
- **Persistence.** Predicting that nothing changes already has a mean
  R^2 of 0.87 on the validation links, so R^2 flatters a model that
  copies the present. Every result below gives skill over persistence
  (mean of 1 - MSE / persistence's MSE) beside R^2.
- **Search.** Release
  [`exp-codec-search-3`](https://github.com/Radonux/compact/releases/tag/exp-codec-search-3)
  (`experiments/codec-search.json` at commit `44469af73`). Searched, as a
  full grid of 18 runs: learning rate 3e-4, 1e-3, 3e-3; encoder width x
  depth 256x2, 256x4, 512x4; weight decay 1e-4, 1e-2. Each run: 1,000,000
  training anchors, 25 epochs, seed 0, 16 ordered latents. All 18
  finished; none stopped early. Validation links (768,620 anchors):

  | Setting | Validation loss | R^2 | Skill |
  |---|---|---|---|
  | lr 1e-3, 512x4, wd 1e-4 (winner by the rule: lowest validation loss) | 0.0325 | 0.960 | 0.607 |
  | lr 1e-3, 512x4, wd 1e-2 | 0.0327 | 0.960 | 0.605 |
  | best of 256x2 (lr 3e-3, wd 1e-4) | 0.0334 | 0.959 | 0.594 |
  | best of 256x4 (lr 3e-3, wd 1e-4) | 0.0335 | 0.959 | 0.597 |
  | worst (lr 3e-4, 256x2, wd 1e-2) | 0.0386 | 0.954 | 0.530 |

  Persistence: R^2 0.869. The winner's skill by number of latents kept:
  1: -2.75, 2: -0.16, 3: 0.32, 4: 0.46, 6: 0.52, 8: 0.56, 10: 0.59,
  12: 0.60, 16: 0.61. A negative skill at 1 or 2 latents means that so
  few latents predict worse than copying the present.
- **Reading the search.** Weight decay does not matter. The learning
  rate matters most (3e-4 is too low for 25 epochs). Size matters
  little: the gap between the smallest and the largest network at their
  best learning rates is 0.0009 in loss. Training and validation losses
  are equal, so nothing overfits and the limit is data and steps, not
  regularization. The winner is the largest network searched, so the
  grid's edge was not passed; a larger network is UNRUN.
- **Local probes** (4 of 20 shards, 8 epochs, default untuned settings,
  validation links, one seed; run on the development machine, not a
  release, so design hints only):

  | Probe | Skill | Note |
  |---|---|---|
  | 16 ordered latents, one message | 0.47 | by width: 3: 0.15, 4: 0.27, 6: 0.40, 8: 0.44, 10: 0.45, 16: 0.47 |
  | 64 latents, all kept | 0.52 | no narrow layer |
  | input of the last 4 or 8 messages | 0.40 | validation loss within 3% of one message; anchors differ slightly |
  | utility terms as extra targets | 0.07 over all targets | share of capacity and queueing delay are not visible to the sender, so their persistence is an oracle; R^2 0.76 to 0.95 on them |

  On this evidence the message history is not adopted. Neither probe is
  settled: both are UNRUN as experiments.
- **Dispatched.** `experiments/codec-confirm.json`: the winner on all
  4.0 million training anchors, 10 epochs: 16 ordered latents (5 seeds),
  64 latents all kept (5 seeds), and fixed widths 2, 3, 4, 6, 8, 12, 16
  (3 seeds each). No result yet.
- **UNRUN.** The confirmation's results; the choice of the knee; the
  information bottleneck; anything on the test links; grounding.

## 2026-11-11: setup

- **Documents.** Rewritten for compact: the question, claims, terms,
  protocol and plan ([methodology.md](methodology.md)), the emulator
  ([architecture.md](architecture.md), [validation.md](validation.md)),
  the commands ([training.md](training.md)) and the literature's
  candidate concepts ([concepts.md](concepts.md)).
- **Phase 0 inputs as releases of this repository.** The two teachers:
  [`exp-phase0-teachers-1`](https://github.com/Radonux/compact/releases/tag/exp-phase0-teachers-1)
  (`teachers/rule-bdp.onnx` and `teachers/td3bc.onnx`, 16,152,577 bytes
  each, SHA-256 in the release notes). The step data:
  [`exp-phase0-steps-2`](https://github.com/Radonux/compact/releases/tag/exp-phase0-steps-2)
  (20 complete shards, 2.2 GB unpacked). Both were copied unchanged by
  `experiments/phase0-teachers.json` and `experiments/phase0-steps.json`,
  whose descriptions record the source. An experiment reads them as
  `"inputs": {"teachers": "release:exp-phase0-teachers-1", "steps":
  "release:exp-phase0-steps-2"}`.
- **Tests.** Without a UML build or the Mahak and Sage checkouts,
  `uv run pytest -q` gave 202 passed and 57 skipped. The UML tests were
  not run locally (UNRUN here; CI runs them).
- **Nothing was trained or evaluated.**

## Before 2026-11-11: the Phase 0 prototype

A prototype ran on 2 CPUs before this repository was set up (its date
was not recorded). It is not a
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
