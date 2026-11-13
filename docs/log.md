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
| 4 Grounding | first look done (2026-11-13) on codecs trained on the first copy of the data: probe `exp-ground-probe-10`, selection `exp-ground-select-12`; codecs trained on the new data are in `exp-codec-knee-11` (2026-11-13); probe and selection over them (`experiments/ground-knee.json`) dispatched, no result yet | the labels and the step message in `steps.parquet`; `experiments/ground.py probe` and `select`; the trained weights in `exp-codec-knee-11` | formulas beyond a linear fit (PySR on the residual); the operator-family sensitivity |
| 5 Control | UNRUN | a recurrent actor on raw measurements (`train ppo`, `inputs: "raw"`), trained by PPO, TD3+BC or DAgger; closed-loop rollouts and scoring | a controller that reads the codec's latents or fitted concepts; DAgger from a teacher given as a graph; the bottlenecked student in the closed loop |
| 6 Retrain and compare | UNRUN | `train run` for net1 and an offline net2; `train ppo` for a closed-loop net2; oracles on the seven LCFs | labels from fitted formulas; net1 on them; the live truth for them |

C1 has first evidence from a release, on validation links only (entry of
2026-11-12). C2 has a first look at the level of the latent space (entry
of 2026-11-13). C3 to C5 have none.

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
