# Status

Everything that changes with time: what exists, what has run, what it
showed and what is open. This page is rewritten in place and holds no
dated entries; its history is in git. The steps are those of the
[plan](methodology.md#plan).

## Where the project stands

No claim has evidence yet. The closed-loop runs so far: the start taught
from the fixed rule
([below](#the-start-taught-from-the-fixed-rule)), the bootstrap from
Cubic and BBRv3, the reinforcement-learning stage at one point, its
settings search and the confirmation of the two best settings per
teacher. The confirmation gives the first comparisons with the
reference at tuned settings, over seeds only and without the share
with a competing flow, so they are not yet results for a claim.

| Step | State | Built | Not built |
|---|---|---|---|
| 1 Setup | done | the documents; the estimator of the seven LCFs and the rule graph as releases | |
| 2 The controller in the closed loop | the bootstrap has run; reinforcement learning is tuned at width 10 and confirmed at widths 7, 10 and 16 over three seeds | `PPOSpec.latents`; `train ppo` by cloning a Linux controller, PPO, DAgger or TD3+BC; closed-loop rollouts and scoring; `experiments/net2_ppo_setup.py`, `net2_ppo_report.py`; the collapse check on rollouts (`ground.py probe`); the reinforcement-learning stage (`experiments/controller-rl.json`) and its scores by width (`experiments/width_curve.py`); its settings search and confirmation (`experiments/controller-rl-search.json`, `controller-rl-confirm.json`, `settings_search.py`) | the share with a competing flow; intervals over links; the teachers' own scores on the validation links |
| 3 Grounding | first readings on the untuned and the confirmed controllers, at widths 7 to 16 with the sender's own window held; the decision's change is mostly unexplained | `experiments/ground.py`: `rollouts` of a controller in the closed loop, `probe` and `select` on its latents there, with the sensitivity check; a library of every stored label (37 columns over 100 ms and 10 ms windows) and 12 composites in both windows; `experiments/ground-teachers.json`; a stress test on planted latents, `experiments/ground_stress.py` | formulas beyond two terms of four factors; the operator-family sensitivity; matching concepts across seeds |
| 4 Retrain and compare | UNRUN | `train run` for net1; `train ppo` for a closed-loop net2; net1 on any stored label or composite; the live truth for every label but `queue_delay_ns`, so oracles on them | labels from fitted formulas beyond the composites; net1 on them |

## The bootstrap

What it trained: the discovery controller at widths 1, 2, 4, 7, 10, 16 and
64, three seeds each, once from Cubic and once from BBRv3. The teacher
drives every training rollout, and the controller learns the window that
sends as the teacher did. A reference without an encoder (k 0 in the
experiment file) is the recurrent network on the raw measurements, its
window on a measured bandwidth-delay product.

Why these widths: 7 is expliCCit's budget, 10 was the owner's first
proposal, and 64 stands for no constraint. Widths 1 and 2 are there to
show action collapse.

What was read from it (release `exp-controller-teachers-22`, 48 runs, no
failed rollout): how each start drives in the closed loop, in log utility
on 21 validation links under 6 conditions against the fixed rule, the
noise-free policy after 40 iterations. Mean of three seeds, with the
lowest and the highest:

| Width | From Cubic | From BBRv3 |
|---|---|---|
| none | -1.34 (-1.57 to -1.07) | -1.49 (-1.82 to -1.23) |
| 1 | -1.58 (-1.70 to -1.36) | -1.89 (-2.37 to -1.08) |
| 2 | -1.29 (-1.46 to -1.10) | -1.74 (-2.36 to -1.39) |
| 4 | -1.20 (-1.34 to -1.13) | -1.19 (-1.63 to -0.54) |
| 7 | -1.46 (-1.56 to -1.29) | -1.63 (-2.38 to -0.89) |
| 10 | -0.99 (-1.07 to -0.88) | -1.07 (-1.52 to -0.56) |
| 16 | -1.39 (-1.74 to -1.05) | -1.63 (-2.01 to -1.01) |
| 64 | -1.00 (-1.32 to -0.55) | -1.48 (-2.10 to -0.68) |

Every start is below the rule, by 0.55 to 2.38. Before any training the
controller with latents scores -0.79 and the reference -0.55, and 44 of
the 48 runs end below where they began. The score also moves between
evaluations of one run, by 0.83 at the median, while the imitation
loss falls steadily to 0.01 to 0.04. So a low imitation loss on the
teacher's rollouts does not give a stable controller here. No width
stands out, and the spread across seeds is as large as any difference
between widths.

What this does not say: whether the clones drive like their teachers.
The teachers' own scores on the validation links are UNRUN, so a clone at
-1.3 may be a faithful copy of a teacher that scores -1.3 against the
rule, or a poor copy of a better one. The teacher drove every training
rollout, so the clone never saw the states its own mistakes lead to.
Nothing is chosen from this run, and its width curve is not a count,
because a teacher's window is one number. Every width goes on to
reinforcement learning.

What to keep in mind when reading it:

- The label for BBRv3 is its pacing rate times its smoothed RTT, capped
  by its window. That reproduces its sending rate under the `agent`
  controller, not its internal state machine.
- The MSS in that conversion is a constant, 1448 bytes. It is UNRUN
  against a recorded `TCP_INFO`.
- The estimator of the seven LCFs serves only as the measurement scaler
  and as the rule the evaluations compare with. That estimator and rule
  were fitted on episodes from the guest before the cross-flow fix
  (below).

## The start taught from the fixed rule

`experiments/controller-clone.json` taught the same widths by DAgger from
the fixed rule on the seven hand-picked LCFs, three seeds each. Its
release is `exp-controller-clone-21`. All 24 runs finished their 40
iterations.

Score: mean log utility minus the fixed rule's, on the 21 validation
links under 6 conditions, for the policy after the last iteration.

| width | mean over 3 seeds | the seeds |
|---|---|---|
| 0 (no encoder) | -0.028 | -0.031, -0.040, -0.013 |
| 1 | -0.041 | -0.049, -0.042, -0.032 |
| 2 | -0.037 | -0.031, -0.020, -0.060 |
| 4 | -0.046 | -0.047, -0.021, -0.071 |
| 7 | -0.054 | -0.078, -0.046, -0.038 |
| 10 | -0.029 | -0.044, -0.040, -0.003 |
| 16 | -0.037 | -0.042, -0.047, -0.022 |
| 64 | -0.033 | -0.035, -0.005, -0.058 |

What it shows:

- The narrow controller's exported graph runs in the guest, at every
  width.
- Every width ends 0.03 to 0.05 below the rule. Before teaching, the
  encoder controllers were 0.79 below.
- The curve is flat. Width 1 is 0.008 below width 64, and seeds of one
  width differ by up to 0.05. So one latent carries the rule's decision.
  That was expected: the rule's window is one number. It is why imitation
  alone cannot count concepts.

What it does not show: anything about concepts. The runs are untuned, the
teacher is the hand-built rule, and the decision was not regressed on the
latents, because these runs kept no rollouts. This start leans toward the
seven hand-picked LCFs, which is why the Cubic and BBRv3 bootstrap
replaced it. It stays as a third start for C5.

## The reinforcement-learning stage

`experiments/controller-rl.json` continued every bootstrapped controller
by PPO on the log utility, from the clone's actor and value network, for
120 iterations (release `exp-controller-rl-24`, 48 runs, no failed
rollout). Its settings are one point, taken from PPO runs made before
this repository, and are not searched.

Its rules were fixed in the file before it ran. A run's score is that of
the policy it leaves, in log utility against the fixed rule on the 21
validation links under 6 conditions. The knee is the smallest width
within one pooled standard error of the best width. Each controller with
latents then drove rollouts of its own, and its decision was regressed on
its latents.

Mean score of three seeds, with the lowest and the highest:

| Width | From Cubic | From BBRv3 |
|---|---|---|
| none | -0.16 (-0.19 to -0.13) | -0.13 (-0.14 to -0.13) |
| 1 | -0.25 (-0.29 to -0.21) | -0.47 (-0.53 to -0.37) |
| 2 | -0.38 (-0.53 to -0.26) | -0.46 (-0.60 to -0.32) |
| 4 | -0.30 (-0.54 to -0.15) | -0.22 (-0.30 to -0.14) |
| 7 | -1.46 (-2.44 to -0.22) | -0.37 (-0.80 to -0.15) |
| 10 | -0.21 (-0.28 to -0.15) | -0.47 (-0.51 to -0.41) |
| 16 | -0.24 (-0.31 to -0.13) | -0.31 (-0.39 to -0.26) |
| 64 | -0.29 (-0.45 to -0.19) | -0.50 (-0.83 to -0.26) |

What it shows:

- PPO raises the clones' log utility. They started 0.5 to 2.4 below the
  rule, and 46 of the 48 runs end 0.13 to 0.83 below it. The references
  without an encoder end 0.13 and 0.16 below.
- Three of the six validation conditions have a competing flow (BBR from
  4 s, Cubic from 0 s, two Cubic flows), and there a larger share raises
  the score. The only cells above the rule are of this kind: the
  references under Cubic from 0 s (+0.12 from Cubic) and under two Cubic
  flows (+0.14 and +0.05). Whether these come from a larger share is
  UNRUN: the share is not yet computed, and the table above does not
  split the conditions.
- Among the controllers trained the same way, by mean and spread over 3
  seeds. Intervals over links and seeds are UNRUN, so these are readings
  and not verdicts:
  - The two references without an encoder are 0.03 apart, with a spread
    between seeds of 0.01 and 0.03.
  - The width-64 controllers are 0.13 (Cubic) and 0.37 (BBRv3) below their
    references. Passing the decision through latents costs something at
    this point of the settings.
  - The narrower widths against width 64 are unresolved.
    Width 64's own spread is 0.14 and 0.30, larger than the
    margin. Cubic at width 7 is 1.17 below, from two runs that diverged.
- Two runs got worse: Cubic at width 7, seeds 1 and 3, end at -1.71 and
  -2.44. The third seed of that width ends at -0.22, so this is the
  training and not the width.
- The reference without an encoder has the highest mean and the smallest
  spread between seeds for both teachers.
- The widths do not separate. The rule in the file puts a knee at
  width 1 from Cubic (pooled standard error 0.26) and at width 4 from
  BBRv3 (0.11).
- The decision is largely in the latents at widths 1 and 2. A quadratic
  fit of the decision on the latents alone has a held-out R^2 of 0.80 to
  0.93 at width 1 and 0.82 to 0.91 at width 2, and width 1 scores as
  well as width 64. This is action collapse in the closed
  loop. It is what the objective allows
  ([methodology.md](methodology.md#what-the-width-cannot-count)), so the
  width is no longer read as a count. The count is read in retraining,
  from performance against the number of fitted concepts.
- Many runs were still improving at the last evaluation, so 120
  iterations is not the end of training.

Width 64 is no longer trained. The controllers that are grounded have 7,
10 and 16 latents ([methodology.md](methodology.md#the-discovery-controller)).
In this release the wider middle trained no better and no more steadily:
width 64 ends 0.29 and 0.50 below the rule with a spread between seeds of
0.14 and 0.30, and widths 10 and 16 end 0.21 to 0.47 below with a spread
of 0.05 to 0.10.

The settings search has run (`exp-controller-rl-search-29`,
`experiments/controller-rl-search.json`): width 10 over a full grid of
learning rate (0.00003, 0.0001, 0.0003), clip (0.1, 0.2) and exploration
noise (0.1, 0.2), from both teachers, seeds 1 and 2. The reason for these
three: over the last 40 iterations the learner at width 10 clipped 23% of
decisions with a step KL of 0.019, against 11% and 0.009 for the
references. It chooses candidates and supports no claim. Scores are the
log utility minus the fixed rule's, as the mean of two seeds, with the
spread between them.

| Teacher | Learning rate | Clip | Noise | Score | Spread | Without a competing flow | With one |
|---|---|---|---|---|---|---|---|
| Cubic | 0.0003 | 0.2 | 0.2 | -0.17 | 0.00 | -0.23 | -0.12 |
| Cubic | 0.00003 | 0.2 | 0.2 | -0.27 | 0.09 | -0.18 | -0.36 |
| Cubic | 0.0001 | 0.2 | 0.2 | -0.28 | 0.01 | -0.15 | -0.40 |
| BBRv3 | 0.0001 | 0.1 | 0.1 | -0.26 | 0.15 | -0.18 | -0.33 |
| BBRv3 | 0.00003 | 0.2 | 0.1 | -0.26 | 0.02 | -0.16 | -0.35 |
| BBRv3 | 0.0003 | 0.2 | 0.2 | -0.29 | 0.07 | -0.29 | -0.29 |

These are the three best of twelve points per teacher. The release's
summary has all of them.

- No point reaches the fixed rule.
- A learning rate of 0.0003 with noise 0.1 fails for both teachers. Its
  four points end 1.05 to 2.94 below the rule, with about half of the
  decisions clipped.
- For Cubic the point that was already in use is the best, by 0.10 over
  the next. Noise 0.2 is better than 0.1 there: the four best points
  all have it.
- For BBRv3 the nine best points lie between -0.26 and -0.40 and the
  spread between two seeds reaches 0.15, so the search does not separate
  them.
- The point that was in use scores higher here than in
  `exp-controller-rl-24`: for BBRv3 -0.29 over two seeds against -0.47
  over three, for Cubic -0.17 against -0.21. Between the two releases
  the trainer changed (below) and one seed is missing, so for BBRv3 more
  of the gain is between releases than between points.
- The scores of the two best points per teacher changed by 0.04 or less
  over the last 20 iterations.

The confirmation has run (`exp-controller-rl-confirm-31`,
`experiments/controller-rl-confirm.json`): the two best points per
teacher at widths 7, 10 and 16 and the reference without an encoder,
seeds 1 to 3, 48 runs, none failed. Its rules were fixed in the file
before it ran.

Each teacher's setting is the point with the higher mean over its nine
runs at widths 7, 10 and 16:

- Cubic: learning rate 0.0003, clip 0.2, noise 0.2, at -0.35 against
  -0.61. Width 7 makes most of the gap, where seed 1 ends at -1.44 and
  -2.99. Without width 7 the order holds, -0.24 against -0.29.
- BBRv3: learning rate 0.00003, clip 0.2, noise 0.1, at -0.25 against
  -0.31. Without width 7 the two points tie at -0.28, so width 7
  decides.

At that setting, each width against the reference trained at the same
point. The score is the log utility minus the fixed rule's, the mean of
three seeds. The difference from the reference has a 95% interval over
seeds (Welch's t, 2 to 4 degrees of freedom), and the verdict follows
the margin of 0.1
([methodology.md](methodology.md#comparing-controllers)).

| Teacher | Width | Score | Minus the reference | 95% interval | Verdict | Minus the reference, without a competing flow | With one |
|---|---|---|---|---|---|---|---|
| Cubic | none | -0.16 | | | | | |
| Cubic | 7 | -0.59 | -0.43 | -2.25 to +1.40 | unresolved | -0.38 | -0.48 |
| Cubic | 10 | -0.17 | -0.01 | -0.14 to +0.12 | unresolved | +0.03 | -0.05 |
| Cubic | 16 | -0.31 | -0.15 | -0.45 to +0.16 | unresolved | +0.06 | -0.35 |
| BBRv3 | none | -0.11 | | | | | |
| BBRv3 | 7 | -0.20 | -0.09 | -0.17 to -0.02 | unresolved | -0.03 | -0.16 |
| BBRv3 | 10 | -0.28 | -0.17 | -0.23 to -0.11 | not comparable | +0.00 | -0.34 |
| BBRv3 | 16 | -0.28 | -0.17 | -0.23 to -0.11 | not comparable | -0.02 | -0.32 |

What it shows:

- No controller reaches the fixed rule. The references end 0.16 (Cubic)
  and 0.11 (BBRv3) below it.
- The shortfalls lie with a competing flow. Without one, every width but
  Cubic's width 7 is within 0.06 of its reference, and Cubic's width 16
  is comparable there (interval -0.06 to +0.18). With one, BBRv3's widths
  10 and 16 are 0.34 and 0.32 below (intervals -0.62 to -0.07 and -0.61
  to -0.03). There a higher score can come from a larger share of the
  link, which is not a gain. The share is not computed, so whether the
  two verdicts of not comparable are a cost of the latents, or the
  reference taking more of the link, is UNRUN.
- The intervals are over seeds only. The trainer kept the paired scores
  per condition and not per link, so intervals over links could not be
  drawn from this release. It now keeps every link and condition.
- Cubic's width 7, seed 1, fails at both points. It starts at -1.54, as
  the other seeds do (-1.29 and -1.56), and ends at -1.44 and -2.99,
  while seeds 2 and 3 end at -0.13 to -0.47. Its training rollouts show
  a window that stays small. Its clone starts with the smallest window
  of the three (u of -4.2, against -2.9 and -3.3), and over the last 40
  iterations its mean u is -3.0 and -5.1 at the two points and the link
  is 46% and 25% used, against -0.6 to -1.4 and 76% to 89% for the other
  seeds. It moves far from its start all the same (KL to the start 18
  and 24, against 3 to 6), so it settles on a small window rather than
  staying put. Both releases train from the same clones, and in
  `exp-controller-rl-24` this width diverged too. Why this clone falls
  into a small window and seed 3, which starts almost as low, does not
  is UNRUN.
- The setting chosen for Cubic gives the results of the search again at
  width 10 (-0.17 in both). The references score a little higher with the
  slower learner for both teachers (Cubic -0.13 against -0.16, BBRv3
  -0.11 against -0.18), which the rules do not use.
- Scores rose by 0.02 to 0.03 on average over the last 20 iterations, by
  up to 0.13 in single runs. Training is not done at 120 iterations.

Longer training and more than three seeds are UNRUN. More seeds need
more clones from the bootstrap. The grounding of the confirmed
controllers has run ([below](#grounding-the-confirmed-controllers)).

Two things changed in the trainer for it. A run that the time budget ends
between evaluations now scores the policy it leaves; before, that policy
had no score. And the value network a clone hands over learned the
teacher's return, so the stage updates it alone for 3 iterations first.

## The grounding library

Every basic concept the emulator knows now takes part in the fitting. The
step tables store 37 label columns: 24 labels, 13 of them over both a
100 ms and a 10 ms window. Twelve composites from the literature are
computed from them in both windows. `experiments/ground.py` reads both
lists from the code, so its library is all of them and the four sender
measurements, each under the eight operators. Its sensitivity check takes
a label out in both windows together with every composite computed from
it. The guest gives a policy the same values live, all but
`queue_delay_ns`, so an oracle on any of them is possible.

The labels, the composites, the reader that recomputes labels from stored
link tables, and the live truth were brought in from the project compact
was forked from, as its commits, with compact's own changes kept.

What this does not change: the bootstrap runs above started on the guest
before this change, and their steps carry the earlier 19 labels. Their
policies read only the measurements, so grounding runs them again on the
current guest.

## Grounding on a controller's own rollouts

`experiments/ground.py` now reads a discovery controller, not logged
steps of other controllers. `rollouts` runs the controller's exported
policy in the closed loop, on a sample of the training links and on the
validation links. `probe` and `select` compute the latents from those
steps through the controller's own encoder. `probe` also regresses the
decision on the latents, which is the action-collapse check. A controller
without latents is refused.

`select` also grounds each controller's decision in place of its latents,
with the same paths, formulas and sensitivity check.

Two releases hold the first readings on real controllers. Both sets of
controllers are untuned, so neither supports a claim about concepts.
`exp-ground-teachers-26` is the bootstrapped controllers: both teachers,
widths 1 to 64, three seeds, latents only. `exp-ground-rl-27` is the
controllers after reinforcement learning at widths 1, 2 and 64, latents
and decision.
All numbers are R^2 on the validation links.

| Bootstrapped, by width | 1 | 2 | 4 | 7 | 10 | 16 | 64 |
|---|---|---|---|---|---|---|---|
| Latents from the whole library, Cubic | 0.96 | 0.80 | 0.67 | 0.71 | 0.66 | 0.72 | 0.67 |
| Latents from the whole library, BBRv3 | 0.96 | 0.82 | 0.77 | 0.76 | 0.76 | 0.79 | 0.69 |
| Latents from the seven hand-picked LCFs, Cubic | 0.62 | 0.38 | 0.30 | 0.37 | 0.28 | 0.39 | 0.38 |
| Latents from the seven hand-picked LCFs, BBRv3 | 0.79 | 0.57 | 0.51 | 0.50 | 0.52 | 0.57 | 0.45 |
| One run's 8 features on another run's latents, Cubic | 0.94 | 0.74 | 0.54 | 0.59 | 0.53 | 0.62 | 0.53 |
| One run's 8 features on another run's latents, BBRv3 | 0.88 | 0.68 | 0.64 | 0.56 | 0.68 | 0.69 | 0.56 |
| Base RTT read from the latents, Cubic | -0.14 | -0.01 | 0.01 | 0.33 | 0.39 | 0.69 | 0.86 |
| Queueing delay read from the latents, Cubic | -0.03 | -0.01 | 0.06 | 0.13 | 0.33 | 0.61 | 0.85 |
| Receive rate read from the latents, Cubic | 0.13 | 0.24 | 0.39 | 0.49 | 0.73 | 0.77 | 0.93 |

| After reinforcement learning | 1 | 2 | 64 |
|---|---|---|---|
| Decision from the latents, quadratic (linear at 64), Cubic | 0.87 | 0.77 | 0.97 |
| Decision from the latents, quadratic (linear at 64), BBRv3 | 0.89 | 0.88 | 0.96 |
| Latents from the whole library, Cubic and BBRv3 | 0.97, 0.98 | 0.85, 0.88 | 0.64, 0.64 |
| Latents from the seven hand-picked LCFs | 0.81, 0.80 | 0.63, 0.67 | 0.33, 0.34 |
| Decision from its first 8 features, linear, range over seeds | 0.95 to 0.99 | 0.96 to 0.98 | 0.89 to 0.99 |

What they show:

- The grounding names the action, not a concept behind it. At every
  width and for both teachers, the first feature picked for the latents
  is the sender's own window (`sender_cwnd`). The first feature picked
  for the decision is, in most runs, the flow's packets in its queue plus
  its rate times the base RTT: the flow's packets in flight, which is the
  window again. The library holds quantities that the controller's own
  decision sets a moment earlier, and a fit to the decision finds them
  first. So the high R^2 of the decision is not evidence of concepts.
  Both releases predate the fix: `select` now holds the sender's own
  window in every fit and grounds the change a decision makes
  ([methodology.md](methodology.md#grounding)). The readings with the
  fix are [below](#grounding-beyond-the-senders-own-window).
- A wider middle holds more of the path's state and is harder to span.
  Base RTT, queueing delay and receive rate are read from the latents
  better as the width grows. The library's linear span of the latents
  falls from 0.96 at width 1 to about 0.7 from width 4 on.
- The seven hand-picked LCFs span the latents about half as well as the
  whole library at widths 4 and above.
- The features do not recur across seeds beyond width 1. One run's 8
  features span another run's latents at 0.53 to 0.69 for widths 4 to
  64. Each seed's list of 8 is different in its lower places.
- The sensitivity check holds in the narrow sense. With one base
  variable taken out, the span at 8 features moves by at most 0.02 at
  width 64, and a near equivalent takes the place (for example the
  delivery rate for the window).

What has run: unit tests on synthetic controllers, and one end-to-end
test in the guest, where a width-3 controller cloned from BBRv3 for one
iteration drives its rollouts and both verbs read them. No number from
either is a result.

That end-to-end test also found a fault. A cloned controller with more
than one rollout per link ran the same episode twice under one id, and
one copy failed. The specification now refuses that. The bootstrap uses
one rollout per link and is not affected.

## Grounding beyond the sender's own window

`exp-ground-rl-30` is the first grounding with the sender's own window
held in every fit. It grounds the controllers of `exp-controller-rl-24`
at widths 7, 10 and 16, both teachers, three seeds. These controllers are
untuned and two of the Cubic controllers at width 7 diverged, so the
release checks the method and supports no claim about concepts. All
numbers are R^2 on the validation links.

| Latents, runs of one width pooled | Cubic 7 | Cubic 10 | Cubic 16 | BBRv3 7 | BBRv3 10 | BBRv3 16 |
|---|---|---|---|---|---|---|
| The window alone, linear | 0.51 | 0.46 | 0.51 | 0.63 | 0.57 | 0.62 |
| With the first 8 features of ground truth, linear | 0.68 | 0.64 | 0.69 | 0.72 | 0.69 | 0.69 |
| With the whole library, linear | 0.74 | 0.71 | 0.76 | 0.79 | 0.77 | 0.78 |
| The window alone, a small network | 0.53 | 0.54 | 0.57 | 0.69 | 0.67 | 0.72 |
| With the first 8 features of ground truth, a small network | 0.83 | 0.80 | 0.84 | 0.83 | 0.83 | 0.84 |
| With the whole library, a small network | 0.86 | 0.84 | 0.87 | 0.86 | 0.85 | 0.87 |

| The change a decision makes, range over runs | Cubic | BBRv3 |
|---|---|---|
| The window alone, linear | -0.03 to 0.05 | 0.25 to 0.80 |
| With the whole library, linear | 0.08 to 0.24 | 0.40 to 0.85 |
| The window alone, a small network | -0.01 to 0.11 | 0.36 to 0.88 |
| With the first 8 features and formulas, a small network | 0.18 to 0.30, one run -1.09 | 0.46 to 0.89 |

What it shows:

- The fix does what it was made for. No path holds the window, and the
  flow's packets in flight no longer come first. The features added
  first to the latents are the base RTT, the drop fraction, the ratio of
  RTT to base RTT, the queueing delay on the ACK path and the receive
  rate. The flow's own arrival rate still appears in some paths. With
  the window held it tells the RTT, so the rule allows it.
- The path's state adds to the latents what the window does not. The
  window alone gives 0.53 to 0.72 by a small network, and eight features
  of ground truth raise that to 0.80 to 0.84, within 0.03 to 0.05 of the
  whole library.
- The change a decision makes is mostly unexplained for the controllers
  bootstrapped from Cubic. The window gives almost nothing and the best
  fit reaches 0.18 to 0.30. Its first features are the ratio of RTT to
  base RTT, the queueing delay on the ACK path and the change in the
  drop fraction. The decoder is recurrent, so the rest may be its own
  memory, which no feature of one step holds. That is not checked.
- For the controllers bootstrapped from BBRv3 the window's own history
  explains most of the change (0.58 to 0.80 in all runs but one), and
  the path adds 0.04 to 0.15. Why the two teachers differ is not known.
  One candidate was that the window in force at a decision is not
  always the window the controller last set. The grounding of the
  confirmed controllers rules it out
  ([below](#the-window-between-decisions)).
- The small network on the whole library scores below the one on eight
  features for several Cubic decisions (down to -0.75). It overfits a
  target this noisy, so that bound is not usable for the decision's
  change as it stands.
- Seeds still do not agree on the features. One run's window and first 8
  features span another run's latents at 0.49 to 0.74, and the lists
  differ.

## Grounding the confirmed controllers

`exp-ground-rl-confirm-32` (`experiments/ground-rl-confirm.json`)
grounds the controllers of `exp-controller-rl-confirm-31` at each
teacher's setting, widths 7, 10 and 16, three seeds, and runs the
references without an encoder on the same validation episodes. Every
episode finished. It checks the method and supports no claim about
concepts.

### Against the reference, with the share

Log utility on the 21 validation links under 6 conditions, each
controller on its own rollouts, seeds averaged within a link. Minus the
reference is paired by link, with a bootstrap 95% interval over links
and the verdict by the margin of 0.1
([methodology.md](methodology.md#comparing-controllers)). The share is
the observed flow's part of what all flows took from the bottleneck,
over the conditions with a competing flow
(`experiments/versus_reference.py`).

| Teacher | Width | Without a competing flow, minus the reference | Verdict | With one, minus the reference | Verdict | Share | Share minus the reference |
|---|---|---|---|---|---|---|---|
| Cubic | none | | | | | 0.65 | |
| Cubic | 7 | -0.38 (-0.58 to -0.19) | not comparable | -0.47 (-0.64 to -0.32) | not comparable | 0.50 | -0.14 (-0.19 to -0.09) |
| Cubic | 10 | +0.03 (-0.04 to +0.10) | comparable | -0.05 (-0.15 to +0.06) | unresolved | 0.60 | -0.04 (-0.10 to +0.01) |
| Cubic | 16 | +0.06 (-0.02 to +0.14) | comparable | -0.35 (-0.52 to -0.20) | not comparable | 0.51 | -0.13 (-0.18 to -0.09) |
| BBRv3 | none | | | | | 0.56 | |
| BBRv3 | 7 | -0.03 (-0.09 to +0.02) | comparable | -0.16 (-0.24 to -0.09) | unresolved | 0.53 | -0.04 (-0.06 to -0.01) |
| BBRv3 | 10 | +0.00 (-0.06 to +0.05) | comparable | -0.35 (-0.49 to -0.21) | not comparable | 0.49 | -0.08 (-0.11 to -0.04) |
| BBRv3 | 16 | -0.02 (-0.09 to +0.03) | comparable | -0.32 (-0.44 to -0.21) | not comparable | 0.50 | -0.06 (-0.09 to -0.03) |

What it shows:

- Without a competing flow the latents cost nothing measurable. Every
  width is comparable with its reference but Cubic's width 7, where seed
  1 failed to train ([above](#the-reinforcement-learning-stage)).
- With a competing flow every width takes a smaller share than its
  reference, and the widths with the larger share gaps are the ones
  that are not comparable. The references take 0.56 (BBRv3) and 0.65
  (Cubic) of what all flows took, where an even split over these three
  conditions averages 0.44.
- A higher score from a larger share is not a gain, so these verdicts
  do not show a cost of the latents. A rough reading, the log of the
  ratio of the mean shares, puts 0.15 of BBRv3 width 10's 0.35 and 0.23
  of Cubic width 16's 0.35 in the share. Whether a gap remains at equal
  share is not settled by these numbers.
- These intervals are over links. They agree with the confirmation's,
  which were over seeds.

### The window between decisions

In all 18 controllers the window in force at a decision is the window
the controller set at its previous decision, at every one of 65,000 to
93,000 decisions per controller. The guest reads that window from the
kernel, so nothing changes the sender's window between decisions in
these episodes. That rules out the candidate raised by
`exp-ground-rl-30` ([above](#grounding-beyond-the-senders-own-window)).

### What the latents and the decision hold

R^2 on the validation links.

| Latents, runs of one width pooled | Cubic 7 | Cubic 10 | Cubic 16 | BBRv3 7 | BBRv3 10 | BBRv3 16 |
|---|---|---|---|---|---|---|
| The window alone, linear | 0.48 | 0.43 | 0.42 | 0.65 | 0.60 | 0.65 |
| With the first 8 features of ground truth, linear | 0.65 | 0.65 | 0.66 | 0.78 | 0.76 | 0.74 |
| With the whole library, linear | 0.71 | 0.73 | 0.73 | 0.82 | 0.80 | 0.79 |
| The window alone, a small network | 0.52 | 0.53 | 0.52 | 0.67 | 0.65 | 0.73 |
| With the first 8 features of ground truth, a small network | 0.80 | 0.82 | 0.83 | 0.85 | 0.86 | 0.86 |
| With the whole library, a small network | 0.86 | 0.86 | 0.86 | 0.85 | 0.87 | 0.87 |

| The change a decision makes, range over runs | Cubic | BBRv3 |
|---|---|---|
| The window alone, linear | -0.07 to 0.09 | 0.02 to 0.11 |
| With the whole library, linear | 0.14 to 0.28 | 0.21 to 0.36 |
| The window alone, a small network | 0.04 to 0.19 | 0.03 to 0.14 |
| With the first 8 features and formulas, a small network | 0.18 to 0.34 | 0.17 to 0.37 |

What it shows:

- The latents read as they did before tuning. The window alone gives
  0.52 to 0.73 by a small network, and eight features of ground truth
  raise that to 0.80 to 0.86, within 0.06 of the whole library.
- The first features differ by teacher and agree across widths. From
  Cubic a drop fraction comes first at every width, then, in varying
  order, the random drop fraction, the receive rate, the capacity or the
  BDP and the queueing delay on the ACK path. From BBRv3 the flow's arrival rate
  comes first at every width (with the window held it gives the RTT),
  then the base RTT and the queueing delay on the ACK path.
- The change a decision makes is mostly unexplained for both teachers
  now. The window alone gives 0.11 or less, where it gave BBRv3's
  untuned controllers 0.25 to 0.80, so the difference between the
  teachers seen there does not appear at the confirmed settings. The
  best fit reaches 0.17 to 0.37. The decoder's memory, which no feature
  of one step holds, is the candidate left. The probe fits the change
  from what the decoder sees at the step, its latents and the window in
  force (`experiments/probe-memory.json`, on this release's rollouts).
  Its first run (`exp-probe-memory-33`) could not be read by its rule:
  its quadratic fit is left out at width 16 and was unstable elsewhere
  (down to -1.13). Linearly, the step explains 0.11 to 0.28 of the change
  for BBRv3 and -0.16 to 0.83 for Cubic. The rule's reading, with a
  small network in place of the quadratic fit, is UNRUN.
- A delay signal leads the decision's change in 12 of the 18 runs (4
  from Cubic, 8 from BBRv3): the ratio of RTT to base RTT, the queueing
  delay, or a formula of the base RTT and the queueing delay. Alone it
  explains 0.02 to 0.11.
- Seeds agree a little more. One run's window and first 8 features span
  another run's latents at 0.52 to 0.72 from Cubic and 0.67 to 0.82 from
  BBRv3, against 0.49 to 0.74 before, and the lists still differ.
- The small network on the whole library still overfits the decision's
  change (down to -0.12).

## A stress test of the fitter

`experiments/ground_stress.py` plants latents with a known answer and
puts them through `select`, as if an encoder had learned exactly the
planted concepts. It asks one question: if training found real concepts,
does grounding give them back in closed form? Release
`exp-ground-stress-23` holds the run: Cubic and BBRv3 for 30 s on 96
training links to fit (73,844 decisions) and on the validation links to
score (81,163 decisions), 125 cases. No controller was trained. These are
checks of the fitter, not results about a controller.

Each case is selected twice. With the curated library the planted concept
is in the vocabulary. With the stored labels alone every composite is
taken out, so its formula has to be found, not looked up. "Named" means
that one of the first 8 features orders the rows as the planted concept
does, at a rank correlation of 0.99 or more.

| Planted | Outcome |
|---|---|
| Each of the 12 literature composites alone, held on a log scale, as a power of its value, or through a tanh, without noise (36 cases) | From the stored labels alone the formula search names all 36, the worst at 0.995. The linear fit on labels without formulas names 6. Normalized power has five factors, one more than the search writes, and is named by a shorter formula that orders the rows alike at 0.995. |
| The same with noise of 0.3 standard deviations (36 cases) | 28 named. In 6 of the other 8 a different formula that orders the rows much as the planted one does is listed, at 0.93 to 0.99. In the last 2 the noise left 0.01 or less to explain. |
| expliCCit's seven concepts as a random mix in 7 or 10 latents, with and without a tanh and noise (12 cases) | The first 8 features are enough in all 12: a small network on them reaches what the noise leaves. The planted basis is not recovered. Most of the seven are picked by name, and stand-ins take the place of the rest (base RTT beside queueing delay for one-way delay). The linear fit on places reads 0.92 to 0.95 without noise where the network reads 0.99. |
| Random sets of 2 and 4 ground-truth variables through a mix and a tanh, with noise (12 cases) | Enough in all 12. The first features span every planted variable in 6. |
| One variable under a windowed operator (12 cases) | Named in 9 from the labels. The 3 misses are a windowed composite, which the formula search does not build. |
| Forms the search cannot write: a minimum, a clipped ratio, a weighted sum, a root over a sum (10 cases) | Enough in all 10. No formula is exact, and the report says so by the gap between the linear and the flexible fit. |
| No relation: white noise, and one random number per link | Held-out R^2 is at most 0.01 for the linear fit and below 0 for the flexible one. Nothing is found. |
| One real latent beside one that hardly moves | The mean R^2 halves. `probe` reports each latent's standard deviation so that such a latent is seen. |
| What the sender measures, from ground truth alone | The smoothed RTT is found as base RTT plus queueing delay (R^2 0.996), a formula nobody listed. |

What changed in `select` because of it:

- Every fit is made on places (ranks). A latent that held a variable as
  its value, its inverse or its tanh gave different linear fits before,
  and now gives one. Normal scores in place of ranks were tried and
  dropped: in a local run on the seven-concept mix the linear fit fell
  from 0.92 to 0.78.
- A formula search over the stored labels follows the selection, and the
  path is selected again with its formulas. Before it, a product of two
  labels could not be written at all.
- The search tries every formula of two factors over every stored label.
  When it only followed the selected features it missed excess demand,
  arrival rate less capacity: the difference of two nearly equal
  variables shows in no linear fit on either.
- Features are ordered by backward elimination, and a small network's
  R^2 on the first 8 features and on the whole library stands beside the
  linear numbers. No fixed bar for R^2 is used.

Limits that remain:

- A mix of several concepts is grounded as a set of features that is
  enough, not as the planted basis. Which features a person would call
  the concepts is not decided by the fit. Recurrence across seeds and
  `probe`, which reads each feature from the latents, are the checks for
  that, and neither is in the stress test.
- Under noise the listed formula can be a near equivalent of the planted
  one. Formulas that order the rows alike at 0.98 count as one, and the
  simplest is kept.
- The search writes two terms with coefficient 1 and at most four
  factors. A weighted sum or a minimum is reported as a gap, not written.
- A latent that hardly moves still counts as much as the others. Whether
  trained controllers have such latents is not known.

## Releases

| Release | What it holds | State |
|---|---|---|
| `exp-estimator-seven-20` | The estimator of the seven LCFs: `estimator/source-net1.pt` as copied, and `estimator/net1.pt` converted to the current spec with weights and scalers untouched (a distribution over 64 bins, 3 members, 70 inputs). Hashes are in the release notes. | input of the bootstrap |
| `exp-steps-behaviour-rule-8` | The fixed rule as an ONNX graph. | input of the bootstrap |
| `exp-controller-teachers-22` | The bootstrap from Cubic and BBRv3: 48 run directories with specs, weights, exported policies and histories. | read above; input of the reinforcement-learning stage and of grounding |
| `exp-controller-rl-24` | The reinforcement-learning stage: 48 run directories with specs, weights, exported policies, histories, the width curve and each controller's probe. | read above |
| `exp-ground-stress-23` | The stress test of the fitter: the rollouts' setup, `stress/stress.json` with every case and `summary.md`. | read above |
| `exp-ground-teachers-26` | Grounding of the bootstrapped controllers on their own rollouts: probe and select per teacher, widths 1 to 64, three seeds. |
| `exp-controller-rl-search-29` | The settings search at width 10: 48 run directories with specs and histories, and the ranking of the twelve points per teacher. | read above |
| `exp-controller-rl-confirm-31` | The confirmation of the two best settings per teacher: 48 run directories with specs, weights, exported policies and histories, at widths 0, 7, 10 and 16. | read above; input of grounding |
| `exp-ground-rl-27` | Grounding of the controllers after reinforcement learning at widths 1, 2 and 64: latents and decision. |
| `exp-probe-memory-33` | The probe of the confirmed controllers with the decision's change fitted from the latents and the window in force; its quadratic fit is unreadable. | read above |
| `exp-ground-rl-confirm-32` | Grounding of the confirmed controllers at widths 7, 10 and 16, the references' rollouts on the same validation episodes, each width against its reference with the share, and the window between decisions. | read above |
| `exp-ground-rl-30` | Grounding of the controllers after reinforcement learning at widths 7, 10 and 16, with the sender's own window held in every fit: latents and the change a decision makes. |
| `exp-controller-clone-21` | The start taught from the fixed rule: 24 run directories with specs, weights, exported policies and histories. | read above |
| `exp-steps-broad-30s-9` | Logged steps on a broad set of links, 14,400 episodes, with every label. | old guest; regenerate before use |
| `exp-phase0-teachers-1`, `exp-phase0-steps-2` | Two policy graphs and logged steps, copied in at setup. | old guest; not used by the plan |

Each experiment file's description records where a copied input came
from.

## A fault in earlier episodes

The guest let a stopped TCP cross flow keep sending the bytes its sender
had already queued, and a handshake that waited behind a queue started a
flow later than planned. The number of flows the labels and scores called
active was therefore not always the number sending.

Commit `a26b711a1` fixes it: both ends are aborted at the stop, a late
connection is refused, and each cross flow's real start is recorded
(`cross_connected_ms`). The result version is 6. The commit message has
the measurements.

Every release made before that commit used the old guest. That covers
all step data listed above, in the episodes that have cross flows. Data
for a claim is generated with the fixed guest.

## Withdrawn work

Before the discovery controller existed, the project fitted predictors
to logged steps: one of the flow's own future measurements, and one of
the logging controllers' window changes, each through k latents, with
linear grounding of both. None of them is a controller and none ran in
the closed loop, so none of it is evidence for any claim here. Their
widths, knees and selected features are not used.

The releases remain, each with notes that say what it ran:
`exp-codec-search-3`, `exp-codec-confirm-4`, `exp-codec-variants-5`,
`exp-codec-dropout-6`, `exp-codec-capacity-7`, `exp-codec-knee-11`,
`exp-ground-probe-10`, `exp-ground-select-12`, `exp-ground-knee-13`,
`exp-actions-knee-14`, `exp-student-search-16`, `exp-student-confirm-17`
and `exp-student-ground-18`. Their code and experiment files are removed
from the tree. Each release names the commit it ran at.

What that work left that is still used: the split by link, and the
selection method in `experiments/ground.py`.

Counting concepts by the width of the latents is dropped. One latent
can always carry the decision itself, and the reinforcement-learning
stage showed it in the closed loop
([above](#the-reinforcement-learning-stage)). Changes that would keep
the decision out of the latents were weighed and not built: several
tasks or a preference the encoder does not see only raise the collapse
to as many numbers as there are tasks, and a penalty on the information
in the latents favours it. The count is read in retraining instead
([methodology.md](methodology.md#what-the-width-cannot-count)).

## Open questions

| Question | Note |
|---|---|
| Whether the latents hold more than the decision | Nothing forces them to. The probe reads each base variable from the latents, and the decision is grounded beside them. If the two paths agree and the latents add nothing, the decision's path alone is used. |
| Whether the order of the decision's path is the order of need | It comes from logged steps. Retraining at each m, against a random order and the baselines, decides. |
| How far reinforcement learning moves a controller from its teacher | If it barely moves, the concepts are the teacher's. The measure is the RMS difference in log2 of the window from the clone. How small counts as barely is not decided. |
| How to compare controllers with a competing flow when their shares differ | With a competing flow the references take a larger share than every width, and a higher score from a larger share is not a gain. A comparison at equal share, or a score that does not reward the share, is not decided. |
| The threshold for C3 | |
| How close to the flexible fit a formula must come | C2 compares a formula with a flexible fit on the whole library. The margin is not decided. |
| How to weigh a latent that hardly moves | Every fit standardizes the latents. A weight from the decoder's use of each latent is a candidate. |
| The kind of output the retrained net1 states | A value, an interval or a distribution. |
| Labels for fitted formulas | A formula over stored labels can be added as a composite with no new data and no guest change ([architecture.md](architecture.md#composites)). A formula that uses a windowed operator over past steps, as the library's features do, has no such path yet. |

## Checks

On the current code: ruff, pyright and `uv run pytest -q` (225 passed, 61
skipped) pass. With the guest booted locally, the 24 tests of
`tests/test_e2e.py` pass: the end-to-end pipeline for every net1 kind and
view, grounding included. The other tests that boot the guest are UNRUN
locally on this code. CI runs them, and its `ci` and `uml` workflows
passed on the commit of the reinforcement-learning stage, `7d24b9ac2`.
