# Status

Everything that changes with time: what exists, what has run, what it
showed and what is open. This page is rewritten in place and holds no
dated entries; its history is in git. The steps are those of the
[plan](methodology.md#plan).

## Where the project stands

No claim has evidence yet. One closed-loop run has finished: the start
taught from the fixed rule, which is a check and not a result for a claim
([below](#the-start-taught-from-the-fixed-rule)).

The discovery controller is built and unit-tested. Its bootstrap from
Cubic and from BBRv3 is running (`experiments/controller-teachers.json`,
dispatched 2026-11-13). The reinforcement-learning stage and the
grounding of a controller's latents on its own rollouts are built and
tested. Both wait for that release.

| Step | State | Built | Not built |
|---|---|---|---|
| 1 Setup | done | the documents; the estimator of the seven LCFs and the rule graph as releases | |
| 2 The controller in the closed loop | the bootstrap is running | `PPOSpec.latents`; `train ppo` by cloning a Linux controller, PPO, DAgger or TD3+BC; closed-loop rollouts and scoring; `experiments/net2_ppo_setup.py`, `net2_ppo_report.py`; the collapse check on rollouts (`ground.py probe`); the reinforcement-learning stage (`experiments/controller-rl.json`) and its width curve with the knee rule (`experiments/width_curve.py`) | the search over the stage's settings; the teachers' own scores on the validation links |
| 3 Grounding | UNRUN | `experiments/ground.py`: `rollouts` of a controller in the closed loop, `probe` and `select` on its latents there, with the sensitivity check; a library of every stored label (37 columns over 100 ms and 10 ms windows) and 12 composites in both windows; `experiments/ground-teachers.json` | formulas beyond a linear fit; the operator-family sensitivity; matching concepts across seeds |
| 4 Retrain and compare | UNRUN | `train run` for net1; `train ppo` for a closed-loop net2; net1 on any stored label or composite; the live truth for every label but `queue_delay_ns`, so oracles on them | labels from fitted formulas beyond the composites; net1 on them |

## The bootstrap

What it trains: the discovery controller at widths 1, 2, 4, 7, 10, 16 and
64, three seeds each, once from Cubic and once from BBRv3. The teacher
drives every training rollout, and the controller learns the window that
sends as the teacher did. A reference without an encoder (k 0 in the
experiment file) is the recurrent network on the raw measurements, its
window on a measured bandwidth-delay product.

Why these widths: 7 is expliCCit's budget, 10 was the owner's first
proposal, and 64 stands for no constraint. Widths 1 and 2 are there to
show action collapse.

What will be read from it: how each start drives in the closed loop, in
log utility on 21 validation links under 6 conditions against the fixed
rule, and whether width 1 already matches width 64. Nothing is chosen
from this run, and its width curve is not a count, because a teacher's
window is one number. Every width goes on to reinforcement learning.

What to keep in mind when reading it:

- The exported graph of the narrow controller, and the cloning code, have
  not run in the guest before this run.
- The label for BBRv3 is its pacing rate times its smoothed RTT, capped
  by its window. That reproduces its sending rate under the `agent`
  controller, not its internal state machine.
- The MSS in that conversion is a constant, 1448 bytes. It is UNRUN
  against a recorded `TCP_INFO`.
- The estimator of the seven LCFs serves only as the measurement scaler
  and as the rule the evaluations compare with. That estimator and rule
  were fitted on episodes from the guest before the cross-flow fix
  (below).
- The teachers' own scores on the validation links are not in the report
  yet, so the report says how a start compares with the rule and not how
  close it is to its teacher in the closed loop.

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

`experiments/controller-rl.json` continues every bootstrapped controller
by PPO on the log utility, from the clone's actor and value network. It
is UNRUN: it reads the bootstrap's release.

Its rules are fixed in the file before it runs. A run's score is that of
the policy it leaves. The knee is the smallest width within one pooled
standard error of the best width. Each controller with latents then
drives rollouts of its own, and its decision is regressed on its latents.
The distance from the clone is the RMS difference in log2 of the window.

Its settings are one point, taken from PPO runs made before this
repository, and are not searched. So its knee will not be a count of
concepts. It will show whether PPO improves on a clone, whether the
widths separate, and what the search should vary.

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

`experiments/ground-teachers.json` does this for the bootstrapped
controllers: both teachers, widths 1 to 64, three seeds. It is UNRUN. It
reads the bootstrap's release, which does not exist yet. It will test the
pipeline on real controllers. It supports no claim about concepts,
because those controllers are untuned starts.

What has run: unit tests on synthetic controllers, and one end-to-end
test in the guest, where a width-3 controller cloned from BBRv3 for one
iteration drives its rollouts and both verbs read them. No number from
either is a result.

That end-to-end test also found a fault. A cloned controller with more
than one rollout per link ran the same episode twice under one id, and
one copy failed. The specification now refuses that. The bootstrap uses
one rollout per link and is not affected.

## A stress test of the fitter

Latents with a known answer were planted and put through `select`. The
rows were 150 episodes run locally under Cubic, BBRv3 and the fixed rule
(90 training links to fit, 60 validation links to score). Each latent was
a formula over ground truth, some with noise added. This was a local
check. Its script is not in the repository, and a released run is UNRUN.

| Planted | Outcome |
|---|---|
| Products of powers, alone or as an invertible mix of three, with and without noise | The right features come first, and R^2 reaches the limit the noise sets. Mixing costs nothing, so the latents need no rotation before the fit. |
| Other forms: a tanh of a variable, the smaller of two, a ratio of sums, products of plain values | In five of seven cases the linear R^2 stops between 0.68 and 0.95. A small network on the first 8 features reaches 0.98 to 1.00 in all seven, so the features were right and the form was not. |
| Four variables through a random mix and a tanh, with noise | Forward selection picks a stand-in for the mix first and keeps it: 0.69 at 4 features. Ordering by backward elimination puts the four planted variables first: 0.77. |
| No relation: one random number per episode, and white noise | Held-out R^2 is at or below 0. Nothing is found. |
| One real latent beside one that hardly moves | The mean R^2 halves, and fits to the noise take the picks after the first. |

Three changes followed. `select` orders its features by backward
elimination. It reports a small network's R^2 on the first 8 features and
on the whole library beside the linear numbers. `probe` reports each
latent's standard deviation. No fixed bar for R^2 is used any more: a
latent computed from the sender's measurements can hold less than all of
a ground-truth variable, and the flexible fit on the whole library says
how much there is to explain.

Two limits remain. The selection criterion is linear, so a variable that
acts only through a product with another is picked late: in one planted
case it came seventh. And a latent that hardly moves still counts as much
as the others. Whether trained controllers have such latents is not known.

## Releases

| Release | What it holds | State |
|---|---|---|
| `exp-estimator-seven-20` | The estimator of the seven LCFs: `estimator/source-net1.pt` as copied, and `estimator/net1.pt` converted to the current spec with weights and scalers untouched (a distribution over 64 bins, 3 members, 70 inputs). Hashes are in the release notes. | input of the bootstrap |
| `exp-steps-behaviour-rule-8` | The fixed rule as an ONNX graph. | input of the bootstrap |
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

## Open questions

| Question | Note |
|---|---|
| What keeps the decision out of the latents | The first failure in [methodology.md](methodology.md#how-it-can-fail). An untuned prototype on logged states, made before this repository and never released, saw it: with only the action to fit, one latent was enough. The bootstrap will show whether it happens in the closed loop. What to change if it does is not decided. |
| How far reinforcement learning moves a controller from its teacher | If it barely moves, the concepts are the teacher's. The measure is the RMS difference in log2 of the window from the clone. How small counts as barely is not decided. |
| Whether one standard error is the right knee rule | With 3 seeds the error is itself noisy. It is pooled over widths for that reason. |
| What the search over the stage's settings covers | Candidates: exploration noise, learning rate, an anchor to the clone, the value warm-up. |
| The scale of retained performance | [methodology.md](methodology.md#retained-performance). |
| The threshold for C3 | |
| How close to the flexible fit a formula must come | C2 compares a formula with a flexible fit on the whole library. The margin is not decided. |
| How to weigh a latent that hardly moves | Every fit standardizes the latents. A weight from the decoder's use of each latent is a candidate. |
| The kind of output the retrained net1 states | A value, an interval or a distribution. |
| Labels for fitted formulas | A formula over stored labels can be added as a composite with no new data and no guest change ([architecture.md](architecture.md#composites)). A formula that uses a windowed operator over past steps, as the library's features do, has no such path yet. |

## Checks

On the current code: ruff, pyright and `uv run pytest -q` (224 passed, 61
skipped) pass. With the guest booted locally, the 24 tests of
`tests/test_e2e.py` pass: the end-to-end pipeline for every net1 kind and
view, grounding included. The other tests that boot the guest are UNRUN
locally on this code. CI runs them, and its `ci` and `uml` workflows
passed on the commit of the reinforcement-learning stage, `7d24b9ac2`.
