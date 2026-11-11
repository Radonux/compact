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
| 2 The codec on logged steps | search done (`exp-codec-search-3`); confirmation over seeds dispatched (`experiments/codec-confirm.json`), no result yet | `experiments/codec.py`, its positive control (`tests/test_codec.py`), the link split | |
| 3 Strengthen the codec | local probes only | the `frames` and `targets` options of the codec | the information-bottleneck term; the run on the test links |
| 4 Grounding | UNRUN | the labels and the step message in `steps.parquet`; the codec's latents | the library, the fits, the stability and sensitivity checks |
| 5 Control | UNRUN | a recurrent actor on raw measurements (`train ppo`, `inputs: "raw"`), trained by PPO, TD3+BC or DAgger; closed-loop rollouts and scoring | a controller that reads the codec's latents or fitted concepts; DAgger from a teacher given as a graph; the bottlenecked student in the closed loop |
| 6 Retrain and compare | UNRUN | `train run` for net1 and an offline net2; `train ppo` for a closed-loop net2; oracles on the seven LCFs | labels from fitted formulas; net1 on them; the live truth for them |

No claim (C1 to C5) has evidence from a release.

## Open items

- **Targets for a value head** of a distilled student (step 5). The
  teacher graphs are policies and come without a critic, and a rule has
  none at all. Q-values on a grid of actions need a critic fitted to
  each teacher first.
- **Missing base variables.** The emulator does not label the arrival
  rate of all flows, the flow's own arrival rate or the number of active
  flows ([concepts.md](concepts.md#what-this-means-for-compact)). Without
  them grounding cannot return a concept about demand or about the other
  flows. They should be added before step 4, and data used for grounding
  must be generated after that.
- **The scale of retained performance** ([methodology.md](methodology.md#retained-performance)).
- **The threshold for C3**, and the choice rule for the knee in C1.
- **The links, conditions and episode lengths** of the closed-loop
  evaluation, to be fixed in step 5's experiment file.
- **The kind of output** the retrained net1 states (value, interval or
  distribution).
- **Live truth for new concepts.** The guest gives a policy the true
  values of the seven LCFs only, and labels are computed when an episode
  is processed ([architecture.md](architecture.md#known-limits)). The
  oracle net2 of step 6 and labels for fitted formulas need both
  extended.

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
