# Log

What has run, what it showed, and what is open. Everything that changes
with time lives here; the other documents state only what stays true.
Entries are dated, newest first. Each entry names its release, what was
searched, the held-out numbers, and what remains UNRUN. The entries of a
direction that was withdrawn are cut down to what they were and which
releases they left; their full text is in the history of this file up
to commit `c72e2f746`.

## State of the plan

Updated with each entry. The steps are those of the
[plan](methodology.md#plan).

| Step | State | Can reuse | Not built |
|---|---|---|---|
| 1 Setup | done (2026-11-11); the estimator of the seven hand-picked LCFs is release `exp-estimator-seven-20` (2026-11-13) | | |
| 2 The controller in the closed loop | the controller with a narrow middle is built and unit-tested (2026-11-13); its taught start (`experiments/controller-clone.json`) is UNRUN | `PPOSpec.latents` (`train/recurrent.py`); `train ppo` by DAgger, PPO or TD3+BC; closed-loop rollouts and scoring; `experiments/net2_ppo_setup.py` and `net2_ppo_report.py` | reinforcement learning after the taught start; the width curve; the check for action collapse on rollouts; a start that does not come from the rule |
| 3 Grounding | UNRUN on a controller | `experiments/ground.py probe` and `select`, with the vocabulary sensitivity check; the labels in `steps.parquet` | grounding on a controller's own rollouts; formulas beyond a linear fit; the operator-family sensitivity |
| 4 Retrain and compare | UNRUN | `train run` for net1 and an offline net2; `train ppo` for a closed-loop net2; oracles on the seven LCFs | labels from fitted formulas; net1 on them; the live truth for them |

No claim has evidence yet. Nothing has run in the closed loop.

## Open items

- **What keeps the decision out of the latents.** The encoder can
  compute a window and pass it through one latent
  ([methodology.md](methodology.md#the-controller)). The prototype below
  saw exactly this on logged states: with only the action to fit, one
  latent was enough. The taught start will show whether it happens in
  the closed loop; what to change if it does is not decided.
- **The scale of retained performance** ([methodology.md](methodology.md#retained-performance)).
- **The threshold for C3.**
- **The knee rule** for the controller's width curve, to be fixed in the
  experiment file of the run that reads it.
- **The kind of output** the retrained net1 states (value, interval or
  distribution).
- **Live truth for new concepts.** The guest gives a policy the true
  values of the seven LCFs only, and labels are computed when an episode
  is processed ([architecture.md](architecture.md#known-limits)). The
  oracle net2 of step 4 and labels for fitted formulas need both
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
  `exp-steps-broad-30s-9` and `exp-phase0-steps-2`, and so every release made from them, in the episodes that have cross
  flows. That work is withdrawn (entry of 2026-11-11 to
  2026-11-13 below) and none of it is rerun. Data for a claim is
  generated with the fixed guest.
- **The start of the closed-loop run.** The seven-LCF estimator and the
  rule built on it were fitted on episodes of the old guest. They serve
  only as the teacher of the taught start (`experiments/controller-clone.json`);
  every rollout and every score of that run comes from the fixed guest,
  the rule included, so the comparison with the rule is fair, but the
  rule may be weaker than one fitted on clean episodes.
- **Checked.** Format, lint, types, the unit tests (213 passed, 58
  skipped) and the guest's clippy. UNRUN here: the tests that boot the
  guest; the experiment workflow runs the guest itself.

## 2026-11-13: the discovery model is a controller

- **What the project trains.** A controller of expliCCit's shape with
  the middle left free: an encoder in net1's place to k latents, a
  recurrent decoder in net2's place to the window, trained in the closed
  loop ([methodology.md](methodology.md#the-controller)). This is what
  the project's owner asked for at the start. The two days before this
  entry went to something else (entry below).
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
  - Widths 1, 2, 4, 7, 10, 16 and 64. 7 is expliCCit's budget, 10 the
    owner's first proposal, 64 stands for no constraint and is the
    unconstrained teacher. 1 and 2 are there to show action collapse:
    if one latent does as well as 64, the latent is the decision and
    the curve counts nothing.
  - The first stage teaches each controller a fixed rule's window by
    DAgger, as a start only. The rule reads the estimator of the seven
    hand-picked LCFs, and its window is one number (2^0.25 times the
    believed bandwidth-delay product), so this stage is the easiest
    case for collapse and its width curve is not read as a count. The
    claim will rest on the reinforcement-learning stage, and a run from
    another start is the check (C5).
  - A reference without an encoder (k 0 in the experiment file): the
    recurrent network on the measurements with its window on a measured
    bandwidth-delay product.
- **Inputs.** Release
  [`exp-estimator-seven-20`](https://github.com/Radonux/compact/releases/tag/exp-estimator-seven-20)
  (`experiments/estimator-seven.json`): the estimator of the seven
  hand-picked LCFs, copied as `estimator/source-net1.pt` and converted
  to the current spec as `estimator/net1.pt` (weights and scalers
  untouched; a distribution over 64 bins, 3 members, 70 measurements
  read; both hashes are in the release notes). Run 19 of the workflow
  failed on the old spec and published nothing. The rule graph of
  `exp-steps-behaviour-rule-8` supplies the training paths' logging
  controller.
- **UNRUN.** Everything in the closed loop: the taught start
  (`experiments/controller-clone.json`), reinforcement learning, the
  width curve, grounding on a controller's rollouts, the start that does
  not come from the rule.

## 2026-11-11 to 2026-11-13: work on logged steps, withdrawn

On 2026-11-11 the discovery model was changed from a controller to
predictors fitted on logged steps, and the next two days built on that.
It was a wrong reading of the goal. The goal is what a trained
controller uses, judged in the closed loop. None of the models below is
a controller, none ran in the closed loop, and no number from them
supports a claim of this project. Their widths, knees and picked
features are not carried forward.

- **What they were.**
  - A model that predicts the flow's own measurements 60 ms to 960 ms
    ahead through k latents (`experiments/codec.py`).
  - A model that predicts, through k latents, how the logging
    controllers (17 kernel algorithms and 5 learned policies) change
    their windows (`codec.py` with `targets=action`).
  - A check of how well sets of inputs predict those window changes
    (`experiments/actions.py`).
  - Linear grounding of both models' latents (`experiments/ground.py`).
- **Releases they left**, all on validation links, the test links never
  read: `exp-codec-search-3`, `exp-codec-confirm-4`,
  `exp-codec-variants-5`, `exp-codec-dropout-6`, `exp-codec-capacity-7`,
  `exp-codec-knee-11`, `exp-ground-probe-10`, `exp-ground-select-12`,
  `exp-ground-knee-13`, `exp-actions-knee-14`, `exp-student-search-16`,
  `exp-student-confirm-17`, `exp-student-ground-18`. Each release's
  notes say what it ran. Their episodes come from the guest before the
  fix in the entry above.
- **What carries over**, as tools and not as results.
  - The split by link (70/15/15, `expliccit_env.train.data.link_key`).
  - `experiments/ground.py probe` and `select`, with the vocabulary
    sensitivity check and the path on ground truth alone.
  - Three more ground-truth labels, `arrival_rate_bps`,
    `flow_arrival_rate_bps` and `active_flows`
    ([architecture.md](architecture.md#ground-truth-labels)), with
    their tests.
  - Step data with those labels: release `exp-steps-broad-30s-9`
    (`experiments/steps-broad-30s.json`, 14,400 episodes), driven in
    part by the rule of release `exp-steps-behaviour-rule-8`. It was
    made by the guest before the fix, so it is regenerated before it is
    used for anything.
- **Two cautions about grounding** that these runs raised and that
  apply to any latents. A library that holds the sender's own readings
  lets them stand in for the path quantities they measure, so the path
  on ground truth alone is the one that names concepts. And the
  features that best span a set of latents were not the ones that best
  predicted actions, so how much of the latents a feature spans is not
  its importance.
- **Not removed.** `experiments/codec.py`, `codec_report.py`,
  `actions.py`, their experiment files and `tests/test_codec.py` stay
  in the repository as the code behind the releases above.

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
(`build.py`, `student.py`, `ground.py`) are not in the repository.

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
