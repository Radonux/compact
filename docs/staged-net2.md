# Taught, then RL: the staged net2

This is the design for moving palaestra to the next architecture, and how
much of it the code already holds. In that architecture, net1 reports a
distribution for each LCF. net2 is first taught by imitating a teacher,
then improved by RL on the reward. The thesis does not change: net2 acts
on net1's output alone, and net1 learns only from ground truth. What
changes is how net2 is trained. Status: the code paths below exist and
are tested. No experiment has run the staged pipeline yet (UNRUN).

## net1

A distribution net1 exists: `BinNet1` (`train/models.py`), 64 bins per
LCF, trained with the log score, optionally an ensemble
(`Net1Spec.bins`, `members`). The rest of the system reads net1's output
through `train/beliefs.py`, which states for each kind (value, interval,
distribution) its means, its patch with the truth and its faults, so
every net2, rule and window target works on every kind, and
`tests/test_e2e.py` checks each pairing end to end. net1 v6 is the current one. A binary LCF,
such as the two flags in expliCCit's 9-LCF list, already fits it. A label
within [0, 1] gets equal bins, so a 0/1 flag puts all its mass in the
first or the last bin, and the distribution is a Bernoulli. Only the
flags' definitions are missing; they wait on the team. A new net1 means
a new net2: net2 reads net1's bins, so a stage continued on a different
net1 is refused (below).

## Why teach first

Three facts about this problem decide the design.

1. **Closed-loop RL is slow here.** Every PPO iteration runs UML rollouts:
   8 links times 4 rollouts, 80 to 120 iterations per run. A return
   averages the reward over many decisions, so each rollout says little
   about any one decision.
2. **RL loses a good start.** On the true LCFs, PPO started 0.040 above
   net1 v6's fixed rule and ended 0.015 to 0.034 below it
   ([`exp-net2-r19-blackbox-164`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19-blackbox-164),
   [results.md](results.md)). Rounds 10 to 17 drifted toward pushing from
   the rule's start in the same way. A policy that starts well needs
   protection until the value network can tell what improves on it.
3. **Imitation alone stops short.** A teacher with privileged information
   (the true LCFs, or the flows sharing the link) is cheap to imitate:
   DAgger gives a label at every decision, on the student's own states, so
   errors do not compound (Ross, Gordon and Bagnell, 2011). But a student
   that cannot see what the teacher sees learns the teacher's action
   averaged over the states it cannot tell apart. Weihs et al. (NeurIPS
   2021, "Bridging the imitation gap by adaptive insubordination") call
   this the imitation gap. Warrington et al. (ICML 2021, "Robust asymmetric
   learning in POMDPs") show that imitating a fixed privileged expert can
   be far from the best policy the agent can reach. Under net1's
   uncertainty, that average is not the best action. A teacher that knows
   the capacity never needs to probe; a student unsure of it might. Only
   the reward can teach the value of information, and that is where
   net1's distribution should pay (claim H5, arms A3 against A2).

So: teach to get a good start cheaply, then let RL correct the imitation
gap, without losing the start.

## The invariants of the handoff

The handoff takes no conversion step: a stage's output is the next
stage's input. Five invariants make that hold, in `train/actors.py`, `learners.py` and `handoff.py`.

| Invariant | How it holds |
|---|---|
| One actor, one head | Every learner (`dagger`, `ppo`, `td3bc`) trains the same `RecurrentActor`, and `policy_u` is the only map from its output to u, in training and in the exported graph. A taught student of any kind (`u`, `gate`, `gate3`) is the RL actor as it stands, and PPO's Gaussian is over the u the deployed graph sets. Tested: the continued policy exports the same ONNX bytes as the taught one. |
| A warm value network | A `dagger` run now trains a value network as PPO does: the same inputs (`critic_truth`, `critic_flows`), the same reward and GAE targets, on the student's own rollouts. It saves it as `critic.pt`, and PPO loads it, so PPO does not start from a blank critic. |
| A protected start | `anchor` adds anchor times KL(start, policy), (mu - mu_start)^2 / (2 sigma^2) per decision, to PPO's loss. `kl_to_start` is logged every iteration, anchored or not. `critic_warmup` still holds the actor while the critic learns. |
| A checked handoff | `init_from` names the earlier run's directory. Its net1 must be tensor-equal to this run's. Its actor fields (view, input_bins, hidden, memory, student, base, inputs, action_max) must match, or the run refuses to start. Its value network is loaded only when it reads the same inputs. The report's `started_from` names what was loaded. |
| Pluggable teachers, one label frame | A teacher labels each decision with the window it would set; the student learns that window as u on its own believed BDP (or, for the black box, its measured BDP). The teachers are the rules on the true flows (`share_k`, a Gate3's `copies`) and, with `teacher_from`, any actor trained on the true LCFs (`inputs` "truth"). That teacher reads the truth's bins and the window the student actually set. |

## The recipe

Stage 1 teaches, and stage 2 continues it by PPO. Both run in one
experiment step, so the handoff needs no release in between:

    jq '.algo = "dagger" | .student = "gate3" | .iterations = 40' ppo-base.json > teach.json
    uv run expliccit train ppo teach.json out/teach --build build/uml --workers 4
    jq --arg from out/teach '.algo = "ppo" | .student = "gate3" | .init_from = $from
        | .anchor = 1 | .critic_warmup = 0' ppo-base.json > rl.json
    uv run expliccit train ppo rl.json out/rl --build build/uml --workers 4

For a teacher on the true LCFs, first train (or take) a run with
`inputs` "truth" (round 19's A1 runs are such runs), then set
`teacher_from` to its directory in the stage 1 spec, with `student` "u".
A student taught this way learns to act as if it knew the path; stage 2
is then where it can learn what net1's uncertainty is worth.

## What is ready, what is not

Ready and tested (`tests/test_ppo.py`):

- a taught `gate3` student continued by PPO computes the same u and exports
  the same graph, and PPO then moves it with the advantage;
- DAgger's value network learns the student's return (explained variance
  above 0.5 on a synthetic reward), and PPO loads it;
- the anchor keeps PPO within half the KL it moves without one;
- `init_from` refuses another net1 or another actor shape;
- a policy teacher's label is its window on the student's BDP, for net1
  and black-box students;
- in UML, a `dagger` run hands over to a `ppo` run, whose first policy is
  the taught one byte for byte (run on a local UML build of 30caf2e).

UNRUN, and the first experiments to run, on validation paths only and
under the tuning rules in [CLAUDE.md](../CLAUDE.md):

1. r18e05 (the round 18 `gate3` student that passed its gate) continued by
   PPO, anchor 0, 1 and 5, three seeds, on round 23's information-rich
   mixture. Does RL improve a taught start without losing it? Its run
   directories (seeds 1 to 3, `ppo-r18-k0.5-sN.tar`) are in
   `model-v6-inputs`. They predate DAgger's value network, so PPO starts
   with a blank critic there and needs `critic_warmup`.
2. A `u` student taught by round 19's A1 actor (`teacher_from`, run
   directories `ppo-r19-truth-sN.tar` in `model-v6-inputs`), against the same student
   taught by the rules, then each continued by PPO, on bins (A3) and on
   means (A2). Does a better teacher give a better start, and does RL from
   it use net1's spread where the means cannot?

Not built, because nothing needs it yet:

- Offline teaching from logged step data into the recurrent actor. The
  offline trainer (`train/fit.py`: `bc`, `td3bc`, `iql`, expliCCit's CQL
  objective) trains a memoryless MLP on the window multiplier, which
  cannot hand over to the recurrent actor. If the team's design teaches
  from logs, the change is behaviour cloning of `RecurrentActor` on
  logged decision sequences (the first DAgger iteration, on logs): about
  a day of work.
- DAgger's mixing of teacher and student rollouts (DAgger's beta). The
  student starts as the fixed rule, close enough to the teacher's
  states that none was needed so far.

## Real paths

The eight Ubuntu hosts (24.04 to 26.04) can run what UML cannot: real
cross traffic, real queues and real paths (R12.3). They cannot give
ground truth, so on them net1 is checked only by the consistency checks
(R2, `experiments/r2_alarms.py`), and controllers are compared by what
the sender and receiver measure. The plan:

1. **What runs unchanged.** The guest runner's step featurizer, policy loop
   (tract on the exported ONNX) and recorder (`guest/src`), and the modules
   `expl_rec`, `expl_tcpobs` and `tcp_agent`. `sch_mmlink` and
   `sch_flowdelay` are not needed: the path is real.
2. **What must change.** A host mode in the runner: no namespaces or veth
   pairs, a remote receiver's address, and wall-clock steps (10 ms) in
   place of UML's virtual time. The receiver box runs a sink. The episode
   record must mark the episode as real, so no step gets labels. A
   failed episode yields nothing, as in UML.
3. **Kernels.** Ubuntu 26.04 ships Linux 7.0, the version palaestra's UML
   builds on, so the modules should build against its headers unchanged
   (UNVERIFIED until built there). Ubuntu 24.04 ships Linux 6.8: check
   each `tcp_congestion_ops` callback and tracepoint `tcp_agent` and
   `expl_tcpobs` use against 6.8's before trusting a run. BBRv3 is a
   patched TCP core, not a loadable module, so on stock kernels only the
   stock controllers (Cubic, BBRv1, Reno, Vegas and the rest) are
   baselines; BBRv3 needs Google's kernel on the boxes.
4. **The protocol.** For each ordered pair of boxes, run the controllers
   back to back in a random order, repeat over days, and compare each one
   with the others run in the same time slot (paired by slot, as the UML
   evaluation pairs by link). Report throughput, RTT and loss per pair,
   and R2's alarm rate for each policy's net1.
5. **Results.** Each box pushes its records with `scripts/stash.sh`, so
   every real-path result becomes a release like the UML ones.

Effort: the host mode and the sink, about two days; building and checking
the modules on both Ubuntu kernels, about one day once the boxes are
reachable from where they are built. Nothing here has run (UNRUN).
