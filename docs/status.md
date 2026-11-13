# Status

Everything that changes with time: what exists, what has run, what it
showed and what is open. This page is rewritten in place and holds no
dated entries; its history is in git. The steps are those of the
[plan](methodology.md#plan).

## Where the project stands

No claim has evidence yet. Nothing has finished in the closed loop.

The discovery controller is built and unit-tested. Its first closed-loop
stage, the taught start, is running
(`experiments/controller-clone.json`, workflow run 21, dispatched
2026-11-13).

| Step | State | Built | Not built |
|---|---|---|---|
| 1 Setup | done | the documents; the estimator of the seven LCFs and the rule graph as releases | |
| 2 The controller in the closed loop | the taught start is running | `PPOSpec.latents`; `train ppo` by DAgger, PPO or TD3+BC; closed-loop rollouts and scoring; `experiments/net2_ppo_setup.py`, `net2_ppo_report.py` | the reinforcement-learning stage; the collapse check on rollouts; a start that does not come from the rule |
| 3 Grounding | UNRUN | `experiments/ground.py probe` and `select`, with the sensitivity check; 19 ground-truth labels in the step tables | grounding on a controller's own rollouts; formulas beyond a linear fit; the operator-family sensitivity |
| 4 Retrain and compare | UNRUN | `train run` for net1; `train ppo` for a closed-loop net2; oracles on the seven LCFs | labels from fitted formulas; net1 on them; the live truth for them |

## The taught start

What it trains: the discovery controller at widths 1, 2, 4, 7, 10, 16 and
64, three seeds each, by DAgger. The controller acts in every rollout
and each decision is labelled with the fixed rule's window. A reference
without an encoder (k 0 in the experiment file) is the recurrent network
on the raw measurements, its window on a measured bandwidth-delay
product.

Why these widths: 7 is expliCCit's budget, 10 was the owner's first
proposal, and 64 stands for no constraint. Widths 1 and 2 are there to
show action collapse.

What will be read from it: how far each start is from the rule, in log
utility on 21 validation links under 6 conditions, and whether width 1
already matches width 64. Nothing is chosen from this run, and its width
curve is not a count, because the rule's window is one number. Every
width goes on to reinforcement learning.

Two caveats on its inputs. The estimator and the rule were fitted on
episodes from the guest before the cross-flow fix (below). Every rollout
and score of the run comes from the fixed guest, the rule included, so
the comparison with the rule is fair, but the rule may be weaker than
one fitted on clean episodes. And the exported graph of the narrow
controller has never run in the guest before this run.

## Releases

| Release | What it holds | State |
|---|---|---|
| `exp-estimator-seven-20` | The estimator of the seven LCFs: `estimator/source-net1.pt` as copied, and `estimator/net1.pt` converted to the current spec with weights and scalers untouched (a distribution over 64 bins, 3 members, 70 inputs). Hashes are in the release notes. | input of the taught start |
| `exp-steps-behaviour-rule-8` | The fixed rule as an ONNX graph. | input of the taught start |
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
and `exp-student-ground-18`. So does their code (`experiments/codec.py`,
`codec_report.py`, `actions.py`, `tests/test_codec.py`), as the record
behind those releases.

What that work left that is still used: the split by link,
`experiments/ground.py`, and three ground-truth labels
(`arrival_rate_bps`, `flow_arrival_rate_bps`, `active_flows`).

## Open questions

| Question | Note |
|---|---|
| What keeps the decision out of the latents | The first failure in [methodology.md](methodology.md#how-it-can-fail). An untuned prototype on logged states, made before this repository and never released, saw it: with only the action to fit, one latent was enough. The taught start will show whether it happens in the closed loop. What to change if it does is not decided. |
| The knee rule for the width curve | To be fixed in the experiment file of the reinforcement-learning stage. |
| The scale of retained performance | [methodology.md](methodology.md#retained-performance). |
| The threshold for C3 | |
| The kind of output the retrained net1 states | A value, an interval or a distribution. |
| Live truth for new concepts | The guest gives a policy the true values of the seven LCFs only, and labels are computed when an episode is processed ([architecture.md](architecture.md#known-limits)). The oracle net2 and labels for fitted formulas need both extended. |

## Checks

On the current code: ruff, pyright, `uv run pytest -q` (213 passed, 58
skipped) and the guest's `cargo fmt` and clippy pass. UNRUN locally: the
tests that boot the guest. CI and every experiment run them.
