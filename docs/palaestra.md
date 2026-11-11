# palaestra's claims and their status at the fork

compact is a fork of [Radonux/palaestra](https://github.com/Radonux/palaestra),
taken after commit `4d576dfa7`. This page keeps palaestra's own research
question, claims, arms and status as they stood then, moved here unchanged
from `methodology.md`. compact's question and claims are in
[methodology.md](methodology.md); it does not pursue H1 to H6 further. The
terms and the evaluation protocol the tables below use are still in force
and stay in [methodology.md](methodology.md#inherited-terms).

Every release cited with a `Radonux/palaestra` URL was published there
before the fork. Nothing on this page was rerun in compact.

palaestra's question: can a learned congestion controller be split into an
estimator of the path's physical state (net1, trained against ground
truth) and a decision maker that sees only net1's output (net2), at small
cost, and does the split make the controller explainable and repairable?
The idea is expliCCit's. The work had two phases: a reproduction of
expliCCit's released models and setup, faithful to the original, and then
research in which only the split was fixed.

## Claims

| | Claim | How it can fail |
|---|---|---|
| H1 | Sufficiency: a policy on the true LCFs does about as well as the best policy we can train on raw measurements | the LCFs omit something the decision needs |
| H2 | Estimability: net1 recovers the LCFs from sender-side measurements, where the measurements allow it | some LCFs are not identifiable at some operating points |
| H3 | Honesty: net1's stated confidence matches its actual error | over-confidence, above all where the link is idle |
| H4 | No leakage: net1's output carries only its beliefs, and net2 acts on nothing else | soft outputs pass on information beyond the LCFs |
| H5 | Repairability: a fault is traced to one LCF and fixed there, without retraining net2 from scratch | errors show only in combination |
| H6 | Affordability: small loss against a black box with the same inputs, data and training | |

## Arms

Same links, seeds and training budget in each arm.

| Arm | net2 sees | Compared with | State |
|---|---|---|---|
| A0 | the full true state of the path | A1: what the LCFs omit | UNRUN; the environment does not provide it |
| A1 | the true LCFs (oracle) | A3: the cost of estimation (H2) | run for the fixed and share rules and for a learned net2 (round 19) |
| A2 | net1's means only (`view: "means"`) | A3: the value of stated uncertainty | run on validation for online TD3+BC (net1 v3) and for rules where the flow cannot see the capacity (net1 v6); and for learned net2 on v6 (rounds 22, 23) |
| A3 | net1's beliefs | | the design under test |
| A4 | raw measurements (a black box) | A3: the cost of the split (H6) | run for a learned net2 (round 19b); a rule on a classical estimate on net1 v3 |

## Interpretability

A belief is a distribution over one physical quantity, so every statement
can be checked against the truth: "capacity is about 24 Mbit/s (90%
interval 22 to 26)", or "at least 18 Mbit/s (95%); the link was not kept
busy". Each statement comes with net1's measured coverage on held-out
links; the claim rests on that measurement. The plan:

- Explain a decision by evaluating net2 on a changed belief: exact
  counterfactuals, and exact Shapley values over the seven LCFs in 2^7 =
  128 evaluations (`train/attribution.py`). Run on expliCCit's released
  policy ([results.md](results.md)); on palaestra's own net2, UNRUN.
- Test leakage (H4) with probes that try to recover non-LCF quantities
  (the controller that produced the episode, raw columns) from net1's
  output. UNRUN.
- Detect a wrong belief without ground truth (R2): physical invariants
  among the beliefs, entropy, and later measurements of delivery rate and
  min RTT.
- Repair a fault in net1 alone (R3): retrain net1, keep net2. Replacing a
  belief with the truth at run time can confirm a diagnosis, with two
  cautions: the LCFs are coupled, so one patched LCF can describe a path
  that cannot exist, and a net2 trained on net1's beliefs has adapted to
  them, so the truth is not automatically a repair.
- Compare net2 on the full beliefs with net2 on a few percentiles per LCF
  in physical units, so that what an engineer reads is what net2 acts on.
  UNRUN.

## Status

| | Verdict | Evidence |
|---|---|---|
| Reproduction | The link model equals a port of Mahimahi's link queue, event for event; none of expliCCit's three checked results reproduced here; the comparison on the original stack is UNRUN | [results](results.md#reproduction-of-expliccit) |
| H1 | Evidence against, for a shared link: rules on the true LCFs fail among copies; told the number of flows, one rule handles copies and Cubic (an oracle bound). For a learned net2 at equal compute (round 19), the policy on the true LCFs ends 0.2 to 0.4 above the black box on raw measurements, and level with the one on net1. Measured: the raw measurements tell copies from a stock flow far better than net1's beliefs (0.753 against 0.496 balanced accuracy, three classes), and net1 learns it with a traffic head (0.909 on validation) | [results](results.md#what-limits-a-rule-is-the-other-flows) |
| H2 | Yes for five LCFs (test R^2 0.88 to 1.00), weaker for capacity (0.81) and free buffer (0.47); estimation costs the fixed rule 0.023 | [results](results.md#net1-estimates-the-path-state-within-its-training-range) |
| H3 | Yes within the training ranges (90% coverage 0.91 to 0.97 on test links); no outside them. Wider training data (net1 v6) made it honest where v5 failed and moved the edge, beyond which it fails again. On real cellular traces, outside every training range, coverage falls to 0.65 to 0.95 | [results](results.md#net1-estimates-the-path-state-within-its-training-range) |
| H4 | Partly: the learned net2 run unchanged on two other seeds of net1 v6 (same bins) kept 70% to 100% of its gains, with wider intervals. But net1's capacity belief under sharing carries the other flows' share: with the true capacity instead, the same net2 collapses as copies (-1.25) | [results](results.md#a-learned-net2-beats-the-rule-against-competitors) |
| Alarms (R2) | Alarms computed without ground truth catch wrong beliefs that the flow's own measurements contradict: 97% to 100% of episodes for faults injected into five LCFs, and 36 of 37 wrong capacity beliefs on real cellular traces, at 8% to 17% false alarms per LCF. They miss what the measurements cannot contradict: an overestimated busy fraction (0 of 58 on cellular) and a low capacity under competition. They name a group of LCFs, one-way delay alone. On expliCCit's own encoder, wrong in 41% of its beliefs in range, they catch 13% to 29% of its errors and not its Case Study 1 fault | [results](results.md#alarms-and-repair) |
| H5 | Two cases, net2 unchanged in both: a capacity fault repaired by retraining net1 alone (alarms without ground truth missed it), and the copies belief's confusion of Cubic with copies, traced to a belief net1 lacked and repaired by a traffic head added to net1 (the cooperative policy's loss with Cubic from the start went from -0.156 to +0.005 to +0.011, its copies gain from +0.211 to +0.20 to +0.32). The second repair left the trade-off between cooperating and pushing in place | [results](results.md#alarms-and-repair) |
| H6 | The simplest net2 on net1 beats every Linux controller and Sage on test links. Against a black box with the same inputs, learner and budget (A4, PPO on the raw measurements with BBR's BDP estimate), the split leads by 0.2 to 0.4 from the black box's own start, and by about 0.05 when the black box starts from a clone of the split's rule: the lead is the start that net1's estimate gives. Over every validation family that black box trails the split by 0.05 on the five-family mean, but cooperates with copies (+0.18 against -0.26); the split costs about 14 times the inference | [results](results.md#the-split-costs-little) |
| Learned net2 | On net1 v6, one candidate (round 18, round 7e's recipe) passed its gate on new validation episodes and on test beat the fixed rule against competitors (Cubic from the start +0.068, +0.023 to +0.120; competition conditions +0.039, +0.005 to +0.075) and matched it in every other family and as copies; five-family mean +0.018. The same recipe passed on net1 v5 (round 7e), whose competition gain did not hold on test. Learned from the reward alone (PPO), net2 gains more against competitors but pushes among copies (-0.22 to -0.44) | [results](results.md#a-learned-net2-beats-the-rule-against-competitors) |
| Stated uncertainty | No measurable value to net2 so far (A2 against A3), including for rules where the flow cannot see the capacity: probing on the spread mostly pushes on shared links, and probing on the means gains nothing over the fixed rule; the passing learned net2 does as well on the means (round 22) | [results](results.md#stated-uncertainty-has-not-helped-net2-yet) |
