# net2 rounds

The record behind [results.md](results.md#a-learned-net2-beats-the-rule-against-competitors):
every gate a learned net2 faced on net1 v5, and the training run of every
round. Terms and scoring are in [methodology.md](methodology.md#terms);
each number is a candidate minus the fixed rule, paired by link, with its
95% interval over links.

Every round on net1 v5, minus the fixed rule on the 21 validation links
(campaign seed 21); the kept candidate unless named. "Competition" is
new-20's three competition conditions.

| Round | Learner, and what changed | Gate | cubic-first-60 | competition | copies-30s |
|---|---|---|---|---|---|
| 1 | Recurrent PPO from the rule; tau, sigma, learning rate, K and the training mix searched | [`exp-net2-ppo-gate-99`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-ppo-gate-99) | -0.007 (-0.032 to +0.015) | +0.005 (-0.029 to +0.037) | -0.127 (-0.235 to -0.031) |
| 2 | PPO with a team reward for copies, or a critic that counts the flows | [`exp-net2-gate-team-104`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-team-104) | -0.019 (-0.042 to +0.004) | -0.023 (-0.057 to +0.008) | -0.128 (-0.301 to +0.020) |
| 2 | TD3+BC, the same graph | [`exp-net2-gate-td3-108`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-td3-108) | -0.021 (-0.040 to -0.001) | -0.008 (-0.043 to +0.030) | -0.121 (-0.254 to -0.022) |
| 3 | PPO and TD3+BC on the fixed graph; residuals on the share rule (kept: TD3+BC residual) | [`exp-net2-gate-r3-111`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r3-111) | +0.109 (+0.048 to +0.176) | +0.044 (-0.009 to +0.090) | -0.315 (-0.399 to -0.234) |
| 4 | DAgger from a privileged teacher; the student learns u | [`exp-net2-gate-r4-114`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r4-114) | +0.106 (+0.054 to +0.164) | +0.058 (+0.004 to +0.102) | -0.291 (-0.383 to -0.196) |
| 4c | DAgger without the overfitting: more links, one pass per iteration | [`exp-net2-gate-r4c-118`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r4c-118) | +0.087 (+0.042 to +0.137) | +0.055 (+0.011 to +0.094) | -0.129 (-0.189 to -0.076) |
| 5 | The student learns only p, its belief that a stock TCP flow is present | [`exp-net2-gate-r5-116`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r5-116) | +0.146 (+0.064 to +0.229) | +0.060 (+0.011 to +0.105) | -0.378 (-0.510 to -0.261) |
| 5t | The same students, pushing only past p = 0.5 (named: teacher gain 0.5) | [`exp-net2-gate-r5t-117`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r5t-117) | +0.096 (+0.052 to +0.143) | +0.036 (+0.002 to +0.068) | -0.104 (-0.171 to -0.048) |
| 5e | One policy on the mean belief of round 5t's three gain-0.5 students | [`exp-net2-gate-r5e-121`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r5e-121) | +0.088 (+0.037 to +0.143) | +0.021 (-0.032 to +0.070) | -0.112 (-0.233 to -0.014) |
| 6 | The threshold also in training; round 5's students continued | [`exp-net2-gate-r6-122`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r6-122) | +0.095 (+0.046 to +0.143) | +0.034 (-0.011 to +0.072) | -0.135 (-0.204 to -0.078) |
| 6b | Round 6 with more copies in training (named: new students) | [`exp-net2-gate-r6b-123`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r6b-123) | +0.079 (+0.040 to +0.123) | +0.032 (-0.013 to +0.072) | -0.081 (-0.161 to -0.010) |
| 7 | A teacher that also cooperates with copies; the student learns p and q, its belief that copies are present | [`exp-net2-gate-r7-125`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7-125) | +0.051 (-0.086 to +0.184) | +0.006 (-0.065 to +0.069) | -0.012 (-0.256 to +0.234) |
| 7t | Round 7's students acting on p past 0.3 (named) or 0.4 instead of 0.5 | [`exp-net2-gate-r7t03-126`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7t03-126), [`-r7t04-127`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7t04-127) | +0.062 (-0.077 to +0.199) | +0.009 (-0.058 to +0.068) | -0.070 (-0.329 to +0.180) |
| 7v | Round 7's students with q vetoing the push instead of cooperating | [`exp-net2-gate-r7v-128`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7v-128) | +0.147 (+0.069 to +0.227) | +0.059 (+0.017 to +0.099) | -0.145 (-0.211 to -0.077) |
| 7vq | The veto with q's threshold at 0.3 or 0 (named: 0) | [`exp-net2-gate-r7vq03-130`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7vq03-130), [`-r7vq0-131`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7vq0-131) | +0.140 (+0.060 to +0.220) | +0.056 (+0.010 to +0.099) | -0.142 (-0.221 to -0.064) |
| 7e | Round 7's three seeds as one policy on their mean beliefs, with the veto and q's threshold at 0 (named: gain 0.5, which passes; gain 1 lost 0.113 as copies) | [`exp-net2-gate-r7e05-133`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7e05-133), [`-r7e1-132`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7e1-132) | +0.086 (+0.040 to +0.139) | +0.054 (+0.004 to +0.107) | -0.005 (-0.064 to +0.049) |
| 7e, again | Round 7e's passing candidate unchanged, on new validation episodes (campaign seed 11) | [`exp-net2-gate-r7e05-s11-137`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r7e05-s11-137) | +0.086 (+0.039 to +0.140) | +0.057 (+0.009 to +0.107) | -0.027 (-0.083 to +0.024) |
| 8e | Round 7's students trained 60 more iterations under the veto, as 7e (named: gain 0.5; gain 1 lost 0.106 as copies) | [`exp-net2-gate-r8e05-136`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r8e05-136), [`-r8e1-135`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r8e1-135) | +0.073 (+0.020 to +0.134) | +0.029 (-0.027 to +0.082) | -0.042 (-0.082 to -0.003) |
| 9 | Round 7's learner on 16 times the training links (named: gain 0.5; gain 1 had competition +0.026) | [`exp-net2-gate-r9links05-140`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r9links05-140), [`-r9links1-139`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r9links1-139) | +0.070 (+0.027 to +0.125) | +0.013 (-0.020 to +0.042) | -0.041 (-0.087 to +0.006) |
| 9b | Those links with stock competitors that join within the first 10 s (named: gain 0.5; gain 1 had +0.020 with Cubic from the start) | [`exp-net2-gate-r9broad05-142`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r9broad05-142), [`-r9broad1-141`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r9broad1-141) | +0.015 (-0.010 to +0.043) | +0.039 (+0.004 to +0.078) | -0.033 (-0.100 to +0.025) |
| 12 | PPO from the reward, the flow alone on about half the training links; checkpoint at iteration 79, picked on these links (an evaluation, not a gate) | [`exp-net2-eval-r12-155`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-r12-155) | +0.026 (-0.064 to +0.129) | +0.013 (-0.060 to +0.088) | -0.099 (-0.406 to +0.199) |

Every candidate's five-family mean is within 0.04 of the rule's. Each
gate's notes name its candidates' policy release (`model-net2-...`). The
training runs: round 1 `exp-net2-ppo-search1-90`, `-search2-96`,
`-seeds-98`; round 2 `exp-net2-ppo-team-101`, `-flows-102`,
`exp-net2-td3bc-103`; round 3 `exp-net2-r3-ppo-105`, `-td3-106`,
`-share-109`; round 4 `exp-net2-r4-dagger-110`, 4c `-r4c-dagger-113`;
round 5 `exp-net2-r5-gate-115`; round 6 `exp-net2-r6-gate-119`, 6b
`-r6b-gate-120`; round 7 `exp-net2-r7-gate3-124`; round 8
`exp-net2-r8-gate3-129`; round 9 `exp-net2-r9-data-138`; round 10
`exp-net2-r10-rl-143`; round 11 run 144 (it published no release:
`model-net2-r11-state`) and `exp-net2-r11-s2-150`; round 12
`exp-net2-r12-mix-147`; round 13 `exp-net2-r13-alone-151`; on net1 v6,
round 14 `exp-net2-r14-v6-154`, round 15 run 156 (no release; its final
copies-arm policies are in `model-net2-v6-cands`), round 16
`exp-net2-r16-long-157`, round 14's stage 2 `exp-net2-r14-s2-158`,
round 17 `exp-net2-r17-copies-161`, round 18
`exp-net2-r18-gate3-v6-162`, round 19 `exp-net2-r19-blackbox-164`, round 19b `exp-net2-r19b-bbr-165`, round 20a `exp-net2-r20a-clone-170`
(its clones in `model-net2-r20a-clones`), round 20b `exp-net2-r20b-bb-ppo-171`. Evaluations on net1 v6:
`exp-net2-eval-v6-159`, `-v6-ens-160`, `-v6-veto-166`; round 18's gate
`exp-net2-gate-r18e05-s11-167`, test `exp-net2-test-r18e05-168` and net1
swap `exp-net2-swap-v6-169`, per-LCF truth patch `exp-net2-patch-v6-172`, net1 v6's capacity belief under sharing `exp-net1-belief-share-173`.

Before the gate existed, learning from other controllers' logs failed
(cloning and IQL on net1 v3: -0.97 to -1.61 on validation, against -0.48
for BBRv3), and online TD3+BC trailed the rule on net1 v4's validation
links ([`exp-net2v4-val-16`](https://github.com/Radonux/palaestra/releases/tag/exp-net2v4-val-16)).

Round 19's black box first aimed its window at the guest's 1 s average
delivery rate, which trails a growing window by about 0.5 s; it started
2.18 below net1 v6's rule and collapsed to 2.8 to 3.0 below. Round 19b
gave it BBR's estimate (the per-ACK delivery rate's maximum over 2 s),
and round 20a a clone of the rule on net1 as its start.
