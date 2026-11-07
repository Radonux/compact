# Results

Every number here comes from a published release unless it says
otherwise; terms, families and the scoring are defined in
[methodology.md](methodology.md#terms). Scores are log utility, paired by
link, with 95% bootstrap intervals over links in brackets. "Minus BBRv3"
means a controller's score minus BBRv3's on the same links.

Runs from #69 on use the guest and TCP settings frozen at `1365f0e`;
the few earlier runs cited here decided questions those settings did not
touch ([validation.md](validation.md#tcp-buffer-limits)). The
evaluation links are one family (constant or stepping capacity of 8 to
100 Mbit/s, one-way delay of 5 to 50 ms, buffers of 0.25 to 4 BDP), so
every ranking below holds for that family and the listed conditions only.
The Linux controllers ran with their defaults; only the rules' constants
were tuned for this score.

## net1 estimates the path state within its training range

net1 v5 recovers five of the seven LCFs with R^2 of 0.88 or more on links
it never saw, and its 90% statements hold 91% to 97% of the time. It is
weak on free buffer, and it is not calibrated outside the ranges of its
training data.

net1 v5 on the broad dataset's held-out links
([`exp-net1v4-recheck-79`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-recheck-79)):
R^2 of the belief's mean on the test links, then the coverage of its 90%
intervals on the test links, with controllers held out of training as the
observed flow (BBRv3, Westwood and a delay-threshold policy), with a BBRv3
or Westwood competitor (held out of training), and on three families
outside the training ranges.

| LCF | R^2 | 90% coverage | held-out controllers | held-out competitors | loss 3% to 6% | one-way delay 61 to 90 ms | buffers 8 to 16 BDP |
|---|---|---|---|---|---|---|---|
| mean queueing delay | 0.95 | 0.97 | 0.94 | 0.97 | 0.90 | 0.80 | 0.95 |
| queue | 0.90 | 0.92 | 0.89 | 0.92 | 0.91 | 0.89 | 0.90 |
| capacity | 0.81 | 0.92 | 0.88 | 0.92 | 0.88 | 0.84 | 0.90 |
| free buffer | 0.47 | 0.91 | 0.89 | 0.92 | 0.88 | 0.89 | 0.44 |
| busy fraction | 0.88 | 0.91 | 0.90 | 0.91 | 0.81 | 0.89 | 0.90 |
| one-way delay | 1.00 | 0.97 | 0.97 | 0.98 | 0.97 | 0.90 | 0.96 |
| receive rate | 0.99 | 0.94 | 0.93 | 0.94 | 0.92 | 0.92 | 0.94 |

- Free buffer depends on the buffer size, which sender-side measurements
  reveal only when the queue fills. net1's free-buffer intervals are
  calibrated (0.91), but its mean explains under half the variance.
- Capacity is identifiable only on a busy link. On net1 v3's held-out
  links, its 90% intervals for capacity were 12.3 Mbit/s wide on busy
  steps and 41.2 Mbit/s on idle ones (links of 4 to 200 Mbit/s; local
  run).
- Outside its ranges net1 is not honest. Its bins end just above the
  training range, so one-way delays above 60 ms look covered (0.90) while
  the mean cannot leave the range (R^2 -2.54). On buffers deeper than any
  in training, its free-buffer statements hold 44% of the time.
- An unseen arrival pattern broke an earlier net1. net1 v3 was trained
  only with competitors that arrived after the observed flow had measured
  the link alone. With a Cubic flow from the start, its median capacity
  belief was 0.59 of the truth and its 90% capacity statements held 26% of
  the time (local run). The broad data, with competitors from the start,
  repaired that ([alarms and repair](#alarms-and-repair)).
- A distribution per LCF was the right output. On expliCCit-like data it
  was at least as accurate as a single value or an interval on every LCF
  but free buffer, and its CRPS was 3 to 6 times lower than the single
  value's absolute error (local runs).

### Split by link, not by row

expliCCit's trainer splits rows at random, which puts a training row of
the same episode 10 ms from each test row. With expliCCit's own settings
on our expliCCit-like data
([`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97),
three seeds), a row split gives every LCF but free buffer an R^2 of 0.96
or more. Split by link, free buffer falls from 0.65 to -0.22, busy
fraction from 0.96 to 0.75, and the mean absolute error of capacity grows
4.0 times (one-way delay 2.1, receive rate 1.9). A row split measures
recall of training links, not estimation.

### Breadth of links mattered more than any setting

On 132 training links (local runs, as is everything on net1 v3 here
without a release), net1 v3's validation loss rose from the first epoch
and its 90% capacity statements held 71% of the time: with few links the
min RTT nearly names a link, and net1 recalls that link's capacity. Adding
1,050 links raised capacity's coverage to 0.89 with the same settings.
On the broad data, six settings of dropout and width span a mean CRPS
skill of only 0.768 to 0.780
([`exp-net1v4-grid-10`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-grid-10)),
lower learning rates did worse
([`exp-net1v4-lr-12`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-lr-12)),
and the choice held over two more seeds
([`exp-net1v4-seeds-13`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-seeds-13)).
In the grid every member's validation loss was lowest after 1 to 4 epochs
and rose after, as the network began to fit its training links; a lower
learning rate did not prevent it. For the same reason a minimal
set of quantities (capacity, queue, buffer size, one-way delay,
cross-traffic share) lost to the seven LCFs (mean CRPS skill 0.74 against
0.81): buffer size and cross-traffic share are constant over a link, so
predicting them rewards recognizing it.

## The split costs little

Even the simplest net2, the fixed rule (a window of 1.19 believed BDPs),
beats every Linux controller on the held-out test links, and the same
rule on the true LCFs does only 0.023 better.

The final test
([`exp-final-test-100`](https://github.com/Radonux/palaestra/releases/tag/exp-final-test-100)),
27 test links, every candidate fixed on validation, no failed episode;
minus BBRv3:

| Controller | training-20 | training-60 | new-20 | deep-long-20 | cubic-first-60 | five-family mean | held-competitors-60 | copies-30s |
|---|---|---|---|---|---|---|---|---|
| fixed rule on the true BDP | +0.152 | +0.148 | +0.291 | +0.205 | -0.014 | +0.156 | +0.120 | not run |
| share rule | +0.083 | +0.101 | +0.285 | +0.227 | +0.067 (-0.055 to +0.186) | +0.153 | +0.149 | -0.271 (-0.590 to +0.027) |
| fixed rule | +0.098 | +0.108 | +0.262 | +0.238 | -0.041 (-0.197 to +0.106) | +0.133 | +0.105 (-0.014 to +0.216) | +0.047 (-0.202 to +0.279) |
| BBRv1 | -0.031 | +0.020 | +0.209 | -0.027 | +0.111 | +0.057 | +0.120 | -0.261 |
| Scalable | -0.509 | -0.423 | -0.116 | -2.537 | +0.375 | -0.642 | +0.234 | not run |
| Sage | -0.364 | -0.494 | -0.356 | -0.491 | -0.852 | -0.511 | -0.881 | +0.004 |
| expliCCit's released policy | -1.573 | -1.433 | -1.358 | -2.200 | -1.676 | -1.648 | -1.581 | -1.288 |

- The fixed rule leads BBRv3 by 0.133 over the five families and BBRv1,
  the best Linux controller on that mean, by 0.076. It leads in every
  family but cubic-first, where it ties BBRv3 and the aggressive
  controllers win (Scalable +0.375, BIC +0.334, H-TCP +0.140, BBRv1
  +0.111).
- The rule on the true BDP scores 0.023 more over the five families: what
  net1 v5's estimate costs this rule. On net1 v3, the same rule on a
  classical estimate (largest delivery rate over 10 s times the min RTT,
  as BBR builds it) scored 0.155 below the rule on net1, and 0.228 below on
  a second held-out set of 158 links: net1 closed about 85% of the gap
  between that estimate and the truth (local runs).
- Copies of the fixed rule hold 0.99 utilization (BBRv3's 0.96) with 22 to
  29 ms of mean queueing delay (28 to 33 ms). With 2 copies they share as
  evenly as BBRv3's (Jain's index 0.979 against 0.984), with 4 less so
  (0.915 against 0.979).
- Against Sage's released policy with the flow alone, the fixed rule has
  7% to 8% more utilization on clean links in the training families and
  14% more on deep, long paths; its queueing delay is within the noise of
  Sage's in the training families and 70% lower on deep, long paths. Under
  1% and 3% random loss it has 1.7 to 2.6 times Sage's utilization.

Nothing was chosen on two outside benchmarks, ccperf's experiments
([`exp-ccperf-77`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-77),
[`exp-ccperf-v5-87`](https://github.com/Radonux/palaestra/releases/tag/exp-ccperf-v5-87))
and ccBench's suites
([`exp-ccbench-merged-92`](https://github.com/Radonux/palaestra/releases/tag/exp-ccbench-merged-92)),
which the rules on net1 v5 ran unchanged:

| Benchmark, measure | Fixed rule | Share rule | BBRv1 | BBRv3 | Cubic | Sage |
|---|---|---|---|---|---|---|
| ccperf single flow, utilization | 0.917 | 0.911 | 0.863 | 0.861 | 0.621 | 0.454 |
| ccperf random loss 0.1% to 5%, utilization | 0.972 | 0.977 | 0.978 | 0.684 | 0.415 | 0.528 |
| ccperf deep buffers, queueing delay (ms) | 2.6 | 2.4 | 19.3 | 8.7 | 74.6 | 4.8 |
| ccperf Poisson rate traces, queueing delay (ms) | 314 | 365 | 244 | 211 | 523 | 540 |
| ccperf against Cubic, share over fair share | 0.847 | 1.088 | 0.976 | 0.746 | 1.076 | 0.303 |
| ccBench solo, games won | 73.5% | 66.0% | 42.4% | 25.0% | 3.4% | 25.2% |
| ccBench, a Cubic flow first, share of capacity | 0.398 | 0.426 | 0.497 | 0.399 | 0.369 | 0.206 |

The rules keep the link full with a short queue alone, under random loss
and on deep buffers, and win most of ccBench's solo games (its score is
throughput squared over delay). They queue more than BBR on traces whose
capacity jumps, and against a Cubic flow that started first they take
less than half the link: the same gap as cubic-first-60.

Compute: outside the guest, one decision of the fixed rule on net1 v5
takes 1.07 ms (median; 99th percentile 1.29 ms) on one core, against
0.84 ms for Sage's released policy; net1's three members dominate
(`tools/policy-bench`, 2026-11-05, local). In the guest, inference costs
no virtual time, so none of this shows in the scores.

## What limits a rule is the other flows

Pushing against a buffer-filling flow pays, pushing against copies of
oneself hurts, and the seven LCFs do not say which is present. Told the
number of flows, a rule on the true state handles both.

- The share rule matches the rule on the true BDP over the five families
  and gains on the fixed rule where competitors fill the buffer (+0.108
  with Cubic from the start, +0.044 with the held-out competitors), but
  its copies lose (-0.271 against BBRv3's copies, where the fixed rule's
  have +0.047; `exp-final-test-100`). Each copy believes its share is
  small and pushes.
- Rules on the true LCFs fail among copies
  ([`exp-share-oracle-val-20`](https://github.com/Radonux/palaestra/releases/tag/exp-share-oracle-val-20),
  net1 v4, validation). Each copy aims at the whole link's BDP, so 2 and 4
  copies of the fixed rule on the truth build queues of 41 and 66 ms and
  score 1.06 below the same rule on net1 v4. net1 believes a shared link
  smaller than it is (0.4 to 0.9 of it in local runs), which helps by
  accident. The capacity this flow can take is not among the LCFs.
- A hand-made switch does not fix it
  ([`exp-mode-val-19`](https://github.com/Radonux/palaestra/releases/tag/exp-mode-val-19)).
  A cooperative window that follows the flow's own share makes copies
  lead the fixed rule's copies by 0.44 (queues of 5 and 8 ms), but yields
  to Cubic (1.7 below the fixed rule with Cubic from the start). Switching
  between it and the share rule on net1's belief of a full buffer kept
  neither gain.

Told the number of flows N (constant and handed to the rule), a rule on
the truth that aims at 1/N of the BDP and pushes only below 1/N of the
capacity in use handles both
([`exp-fairshare-used-val-23`](https://github.com/Radonux/palaestra/releases/tag/exp-fairshare-used-val-23),
validation, minus BBRv3):

| Rule | alone | 2 or 4 copies | Cubic from 0 s |
|---|---|---|---|
| fixed rule on net1 v4 | +0.180 | +0.111 | -0.029 |
| 1/N rule on the truth, k = 2 | +0.199 | +0.460 | -0.286 |
| 1/N rule on the truth, k = 4 | +0.187 | +0.398 | +0.065 |
| 1/N rule on the truth, k = 6 | +0.166 | -0.605 | +0.249 |

At k = 4 it matches the fixed rule alone and leads it with copies (by
0.29) and with Cubic from the start (by 0.09). This is an oracle bound: k
was chosen on these links and N never changes. N is not added to the
LCFs: it describes the other endpoints, not the path, changes as flows
come and go, and one sender's measurements fit several values of it. It
is net2's to infer from how the others respond.

## Learned net2 has not passed its gate

No learned net2 has passed the gate. Reinforcement learning from the rule
never moved off the rule. Imitation of a privileged teacher learns to
tell copies from Cubic in part, but its candidates either lose 0.08 to
0.15 to the fixed rule as copies (rounds 4c to 6b, 7v) or, taught to
cooperate with copies, give up their gain against Cubic (round 7).

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

Every candidate's five-family mean is within 0.04 of the rule's. Each
gate's notes name its candidates' policy release (`model-net2-...`). The
training runs: round 1 `exp-net2-ppo-search1-90`, `-search2-96`,
`-seeds-98`; round 2 `exp-net2-ppo-team-101`, `-flows-102`,
`exp-net2-td3bc-103`; round 3 `exp-net2-r3-ppo-105`, `-td3-106`,
`-share-109`; round 4 `exp-net2-r4-dagger-110`, 4c `-r4c-dagger-113`;
round 5 `exp-net2-r5-gate-115`; round 6 `exp-net2-r6-gate-119`, 6b
`-r6b-gate-120`; round 7 `exp-net2-r7-gate3-124`.

Reinforcement learning stayed at its start. After 60 iterations the
spread of u across states was 0.02 to 0.04 for every PPO and TD3+BC
learner, and the residuals stayed at the share rule. A return over
thousands of decisions, each nudged by exploration noise of sigma 0.05,
carries too little about any one of them. In round 1, PPO drifted toward
the training mix's best constant u: Cubic was a tenth of the links, and
on the rest a lower u is better.

A graph bug cost rounds 1 and 2 about 0.13 as copies. The exported graph
held a decision's window until the next decision, freezing the BDP it was
computed from, where the rule re-reads its belief every step. The
untrained clone of the rule lost 0.132 to the rule as copies with the held
window, and came within 0.003 of it once each step aimed at that step's
believed BDP
([`exp-net2-clone-copies-107`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-clone-copies-107),
fixed in `85de8d0`).

Imitation learns state dependence. DAgger gives a label at every
decision: the teacher is the share rule while a stock TCP flow shares the
link and the fixed rule otherwise, which the student must infer from
net1's output. Round 4's students spread u by 0.16 to 0.48 across states,
and against the share rule at the teacher's gain they kept 62% to 77% of
its gain against Cubic while paying 31% to 34% of its loss as copies: they
tell the two apart in part. Round 4 overfit (about 12 passes per
iteration over a replay drawn from 64 links); round 4c fixed that, and
its copies loss fell from 0.291 to 0.129.

Matching the rule among copies has a floor. From round 4c to 6b, the best
candidates lose 0.08 to 0.14 as copies however training changed. A seed's
copies loss follows how well it told the cases apart in training: the
round-5t seed whose belief p was 0.78 with a stock TCP flow and 0.13
without lost 0.02 as copies, the two with p of 0.44 and 0.47 without lost
0.13 to 0.16. The loss sits on a few links (7 of 42 copies episodes in
round 5e set a window more than 0.1 above the rule's, in log2), where one
copy's push deepens the queue, which then looks like a buffer-filling
flow to the others.

Cooperating with copies moved the trade-off rather than removing it.
Round 7's teacher uses the cooperative window (the mode rule's, above)
while only copies share the link. Its students learned q, a belief that
copies are present: over the last 10 training iterations, 0.55 to 0.72
with copies and 0.21 to 0.26 elsewhere, while p stayed at 0.21 to 0.35
with copies. The copies loss vanished on average (-0.012 for the kept
candidate, +0.054 for the other), but so did most of the gain against
Cubic (+0.051, against +0.079 to +0.146 in rounds 4 to 6b), and its
copies interval (-0.256 to +0.234) is the widest of any round: among
copies the outcome depends strongly on the link.

Per link (from the round-7 gate's episodes), the cooperative window was
the cost. With Cubic from the start, on buffers of 20 to 112 packets
some seeds took Cubic for copies and yielded: their receive share fell
to 0.15 to 0.5 where the rule's was 0.39 to 0.87, losing 0.4 to 1.1 on
those links, while on deeper buffers every seed gained 0.3 to 0.6.
Lowering the threshold on p changed little (round 7t). The beliefs are
trained on their labels whatever the policy does, so the same students
can act differently: with q vetoing the push instead, u = u_rule + p'(1
- q')(share - u_rule) (round 7v), taking Cubic for copies costs at most
the push. The gain-1 candidate then meets every gate condition but
copies, the closest any round has come, and loses 0.145 as copies: at a
threshold of 0.5 the veto among copies is weak.

Before the gate existed, learning from other controllers' logs failed
(cloning and IQL on net1 v3: -0.97 to -1.61 on validation, against -0.48
for BBRv3; the policies inherit the logs' confounding of a full queue with
a large window). Online TD3+BC from the rule drifted below it with
one-step returns and tied it with 40-step returns on net1 v3's validation
links (local runs), and trailed it on net1 v4's
([`exp-net2v4-val-16`](https://github.com/Radonux/palaestra/releases/tag/exp-net2v4-val-16)).

## Reproduction of expliCCit

Three of expliCCit's results were checked here, and none reproduced as
stated.

| expliCCit's result | Here | Evidence |
|---|---|---|
| net1 estimates every LCF with R^2 above 0.9 | Only with its random row split, and not for free buffer even then (0.65). On held-out links net1 v5 has 0.81 for capacity, 0.88 for busy fraction and 0.47 for free buffer | [`exp-net1-row-split-97`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-row-split-97), `exp-net1v4-recheck-79` |
| The split costs about 5% utilization and 9% queueing delay against Sage | The other way for our split (above). expliCCit's released policy, as deployed, has 41% less utilization than Sage on clean test links and 24% to 25% less on clean validation links | `exp-final-test-100`, [`exp-sage-v5-val-94`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-v5-val-94) |
| On a 24/48 Mbit/s link (40 ms one way, 1000-packet buffer), under-utilization traces to capacity underestimated by 25%, and the true capacity restores it | The released policy uses 0.48 of the link (0.34 to 0.53 on the eight links around it), and its capacity belief is within 14%; the true capacity gives 0.43. Its wrong beliefs are busy fraction (1.00 against 0.51) and queue (19 packets against 0); the true busy fraction gives 0.71 on all nine links. The deployed rounding of beliefs is not the cause | [`exp-expliccit-case-42`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-42), [`exp-expliccit-case-91`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-case-91) |

Divergences from expliCCit's and Sage's setting, each a possible cause:

- Linux 7.0 in UML in virtual time, not Sage's patched Linux 4.19 in real
  time; pacing refreshed every step, not every ACK; inference that costs
  no virtual time.
- Sage and expliCCit's policies run on links they were not trained on.
  expliCCit's released net1 was trained on its own emulation, and its
  ground-truth definitions are not in its code, so ours may differ. On our
  expliCCit-like data it had R^2 1.000 for one-way delay (so its "mRTT" is
  the one-way delay), 0.899 for receive rate, 0.847 for queueing delay,
  0.594 for queue, 0.361 for capacity, -0.743 for busy fraction and -59.0
  for free buffer (local run).
- The emulator itself matches Mahimahi: the link model equals a port of
  Mahimahi's link queue event for event, and the state message equals
  `sage.cc`'s at every compared step ([validation.md](validation.md)). The
  run of the same four links on the original stack, which would separate
  the emulator from the rest, is UNRUN: the kit is `tools/crosscheck`, and
  palaestra's side is
  [`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)
  (expliCCit's released policy uses 0.50 of the case-study link there,
  Sage 0.92).

## Alarms and repair

Alarms computed without ground truth catch beliefs that the flow's own
measurements contradict, and miss beliefs they are consistent with. A
capacity fault was repaired by retraining net1 alone, with net2
unchanged.

Alarms (R2,
[`exp-r2-alarms-81`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-alarms-81)):
physical invariants among the beliefs (receive rate at most capacity
times busy fraction; queueing delay equal to queue over capacity; queue
plus free buffer constant), each belief's entropy, and checks against the
delivery rate and min RTT measured three smoothed RTTs later, with
each threshold at the 95th percentile of the alarm's episode mean on
net1 v4's validation links. False alarms stay
near the nominal 5% per alarm. On net1 v3 they catch one-way delay on
deep buffers (0.83 of 168 wrong episodes, AUC 0.91), queueing delay at 61
to 90 ms (0.89 of 90, AUC 0.92) and capacity under 3% to 6% loss (0.63 of
287, AUC 0.79). They miss its capacity under competition: 0.12 of 43 wrong
episodes with Cubic from the start (AUC 0.52). A capacity believed at 0.59
of the truth is still above the flow's own share, so nothing the flow
measures contradicts it (inferred, not tested).

Repair (R3). net1 v3's rule lost to BBRv3 with Cubic from the start
because its capacity belief was low there (above); the same rule on the
true BDP did not, so the belief was the fault. Retraining net1 on the
broad data repaired it with net2 unchanged: on the test links with Cubic
from the start, the fixed rule on net1 v4 beat the same rule on net1 v3
by 0.428 (0.277 to 0.580), each at its validation u
([`exp-rule-net1v4-test-14`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-14)).
Under the frozen settings, on validation
([`exp-r3-capacity-head-val-86`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-val-86)),
the rule minus BBRv3 with Cubic from the start is -0.555 on net1 v3,
-0.037 on net1 v4 and +0.058 on the truth. Retraining only net1 v3's
capacity head
([`exp-r3-capacity-head-83`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-83))
repairs its calibration (90% coverage 0.62 to 0.93) but recovers little
in closed loop: +0.16 with Cubic from the start, where retraining all of
net1 gained +0.52. A new head on frozen features cannot supply cues the
features never learned (inferred).

The case study above shows the other half of the method: replacing one
belief at a time with the truth localized expliCCit's fault to busy
fraction. On our own early models some such patches hurt (a cloned policy
fell to 0.144 utilization with the true busy fraction; local run), because
a net2 has adapted to its net1's errors.

## Stated uncertainty has not helped net2 yet

A net2 on the means of net1's beliefs (arm A2) did at least as well as
one on the full beliefs (A3). Over 60 iterations of online TD3+BC on net1
v3, scored every 10 iterations on the validation links, A3 minus A2 stayed
between -0.030 and +0.001; the one interval that excluded zero favoured
the means (one seed each, local runs, not run on test). A likely reason,
not tested: a window target from the believed BDP needs mostly the means,
and these runs keep the link busy, where capacity is identifiable.
net1's honesty still serves the human reading of its beliefs.
