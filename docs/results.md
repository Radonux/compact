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
- Outside its ranges net1 v5 is not honest. Its bins end just above the
  training range, so one-way delays above 60 ms look covered (0.90) while
  the mean cannot leave the range (R^2 -2.54). On buffers deeper than any
  in training, its free-buffer statements hold 44% of the time.
- An unseen arrival pattern broke net1 v3, trained only with competitors
  arriving after the flow had measured the link alone: with Cubic from
  the start its capacity belief was 0.59 of the truth and its 90%
  statements held 26% of the time (local run), until the broad data
  repaired it ([alarms and repair](#alarms-and-repair)).
- A distribution per LCF was as accurate as a single value or an interval
  on every LCF but free buffer, with a CRPS 3 to 6 times below the single
  value's absolute error (expliCCit-like data, local runs).

Wider training data moves that edge; it does not remove it. net1 v6
has net1 v5's settings and data plus links with one-way delays of 5 to
150 ms, buffers of 0.25 to 32 BDP, base capacities of 1 to 200 Mbit/s and
120 s episodes ([`exp-data-net1-v6-146`](https://github.com/Radonux/palaestra/releases/tag/exp-data-net1-v6-146), [`exp-net1v6-149`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-149)).
On v5's own test links it is as good (CRPS skill 0.782 to 0.784 over
three seeds against 0.778, calibration equal, though its free-buffer mean
is worse: R^2 0.22 to 0.30 against 0.47;
[`exp-net1v6-inrange-152`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v6-inrange-152)). On the families v5 never saw it is
honest (one-way delays of 61 to 90 ms: R^2 0.97, 90% coverage 0.97).
Beyond its own new ranges it fails again, more gently: at one-way delays
of 151 to 250 ms its one-way-delay statements hold 27% of the time (R^2
-3.84, against v5's -24.1). net1 v6 replaces v5 for new work, and the
fixed rule on it keeps u = 0.25 ([`exp-rule-net1v6-val-153`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v6-val-153)).

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

With few links the min RTT nearly names a link and net1 recalls its
capacity: on 132 training links net1 v3's validation loss rose from the
first epoch and its 90% capacity statements held 71% of the time; 1,050
more links raised that to 0.89 (local runs, as is everything on net1 v3
here without a release). On the broad data six settings of dropout and
width span a mean CRPS skill of only 0.768 to 0.780
([`exp-net1v4-grid-10`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-grid-10)),
lower learning rates did worse
([`exp-net1v4-lr-12`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-lr-12)),
the choice held over two more seeds
([`exp-net1v4-seeds-13`](https://github.com/Radonux/palaestra/releases/tag/exp-net1v4-seeds-13)),
and every member's validation loss was lowest after 1 to 4 epochs. For
the same reason a minimal set of quantities (capacity, queue, buffer
size, one-way delay, cross-traffic share) lost to the seven LCFs (mean
CRPS skill 0.74 against 0.81): buffer size and cross-traffic share are
constant over a link, so predicting them rewards recognizing it.

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
  29 ms of queueing delay (28 to 33 ms) and share as evenly as BBRv3's
  with 2 copies (Jain's index 0.979 against 0.984), less so with 4.
- Against Sage's released policy alone, the fixed rule has 7% to 14% more
  utilization on clean links and 1.7 to 2.6 times Sage's under 1% and 3%
  random loss, with up to 70% less queueing delay on deep, long paths.

For a learned net2 the split's lead is its start. The same recurrent
PPO, mixture and budget on three inputs (rounds 19, 19b and 20b,
[`exp-net2-r19-blackbox-164`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19-blackbox-164),
[`-r19b-bbr-165`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r19b-bbr-165),
[`-r20b-bb-ppo-171`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20b-bb-ppo-171)),
minus net1 v6's fixed rule on the new family's validation links, three
seeds each, at the start and at the last evaluation (iterations 79 to
119):

| The actor reads | Start | Last evaluation |
|---|---|---|
| the true LCFs (A1) | +0.040 | -0.034 to -0.015 |
| net1 v6's beliefs (A3) | -0.008 | -0.040 to +0.017 |
| the raw measurements, its window on BBR's BDP estimate (A4) | -0.554 | -0.387 to -0.217 |
| the same, started from a clone of the rule on net1 (A4) | -0.087 to -0.064 | -0.096 to -0.036 |

From its own start, a window of 1.19 times BBR's estimate, the black box
learned (u rose from 0.25 to 0.5) but stayed 0.2 to 0.4 behind; with
Cubic from the start it ended 0.63 below. A window aimed at the flow's
own delivery rate shrinks as a competitor takes share, where net1's
capacity estimate does not (inferred). Started instead from a clone of
the rule on net1, learned from the raw measurements by DAgger with that
rule as the teacher
([`exp-net2-r20a-clone-170`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20a-clone-170):
40 iterations brought it from 0.554 to 0.064 to 0.087 below the rule,
and 80 more only to 0.031 to 0.067 below,
[`exp-net2-r20c-clone-more-175`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-r20c-clone-more-175)),
the same PPO ended 0.036 to 0.096 below, about 0.05 below the arm on net1
on average and level with it on one seed. Given the split's start, the
raw measurements learn from the reward nearly as well; what the split
gives is that start, a window on an estimate of the path rather than of
the flow's own throughput, which the black box reached only by imitating
the split. Off the family they trained on, the inputs part ways. On every
validation family
([`exp-net2-eval-ladder-174`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-ladder-174),
final policies, campaign seed 21) the black box trailed the arm on net1 in
each family, by 0.046 on the five-family mean (-0.063 against -0.017;
-0.015 on the truth), and gained less with Cubic from the start (+0.045
against +0.130), but as copies it beat the rule (+0.183, +0.009 to +0.351)
where the arms on net1 and on the truth lost (-0.256 and -0.706): a
window on the flow's own delivery rate cooperates with copies and yields
to Cubic, the trade-off of the rules in the next section. Inference is the
split's cost: a decision of the policy on net1
v6 takes about 1 ms on one core, the black box's 0.07 ms
(`tools/policy-bench`, local). On the true LCFs, the learner lost its
start's lead and ended level with the arm on net1: what the truth adds,
learning from the reward did not use, and the drift toward pushing (next
sections) is the learner's, not net1's errors.

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
and on deep buffers; they queue more than BBR on traces whose capacity
jumps, and take less than half the link against a Cubic flow that started
first, the same gap as cubic-first-60.

Compute: one decision of the fixed rule on net1 v5 takes 1.07 ms on one
core (Sage's policy 0.84 ms; `tools/policy-bench`, local); in the guest
inference costs no virtual time.

## What limits a rule is the other flows

Pushing against a buffer-filling flow pays, pushing against copies of
oneself hurts, and the seven LCFs do not say which is present. Told the
number of flows, a rule on the true state handles both.

The measurements do say it; net1's beliefs lose it. A per-step classifier
of the link's other traffic (none, copies, a stock TCP flow), scored on
held-out validation links with net1 v6's fixed rule as the observed flow,
reaches a balanced accuracy of 0.753 from the raw measurements and 0.496
from net1 v6's beliefs (chance 0.333), and calls a stock flow copies 14%
of the time from the measurements against 33% from the beliefs
([`exp-net2-competitor-info-177`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-competitor-info-177)).
net1 can learn it: a head on net1's scaled measurements, trained on the
training links (round 21,
[`exp-net1-traffic-data-178`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-data-178)), reaches 0.909
on the same validation links and calls a stock flow copies 15% of the time
(BBRv1 6%, Cubic 11%, the held-out BBRv3 and Westwood 19% and 25%;
[`exp-net1-traffic-head-val-180`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-traffic-head-val-180)).

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
  smaller than it is, which helps by accident: net1 v6's capacity belief
  is 1.03 to 1.04 of the link alone, 0.72 to 0.82 of it with 2 copies and
  0.46 to 0.61 with 4, between the fair share and the link
  ([`exp-net1-belief-share-173`](https://github.com/Radonux/palaestra/releases/tag/exp-net1-belief-share-173)).
  The capacity this flow can take is not among the LCFs.
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

## A learned net2 beats the rule against competitors

A learned net2 on net1 v6 beat the fixed rule against competitors on the
test links and matched it everywhere else, copies included. It is round
7e's recipe trained again on v6 (round 18): DAgger students that learn,
from net1's output alone, two beliefs (p, that a stock TCP flow shares
the link, and q, that copies of itself do), run as one policy on three
seeds' mean beliefs, with q vetoing the push. It was picked among seven
candidates on the validation links
([`exp-net2-eval-v6-veto-166`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-veto-166)),
passed its gate alone on new validation episodes
([`exp-net2-gate-r18e05-s11-167`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-gate-r18e05-s11-167)),
and then ran once on the 27 test links
([`exp-net2-test-r18e05-168`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r18e05-168)),
with no failed episode:

| Family | net2 minus the rule | net2 minus BBRv3 | rule minus BBRv3 |
|---|---|---|---|
| training-20 | -0.002 (-0.013 to +0.010) | +0.093 | +0.095 |
| training-60 | +0.011 (-0.001 to +0.026) | +0.123 | +0.112 |
| new-20 | +0.018 (+0.001 to +0.035) | +0.271 | +0.253 |
| deep-long-20 | -0.005 (-0.025 to +0.010) | +0.257 | +0.262 |
| cubic-first-60 | +0.068 (+0.023 to +0.120) | -0.041 | -0.109 |
| five-family mean | +0.018 | +0.141 | +0.123 |
| new-20, competition | +0.039 (+0.005 to +0.075) | | |
| held-competitors-60 | | +0.104 | +0.089 |
| copies-30s | -0.005 (-0.067 to +0.053) | -0.017 | -0.012 |

The gains are small and held across both validation seeds and the test:
with Cubic from the start +0.049, +0.056 and +0.068, in the competition
conditions +0.031, +0.027 and +0.039, each interval above zero. The same
recipe on net1 v5 (round 7e) passed one of 19 gates there, but on test
its competition gain shrank to +0.019 (-0.015 to +0.051), with +0.042
(+0.004 to +0.083) with Cubic from the start
([`exp-net2-test-r7e-k0.5-134`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-test-r7e-k0.5-134)).
Reinforcement learning from the reward gains more against competitors
but pushes among copies (below). What worked, twice, is imitation of a
privileged teacher, explicit beliefs about the other flows, and averaging
the seeds.

The candidate kept most of its gain on a net1 it was never trained on.
net1 v6's other two seeds share seed 0's LCFs, bins and input scaler, and
the unchanged policy on them, each against the rule on the same seed
(validation seed 11,
[`exp-net2-swap-v6-169`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-swap-v6-169)),
had +0.036 and +0.046 with Cubic from the start (+0.051 on seed 0) and
+0.031 and +0.034 in the competition conditions (+0.035), but with
intervals reaching below zero, and -0.063 (-0.145 to +0.008) and +0.041
as copies. net2 acts mostly on what the beliefs mean; another seed costs
part of the margin.

On net1 v6, PPO from the reward gained more against competitors than any
other net2 (up to +0.23 with Cubic from the start) and lost 0.22 to 0.44
as copies, averaged over seeds or not; more training made it worse, and a
veto by round 18's copies belief halved the copies loss but not enough
(rounds 14, 15 and 17;
[`exp-net2-eval-v6-159`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-159), table in
[rounds.md](rounds.md)).

Reinforcement learning from the reward follows its training mixture.
With noise drawn anew at each decision every learner stayed at the rule;
with noise held 200 ms PPO learned to push, which pays where competitors
fill the mixture and costs alone and among copies; a longer horizon pushed
harder, and longer episodes did not help at equal compute (rounds 1 to 16,
[rounds.md](rounds.md)). Imitation of a privileged teacher learns state
dependence instead: DAgger students learn from net1's output p, that a
stock TCP flow shares the link, and q, that copies of itself do. With q
vetoing the push, `u = u_rule + p'(1 - q')(share - u_rule)`, mistaking
Cubic for copies costs at most the push, and one policy on three seeds'
mean beliefs passed on net1 v5 and again on v6. Following the cooperative
window where q believes copies instead trades Cubic for copies on v6,
even averaged over seeds: +0.211 (+0.028 to +0.398) as copies but -0.156
(-0.288 to -0.037) with Cubic from the start
([`exp-net2-eval-v6-coop-176`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-coop-176)).
Yielding to Cubic costs more than pushing among copies, so the veto,
which only withholds the push, is the safer use of an imperfect q.

A better q moves the trade-off but does not remove it. With q from net1's
traffic head (round 21, above) in place of the students', each minus the
rule at campaign seed 21
([`exp-net2-eval-v6-head-179`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-head-179),
[`-head2-182`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-head2-182)):

| q from | Uses q to | Five-family mean | cubic-first-60 | copies-30s |
|---|---|---|---|---|
| the students (r18e05) | veto the push | +0.017 | +0.049 | -0.022 |
| net1's head, past 0.7 | veto the push | +0.018 | +0.051 (+0.009 to +0.100) | -0.002 |
| the students | cooperate | | -0.156 | +0.211 |
| net1's head | cooperate | -0.012 | -0.038 | +0.290 (+0.097 to +0.510) |
| net1's head, past 0.5 to 0.9 | cooperate | +0.000 to +0.005 | +0.005 to +0.011 | +0.204 to +0.315 |
| net1's head, p from it too (no learned output) | cooperate | -0.018 | -0.093 (-0.282 to +0.054) | +0.418 (+0.211 to +0.617) |

The copies belief can live in net1 at no cost. Cooperating on it repairs
most of the Cubic loss and gains more as copies, net2 unchanged, but gives
up the gain against Cubic at every threshold, with wide intervals: a flow
that yields to Cubic by mistake seems not to win its share back
(inferred).

Every gate, evaluation and training run of each round, on net1 v5 and
v6, is listed in [rounds.md](rounds.md).

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
  time, and inference that costs no virtual time. The runs above used
  `agent`, which paces once a step at cwnd * MSS / srtt; Sage's kernel
  paces every ACK at max(cwnd, packets_out) * MTU / srtt (read from its
  binary). `agent_pure` now does exactly that
  ([validation.md](validation.md)); its runs are UNRUN at scale.
- The policy computation is not a divergence: on our recorded messages,
  expliCCit's own deployed code takes our action at 11,972 of 11,973
  steps (local run, [validation.md](validation.md)).
- Sage and expliCCit's policies run on links they were not trained on,
  and expliCCit's ground-truth definitions are not in its code, so ours
  may differ: on our expliCCit-like data its released net1 had R^2 1.000
  for one-way delay, 0.361 for capacity, -0.743 for busy fraction and
  -59.0 for free buffer (local run).
- The emulator itself matches Mahimahi: the link model equals a port of
  Mahimahi's link queue event for event, and the state message equals
  `sage.cc`'s at every compared step ([validation.md](validation.md)). The
  run of the same four links on the original stack, which would separate
  the emulator from the rest, is UNRUN: the kit is `tools/crosscheck`, and
  palaestra's side is
  [`exp-crosscheck-93`](https://github.com/Radonux/palaestra/releases/tag/exp-crosscheck-93)
  (expliCCit's released policy uses 0.50 of the case-study link there,
  Sage 0.92).

### What the released policy's decisions are attributed to

Each decision of the released policy, as deployed (beliefs clamped and
rounded), was split into exact Shapley values over its seven LCFs: 2^7
evaluations per decision, against the mean LCF vector of the same
campaign's steps. The values sum to each decision's change within
2.4e-8 on every campaign
([`exp-expliccit-attribution-190`](https://github.com/Radonux/palaestra/releases/tag/exp-expliccit-attribution-190),
validation links, every fifth step).

| Campaign | Steps | Share of a decision's total abs value held by its largest three LCFs | Largest mean shares |
|---|---|---|---|
| Case Studies 1 and 2's links | 3,178 | 0.82 | one-way delay 0.36, receive rate 0.18, capacity 0.17 |
| New family | 45,085 | 0.77 | one-way delay 0.20, capacity 0.20, queue delay 0.18 |
| Cellular traces | 8,189 | 0.80 | queue delay 0.35, queue 0.19, one-way delay 0.13 |
| Rising capacity | 11,290 | 0.77 | capacity 0.25, one-way delay 0.18, queue 0.17 |

On the case-study links busy fraction gets exactly 0. As deployed, its
belief is the same at every step there, so it never differs from the
reference. Yet it is the wrong belief on that link (1.00 against 0.51),
and the true value lifts utilization from 0.48 to 0.71 (above). A
Shapley value against a mean reference says how a decision varies with
an input, not whether the input is right. A belief that is wrong but
constant gets no attribution, so attribution alone could not find this
fault; replacing the belief with the truth did. The leave-one-out
intervention tracks the Shapley value closely (correlation 0.83 to 0.99
for every LCF but capacity, 0.60 to 0.90 for capacity), so the policy
is close to additive over its LCFs, and capacity is the one that
interacts most with the others.

## Baselines

Each controller ran as the observed flow on the 27 test links of every
held-out family, and as 2 and 4 copies of itself:

- expliCCit's released policy as deployed;
- Sage's released policy;
- BBRv1, BBRv3, Cubic and Vegas, each with Linux's defaults.

That is 2,928 episodes, and none failed
([`exp-baselines-192`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-192);
report in
[`exp-baselines-report-193`](https://github.com/Radonux/palaestra/releases/tag/exp-baselines-report-193)). Families run for 30 s,
except Cubic first (60 s) and expliCCit's case-study links (80 s). Share
is the observed flow's throughput over the link's mean capacity. Delay is
the mean queueing delay of its packets at the bottleneck, in ms. The
report adds 95% intervals, the 95th percentile of delay and, under
competition, Jain's index among the TCP flows.

| Family: condition | expliCCit | Sage | BBRv1 | BBRv3 | Cubic | Vegas |
|---|---|---|---|---|---|---|
| training: clean | 0.54, 14 | 0.92, 10 | 0.95, 14 | 0.94, 15 | 0.97, 34 | 0.95, 10 |
| training: loss 1% | 0.56, 12 | 0.46, 3 | 0.92, 10 | 0.72, 4 | 0.15, 2 | 0.16, 2 |
| training: UDP from 5 s, Cubic from 8 s | 0.47, 28 | 0.45, 21 | 0.74, 30 | 0.67, 28 | 0.65, 35 | 0.38, 20 |
| new: ACK loss 1% | 0.54, 13 | 0.92, 10 | 0.95, 14 | 0.93, 15 | 0.97, 34 | 0.95, 11 |
| new: loss 0.1% | 0.56, 14 | 0.84, 5 | 0.95, 13 | 0.91, 13 | 0.45, 7 | 0.43, 4 |
| new: loss 3% | 0.59, 12 | 0.40, 3 | 0.91, 8 | 0.34, 2 | 0.07, 1 | 0.08, 2 |
| new: Cubic from 0 s | 0.41, 33 | 0.31, 35 | 0.65, 37 | 0.50, 35 | 0.53, 36 | 0.23, 29 |
| new: Cubic from 2 s and 10 s | 0.46, 33 | 0.31, 31 | 0.67, 38 | 0.55, 34 | 0.53, 37 | 0.25, 30 |
| new: BBRv1 from 4 s | 0.41, 32 | 0.24, 20 | 0.57, 32 | 0.35, 27 | 0.35, 34 | 0.20, 20 |
| cubic-first (60 s) | 0.41, 32 | 0.31, 35 | 0.66, 37 | 0.53, 35 | 0.52, 36 | 0.17, 29 |
| deep-long: clean | 0.28, 8 | 0.86, 98 | 0.95, 70 | 0.94, 83 | 0.98, 489 | 0.98, 200 |
| deep-long: loss 1% | 0.29, 8 | 0.54, 9 | 0.92, 64 | 0.77, 38 | 0.08, 2 | 0.08, 3 |
| deep-long: UDP from 5 s, Cubic from 8 s | 0.16, 145 | 0.36, 217 | 0.57, 226 | 0.62, 247 | 0.84, 503 | 0.61, 327 |
| rise: clean | 0.48, 16 | 0.77, 15 | 0.80, 26 | 0.76, 30 | 0.81, 29 | 0.70, 18 |
| cellular: clean | 0.33, 143 | 0.72, 141 | 0.80, 144 | 0.72, 126 | 0.66, 138 | 0.57, 85 |
| cellular: Cubic from 0 s | 0.30, 160 | 0.47, 133 | 0.69, 149 | 0.44, 131 | 0.37, 127 | 0.23, 108 |
| case studies (2 links, 80 s) | 0.73, 11 | 0.95, 13 | 0.96, 26 | 0.97, 51 | 1.00, 218 | 0.99, 28 |

| Copies | expliCCit | Sage | BBRv1 | BBRv3 | Cubic | Vegas |
|---|---|---|---|---|---|---|
| 2: Jain's index | 0.93 | 0.95 | 0.99 | 0.98 | 0.98 | 0.87 |
| 2: link utilization | 0.72 | 0.97 | 0.95 | 0.95 | 0.98 | 0.96 |
| 2: mean queueing delay (ms) | 16 | 16 | 33 | 28 | 36 | 14 |
| 4: Jain's index | 0.89 | 0.91 | 0.94 | 0.98 | 0.97 | 0.84 |
| 4: link utilization | 0.84 | 0.98 | 0.97 | 0.96 | 0.98 | 0.98 |
| 4: mean queueing delay (ms) | 22 | 21 | 40 | 33 | 39 | 18 |

What the tables say:

- expliCCit's released policy keeps the queue short and leaves capacity
  unused: 0.28 to 0.73 of the link alone (0.54 on clean training links,
  where Sage has 0.92), and 0.72 as two copies. This repeats
  `exp-final-test-100` (above). Its share holds under 1% to 3% random
  loss, where Cubic and Vegas fall below 0.16. On clean deep-buffer
  links it keeps 8 ms of queue where the others keep 70 to 489 ms, at
  0.28 of the link.
- Sage on clean links (0.92 of the link, 10 ms) is close to Vegas (0.95,
  10 ms), with a lower 95th percentile of delay (22 against 29 ms). It
  keeps 0.84 under 0.1% loss, where Cubic and Vegas fall to 0.45 and 0.43.
  In the new family's three competition conditions and with Cubic first,
  it gets 0.24 to 0.31 of the link, the least of any controller but
  Vegas. In clean deep buffers it keeps 98 ms of queue.
- BBRv1 has the most throughput under random data loss (0.1% to 3%) and
  in every competition but one: with deep buffers, Cubic fills them and
  takes 0.84. Against Cubic, BBRv1 takes about two thirds of the link;
  BBRv3 takes 0.50 to 0.55, but keeps only 0.34 under 3% loss, where
  BBRv1 keeps 0.91.
- Among copies, BBRv1, BBRv3 and Cubic are the fairest (Jain's index 0.94
  to 0.99) and Vegas the least fair (0.84 to 0.87). The two learned
  policies sit between them on fairness, and between Vegas and the other
  three on delay.

Orca and Aurora are not in these tables. Orca's released model needs a
port: Cubic under the agent and Orca's per-period statistics from the
kernel, both under way. Aurora has no released model (PCC-RL at
`64bea6e` holds only training code), so it must be trained first. Either
can run later on the same links and seeds and join the report.

## Alarms and repair

Alarms computed without ground truth catch beliefs that the flow's own
measurements contradict, and miss beliefs they are consistent with. A
capacity fault was repaired by retraining net1 alone, with net2
unchanged.

Alarms (R2,
[`exp-r2-alarms-81`](https://github.com/Radonux/palaestra/releases/tag/exp-r2-alarms-81)):
physical invariants among the beliefs, each belief's entropy, and checks
against the delivery rate and min RTT measured three smoothed RTTs later,
thresholds at the 95th percentile on net1 v4's validation links (false
alarms near 5%). On net1 v3 they catch one-way delay on deep buffers
(0.83 of 168 wrong episodes, AUC 0.91), queueing delay at 61 to 90 ms
(0.89 of 90) and capacity under 3% to 6% loss (0.63 of 287), and miss its
capacity under competition (0.12 of 43, AUC 0.52): a capacity believed at
0.59 of the truth is still above the flow's own share, so nothing the flow
measures contradicts it (inferred).

Repair (R3). net1 v3's rule lost to BBRv3 with Cubic from the start
because its capacity belief was low there; the same rule on the true BDP
did not. Retraining net1 on the broad data repaired it with net2
unchanged: on the test links with Cubic from the start, the fixed rule on
net1 v4 beat the same rule on net1 v3 by 0.428 (0.277 to 0.580)
([`exp-rule-net1v4-test-14`](https://github.com/Radonux/palaestra/releases/tag/exp-rule-net1v4-test-14);
under the frozen settings on validation, -0.555 on net1 v3, -0.037 on v4,
+0.058 on the truth, [`exp-r3-capacity-head-val-86`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-val-86)).
Retraining only the capacity head
([`exp-r3-capacity-head-83`](https://github.com/Radonux/palaestra/releases/tag/exp-r3-capacity-head-83))
repairs its calibration (90% coverage 0.62 to 0.93) but recovers +0.16 in
closed loop where all of net1 gained +0.52: a new head on frozen features
cannot supply cues the features never learned (inferred).

Replacing one belief at a time with the truth localizes a fault (busy
fraction, in expliCCit's case study above), but a net2 adapts to its
net1's errors, so a patch can also hurt. For the learned net2 on net1 v6
(validation seed 11,
[`exp-net2-patch-v6-172`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-patch-v6-172)),
six of the seven patches moved it by 0.026 or less in every measure. The
true capacity gained against competitors (the competition conditions
+0.098 against +0.029) and collapsed as copies (-1.247 against -0.008),
as with all seven true (-1.205): each copy then aims at the whole link,
where net1's lower belief on a shared link (above) kept the copies safe.
That belief carries something about the other flows that the LCF leaves
out, and net2 relies on it.

## Stated uncertainty has not helped net2 yet

A net2 on the means of net1's beliefs (arm A2) did at least as well as
one on the full beliefs (A3). Over 60 iterations of online TD3+BC on net1
v3, scored every 10 iterations on the validation links, A3 minus A2 stayed
between -0.030 and +0.001; the one interval that excluded zero favoured
the means (one seed each, local runs, not run on test).

Those links left the spread little to say. The fixed rule alone keeps 96%
of validation steps on a fully busy link, where net1's capacity belief is
narrow (median log2 sd 0.14). Off a busy link the spread widens 3 to 6
times, stays honest (90% coverage 0.95 to 0.98) and tracks the error
(correlation 0.82; run 177's steps, local). Two families put the flow
where it cannot see the capacity: rise (alone, the capacity alternating
between half and twice its level every 2 s) and info (a Cubic or BBRv1
flow leaving at 10 s, UDP at half the capacity on and off every 2 s,
Cubic on and off every 3 s). There, rules that probe above the believed
BDP did no better than the fixed rule, whether they read the spread or
only the means. Each raises the window target by k times a signal: the
spread of log2 capacity past s0, or how idle net1 believes the link is
(1 minus its mean busy fraction, past 0.05). Minus the fixed rule
([`exp-net2-info-val-184`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-info-val-184), validation):

| Rule probes on | rise-30 | info-30 | on/off UDP | five-family mean | cubic-first-60 | copies-30s |
|---|---|---|---|---|---|---|
| believed idle share, k 1 (means) | -0.008 | -0.006 | -0.005 | +0.001 | +0.034 | +0.035 |
| believed idle share, k 2 (means) | -0.021 | -0.007 | -0.016 | -0.004 | +0.036 | -0.027 |
| spread, k 1, s0 0.2 | -0.087 | -0.035 | -0.131 | +0.022 | +0.138 | -0.389 |
| spread, k 2, s0 0.4 | -0.117 | -0.028 | -0.070 | +0.008 | +0.121 | -0.379 |
| spread, k 1, s0 0.2, times 1 - P(copies) | -0.051 | -0.021 | -0.059 | +0.007 | +0.091 | -0.049 |
| (BBRv3, for scale) | -0.627 | -0.161 | -0.115 | -0.142 | -0.007 | +0.100 |

The fixed rule's margin of 19% over a capacity net1 keeps re-estimating
already follows each fourfold rise; extra probing buys less than its
queue costs. net1 is as unsure of the capacity on any shared link
(median spread 0.34 to 0.42 with copies or a stock flow), so probing on
the spread mostly pushes there: it gains against Cubic and loses among
copies and under on/off UDP. For a window rule, the spread has so far
added nothing the means do not.

The learned net2 that passed did not need it either: round 18's recipe
trained on net1 v6's means (round 22) scores as r18e05 on its full
beliefs (+0.016 against +0.018 on the five-family mean, +0.053 each with
Cubic from the start, -0.004 and -0.003 on rise;
[`exp-net2-eval-r23-186`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-r23-186)). PPO trained where
probing can pay (round 23) learned a window that rises where net1
believes the link idle and drops below the rule's where it is busy. That
wins on rise, info and on/off UDP (up to +0.23) and as copies (+0.43),
and loses with Cubic from the start (-0.29 to -0.36). Where net1 is
unsure of the capacity, those policies back off: their u falls with the
spread. Holding exploration 1 s instead of 200 ms did not help. net1's
honesty still serves the human reading of its beliefs.
