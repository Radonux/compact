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
40 iterations brought it from 0.554 to 0.064 to 0.087 below the rule),
the same PPO ended 0.036 to 0.096 below, about 0.05 below the arm on net1
on average and level with it on one seed. Given the split's start, the
raw measurements learn from the reward nearly as well; what the split
gives is that start, a window on an estimate of the path rather than of
the flow's own throughput, which the black box reached only by imitating
the split. Inference is the split's cost: a decision of the policy on net1
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

On net1 v6, PPO gains more against competitors than any earlier net2
and loses more as copies. Each candidate over three seeds, minus v6's
fixed rule on the validation links (campaign seed 21;
[`exp-net2-eval-v6-159`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-159),
[`-v6-ens-160`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-ens-160),
[`-v6-veto-166`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-eval-v6-veto-166)):

| Round | Five-family mean | cubic-first-60 | competition | copies-30s |
|---|---|---|---|---|
| 14, tau 1 s | +0.002 | +0.068 (-0.033 to +0.177) | +0.059 (-0.008 to +0.129) | -0.223 (-0.524 to +0.042) |
| 14, tau 5 s | -0.012 | +0.110 (-0.028 to +0.262) | +0.032 (-0.045 to +0.114) | -0.258 (-0.514 to -0.022) |
| 15, more copies in training | +0.004 | +0.174 (+0.057 to +0.292) | +0.085 (+0.011 to +0.161) | -0.300 (-0.605 to -0.007) |
| 15, one policy on the seeds' mean u | +0.011 | +0.149 (+0.044 to +0.262) | +0.065 (-0.005 to +0.139) | -0.258 (-0.594 to +0.043) |
| 14, stage 2's best checkpoint (iteration 39) | +0.031 | +0.230 (+0.062 to +0.419) | +0.066 (-0.036 to +0.178) | -0.444 (-0.776 to -0.146) |
| 15, push vetoed by round 18's copies belief | +0.017 | +0.097 (+0.027 to +0.166) | +0.053 (-0.012 to +0.121) | -0.143 (-0.291 to -0.022) |
| 18, DAgger students as round 7e, gain 1 | +0.028 | +0.098 (+0.040 to +0.163) | +0.030 (-0.022 to +0.082) | -0.019 (-0.078 to +0.030) |
| 18, the same at gain 0.5 (passed; above) | +0.017 | +0.049 (+0.017 to +0.085) | +0.031 (+0.006 to +0.056) | -0.022 (-0.058 to +0.017) |

Averaging the seeds removed little of PPO's copies loss, so unlike round
7's misfires (below) the seeds share the push: what wins against Cubic
deepens the queue among copies (inferred). More training made it worse:
continued up to 120 more iterations (round 14's stage 2), every cell lost
ground after about 40, mostly where the flow was alone (tau 5 s peaked
at +0.021 and +0.032 on the new family and ended at -0.114 and -0.039;
tau 1 s fell to -0.035 to -0.075), and more copies in training (round 17)
fell the same way. A veto by round 18's copies belief halved the copies
loss but not enough.

Reinforcement learning from the reward follows its training mixture. In
rounds 1 to 3 every PPO and TD3+BC learner stayed at the rule (spread of
u 0.02 to 0.04): noise of sigma 0.05 drawn anew at each decision tells a
return over thousands of decisions too little about any one (rounds 1
and 2 also lost 0.13 as copies to a graph bug, fixed in `85de8d0`,
[`exp-net2-clone-copies-107`](https://github.com/Radonux/palaestra/releases/tag/exp-net2-clone-copies-107)).
Noise held 200 ms at sigma 0.2 (round 10) moved PPO toward larger
windows: +0.05 to +0.07 with Cubic from the start, -0.22 to -0.39 alone,
as competition on 80% of the training links left the flow alone on 8%.
A longer horizon pushed harder (round 11, tau 20 s: windows of 4 to 7
believed BDPs, 1.5 to 2.2 below), always alone the learner beat the rule
there by up to 0.08 (round 13), and alone on half the links it matched
the rule on every family but lost 0.1 as copies (round 12). Longer
episodes did not help at equal compute (round 16: 120 s no better than
30 s at tau 1 s, pushing at tau 5 s), since they hold long stretches
where a competitor yields and give two links per update, not eight
(inferred).

Imitation learns state dependence. DAgger labels each decision with a
privileged teacher's u (the share rule while a stock TCP flow shares the
link, the fixed rule otherwise), which the student must infer from net1's
output. Round 4's students kept 62% to 77% of the share rule's gain
against Cubic for 31% to 34% of its copies loss; from round 4c to 6b a
seed's copies loss (0.02 to 0.16) followed how well it told the cases
apart, since one copy's push looks like a buffer-filling flow to the
others. Round 7's students also learned q, a belief that copies are
present (0.55 to 0.72 with copies, 0.21 to 0.26 elsewhere). Cooperating
with copies cost most of the gain against Cubic, as some seeds took
Cubic for copies and yielded; with q vetoing the push instead,
`u = u_rule + p'(1 - q')(share - u_rule)` (round 7v), mistaking Cubic
for copies costs at most the push. The copies loss left (-0.145) was
single seeds misfiring (in 12 of 42 pairs of link and number of copies
one seed lost 0.5 or more, never all three; 72% of the loss), and one
policy on the three seeds' mean beliefs (round 7e) passed. Neither more
iterations (round 8) nor 16 times the links (round 9) beat it. On net1 v6
the students separate less sharply (q 0.53 to 0.76 with copies, 0.10 to
0.52 elsewhere), and the same recipe passed again.

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
  time; pacing refreshed every step, not every ACK; inference that costs
  no virtual time.
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
the means (one seed each, local runs, not run on test). A likely reason,
not tested: a window target from the believed BDP needs mostly the means,
and these runs keep the link busy, where capacity is identifiable.
net1's honesty still serves the human reading of its beliefs.
