# Validation

This page records every check that ran, with its test or command and its
result, and what has not run. A check ran in this environment unless it
says CI. Anything not run is marked UNRUN.

Every check below that names a test is part of the test suite, and the
suite passed in its last full run (M15), apart from the two Sage
comparisons. A test asserts its property, not the exact numbers quoted
next to it; those numbers come from the run of the milestone where the
entry was written. Entries without a test are one-off runs, each tied to
its milestone; they describe the build of that milestone, not necessarily
the current one.

## Summary (M16, 2026-11-02)

The suite has 117 tests: 80 unit tests and 37 that need a UML build or
another checkout. On 2026-11-02, with a UML build at the multi-agent change
and without Mahak, 109 passed and 8 skipped (6 need Mahak, 2 are the Sage
comparisons).

Before that change, the suite had 105 tests: 77 unit tests and 28 that
need a UML build.
The last full local run with a Mahak checkout had 93 of them: 91 passed,
and the other 2, the comparisons of Sage's step with `sage.cc`
(`test_step_matches_sage_cc`), skipped: UNRUN. On 2026-11-02 the 94 tests
of that day ran with a UML build but without Mahak: 86 passed, and 8
skipped (6 need Mahak, 2 are the Sage comparisons). The 11 tests of the
experiment file parser (`tests/test_ci_experiment.py`) came after that
run and pass.

The passing tests show that:

- the kernel's link model equals an independent port of Mahimahi's
  `LinkQueue`, event for event, in both directions, also under random loss
  and cross traffic;
- trace generation, the `mm-th-delay` metrics and Mahak's driver equal
  Mahak's own code;
- the same build, plan and seed give byte-identical record files;
- the trainer loads a campaign's steps in one order (episode id, then
  step), whatever order the workers finished in; before `be72bd7` it kept
  the shards' order, so a net1 retrained on regenerated data could differ
  slightly in its minibatches;
- the closed loop applies Sage's window rule exactly, and a policy acts on
  exactly the message and the truth that its step recorded;
- the truth a policy can read equals the host's labels at every step;
- any number of policies drive their own sockets on one bottleneck, next to
  stock and UDP flows, reproducibly, and single-agent episodes are
  unchanged by that support.
- a distribution net1 states calibrated probabilities on data with known
  noise, and its exported policy, patched or not, runs in the guest as
  torch computes it;
- held exploration noise reproduces the per-step stream block by block,
  and a policy that reads the sender's window sets an absolute window.

A one-off check (M6, repeated at M7) showed that adding BBRv3 left seeded
Cubic and BBRv1 episodes byte-identical.

The main gaps, detailed under "Not run": reference runs of real Mahimahi
against the link model are prepared but not run, and UML's virtual time is
never checked against a real-time kernel.

## Environment and how to rerun

- Host: x86_64 cloud container, 4 vCPU, 15 GiB RAM, host kernel 6.18.44,
  Ubuntu 24.04 userland, gcc 13.3.0, Rust 1.97.0, Python 3.13, perl 5.38.
- Kernel: UML from `third_party/linux` (Linux 7.0 plus the patches in
  `third_party/README.md`), `kernel/uml.config`, `time-travel=inf-cpu`.

To rerun the suite:

    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    uv sync --extra train                  # the trainer's tests need PyTorch
    git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C /tmp/mahak checkout 5077583
    EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q

Without `EXPLICCIT_UML_DIR` the 28 UML tests skip, and without
`EXPLICCIT_MAHAK_DIR` the tests that run Mahak's own code skip. The
comparison with `sage.cc` also needs `EXPLICCIT_SAGE_DIR` (a Sage checkout
at `eedd281`; the full hash is in `tests/sage_ref.py`) and g++.

## The link model and the ports (test suite)

| Property | Check | Result |
|---|---|---|
| The trace generator equals Mahak's | `test_mahak_trace_matches_helper_py`: Mahak's own `helper.py` writes the file, and ours must match it bit for bit; 5 settings, including float-rounding traps (7 x 0.3, 45 x 1.7) | pass |
| The metric port equals the Perl script | `test_mm_th_delay_matches_perl`: 250 generated logs (175 defined, 75 where Perl dies), output compared as the exact `%.15g` strings | pass |
| The kernel link equals Mahimahi's `LinkQueue` | `test_kernel_link_matches_mahimahi_oracle`: 14 s Cubic, BBRv1 and BBRv3 episodes cross Mahak's 12 s step (20 to 4 Mbit/s) into a 30-packet buffer. Every kernel arrival (with the queue it found), departure, drop and per-ms opportunity record equals an independent Python port of `LinkQueue` fed the same arrivals. | pass, both directions |
| The delay line is exact | `test_ack_delay_line_is_exact`: each ACK reaches the sender between 0 and 2 us after its departure plus the configured delay (the 256 ns UML tick) | pass, 0 to 1.3 us |
| Mahak's metrics from kernel records equal Perl's | `test_mahak_metrics_agree_with_perl` | pass |
| Oracle laws | `test_link_oracle_laws` (Hypothesis, 300 cases): conservation, FIFO, never departing in the arrival ms, droptail bounds, opportunity accounting | pass |
| Record ABI | `test_dtypes_match_c_structs`: every numpy offset equals the C `offsetof`; `test_load_rejects_sequence_gap`, `test_load_rejects_truncated_file` | pass |
| Reproducibility | `test_same_seed_is_bit_identical` (BBRv1) | byte-identical records for one seed; different records for another |
| Spec laws | `test_spec.py`: a delay is exactly one of mRTT and base RTT; unknown fields are rejected; campaign expansion is deterministic; loss probabilities lie in [0, 1); cross-flow windows lie inside the episode; conditions multiply a campaign without changing its plain arm; Mahak's mRTT is one-way; loss and cross traffic reach the guest plan; a paths sweep takes any trace, and conditions override its loss; episode ids are content hashes; the examples parse | pass |
| Trace helpers | `test_dense_counts_round_trip`, `test_mahak_trace_levels` (5 settings), `test_periodic_trace_has_exact_rates_and_period`, `test_mm_log_format_round_trip`, `test_mahak_first_power` | pass |

## Closed loop and pipeline (test suite)

| Check | Result |
|---|---|
| `test_hold_agent_follows_sage_window`: constant a = 0.1, `agent` | every window set equals Sage's accumulator (from 10, times 2^a, rounded up, capped at 20000); one step per 10 ms; column 76 of the next step reports a; the socket holds the set window |
| `test_policy_reads_the_observation`: a = 0.5 x70 - 0.35 | every action equals the policy applied, in float32, to the very message the step table recorded |
| `test_recurrent_state_round_trips`: a counter in the state | the state is fed back each step: a = 0.001 k exactly |
| `test_reno_agent_grows_between_actions`: `agent_reno` | Reno regrows the window between actions |
| `test_bad_policy_output_fails_the_episode`: a = 20 | an explicit failure row, no data |
| `test_step_table` (Cubic, 24 then 12 Mbit/s): steps, labels at steps, reward | column 76 equals the logged window ratio; the action is the next step's column 76; capacity labels are 24 and 12 Mbit/s within 2%; busy fraction lies in [0, 1]; the reward is finite and peaks above 10 (Sage's scale tops out at 25) |
| `test_step_record_matches_the_guest` | the Python step record layout equals the Rust constants |
| `test_campaign_shards_and_tables`: 8 episodes in 2 shards | all ok; obs and labels are row-aligned |
| `test_cli_episode_and_mahak_query` | a CLI episode and the Mahak oracle replacement work |

## Speed and storage (M0, and before M6)

Measured before M6, once the link model existed: one core per guest, 20 s
episodes, wall time including boot. `docs/m0-report.md` (item 11) has every
row but the 70 Mbit/s one, which is the probe grid of the next section.

| Setting | Wall time | Speed |
|---|---|---|
| 5 Mbit/s, 2 to 4000 pkts | 0.37 to 0.67 s | 30 to 53x |
| 70 Mbit/s grid below | 0.55 to 0.97 s | 21 to 36x |
| 100 Mbit/s, 10 to 100 ms one-way, up to 20000 pkts | 1.47 to 1.52 s | 13 to 14x |
| 100 to 800 Mbit/s step, 8000 pkts | 2.42 s | 8x |

Peak RSS per guest: 85 to 117 MiB.

Aggregate throughput, measured on 2026-10-30, before M6 and before the step
loop existed: shard 0/50 of `examples/campaign-icnp25-sample.json` (40
episodes drawn log-uniformly from the ICNP'25 space, 20 s each, Cubic and
BBRv1) took 6.6 s with 4 workers on 4 cores, and every episode was ok. That
is about 5400 episodes per core-hour for that mix. The output was 98 MB of
zstd Parquet, about 2.5 MB per episode (link_events 60%, obs 24%, link_opp
14%, labels 4%).

Planning consequence (extrapolated, not measured at scale): one 512-core
node would produce on the order of 2.8 million episodes and 7 TB per hour,
so storage, not CPU, bounds campaigns. `expliccit campaign run --tables`
keeps only the named tables, and dropping `link_events` and `link_opp`
saves most of the space. For example, the 360-episode teacher campaign of
`docs/training.md` takes 216 MB with `--tables steps`.

## An illustrative Mahak-style probe (before M6, one seed)

This is one seed per cell, not a reproduction. 70 Mbit/s, step 0.2 (a 5x
drop at 12 s) or 1.0, mRTT 4 or 40 ms, buffer 100 or 8000 packets, 20 s,
metrics over the whole episode as Mahak computes them:

| cc | step | mRTT | buffer | util | mean qdelay ms | delay/mRTT | first_power | retrans |
|---|---|---|---|---|---|---|---|---|
| bbr | 0.2 | 4 | 8000 | 0.983 | 7.2 | 1.79 | 0.549 | 0 |
| cubic | 0.2 | 4 | 8000 | 0.997 | 408.4 | 102.10 | 0.010 | 0 |
| bbr | 0.2 | 40 | 8000 | 0.950 | 69.8 | 1.75 | 0.545 | 0 |
| cubic | 0.2 | 40 | 8000 | 0.960 | 379.8 | 9.49 | 0.101 | 0 |
| bbr | 0.2 | 40 | 100 | 0.940 | 10.7 | 0.27 | 3.524 | 5015 |
| cubic | 0.2 | 40 | 100 | 0.932 | 13.9 | 0.35 | 2.689 | 716 |

The full 16-row grid is `examples/campaign-bbr2-probe.json`; its 16
episodes took 4.9 s on 4 cores. Mahak reports BBR2 failing in the first
cell, while BBRv1 here keeps 7 ms of queueing delay. That is consistent with
the failure being specific to BBR2, but one seed is not a map.

## BBRv3 (M6)

Google's BBRv3 is imported from google/bbr (`third_party/README.md`): 12
core TCP prerequisites plus the `tcp_bbr3` module, ported to 7.0.

**Stock congestion control stays unmodified.** With kernel build metadata
pinned (so two builds of different source share one random stream per
seed), four seeded 14 s episodes (Cubic and BBRv1, step and constant
traces, 20 to 2000-packet buffers) were run on the kernel without the
series and on the final kernel, and every record file was compared byte for
byte. The steps that led there:

| Kernel | Cubic, shallow | BBRv1 x2 | Cubic, deep (send-buffer-limited) |
|---|---|---|---|
| all 12 prerequisites, Google's code unchanged | identical | identical | links identical; `TCP_OBS` app-limited fields differ |
| plus `tcp_bbr3` built in (`=y`) | identical | **differ** | differ |
| final: app-limited checks opt-in, `tcp_bbr3` a module loaded only for `bbr3` | **identical** | **identical** | **identical** |

Two mechanisms were found and removed. The broadened app-limited checks
changed rate-sample marking for every socket; they are now opt-in per
congestion control. Building BBRv3 into the kernel changed seeded BBRv1 runs
through boot-time work alone; it is now a module. The procedure needs two
kernel builds, so it is not in CI. Rerun it when the kernel patch set
changes.

**BBRv3 runs correctly over the link model.** `bbr3` is a case of
`test_kernel_link_matches_mahimahi_oracle`, which also asserts that the
socket really ran `bbr3`: pass.

**First look** (one seed per cell, not a map): BBRv1 against BBRv3 on the
probe grid, 70 Mbit/s, 20 s.

| cc | step | mRTT | buffer | util | mean qdelay ms | first_power | retrans |
|---|---|---|---|---|---|---|---|
| bbr | 0.2 | 4 | 100 | 0.983 | 6.5 | 0.609 | 287 |
| bbr3 | 0.2 | 4 | 100 | 0.986 | 5.5 | 0.721 | 30 |
| bbr | 0.2 | 4 | 8000 | 0.982 | 6.7 | 0.583 | 0 |
| bbr3 | 0.2 | 4 | 8000 | 0.988 | 14.6 | 0.271 | 0 |
| bbr | 1.0 | 40 | 100 | 0.942 | 3.8 | 9.897 | 2735 |
| bbr3 | 1.0 | 40 | 100 | 0.951 | 2.8 | 13.480 | 941 |

BBRv3 retransmits far less with shallow buffers, as designed. In Mahak's
BBR2 failure cell (5x drop, 8000 packets, mRTT 4 ms), its queueing delay is
about twice BBRv1's, the direction Mahak reported for BBR2. A map over
seeds is needed before calling it a finding.

## Random loss and cross traffic (M7)

`test_loss_and_cross_traffic`: BBRv1 on 20 Mbit/s, 20 ms base RTT, a
60-packet buffer, 10 s; 1% random loss on data and 0.5% on ACKs; 5 Mbit/s
of constant-rate UDP over [2 s, 8 s) and a Cubic flow from 6 s.

| Check | Result |
|---|---|
| The kernel link equals the `LinkQueue` oracle (both directions, all flows) | pass |
| Random drops are Bernoulli per entering packet (within 5 sigma) | data 176 of 16916 (1.04%), ACKs 37 of 7251 (0.51%) |
| Every flow crossed the bottleneck; `TCP_OBS` holds only the observed flow | pass |
| The UDP sender wrote exactly 2500 packets (one per 2.4 ms for 6 s) | pass |
| `cross_traffic_share` label: 0 before 2 s; UDP alone; UDP plus Cubic; Cubic alone | median 0, 0.247, 0.359, 0.145 |
| The same seed, run a second time | byte-identical record files |

The UDP-only share is 0.25 (5 Mbit/s of 1504-byte units on 20 Mbit/s),
less the 1% lost before the link. Cubic gets a small share because 1%
random loss throttles it, while BBRv1 does not back off on loss.

**Loss-free, single-flow episodes are unchanged.** The four seeded stock
episodes of the BBRv3 check were rerun on the M7 build and compared with
the final BBRv3 build's output. Every record file is byte-identical, and
the `LINK_STATS` counters are identical apart from the appended
`drops_random` field (0) that came with record ABI 3.

## Mahak's driver (M8)

**The port equals Mahak's own driver.** `test_driver_matches_mahak_online`
runs Mahak's unmodified `mahak_online.py` in a separate environment,
through `tests/mahak_ref.py`, which only replaces `helper.oracle_query`
with a synthetic function and seeds numpy. It then runs the port on the
same oracle and seed over `run_mahak.sh`'s space (51000 points, budget 15).

| Mahak's environment | Queries | Map (posterior mean) |
|---|---|---|
| scikit-learn 1.7.2, numpy and scipy as locked here (the test) | 16 of 16 identical | identical |
| Mahak's pinned one: Python 3.9, scikit-learn 1.1.3, numpy 1.23.5, scipy 1.9.3, pandas 1.5.2 (run once) | 16 of 16 identical | max difference 1.4e-13 |

`test_grid_follows_mahaks_arange` pins the grid quirks the port keeps (axes
can pass their maximum, and a float step can add a point), and
`test_failed_queries_never_become_labels` pins the failure rule.
`test_cli_mahak_driver` runs `expliccit mahak` end to end in UML.

**First maps** (`run_mahak.sh`'s command, first_power, one core): for each
of Cubic, BBRv1 and BBRv3, a map of 21 episodes took 13 to 16 s of wall
time, with no failed query. Each map was then compared with 60 held-out
grid points run in UML:

| cc | budget | Spearman | Pearson | MAE | RMSE | lowest predicted value |
|---|---|---|---|---|---|---|
| cubic | 20 | 0.50 | 0.69 | 0.90 | 1.82 | -1.16 |
| cubic | 100 (84 s) | 0.90 | 0.56 | 0.56 | 1.86 | -0.63 |
| bbr | 20 | 0.81 | 0.78 | 3.49 | 5.16 | -1.23 |
| bbr3 | 20 | 0.74 | 0.73 | 3.06 | 3.92 | -0.26 |

At Mahak's default budget the maps rank regions only roughly and predict
impossible negative powers. A budget of 100 ranks well (0.90), while a few
very high-power points (small mRTT) keep the RMSE up. This is the
algorithm's behaviour, not the port's. A UML query costs under a second, so
budgets in the hundreds are cheap.

## Sage's step and the trainer (M9)

`tests/test_train.py`:

- expliCCit's shapes match its released checkpoints (1,265,927 and 68,097
  parameters);
- the input stage ignores non-finite values when fitting and maps them to 0
  when running;
- net1 learns a synthetic capacity (R^2 > 0.9 on held-out episodes), and
  net2 trains;
- the deployed rounding is exact;
- the export is a `sage-v1` graph without IsInf or IsNaN;
- a path's episodes share one split, and with a link split so do a link's
  episodes under every condition;
- a freshly exported graph runs in the guest's closed loop (`tract`), with
  its actions within the action bound.

On real data at M9 (the 144-episode campaign of `docs/training.md`,
section 1, without its loss condition; one seed): our net1 reached R^2 0.87
to 0.997 on held-out paths for six of expliCCit's seven LCFs (free buffer
0.50). expliCCit's released net1 on the same steps got 0.36 to 1.0 on five
and negative R^2 on free buffer and busy fraction. Its released policy ran
in our closed loop at 0.58 to 0.95 utilization. Our first net2 collapsed
the window. Later results, tuned and over seeds, are in
`docs/training.md`, section 7.

### The port of Sage's step against `sage.cc`

`test_step_matches_sage_cc` (`tests/sage_ref.py`) compiles Sage's own step
code, taken by line range from `sage.cc` at `eedd281` with only the clock
and the `TCP_INFO` read stubbed. It replays the recorded readings of a
Cubic episode and a BBRv1-with-loss episode, and compares every column
printed with `%.7f` (all but the four the port does not take from
`sage.cc`).

Its first run, at M9, died with SIGSEGV. The cause is in Sage's
`define.h`: three assignment operators and the window's `add()` end
without returning a value, undefined behaviour that g++ 13 compiles into
a trap (an illegal instruction even at -O0, a crash at -O1). No caller
reads those values, so the harness now compiles a copy of `define.h` with
the four returns g++ suggests added, and nothing else changed. On
2026-11-04, with a Sage checkout at `eedd281` and g++ 13.3, both
comparisons passed: every compared column of every step (1,382 on the
Cubic episode) equals `sage.cc`'s output as printed. The test skips unless
`EXPLICCIT_SAGE_DIR` is set, so CI does not run it.

### Sage's released policy under `sage-v1`

Sage's policy (its released `models/ckpt-1`, MIT-licensed) is a GRU of
1,024 units between two layer-normed encoders, a residual MLP and a
five-component Gaussian mixture, whose mean, clipped to [-2, 2] and
rounded to 3 decimals, is the log2 of the window multiplier.
`experiments/sage_weights.py` builds it with Sage's own stack (TensorFlow
2.4.1, sonnet 2.0.0, acme 0.2.0) exactly as its evaluation actor does,
restores the checkpoint, and records its actions on the 1,382 steps of a
recorded Cubic episode; `experiments/sage_export.py` rewrites it in torch
and exports it. On 2026-11-04 the torch version matched TensorFlow within
2.9e-6 at every step, and the exported graph, run with its state fed back,
gave TensorFlow's rounded action at 1,380 of the 1,382 steps (the other two
differ by 0.001, where the two float32 computations straddle a rounding
boundary). It runs as a `hold` agent in UML. The exported policy is
published as
[`exp-sage-model-24`](https://github.com/Radonux/palaestra/releases/tag/exp-sage-model-24),
built in CI from Sage's checkout at `eedd281` with the same check, and its
validation and test runs (`exp-sage-val-25`, `exp-sage-test-26`) are in
`docs/training.md`, section 7. Where they depart from Sage's own setting:
Linux 7.0 in place of Sage's patched 4.19 kernel, pacing refreshed at each
step in place of each ACK, no inference delay, and links Sage was not
trained on.

### Per-flow, per-second bytes and delays (`flow_seconds`)

`test_loss_and_cross_traffic` checks the table on an episode with a
constant-rate UDP flow and a Cubic flow: each flow's bytes over its seconds
equal its `flow_bytes`, the UDP flow's packets fall in the seconds it was
on, and the delays over every flow and second give exactly the episode's
Mahimahi mean delay.

### Cross flows with RTTs of their own (`sch_flowdelay`)

`test_a_cross_flow_has_its_own_rtt` runs three agents that hold 10 packets
each on a 30 Mbit/s link with a 20 ms base RTT (a BDP of 50 packets, so
nothing queues), one of them with `extra_rtt_ms` 30. The minimum RTT each
flow's step message reports is 20 to 22 ms for the two others and 50 to 52
ms for that one; its base RTT label is exactly 50 ms, and its one-way delay
label and live truth exactly 25 ms, while the others keep 20 ms. In a local
check on 2026-11-04 with two Reno cross flows for 8 s on the same link, the
flow with 30 ms more RTT carried 3.4 times fewer bytes than the other (6.4
against 21.8 MB) and with 150 ms more, 33 times fewer; without extra RTT the
two carried 18.8 and 10.0 MB. Seeded episodes of ccperf's Cubic-fairness,
single-flow and start-time-fairness campaigns, which have no extra RTT, gave
the same `flow_seconds` tables, cell for cell, before and after the qdisc
was added.

### TCP buffer limits (2026-11-04)

Linux sizes TCP's buffer limits from memory, so in the 1 GiB guest the send
buffer could hold at most 4 MB: a flow could not keep more than that in
flight. On a real host with Sage's or ccBench's setup scripts the limit is
64 MB. Of ccBench's 150 constant single-flow cells, 31 need more than 4 MB
in flight to fill their buffer (7 more than 16 MB, none more than 64 MB).
The guest now sets those scripts' limits (`docs/architecture.md`). Rerun
with it, the same seeded specs of ccBench cells from `exp-ccbench-27`
(which ran before the change) gave:

| Cell | Needs in flight | Before: utilization, mean and max queueing delay | After |
|---|---|---|---|
| Cubic, 12 Mbit/s, 10 ms RTT, 1 BDP | 0.02 MB | 1.002, 7 and 10 ms | the same |
| Cubic, 48 Mbit/s, 160 ms RTT, 8 BDP (full: 1,280 ms) | 8.6 MB | 0.965, 347 and 509 ms | 0.964, 711 and 1,280 ms |
| Cubic, 96 Mbit/s, 160 ms RTT, 4 BDP (full: 640 ms) | 9.6 MB | 0.959, 133 and 206 ms | 0.959, 445 and 640 ms |
| Reno, 96 Mbit/s, 160 ms RTT, 4 BDP | 9.6 MB | 0.959, 142 and 213 ms | 0.959, 276 and 640 ms |
| Cubic, 192 Mbit/s, 160 ms RTT, 16 BDP (full: 2,560 ms) | 65 MB | 0.722, 34 and 76 ms | 0.953, 727 and 1,407 ms |
| BBRv1, the same cell | 65 MB | 0.673, 2 and 38 ms | 0.930, 22 and 315 ms |

At 192 Mbit/s and 160 ms the BDP alone (3.8 MB) is about the old limit, so
the old runs lost throughput too. `test_a_flow_can_fill_a_deep_buffer_on_a_long_fat_path`
runs the Reno cell for 4 s: its queue reaches the full 640 ms, and with the
old 4 MB send limit set back through the spec's `sysctls` it stops at 213
ms.

What ran before the change: none of the train, validation or test links of
the training, new and cubic-first families needs more than 3.5 MB, but the
deep-long family (2.5 times the delay, 4 to 16 BDP) does on 8 of its 21
validation links and 10 of its 27 test links, so earlier deep-long rows
(for example in `exp-rule-net1v4-test-14`, `exp-share-test-18` and
`exp-sage-test-26`) understate how far buffer-filling controllers queue
there. In net1 v4's broad training data about 7% of links need more than
4 MB (1.7% have a BDP above 3 MB), and 22% and 34% of the out-of-distribution
probes ood-rtt and ood-buffer. Labels were exact either way; what changes
is which states occurred. `exp-ccbench-27` and `exp-ccperf-29` ran before
the change.

Reruns (scheduled 2026-11-04; each result replaces the old one where it is
cited). The six deep-long reruns are done, on the same episode ids, with no
failed episode (`exp-*-deeplong-32` to `-37`, `docs/training.md`,
"Deep-long with the original setups' TCP limits"): on deep-long the rules
move by at most 0.02 on the test links and 0.06 on the validation links
(their five-family means by at most 0.004 and 0.013), loss-based
controllers fall by 0.2 to 0.9, and every validation choice stands.

With 64 MB, a policy whose window grows without bound floods the link,
and only the send buffer stops it. One 60 s episode of the 60 s data (a
threshold policy with theta 0.1 on 49 to 196 Mbit/s, 5 ms one way, a
26-packet buffer) wrote 1.75 GB of arrival records and 1.72 GB of drop
records against 0.03 GB of departure records, and seven of that run's
shards lost their runner to it and its like. Those are record bytes, 64 a
packet; rerun locally on 2026-11-04 and counted from the records, the
downlink saw 27.2 million arrivals (40.7 GB of packets, 5.4 Gbit/s on
average) and 26.8 million drops against 378,000 departures (76 Mbit/s), 71
arrivals per departure. The flood comes from the window: a host with the
setups' 64 MB would flood the same way, so the limit that follows is a
deliberate choice of how far a window may run, not the removal of an
artifact. What is particular to this environment is the rate, since
sending here costs no virtual time; whether a real sender and Mahimahi's
single userspace process pass 450,000 packets a second is UNVERIFIED. In
the rerun 30 s data, 99 of 14,400 episodes (all policies) sent more than 3
times the capacity, against none before. The send buffer is therefore sized to the path (`docs/architecture.md`):
16 times what the path holds, at least 1 MB and at most 64 MB. With it the
same episode records 290 MB of arrivals. `test_a_window_far_beyond_the_path_is_bounded`
runs an agent that grows its window every step on 50 Mbit/s, 10 ms and 26
packets for 4 s: 8 arrivals per departure with the cap, 54 with 64 MB. Rerun
with the cap, 148 episodes that the old regime let finish normally (six
ccperf campaigns with Cubic, BBRv1 and Reno, and 43 of the broad data's
episodes with a stock observed flow) gave the same `flow_seconds` tables,
cell for cell (10,070 cells).

That check was too narrow to show that the cap binds only floods. The 30 s
data rerun with the cap (`exp-data-net1-broad-30s-49`) against the one
without it (`-38`), the same 14,400 ids, changed 1,482 episodes:

| Episodes | Changed | Utilization change of those changed | Drop fraction without the cap (median) |
|---|---|---|---|
| a policy among the flows | 1,372 of 3,788 | mean -0.050, 449 by more than 0.05, at most -0.61 | 0.93 |
| Hybla among the flows, no policy | 98 of 929 | mean -0.002, one by more than 0.05 (-0.094) | 0.05 |
| any other | 12 of 9,683 | mean -0.004, at most -0.032 | 0.003 |

By policy, the 1,372 changed policy episodes are threshold-0.1's 573 of
616 (567 of them dropped more than half of what reached the link),
threshold-0.3's 411 of 656 (407), threshold-0.6's 165 of 592 (159),
threshold-0.9's 4 of 600 (4), and the fixed rule's (u = 0.25 on net1) 219
of 1,324 (none; mean utilization change 0.0000, not traced further). The
floods are the delay-threshold teachers' (`examples/net1_varied.py`): each
grows the window by 2^0.01 per 10 ms step until min RTT over smoothed RTT
falls below theta, a standing queue of (1/theta - 1) base RTTs, and reacts
to nothing else, loss included. Where the buffer holds less than that
queue, drops keep it from forming and the window grows for the rest of the
episode; the target is 9 base RTTs for theta 0.1 and 0.11 for theta 0.9.
The data generation is left as it is: these episodes are what the teachers
do.

Local reruns (2026-11-04) of 18 of them reproduce both releases bit for bit.
Three Hybla and two policy episodes give release 38's result with the
send buffer at 64 MB and release 49's with the cap, so the cap is the
cause there. The 12 others give release 38's result at its commit
(`1a9ed5a`) and release 49's at the cap's, with or without the cap: what
moves them is the plan's added `tcp_wmem` write itself, not its value
(not traced further; M10 below found guest changes shifting the kernel's
random stream). So the cap binds floods (83% of the changed policy
episodes dropped more than half of what reached the link, and the most
changed, a threshold policy on a 10-packet buffer at the 1 MB floor, fell
from 0.95 to 0.34) and Hybla's slow start, which grows the window by 2^ρ - 1
packets per ACK (ρ the RTT over 25 ms) and overshoots the path on long
RTTs. With either
setting a flood is unlike the original setups', whose senders could not
push 64 MB per round trip into Mahimahi, so a limit stays; but 16 times the
path and a 1 MB floor were a guess, well above what host tuning gives a
small path.

The limit now starts from the host-tuning rule (`docs/architecture.md`).
That rule sizes a socket buffer to the bandwidth-delay product, bandwidth
times RTT (ESnet fasterdata, "Background Information", edited 2024-01-07:
"2 * bandwidth * delay" with one-way delay, or "bandwidth * RTT" from ping;
its Linux page, edited 2025-03-06, sets 32 to 64 MB for 10 Gbit/s hosts).
Here the bandwidth is the peak capacity over 1 s and the RTT that of a full
queue, which gives N, the packets the path holds. Linux counts the buffer
in memory: `tcp_sndbuf_expand` gives a window of W segments 2 W (BBR 3 W)
times `per_mss`, 2304 bytes in this build, read by compiling against its
headers (`sk_buff` 256 and `skb_shared_info` 320 bytes after alignment,
`MAX_TCP_HEADER` 256, MSS 1460, so roundup_pow_of_two(2036) + 256).

A first version granted a window of N (`428b121`) and touched ordinary
flows. On 375 episodes sampled from the 30 s data (seed 7) and paired with
release 49 it changed 58 of 160 stock episodes (none by more than 0.05 in
utilization, mean -0.0002), 109 of 120 with BBR on a buffer shallower than
the BDP (2 by more than 0.05), 41 of 60 with Hybla and 22 of 35 with a
threshold policy (13 by more than 0.05, mean -0.065); four of the BBR ones
rerun at 64 MB reproduce release 49 exactly. A buffer below what autotuning
asks for changes when the sender writes, and the episode takes another
course even where no data in flight is stopped.

A limit should leave well-behaved controllers alone, so their windows were
measured: 340 of those episodes at 64 MB (2026-11-04, local), the largest
window of each episode's observed flow against N (the records keep its
window only).

| Controller | Flows | Largest window over N | Largest request in `per_mss` per packet of N |
|---|---|---|---|
| BBRv1 | 36 | 2.15 | 6.44 |
| BBRv3 | 37 | 1.88 | 5.63 |
| loss-based and delay-based (14 controllers) | 231 | 2.14 | 4.28 |
| Hybla | 36 | 6.24 | 12.47 |

Slow start overshoots to about twice the path before the loss is seen, and
BBR keeps two BDPs in flight on a shallow buffer. The limit therefore
grants what autotuning asks for a window of 4 N at BBR's factor, 12
`per_mss` per packet of N (about 18 times the path's bytes), twice the
largest window measured, and at least 10 segments and at most 64 MB. Checks
on 2026-11-04, local:

- The same 340 episodes with the limit, paired with themselves at 64 MB:
  all 160 stock and all 120 BBR ones are identical, and 57 of the 60 with
  Hybla; the other three moved by at most 0.0003 in utilization.
- The UML tests pass: Reno still fills a 640 ms buffer at 96 Mbit/s and
  160 ms, and the growing agent on 50 Mbit/s, 10 ms and 26 packets for 4 s
  (a 1.9 MB limit) puts 9.0 arrivals on the link per departure, against
  54.4 with 64 MB; its utilization is 0.71 against 0.93.

The setup scripts' other lines (the initial buffer sizes 87380 and 65536
bytes, no cached metrics, no autocorking) change every episode a little,
not only those the cap reached. Paired by episode, the out-of-distribution
data rerun (`exp-data-net1-broad-ood-40` against `-9`, the same 3,600 ids)
shows, on links that need less than 3 MB in flight, utilization higher by
0.007 to 0.012 and queueing delay by about 3% (intervals over episodes
excluding zero in three of the four sets), and on links that need more,
delay higher by 4% to 16%. That is small against the differences between
controllers, but it makes a mean of old families and rerun deep-long rows
a mix of two guests, so the six validation and test runs are also rerun
whole (runs #43 to #48), and their deep-long rows must equal the
`-deeplong` reruns'.

- The deep-long family of the six runs that report it and are still used:
  the fixed rule's validation and test (`exp-rule-net1v4-val-11`,
  `exp-rule-net1v4-test-14`), the share rule's (`exp-share-copies-val-17`,
  `exp-share-test-18`) and Sage's (`exp-sage-val-25`, `exp-sage-test-26`),
  each as `experiments/*-deeplong.json`: the same controllers, links and
  seeds, so the same episodes, and only the guest differs. The validation
  reruns show whether u = 0.125 and k = 1 would still be chosen.
- net1 v4's data (`exp-data-net1-broad-30s-5`, `-60s-8`, `-ood-9`), with the
  same specs, to count the episodes that changed and to measure net1 v4's
  honesty on what the new guest produces; net1 is retrained only if that
  honesty slips.
- ccBench and ccperf (`experiments/ccbench.json`, `ccperf.json`).

Not rerun: the learned net2 on net1 v4 (`exp-net2v4-online-15`,
`exp-net2v4-val-16`), the mode rule (`exp-mode-val-19`) and the share rule
on the truth (`exp-share-oracle-val-20`) are negative results that later
work replaced, decided on conditions the limit did not reach (copies and
competitors on the plain links); the fairshare runs, copies and held-out
competitors used only links that need at most 3.1 MB in flight in all; and
`exp-test-links-duration-2` and net1 v3 are superseded.

Reruns for the send-buffer limit (scheduled 2026-11-04): the broad data
(30 s, 60 s and out-of-distribution), the fixed rule's u over three
validation seeds, the share rule's and Sage's validation, the case study,
ccperf and ccBench, every one with the limit; the test runs follow once the
validation choices are settled on these, so each test link set runs once
more, not twice. Runs with the 16 times cap are superseded (`-49` and `-51`
published; #50 and #52 to #58 were cancelled), and so are #59 to #68,
cancelled, which had the first version of the limit. The `-deeplong` reruns and
runs #43 to #48 remain the record of 64 MB without a limit. The runs listed
above as not rerun stay so, for the reasons given there.

Done under the frozen settings, with no failed episode: the broad data
(`exp-data-net1-broad-30s-73`, `-60s-74`, `-ood-75`), Sage's validation
(`exp-sage-val-72`), the fixed rule's u and the share rule's k on net1 v5
(`exp-rule-net1v5-val-82`, `exp-share-net1v5-val-85`), the case study
(`exp-expliccit-case-91`), ccperf (`exp-ccperf-77`, `exp-ccperf-v5-87`) and
ccBench (`exp-ccbench-78`, `exp-ccbench-v5-89`, merged in
`exp-ccbench-merged-92`). The test runs wait for 2026-11-16.

### Freeze (2026-11-04)

The guest, the kernel and the TCP settings are frozen at `1365f0e`, the
send-buffer limit's commit: the guest and kernel as built from it, the
setup scripts' sysctls, and `EpisodeSpec._send_buffer`. Every run from #69
on uses them. A later change to any of them restarts the critical path
(net1's data, net1, then the rules' choices), so one is made only for a real
bug, and recorded here with its commit.

## Teacher modules, and seeded runs across guest versions (M10)

Building BIC, CDG, HighSpeed, H-TCP, Hybla, Illinois, LP, NV, Scalable,
Vegas, Veno, Westwood and YeAH into `kernel/uml.config` as modules changed
the kernel image. They are instead built from the unmodified in-tree
sources as out-of-tree modules, and the kernel image is bit-identical to
before (md5 3ea96e84). Vegas, Veno, YeAH, LP and Westwood each ran an
episode (utilization 0.94 to 0.99).

The four seeded episodes of the M6 check were rerun on the M10 guest and
compared with the M7 outputs. Both Cubic episodes and one BBRv1 episode are
byte-identical. The other BBRv1 episode diverges after 10.4 s, at a BBR
pacing-gain change (its random gain-cycle phase). The same guest on the
modified and the original kernel gives identical files, so the cause is the
guest's own changes since M7, which shift the kernel's random stream.
Seeded runs are reproducible for one build (kernel and guest), not across
guest versions. Compare versions on Cubic, or over many seeds.

## Online training and interval net1 (M11)

Tests:

- `tests/test_agent.py::test_exploration_noise_is_seeded_and_has_the_requested_spread`:
  with a zero policy and sigma 0.1, the applied actions have a mean below
  0.02 and a standard deviation in [0.08, 0.12]. The same episode seed gives
  the same noise, and another seed gives different noise.
- `tests/test_train.py`: the interval net1 covers held-out rows at 0.83 to
  0.97 for a nominal 0.90, and its bounds never cross; the utility reward
  equals hand-computed values; one online iteration (two rollouts, an
  update, an evaluation) runs in UML and writes a policy.
- `tests/test_uml.py`: on a one-flow episode, the flow's drop fraction
  equals the qdisc's own counters (`drops_queue` over `arrivals`).

Runs at scale, one seed each: an interval net1 and a cloned net2 on the
expliCCit-like data; four online runs of 12 or 20 iterations (16 or 24
episodes each, 4 to 6 minutes on 4 cores, no failed episode); and one
evaluation campaign of the four teachers and ten policies on 18 paths for
45 s (252 episodes, none failed). The results and their reading are in
`docs/training.md`. At M11, with untuned settings, online training degraded
the cloned policy, and among the offline net2 only cloning kept its drops
near the teachers'. M12's tuning superseded the online result (below).

## Tuning net2 (M12)

`tests/test_train.py`: IQL prefers the better of two logged actions where
cloning averages them; TD3+BC reports most target actions outside the data
when the logged actions are tiny; the teacher filter keeps only the named
controllers' episodes.

Runs at scale, with a harness kept outside the repository (results in
`docs/training.md`, section 7, "Tuned comparison"): 214 offline training
runs and 48 online runs (11,264 rollout episodes); 7,512 validation
episodes (614 policy checkpoints and the teachers); and one test campaign
of 468 episodes. The first round's validation run, 3,216 closed-loop
episodes of 30 s, took about 35 minutes on 4 cores. No episode failed in any of them, except in a first run of the
test campaign that filled the disk (304 of 468 failed). That run was
discarded, and the campaign was run again after freeing space. A scorer
that averages only the surviving episodes gives each controller a mean
over different paths, so the harness now refuses to score a campaign with
any failed episode.

On the held-out test paths, learned policies reach +0.55 (cloning) to +0.64
(TD3+BC) utility, against +0.82 for BBRv1 and +0.29 for expliCCit's released
policy. Online fine-tuning changes the cloned policy by +0.03 (95% interval
-0.13 to +0.16).

## The path's true state as a policy input (M13)

`tests/test_truth.py`, in UML:

- On a 14 s BBRv1 episode with a capacity step (20 to 4 Mbit/s), a
  60-packet buffer, random loss in both directions, and UDP and TCP cross
  traffic, the truth the guest computed at each of 1,378 steps equals the
  host's labels for all seven LCFs, bit for bit and NaN for NaN. Steps fall
  on whole milliseconds, the instants of delivery opportunities. On a
  rerun, 753 of the 1,378 steps had a departure at exactly their
  nanosecond, and the truth still matched, so the qdisc's events at an
  instant are drained before the step reads them.
- A policy that reads only the true queue acts, at every step, exactly as
  the float32 function of the truth that the step recorded.
- A net2 on the true LCFs, exported as the trainer exports it, acts in the
  guest as torch computes it from the recorded truth (within 1e-6).

`tests/test_train.py`: a patched graph replaces exactly the named LCFs (the
centre and both bounds for an interval net1), an oracle graph ignores the
message, and NaN in the truth is zeroed as in net2's training targets.
A net2 with memory exports with the `state` input and the next state as
its second output, and the ONNX graph, run three steps with its state fed
back, matches torch at each. `tests/test_records_abi.py` pins the step
record's new layout.

Patched runs of expliCCit's released policy on its case-study link are
reported in `docs/methodology.md`, section 6.

## CI

Each check runs only when what it checks can have changed; everything
else runs on demand.

- `.github/workflows/ci.yml`, on every commit that changes more than
  docs: Python (ruff, pyright strict on `src`, and pytest with the Mahak
  reference and the `train` extra; the UML tests skip there), and
  workflow and shell lint (actionlint, zizmor, shellcheck).
- `.github/workflows/uml.yml`, on a push that changes the guest, the
  kernel tree or our modules, `src`, the tests or the build, and on
  demand: Rust (fmt, clippy, the x86_64 musl build of the guest) and UML
  (kernel style with `scripts/check-kernel.sh`, the UML build from the
  cache or from source, the UML tests, a smoke campaign through the CLI).
- On demand only (`uml.yml` with `mainline` set): the tripwire that
  applies our kernel patches to the newest mainline kernel and runs the
  UML tests on it. It takes mainline's tarball from GitHub's mirror of
  Linus' tree, since git.kernel.org refuses downloads from Actions
  runners.

Its first run on demand (2026-11-02, mainline 7.4-rc1) downloaded the
source and applied our patches up to `include/linux/tcp.h`, where one
hunk no longer applies. Our pinned 7.0 tree is unaffected; carrying the
patches to a newer kernel would start there.

This page does not track CI results; the Actions tab has the current state.
Two earlier records, both from before M6: on the first CI run on GitHub,
the Python, Rust and lint jobs passed, and the UML job exposed a
toolchain-selection bug in `build-guest.sh`, since fixed. Before the agent
controllers were added, the tripwire was run once locally on kernel.org
mainline 7.3.0 plus our patches: the patches applied, the modules built,
and 6 UML tests passed.

## net1 as a distribution per LCF (M14)

`tests/test_train.py`: quantiles read off a binned distribution match a
uniform and a one-bin distribution; bins are equal for a fraction and
log-spaced for a rate, with a bin for zero; CRPS matches its known
values (1/12 for a uniform scored at its centre, the absolute error for a
point forecast); on synthetic data with a known noise, a distribution
net1 covers the truth at the nominal rates (within 0.05 at 50% and 90%)
and gains between 0.6 and 1.1 bits, against at most 1.04 that the noise
allows; a distribution net1 saves and loads exactly, feeds net2, and a
truth patch puts all probability on the true value's bin while a NaN
truth keeps net1's belief; an online spec whose net2 reads the truth is
rejected. `tests/test_truth.py`: an exported distribution policy with a
patched LCF acts in the guest as torch computes it (within 1e-6).

Runs at scale, with a harness kept outside the repository (results in
`docs/training.md`, section 7, "net1 on held-out paths"): 42 tuning runs
on the teacher data and 8 seed runs. Three runs in parallel exceeded the
container's 15 GB (about 6.3 GB each, 7.3 GB with 128 bins), so the runs
went two at a time, and the 128-bin ones alone; the killed runs were
rerun, and none of them produced a result. A container restart stopped
the search once; it resumed from the finished runs.

Two scoring mistakes were found during the search and fixed before any
result was taken: coverage was first tested against exact values inside
a bin, which fails for quantities that take a few exact values (capacity
levels, one-way delays), and is now tested on bins; and the log score was
first used to compare different numbers of bins, which favours coarse
bins, and CRPS now makes that choice.

## net1 on unseen links, and net2's window target (M15)

`tests/test_agent.py`: with `noise_hold_steps` 5, the applied actions
equal the per-step run's draw at the start of each block of 5 (the
generator advances every step either way); a policy computing
a = log2(100) - log2(cwnd) from the `cwnd` input holds the window at
exactly 100 packets from the third action on, and the step table's `cwnd`
column is the window the policy read. `tests/test_train.py`: a link split
keeps a link's episodes together across cross-traffic and loss
conditions; a paired stride keeps each sampled step's successor; the log
utility floors the share at 0.01; with the window target, net2's input
pools the bins into distributions, the believed BDP equals the one from
net1's means, the logged action is log2 of the next window over that BDP
(none where the next step is missing), and the exported graph inverts
it (the window it sets, over the believed BDP, is 2^u); an oracle net2 on
a distribution net1 sees all mass on the true bin, and net1's own belief
where the truth is unknown; the means view hands net2 each
distribution's mean, standardized as net1's targets are, and its graph
inverts the window target as well; n-step transitions sum the rewards
of consecutive steps only (never across a gap in the steps or an
episode's end) and bootstrap n steps ahead; online training runs one
iteration with each action kind, drawing its rollouts under two
conditions.
`tests/test_spec.py`: `ccs_per_point` runs the same seeded subset of
controllers under every condition of a point, and existing campaigns
expand to the same episodes as before (checked by hashing the expansions
of the icnp25 sample, the smoke campaign and the 180-link campaign before
and after the change).

Found and fixed while building it: the ONNX export left the caller's
graph in training mode (a fresh wrapper module starts in training mode,
and the exporter restores the wrapper's mode), so a torch reference
computed after an export disagreed with the guest; the guest was right.
The oracle net2 on a distribution net1 trained on K scaled values while
its graph fed K x B bins.

Runs at scale (harness outside the repository; `docs/training.md`,
section 7, "net1 on unseen links"): two data campaigns, 11,340 and 9,450
episodes, all ok; 12 tuning runs, the 3-member ensemble, and its single
report on the test links. Memory: two trainings at once were killed by
the container (12.9 GB for one) until the loader kept float32 arrays per
row group and net1 scaled its inputs in batches; then two runs peaked at
10.7 GB together. Killed runs were rerun and produced no result.

net2 runs (`docs/training.md`, section 7, "net2 on the fixed net1"): 5
offline runs, 6 constant multiples and 2 constant oracles, 5 online runs
(one-step TD3+BC twice, IQL, and n-step TD3+BC with 20 and 40 steps,
each scored on the validation links every 5 iterations), then one test
report of 5 policies on 27 links (405 episodes, all ok). Three online
runs (both one-step TD3+BC runs and IQL) were stopped after 5 to 10
iterations, once their evaluations stayed below their start; their
evaluations are reported as such. Two more seeds of the 40-step run
followed, each scored once on the test links; then the constant, the
oracle and the three seeds were scored once on a second held-out set
(158 links, 2,370 episodes, all ok). Disk filled to 99% during that run
from the online runs' rollout tables, which were deleted (their
transitions had been used); no episode failed. The rule on a classical
BDP estimate was tuned the same way (6 multiples on the validation
links) and scored once on both held-out sets. Later, the constant's test
run was repeated with the same policy and seed and reproduced its score
exactly (-0.3927), to split it by condition, as did the learned net2's
seed 0 (-0.348) and the constant's validation run (-0.369); and two
online runs of 60
iterations (arms A3 and A2, one seed each, 1,920 rollout episodes each)
were scored on the validation links every 10 iterations, next to the
constant run again there. Kernel
controllers are scored from the data campaign's episodes on the same
links, under the same conditions but other episode seeds.

## Several learned flows on one bottleneck (M16, 2026-11-02)

Every TCP cross flow can now be an agent with its own policy. Each action
names its socket by the writer's file descriptor and is kept per socket
(`docs/architecture.md`, D25).

`tests/test_multi_agent.py`, in UML:

- `test_each_agent_drives_only_its_own_socket`: an observed agent holding
  10 packets, a cross agent growing by 2^0.05 per step and a cross
  `agent_reno` shrinking by 2^-0.02, next to Cubic and UDP. Each socket's
  accepted windows equal Sage's accumulator for its own policy, its action
  numbers run 1, 2, 3 and so on, and a held socket reads at each step the
  window its own policy set at the previous one (more than 95% of steps,
  the rest being the core stack's recovery). The observed socket stays at
  10 packets while the cross agent grows past 100.
- `test_a_multi_agent_episode_is_reproducible`: the same episode twice gives
  the same actions, windows, step times and departures, bit for bit.
- `test_many_agents_share_the_bottleneck`: ten copies of one policy holding
  30 packets, on a 20 Mbit/s link with a 20 ms RTT. Their receive rates over
  the last second have a Jain index above 0.95 and sum to 80% to 101% of
  capacity, and each flow's live truth equals its own labels, bit for bit,
  at every step.
- `test_spec.py::test_cross_flows_take_their_own_agents`: each agent cross
  flow gets its own agent section and noise stream, a shared policy file is
  written once, and a stock cross flow keeps its plain form.

Single-agent episodes are unchanged. Three episodes (`agent` with a
constant policy; the fixed rule on net1 v3 with a Cubic flow and 1% loss;
`agent_reno` with a BBRv1 flow) were run before and after the change on
the same build. Every table (observations, labels, link events,
opportunities, steps, actions) was equal, NaN for NaN; actions only gained
their port. The 11,340 episode ids of `examples/campaign-net1-varied.json`
are the same before and after.

The truth with several step loops. Without the 1 ns settle step, 16 agents
on one link gave 1 step of 6,330 whose queue truth was one packet above
the labels: kernel work at that instant had not been drained yet. With it,
10 agents (4,832 steps) and 16 agents (6,266 steps) matched exactly. With
64 agents (the cap), 1 step of 7,909 still differed (by 2 packets, with
the busy fraction and mean delay of that window), because at that load the
qdisc emits some records after their instant. The labels are exact either way; only the
live truth input can lag. A 6 s episode with 64 agents took 41 s of wall
time in a 1 GiB guest.

## Not run (and why)

| Item | Status |
|---|---|
| The link model against real Mahimahi | Prepared, UNRUN. `tools/crosscheck` holds four links (expliCCit's case-study link, a 1 BDP and a 4 BDP cell, the second with 1% data loss, and a ccBench solo cell) for the team to run on the original stack (Ubuntu 16.04, Sage's Linux 4.19, ccBench's Mahimahi) with Cubic, BBRv1, Vegas, Sage and expliCCit's released policy, and `compare.py` computes the same metrics from both sides. palaestra's side is `exp-crosscheck-93` (60 episodes, none failed). The kit's commands for the original stack are read from its sources, not run. A run on current Mahimahi would first need Mahak's `highbdp.patch` ported (`configure.ac` and `linkshell.cc` changed). |
| UML against a real-time kernel | Will not run: the native tier was dropped (`docs/architecture.md`, D19), so virtual time is never compared with real time. |
| Mahak's failure regions (BBR, Orca, Sage) | Partial: the one-seed probes above (before M6, and M6). Maps over seeds are UNRUN. Orca's trained model is not in this repository. Sage's released policy ran on the validation and test links (`docs/training.md`, section 7), not over Mahak's regions. |
| DeepCC pacing (Sage's patched kernel, `PACE_WITH_SRTT`, coefficient 100) | UNVERIFIED: the kernel patch's source is not public. The guest paces at cwnd * mss / srtt. |
| Inference latency in the closed loop | UNRUN (parked): inference is free in virtual time. The plan is to measure it outside the environment and add a fixed decision delay. |
| Automatic reruns of the M6 check that stock congestion control is unchanged | Not in CI: it needs two kernel builds. Rerun it by hand when the kernel patch set changes. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for 7.0. W=1 and checkpatch --strict are clean. |
| Large campaigns on the CSLab Slurm cluster | UNRUN: script generation is tested, but no script was ever submitted. |
| Training on GPU (the CUDA wheel in `docs/training.md`) | UNRUN: there is no GPU in this environment. |
| The oracle arm A1: a net2 trained on the true LCFs, compared in closed loop with the same net2 on net1's beliefs | UNRUN for a trained net2. The environment supports it since M13 (above). A fixed rule has run on the true BDP against net1's (M15, above); the case study in `docs/methodology.md`, section 6, only patches beliefs of expliCCit's released policy, on one path. |
| Training settings not searched at M12: reward scale 1 for TD3+BC, discount 0.995 or 0.999, noise held for an RTT, one decision per RTT | UNRUN (`docs/training.md`, section 7). |
| The experiments of `docs/methodology.md`, section 5 | Partly run, with fewer seeds than designed there (arms A2 and A3, and A1 for a fixed rule). Its section 6 says what has run and the status of each piece the environment needs. |
