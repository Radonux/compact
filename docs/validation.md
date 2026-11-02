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

## Summary (M15, 2026-11-02)

The suite has 105 tests: 77 unit tests and 28 that need a UML build.
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
- the closed loop applies Sage's window rule exactly, and a policy acts on
  exactly the message and the truth that its step recorded;
- the truth a policy can read equals the host's labels at every step.
- a distribution net1 states calibrated probabilities on data with known
  noise, and its exported policy, patched or not, runs in the guest as
  torch computes it;
- held exploration noise reproduces the per-step stream block by block,
  and a policy that reads the sender's window sets an absolute window.

A one-off check (M6, repeated at M7) showed that adding BBRv3 left seeded
Cubic and BBRv1 episodes byte-identical.

The main gaps, detailed under "Not run": the port of Sage's step has not
been compared with `sage.cc` (UNRUN); the comparison with Sage is UNRUN,
and Sage's released checkpoint has not been run in UML; reference runs of
real Mahimahi against the link model have not started; and UML's virtual
time is never checked against a real-time kernel.

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

### The port of Sage's step against `sage.cc` (UNRUN)

`test_step_matches_sage_cc` (`tests/sage_ref.py`) is written to compile
Sage's own step code, taken by line range from `sage.cc` at `eedd281` with
only the clock and the `TCP_INFO` read stubbed. It replays the recorded
readings of a Cubic episode and a BBRv1-with-loss episode, and compares
every column printed with `%.7f`. Its first run, at M9, built the harness,
but the harness died with SIGSEGV before printing anything. The cause has
not been found: the message fits Sage's 1000-byte buffers (873 bytes at
most on that episode), and the struct layout matches (224 bytes). Debugging
means running Sage's compiled code again, which the permission policy of
the session that ran it then blocked. The test skips unless
`EXPLICCIT_SAGE_DIR` is set. Until it passes, the port is checked only by
review and by the tests above. UNRUN.

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
`tests/test_records_abi.py` pins the step record's new layout.

Patched runs of expliCCit's released policy on its case-study link are
reported in `docs/methodology.md`, section 6.

## CI

`.github/workflows/ci.yml` runs five jobs:

- Python: ruff, pyright (strict on `src`), and pytest with the Mahak
  reference and the `train` extra; the UML tests skip there;
- Rust: fmt, clippy and the x86_64 musl build of the guest;
- UML: kernel style (`scripts/check-kernel.sh`), the UML build, the UML
  tests, and a smoke campaign through the CLI;
- a non-blocking tripwire that applies our kernel patches to the newest
  mainline kernel and runs the UML tests on it;
- workflow and shell lint: actionlint, zizmor and shellcheck.

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

## Not run (and why)

| Item | Status |
|---|---|
| The port of Sage's step against `sage.cc`'s own code | UNRUN: the harness died with SIGSEGV on its first run (M9, above). Until it passes, the port is checked only by review and by the step tests. |
| expliCCit's comparison with Sage (expliCCit reports that the split costs about 5% utilization and 9% queueing delay against Sage) | UNRUN. Sage's released checkpoint is available locally, but it has not been exported to ONNX or run in UML. The agent controllers can run it once it is exported under `sage-v1`, as a recurrent graph with its LSTM state in `state`. |
| The link model against real Mahimahi | Not started (UNRUN): it needs a root host (any kernel). Mahimahi is vendored. Mahak's `highbdp.patch` does not apply to current Mahimahi (`configure.ac` and `linkshell.cc` changed) and must be ported first. |
| UML against a real-time kernel | Will not run: the native tier was dropped (`docs/architecture.md`, D19), so virtual time is never compared with real time. |
| Mahak's failure regions (BBR, Orca, Sage) | Partial: the one-seed probes above (before M6, and M6). Maps over seeds are UNRUN. Orca's trained model is not in this repository. Sage's: see the comparison with Sage above. |
| DeepCC pacing (Sage's patched kernel, `PACE_WITH_SRTT`, coefficient 100) | UNVERIFIED: the kernel patch's source is not public. The guest paces at cwnd * mss / srtt. |
| Inference latency in the closed loop | UNRUN (parked): inference is free in virtual time. The plan is to measure it outside the environment and add a fixed decision delay. |
| Automatic reruns of the M6 check that stock congestion control is unchanged | Not in CI: it needs two kernel builds. Rerun it by hand when the kernel patch set changes. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for 7.0. W=1 and checkpatch --strict are clean. |
| Large campaigns on the CSLab Slurm cluster | UNRUN: script generation is tested, but no script was ever submitted. |
| Training on GPU (the CUDA wheel in `docs/training.md`) | UNRUN: there is no GPU in this environment. |
| The oracle arm A1: a net2 trained on the true LCFs, compared in closed loop with the same net2 on net1's beliefs | UNRUN for a trained net2. The environment supports it since M13 (above). A fixed rule has run on the true BDP against net1's (M15, above); the case study in `docs/methodology.md`, section 6, only patches beliefs of expliCCit's released policy, on one path. |
| Training settings not searched at M12: reward scale 1 for TD3+BC, discount 0.995 or 0.999, noise held for an RTT, one decision per RTT | UNRUN (`docs/training.md`, section 7). |
| The experiments of `docs/methodology.md`, section 5 | Partly run, with fewer seeds than designed there (arms A2 and A3, and A1 for a fixed rule). Its section 6 says what has run and the status of each piece the environment needs. |
