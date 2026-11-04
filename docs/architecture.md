# Architecture

This page explains how one episode runs and why it is built this way: the
emulated path and its bottleneck, virtual time, the record stream and the
ground truth computed from it, the control loop a learned policy runs in,
the known limits, the design decisions (D1 to D23), and how to extend the
system. `README.md` states the guarantees and how to check them,
`docs/validation.md` records the checks that ran, and `docs/methodology.md`
is the research plan.

## Data flow and trust boundaries

    EpisodeSpec / CampaignSpec (JSON)          src/expliccit_env/spec.py   <- trust boundary
            |  guest_plan(): traces as per-ms counts, policy file
            v
    plan dir: plan.json, *.trace, policy.onnx  src/expliccit_env/guest_plan.py
            |  uml.py: boots UML with time-travel=inf-cpu and the spec's seed
            v
    expliccit-guest (PID 1 in UML)                                 guest/src/
            |  Plan::load                                          <- trust boundary
            |  namespaces, veths, 2 x mmlink qdisc (rtnetlink), modules
            |  bulk TCP pair, cross flows, optional Sage step loop
            |  (+ policy, + live truth) -> steps.bin
            |  /dev/expl_rec -> one file per record type, framing + seq checked
            v
    raw dir: *.bin, steps.bin, result.json, console.log
            |  episode.py: GuestResult (pydantic) + load_records    <- trust boundary
            |  + load_steps; validity checks, labels.py, steps.py, Mahak metrics
            v
    shard dir: episodes/obs/labels/link_events/link_opp/agent_action
               (+ steps) .parquet, DONE

Everything that crosses a process or kernel boundary is parsed once, where
it enters, into types that cannot hold invalid values:

- specs: pydantic models with closed sums;
- the guest plan: Rust `Plan::load`;
- netlink options: the kernel's `nla_parse_nested` with a policy;
- agent actions: the kernel's `agent_write`;
- the policy graph's inputs: `Policy::load` accepts one `obs` and at most
  one each of `state`, `truth` and `cwnd`, matched by name;
- records: numpy dtypes that mirror the C structs, checked for size, type,
  ABI and sequence continuity;
- `steps.bin`: a numpy dtype, checked for whole records and increasing
  times;
- `result.json`: pydantic.

## Topology inside one episode

    snd 10.0.0.1 s0 ---- p0 10.0.0.2 [path] 10.0.1.2 p1 ---- r0 10.0.1.1 rcv
                          ^ uplink mmlink           ^ downlink mmlink
                          (link, then delay)         (delay, then link)

The guest creates three fresh network namespaces (sender, path, receiver)
joined by veth pairs. The bottleneck sits in the middle namespace. Every skb
is orphaned when it crosses into that namespace, so the sender's TCP Small
Queues see a remote bottleneck, just as a Mahimahi shell looks to the server
outside it. Data packets cross `mm-delay`, then `mm-link`; ACKs cross the
uplink, then the delay. This matches Mahimahi's nesting. IPv6 is off, and
ARP bypasses the model (Mahimahi works on IP packets), so only the
experiment's traffic is shaped.

The sender listens with the congestion control and the observation mark
already set, as in Mahak's `server-mahimahi`. The receiver connects at trace
time 0, and the sender writes until the episode ends. Both set
`TCP_NODELAY`.

Cross traffic shares the same path in the same direction, sender to
receiver. Cross flow `i` sends from port `6000 + i`. The observed flow sends
from port 5001, and that port is how link records and labels tell the flows
apart. A TCP cross flow is a bulk transfer with its own congestion control,
connected at its start time and shut down at its stop time. That control is
a stock controller by name, or an agent with its own policy, exactly as the
observed flow's `cc` is written; any number of cross flows can be agents,
each with its own step loop, policy state and exploration noise (D25). A UDP
cross flow sends one 1500-byte IP packet every `12000 / rate` seconds over
its window (rate in bit/s). Cross sockets carry no observation mark, so
`TCP_OBS` holds the observed flow only. Threads drive each flow and sleep on
absolute deadlines, so no flow spins in virtual time. Each TCP cross flow
reports the controller its socket ran, and an episode whose flows ran other
controllers than planned is invalid.

## The link model (`sch_mmlink`)

`sch_mmlink` is a port of `third_party/mahimahi/src/frontend/link_queue.cc`:

- Time is integer milliseconds since the qdisc's `base_ns`, shared by both
  directions. Trace entry `i` is the number of 1504-byte delivery
  opportunities at ms `i`. The trace repeats with a period equal to its last
  timestamp, so the last ms of a cycle and ms 0 of the next coincide, as in
  Mahimahi.
- Before a packet that arrives in ms `a` joins the queue, every opportunity
  at ms `<= a` is used. A packet therefore never departs in its arrival ms.
- The head of the queue moves "in transit" and may span several
  opportunities. It departs at the opportunity that carries its last byte.
  Droptail limits (packets, bytes or both) count queued packets only, never
  the one in transit.
- A packet's size is its IPv4 total length plus the 4-byte TUN header that
  Mahimahi counts, so a full 1500-byte packet uses one 1504-byte
  opportunity.
- The delay line releases each packet exactly `delay_ns` after it entered.
  A packet that leaves the uplink at opportunity ms `t` enters its delay
  line at the instant of `t`.
- GSO super-packets are segmented on entry, because Mahimahi only ever sees
  MTU-sized packets.
- Random loss is optional (Mahimahi's `mm-loss`). Each packet that enters
  the qdisc is dropped independently with the direction's probability,
  before the delay line and the link, so a lost packet never uses link
  capacity and never appears as an arrival. The qdisc draws from the kernel
  RNG only when the probability is nonzero, so loss-free episodes are
  unchanged.

The qdisc processes events lazily (on enqueue, on dequeue and from its
watchdog), but always in time order. It uses the semantic instants
(opportunity times, release times), never the moment its code happened to
run. So neither the 256 ns UML clock tick nor timer slack can change the
model's decisions. Cost: O(1) per packet event plus O(1) per elapsed trace
ms.

The trace travels as one netlink attribute of per-ms u8 counts. Netlink
attributes carry a 16-bit length, and a list of up to 900000 timestamps
(Mahak's 100 Mbit/s x 8 grid corner) does not fit in one. Per-ms counts fit
any trace up to 65 s long and 3 Gbit/s (`ponytail` limits, enforced on both
sides).

## Time and determinism

`time-travel=inf-cpu` makes CPU work free: virtual time advances only when
the guest idles or sleeps. Three consequences shape the design (measured in
`docs/m0-report.md`):

- Every episode needs a time-based bottleneck. Without one, a transfer is
  infinitely fast and never reaches its end time. The host enforces a
  wall-clock timeout on every guest.
- Guest code must block or sleep, never spin: UML advances a jiffy after
  10000 unscheduled userspace switches. The record reader sleeps 20 ms
  between drains, and the step loop and the cross-traffic threads sleep
  until absolute deadlines.
- UML's clock ticks every 256 ns, and every clock read from process context
  advances virtual time by one tick. Sender observation timestamps carry
  this sub-microsecond quantization. The link model does not (see above).

With `CONFIG_UML_SEEDED_RANDOM` (our patch, see `third_party/README.md`),
`um_seed=` and a fixed `time-travel-start=`, an episode is a pure function
of the kernel build, the guest build, the plan and the seed: the same
inputs give byte-identical record files. `uml.py` passes the spec's seed,
and campaign seeds are drawn deterministically from the campaign seed. A
change to the guest can shift the kernel's random stream, so seeded runs
are comparable across guest versions only on Cubic or over many seeds
(`docs/validation.md`, M10).

## Records

`kernel/modules/expliccit_rec.h` is the single source of truth (record ABI
3). Every record has a 16-byte header: type, length, sequence number and
time. Producers (`sch_mmlink`, `expl_tcpobs`, `tcp_agent`) append to one
ring in `expl_rec`, so the episode has one global sequence. A full ring
drops whole records. The guest detects the gap in the sequence, and the host
checks for it again.

| Type | Producer | Meaning (t_ns is the event's own instant) |
|---|---|---|
| META | expl_rec | ABI, HZ, ring size; first record after open |
| LINK_ARRIVAL / DROP / DEPART | sch_mmlink | Mahimahi's `+`, `d`, `-` events, with queue state, flow ports and TCP seq/ack; a drop names its reason (full queue, random loss, or an error) |
| LINK_OPP | sch_mmlink | Mahimahi's `#` events, one record per ms with count and bytes used; a ms with no queued packet may have no record, and the trace is the authority on capacity |
| LINK_STATS | sch_mmlink | counters at qdisc deletion; nonzero error counters invalidate the episode |
| TCP_OBS | expl_tcpobs | sender state just before each incoming segment is processed |
| AGENT_ACTION | tcp_agent | an accepted policy action |

The guest splits the stream into one file per type, so the host maps each
file as a packed numpy array with no per-record Python work.

`TCP_OBS` uses the `tcp:tcp_probe` tracepoint, because UML has no kprobes
and stock congestion controls must stay unmodified. The tracepoint fires at
the top of `tcp_rcv_established`. Record k is therefore the state left by
segment k-1 plus any timer-driven changes since then. Labels and training
code must read it as "the state observed when segment k arrived".

The step loop writes `steps.bin` itself; it is not part of the ring (see
"Step data" below).

## Ground-truth labels

`src/expliccit_env/labels.py` computes the labels as pure functions of an
episode's records and its path constants. They describe the bottleneck in
the data direction (the downlink). The host computes them at every sender
observation (`labels.parquet`, row for row with `obs.parquet`) and at every
step (the `label_*` columns of `steps.parquet`). Windowed labels cover the
trailing 100 whole trace ms before the sample.

The seven LCFs that expliCCit uses are also the columns of a policy's
`truth` input (`policy.TRUTH_COLUMNS`). Each depends only on what happened
at or before the sample, which is what lets the guest compute them live:

| Label | Definition |
|---|---|
| `queue_delay_mean_ns` | mean queueing delay of the packets (all flows) that left the bottleneck in the window; NaN if none did |
| `queue_pkts` | the queue after the last link event at or before the sample, not counting the packet in transit (Mahimahi's droptail definition) |
| `capacity_bps` | the trace's delivery opportunities in the window, in bit/s (1504-byte opportunities) |
| `free_buffer_pkts` | the packet limit minus `queue_pkts`; NaN without a packet limit |
| `busy_fraction` | bytes the bottleneck used over the bytes the trace offered in the window; NaN where capacity is 0 |
| `one_way_delay_ns` | one delay line: what Mahak and expliCCit call mRTT |
| `receive_rate_bps` | the observed flow's bits leaving the bottleneck per second in the window (link bytes, as capacity counts them); the receiver sees them one propagation delay later |

The other labels:

- `queue_bytes`: the queue in bytes, defined as `queue_pkts` is.
- `queue_delay_ns`: the exact queueing delay of the latest data packet that
  reached the bottleneck at or before the sample and was not dropped. It
  uses that packet's later departure, and it is NaN if the packet had not
  left by the end of the episode.
- `base_rtt_ns`: both delay lines.
- `buffer_pkts`, `buffer_bytes`: the droptail limits (0 means no limit).
- `buffer_fill`: the queue over its limit, the larger of the packet and the
  byte fraction; 1 means the next arrival is dropped.
- `drop_fraction`: the observed flow's packets that the full queue dropped
  in the window, over those that reached the bottleneck; NaN if none did.
- `random_loss_rate`: the data direction's configured loss probability.
- `cross_traffic_share`: bytes that cross traffic took from the bottleneck
  in the window, over the trace's capacity in the same window; NaN where
  the capacity is 0.

## Sage's step and the policy contract (`sage-v1`)

expliCCit's networks read what Sage's sender computes, so the guest computes
it the same way: `guest/src/step.rs` ports the step of Sage's `sage.cc`
(`CntThread`, commit `eedd281`). The step loop runs when the spec sets
`steps` or runs a policy.

### The step

Every 10 ms of virtual time the step loop reads `TCP_INFO` from the
observed flow's socket. Once the ACK state has moved (bytes acked or packets
delivered), it builds Sage's 77-value message. Otherwise it polls again
every 1 ms, where Sage busy-polls. The message covers the current RTT,
delivery rate and window; their averages, minima and maxima over 10, 200
and 1000 steps; and Sage's derived rates. Sage's quirks are kept (an empty
window's minimum is 1e6, the incremental mean, integer averages), because a
policy trained on Sage's numbers must see Sage's numbers.

Some columns need a note:

- Columns 1 and 75 are not observations and are written as 0: column 1 is
  Sage's nominal capacity, taken from its command line, and column 75 is
  its reward, which uses that capacity.
- Column 76 is the log2 of the window multiplier since the previous step,
  rounded to three decimals as `sage.cc` prints it. Under a policy, the
  multiplier is the 2^a it applied. Under a stock controller, it is the
  observed change of the window (Sage's collector logs cwnd over the
  previous cwnd, starting from 10).
- Column 77 is the kernel's min RTT in ms, which expliCCit's `sage.cc` adds
  to the message.

### The policy contract

`src/expliccit_env/policy.py` defines the contract, and `step.rs` mirrors
it. An ONNX graph takes these inputs, matched by name:

- `obs` float32 [1, 78]: the full message (Sage's 77 values plus
  expliCCit's min RTT);
- `state` float32 [1, S], optional: a recurrent state, zeros at the first
  step, which the guest feeds back from the graph's second output;
- `truth` float32 [1, 7], optional: the path's true state at the step (see
  below);
- `cwnd` float32 [1, 1], optional: the sender's window in packets
  (TCP_INFO's `snd_cwnd`, the step table's `cwnd` column), so a graph can
  choose an absolute window and answer with a = log2(target) - log2(cwnd).

The graph returns a, the log2 of the window multiplier. It selects its
columns and scales them itself, so a new network design needs no guest
change. The trainer's exporter (`train/export.py`) gives a graph the
`state` input and the next state as its second output when net2 declares
`state_dim` and maps (input, state) to (u, next state); a net2 without
memory exports as before, byte for byte. A message column can be infinite or NaN where `sage.cc` divides by
zero (for example column 68 before any sending rate is known), and the graph
must keep such a column from reaching its output.

The window follows Sage: a precise accumulator starting at 10 packets,
multiplied by 2^a, rounded up and kept in [4, 20000]. Pacing is
cwnd * mss / srtt, refreshed every step. A non-finite a, or |a| > 8, fails
the episode instead of being clamped, so a broken policy cannot silently
produce data. For online training, the spec's `noise_sigma` adds Gaussian
noise to each a, seeded from the episode's seed, and `noise_hold_steps`
keeps each draw for that many steps (the generator still advances every
step, so the stream does not depend on the hold). Held noise matters for a
policy that sets an absolute window: independent draws each step would
average out within an RTT. The sum is clamped to [-8, 8], and the step
records the action actually applied.

### The truth input

`truth` holds the seven LCFs of `policy.TRUTH_COLUMNS` at the step, in the
labels' units. An oracle policy (net2 on the true LCFs, arm A1 of
`docs/methodology.md`) or a patched one (some of net1's beliefs replaced by
the truth) reads it.

`guest/src/truth.rs` computes it live from the downlink's link records,
with the labels' definitions. The recorder's drain of the record ring is
shared with the step loop, which drains the ring at its own instant, so
every link record emitted by then counts. Each value is causal: the queue
after the last event at or before the step, and sums over the trailing 100
trace ms. Integer sums are kept exact and divided in the host's order. The
truth therefore equals the labels the step is stored with, bit for bit and
NaN for NaN (`tests/test_truth.py`). A value is NaN where its label is
(mean queueing delay with no departure in the window, free buffer without a
packet limit, busy fraction at zero capacity), and the graph must handle
that.

The truth exists whenever the step loop runs, and every step records it in
`steps.bin`, whether or not the policy reads it. D23 records why it is
built this way.

### Step data

`steps.bin` holds one fixed-size record per step: the step's
CLOCK_MONOTONIC time in us, the 224-byte `TCP_INFO` prefix the message was
computed from, the 78-value message, the action applied after it (NaN under
a stock controller), the window that action set (0 if none), and the truth.
`steps.py` decodes it, and the guest's and host's layouts are checked
against each other (`tests/test_records_abi.py`). An agent cross flow `i`
writes the same records to `steps-cross<i>.bin`; its truth is computed for
its own port. The guest's `result.json` carries result version 5.

The training table (`steps.py`, `steps.parquet`) gives each step:

- the labels at its instant;
- the action taken from it: the next step's column 76, as Sage's
  `gen_offline_data.py` derives it;
- Sage's single-flow reward, computed with the exact capacity where Sage
  uses the nominal one.

The `cross_steps` table holds the same columns for every agent cross flow,
with `flow`, its index in the spec's `cross` list; its labels are that
flow's own (receive rate, drops and the share every other flow took), and
the path's.

Every episode also has a `flow_seconds` table: for each flow (0 the
observed one, `i` the `i`-th cross flow) and each whole second, the bytes
and packets that left the bottleneck and their queueing delays (departure
minus arrival ms, Mahimahi's resolution) as a sum, a 95th percentile and a
maximum. Means over any window of whole seconds are exact from the sums;
the percentiles hold for single seconds only. It is small, so campaigns
keep it whatever else they drop.

## Agent controllers (`tcp_agent`)

After each step, the step loop writes the policy's window and pacing rate
as `struct expl_agent_action` to `/dev/expl_agent`, naming the socket by
its own file descriptor (agent ABI 2). `tcp_agent` keeps each socket's
latest action in that socket's congestion-control state, so any number of
agent sockets run side by side, and records each accepted action as
`AGENT_ACTION` with the socket's port. The action takes effect at that
socket's next ACK; a write for a socket that does not run an agent
controller is refused.

- `agent` (Sage-style): `cong_control` holds cwnd and pacing at the latest
  action. The initial window is held until the first action.
- `agent_reno` (Orca-style): each new action sets cwnd once, from
  `in_ack_event`, so it applies even when the flow is not cwnd-limited.
  Reno and the core loss response run in between. Orca proper runs Cubic
  underneath, but Cubic's update is not exported, so this base is Reno.

## Mahak's driver (`mahak.py`)

`mahak.py` ports Mahak's `mahak_online.py` and the query and result
functions of its `helper.py`. These parts are Mahak's:

- the grid;
- the features: min-max scaling, then a full PCA;
- the model: scikit-learn's Gaussian process, with an RBF kernel of length
  scale 0.1, 10 optimizer restarts, normalized targets and
  `random_state=100`;
- the first point: `choice` on a legacy numpy RandomState, which is what
  Mahak does under `np.random.seed`;
- the query rule: the largest predictive standard deviation, first index on
  ties.

The oracle is one UML episode per query, 20 s long as in `helper.py`, with
the same seed for every query. It is therefore a pure function of the point,
and a repeated point reuses its result.

Mahak's oracle crashes on a failed emulation. Here a failed episode, an
undefined metric or a non-finite value is recorded with its reason, the
point leaves the candidate pool, and it never becomes a label. Every queried
episode is also written to a dataset shard, so a map's queries can be
reused as training data.

## Experiments in GitHub Actions

Large runs go to GitHub Actions. For this public repository that is 20
concurrent jobs, each on a runner with 4 cores, 16 GB of memory and 14 GB
of disk, for up to 6 hours, and at most 256 jobs per workflow run. An
experiment is one file under `experiments/` (its format is in
`tools/ci/experiment.py`), dispatched to `.github/workflows/experiment.yml`:

1. **plan** parses the file once, expands its cells (the shards of a
   campaign, or the product of matrix axes) and runs `prepare`, for
   example `campaign plan`.
2. **uml** restores the UML build from the Actions cache, or builds it
   (`.github/actions/uml`). The key is the git trees of the kernel, our
   modules and config, the guest and both build scripts, so a build is
   reused until one of them changes. The `uml` job of `uml.yml` uses the
   same action.
3. **cells** runs `run` once per cell, 20 at a time. A cell writes only
   paths no other cell writes, and its output is an artifact kept 7 days.
4. **collect** merges the cells' outputs, runs `collect`, and packs the
   result as `NAME.tar.zst` in parts under 2 GiB.
5. **publish** makes the result a release, `exp-NAME-RUN`, with the
   experiment's record and its `summary.md` in the notes. The release is
   tagged at the commit that produced it, or at main's head if main's
   workflow files have changed since (the job's token may tag only a
   commit whose workflows match main's); the notes name the producing
   commit either way. Every successful run is published: releases are where
   results live, and anything shown to others comes from one. Artifacts
   only carry data between jobs, for 7 days.

Episodes are pure functions of their specs, so any runner can run any
shard, and "Re-run failed jobs" repeats only the cells that failed.
collect starts only once every cell has succeeded, so a partial campaign
never produces metrics. A later experiment reads an earlier one's result
through `inputs`, normally `release:TAG` (`run:RUN_ID/NAME` reads an
artifact of a run, while it lasts).

    gh workflow run experiment.yml -f experiment=experiments/smoke.json

The workflow is general compute, not a check on commits. It also takes
the experiment's JSON inline, with a `name`, so a one-off job needs no
commit; the JSON is recorded in the release notes either way.

Releases double as the object store. Tag names say what a release holds:
`exp-NAME-RUN` for a run's result, `model-...` for weights, `data-...` for
datasets. `scripts/stash.sh TAG FILE...` stores local files, each under
git's 100 MB limit on GitHub: it pushes a branch `stash/TAG` holding main
plus the files, and `.github/workflows/stash.yml` publishes them as release
TAG, tagged at that main commit, and deletes the branch, so the files never
enter main's history. (A ruleset that restricts branch deletion makes that
last step fail with a warning: the branch then stays, and a full clone
fetches its files, until the ruleset excludes `stash/**`.) An experiment reads a release with
`"inputs": {"NAME": "release:TAG"}`, and anyone can download a file from
`https://github.com/Radonux/palaestra/releases/download/TAG/FILE`.

## Benchmark

`experiments/ccbench.py` plans ccBench's two suites (github.com/Soheil-ab/ccBench
at `49b5ada`; Abbasloo, "Internet Congestion Control Benchmarking", arXiv
2307.10054) and three it lacks, and `experiments/ccbench_report.py` scores
them from `flow_seconds` (`experiments/ccbench.json` runs both for every
controller):

| Suite | Runs per controller | What varies |
|---|---|---|
| `solo` (ccBench1) | 690 | 12 to 192 Mbit/s, one-way delay 5 to 80 ms, buffer 0.5 to 16 BDP; constant, or 2x or 4x up (under 100 Mbit/s) or down from 7 s to 14 s; 30 s |
| `friendly` (ccBench2) | 125 | a Cubic flow from 0 s, the controller from 5 s; buffer 1 to 16 BDP; 125 s |
| `loss` | 150 | `solo`'s constant links with buffers of 1 and 4 BDP, and 0.1%, 1% or 3% random loss |
| `copies` | 150 | 2, 4 or 8 flows of the controller starting together, the same links; 60 s |
| `competitors` | 100 | `friendly` with BBRv1 or BBRv3 in place of Cubic, the same links |

The report keeps ccBench's leagues: in each run and window, the controllers
within 10% of the best round(throughput)^2 / round(delay) win (`solo`,
`loss`), or those whose throughput is closest to half the capacity
(`friendly`, `competitors`); a controller's rank is its share of games won,
among the controllers of that run only. Next to the ranks it gives absolute
means (utilization, queueing delay, share against the competitor, Jain's
index of copies). Where it departs from ccBench:

- There is no Pantheon tunnel: throughput and delay are measured at the
  bottleneck, exactly, from the records. ccBench takes each packet's delay
  from Pantheon's tunnel log and adds the one-way delay again; the report
  uses queueing delay plus twice the one-way delay, assuming the tunnel's
  delay includes the emulated one-way delay (UNVERIFIED).
- ccBench's runs start 10 s into its traces, after its setup time; here the
  flow starts at 0 s and the step traces are shifted to match (the change
  from 7 s to 14 s of the flow).
- Only kernel controllers and policies under `sage-v1` run; ccBench's
  user-space schemes (Copa, Vivace, PCC, Indigo and others) do not.
- Step traces are 7 s levels that repeat after 35 s; runs last 30 s, so
  each sees one raised or lowered period, as in ccBench.

ccBench asks how a controller trades throughput against delay on many
links. ccperf (Vielhaus, gitlab.com/ccperf/ccperf at `18d1620`, MIT) asks
more kinds of question: about responsiveness, buffers, fairness among
copies and against other controllers. `experiments/ccperf.py` plans the
experiments of ccperf's benchmark that this emulator can express, with the
semantics of ccperf's ns-3 backend (ccperf-ns3 at `7c1709f`; it is ns-3
simulation code, which does not carry over to a UML kernel, so only what it
sets up is taken). `experiments/ccperf_report.py` scores them, and
`experiments/ccperf.json` runs both for every controller, 5 times each:

| Experiment | Runs per controller and repeat | What varies |
|---|---|---|
| `single_flow` | 5 | ccperf's five static paths, 16 to 115 Mbit/s and 15 to 800 ms base RTT |
| `synchronous_fairness` | 7 | 5 flows on each static path; 2 and 10 flows on the default path |
| `start_time_fairness` | 3 | 2, 5 or 10 flows, all but the first starting in [0, 5) s |
| `loss_resilience` | 7 | 0.1% to 5% random loss |
| `deep_buffers`, `shallow_buffers` | 5, 6 | buffers of 1 packet to 16 BDP; 5 flows with 10 packets |
| `rate_step`, `rate_pulse`, `rate_random_walk` | 4, 2, 2 | capacity steps up or down at 10 s, 5 s pulses, a bounded random walk every 0.1 or 1 s |
| `replay_rate_trace` | 3 | ccperf's three Poisson rate traces (0.57 to 3.04 Mbit/s as named, from Pantheon; in `experiments/traces/ccperf`) on its dynamic paths |
| `reno_fairness`, `cubic_fairness`, `vegas_fairness` | 3 each | 2, 4 or 10 flows, half of them the competitor |

Per run the report gives utilization, the queueing delay of the flows under
test, Jain's index of their throughputs, and their share against a
competitor over their fair share; `summary.md` averages them per experiment
and controller, and `ccperf.csv` keeps every variant. ccperf has no single
score, and neither does this.

That is 51 of ccperf's 121 experiments per controller. The rest need
emulator features this one lacks: per-flow RTT (RTT fairness),
time-varying delay and loss (RTT and loss steps, pulses and walks), AQM
(ccperf's AQM runs, and its scalability runs, which use CoDel), ECN, ACK
compression, a policer, Poisson UDP, and an observed flow that stops early
(TCP on-off). Scalability with drop-tail does not test scalability: the
buffer of one BDP at the higher rate lets a loss-based controller fill it
before the step, so Cubic used 0.999 of a step from 16 to 512 Mbit/s in a
local check. The planner's docstring names each divergence; the main ones
are greedy flows where ccperf sends a fixed size, and random loss before the
bottleneck where ccperf drops after it.

## Known limits

Limits marked `ponytail` in the code name their upgrade path there.

- One global record ring per kernel, which is fine for one episode per UML
  guest.
- A policy's decision costs no virtual time: a learned controller acts the
  instant its step is read. Real controllers decide a few milliseconds late
  (an inference or a userspace round trip). ponytail: no decision delay;
  add one per flow, applied before the action is written, when comparisons
  between learned controllers need it.
- With several step loops, each sleeps 1 ns of virtual time after reading
  its step, so that kernel work at the same instant (softirqs deferred to
  `ksoftirqd`) is drained before its truth; one loop never needed it, so
  single-agent episodes are unchanged. Under very heavy load the qdisc can
  also emit a record after its instant: with 64 agents, 1 of 7,909 steps
  saw a truth that differed from the labels (docs/validation.md).
  The labels are always exact; only the live truth input can lag.
- Per-ms u8 trace counts: at most 255 opportunities per ms (3 Gbit/s), and
  traces up to about 65 s.
- Each path has one bottleneck with a droptail queue, fixed delay lines and
  i.i.d. random loss: no AQM or ECN, no delay jitter, no bursty loss and no
  multiple bottlenecks.
- At most 64 cross flows per episode (ports 6000 to 6063), all sender to
  receiver; each costs two sockets and up to four guest threads. No reverse
  traffic shares the ACK path.
- The record reader polls every 20 ms of virtual time. The step loop also
  drains the ring at each step.
- The step loop decides once per step (Sage's 10 ms period, extended until
  the ACK state moves), not once per RTT. Exploration noise is Gaussian,
  drawn per step or held for a fixed number of steps. After an idle step it polls every 1 ms, where
  Sage busy-polls. It refreshes pacing once per step, where Sage's patched
  kernel may pace per ACK (that kernel's source is not public).
- The `truth` input carries the seven LCFs only. The path's full state
  beyond them (arm A0 of `docs/methodology.md`) is not available to a
  policy.
- Labels are computed when an episode is processed. Shards keep
  `link_events` and `link_opp`, which hold what a label needs, but nothing
  reads them back yet (see "Extending").
- `parse_mm_log` (Mahimahi text logs) loops in Python. It is meant for
  fixtures and reference logs, not campaign data.
- Mahak's metrics (`utilization`, `mean_delay_ms`, powers) are computed over
  the link log as `mm-th-delay` would compute them, so with cross traffic
  they describe the whole link. `goodput_mbps` and `drop_fraction` (the
  share of the flow's packets reaching the bottleneck that the full queue
  dropped) are the observed flow's own.
- The trainer exports with torch's TorchScript-based ONNX exporter, which
  emits simple ops that `tract` runs. Move to the dynamo exporter (it needs
  `onnxscript`) when the old one is removed. The graph tests finiteness as
  |x| < 3e38, because `tract` does not analyse IsInf and IsNaN.
- Inference is free (parked). Under `inf-cpu` a policy's inference costs
  no virtual time, while a real sender spends time on it (Sage crosses
  shared memory into Python every step). The plan is to measure inference
  time outside the environment and add it back as a fixed decision delay
  (UNRUN).
- UML's virtual time is never checked against a real-time kernel (D19).

## Decisions

Each row records a decision as it was made, and a later row can change an
earlier one. D1 to D5 came from the project brief: real kernel TCP; Tier 1
on UML; Tier 2 on native 7.0; kernel 7.0 only; custom controllers as kernel
modules. M0 confirmed D2 and D5 and measured their consequences. D19 later
dropped D3, which leaves UML as the only tier. D6 to D24 were made during
the build, where the brief did not specify (its later sections were
missing).

| | Decision | Why, and what was rejected |
|---|---|---|
| D6 | The link model is a kernel qdisc (`sch_mmlink`) | It runs inside the kernel under test, with no per-packet user/kernel crossings and exact semantic times. Rejected: Mahimahi itself inside the guest (TUN copies, a heavy build, ms timers in userspace); netem for the delay plus a separate link (four namespaces, two models to validate). |
| D7 | One shared record ring, one global sequence, split by type in the guest | Loss is detectable (sequence gaps), events have a total order, and the host decodes them vectorized. |
| D8 | Observations come from the `tcp_probe` tracepoint | UML has no kprobes, and stock CCs stay unmodified. Rejected: polling `TCP_INFO` (not per ACK, and spinning distorts virtual time). |
| D9 | `arch/um` HZ patched to 1000 | UML hard-codes 100, which would make TCP's jiffy timers 10x coarser than on an HZ=1000 host kernel. |
| D10 | A seeded-randomness patch for UML | Bit-reproducible episodes. Host entropy leaked in through three paths (see `third_party/README.md`). |
| D11 | The kernel and Mahimahi are vendored as unmodified import commits plus patch commits | Builds are self-contained (kernel.org outages), and `git diff <import>` is the patch series. |
| D12 | Parquet shards with a DONE marker | Few large files per worker (HPC filesystems), atomic visibility, rerunnable shards. |
| D13 | ONNX via `tract` in the guest | Pure Rust, static musl, deterministic, zero virtual-time cost. It needs a C compiler for its SIMD kernels at build time (`musl-tools`). |
| D14 | BBRv3 from google/bbr as a loadable module, with its core changes kept inert for other CCs | The 12 core prerequisites are imported unchanged except one, which is made opt-in per CC (`TCP_CONG_BROAD_APP_LIMITED`). `tcp_bbr3` is a module loaded only for `bbr3` episodes, because building it in changed seeded BBRv1 runs. Verified by byte-identical stock episodes across the change. |
| D15 | Pinned kernel build metadata | uname's version string seeds the guest's random pool. Pinning it makes seeded runs comparable across kernel builds, which is what the D14 check relies on. (It does not make them comparable across guest versions: `docs/validation.md`, M10.) |
| D16 | Random loss sits in `sch_mmlink`, before the link, in both directions | One qdisc keeps one event order and one record stream. Lost packets never consume capacity, so the capacity and cross-traffic labels stay exact. No single Mahimahi shell nesting matches this in both directions (`mm-loss` between `mm-delay` and `mm-link` matches it for data only). Mahak uses no loss, so nothing in its semantics depends on the choice. Rejected: netem (a second model and a fourth namespace). |
| D17 | The guest runner generates cross traffic through the same bottleneck, told apart by sender port | No extra hosts or namespaces, deterministic under the seed, and the link records already carry ports. Rejected: replaying captured traffic (not reactive), and observing cross flows (the label is the share they take, not their state). |
| D18 | Mahak's driver is ported onto scikit-learn, not run from Mahak's repository | Mahak vendors an old modAL that breaks on scikit-learn 1.8 and later, and the algorithm itself is about 30 lines. The port is checked against the unmodified original, run in a separate environment, on a shared synthetic oracle (`tests/test_mahak.py`). |
| D19 | The native tier (D3: Linux 7.0 on Ubuntu 26.04 hosts, as root, in real time) is dropped | No environment we have can easily boot a patched kernel. Ubuntu 26.04 ships stock 7.0, so a native tier could run Cubic, BBRv1 and the agent controllers, but never BBRv3 or later kernel changes. Packaging a custom image is too much work for what it adds. Cost: UML's virtual time is never checked against a real-time kernel, so the effects of CPU cost (free under `inf-cpu`) on TCP are not measured. Its code was last present in commit `faad5be` (`src/expliccit_env/native.py`, `scripts/tier2-setup.sh`). |
| D20 | Sage's step is ported into the guest, and that port is the only implementation | Training data and deployment use the same code, so there is no train/serve skew, and the message equals Sage's, so Sage's and expliCCit's column choices apply unchanged. Rejected: computing features on the host from per-ACK records (a second implementation for deployment), and feeding raw `TCP_INFO` to the graph (it would have to reimplement Sage's windows in ONNX). The raw `TCP_INFO` of every step is kept in `steps.bin` for other feature designs. |
| D21 | One policy contract: the whole message in, the graph selects and scales, optional recurrent state | Any network design (MLP, recurrent, a different column set, uncertainty heads) runs without guest changes. It replaced `tcpinfo-v1` (8 features in, absolute cwnd and pacing out), which no policy used. |
| D22 | The trainer lives here, on PyTorch, with architectures and algorithms chosen by name, and exports one ONNX graph per policy | A network redesign is a registry entry, not a new pipeline. The scaling travels inside the graph, so deployment needs no scaler files. expliCCit's layer layout and parameter names are kept, so its released checkpoints load unchanged (from a local copy; its code is not in this repository). Splits are by path, never by row, so held-out numbers are not training accuracy. Default scaling is robust (quantile clipping, then standardization). TD3+BC is the default net2 objective, and expliCCit's own is kept for comparison. |
| D23 | The truth a policy can read is computed live in the guest from the link records, with the labels' definitions, and checked equal to them | Arms that feed true or patched LCFs to a running policy (oracle, repair) need ground truth during the episode, and a policy must see exactly what training called the truth. Rejected: an independent estimate in the kernel module (a second definition), and replaying an episode with the truth injected (the closed loop would change the trajectory). |
| D24 | Large runs are GitHub Actions runs of an experiment file, and every result is published as a release | Twenty 4-core runners for 6 hours each, free for a public repository, with no queue to wait in and no cluster account. The fan-out is the campaign's own sharding, so the workflow holds no experiment logic. Artifacts expire after at most 90 days, so every result becomes a release, pinned to its commit. Rejected: the Slurm array script (generated and checked, but no cluster is available), and one workflow per experiment (the same plan, cells, collect steps each time). |
| D25 | Each agent action names its socket by the writer's own file descriptor, and is kept in that socket's congestion-control state | Any number of learned flows share one bottleneck with no kernel change and no limit in the module: the guest runner owns every socket, so one file descriptor table covers them all, and the action lives as long as the socket. The global lock and the point where an action takes effect are unchanged, so single-agent episodes reproduce exactly (checked on the steps, actions and link records of three episodes). Rejected: slots keyed by `SO_MARK` (a fixed count and mark bookkeeping, and the observation module matches marks), and per-agent processes (separate file tables, for no gain while the runner drives every flow). |

## Extending

- **A new candidate LCF (label).** Add a pure function to
  `src/expliccit_env/labels.py` that maps an episode's records and constants
  to one value per observation row, and add a column in `labels_table`.
  Existing datasets keep `link_events` and `link_opp`, which hold what a new
  label needs, but nothing reads them back yet. Today a new label means
  rerunning the episodes, or keeping raw records with `--keep-raw`. For a
  policy to read the label, it must also be added to `guest/src/truth.rs`
  and `policy.TRUTH_COLUMNS` with the same definition. That changes the step
  record (`steps.py`, `tests/test_records_abi.py`), and
  `tests/test_truth.py` checks that the guest's value equals the label.
- **A new trace shape.** Add a variant next to `MahakStep` and
  `PeriodicLevels` in `spec.py`, with a `timestamps()` method that returns
  Mahimahi trace timestamps. The kernel only ever sees per-ms counts.
  `TraceFile` already accepts any Mahimahi trace (pinned by sha256).
- **A learned controller.** Export it to ONNX under the `sage-v1` contract
  (`src/expliccit_env/policy.py`: input `obs` float32 [1, 78], optional
  `state`, `truth` and `cwnd`; output a, the log2 of the window
  multiplier). Then
  use
  `{"kind": "agent", "mode": "hold" | "reno", "policy": {"path": ..., "sha256": ...}}`
  as the spec's `cc`, or as a TCP cross flow's `cc` to run it as a
  competitor; any number of flows can be learned, with the same policy or
  different ones.
- **A new kind of cross traffic.** Add a variant next to `TcpCross` and
  `UdpCross` in `spec.py` and `guest_plan.py`, and next to `CrossFlow` in
  `guest/src/plan.rs`, then start it in `guest/src/cross.rs`. It must sleep
  on absolute deadlines and stop when `Running::finish` shuts its sockets.
- **A new in-kernel congestion control.** Build it as a module: next to
  `tcp_agent`, or from the in-tree source as `scripts/build-guest.sh` does
  for the extra teachers. Enabling it in `kernel/uml.config` changes the
  kernel image, and building BBRv3 in changed seeded BBRv1 runs (D14). The
  guest loads a module `tcp_<name>.ko` only for episodes that use `<name>`.
  Then add the name to `StockName` in `spec.py`.
- **A kernel change.** Commit it under `third_party/linux` as its own
  commit. `scripts/check-kernel.sh` style-checks every such commit except
  the upstream imports listed in `third_party/linux-imports.txt`, and
  `git diff e8eadca -- third_party/linux` lists them all. Record ABI changes
  go in `kernel/modules/expliccit_rec.h` together with
  `guest/src/recorder.rs` and `src/expliccit_env/records.py` (bump
  `EXPL_REC_ABI`; `tests/test_records_abi.py` checks that the layouts
  agree).
