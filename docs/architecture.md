# Architecture

How the emulator runs one episode. compact uses three things from it: the
exact hidden state of the path at every instant, a closed loop in which a
policy graph drives a real TCP sender, and a log of every step. The
guarantees are in [README.md](../README.md#guarantees), the evidence for
them in [validation.md](validation.md), and the commands in
[training.md](training.md).

## Data flow and trust boundaries

    spec / campaign JSON ─► spec.py (parse) ─► guest_plan.py: plan.json, *.trace, policy-<hash>.onnx
       ─► uml.py: UML with time-travel=inf-cpu, the spec's seed, a wall-clock timeout
       ─► guest runner, PID 1 (Plan::load parses again): namespaces, qdiscs, modules, flows,
          the step loop, the policy, the live truth; /dev/expl_rec ─► one file per record type
       ─► episode.py (parses result.json, records, steps again): validity checks,
          labels.py, steps.py, link metrics
       ─► shard: episodes, obs, labels, link_events, link_opp, agent_action, steps,
          cross_steps, flow_seconds (.parquet), DONE

Whatever crosses a process or kernel boundary is parsed once where it
enters, into a type that cannot hold an invalid value. Specs are pydantic
models with closed sums. The plan is parsed in Rust, netlink options by
the kernel's `nla_parse_nested`, agent actions in `agent_write`, policy
graphs by input name, and records and steps as numpy dtypes checked for
size, type, ABI and sequence. The host checks again what the guest
checked. An episode that fails any check becomes a `failed` row with its
reason. It gives no labels and no metrics.

## One episode

    snd 10.0.0.1 s0 ---- p0 10.0.0.2 [path] 10.0.1.2 p1 ---- r0 10.0.1.1 rcv
                          ^ uplink mmlink           ^ downlink mmlink

Three namespaces (sender, path, receiver) are joined by veth pairs, with
the bottleneck in the middle. One bulk TCP flow is observed. Every skb is
orphaned on entering the path, so the sender's TCP Small Queues see a
remote bottleneck. Data cross the delay line and then the link, and ACKs
go the other way. IPv6 is off and ARP bypasses the model. The receiver
connects at trace time 0, and both ends set `TCP_NODELAY`.

TCP settings follow the setup scripts of Sage and ccBench: `tcp_rmem` and
`tcp_wmem` up to 64 MB, `rmem_max` and `wmem_max` 128 MB, no cached
metrics, no autocorking. Only the send buffer is sized to the path
(`EpisodeSpec._send_buffer`). A window far beyond the path costs nothing
in virtual time, so without that limit it would flood the link
([validation.md](validation.md#tcp-buffer-limits)).

Cross traffic shares the path in the same direction. Cross flow `i` sends
from port `6000 + i` and the observed flow from 5001, so ports tell flows
apart in the records. A TCP cross flow runs a Linux controller or a
policy of its own. A UDP cross flow sends at a constant rate. A TCP cross
flow with `extra_rtt_ms` has its ACKs held that much longer by
`sch_flowdelay`. At most 64 cross flows run, all from sender to receiver.

A TCP cross flow sends only inside its planned window, but it does not
start exactly at `start_ms`. Its receiver connects then, and the flow
begins when the handshake completes. Behind a standing queue that can
take seconds, or fail if the SYN-ACKs are dropped. The episodes table
records when each flow connected (`cross_connected_ms`, null for UDP and
for a flow that never connected and so sent nothing). The privileged
critic's rival counts read that time and not the plan.

At `stop_ms` both ends are aborted. The sender's unsent data is
discarded, where a graceful shutdown would go on sending it for seconds
on a deep buffer, and a handshake that completes later is refused. After
the stop, data in flight reaches the link for one more one-way delay,
resets answer the ACKs in flight for about one RTT, and the flow's
packets already queued at the bottleneck drain.

## The link model (`sch_mmlink`)

A port of Mahimahi's `link_queue.cc` as a qdisc, one per direction.

- Time is integer milliseconds. Trace entry `i` is the number of
  1504-byte delivery opportunities at ms `i`. The trace repeats with its
  last timestamp as its period.
- Before a packet arriving in ms `a` is queued, every opportunity up to
  `a` is used, so no packet departs in its arrival ms. The head packet
  departs at the opportunity that carries its last byte. Droptail limits
  count queued packets only.
- A packet's size includes Mahimahi's 4-byte TUN header. GSO packets are
  segmented on entry. The delay line releases each packet exactly
  `delay_ns` after entry.
- Random loss drops each entering packet independently, before the delay
  and the link. A lost packet never uses capacity, so the labels stay
  exact.

Events are processed in time order at their own instants (opportunity and
release times), never at the time the code happened to run. UML's clock
tick and timer slack therefore cannot change a decision.

## Time and determinism

`time-travel=inf-cpu` makes CPU work free. Virtual time advances only
when the guest idles or sleeps (findings F1 to F5 in
[validation.md](validation.md#feasibility-findings)). It follows that:

- every episode needs a timed bottleneck, and the host enforces a
  wall-clock timeout;
- guest code must block or sleep, never spin;
- inference costs no virtual time, so a policy acts the instant it reads
  its step.

With the seeded-randomness kernel patch, `um_seed=` and a fixed
`time-travel-start=`, an episode is a pure function of the kernel build,
the guest build, the plan and the seed. A guest change can shift the
kernel's random stream, so seeded runs compare across guest versions only
over many seeds.

## Records

`kernel/modules/expliccit_rec.h` defines every record (ABI 4): a 16-byte
header (type, length, sequence, time) and a body. All producers append to
one ring, so an episode has one global sequence. A full ring drops whole
records, and the guest and the host both detect the gap.

| Type | Producer | Meaning, at the event's own instant |
|---|---|---|
| META | expl_rec | ABI, HZ, ring size |
| LINK_ARRIVAL, DROP, DEPART | sch_mmlink | every packet's arrival, drop and departure, with queue state, ports and TCP seq/ack; a drop names its cause |
| LINK_OPP | sch_mmlink | the delivery opportunities of each ms; the trace is the authority on capacity |
| LINK_STATS | sch_mmlink | counters at deletion; a nonzero error counter invalidates the episode |
| TCP_OBS | expl_tcpobs | sender state just before each incoming segment is processed (`tcp:tcp_probe`) |
| AGENT_ACTION | tcp_agent | an accepted policy action, with its socket's port |

## Ground-truth labels

The labels are the hidden state of the path. net1 is supervised on them,
and they are the base variables of the grounding library
([methodology.md](methodology.md#terms)).

`labels.py` computes them as pure functions of an episode's records, for
the data direction. `labels_at` gives them at any instants: at every
sender observation (`labels.parquet`) and at every step (the `label_*`
columns of `steps.parquet`).

A windowed label covers the whole trace ms just before the sample. The
steps store each one twice: over 100 ms under its own name, and over
10 ms as `<name>_10ms`. `labels.parquet` has a row per ACK and keeps the
100 ms window only. With both windows the steps hold 37 label columns.

Labels of the instant:

| Label | Definition |
|---|---|
| `queue_pkts`, `queue_bytes` | the queue after the last link event, without the packet in transit |
| `flow_queue_pkts` | the observed flow's packets in `queue_pkts`; over all flows they add up to it |
| `queue_delay_ns` | the latest data packet's exact queueing delay |
| `free_buffer_pkts` | the packet limit minus the queue; NaN without a limit |
| `buffer_pkts`, `buffer_bytes`, `buffer_fill` | the buffer's limits, and the queue as a fraction of the limit |
| `one_way_delay_ns`, `base_rtt_ns` | one delay line, and twice it |
| `random_loss_rate` | the probability of random loss the link applies on the data direction |

Windowed labels:

| Label | Definition |
|---|---|
| `capacity_bps` | the trace's delivery opportunities in the window, in bit/s |
| `capacity_ahead_bps` | `capacity_bps` of the window that starts with the sample's ms; the one label that looks forward |
| `receive_rate_bps` | the observed flow's bits leaving the bottleneck per second |
| `busy_fraction` | bytes sent over bytes the trace offered; NaN at zero capacity |
| `queue_delay_mean_ns` | mean queueing delay of the packets (all flows) that left the bottleneck; NaN if none did |
| `ack_queue_delay_mean_ns` | the same on the ACK direction's link: delay the sender sees that the bottleneck queue did not cause |
| `queue_min_pkts` | the smallest queue at any instant of the window (the standing queue); NaN for an empty window |
| `drop_fraction` | the flow's packets dropped by the full queue over those that reached it |
| `random_drop_fraction` | the flow's packets the random loss dropped before the bottleneck, over those it sent toward it; NaN if none |
| `cross_traffic_share` | the share of the bottleneck's capacity that every other flow used |
| `arrival_rate_bps` | bits of every flow reaching the bottleneck queue per second, those the full queue then dropped included |
| `flow_arrival_rate_bps` | the same for the observed flow alone |
| `active_flows` | flows with a packet reaching the bottleneck in the window |

The seven LCFs of expliCCit are `queue_delay_mean_ns`, `queue_pkts`,
`capacity_bps`, `free_buffer_pkts`, `busy_fraction`, `one_way_delay_ns`
and `receive_rate_bps`, over 100 ms.

A label added later needs no rerun where the link tables were kept:
`dataset.read_episode` turns a shard's `obs`, `link_events` and
`link_opp` back into the records, and `labels_at` computes the label from
them. A shard written with `--tables steps` holds no link tables and must
be rerun.

### Composites

A composite is a fixed formula over the label columns of one window
(`train/lcfs.py`). It is not stored. It is computed from the stored
columns when it is asked for, so a new composite needs no new data. Each
comes in both windows. A set that holds a composite beside the columns it
is made of is redundant by construction.

With C the capacity, U the busy fraction, y the arrival rate, q the queue
in bytes, R the base RTT and N the active flows:

| Composite | Formula | Source ([concepts.md](concepts.md)) |
|---|---|---|
| `departure_rate_bps` | C U | RFC 5136 (usage), HPCC |
| `available_capacity_bps` | C (1 - U) | RFC 5136 |
| `excess_demand_bps` | y - C | XCP |
| `load_factor` | y / C; TIMELY's queue growth rate is this less 1 | the OSU scheme, with the target at capacity |
| `bdp_pkts` | C R, in 1504-byte packets | CCmatic |
| `littles_delay_ns` | q / (C U) | PIE |
| `normalized_inflight` | q / (C R) + U | HPCC |
| `normalized_power` | (1 + q / (C R)) y / C | PowerTCP |
| `cross_arrival_rate_bps` | y less the observed flow's arrival rate | Dovrolis 2001 |
| `fair_share_bps` | C / N | RCP |
| `sending_ratio` | the flow's arrival rate over its receive rate | Aurora, measured at the link |
| `rtt_ratio` | (R + mean queueing delay) / R | Aurora, Remy, on the true delays |

A ratio is NaN where its denominator is 0.

Every stored label and every composite, in each window, is a name net1
can be trained on (`net1.lcfs`) and a base variable of the grounding
library (`experiments/ground.py`). Neither place lists them: a label
added to `labels.py` or a formula added to `lcfs.COMPOSITES` joins both.

## The step and the policy contract (`sage-v1`)

The sender's side of every learned controller is one step loop,
`guest/src/step.rs`, a port of the step of Sage's `sage.cc` (commit
`eedd281`). Training data and deployment use this one implementation, so
the two cannot differ. Every 10 ms of virtual time it reads `TCP_INFO`.
Once the ACK state has moved it builds a 78-value message: RTT, delivery
rate and window, their statistics over 10, 200 and 1000 steps, the log2
window change since the last step (column 76) and the min RTT
(column 77). These are the sender metrics. They are all a policy can see,
and they are the other half of the grounding library.

A policy is an ONNX graph (`policy.py`) with inputs matched by name.

| Input | Shape | Meaning |
|---|---|---|
| `obs` | [1, 78] | the message; the graph selects and scales its own columns |
| `state` | optional | a recurrent state fed back from the graph's second output |
| `truth` | [1, W], optional | the path's true state at this step (an oracle): the first W of `policy.TRUTH_COLUMNS` |
| `cwnd` | optional | the current window, so a graph can answer a = log2(target) - log2(cwnd) |

The graph returns a, the log2 window multiplier. The window starts from
10 packets and is multiplied by 2^a each step, rounded up, within
[4, 20000]. Pacing is cwnd mss / srtt, refreshed each step. A non-finite
a, or |a| > 8, fails the episode. `noise_sigma` adds seeded Gaussian
noise to a. Graphs are exported with a batch dimension of 1.

`guest/src/truth.rs` computes the truth live from the links' records
with the labels' definitions, so it equals the labels stored with the
step bit for bit (D23). The truth is `policy.TRUTH_COLUMNS`, 36 values:
every stored label, over both windows, except `queue_delay_ns`, the delay
of a packet that may not have left yet. `capacity_ahead_bps` looks
forward but is in the truth, because it reads the trace, which the guest
holds. A composite is not a column of the truth. An exported policy
computes it from the columns with the labels' own formula (`lcfs.value`,
`graph.TruthColumns`). expliCCit's seven come first and a policy declares
the width it reads, so a policy exported with `truth` [1, 7] still runs
and reads the same values.

`steps.parquet` holds per step the time, the raw `TCP_INFO`, the message,
the labels, the action applied and the window set. A policy cross flow
writes its own steps (`cross_steps`). `campaign run --tables` keeps only
the named tables, and most experiments keep only `steps`.

## Controllers

All 17 kernel controllers (`StockName` in `spec.py`) run unmodified.
BBRv3 is Google's, as a module
([third_party/README.md](../third_party/README.md)).

A policy graph runs under `tcp_agent`, which keeps each action in its
socket's congestion-control state and applies it at the next ACK. Any
number of policies can therefore share a bottleneck. The `agent`
controller holds the window and pacing the policy set. compact's own
controllers train and run under it.

Three more modes run published controllers the way their own code does.

| Mode | Controller | What it does |
|---|---|---|
| `agent_pure` | Sage | the same window, paced at every delivery |
| `agent_cubic` | Orca | the window times 2^a once per action, Cubic in between, its statistics through `TCP_CC_INFO` |
| `agent_rate` | Aurora | a paced sending rate, one step per half RTT |

expliCCit's released net1 and actor are in
`third_party/expliccit-models`. Every model runs on Linux 7.0's stack as
it is. Nothing of the 4.x kernels those controllers were built for is
emulated, so results may differ from their original deployments.

## Experiments in GitHub Actions

An experiment is one JSON file under `experiments/` (format in
`tools/ci/experiment.py`), dispatched with
`gh workflow run experiment.yml -f experiment=experiments/NAME.json`.
`.github/workflows/experiment.yml` runs five jobs:

1. plan expands the cells and runs `prepare`;
2. uml takes the UML build from the Actions cache;
3. cells runs `run` per cell (at most 256 cells, 20 at a time, 6 h each,
   on 4-core runners);
4. collect runs only once every cell succeeded;
5. publish makes release `exp-NAME-RUN`, with the experiment's JSON and
   `summary.md` in its notes.

Releases are the record. Artifacts only carry data between jobs, and they
expire. `republish.yml` publishes a run whose publish job failed, while
its artifact lasts.

A later experiment reads a release of this repository through
`"inputs": {"NAME": "release:TAG"}`. Outside files enter a release in one
of two ways. An experiment's `run` downloads them and its description
records the source (`experiments/estimator-seven.json` is one). Or
`scripts/stash.sh TAG FILE...` takes local files under 100 MB each: it
pushes a branch `stash/TAG`, and `stash.yml` publishes the files and
deletes the branch, so they never enter main's history.

Many experiment files under `experiments/` came with the fork and name
inputs that are not among compact's releases. They remain as worked
examples of the commands. `experiments/smoke.json` runs as it is.

## Known limits

Limits marked `ponytail` in the code name their upgrade path there.

| Limit | Consequence |
|---|---|
| The live `truth` leaves out `queue_delay_ns` | an oracle or a patched belief cannot use that one label; the window's mean queueing delay is there |
| Labels are stored when an episode is processed | a label added later is computed from the stored link tables; a shard written with `--tables steps` has none and must be rerun |
| Inference and decisions cost no virtual time | the effect of decision latency is not measured |
| One bottleneck: droptail, fixed delay, i.i.d. loss | no AQM, ECN, jitter, bursty loss or multiple bottlenecks |
| One decision per 10 ms step | pacing is refreshed per step, not per ACK |
| Several step loops each sleep 1 ns before reading their truth | at 64 policies, 1 of 7,909 steps saw a truth that differed from the labels; labels are always exact |
| Per-ms u8 trace counts | at most 3 Gbit/s and about 65 s per trace |
| Whole-link metrics | with cross traffic, `utilization` and `mean_delay_ms` include other flows |
| Virtual time is never checked against a real-time kernel | the effect of CPU cost on TCP is not measured (D19) |

## Decisions

The design choices a reader is most likely to question. The numbers are
the ones the code's comments use.

| | Decision | Reason, and what was rejected |
|---|---|---|
| D2 | The kernel runs as UML in virtual time | No root or KVM, any host, reproducible. |
| D4 | Linux 7.0 only | A tripwire (`uml.yml` with `mainline`, `tools/ci/mainline_tree.py`) shows on demand where the kernel patches stop applying to the newest mainline. |
| D5 | Congestion controls are kernel modules | BPF `struct_ops` cannot register in UML (no BPF JIT on arch/um). |
| D6 | The link model is a kernel qdisc | In the kernel under test, without per-packet crossings, at exact instants. Rejected: Mahimahi in the guest; netem plus a separate link. |
| D7 | One record ring with one global sequence | Loss is detectable, events are ordered, decoding is vectorized. |
| D13 | ONNX through `tract` in the guest | Pure Rust, static, deterministic, no virtual-time cost. |
| D19 | No real-time tier | No environment here boots a patched kernel easily. Cost: virtual time is never checked against real time. |
| D21 | One policy contract: the whole message in, the graph selects and scales | Any network runs without guest changes. |
| D23 | A policy's truth is computed live from link records with the labels' definitions | Oracles need exactly the training truth during the episode. Rejected: a second definition in the kernel; replaying episodes. |
| D24 | Large runs are GitHub Actions experiments, every result a release | Twenty 4-core runners, free for a public repository; artifacts expire. |

## Extending

| To add | Change |
|---|---|
| A label | `labels_at` in `labels.py` (it feeds `labels.parquet` and the step columns). If it is known at the step, add it to `guest/src/truth.rs` and at the end of `policy.TRUTH_COLUMNS` (released policies read a prefix); `tests/test_records_abi.py` and `tests/test_truth.py` then check the step record and that guest and host agree. |
| A composite | a formula in `lcfs.COMPOSITES` over label columns. It needs no new data and no guest change. |
| A kind of belief for net1, beside a value, an interval and a distribution | A subclass of `beliefs.Beliefs` and the net1 module that states it. `tests/test_e2e.py` then trains, exports and runs it with every net2. |
| A net2 input or window target | A `Front` option or a target module in `train/graph.py`. Training reads net2's inputs through the same front end the exported graph runs, so the two cannot drift apart. |
| A learned controller | An ONNX graph under `sage-v1`, used as `{"kind": "agent", "mode": "hold", "policy": {"path": ..., "sha256": ...}}` for the observed flow or a TCP cross flow. |
| A trace shape | A variant next to `MahakStep` and `PeriodicLevels` in `spec.py`. `TraceFile` takes any Mahimahi trace. |
| Cross traffic | A variant next to `TcpCross` and `UdpCross` in `spec.py` and `guest_plan.py`, and next to `CrossFlow` in `guest/src/plan.rs`, sleeping on absolute deadlines. |
| A kernel change | Its own commit under `third_party/linux`, recorded in `third_party/README.md`. A record layout change touches `expliccit_rec.h`, `guest/src/recorder.rs` and `records.py` together and bumps `EXPL_REC_ABI`. |
