# Architecture

How the emulator runs one episode, and what it gives compact: an exact
ground truth for every instant (the vocabulary concepts are grounded in),
a closed loop in which any policy graph drives a real TCP sender, and
logged steps to distill from. The guarantees are in
[README.md](../README.md#guarantees), the evidence for them in
[validation.md](validation.md), and the commands in
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

Everything that crosses a process or kernel boundary is parsed once where
it enters, into types that cannot hold invalid values: specs as pydantic
models with closed sums, the plan in Rust, netlink options with the
kernel's `nla_parse_nested`, agent actions in `agent_write`, policy graphs
by input name, records and steps as numpy dtypes checked for size, type,
ABI and sequence. The host re-checks what the guest already checked. An
episode that fails any check becomes a `failed` row with its reason, and
gives no labels and no metrics.

## One episode

    snd 10.0.0.1 s0 ---- p0 10.0.0.2 [path] 10.0.1.2 p1 ---- r0 10.0.1.1 rcv
                          ^ uplink mmlink           ^ downlink mmlink

Three namespaces (sender, path, receiver) are joined by veth pairs, with
the bottleneck in the middle. One bulk TCP flow is observed. Every skb is
orphaned on entering the path, so the sender's TCP Small Queues see a
remote bottleneck. Data cross the delay line and then the link, ACKs the
reverse. IPv6 is off and ARP bypasses the model. The receiver connects at
trace time 0; both ends set `TCP_NODELAY`.

TCP settings follow the setup scripts of Sage and ccBench: `tcp_rmem` and
`tcp_wmem` up to 64 MB, `rmem_max` and `wmem_max` 128 MB, no cached
metrics, no autocorking. The send buffer alone is sized to the path
(`EpisodeSpec._send_buffer`), because a window far beyond the path costs
nothing in virtual time and would flood the link
([validation.md](validation.md#tcp-buffer-limits)).

Cross traffic shares the path in the same direction. Cross flow `i` sends
from port `6000 + i` and the observed flow from 5001, so ports tell flows
apart in the records. A TCP cross flow runs a Linux controller or a policy
of its own; a UDP cross flow sends at a constant rate. A TCP cross flow
with `extra_rtt_ms` has its ACKs held that much longer by `sch_flowdelay`.
At most 64 cross flows run, all from sender to receiver.

## The link model (`sch_mmlink`)

A port of Mahimahi's `link_queue.cc` as a qdisc, one per direction:

- Time is integer milliseconds. Trace entry `i` is the number of 1504-byte
  delivery opportunities at ms `i`; the trace repeats with its last
  timestamp as its period.
- Before a packet arriving in ms `a` is queued, every opportunity up to `a`
  is used, so no packet departs in its arrival ms. The head packet departs
  at the opportunity that carries its last byte, and droptail limits count
  queued packets only.
- A packet's size includes Mahimahi's 4-byte TUN header; GSO packets are
  segmented on entry. The delay line releases each packet exactly
  `delay_ns` after entry.
- Random loss drops each entering packet independently before the delay
  and the link, so a lost packet never uses capacity and the labels stay
  exact.

Events are processed in time order at their semantic instants
(opportunity and release times), never when the code happened to run, so
UML's clock tick and timer slack cannot change a decision.

## Time and determinism

`time-travel=inf-cpu` makes CPU work free: virtual time advances only when
the guest idles or sleeps (findings F1 to F5 in
[validation.md](validation.md#feasibility-findings)). Three consequences:

- every episode needs a timed bottleneck, and the host enforces a
  wall-clock timeout;
- guest code must block or sleep, never spin;
- inference costs no virtual time: a policy acts the instant it reads its
  step.

With the seeded-randomness kernel patch, `um_seed=` and a fixed
`time-travel-start=`, an episode is a pure function of the kernel build,
the guest build, the plan and the seed. A guest change can shift the
kernel's random stream, so seeded runs compare across guest versions only
over many seeds.

## Records

`kernel/modules/expliccit_rec.h` defines every record (ABI 4): a 16-byte
header (type, length, sequence, time) and a body. All producers append to
one ring, so an episode has one global sequence; a full ring drops whole
records, and the guest and the host both detect the gap.

| Type | Producer | Meaning (at the event's own instant) |
|---|---|---|
| META | expl_rec | ABI, HZ, ring size |
| LINK_ARRIVAL, DROP, DEPART | sch_mmlink | every packet's arrival, drop and departure, with queue state, ports and TCP seq/ack; a drop names its cause |
| LINK_OPP | sch_mmlink | the delivery opportunities of each ms; the trace is the authority on capacity |
| LINK_STATS | sch_mmlink | counters at deletion; a nonzero error counter invalidates the episode |
| TCP_OBS | expl_tcpobs | sender state just before each incoming segment is processed (`tcp:tcp_probe`) |
| AGENT_ACTION | tcp_agent | an accepted policy action, with its socket's port |

## Ground-truth labels

The labels are the hidden state of the path. They are what net1 is
supervised on, and they are the base variables of the grounding library
([methodology.md](methodology.md#terms)).

`labels.py` computes them as pure functions of an episode's records, for
the data direction; windowed labels cover the trailing 100 trace ms.
`labels_at` gives them at any instants: at every sender observation
(`labels.parquet`) and at every step (the `label_*` columns of
`steps.parquet`).

| Label | Definition |
|---|---|
| `capacity_bps` | the trace's delivery opportunities in the window, in bit/s |
| `receive_rate_bps` | the observed flow's bits leaving the bottleneck per second, over the window |
| `busy_fraction` | bytes sent over bytes the trace offered in the window; NaN at zero capacity |
| `queue_pkts`, `queue_bytes` | the queue after the last link event, without the packet in transit |
| `queue_delay_mean_ns` | mean queueing delay of the packets (all flows) that left the bottleneck in the window; NaN if none did |
| `queue_delay_ns` | the latest data packet's exact queueing delay |
| `free_buffer_pkts` | the packet limit minus the queue; NaN without a limit |
| `buffer_pkts`, `buffer_bytes`, `buffer_fill` | the buffer's limits, and the queue as a fraction of the limit |
| `one_way_delay_ns`, `base_rtt_ns` | one delay line, and twice it |
| `drop_fraction` | the flow's packets dropped by the full queue over those that reached it, in the window |
| `random_loss_rate` | the configured random loss on the data direction |
| `cross_traffic_share` | the share of the bottleneck's capacity that every other flow used in the window |
| `arrival_rate_bps` | bits of every flow's packets that reached the bottleneck queue per second, over the window, counting those the full queue then dropped |
| `flow_arrival_rate_bps` | the same for the observed flow alone |
| `active_flows` | flows (protocol and sender port) with at least one packet reaching the bottleneck in the window, the observed flow included |

The seven LCFs of expliCCit are `queue_delay_mean_ns`, `queue_pkts`,
`capacity_bps`, `free_buffer_pkts`, `busy_fraction`, `one_way_delay_ns`
and `receive_rate_bps`. The arrival rates and the flow count are there
because many published concepts are closed forms over them
([concepts.md](concepts.md#what-this-means-for-compact)).

## Sage's step and the policy contract (`sage-v1`)

The sender's side of every learned controller is one step loop,
`guest/src/step.rs`, a port of the step of Sage's `sage.cc` (commit
`eedd281`). It is the only implementation, for training data and for
deployment alike, so there is no train/serve skew. Every 10 ms of virtual
time it reads `TCP_INFO`; once the ACK state has moved it builds a
78-value message: RTT, delivery rate and window, their statistics over 10,
200 and 1000 steps, the log2 window change since the last step (column
76) and the min RTT (column 77). These are the sender metrics: what a
policy can see, and the other half of the grounding library.

A policy is an ONNX graph (`policy.py`) with inputs matched by name:

| Input | Shape | Meaning |
|---|---|---|
| `obs` | [1, 78] | the message; the graph selects and scales its own columns |
| `state` | optional | a recurrent state fed back from the graph's second output |
| `truth` | [1, 7], optional | the true values of the seven LCFs at this step (an oracle) |
| `cwnd` | optional | the current window, so a graph can answer a = log2(target) - log2(cwnd) |

It returns a, the log2 window multiplier. The window starts from 10
packets and is multiplied by 2^a each step, rounded up, in [4, 20000];
pacing is cwnd mss / srtt, refreshed each step. A non-finite a, or
|a| > 8, fails the episode. `noise_sigma` adds seeded Gaussian noise to a.
Graphs are exported with a batch dimension of 1.

`guest/src/truth.rs` computes the truth live from the downlink's records
with the labels' definitions, so it equals the labels stored with the step
bit for bit (D23). It covers the seven LCFs only.

`steps.parquet` holds per step the time, the raw `TCP_INFO`, the message,
the labels, the action applied and the window set; a policy cross flow
writes its own steps (`cross_steps`). `campaign run --tables` keeps only
the named tables; most experiments keep only `steps`.

## Controllers

- **Linux's own.** All 17 kernel controllers (`StockName` in `spec.py`),
  unmodified; BBRv3 is Google's, as a module
  ([third_party/README.md](../third_party/README.md)).
- **Policy graphs.** `tcp_agent` keeps each action in its socket's
  congestion-control state and applies it at the next ACK, so any number
  of policies share a bottleneck. The `agent` controller holds the window and
  pacing the policy set, and is what our own policies train and run
  under.
- **Other learned controllers.** Three more modes run published
  controllers as their own code runs them: `agent_pure` (Sage: the same
  window, paced at every delivery), `agent_cubic` (Orca: the window times
  2^a once per action, Cubic in between, its statistics through
  `TCP_CC_INFO`) and `agent_rate` (Aurora: a paced sending rate, one step
  per half RTT). expliCCit's released net1 and actor are in
  `third_party/expliccit-models`. These are candidates for further
  teachers; none is one yet.

Every model runs on Linux 7.0's stack as it is. Nothing of the 4.x
kernels those controllers were built for is emulated, so results may
differ from their original deployments.

## Experiments in GitHub Actions

An experiment is one JSON file under `experiments/` (format in
`tools/ci/experiment.py`), dispatched with
`gh workflow run experiment.yml -f experiment=experiments/NAME.json`.
`.github/workflows/experiment.yml` runs plan (expand the cells, run
`prepare`), uml (the UML build from the Actions cache), cells (`run` per
cell: at most 256, 20 at a time, 6 h each, on 4-core runners), collect
(only once every cell succeeded) and publish: release `exp-NAME-RUN`, with
the experiment's JSON and `summary.md` in its notes.

Releases are the record; artifacts only carry data between jobs and
expire. `republish.yml` publishes a run whose publish job failed while its
artifact lasts.

A later experiment reads a release of this repository through
`"inputs": {"NAME": "release:TAG"}`. Two ways put outside files into a
release:

- an experiment whose `run` downloads them and whose description records
  the source (`experiments/phase0-teachers.json`,
  `experiments/phase0-steps.json`);
- `scripts/stash.sh TAG FILE...` for local files under 100 MB each: it
  pushes a branch `stash/TAG`, and `stash.yml` publishes the files and
  deletes the branch, so they never enter main's history.

Many experiment files under `experiments/` predate compact and name
inputs that are not among its releases. They remain as worked examples of
the commands; `experiments/smoke.json` runs as it is.

## Known limits

Limits marked `ponytail` in the code name their upgrade path there.

| Limit | Consequence |
|---|---|
| The live `truth` carries the seven LCFs only | an oracle on any other concept needs `truth.rs` and `policy.TRUTH_COLUMNS` extended |
| net1 can be trained only on the named labels (`Lcf` in `train/spec.py`) | a label defined by a fitted formula needs that type and `labels.py` extended |
| Labels are computed when an episode is processed | no code recomputes them from stored link tables; a new label means rerunning the episodes |
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

- **A label**: add it to `labels_at` in `labels.py` (it feeds
  `labels.parquet` and the step columns) and rerun the episodes. For net1
  to predict it, add it to `Lcf` in `train/spec.py`. For a policy to read
  it live, add it to `guest/src/truth.rs` and `policy.TRUTH_COLUMNS`;
  `tests/test_records_abi.py` and `tests/test_truth.py` then check the
  step record and that guest and host agree.
- **A kind of belief for net1** (beside a value, an interval and a
  distribution): a subclass of `beliefs.Beliefs` and the net1 module that
  states it; `tests/test_e2e.py` then trains, exports and runs it with
  every net2.
- **A net2 input or window target**: a `Front` option or a target module
  in `train/graph.py`. Training reads net2's inputs through the same front
  end the exported graph runs, so the two cannot drift apart.
- **A learned controller**: an ONNX graph under `sage-v1`, used as
  `{"kind": "agent", "mode": "hold", "policy": {"path": ..., "sha256": ...}}`
  for the observed flow or a TCP cross flow.
- **A trace shape**: a variant next to `MahakStep` and `PeriodicLevels` in
  `spec.py`; `TraceFile` takes any Mahimahi trace.
- **Cross traffic**: a variant next to `TcpCross` and `UdpCross` in
  `spec.py` and `guest_plan.py` and next to `CrossFlow` in
  `guest/src/plan.rs`, sleeping on absolute deadlines.
- **A kernel change**: its own commit under `third_party/linux`, recorded
  in `third_party/README.md`. A record layout change touches
  `expliccit_rec.h`, `guest/src/recorder.rs` and `records.py` together and
  bumps `EXPL_REC_ABI`.
