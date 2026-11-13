# Validation

What was checked about the emulator and the training pipeline, how, and
what was not. compact's claims rest on four properties: the ground truth
is exact, the TCP stack is the kernel's own, episodes are reproducible,
and a policy acts in the closed loop exactly as it was trained. This page
is the evidence for them.

Commands assume a UML build (`scripts/build-uml.sh build/uml &&
scripts/build-guest.sh build/uml`), `uv sync --extra train`, and, where
named, `EXPLICCIT_UML_DIR=build/uml`, a Mahak checkout at `5077583` in
`EXPLICCIT_MAHAK_DIR`, or a Sage checkout at `eedd281` and g++ in
`EXPLICCIT_SAGE_DIR`. Tests check their property, not the exact numbers
quoted, which come from the run where each check was first made.

Without a UML build or the outside checkouts, the tests that need them
skip. The UML tests run in CI ([below](#ci)).

## The ground truth is exact

| Property | How it was shown | Result | Rerun |
|---|---|---|---|
| The bottleneck equals Mahimahi's link queue | every arrival, departure, drop and opportunity of Cubic, BBRv1 and BBRv3 episodes across a 20 to 4 Mbit/s step, both directions, equals an independent Python port of `LinkQueue` fed the same arrivals; again under random loss and cross traffic | equal, event for event | `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"` |
| The Python port obeys queue laws | Hypothesis, 300 cases: conservation, FIFO, no departure in the arrival ms, droptail bounds, opportunity accounting | pass | `pytest tests/test_mahimahi.py -k laws` |
| The delay line is exact | each ACK reaches the sender after its departure plus the configured delay | within 0 to 1.3 us (the 256 ns tick) | `pytest tests/test_uml.py -k ack_delay` |
| Random loss is Bernoulli per packet | 1% on data and 0.5% on ACKs, with UDP and Cubic cross traffic | 1.04% and 0.51%, within 5 sigma | `pytest tests/test_uml.py -k loss_and_cross` |
| Records are decoded as written | numpy offsets against the C structs; sequence gaps and truncated files | equal; rejected | `pytest tests/test_records_abi.py` |
| The live truth equals the labels | a 14 s BBRv1 episode with a capacity step, loss both ways, UDP and TCP cross traffic | all 36 truth values (every stored label but `queue_delay_ns`, in both windows) equal at every one of more than 1,300 steps, bit for bit and NaN for NaN | `pytest tests/test_truth.py` |
| Bad data never becomes labels or metrics | an out-of-range policy action; a failed query; an incomplete shard | a `failed` row; never a label; an error | `pytest tests/test_agent.py tests/test_mahak.py tests/test_score.py -k "bad_policy or failed_queries or incomplete_shard"` |
| Trace generation and link metrics equal Mahak's code | traces against Mahak's `helper.py`; metrics against its Perl `mm-th-delay` on 250 logs | bit for bit; exact strings | `pytest tests/test_mahimahi.py tests/test_mahak.py` (Mahak) |

Smaller checks:

- `flow_seconds`: each flow's bytes over its seconds equal its total, and
  the delays give the episode's mean link delay exactly
  (`test_loss_and_cross_traffic`).
- `sch_flowdelay`: a cross flow with 30 ms extra RTT reports a 50 to 52 ms
  minimum RTT and a base-RTT label of exactly 50 ms
  (`test_a_cross_flow_has_its_own_rtt`).
- Several step loops: without a 1 ns settle step, 16 policies gave 1 step
  of 6,330 whose queue truth was one packet off; with it, 10 and 16
  matched exactly, and 64 (the cap) matched at 7,908 of 7,909 steps.
  Labels are exact either way.

## The stack is the kernel's own, and episodes are reproducible

| Property | How it was shown | Result | Rerun |
|---|---|---|---|
| Episodes are reproducible | the same build, plan and seed, run twice, with one or many policies | byte-identical records | `pytest tests/test_uml.py tests/test_multi_agent.py -k "bit_identical or reproducible"` |
| Linux's controllers are unchanged by the kernel patches | four seeded episodes on the kernel without and with the BBRv3 series, compared byte for byte | identical ([below](#bbrv3-and-stock-controllers)) | by hand: two kernel builds |
| The policy controllers leave Linux's controllers unchanged | 12 seeded 30 s episodes (Cubic, BBRv1, BBRv3 and Vegas; a stepping link, 1% random loss and a Cubic competitor) on the guest before and after the `agent_cubic`, `agent_pure` and `cc_info` changes, with the same kernel image | all 120 record files byte-identical (local run, 2026-11-09) | by hand: two guest builds |
| The send-buffer limit leaves Linux's controllers alone | 340 episodes with and without it | 3 changed (Hybla), by at most 0.0003 in utilization ([below](#tcp-buffer-limits)) | by hand |
| Policies drive only their own sockets | three policies next to Cubic and UDP; ten copies of one policy | each socket follows its own policy; Jain's index above 0.95; each flow's truth equals its labels | `pytest tests/test_multi_agent.py` |

## A policy acts as it was trained

| Property | How it was shown | Result | Rerun |
|---|---|---|---|
| The closed loop applies what the graph says | constant, observation-reading and recurrent policies; the window accumulator; the window input | every window and action as computed from the recorded message | `pytest tests/test_agent.py` |
| The step message is `sage.cc`'s | `sage.cc`'s own step code, compiled with the clock and `TCP_INFO` stubbed, replays a Cubic and a BBRv1-with-loss episode | every compared column of every step equal (1,382 steps on the Cubic episode); passed 2026-11-04; CI skips it | `pytest tests/test_sage.py` (Sage) |
| An exported policy acts in the guest as in PyTorch | distribution, oracle and patched graphs run in UML from the recorded inputs | within 1e-6 | `pytest tests/test_truth.py tests/test_train.py` |
| Every kind of net1 trains with every net2 and exports a graph that computes what was trained | single values, intervals and distributions, each with the offline actor and the recurrent actor taught by DAgger and continued by PPO, on all beliefs or their means; ONNX against torch step by step, and the whole CLI pipeline in UML | pass | `pytest tests/test_e2e.py` |
| The trainer measures what it claims | a distribution net1 on data with known noise covers at its nominal rates; CRPS and quantiles match known values; link splits keep links together | pass | `pytest tests/test_train.py` |
| Scores follow their rules | log utility and pairing by link | pass | `pytest tests/test_score.py` |

## Other learned controllers

The emulator runs four published controllers as their own code runs them.
Each check below was made before compact existed, against a release that
is not among compact's, and was not rerun here. compact's plan uses none
of these controllers. Rerun the check before one is used as a baseline.

| Controller | Check | Result |
|---|---|---|
| Sage's released policy | the torch rewrite against Sage's own TensorFlow stack on 1,382 recorded steps | within 2.9e-6; the exported graph's rounded action equal at 1,380 steps |
| Sage's pacing (`agent_pure`) | the pacing formula read from Sage's kernel binary, against the kernel's pacing rate at every arriving segment | equal at 3,008 of 3,008 segments |
| Orca's released policy | Orca's own state code, history and actor against the export on 600 server messages | the window percent equal at 600 of 600 steps |
| Aurora, trained with PCC-RL's code | PCC-RL's own features, history and agent against the export on 600 synthetic monitor intervals | the policy's inputs within 4.6e-5, the rate factor within 7.7e-6 |
| expliCCit's released policy | expliCCit's own evaluation code on the messages the guest recorded in 9 episodes | the applied action equal to three decimals at 11,972 of 11,973 steps (local run, 2026-11-09) |

## Feasibility findings

Probes (`tools/m0/`, 2026-10-30) asked whether Linux 7.0 runs as UML in
virtual time with the networking, modules and hooks the emulator needs. It
builds in under two minutes as a static binary, boots under
`time-travel=inf-cpu` (a guest `sleep 10` measured 10.000064512 s), runs
three namespaces with veth and TCP, loads out-of-tree modules, and lets a
module attach to `tcp:tcp_probe`. arch/um hard-codes HZ at 100 (patched to
1000). It has no kprobes, and BPF `struct_ops` cannot register, so
controllers are modules (D5). The code cites five findings:

- F1. With no rate limit on the path, CPU work costs no virtual time, so a
  bulk transfer is infinitely fast and never ends: every episode needs a
  timed bottleneck, and the host a wall-clock timeout on every guest.
- F2. A guest process that makes many syscalls without blocking advances
  virtual time artificially (one jiffy per 10000 unscheduled userspace
  switches): guest code must block or sleep, never spin.
- F3. Large queues are fine with `sch_mmlink`: 20000-packet buffers at 100
  Mbit/s run 14 times faster than real time.
- F4. UML's clock ticks every 256 ns, and every clock read from process
  context advances virtual time one tick: sender timestamps carry that
  grain, the link model does not.
- F5. Host entropy reaches UML through `setup_arch()`,
  `arch_get_random_longs()` and the command line; the seeded-randomness
  patch removes all three.

Speed and memory, measured before the step loop existed (20 s episodes,
one core each, boot included): 8 to 53 times faster than real time, peak
RSS 85 to 117 MiB per guest, and about 2.5 MB of Parquet per episode, most
of it link tables. Storage, not CPU, bounds large campaigns; `campaign run
--tables` keeps only what is needed.

## BBRv3 and stock controllers

Google's BBRv3 adds 12 commits to the TCP core. With kernel build metadata
pinned, four seeded 14 s episodes (Cubic and BBRv1, step and constant
traces, buffers of 20 to 2000 packets) were compared byte for byte on the
kernel without the series and with it. Two changes broke equality and were
removed: the series' app-limited checks altered every socket's rate
samples (now opt-in per controller, `56f58a1`), and building `tcp_bbr3`
into the kernel changed seeded BBRv1 runs through boot-time work alone
(now a module loaded only for `bbr3`). The final kernel gives identical
files. The extra Linux controllers are likewise built as out-of-tree
modules, leaving the kernel image bit-identical. The check needs two
kernel builds, so it is not in CI; rerun it when the patch set changes.

Seeded runs are reproducible for one build, not across guest versions:
rerun on a later guest, both Cubic episodes and one BBRv1 episode stayed
byte-identical, but the other BBRv1 episode diverged after 10.4 s, at a
random pacing-gain phase, because guest changes shift the kernel's random
stream.

## TCP buffer limits

The guest's TCP memory limits are 64 MB, as Sage's and ccBench's setup
scripts set them. With Linux's defaults for 1 GiB of memory (a 4 MB send
buffer), buffer-filling controllers queue less than they would on long,
deep paths: the queue of a Reno flow on 96 Mbit/s, 160 ms and 4 BDP
reaches the full 640 ms with 64 MB, against 213 ms with 4 MB
(`test_a_flow_can_fill_a_deep_buffer_on_a_long_fat_path`).

With 64 MB, a window that grows without bound floods the link, since
sending costs no virtual time: one 60 s episode of a delay-threshold
policy offered 5.4 Gbit/s to a 76 Mbit/s link, 71 arrivals per departure.
A real host with 64 MB would flood too; whether at that rate is UNRUN.
The send buffer is therefore sized to the path
([architecture.md](architecture.md#one-episode)): what Linux's autotuning
asks for four times the packets the path holds at BBR's factor (12 times
2304 bytes per packet), at least 10 segments and at most 64 MB. Linux's
own controllers' largest windows, measured on 340 episodes, were at most
2.15 times the path (6.24 for Hybla's slow start). With the limit, all 160
stock and 120 BBR episodes of that sample are identical to 64 MB, and 57
of 60 Hybla ones (the others moved by at most 0.0003 in utilization); the
growing policy of `test_a_window_far_beyond_the_path_is_bounded` puts 9.0
arrivals per departure on the link, against 54.4 with 64 MB.

The guest, the kernel and these settings have been frozen since `1365f0e`.
Step data generated under other settings is not comparable with data
generated under these; every episode row records its `repo_commit`.

## CI

`.github/workflows/ci.yml` runs on every commit that changes more than
docs: ruff, pyright and pytest with the Mahak reference and the trainer
(UML tests skip), plus workflow and shell lint. `uml.yml` runs when the
guest, kernel, modules, `src`, tests or build change: Rust format, clippy
and build, kernel style (`scripts/check-kernel.sh`), the UML build and
tests, and a smoke campaign. On demand it also runs the mainline tripwire
(D4), which on 2026-11-02 applied the kernel patches to mainline 7.4-rc1
up to `include/linux/tcp.h`, where one hunk no longer applies; the pinned
7.0 tree is unaffected.

## Not run

| Item | Status |
|---|---|
| The link model against real Mahimahi on a real host | UNRUN. `tools/crosscheck` is a kit for it. |
| UML's virtual time against a real-time kernel | Will not run (D19). |
| Decision latency in the closed loop | UNRUN: inference is free in virtual time. |
| Sage's per-ACK pacing on its own kernel | UNRUN: that kernel's source is not public. |
| sparse (`C=1`) on the modules | UNRUN: Ubuntu 24.04's sparse is too old for Linux 7.0; W=1 and checkpatch `--strict` are clean. |
| Training on GPU | UNRUN: no GPU here. |
