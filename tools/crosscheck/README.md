# Cross-check kit: palaestra against the original stack

This kit compares palaestra with the stack that Sage and expliCCit ran on:
Ubuntu 16.04, Sage's patched Linux 4.19 (`4.19.112-0062`) and ccBench's
patched Mahimahi. The team runs four links there with five controllers and
sends back mm-link's downlink logs. `compare.py` computes the same metrics
from those logs and from palaestra's runs of the same links, and prints them
side by side. That is the reproduction phase's check that UML with
`sch_mmlink` reproduces the original setup and roughly its results; the
top-level README lists the reference runs of real Mahimahi as UNRUN.

| File | What |
|---|---|
| `links.py` | The four links as palaestra specs. `write DIR` writes their trace files and `links.json`; `plan OUT` plans palaestra's campaign of them |
| `compare.py` | Reads the team's logs and palaestra's episodes, and prints the table |
| `experiments/crosscheck.json` | palaestra's side as a CI experiment: every link with Cubic, BBRv1, Vegas, Sage's released policy and expliCCit's released policy as deployed, seeds 1 to 3. Its release holds the episodes `compare.py` reads, and the kit |

## The links

| Link | What | Capacity | mm-delay (ms, one way) | Data loss | Buffer (packets, each way) | Run (s) | Window (s after the first data packet) |
|---|---|---|---|---|---|---|---|
| a | expliCCit's case-study link (`experiments/expliccit_case.py`) | 24 and 48 Mbit/s, alternating every 7 s | 40 | none | 1000 | 45 | 2 to 44 |
| b | constant, 1 BDP | 48 Mbit/s | 20 | none | 160 | 30 | 2 to 29 |
| c | constant, 4 BDP, random loss | 12 Mbit/s | 40 | 1% | 320 | 30 | 2 to 29 |
| d | a cell of ccBench's solo suite (`experiments/ccbench.py`) | 96 Mbit/s | 80 | none | 5120 | 30 | 2 to 29 |

Buffers are in packets, as ccBench's `solo_runall.sh` computes them and
palaestra's ccBench suite copies: one BDP is D * B // 6 packets for a one-way
delay of D ms and B Mbit/s (b: 160; c: 4 x 80; d: 4 x 1280).

Make the files with

    uv run python tools/crosscheck/links.py write kit

or take `kit/` from the release of `experiments/crosscheck.json`, which holds
the files that palaestra's run used. Each `.trace` is a Mahimahi trace: one
millisecond timestamp per line, each line one delivery opportunity of 1504
bytes (a 1500-byte packet and the 4-byte header Mahimahi counts), and the
file repeats with its last timestamp as its period. The timestamps are the
ones palaestra's specs generate (`expliccit_env.mahimahi.periodic_trace`),
and `links.py` checks that the file reads back as the same trace.
`b.trace` equals the first 7 s of ccBench's `traces/wired48`. `c.trace` and
`d.trace` deliver what ccBench's `wired12` and `wired96` deliver from ms 1
on (`wired96` also has 8 opportunities at ms 0). `links.json` gives, per
link, the trace file and its sha256, the mm-delay, mm-loss and mm-link
arguments, the shell line, the duration, the window, and the palaestra spec.

## How palaestra's path maps to Mahimahi

palaestra's bottleneck is Mahimahi's `mm-delay`, `mm-loss` and `mm-link`
ported to a qdisc, one in each direction (`docs/architecture.md`, "The link
model"). To build the same path from Mahimahi's shells:

- Run the sender outside every shell and the receiver inside, so data
  crosses the downlink and ACKs cross the uplink. The receiver connects to
  the sender at `$MAHIMAHI_BASE`, which the outermost shell sets to the
  host's address.
- Nest the shells as `mm-delay D [mm-loss downlink P] mm-link UP DOWN ... --
  RECEIVER`. A data packet then crosses the delay, the loss and the link
  queue, in that order, and an ACK crosses the uplink queue, then the delay,
  as in palaestra. palaestra drops a lost packet before its delay line, and
  this nesting drops it after the delay. With i.i.d. loss that makes no
  difference: a lost packet never reaches mm-link either way. Sage's own
  harness nests the shells in this order (`sage_rl/src/sage.cc`), and
  ccBench's `cc_solo.sh` puts `mm-loss` outside `mm-delay`, which is the
  same for the same reason.
- The ACK path (uplink). Unless a spec sets `uplink_trace` or `loss.uplink`,
  palaestra gives the ACK path the data direction's trace, the same
  droptail packet limit and the same one-way delay, and no loss, as Mahak
  does. To match it in Mahimahi, pass the same trace file for both of
  mm-link's trace arguments (mm-link takes the uplink trace first), give
  `--uplink-queue=droptail --uplink-queue-args=packets=N` with the data
  direction's N, and add no `mm-loss uplink`. Sage's and ccBench's harnesses
  already do the first two.
- Use ccBench's patched Mahimahi (`patches/mahimahi.core.v2.2.patch`). It
  raises the shells' txqueuelen to 400000; ccBench's notes say that without
  it a long, fast path such as link d may not fill.

## Host setup

Do this once per boot, on Sage's kernel (`uname -r` prints `4.19.112-0062`):

    bash ~/sage/sage_rl/set_sysctl.sh
    sudo modprobe tcp_bbr
    sudo modprobe tcp_vegas
    sysctl net.ipv4.tcp_available_congestion_control    # must list cubic, bbr and vegas

`set_sysctl.sh` sets what palaestra's guest sets (`docs/architecture.md`,
"Topology inside one episode"): `tcp_rmem` and `tcp_wmem` up to 64 MB,
`rmem_max` and `wmem_max` of 128 MB, `tcp_no_metrics_save` 1 and
`tcp_autocorking` 0. It also sets `ip_forward` 1, which Mahimahi needs.
ccBench's `system_setup.sh` has the same lines. The Mahimahi shells refuse
to run as root, so run them as your own user.

## Kernel controllers: Cubic, BBR, Vegas

The sender is an iperf3 server outside the shells. The receiver is an iperf3
client inside them, in reverse mode (`-R`): the client receives and the
server sends. The sender's socket takes the host's default congestion
control, so set the default to the controller CC, start the server, and run
each link. These are the four command lines, with CC (`cubic`, `bbr` or
`vegas`) and the run R (1, 2 or 3) to fill in, run from `kit/`:

    mkdir -p logs
    sudo sysctl -w net.ipv4.tcp_congestion_control=CC
    pkill -x iperf3; iperf3 -s -p 5201 -D

    # link a
    mm-delay 40 mm-link a.trace a.trace --downlink-log=logs/a.CC.R.log --uplink-queue=droptail --uplink-queue-args=packets=1000 --downlink-queue=droptail --downlink-queue-args=packets=1000 -- sh -c 'iperf3 -c $MAHIMAHI_BASE -p 5201 -R -C CC -t 45'
    # link b
    mm-delay 20 mm-link b.trace b.trace --downlink-log=logs/b.CC.R.log --uplink-queue=droptail --uplink-queue-args=packets=160 --downlink-queue=droptail --downlink-queue-args=packets=160 -- sh -c 'iperf3 -c $MAHIMAHI_BASE -p 5201 -R -C CC -t 30'
    # link c
    mm-delay 40 mm-loss downlink 0.01 mm-link c.trace c.trace --downlink-log=logs/c.CC.R.log --uplink-queue=droptail --uplink-queue-args=packets=320 --downlink-queue=droptail --downlink-queue-args=packets=320 -- sh -c 'iperf3 -c $MAHIMAHI_BASE -p 5201 -R -C CC -t 30'
    # link d
    mm-delay 80 mm-link d.trace d.trace --downlink-log=logs/d.CC.R.log --uplink-queue=droptail --uplink-queue-args=packets=5120 --downlink-queue=droptail --downlink-queue-args=packets=5120 -- sh -c 'iperf3 -c $MAHIMAHI_BASE -p 5201 -R -C CC -t 30'

The single quotes keep `$MAHIMAHI_BASE` for the shell inside. The same 36
runs as one loop:

```bash
run() {  # LINK CC R
    local d q t loss=""
    case $1 in
        a) d=40 q=1000 t=45 ;;
        b) d=20 q=160 t=30 ;;
        c) d=40 q=320 t=30 loss="mm-loss downlink 0.01" ;;
        d) d=80 q=5120 t=30 ;;
    esac
    # $loss is unquoted on purpose: empty, or the shell's three words.
    # shellcheck disable=SC2086
    mm-delay "$d" $loss mm-link "$1.trace" "$1.trace" --downlink-log="logs/$1.$2.$3.log" \
        --uplink-queue=droptail --uplink-queue-args="packets=$q" \
        --downlink-queue=droptail --downlink-queue-args="packets=$q" \
        -- sh -c "iperf3 -c \$MAHIMAHI_BASE -p 5201 -R -C $2 -t $t"
}
cd kit && mkdir -p logs
for cc in cubic bbr vegas; do
    sudo sysctl -w net.ipv4.tcp_congestion_control="$cc"
    pkill -x iperf3; sleep 1; iperf3 -s -p 5201 -D; sleep 1
    for r in 1 2 3; do
        for link in a b c d; do run "$link" "$cc" "$r"; sleep 5; done
    done
done
pkill -x iperf3
```

During a run, `ss -tin` on the host shows the sender socket's controller
by name. If Ubuntu 16.04's iperf3 does not accept `-C`, drop it: the
host's default already sets the sender's controller.

## Sage

Sage's harness builds the Mahimahi command itself (`sage_rl/src/sage.cc`,
in `start_server`):

    mm-delay D mm-loss downlink P mm-loss uplink P mm-link ~/sage/traces/UP ~/sage/traces/DOWN --downlink-log=~/sage/sage_rl/log/down-LOG --uplink-queue=droptail --uplink-queue-args="packets=Q" --downlink-queue=droptail --downlink-queue-args="packets=Q" -- sh -c '... client $MAHIMAHI_BASE 1 PORT'

That is the nesting and the flags above, so a run needs only our traces in
`~/sage/traces` and the link's arguments. After Sage's own setup (its
README: `build.sh`, `set_sysctl.sh`, `cp_models.sh`), for run R:

    cp kit/*.trace ~/sage/traces/
    cd ~/sage/sage_rl
    ./sage.sh 40 44545 a.trace a.trace R 1000 24 48 7 0 crosscheck 45 1 0 1 1 0    # link a
    ./sage.sh 20 44545 b.trace b.trace R 160 48 48 7 0 crosscheck 30 1 0 1 1 0     # link b
    ./sage.sh 40 44545 c.trace c.trace R 320 12 12 7 0 crosscheck 30 1 0 1 1 0     # link c, after the loss edit below
    ./sage.sh 80 44545 d.trace d.trace R 5120 96 96 7 0 crosscheck 30 1 0 1 1 0    # link d

The arguments, in `sage.sh`'s order: one-way delay (ms), port, downlink
trace, uplink trace, iteration (the run R, which only names the log),
buffer (packets), the capacity of the trace's first and second level in
Mbit/s, the level period (7 s), 0 for evaluation mode, the log prefix, the run's seconds, actor id, steps (0: run
by time), flows, 1 to save the downlink log, 0 to skip its plots. They are
`run_sample.sh`'s arguments with the link's values. The log lands at
`~/sage/sage_rl/log/down-sage-crosscheck-1-L.trace-L.trace-D-Q-1-BW-BW2-1-R`;
copy it to `logs/L.sage.R.log`. The two capacities and the period tell Sage
the current level's capacity, which it writes into column 1 of its state
message and uses in its reward (column 75); palaestra's port writes both
columns as 0 (`docs/architecture.md`, "The step").

Loss on link c: `sage.sh` fixes `loss="0"`, and `sage.cc` applies its one
rate to both directions, while palaestra's link c loses 1% of data packets
only. To match it, set `loss="0.01"` in `sage.sh` for link c's runs, and in
`sage.cc` pass `0.0` for the uplink rate: in both `sprintf(cmd, ...)` calls
under `#ifdef STANDALONE`, `mm_loss_rate` appears twice after `delay_ms`;
replace the second one, then rebuild with `bash build.sh`. The other links
are unchanged by the edit.

All of this is read from Sage's source at commit `eedd281`; none of it was
run here (UNVERIFIED).

## expliCCit's released policy

palaestra runs expliCCit's released net1 and actor
(`third_party/expliccit-models`) as deployed. On the original stack, run them
with expliCCit's own scripts, the way its case study ran, on the four links:
our trace in both directions, the one-way delay, the droptail queue in both
directions, the duration, and the downlink log saved with
`--downlink-log`. Data loss on link c only, as for Sage. If expliCCit's
scripts build their Mahimahi command as Sage's do, the same arguments
apply. Name the logs `L.expliccit.R.log`. This repository holds none of
expliCCit's code, so none of these steps is checked against it
(UNVERIFIED).

## Naming the logs and sending them back

Each run's log is mm-link's downlink log (`--downlink-log=FILE`); the uplink
log is not needed. Name each one `LINK.CONTROLLER.RUN.log`, with LINK `a`
to `d`, CONTROLLER one of `cubic`, `bbr`, `vegas`, `sage`, `expliccit`, and
RUN 1, 2 or 3, as in `a.cubic.1.log`. `compare.py` refuses other names
ending in `.log`. ccBench's patched Mahimahi also writes `FILE_init_timestamp`
next to each log; it is not needed.

Then record the host next to the logs and pack them:

    cd kit
    { uname -a; lsb_release -ds; iperf3 --version; git -C ~/ccBench/mahimahi rev-parse HEAD
      git -C ~/sage rev-parse HEAD; sha256sum ./*.trace
      sysctl net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.ipv4.tcp_no_metrics_save net.ipv4.tcp_autocorking net.core.rmem_max net.core.wmem_max
    } > logs/host.txt 2>&1
    tar -cf - logs | xz -9 > crosscheck-logs.tar.xz

and send us `crosscheck-logs.tar.xz`. Written from palaestra's run, a 30 s
log of link b with Cubic is about 5 MB, and 0.13 MB after xz. We store the file as a release with
`scripts/stash.sh` (each file under 100 MB) and cite that release next to
the comparison.

## Comparing

    gh release download exp-crosscheck-RUN --dir rel && mkdir release
    cat rel/crosscheck.tar.zst.part-* | zstd -dc | tar -x -C release
    tar -xJf crosscheck-logs.tar.xz        # their logs, in logs/
    uv run python tools/crosscheck/compare.py --mahimahi logs --palaestra release --out crosscheck.md

The table has one row per link and controller: the runs on each side that
gave metrics; for each metric, each side's mean over runs (with their range
where the runs differ) and the difference, Mahimahi minus palaestra; and
each side's data start. Without `--mahimahi` it gives palaestra's side alone,
which is what the experiment writes to its `summary.md`.

Both sides are computed the same way, because both become a Mahimahi log
first: the team's file through `parse_mm_log`, and palaestra's link records
through `mm_log_from_records`, the converter its own mm-th-delay metrics use.
On that log:

- t0, the data start, is the arrival of the first packet of at least 1000
  bytes at mm-link: the flow's first data segment. iperf3 opens a control
  connection first and Sage's client sends a request, so each side starts
  from its own flow's start.
- The window runs from t0 + 2 s to t0 + duration - 1 s.
- Utilization is the bytes that departed in the window over the bytes of
  the window's delivery opportunities (Mahimahi bytes on both).
- Queueing delay is, for each packet that departed in the window, its
  departure minus its arrival ms in the log: its time in the link queue,
  without mm-delay. The table gives the mean and the 95th percentile.
- A log that ends before its window, cannot be parsed, or has no data, and
  a palaestra episode that failed, give no metrics; the table says which
  and why.

On link a, the trace's phase at t0 differs between the sides: palaestra
starts its flow at trace time 0, while mm-link starts its trace before the
receiver connects. The data start columns show the offset. On constant
links the phase does not matter.

## Known differences from the original stack

- Kernel. palaestra runs Linux 7.0 in UML, in virtual time; the original
  runs 4.19 on hardware, in real time. Cubic, BBRv1 and Vegas are each
  kernel's own code, which changed between the two versions.
- Sender and receiver. palaestra's guest sets `TCP_NODELAY` on both sockets
  and writes from the moment the connection is up; iperf3 sets
  `TCP_NODELAY` only with `-N`, which matters little for a bulk sender of
  full segments.
- Send buffer. palaestra caps each episode's `tcp_wmem` maximum at what the
  path needs, at most the setups' 64 MB (`docs/architecture.md`). Of 340
  stock-controller episodes checked against 64 MB, it changed 3, all with
  Hybla, by at most 0.0003 in utilization (`docs/validation.md`).
- Sage and expliCCit. palaestra runs ONNX exports of their released models
  (Sage's in release `exp-sage-model-24`) through the guest's port of
  `sage.cc`'s step (`guest/src/step.rs`), and inference costs no virtual
  time. There, their own code runs them in TensorFlow or PyTorch, in real
  time.
- Random loss. palaestra's seeded kernel draws the losses from its own
  stream, so the lost packets differ from Mahimahi's; only the rate is the
  same.

## What was checked

Run on 2026-11-05 in this environment (not through GitHub Actions):

- `links.py write` wrote the four traces. Counting lines with awk gives 36
  Mbit/s for `a.trace` (24 and 48 Mbit/s over its two 7 s halves, period
  14000 ms), and 48, 12 and 96 Mbit/s for `b`, `c` and `d`. `b.trace`
  equals the first 28000 lines of ccBench's `wired48` at commit `49b5ada`,
  and `d.trace` equals `wired96` from ms 1 to 7000.
- `compare.py` on a synthetic log in mm-link's format (its header lines,
  opportunities, arrivals, departures with the fifth field ccBench's AQM
  patch adds, a drop line) built to have known metrics gave them exactly:
  utilization 0.500, mean queueing delay 7.00 ms, 95th percentile 25.00 ms,
  data start 500 ms. A log that ends before its window and a log with a
  malformed departure line gave no metrics, with their reasons.
- Self-check on one Cubic episode of link b in UML (30 s, seed 1). Its raw
  records, written as a Mahimahi log with the repository's converter
  (`mm_log_from_records`, `format_mm_log`), and its stored `link_events`
  give the same metrics bit for bit, both equal to a direct computation from
  the records. The written log also gives the episode's stored mm-th-delay
  metrics exactly.
- The experiment's `prepare`, `run` and `collect` commands ran end to end
  on 4 local cores: 60 episodes, all valid, in about 3 minutes, with 219 MB
  of `link_events`. On all 60, the bytes, packets and queueing delays
  `compare.py` rebuilds from `link_events` equal the episodes' own
  `flow_seconds` table over the same whole seconds. Nothing from this local
  run is a result; results come from the experiment's release.
- `tests/test_ci_experiment.py` passes with `experiments/crosscheck.json`.

UNVERIFIED, since no environment here runs the original stack: every
command for it above (the shells, iperf3's flags on Ubuntu 16.04, the
sysctl and modprobe lines, Sage's `sage.sh` arguments and the edit for link
c, read from the sources of Sage at `eedd281`, ccBench at `49b5ada` and
Mahimahi at `f1346c3`), and expliCCit's slot. The CI experiment itself is
UNRUN: it was not dispatched.
