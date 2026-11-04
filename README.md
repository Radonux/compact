# palaestra: the expliCCit experiment environment

palaestra (Python package `expliccit-env`, command `expliccit`) runs the
Linux 7.0 TCP stack in User-Mode Linux (UML), in virtual time. Each episode
sends one bulk TCP flow, and optional cross traffic, through an emulated
bottleneck with Mahimahi's semantics: a capacity trace, a one-way delay, a
droptail buffer and optional random loss. The kernel records every arrival,
drop and departure at the bottleneck, so the true state of the path is known
at every instant.

It exists to test expliCCit's idea: split a learned congestion controller
into an estimator, net1, and a decision maker, net2. net1 turns sender-side
measurements into beliefs about a few physical properties of the path, the
latent congestion factors (LCFs). net2 sets the congestion window from those
beliefs alone. Testing this needs real kernel TCP, the true value of every
LCF at every decision, and many episodes. `docs/methodology.md` states the
claims, the plan and where the work stands.

## What it produces

- Labeled data for net1: per-ACK sender observations and, every 10 ms,
  Sage's state message (what expliCCit's networks read), each paired with
  the exact LCFs at that instant. The seven LCFs are capacity, queue
  length, mean queueing delay, free buffer, busy fraction (the share of
  capacity used), one-way delay and the flow's receive rate.
- Closed-loop runs of learned policies. An ONNX graph sets the congestion
  window at every step. It can also read the true LCFs (an oracle), or have
  some of net1's beliefs replaced by them (a patch).
- A trainer for net1 and net2 (cloning, offline RL, and online RL in closed
  loop). It also imports expliCCit's released models.
- Mahak-style performance maps (utilization, queueing delay, power) for any
  of the kernel's congestion controls: Mahak's active-learning driver,
  ported, with UML episodes as its oracle.

## Architecture

```text
                        HOST (Python, src/expliccit_env)
  spec.json ──► parse ──► plan ──► launch ─────────────┐        ┌──► validate records
  campaign.json  (specs)  (guest   (uml.py: boots      │        │    ──► labels + metrics
                           plan)     UML, seeded)      │        │    ──► Parquet shard
                                                       ▼        │
  ┌───────────────────────── ONE UML KERNEL (Linux 7.0) ────────┴─────────────┐
  │  guest runner (Rust, static, guest/): builds the topology, runs traffic,  │
  │  runs Sage's step and the policy, computes the true LCFs, drains records  │
  │                                                                           │
  │   [snd netns]  ── veth ──  [path netns]  ── veth ──  [rcv netns]          │
  │   TCP sender               sch_mmlink x2             TCP receiver         │
  │   (cubic | bbr | bbr3 |    (Mahimahi loss, link      cross receivers      │
  │    tcp_agent | ...)         and delay, each way)                          │
  │   cross TCP and UDP                                                       │
  │   senders                                                                 │
  │        │                         │                                        │
  │   expl_tcpobs ─────────────► expl_rec ◄────── tcp_agent actions           │
  │   (per-ACK state)           (record ring, /dev/expl_rec)                  │
  └───────────────────────────────────────────────────────────────────────────┘
```

| Part | Where | Responsibility |
|---|---|---|
| Specs and campaigns | `src/expliccit_env/spec.py`, `campaign.py` | What to run: path (trace, delay, buffer, random loss), congestion control, cross traffic, duration, seed. A campaign is a grid or sample of Mahak's space, or any list of paths, each run under a list of loss and cross-traffic conditions. |
| Launcher | `uml.py` | Boots a UML guest in virtual time (`time-travel=inf-cpu`), seeded, under a wall-clock timeout. |
| Guest runner | `guest/` (Rust) | Inside the kernel: namespaces, qdiscs, the bulk TCP pair, cross-traffic flows, Sage's step loop and the ONNX policy, the live truth, record collection. |
| Policy contract | `policy.py`, `guest/src/step.rs` | `sage-v1`. In: Sage's 77-value state message (a port of `sage.cc`) plus expliCCit's min RTT column, 78 values; an optional recurrent state; optionally the path's true LCFs (`guest/src/truth.rs`), for oracle and patched policies; optionally the sender's window. Out: a, the log2 of the window multiplier. The graph selects and scales its own inputs. |
| Step data | `steps.py` | Each step with the labels at its instant, Sage's reward from the exact capacity, and the action taken: the rows net1 and net2 train on. |
| Trainer | `train/` (optional `train` extra, PyTorch) | net1 supervised (point estimates, intervals, or a distribution per LCF from an ensemble); net2 by cloning, TD3+BC, IQL or expliCCit's objective, offline or online in closed loop, as a window multiplier or a window target relative to the believed bandwidth-delay product; one ONNX policy graph out. Imports expliCCit's released models from a local copy. |
| Link model | `kernel/modules/sch_mmlink.c` | The bottleneck: Mahimahi's `mm-link` and `mm-delay` (and `mm-loss`), as a qdisc. |
| Observation | `kernel/modules/expl_tcpobs.c` | Sender state at every ACK, for any congestion control. |
| Agent controllers | `kernel/modules/tcp_agent.c` | `agent` (Sage-style) and `agent_reno` (Orca-style), each socket driven by its own policy, any number per bottleneck. |
| Records | `kernel/modules/expl_rec.c`, `expliccit_rec.h` | One ring, one sequence of events: the only kernel-to-host data path. |
| Decode, labels, metrics | `records.py`, `labels.py`, `mahimahi.py`, `episode.py` | Validate records, derive ground truth, compute Mahak's metrics. |
| Mahak driver | `mahak.py` | Mahak's Gaussian-process active learning over its grid, querying UML episodes. |
| Datasets | `dataset.py` | Parquet shards joined on `episode_id`. |
| Kernel and Mahimahi | `third_party/linux`, `third_party/mahimahi` | Vendored sources; our changes are separate commits. |

### Where it runs

Everything runs in User-Mode Linux 7.0 with `time-travel=inf-cpu`. The
kernel is an ordinary unprivileged process, and time is virtual: CPU work
costs no time, so a guest runs 8x to 53x faster than real time. For a given
build, a seed fixes every random choice. It needs x86_64 Linux but neither
root nor KVM, so it runs on laptops and in GitHub Actions. Running it on a
Slurm cluster is UNRUN.

## Running it

It needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
musl-tools, perl, rustup and uv; no root.

    scripts/build-uml.sh build/uml          # UML 7.0 from third_party/linux
    scripts/build-guest.sh build/uml        # modules, runner, guest root
    uv sync

    # one episode
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml

    # a campaign: plan once, run shards anywhere
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4
    uv run expliccit campaign summary out/smoke   # per-controller means, 95% intervals over paths

    # a Slurm array job (one shard per task, one guest per core); never submitted (UNRUN)
    uv run expliccit campaign slurm out/smoke --build build/uml --shards 200 --cpus-per-task 32

    # an experiment in GitHub Actions: up to 256 cells, 20 at a time, 6 h each;
    # its result is published as a release (docs/architecture.md)
    gh workflow run experiment.yml -f experiment=experiments/smoke.json

    # score runs from ground truth: log utility per controller, link and
    # condition, paired with a reference by link
    uv run expliccit score out/smoke --reference cubic --markdown out/smoke/summary.md

    # store local files (weights) as a release that experiments can read
    scripts/stash.sh model-NAME weights.pt NOTES.md

    # Mahak's oracle_query(X, metric, CC), in UML
    uv run expliccit mahak-query --build build/uml --cc bbr --metric first_power \
        --buffer 8000 --mrtt 4 --bw 70 --step 0.2

    # Mahak's whole run (run_mahak.sh's arguments work as written): the map
    # as Mahak_cubic.csv, every query, and a dataset of the queried episodes
    uv run expliccit mahak --build build/uml --out out/map -cc cubic -budget 20 \
        -metric first_power -buffer 2,200,4 -mRTT 5,50,5 -bw 2,20,5 -change 0.1,2,0.1

    # train net1 and net2 offline, then net2 online (specs: docs/training.md)
    uv sync --extra train
    uv run expliccit train run train.json out/model
    uv run expliccit train online online.json out/online --build build/uml --workers 4

`docs/training.md` covers the whole path from teacher campaigns to a
policy: the training specs, running a policy, oracle and patched policies,
and importing expliCCit's released models.

Each dataset shard holds `episodes.parquet` (one row per episode:
parameters, status, metrics, provenance), `obs.parquet` (sender
observations), `labels.parquet` (ground truth, row for row with `obs`),
`link_events.parquet` and `link_opp.parquet` (the full bottleneck log),
`agent_action.parquet`, and, for episodes that record Sage's steps or run a
policy, `steps.parquet` (one row per step: the message, the labels at that
instant, Sage's reward, the action). `campaign run --tables` keeps a
subset.

## Guarantees, and how to check them

The system is built around these properties. Each has a check you can run;
a change that weakens one should say so in its commit message. The UML tests
need a build (above) and `EXPLICCIT_UML_DIR`; the Mahak comparisons need a
Mahak checkout in `EXPLICCIT_MAHAK_DIR`.

1. **Real kernel TCP, unmodified congestion control.** Cubic, BBRv1, Reno
   and the other controllers Linux 7.0 ships are the kernel's own code;
   BBRv3 is Google's. Every change to the kernel is a separate commit on top of the unmodified import: today HZ set
   to 1000, seeded randomness, and Google's BBRv3 with its core
   prerequisites (`third_party/README.md`). Adding BBRv3 left seeded Cubic
   and BBRv1 episodes byte-identical (`docs/validation.md`). List the
   changes with `git diff e8eadca -- third_party/linux`.
2. **The link model is Mahimahi's.** The kernel's arrivals, drops,
   departures and delivery opportunities equal those of an independent
   Python port of Mahimahi's `LinkQueue`, event for event, also under random
   loss and cross traffic:
   `EXPLICCIT_UML_DIR=build/uml uv run pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"`.
3. **Mahak's numbers are Mahak's.** Trace generation matches Mahak's
   `helper.py` bit for bit, metrics match its Perl `mm-th-delay` exactly, and
   the driver makes the same queries and the same map as Mahak's own
   `mahak_online.py` given the same oracle:
   `EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest tests/test_mahimahi.py tests/test_mahak.py`.
4. **Episodes are reproducible.** The same build, spec and seed give
   byte-identical records, random loss and cross traffic included:
   `EXPLICCIT_UML_DIR=build/uml uv run pytest tests/test_uml.py -k "bit_identical or loss_and_cross"`.
   Across guest versions they need not be (`docs/validation.md`, M10).
5. **No silent data.** An episode that times out, overflows a buffer it must
   not, drops records, or fails validation becomes a row with
   `status = "failed"` and a reason, never metrics or labels. Checked in
   part by `tests/test_records_abi.py` (sequence gaps, truncated files),
   `tests/test_agent.py::test_bad_policy_output_fails_the_episode` and
   `tests/test_mahak.py::test_failed_queries_never_become_labels`.
6. **Provenance.** Every row of `episodes.parquet` names the kernel source
   tree hash (`kernel_tree`), the kernel config hash (`config_sha256`), the
   repository commit (`repo_commit`) and the seed.
7. **The oracle sees the labels.** The true state a policy can read
   (`truth`) equals the labels its step is stored with, value for value:
   `EXPLICCIT_UML_DIR=build/uml uv run pytest tests/test_truth.py`.
8. **The mRTT trap is explicit.** Mahak's "mRTT" is a one-way delay (the
   base RTT is twice it). A spec says `{"mahak_compat_mrtt_ms": 20}` or
   `{"base_rtt_ms": 40}` (the same path) and cannot say both:
   `uv run pytest tests/test_spec.py`.

Run every check at once:

    uv sync --extra train
    git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C /tmp/mahak checkout 5077583
    EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q

The suite has 105 tests. All pass except 2 that compare our port of Sage's
step with `sage.cc`'s own code; they need a Sage checkout and g++ and are
UNRUN. CI runs the Python and lint checks on every commit that changes
more than docs (`.github/workflows/ci.yml`), the guest and UML checks when
the code they check changes (`uml.yml`), and the newest-mainline tripwire
on demand.

## Status

### The environment

| Area | State |
|---|---|
| UML build, link model, observations, records, datasets, campaigns, CLI | done, tested |
| Random loss and cross traffic (TCP of any stock controller, constant-rate UDP), with their labels | done, tested |
| BBRv3 (Google's, as `bbr3`) | done, tested |
| Other Linux 7.0 controllers as teachers (BIC, CDG, HighSpeed, H-TCP, Hybla, Illinois, LP, NV, Scalable, Vegas, Veno, Westwood, YeAH), built as modules | done; Vegas, Veno, YeAH, LP and Westwood each ran an episode |
| Agent controllers with ONNX policies, closed loop (`sage-v1`) | done, tested |
| Sage's step and step training data (labels, reward, action per step) | done, tested; equal to `sage.cc`'s own output at every step compared (run locally, as CI has no Sage checkout) |
| The true LCFs as a policy input (oracle and patched policies) | done, tested: equal to the labels at every step |
| Mahak metrics, trace generation and active-learning driver | done, matched to the originals |
| Trainer: net1, net2, export, expliCCit import; online training in closed loop | runs end to end, tested |
| Slurm array scripts | generated and checked locally; never submitted to a cluster (UNRUN) |

Measured speed, on 2026-10-30 with stock controllers, before BBRv3 and the
step loop existed: 8x to 53x real time per core, and about 5400
twenty-second episodes per core-hour on Mahak's ICNP'25 space, with about
2.5 MB of Parquet per episode. Details: `docs/validation.md`.

### Open items and limits

| Item | State |
|---|---|
| Reference runs of real Mahimahi against the link model | UNRUN: needs a root host and a port of Mahak's `highbdp.patch` |
| UML's virtual time against a real-time kernel | will not run: the native tier (Linux 7.0 on Ubuntu 26.04 hosts, as root, in real time) was dropped. No environment we have can easily boot a patched kernel, so it could never run BBRv3 or later kernel changes (`docs/architecture.md`, D19) |
| Inference latency | not modelled: a policy's inference costs no virtual time; to be measured outside and added as a delay |
| Labels recomputed from stored shards | not yet: `link_events` and `link_opp` hold what a new label needs, but nothing reads them back, so a new label means rerunning the episodes |
| Several learned flows on one bottleneck (any mix of policies, stock controllers and UDP; up to 64 cross flows) | done, tested |
| The full true state of the path beyond the seven LCFs, as a policy input (arm A0) | not yet |
| AQM or ECN, delay jitter, bursty loss, multiple bottlenecks | not supported |

### The research

`docs/methodology.md`, section 6, is the current status. A summary:

**Reproduction.** expliCCit reports R^2 above 0.9 for every LCF; on
held-out paths our net1 reaches 0.95 to 0.99 for 5 of the 7 (busy fraction
0.45, free buffer 0.09). expliCCit reports that its split costs about 5%
utilization and 9% queueing delay against Sage; here, with Sage's released
policy outside its own kernel and links, the fixed rule on net1 v4 has 3 to
15% more utilization and 46 to 55% less queueing delay on clean test links.
expliCCit's case study, which traces under-utilization to a capacity
belief 25% too low, did not reproduce: here the released policy's capacity
belief is about right.

**Results so far.** net1 states a probability distribution for each
LCF. On held-out paths it is at least as accurate as a single-value or
interval net1 on six of the seven LCFs. On links it has never seen, with
three controllers held out of training, an ensemble of three trained on
1,230 sampled links states each LCF honestly or cautiously: its 90%
statements hold 88.8% to 97.1% of the time, on busy and idle links alike,
and 80% to 97% under the held-out controllers. Capacity is stated to
within 12 Mbit/s (90%) on a busy link and 41 Mbit/s on an idle one; free
buffer is the least predictable LCF (R^2 0.56). Breadth of links mattered
more than any setting: on 132 training links, net1 recognized links and
was over-confident about capacity (0.71 at 90%).

net2 acts on that fixed net1 by setting the window to 2^u times the
believed bandwidth-delay product. Scored once on 27 held-out links in log
utility (log of the throughput share, minus delay and loss penalties):

- **Fixed multiple.** A fixed multiple (u = 0.25) beats every kernel
  controller on average: it scores -0.39, against -0.47 for BBRv3, the
  best of them, and -1.21 for Cubic. The average is over three equally
  weighted conditions: it beats all 17 controllers on clean links and
  under 1% random loss, but trails BBRv3, Illinois, Scalable and BIC
  under cross traffic, where a fixed window takes a small share against a
  growing Cubic flow. With 60 s episodes instead of 20 s, that loss grows
  and the fixed multiple ties BBRv3 and BBRv1 on average.
- **Unseen competition.** With a Cubic flow present from the start, a
  case no training campaign had, net1 believes the link is about 0.59 of
  its capacity, and its 90% capacity statements hold 26% of the time.
  Its honesty holds only within the conditions it was trained on.
- **True BDP and a classical estimate.** The same rule on the true BDP
  scores -0.33. That 0.065 difference (0.041 on a second held-out set of
  158 links) is what estimation costs this rule. On a classical estimate
  (largest delivery rate times min RTT, as BBR builds it) the rule scores
  -0.55, so net1's belief is worth 0.155 (0.228 on the 158 links).
- **Learned online.** A net2 trained online scores -0.35 over three
  seeds by competing harder with Cubic cross traffic. That is only 0.04
  better than the fixed multiple here, 0.01 on the 158 links, and no
  better on the validation links, also after 60 iterations instead of 20.
- **Means against distributions.** On the validation links, a net2 that
  sees only the means of net1's distributions (arm A2) does at least as
  well as one that sees the distributions (A3). So far net2 gets no
  measurable value from net1's stated uncertainty. Neither went to test.
- **Earlier design and expliCCit.** The earlier design scores -1.00 on
  these links: a multiplicative action on a single-value net1, tuned
  over 262 runs (`docs/training.md`, "Tuned comparison"). expliCCit's
  released policy scores -1.83.

These net2 results, and net1's on unseen links, use one family of links
(constant or stepping capacity, 8 to 100 Mbit/s) and three conditions
(clean, UDP plus a Cubic flow, 1% random loss). Test links are new
combinations of these, but the conditions are not held out, and the
ranking against the kernel controllers holds for this mix only. Under
conditions outside it (`docs/training.md`, "New conditions"), net1 stays
honest about random loss and new paths but not about competition it has
not seen.

## Working conventions

- **Commits** go directly to `main`, in Conventional Commits form, each
  ending with the single trailer
  `Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>`.
  No session ids in commit messages. Force-push is disabled; history is
  never rewritten.
- **Kernel changes** are their own commits under `third_party/linux`;
  `scripts/check-kernel.sh` style-checks them and our modules.
- **Agent skills** come from [BTreeMap/SKILLs](https://github.com/BTreeMap/SKILLs),
  a submodule at `.github/skills` aliased as `.claude/skills` and
  `.agents/skills`. `.github/workflows/sync-skills.yml` (installed verbatim
  from that repository) fast-forwards it twice a day. Clone with
  `git clone --recurse-submodules`.
- Instructions for coding agents: `AGENTS.md` (also `CLAUDE.md`).

## Where the details are

| Document | Contents |
|---|---|
| `docs/methodology.md` | The research goal and plan: claims H1 to H6, why net1 reports beliefs, training net1 and net2, interpretability, experiment arms A0 to A4, and where things stand |
| `docs/training.md` | From teacher campaigns to a policy: train, run and evaluate, oracle and patched policies, expliCCit's released models, online training, results so far |
| `docs/architecture.md` | How one episode runs and why: the emulated path and its bottleneck, virtual time, records and the ground truth computed from them, the policy contract, known limits, decisions D1 to D23, how to extend |
| `docs/validation.md` | Every check that ran, with commands and results, and what has not run |
| `docs/m0-report.md` | The dated feasibility report (M0, 2026-10-30): UML build, virtual time, speed, memory, determinism, BPF |
| `kernel/modules/README.md` | The kernel modules and their ABIs |
| `third_party/README.md` | Vendored sources and our patches on them |
