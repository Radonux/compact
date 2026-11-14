# compact

compact derives the concepts of a concept-bottleneck congestion
controller from a trained controller. Today such concepts are picked by
hand.

## The question

expliCCit splits a learned congestion controller in two. net1, an
estimator, maps the sender's measurements to a few latent congestion
factors (LCFs): physical state of the path, such as capacity and queueing
delay, with an exact value in emulation. net2, the decision maker, sets
the window from net1's estimates alone. Its LCFs were chosen by hand, and
nothing says which to use or how many.

compact trains the same shape with the middle left free, then names what
the middle learned. A controller whose decision passes through learned
latents is trained in the closed loop. What it learned is grounded as
short closed-form formulas over the emulator's ground truth. A normal
net1 and net2 are then retrained on those formulas, so the controller
that ships passes every decision through exact, checkable quantities.
Discovery is data driven, and deployment stays by design.

The design, the claims and the plan are in
[docs/methodology.md](docs/methodology.md).

## The emulator

Everything runs in the emulator this repository holds (Python package
`expliccit-env`, command `expliccit`): the Linux 7.0 TCP stack in
User-Mode Linux (UML) in virtual time, on any x86_64 Linux host without
root, including GitHub Actions.

```text
  spec.json ─► parse ─► plan ─► boot UML ─┐   ┌─► validate records ─► labels, metrics, Parquet
                                          │   │
  ┌───────────────────────────────────────┬───┴─ UML kernel, Linux 7.0 ────┐
  │ guest runner (Rust): namespaces, flows, the step loop, the policy,     │
  │ the live truth, record drain                                           │
  │                                                                        │
  │  [sender] ──── veth ──── [path] ──── veth ──── [receiver]              │
  │  Linux controller        sch_mmlink, one       receiver,               │
  │  or tcp_agent + policy   each way: Mahimahi's  cross receivers         │
  │  cross TCP and UDP       link, delay, loss                             │
  │                                                                        │
  │  expl_tcpobs (per ACK) ──► expl_rec (one record ring) ◄── tcp_agent    │
  └────────────────────────────────────────────────────────────────────────┘
```

compact needs three things from it.

| | |
|---|---|
| Ground truth | Every arrival, drop and departure at the bottleneck is recorded, so the path's hidden state is exact at every instant. These labels supervise net1, and concepts are grounded in them. |
| A closed loop | A policy is an ONNX graph that sets the window of a real kernel TCP sender every 10 ms from the sender's own measurements, beside any of Linux's 17 controllers as competitors. |
| Step data | Every step's measurements, labels and action are logged. |

## Quick start

It needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
musl-tools, perl, rustup and uv. Clone with `--recurse-submodules`.

    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    uv sync

    # one episode, then a campaign: plan once, run shards anywhere
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4
    uv run expliccit campaign summary out/smoke

    # an experiment in GitHub Actions; its result becomes release exp-NAME-RUN
    gh workflow run experiment.yml -f experiment=experiments/smoke.json

    # training: an estimator, then a controller in the closed loop
    uv sync --extra train
    uv run expliccit train run train.json out/model
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

## Guarantees

A change that weakens one of these says so in its commit message. UML
checks need `EXPLICCIT_UML_DIR=build/uml`.

| Property | Check |
|---|---|
| Real kernel TCP: every Linux controller is the kernel's own code, and every kernel change is a separate commit | `git log --oneline e8eadca.. -- third_party/linux`; [third_party/README.md](third_party/README.md) |
| The bottleneck equals an independent port of Mahimahi's link queue, event for event, also under loss and cross traffic | `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"` |
| The same build, spec and seed give byte-identical records | `pytest tests/test_uml.py -k "bit_identical or loss_and_cross"` |
| A failed or invalid episode becomes a `failed` row with a reason, never metrics or labels | `pytest tests/test_records_abi.py tests/test_agent.py::test_bad_policy_output_fails_the_episode tests/test_score.py` |
| The truth a policy reads equals the labels stored with its step | `pytest tests/test_truth.py` |
| Every kind of net1 (single values, intervals, distributions) trains with every net2, exports a graph that computes what was trained, and runs in the guest | `pytest tests/test_e2e.py` |
| A spec names one-way delay or base RTT, never both | `pytest tests/test_spec.py` |
| Every episode row names its kernel tree, kernel config, repository commit and seed | columns `kernel_tree`, `config_sha256`, `repo_commit`, `seed` of `episodes.parquet` |

Everything at once: `uv sync --extra train`, then
`EXPLICCIT_UML_DIR=build/uml uv run pytest -q`. Some comparisons with
outside code need a checkout of it and skip without one.

## Documents

| Document | Read it for |
|---|---|
| [docs/methodology.md](docs/methodology.md) | The design, how it can fail, the claims, the rules of evidence, the plan and the terms |
| [docs/status.md](docs/status.md) | What has run, what it showed and what is open |
| [docs/architecture.md](docs/architecture.md) | How an episode runs: the link model, the labels, the policy contract, experiments, limits |
| [docs/training.md](docs/training.md) | The commands: step data, the estimator, the closed-loop trainer, scoring |
| [docs/validation.md](docs/validation.md) | What was checked about the emulator and the pipeline, and what was not |
| [docs/concepts.md](docs/concepts.md) | Concepts the literature defines: the baseline way of choosing them, names for fitted formulas, and the grounding vocabulary |
| [kernel/modules/README.md](kernel/modules/README.md) | The kernel modules and their ABIs |
| [third_party/README.md](third_party/README.md) | Vendored sources and the kernel patches |
| [AGENTS.md](AGENTS.md) | How to work in this repository: where each fact lives, engineering, git and the checks before a push |

## Origin

compact began as a fork of
[Radonux/palaestra](https://github.com/Radonux/palaestra), which built
the emulator and the trainer. The history up to commit `4d576dfa7` is
that project's. compact is independent of it: its question, documents,
models and releases are its own, and each input taken from there is
copied into a compact release by an experiment file that records the
source.
