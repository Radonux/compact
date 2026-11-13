# compact

compact chooses the concepts of a concept-bottleneck congestion controller
from what a trained controller actually uses, instead of picking them from
a literature review.

## The question

expliCCit splits a learned congestion controller in two. net1, an
estimator, maps end-host measurements to a few latent congestion factors
(LCFs): physical path state such as capacity, queueing delay and base RTT,
with exact ground truth in emulation. net2, the decision maker, sets the
window from net1's estimates alone. Its LCF set was chosen by hand, and
nothing says which LCFs to use or how many.

compact asks whether the set can be derived:

1. **Discover.** Train a controller of the same shape with the middle
   left free: an encoder in net1's place maps measurements to k latents,
   and a decoder in net2's place sets the window from the latents alone.
   Its closed-loop performance against k says how many latents a
   controller needs, and the latents are what it uses.
2. **Ground.** Fit the latents to short closed-form formulas over emulator
   ground truth and basic sender metrics, and keep the formulas that are
   stable across seeds. A person names each one.
3. **Retrain.** Train a normal net1 and net2 on those formulas as the LCF
   labels. Only this controller ships, so every decision still passes
   through exact, checkable quantities.

The measure is retained performance: how much of an unconstrained
teacher's closed-loop performance the retrained controller keeps, against
other ways of choosing the same number of concepts. The claim is not that
a bottleneck makes a network explainable. The bottleneck counts the
concepts; grounding against ground truth says which they are.
[docs/methodology.md](docs/methodology.md) has the claims, the terms and
the plan; [docs/log.md](docs/log.md) records what has run and what it
showed.

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

It gives compact three things:

- **Ground truth.** Every arrival, drop and departure at the bottleneck is
  recorded, so the path's hidden state is exact at every instant. These
  labels supervise net1 and are the variables concepts are grounded in.
- **A closed loop.** A policy is an ONNX graph that sets the window of a
  real kernel TCP sender every 10 ms from the sender's own measurements,
  beside any of Linux's 17 controllers as competitors.
- **Step data.** Every step's measurements, labels and action are logged,
  for distillation and for supervised training.

[docs/architecture.md](docs/architecture.md) has the design and its
reasons.

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

    # training: net1 and an offline net2, then a net2 in the closed loop
    uv sync --extra train
    uv run expliccit train run train.json out/model
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

[docs/training.md](docs/training.md) covers data, training, running and
scoring policies.

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
outside code need a checkout of it and skip without one. What each check
showed, and what was never run: [docs/validation.md](docs/validation.md).

## Working conventions

Commits go directly to `main` in Conventional Commits form, without pull
requests; kernel changes are their own commits under `third_party/linux`.
Results come only from experiments in GitHub Actions, published as
releases of this repository. Agent instructions are in `AGENTS.md` (also
`CLAUDE.md`); agent skills come from the
[BTreeMap/SKILLs](https://github.com/BTreeMap/SKILLs) submodule at
`.github/skills`.

## Documents

| Document | Read it for |
|---|---|
| [docs/methodology.md](docs/methodology.md) | The idea, the claims, the terms, the evaluation protocol and the plan |
| [docs/log.md](docs/log.md) | What has run, what it showed, the state of each step and the open items |
| [docs/concepts.md](docs/concepts.md) | Candidate concepts from the literature: the baseline way of choosing them, names for fitted formulas, and the grounding vocabulary |
| [docs/training.md](docs/training.md) | Producing data, training net1 and net2, running and scoring a policy |
| [docs/architecture.md](docs/architecture.md) | How an episode runs: the link model, the labels, the policy contract, experiments, limits |
| [docs/validation.md](docs/validation.md) | What was checked about the emulator and the pipeline, and what was not |
| [kernel/modules/README.md](kernel/modules/README.md) | The kernel modules and their ABIs |
| [third_party/README.md](third_party/README.md) | Vendored sources and the kernel patches |

## Origin

compact began as a fork of
[Radonux/palaestra](https://github.com/Radonux/palaestra), which built the
emulator and the trainer; the history up to commit `4d576dfa7` is that
project's. compact is independent of it: its question, documents, models
and releases are its own, and the inputs it took from there are copied
into its own releases with their source recorded
(`experiments/phase0-teachers.json`, `experiments/phase0-steps.json`).
