# compact

compact chooses the concepts of a concept-bottleneck congestion controller
from what a trained controller actually uses, instead of picking them from
a literature review.

It includes the emulator the work runs in (Python package `expliccit-env`,
command `expliccit`): the Linux 7.0 TCP stack in User-Mode Linux (UML) in
virtual time, through a bottleneck that behaves as Mahimahi's does, with
the true state of the path recorded at every instant. It runs on any
x86_64 Linux host without root, including GitHub Actions, where every
result is published as a release.

## The question

expliCCit splits a learned congestion controller in two. net1, an
estimator, maps end-host measurements to a few latent congestion factors
(LCFs): physical path state such as capacity, queueing delay and base RTT,
with exact ground truth in emulation. net2, the decision maker, sets the
window from net1's estimates alone. Its LCF set was chosen by hand, and
nothing says which LCFs to use or how many.

compact asks whether the set can be derived:

1. **Discover.** Train a controller whose decision passes through a narrow
   layer of k latents. The width at which performance stops improving
   estimates how many concepts the task needs.
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
[docs/methodology.md](docs/methodology.md) has the claims, the terms, the
plan and the status.

## Status

No result is a release yet. A local prototype on logged states (not
tuned, no closed loop, so a feasibility signal only) recovered a fixed
rule's one concept, log BDP, from a one-latent student, and levelled off
at three latents for two teachers once the student also had to predict
the next state. Both teachers already act through an estimator of seven
hand-picked LCFs, so that is recovery of a known bottleneck, not
discovery. The reproduction of it as a release, the black-box teacher,
the grounding and the retraining are all UNRUN
([status](docs/methodology.md#status)).

## How it works

```text
  spec.json ─► parse ─► plan ─► boot UML ─┐   ┌─► validate records ─► labels, metrics, Parquet
                                          │   │
  ┌───────────────────────────────────────┬───┴─ UML kernel, Linux 7.0 ────┐
  │ guest runner (Rust): namespaces, flows, Sage's step loop, the policy,  │
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

Each episode sends one bulk TCP flow, and optional cross traffic, through
a trace-driven bottleneck with a fixed delay, a droptail buffer and
optional random loss. Every arrival, drop and departure is recorded, so
the LCFs are exact at every instant. A policy is an ONNX graph that sets
the window every 10 ms step from Sage's state message, and can also read
the true LCFs (an oracle). [docs/architecture.md](docs/architecture.md)
has the design and its reasons.

## Quick start

It needs x86_64 Linux with gcc, make, flex, bison, bc, libelf-dev,
musl-tools, perl, rustup and uv.

    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    uv sync

    # one episode, then a campaign: plan once, run shards anywhere
    uv run expliccit episode examples/episode-mahak.json out/one --build build/uml
    uv run expliccit campaign plan examples/campaign-smoke.json out/smoke
    uv run expliccit campaign run out/smoke --build build/uml --workers 4
    uv run expliccit campaign summary out/smoke

    # an experiment in GitHub Actions; its result becomes release exp-NAME-RUN
    gh workflow run experiment.yml -f experiment=experiments/smoke.json

    # training (net1 and an offline net2; learned net2 in closed loop)
    uv sync --extra train
    uv run expliccit train run train.json out/model
    uv run expliccit train ppo ppo.json out/ppo --build build/uml --workers 4

[docs/training.md](docs/training.md) covers data, training, running and
scoring policies.

## Guarantees

A change that weakens one of these says so in its commit message. UML
checks need `EXPLICCIT_UML_DIR=build/uml`; the Mahak checks need a Mahak
checkout (`git clone https://github.com/Ppazhooh/mahak /tmp/mahak && git -C
/tmp/mahak checkout 5077583`) in `EXPLICCIT_MAHAK_DIR`.

| Property | Check |
|---|---|
| Real kernel TCP: every Linux controller is the kernel's own code, BBRv3 is Google's, and every kernel change is a separate commit | `git log --oneline e8eadca.. -- third_party/linux`; [third_party/README.md](third_party/README.md) |
| The bottleneck equals an independent port of Mahimahi's link queue, event for event, also under loss and cross traffic | `pytest tests/test_uml.py -k "mahimahi_oracle or loss_and_cross"` |
| Mahak's traces, metrics and active-learning driver equal Mahak's own code | `pytest tests/test_mahimahi.py tests/test_mahak.py` |
| The same build, spec and seed give byte-identical records | `pytest tests/test_uml.py -k "bit_identical or loss_and_cross"` |
| A failed or invalid episode becomes a `failed` row with a reason, never metrics or labels | `pytest tests/test_records_abi.py tests/test_agent.py::test_bad_policy_output_fails_the_episode tests/test_mahak.py::test_failed_queries_never_become_labels tests/test_score.py` |
| The truth a policy reads equals the labels stored with its step | `pytest tests/test_truth.py` |
| Every released policy (each `.onnx` asset of the `model-*` releases) runs on the current guest; rerun when the guest, its modules or the policy contract change | `experiments/onnx-compat.json` |
| Every kind of net1 (single values, intervals, distributions) trains with every net2 (offline, or taught then PPO; all beliefs or their means), exports a graph that computes what was trained, and runs in the guest | `pytest tests/test_e2e.py` |
| Mahak's "mRTT" is one-way; a spec names one-way delay or base RTT, never both | `pytest tests/test_spec.py` |
| Every episode row names its kernel tree, kernel config, repository commit and seed | columns `kernel_tree`, `config_sha256`, `repo_commit`, `seed` of `episodes.parquet` |

Everything at once: `uv sync --extra train`, then
`EXPLICCIT_UML_DIR=build/uml EXPLICCIT_MAHAK_DIR=/tmp/mahak uv run pytest -q`.
The two comparisons with Sage's `sage.cc` also need a Sage checkout and
g++ (`EXPLICCIT_SAGE_DIR`); CI skips them. What each check showed, and
what was never run: [docs/validation.md](docs/validation.md).

## Working conventions

Commits go directly to `main` in Conventional Commits form, without pull
requests, kernel changes as their own commits under `third_party/linux`. Results come only from
experiments in GitHub Actions, published as releases. Agent instructions
are in `AGENTS.md` (also `CLAUDE.md`); agent skills come from the
[BTreeMap/SKILLs](https://github.com/BTreeMap/SKILLs) submodule at
`.github/skills`, so clone with `--recurse-submodules`.

## Documents

| Document | Read it for |
|---|---|
| [docs/methodology.md](docs/methodology.md) | The idea, the claims, the terms, the evaluation protocol, the plan and the status |
| [docs/lcf-concepts.md](docs/lcf-concepts.md) | Candidate LCFs from the literature: the way of choosing concepts this project tests against, and the source of its hand-picked baseline |
| [docs/training.md](docs/training.md) | How to produce data, train net1 and net2, and run and score a policy |
| [docs/architecture.md](docs/architecture.md) | How an episode runs and why; known limits; decisions; extending |
| [docs/validation.md](docs/validation.md) | What was checked about the emulator, how, and what was not |
| [kernel/modules/README.md](kernel/modules/README.md) | The kernel modules and their ABIs |
| [third_party/README.md](third_party/README.md) | Vendored sources and our patches |
| [tools/crosscheck/README.md](tools/crosscheck/README.md) | A kit that compares the emulator with the stack Sage and expliCCit ran on |

## Origin

compact began as a fork of
[Radonux/palaestra](https://github.com/Radonux/palaestra), which built the
emulator and the trainer; the history up to commit `4d576dfa7` is that
project's. compact is independent of it: its question, documents, models
and releases are its own, and the few inputs it took from there are
copied into its own releases with their source recorded
(`experiments/phase0-teachers.json`, `experiments/phase0-steps.json`).
