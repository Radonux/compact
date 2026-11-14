# Instructions for coding agents

compact derives the concepts of a concept-bottleneck congestion
controller from a trained controller. Discovery is data driven, and
deployment stays by design. `README.md` gives the vision. Read it first,
then `docs/methodology.md` and `docs/status.md`.

## Where each thing lives

Each fact has one home, so a change to one fact is a change to one file.
Every other file links to the home and does not restate it.

| File | Holds |
|---|---|
| `README.md` | The vision: the question, the idea, the emulator and what it guarantees |
| `docs/methodology.md` | The research design in full: the design, the claims, the rules of evidence, how controllers are compared, the plan and the terms |
| `docs/status.md` | Everything that changes with time: what has run, results, releases, open questions and dropped directions |
| `docs/architecture.md` | How an episode runs |
| `docs/training.md` | The commands |
| `docs/validation.md` | What was checked about the emulator |
| `docs/concepts.md` | Concepts from the literature and the grounding vocabulary |
| This file | How to work in this repository |

- When something changes, rewrite the affected document so it reads as
  one current account. Git holds the history, so documents carry no dated
  entries and no earlier versions.
- A direction that was dropped gets one short paragraph in
  `docs/status.md`.
- Plain English, short sentences, no em dashes. Label anything not run as
  UNRUN.

## How to work

- Reason from the problem about what would show whether the idea works.
  The plan is a starting point: follow the evidence, and change
  `docs/methodology.md` to match.
- Before a run that changes what is trained or measured, say so to the
  owner in one sentence.
- A step is done when it has a release with code, configuration, seeds
  and raw outputs, and `docs/status.md` says what ran, what was searched,
  the held-out numbers and what remains UNRUN.
- compact is an independent project. Name every model, release and
  experiment for what it is in compact. An input taken from the project
  it was forked from is copied into a compact release by an experiment
  file whose description records the source, and experiments read inputs
  from compact's releases.

## Engineering

- Choose what to reuse on technical grounds. This is an academic project
  and can be dual-licensed to fit upstream code or data. Record each
  source and its license in `third_party/README.md`.
- expliCCit's original code stays out of this repository and every public
  place until expliCCit is published: it is its co-authors' unpublished
  work. Our own rewrites of it may be public, and so may its released
  weights (`third_party/expliccit-models`).
- Write the least code that works, with the discipline of the
  `pl-theorist` and `ponytail` skills. Mark a deliberate limit with
  `ponytail: <ceiling>, <upgrade path>`.
- Results are GitHub releases, made by experiments
  ([docs/architecture.md](docs/architecture.md#experiments-in-github-actions))
  and cited in `docs/status.md`.
- Before a push, run the checks the change touches
  ([docs/validation.md](docs/validation.md#ci)) and report only what
  actually ran.

## Git

- Push directly to `main`. History only grows: pull requests and
  force-pushes are not used.
- Write commit messages in Conventional Commits form (the `git-commit`
  skill in `.github/skills` has the rules), free of session ids and
  links.
- End every commit message with exactly this trailer, and no other
  `Co-Authored-By` line:

      Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>

- Kernel changes are separate commits under `third_party/linux`, on top
  of the unmodified import. `third_party/README.md` records each one and
  how upstream patches are imported.
- `.github/workflows/sync-skills.yml` and the `.github/skills` submodule
  are kept by their own workflow, from BTreeMap/SKILLs.
