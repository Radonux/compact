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

## Engineering rules

- Licensing is never a blocker: this is an academic project, and it can
  be dual-licensed to fit upstream code or data. Decide what to reuse on
  technical grounds only, and record where it came from and its license.
- expliCCit's original code stays out of this repository and every public
  place until expliCCit is published: it is its co-authors' unpublished
  work. Our own rewrites of it may be public, and so may its released
  weights (`third_party/expliccit-models`).
- Follow the `pl-theorist` and `ponytail` skills: parse untrusted input
  once at its boundary, make invalid states unrepresentable, keep the
  core pure, and write the least code that works. Mark deliberate limits
  with `ponytail: <ceiling>, <upgrade path>`.
- Results are published as GitHub releases, with no exceptions. Run
  experiments through `.github/workflows/experiment.yml` (an experiment
  file under `experiments/`), which publishes every successful run, and
  cite the release in `docs/status.md`. Artifacts are transport only and
  expire.
- Guest code runs under UML's `time-travel=inf-cpu`: it must block or
  sleep, never spin, because spinning distorts virtual time.

## Git

- Push directly to `main`. Pull requests are not used. Force-push is
  disabled; never rewrite history.
- Write commit messages in Conventional Commits form (the `git-commit`
  skill in `.github/skills` has the rules).
- End every commit message with exactly this trailer, and no other
  `Co-Authored-By` line:

      Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>

- Never put session ids or links (for example `Claude-Session:`) in a
  commit message, and keep them out of pull request bodies too, since a
  squash merge copies the body into a commit.
- Change the kernel only as separate commits under `third_party/linux`,
  on top of the unmodified import
  (`git diff e8eadca -- third_party/linux`). Upstream kernel patches (for
  example Google's BBRv3 series) are imported with `git am`, keeping
  their original author and message; only the trailer above is added.
  Record each import in `third_party/README.md`.
- Do not edit `.github/workflows/sync-skills.yml` or the `.github/skills`
  submodule by hand: the file is the verbatim updater from
  BTreeMap/SKILLs, and the workflow bumps the submodule itself.
- `scripts/stash.sh TAG FILE...` pushes a `stash/TAG` branch;
  `.github/workflows/stash.yml` publishes release TAG from it and deletes
  the branch. Delete no stash branch whose release is incomplete. If one
  is left over, dispatch `.github/workflows/stash-sweep.yml`, which
  deletes each stash branch whose release holds all its files.

## Before pushing

Run what the change touches, and report only what actually ran:

    uv run ruff format --check src tests experiments && uv run ruff check src tests experiments && uv run pyright
    uv sync --extra train                                  # the trainer's tests need PyTorch
    uv run pytest -q                                       # unit tests
    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    EXPLICCIT_UML_DIR=build/uml uv run pytest -q            # adds the UML tests
    (cd guest && cargo fmt --check && cargo clippy --release --target x86_64-unknown-linux-musl -- -D warnings)
    scripts/check-kernel.sh                                # checkpatch on modules and kernel patches
    shellcheck scripts/*.sh tools/m0/*.sh
    uvx --from actionlint-py==1.7.12.25 actionlint && uvx zizmor==1.31.1 --offline .github/workflows
