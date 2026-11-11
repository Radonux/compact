# Instructions for coding agents

Read `README.md` first: it is the architecture and the guarantees a change
must keep. Implementation detail is in `docs/architecture.md`.

## Git

- Push directly to `main`. Pull requests are not used. Force-push is
  disabled; never rewrite history.
- Write commit messages in Conventional Commits form (the `git-commit` skill
  in `.claude/skills` has the rules).
- End every commit message with exactly this trailer, and no other
  `Co-Authored-By` line:

      Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>

- Never put session ids or links (for example `Claude-Session:`) in a commit
  message, and keep them out of pull request bodies too, since a squash merge
  copies the body into a commit.
- Change the kernel only as separate commits under `third_party/linux`, on
  top of the unmodified import (`git diff e8eadca -- third_party/linux`).
  Upstream kernel patches (for example Google's BBRv3 series) are imported
  with `git am`, keeping their original author and message; only the
  trailer above is added. Record each import in `third_party/README.md`.
- Do not edit `.github/workflows/sync-skills.yml` or the `.github/skills`
  submodule by hand: the file is the verbatim updater from BTreeMap/SKILLs,
  and the workflow bumps the submodule itself.
- `scripts/stash.sh TAG FILE...` pushes a `stash/TAG` branch;
  `.github/workflows/stash.yml` publishes release TAG from it and deletes
  the branch. Delete no stash branch whose release is incomplete. If one
  is left over, dispatch `.github/workflows/stash-sweep.yml`, which deletes
  each stash branch whose release holds all its files.

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

## Research direction

The goal is to choose the concepts (LCFs) of expliCCit's two-net
controller from what a trained controller actually uses, instead of
picking them from a literature review. Discovery is data driven;
deployment stays by design. `docs/methodology.md` holds the claims and
the plan. `docs/log.md` holds everything that changes with time: what
ran, what it showed, the state of each step and the open items. Keep
status, dates and results out of the README and the other documents.

- **Discover.** Train a model whose output passes through k latents (the
  codec, `experiments/codec.py`). The width at which its skill stops
  improving counts the concepts.
- **Ground.** Fit the latents to short closed-form formulas over emulator
  ground truth and sender metrics. Keep those stable across seeds.
- **Retrain.** Train a normal net1 and net2 on the fitted formulas as LCF
  labels, through the existing pipeline. Only this controller ships; the
  student is thrown away. The metric is retained performance against the
  unconstrained teacher, in closed loop, compared with other ways of
  choosing concepts at the same budget.

Rules for this work:

- The plan is a starting point. Reason from the problem about what would
  show whether the idea works; reorder, add or drop steps when the
  evidence says so, change the plan in `docs/methodology.md`, and record
  why in `docs/log.md`.
- Never claim that a bottleneck makes a network explainable. The claim is
  that the bottleneck counts the concepts and that grounding against
  ground truth names them. A closed-form concept is exact, not
  necessarily readable.
- A person still chooses the grounding library, which is the standing
  objection to the method. Every grounding result comes with its
  vocabulary sensitivity check.
- Report the codec's skill over persistence beside its R^2. Check every
  teacher's student for action collapse (z against the teacher's action).
- A phase is done when it has a release with code, configuration, seeds
  and raw outputs, and an entry in `docs/log.md` saying what ran, what was searched,
  the held-out numbers and what remains UNRUN.
- compact is an independent project. Name every model, release and
  experiment for what it is in compact, never after the project compact
  was forked from. An input taken from there is copied into a compact
  release by an experiment file whose description records the source;
  experiments read inputs only from compact's releases.

## Engineering rules

- Licensing is never a blocker: this is an academic project, and it can be
  dual-licensed to fit upstream code or data. Decide what to reuse on
  technical grounds only, and record where it came from and its license.
- expliCCit's original code stays out of this repository and every public
  place until expliCCit is published: it is its co-authors' unpublished
  work. Our own rewrites of it may be public, since the experiment is done
  in public, and so may its released weights (`third_party/expliccit-models`).
- Follow the `pl-theorist` and `ponytail` skills: parse untrusted input once at
  its boundary, make invalid states unrepresentable, keep the core pure, and
  write the least code that works. Mark deliberate limits with
  `ponytail: <ceiling>, <upgrade path>`.
- Never let a failed or invalid episode produce metrics or labels.
- Results are published as GitHub releases, with no exceptions: run
  experiments through `.github/workflows/experiment.yml` (an experiment
  file under `experiments/`), which publishes every successful run, and
  cite the release in the docs. Artifacts are transport only and expire.
- Make no claim about a learned model's performance before tuning it:
  inspect its training diagnostics (critic and actor scales, saturation,
  actions outside the data), search the hyperparameters that matter on
  validation paths (brute-force grid where affordable, successive halving
  where not), confirm the winners over several seeds, then report once on
  the held-out test paths, saying what was searched.
- Guest code runs under UML's `time-travel=inf-cpu`: it must block or sleep,
  never spin, because spinning distorts virtual time.
- Docs, comments and reports use plain English with no em dashes. Label
  anything not run as UNRUN.
