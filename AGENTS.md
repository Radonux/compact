# Instructions for coding agents

Read `README.md` first, then `docs/methodology.md` (the design, the
claims, the plan) and `docs/status.md` (what has run). Emulator detail is
in `docs/architecture.md`.

## Research direction

Goal: derive the concepts (LCFs) of expliCCit's two-net controller from a
trained controller. Discovery is data driven. Deployment stays by design.

1. Discover. Train a controller whose decision passes through latents:
   an encoder in net1's place, a decoder in net2's place
   (`PPOSpec.latents`), in the closed loop. Bootstrap it from a
   traditional controller (Cubic, BBRv3), then continue by reinforcement
   learning so it is not tied to its teacher. The width of the latents
   counts nothing: one latent can always carry the decision itself.
2. Ground. Fit the wide controller's latents and its decision to short
   closed-form formulas over emulator ground truth and sender metrics.
   Keep those that recur across seeds, in the order the decision needs
   them.
3. Retrain. Train a normal net1 and net2 on the first m fitted formulas
   as LCF labels, through the existing pipeline. Only this controller
   ships. Closed-loop performance against m counts the concepts. The
   metric is the smallest m at which it is comparable with the
   unconstrained controller, against other ways of choosing concepts at
   the same budget.

Rules for this work:

- Never call one controller better than another because its training
  utility is higher. Comparable performance is enough: a mean log utility
  no more than 0.1 below the other's. ccperf is the last benchmark, for
  controllers that have trained well. It is deferred, and daily work
  does not run it.
- The discovery model is a controller, trained and judged in the closed
  loop. Fit to logged steps is never evidence about a controller. Do not
  replace the controller with an offline predictor.
- Before a run that changes what is trained or measured, say so to the
  owner in one sentence.
- The plan is a starting point. Reason from the problem about what would
  show whether the idea works. Reorder, add or drop steps when the
  evidence says so, and change `docs/methodology.md` to match.
- Check every discovery controller for action collapse: regress its
  decision on the latents, and keep widths 1 and 2 in every width curve.
- Never claim that a bottleneck makes a network explainable, or that its
  width counts concepts. Grounding against ground truth names the
  concepts, and retraining on them counts them. A closed-form concept is
  exact, not necessarily readable.
- A person chooses the grounding library, which is the standing objection
  to the method. Every grounding result comes with its vocabulary
  sensitivity check.
- A step is done when it has a release with code, configuration, seeds
  and raw outputs, and `docs/status.md` says what ran, what was searched,
  the held-out numbers and what remains UNRUN.
- compact is an independent project. Name every model, release and
  experiment for what it is in compact, never after the project it was
  forked from. An input taken from there is copied into a compact release
  by an experiment file whose description records the source.
  Experiments read inputs only from compact's releases.

## Documents

- One home per fact. The design, claims, plan and terms are in
  `docs/methodology.md`. Everything that changes with time (status,
  dates, results, open questions) is in `docs/status.md` and nowhere
  else.
- When something changes, rewrite the affected document from the ground
  up so it reads as one current account. Do not append dated entries, and
  do not describe earlier versions. Git holds the history.
- A direction that was dropped gets one short paragraph in
  `docs/status.md` and is removed everywhere else.
- Plain English, short sentences, no em dashes. Label anything not run as
  UNRUN.

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
- Never let a failed or invalid episode produce metrics or labels.
- Results are published as GitHub releases, with no exceptions. Run
  experiments through `.github/workflows/experiment.yml` (an experiment
  file under `experiments/`), which publishes every successful run, and
  cite the release in `docs/status.md`. Artifacts are transport only and
  expire.
- Make no claim about a learned model's performance before tuning it.
  Inspect its training diagnostics (critic and actor scales, saturation,
  actions outside the data), search the hyperparameters that matter on
  validation paths (brute-force grid where affordable, successive halving
  where not), confirm the winners over several seeds, then report once on
  the held-out test paths, saying what was searched.
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
