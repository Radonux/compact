#!/usr/bin/env bash
# Store local files as a GitHub release, for example trained weights.
#
# Usage: scripts/stash.sh TAG FILE... [NOTES.md]
#
# Pushes a branch stash/TAG whose one commit is origin/main plus the files
# under stash/ (made with git plumbing, so the working tree and index are
# untouched). .github/workflows/stash.yml then publishes the files as release
# TAG, tagged at origin/main's commit, and deletes the branch, so the files
# live only in the release. A file named NOTES.md becomes the release notes,
# and its first line (without a heading mark) the release title.
# Each file must be under 100 MB, git's limit on GitHub; experiments read the
# release back with "inputs": {"name": "release:TAG"}.
set -euo pipefail
TAG="${1:?usage: stash.sh TAG FILE...}"
shift
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "TAG must match ^[a-z0-9][a-z0-9._-]*$" >&2; exit 1; }
(( $# > 0 )) || { echo "no files" >&2; exit 1; }
git fetch --quiet origin main
index="$(mktemp)"
trap 'rm -f "$index"' EXIT
export GIT_INDEX_FILE="$index"
git read-tree origin/main
for f in "$@"; do
    [[ -f "$f" ]] || { echo "not a file: $f" >&2; exit 1; }
    git update-index --add --cacheinfo "100644,$(git hash-object -w "$f"),stash/$(basename "$f")"
done
tree="$(git write-tree)"
unset GIT_INDEX_FILE
msg="chore: Stash $# file(s) as release $TAG

Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>"
commit="$(git commit-tree "$tree" -p origin/main -m "$msg")"
git push --quiet origin "$commit:refs/heads/stash/$TAG"
echo "pushed stash/$TAG; the stash workflow publishes release $TAG"
