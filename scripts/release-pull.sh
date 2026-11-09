#!/usr/bin/env bash
# Extract only the members of a packed experiment release that match the
# given patterns, without storing the release: its NAME.tar.zst.part-*
# assets are streamed in order through zstd and tar. A training release can
# hold every iteration's snapshot (round 23's is 10 GB); a pattern's * does
# not cross /, so './ppo-*/policy.onnx' keeps only each run's final policy.
#   scripts/release-pull.sh TAG DEST PATTERN...
# Example:
#   scripts/release-pull.sh exp-net2-r23-probe-185 out/r23 \
#     './ppo-r23-*/policy.onnx' './ppo-r23-*/ppo_spec.json'
set -euo pipefail
if (($# < 3)); then
  echo "usage: $0 TAG DEST PATTERN..." >&2
  exit 2
fi
tag=$1 dest=$2
shift 2
repo=${GITHUB_REPOSITORY:-Radonux/palaestra}
mkdir -p "$dest"
mapfile -t ids < <(gh api "repos/$repo/releases/tags/$tag" \
  --jq '[.assets[] | select(.name | test("\\.tar\\.zst\\.part-"))] | sort_by(.name) | .[].id')
if ((${#ids[@]} == 0)); then
  echo "$tag: no packed parts" >&2
  exit 1
fi
for id in "${ids[@]}"; do
  gh api -H 'Accept: application/octet-stream' "repos/$repo/releases/assets/$id"
done | zstd -dcq | tar -x -C "$dest" --wildcards --no-wildcards-match-slash "$@"
