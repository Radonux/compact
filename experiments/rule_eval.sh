#!/usr/bin/env bash
# Write and plan the held-out campaigns for comparing fixed rules:
#   experiments/rule_eval.sh OUT SPLIT SEED_TRAINING SEED_OTHER STOCK [--policy NAME=PATH ...]
# The training conditions at 20 s and 60 s (campaign seed SEED_TRAINING), and
# the families no training campaign of net1 v3 had (experiments/heldout.py):
# new and deep-long at 20 s, cubic-first at 60 s (seed SEED_OTHER). Each is
# planned into OUT/c-FAMILY-LENGTH; OUT/aliases.json maps policy hashes to
# names, for experiments/rule_report.py. RULE_EVAL_FAMILIES (for example
# "deep-long:20") plans only those families; each family's episodes do not
# depend on which others are planned.
set -euo pipefail
out=$1 split=$2 seed_training=$3 seed_other=$4 stock=$5
shift 5
read -ra families <<< "${RULE_EVAL_FAMILIES:-training:20 training:60 new:20 deep-long:20 cubic-first:60}"
for fd in "${families[@]}"; do
  f=${fd%:*} d=${fd#*:} seed=$seed_other
  [[ $f == training ]] && seed=$seed_training
  uv run python experiments/heldout.py "$out/spec-$f-$d" --split "$split" --family "$f" \
    --duration-ms "${d}000" --seed "$seed" --stock "$stock" "$@"
  uv run expliccit campaign plan "$out/spec-$f-$d/campaign.json" "$out/c-$f-$d"
done
first=${families[0]}
cp "$out/spec-${first%:*}-${first#*:}/aliases.json" "$out/aliases.json"
