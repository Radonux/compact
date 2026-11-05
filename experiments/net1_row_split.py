"""net1 on a random row split against a link split, on the expliCCit-like
data (examples/campaign-expliccit-teachers.json).

expliCCit's net1 trainer splits rows at random (train_test_split twice: 20%
for test, then 20% of the rest for validation) and reports R^2 above 0.9 on
the held-out rows. A row split puts each step on its own, so a test row has
training rows of its own episode 10 ms away; a link split holds whole links
out. Both runs here use expliCCit's settings (its inputs and LCFs, min-max
scaling, batch 256, up to 500 epochs with patience 20) and its fractions,
and differ only in the split. The gap is what a row split overstates about
paths net1 has not seen.

Usage: python experiments/net1_row_split.py OUT --data DIR --split row|link --seed S
       python experiments/net1_row_split.py OUT --report
The first writes OUT/SPLIT-sS/report.json (seed S picks both the split and
the initial weights); the second reads every one and writes OUT/summary.md.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from expliccit_env.train import data, fit
from expliccit_env.train.spec import Net1Spec

TEST, VAL = 0.2, 0.8 * 0.2  # expliCCit's two train_test_split calls
SPEC = Net1Spec(scaling="minmax", batch_size=256, epochs=500, patience=20)


def train(out: Path, data_dir: Path, split: data.SplitBy, seed: int) -> None:
    steps = data.load([data_dir], seed=seed, val=VAL, test=TEST, split_by=split)
    start = time.monotonic()
    net1, log = fit.fit_net1(SPEC, steps, seed)
    report: dict[str, Any] = {
        "split": split,
        "seed": seed,
        # Every cell regenerates the same campaign; collect checks it is the same.
        "data": {
            "episodes": len(steps.episode_ids),
            "rows": int(steps.obs.shape[0]),
            "obs_sha256": hashlib.sha256(steps.obs.tobytes()).hexdigest(),
        },
        "rows": {s: int(steps.rows(s).sum()) for s in ("train", "val", "test")},
        "train_s": time.monotonic() - start,
        "training": log,
        "test": fit.report_net1(net1, steps, "test"),
    }
    d = out / f"{split}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")


def _cell(v: list[float], digits: int) -> str:
    return f"{np.mean(v):.{digits}f} ({min(v):.{digits}f} to {max(v):.{digits}f})"


def summarize(out: Path) -> str:
    reps = [json.loads(p.read_text()) for p in sorted(out.glob("*-s*/report.json"))]
    datasets = {json.dumps(r["data"], sort_keys=True) for r in reps}
    if len(datasets) != 1:
        raise SystemExit(f"the cells saw different data: {sorted(datasets)}")
    by = {s: [r for r in reps if r["split"] == s] for s in ("row", "link")}
    if not all(by.values()):
        raise SystemExit(f"missing a split: {[s for s, v in by.items() if not v]}")
    d = reps[0]["data"]
    lines = [
        f"net1 with expliCCit's settings on the expliCCit-like data ({d['episodes']} episodes,"
        f" {d['rows']:,} steps), test rows of a random row split against test links of a"
        f" link split; {len(by['row'])} and {len(by['link'])} seeds, mean (range).",
        "",
        "| LCF | unit | R^2, row split | R^2, link split | MAE, row split | MAE, link split |",
        "|---|---|---|---|---|---|",
    ]
    for lcf, v in reps[0]["test"].items():
        r2 = {s: [r["test"][lcf]["net1"]["r2"] for r in by[s]] for s in by}
        mae = {s: [r["test"][lcf]["net1"]["mae"] for r in by[s]] for s in by}
        lines.append(
            f"| {lcf} | {v['unit']} | {_cell(r2['row'], 3)} | {_cell(r2['link'], 3)}"
            f" | {_cell(mae['row'], 3)} | {_cell(mae['link'], 3)} |"
        )
    lines += [
        "",
        "| split | seed | train / val / test rows | training (min) |",
        "|---|---|---|---|",
    ]
    for r in reps:
        n = r["rows"]
        lines.append(
            f"| {r['split']} | {r['seed']} | {n['train']:,} / {n['val']:,} / {n['test']:,}"
            f" | {r['train_s'] / 60:.1f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--split", choices=["row", "link"])
    p.add_argument("--seed", type=int)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        print(summarize(a.out), end="")
    elif a.data is None or a.split is None or a.seed is None:
        p.error("training needs --data, --split and --seed")
    else:
        train(a.out, a.data, a.split, a.seed)


if __name__ == "__main__":
    main()
