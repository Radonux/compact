"""Fairness and efficiency of copies (experiments/copies.py): per controller
and flow count n, the mean over links of Jain's index of the flows' bytes
through the bottleneck, the link's utilization and its mean queueing delay,
with 95% bootstrap intervals over links. Appends a table to OUT/summary.md
(experiments/rule_report.py writes the log utility of the same episodes).

Usage: python experiments/copies_report.py OUT
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from expliccit_env.train.data import link_key


def interval(x: np.ndarray[Any, np.dtype[np.float64]]) -> str:
    rng = np.random.default_rng(0)
    boot = x[rng.integers(0, x.size, (4000, x.size))].mean(axis=1)
    return f"{x.mean():.3f} ({np.quantile(boot, 0.025):.3f} to {np.quantile(boot, 0.975):.3f})"


def main(out: Path) -> None:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    lines: list[str] = []
    for c in sorted(out.glob("c-copies-*")):
        rows: dict[tuple[str, int], dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        failed = 0
        for f in sorted((c / "shards").glob("*/episodes.parquet")):
            for e in pq.read_table(f).to_pylist():  # pyright: ignore[reportUnknownMemberType]
                if e["status"] != "ok":
                    failed += 1
                    continue
                x = np.asarray(e["flow_bytes"], dtype=np.float64)
                key = (aliases.get(e["controller"], e["controller"]), x.size)
                per = rows[key][link_key(e["spec_json"])]
                per["jain"].append(float(x.sum() ** 2 / (x.size * (x**2).sum())))
                per["utilization"].append(float(e["utilization"]))
                per["delay"].append(float(e["mean_delay_ms"]))
        lines += [
            "",
            f"## {c.name[2:]}: fairness and efficiency of n copies",
            "",
            "| controller | n | Jain's index | utilization | mean queueing delay (ms) |",
            "|---|---|---|---|---|",
        ]
        for (name, n), links in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            cols = {
                m: np.array([np.mean(v[m]) for v in links.values()])
                for m in ("jain", "utilization", "delay")
            }
            lines.append(
                f"| {name} | {n} | {interval(cols['jain'])} | {interval(cols['utilization'])}"
                f" | {interval(cols['delay'])} |"
            )
        lines.append(f"\nFailed episodes, not scored: {failed}.")
    with (out / "summary.md").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
