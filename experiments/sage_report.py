"""Utilization and queueing delay against Sage, as expliCCit compares its
split with Sage: on the conditions where the observed flow is alone on the
link (clean, random loss, ACK loss), per campaign, condition and
controller, the mean over links of the link's utilization and of its mean
queueing delay (what mm-th-delay prints), and the ratio of each mean to
Sage's on the same links, minus 1, with 95% bootstrap intervals over
links. Appends a table to OUT/summary.md.

Usage: python experiments/sage_report.py OUT [--reference sage]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from expliccit_env.train.data import link_key
from expliccit_env.train.score import condition

Vec = np.ndarray[tuple[int], np.dtype[np.float64]]


def ratio(x: Vec, ref: Vec) -> str:
    """mean(x) / mean(ref) - 1, with a bootstrap interval over links."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, x.size, (4000, x.size))
    boot = x[idx].mean(axis=1) / ref[idx].mean(axis=1) - 1
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return f"{x.mean() / ref.mean() - 1:+.1%} ({lo:+.1%} to {hi:+.1%})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--reference", default="sage")
    a = p.parse_args()
    aliases: dict[str, str] = json.loads((a.out / "aliases.json").read_text())
    lines = [
        "",
        f"## Utilization and queueing delay against {a.reference}, the flow alone",
        "",
        "Means over links, per condition without cross traffic; the ratio to"
        f" {a.reference}'s mean on the same links, minus 1.",
        "",
        f"| campaign | condition | controller | links | utilization | queueing delay (ms) |"
        f" utilization vs {a.reference} | delay vs {a.reference} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for camp in sorted(a.out.glob("c-*")):
        # condition -> controller -> link -> (utilization, delay) per episode
        per: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for f in sorted(camp.glob("shards/*/episodes.parquet")):
            cols = ["controller", "spec_json", "status", "utilization", "mean_delay_ms"]
            for r in pq.read_table(f, columns=cols).to_pylist():
                if r["status"] != "ok" or json.loads(r["spec_json"]).get("cross"):
                    continue
                name = aliases.get(r["controller"], r["controller"])
                cell = per[condition(r["spec_json"])][name][link_key(r["spec_json"])]
                cell.append((r["utilization"], r["mean_delay_ms"]))
        for cond in sorted(per):
            ref = per[cond].get(a.reference)
            if ref is None:
                continue
            for name in sorted(per[cond], key=lambda n: (n != a.reference, n)):
                links = sorted(set(per[cond][name]) & set(ref))
                if not links:
                    continue
                mine = np.array([np.mean(per[cond][name][k], axis=0) for k in links])
                theirs = np.array([np.mean(ref[k], axis=0) for k in links])
                vs = (
                    ("", "")
                    if name == a.reference
                    else (ratio(mine[:, 0], theirs[:, 0]), ratio(mine[:, 1], theirs[:, 1]))
                )
                lines.append(
                    f"| {camp.name[2:]} | {cond} | {name} | {len(links)} |"
                    f" {mine[:, 0].mean():.3f} | {mine[:, 1].mean():.1f} | {vs[0]} | {vs[1]} |"
                )
    with (a.out / "summary.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
