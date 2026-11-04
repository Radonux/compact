"""Score a benchmark planned by experiments/ccbench.py, from each episode's
flow_seconds table (bytes and queueing delay per flow and second).

ccBench's leagues (its league.sh and league-piecewise*.py at 49b5ada):

    solo, loss   In each run and window ([0, 3], [3, 10], [10, 17], [17, 24] s
                 of the flow), a controller scores round(T)^2 / round(d + D),
                 T its throughput in Mbit/s and d + D its mean delay in ms: d
                 the one-way delay of its packets (queueing plus D, as
                 Pantheon's tunnel measures it, assumed) and D the one-way
                 delay ccBench adds again. Every controller within 10% of the
                 best score wins that game.
    friendly,    In each run and window ([5, 35], ..., [95, 125] s), the
    competitors  controller's throughput T against the competitor; with m the
                 smallest |B / 2 - T| of any controller there, every one with
                 T in [(B / 2 - m) 0.9, (B / 2 + m) 1.1] wins.

A controller's rank is the share of games it wins. Games are between the
controllers of this run only, so ranks compare them with one another. Next
to them, absolute means over runs and windows: utilization (all flows'
bytes over the trace's capacity), the controller's mean queueing delay, its
share of the capacity against a competitor, and Jain's index of copies.

Usage: python experiments/ccbench_report.py OUT
Appends to OUT/summary.md.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from expliccit_env import mahimahi as mm
from expliccit_env.spec import AgentCC, EpisodeSpec

WINDOWS = {
    "solo": [(0, 3), (3, 10), (10, 17), (17, 24)],
    "loss": [(0, 3), (3, 10), (10, 17), (17, 24)],
    "friendly": [(5, 35), (35, 65), (65, 95), (95, 125)],
    "competitors": [(5, 35), (35, 65), (65, 95), (95, 125)],
    "copies": [(0, 15), (15, 30), (30, 45), (45, 60)],
}
MARGIN = 0.10


def name_of(cc: Any, aliases: dict[str, str]) -> str:
    label = cc if isinstance(cc, str) else cc.get("name") if cc["kind"] == "stock" else None
    label = label or AgentCC.model_validate(cc).label
    return aliases.get(label, label)


def capacity_mbps(spec: EpisodeSpec, lo: int, hi: int) -> float:
    """The trace's mean capacity over [lo, hi) s, in Mbit/s (Mahimahi's 1504-byte units),
    the trace repeating as the link repeats it."""
    ts = mm.opportunity_times(spec.path.trace.timestamps(), hi * 1000)
    return float((ts >= lo * 1000).sum()) * mm.PACKET_SIZE * 8 / (hi - lo) / 1e6


Games = dict[tuple[str, int], dict[str, float]]
Metrics = dict[str, dict[str, list[float]]]


def collect(
    camp: Path, suite: str, aliases: dict[str, str]
) -> tuple[Games, dict[tuple[str, int], float], Metrics]:
    """Per game (run and window) the value each controller is judged on, the
    capacity there, and each controller's absolute metrics."""
    games: Games = defaultdict(dict)
    cap_of: dict[tuple[str, int], float] = {}
    metrics: Metrics = defaultdict(lambda: defaultdict(list))
    versus = suite in ("friendly", "competitors")
    for shard in sorted(camp.glob("shards/*")):
        eps = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "spec_json", "status"]
        )
        fs = pq.read_table(shard / "flow_seconds.parquet").to_pydict()
        rows: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(fs["episode_id"]):
            rows[e].append(i)
        for eid, sj, status in zip(*(eps[c].to_pylist() for c in eps.column_names), strict=True):
            if status != "ok":
                continue
            spec = EpisodeSpec.model_validate_json(sj)
            raw = json.loads(sj)
            who = name_of(raw["cross"][0]["cc"] if versus else raw["cc"], aliases)
            key = json.dumps([raw["path"], raw["cc"]["name"] if versus else ""], sort_keys=True)
            one_way = spec.path.delay.one_way_ns / 1e6
            col = {
                c: np.array([fs[c][i] for i in rows[eid]], float)
                for c in ("flow", "second", "bytes", "packets", "delay_sum_ms")
            }
            for w, (lo, hi) in enumerate(WINDOWS[suite]):
                inw = (col["second"] >= lo) & (col["second"] < hi)
                mine = inw & (col["flow"] == (1 if versus else 0))
                t = col["bytes"][mine].sum() * 8 / (hi - lo) / 1e6
                d = col["delay_sum_ms"][mine].sum() / max(col["packets"][mine].sum(), 1)
                cap = capacity_mbps(spec, lo, hi)
                m = metrics[who]
                m["utilization"].append(col["bytes"][inw].sum() * 8 / (hi - lo) / 1e6 / cap)
                m["queueing delay (ms)"].append(d)
                if versus:
                    m["share of capacity"].append(t / cap)
                    games[(key, w)][who] = t
                    cap_of[(key, w)] = cap
                elif suite == "copies":
                    per = np.array(
                        [
                            col["bytes"][inw & (col["flow"] == f)].sum()
                            for f in np.unique(col["flow"])
                        ]
                    )
                    m["Jain's index"].append(per.sum() ** 2 / (per.size * (per**2).sum()))
                else:
                    games[(key, w)][who] = round(t) ** 2 / max(round(d + one_way + one_way), 1)
    return games, cap_of, metrics


def league(
    suite: str, games: Games, cap_of: dict[tuple[str, int], float]
) -> tuple[dict[str, int], dict[str, int]]:
    """Games won and played per controller, by ccBench's rules."""
    wins: dict[str, int] = defaultdict(int)
    played: dict[str, int] = defaultdict(int)
    for g, scores in games.items():
        for who in scores:
            played[who] += 1
        if suite in ("friendly", "competitors"):
            half = cap_of[g] / 2
            best = min(abs(half - round(t)) for t in scores.values())
            lo, hi = (half - best) * (1 - MARGIN), (half + best) * (1 + MARGIN)
            won = [w for w, t in scores.items() if lo <= t <= hi]
        else:
            top = max(scores.values())
            won = [w for w, s in scores.items() if top > 0 and s >= top * (1 - MARGIN)]
        for who in won:
            wins[who] += 1
    return wins, played


def main(out: Path) -> None:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    lines: list[str] = []
    for camp in sorted(out.glob("c-*")):
        suite = camp.name[2:]
        if suite not in WINDOWS:
            continue
        games, cap_of, metrics = collect(camp, suite, aliases)
        wins, played = league(suite, games, cap_of)
        names = sorted(metrics, key=lambda n: (-wins[n] / max(played[n], 1), n))
        cols = list(next(iter(metrics.values())).keys()) if metrics else []
        lines += ["", f"## {suite}", ""]
        lines += ["| controller | games won |" + "".join(f" {c} |" for c in cols)]
        lines += ["|---|---|" + "---|" * len(cols)]
        for n in names:
            won = "" if suite == "copies" else f"{wins[n] / max(played[n], 1):.1%} of {played[n]}"
            vals = "".join(f" {np.nanmean(metrics[n][c]):.3f} |" for c in cols)
            lines.append(f"| {n} | {won} |{vals}")
    with (out / "summary.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
