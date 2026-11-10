"""Throughput and delay per scenario, and fairness with several flows, for
each controller of a baselines campaign (experiments/baselines.json).

A scenario is a family (one campaign, c-FAMILY-SECONDS) and a condition (its
cross traffic and loss, score.condition, with "itself" named, so Cubic's run
against Cubic sits with the others' runs against Cubic); a copies
campaign's scenario is its number of flows. For each scenario and controller, over the links (95%
bootstrap intervals):

- the observed flow's throughput through the bottleneck (Mbit/s) and its
  share of the capacity;
- its queueing delay per packet: the mean and the 95th percentile over the
  episode (episode row flow_delay_mean_ms, flow_delay_p95_ms);
- where other TCP flows share the link: the observed flow's share of the
  TCP flows' bytes, and Jain's index over them (UDP is left out of both);
- for copies: Jain's index over the copies, the link's utilization by all
  of them, and their mean queueing delay;
- on Aurora's training links (family aurora-range, the flow alone): its
  loss, and Aurora's own reward, PCC-RL's per-interval reward without its
  0.001 scale (src/gym/network_sim.py at 64bea6e: 10 * packets a second
  - 1000 * RTT in s - 2000 * loss ratio) from the episode's means, the RTT
  as the base RTT plus the mean queueing delay and the loss as the queue's
  drops and the link's random loss together.

The summary leaves aurora-range out, so it stays comparable with earlier
reports, and gives it its own table.

Writes OUT/baselines.md and OUT/baselines.csv (one row per episode), from
the campaigns of every DIR (each with its aliases.json), so a later run of
more controllers on the same links joins an earlier one.

Usage: python experiments/baselines_report.py OUT DIR...
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from expliccit_env.spec import EpisodeSpec
from expliccit_env.train.data import link_key
from expliccit_env.train.score import condition

type Vec = np.ndarray[tuple[int], np.dtype[np.float64]]


def interval(x: Vec, digits: int = 2) -> str:
    x = x[np.isfinite(x)]
    if not x.size:
        return "n/a"
    boot = x[np.random.default_rng(0).integers(0, x.size, (4000, x.size))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return f"{x.mean():.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def jain(x: Vec) -> float:
    return float(x.sum() ** 2 / (x.size * (x**2).sum())) if x.size and x.any() else float("nan")


def episode(e: dict[str, Any], name: str, family: str) -> dict[str, Any]:
    spec = json.loads(e["spec_json"])
    seconds = spec["duration_ms"] / 1000
    nbytes = np.asarray(e["flow_bytes"], np.float64)
    tcp = np.asarray([True] + [c["kind"] == "tcp" for c in spec.get("cross") or []])
    cap = e.get("average_capacity_mbps") or float("nan")
    thr = nbytes[0] * 8 / seconds / 1e6
    copies = family.startswith("copies")
    loss = reward = float("nan")
    if family.startswith("aurora-range"):
        path = EpisodeSpec.model_validate_json(e["spec_json"]).path
        loss = 1 - (1 - e["drop_fraction"]) * (1 - path.loss.downlink)
        rtt_s = (2 * path.delay.one_way_ns / 1e6 + e["flow_delay_mean_ms"][0]) / 1e3
        reward = 10 * thr * 1e6 / (8 * 1500) - 1e3 * rtt_s - 2e3 * loss
    return {
        "family": family,
        "scenario": f"{nbytes.size} flows"
        if copies
        else condition(e["spec_json"]).replace("itself", name),
        "controller": name,
        "link": link_key(e["spec_json"]),
        "throughput_mbps": thr,
        "capacity_share": thr / cap,
        "delay_mean_ms": e["flow_delay_mean_ms"][0],
        "delay_p95_ms": e["flow_delay_p95_ms"][0],
        "tcp_share": nbytes[0] / nbytes[tcp].sum() if tcp.sum() > 1 else float("nan"),
        "jain": jain(nbytes[tcp]) if tcp.sum() > 1 else float("nan"),
        "link_utilization": e.get("utilization", float("nan")),
        "all_delay_mean_ms": e.get("mean_delay_ms", float("nan")),
        "loss": loss,
        "aurora_reward": reward,
    }


def load(dirs: list[Path]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failed = 0
    for d in dirs:
        r, f = load_one(d)
        rows += r
        failed += f
    return rows, failed


def load_one(out: Path) -> tuple[list[dict[str, Any]], int]:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    rows: list[dict[str, Any]] = []
    failed = 0
    for c in sorted(out.glob("c-*")):
        for f in sorted((c / "shards").glob("*/episodes.parquet")):
            for e in pq.read_table(f).to_pylist():  # pyright: ignore[reportUnknownMemberType]
                if e["status"] != "ok" or e.get("flow_delay_p95_ms") is None:
                    failed += 1
                    continue
                cc = str(e["controller"])
                rows.append(episode(e, aliases.get(cc, cc), c.name[2:]))
    return rows, failed


def per_link(rows: list[dict[str, Any]], metric: str) -> Vec:
    """The metric's mean per link (one episode per link and scenario here)."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["link"]].append(float(r[metric]))
    return np.asarray([np.mean(v) for v in by.values()], np.float64)


def main(out: Path, dirs: list[Path]) -> None:
    rows, failed = load(dirs)
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        groups[(r["family"], r["scenario"])][r["controller"]].append(r)
    lines = [
        "# Baselines: throughput and delay per scenario, fairness with several flows",
        "",
        f"{len(rows)} episodes scored, {failed} failed and not scored. Means over links,"
        " with 95% bootstrap intervals over links. Delay is the observed flow's queueing"
        " delay at the bottleneck, per packet.",
        "",
        "## Summary over the scenarios, copies apart",
        "",
        "Each scenario (family and condition) weighs the same: the median over scenarios of"
        " each controller's mean over links.",
        "",
        "| controller | share of capacity | mean delay (ms) | p95 delay (ms) | scenarios |",
        "|---|---|---|---|---|",
    ]
    names = sorted({r["controller"] for r in rows})
    for name in names:
        per = [
            by[name]
            for (family, _), by in groups.items()
            if not family.startswith(("copies", "aurora-range")) and by.get(name)
        ]
        if not per:
            continue
        cells = [
            f"{np.median([per_link(rs, m).mean() for rs in per]):.{d}f}"
            for m, d in (("capacity_share", 3), ("delay_mean_ms", 1), ("delay_p95_ms", 1))
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {len(per)} |")
    aurora = {k: by for k, by in groups.items() if k[0].startswith("aurora-range")}
    if aurora:
        lines += [
            "",
            "## Summary on Aurora's training links",
            "",
            "The mean over aurora-range's conditions of each controller's mean over links.",
            "",
            "| controller | share of capacity | mean delay (ms) | loss | Aurora's reward |",
            "|---|---|---|---|---|",
        ]
        for name in names:
            per = [by[name] for by in aurora.values() if by.get(name)]
            if per:
                cells = [
                    f"{np.mean([per_link(rs, m).mean() for rs in per]):.{d}f}"
                    for m, d in (
                        ("capacity_share", 3),
                        ("delay_mean_ms", 1),
                        ("loss", 4),
                        ("aurora_reward", 0),
                    )
                ]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for (family, scenario), by in sorted(groups.items()):
        n = len({r["link"] for rs in by.values() for r in rs})
        lines += ["", f"## {family}: {scenario} ({n} links)", ""]
        if family.startswith("copies"):
            lines += [
                "| controller | Jain's index | link utilization | mean queueing delay (ms) |"
                " observed flow p95 delay (ms) |",
                "|---|---|---|---|---|",
            ]
            for name, rs in sorted(by.items()):
                lines.append(
                    f"| {name} | {interval(per_link(rs, 'jain'), 3)} |"
                    f" {interval(per_link(rs, 'link_utilization'), 3)} |"
                    f" {interval(per_link(rs, 'all_delay_mean_ms'), 1)} |"
                    f" {interval(per_link(rs, 'delay_p95_ms'), 1)} |"
                )
            continue
        shared = any(np.isfinite(r["tcp_share"]) for rs in by.values() for r in rs)
        scored = family.startswith("aurora-range")
        head = (
            "| controller | throughput (Mbit/s) | share of capacity | mean delay (ms) |"
            " p95 delay (ms) |"
        )
        lines += [
            head
            + (" share of TCP bytes | Jain's index |" if shared else "")
            + (" loss | Aurora's reward |" if scored else ""),
            "|---|---|---|---|---|"
            + ("---|---|" if shared else "")
            + ("---|---|" if scored else ""),
        ]
        for name, rs in sorted(by.items()):
            cells = [
                interval(per_link(rs, "throughput_mbps"), 2),
                interval(per_link(rs, "capacity_share"), 3),
                interval(per_link(rs, "delay_mean_ms"), 1),
                interval(per_link(rs, "delay_p95_ms"), 1),
            ]
            if shared:
                cells += [
                    interval(per_link(rs, "tcp_share"), 3),
                    interval(per_link(rs, "jain"), 3),
                ]
            if scored:
                cells += [
                    interval(per_link(rs, "loss"), 4),
                    interval(per_link(rs, "aurora_reward"), 0),
                ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.md").write_text("\n".join(lines) + "\n")
    with (out / "baselines.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["family"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]])
