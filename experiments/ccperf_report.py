"""Score a benchmark planned by experiments/ccperf.py, from each episode's
flow_seconds table, with the measures ccperf reports for its experiments.

Per run, over its 30 s:

    utilization        all flows' bytes over the trace's capacity
    queueing delay     mean bottleneck queueing delay of the packets of the
                       flows under test, in ms
    Jain's index       of the throughputs of the flows under test, each over
                       its own time on the link (runs with several of them)
    share / fair       the flows under test's share of all bytes over their
                       share of the flows (competition runs: 1 is fair, above
                       1 takes more)

summary.md gets, per experiment, each controller's means over variants and
repeats; ccperf.csv has the means per experiment, variant and controller.

Usage: python experiments/ccperf_report.py OUT
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ccbench_report import capacity_mbps, name_of

from expliccit_env.spec import EpisodeSpec

UTIL, DELAY, JAIN, SHARE = "utilization", "queueing delay (ms)", "Jain's index", "share / fair"
METRICS = (UTIL, DELAY, JAIN, SHARE)


def scores(spec: EpisodeSpec, n_test: int, fs: dict[str, np.ndarray]) -> dict[str, float]:
    """The measures of one run, from its flow_seconds rows."""
    secs = spec.duration_ms // 1000
    starts = [0] + [c.start_ms for c in spec.cross]
    sent = np.array([fs["bytes"][fs["flow"] == f].sum() for f in range(len(starts))])
    test = fs["flow"] < n_test
    out = {
        UTIL: sent.sum() * 8 / secs / 1e6 / capacity_mbps(spec, 0, secs),
        DELAY: fs["delay_sum_ms"][test].sum() / max(fs["packets"][test].sum(), 1),
    }
    if n_test > 1:
        thr = sent[:n_test] / (secs - np.array(starts[:n_test]) / 1000)
        out[JAIN] = thr.sum() ** 2 / (n_test * (thr**2).sum())
    if n_test < len(starts):
        out[SHARE] = sent[:n_test].sum() / sent.sum() / (n_test / len(starts))
    return out


def collect(camp: Path, aliases: dict[str, str]) -> dict[tuple[str, str], list[dict[str, float]]]:
    """(variant, controller) -> the measures of each of its runs."""
    variants: dict[str, list[str | int]] = json.loads((camp / "variants.json").read_text())
    out: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    cols = ("flow", "bytes", "packets", "delay_sum_ms")
    for shard in sorted(camp.glob("shards/*")):
        eps = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "spec_json", "status"]
        )
        fs = pq.read_table(shard / "flow_seconds.parquet", columns=["episode_id", *cols])
        ids = np.array(fs["episode_id"].to_pylist())
        arr = {c: fs[c].to_numpy().astype(float) for c in cols}
        for eid, sj, status in zip(*(eps[c].to_pylist() for c in eps.column_names), strict=True):
            if status != "ok":
                continue
            variant, n_test = variants[eid]
            spec = EpisodeSpec.model_validate_json(sj)
            rows = ids == eid
            who = name_of(json.loads(sj)["cc"], aliases)
            out[(str(variant), who)].append(
                scores(spec, int(n_test), {c: v[rows] for c, v in arr.items()})
            )
    return out


def mean(runs: list[dict[str, float]], col: str, digits: int) -> str:
    vals = [r[col] for r in runs if col in r]
    return f"{np.mean(vals):.{digits}f}" if vals else ""


def main(out: Path) -> None:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    lines: list[str] = ["", "# ccperf's experiments", ""]
    table: list[list[str]] = []
    for camp in sorted(out.glob("p-*")):
        exp = camp.name[2:]
        runs = collect(camp, aliases)
        planned = len(json.loads((camp / "variants.json").read_text()))
        got = sum(len(v) for v in runs.values())
        per: dict[str, list[dict[str, float]]] = defaultdict(list)
        for (variant, who), rs in sorted(runs.items()):
            per[who] += rs
            table.append([exp, variant, who, str(len(rs))] + [mean(rs, c, 4) for c in METRICS])
        cols = [c for c in METRICS if any(mean(rs, c, 1) for rs in per.values())]
        lines += [f"## {exp} ({got} of {planned} runs ok)", ""]
        lines += ["| controller | runs |" + "".join(f" {c} |" for c in cols)]
        lines += ["|---|---|" + "---|" * len(cols)]
        for who in sorted(per, key=lambda w: -np.mean([r[UTIL] for r in per[w]])):
            vals = "".join(f" {mean(per[who], c, 3)} |" for c in cols)
            lines.append(f"| {who} | {len(per[who])} |{vals}")
        lines.append("")
    with (out / "ccperf.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "variant", "controller", "runs", *METRICS])
        w.writerows(table)
    with (out / "summary.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
