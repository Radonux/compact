"""Compare the original stack's Mahimahi downlink logs with palaestra's
episodes on the cross-check links (tools/crosscheck/README.md).

Usage: uv run python tools/crosscheck/compare.py [--mahimahi DIR] --palaestra RELEASE_DIR
           [--out FILE]

DIR holds one mm-link downlink log per run, named LINK.CONTROLLER.RUN.log
(a.cubic.1.log). RELEASE_DIR is an unpacked crosscheck release, or any
directory under which palaestra wrote episodes.parquet with
link_events.parquet (searched recursively), and aliases.json at its top for
the policies' names. Without --mahimahi, palaestra's side alone (the
experiment's summary). The table goes to stdout, and to FILE with --out.

Both sides become the same thing, a Mahimahi link log
(expliccit_env.mahimahi.MmLog): the team's file through parse_mm_log, and
palaestra's link records through mm_log_from_records, the converter its own
mm-th-delay metrics use. One function computes every metric from that log,
over the link's window (links.py): from 2 s after t0 to 1 s before the run's
duration after it, where t0 is the arrival of the first packet of at least
1000 bytes, the flow's first data segment. So each side starts at its own
flow's start, whatever its harness did before.

    utilization     bytes departed over the bytes of the delivery
                    opportunities in the window (both in Mahimahi bytes,
                    1504 per opportunity)
    queueing delay  per departed packet, its departure minus its arrival ms
                    at mm-link: the sojourn in the link queue, without
                    mm-delay; the mean and the 95th percentile

A run whose log does not cover its window, or a palaestra episode that
failed, gives no metrics; the table counts it and lists why.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from links import BY_NAME, LINKS, Link, link_of

from expliccit_env import mahimahi as mm
from expliccit_env import records as rec
from expliccit_env.labels import DOWNLINK
from expliccit_env.spec import AgentCC, EpisodeSpec

DATA_BYTES = 1000  # a data segment: no ACK, handshake or iperf3 control message is this long
LOG_NAME = re.compile(r"^(?P<link>[a-z])\.(?P<cc>[a-z0-9_-]+)\.(?P<run>[0-9]+)\.log$")
ORDER = ("cubic", "bbr", "vegas", "sage", "expliccit")
EVENTS = ["episode_id", "event", "link_id", "bytes", "ms", "arrival_ms", "reason"]


@dataclass(frozen=True, slots=True)
class Metrics:
    t0_ms: int  # trace ms of the first data packet's arrival
    utilization: float
    delay_mean_ms: float
    delay_p95_ms: float


@dataclass(frozen=True, slots=True)
class Invalid:
    reason: str


@dataclass(frozen=True, slots=True)
class Run:
    side: str  # "mahimahi" or "palaestra"
    link: str
    controller: str
    source: str  # the log file, or the episode and its seed
    result: Metrics | Invalid


def metrics(log: mm.MmLog, link: Link) -> Metrics | Invalid:
    """Every metric of one run, from its downlink log."""
    rel = log.ts - log.base_timestamp
    data = (log.kind == mm.Kind.ARRIVAL) & (log.nbytes >= DATA_BYTES)
    if not data.any():
        return Invalid("no data packet reached the link")
    t0 = int(rel[data].min())
    lo, hi = (t0 + 1000 * s for s in link.window_s)
    if int(rel.max()) < hi:
        return Invalid(f"the log ends at {int(rel.max())} ms, before the window's end at {hi} ms")
    inside = (rel >= lo) & (rel < hi)
    capacity = int(log.nbytes[inside & (log.kind == mm.Kind.OPPORTUNITY)].sum())
    dep = inside & (log.kind == mm.Kind.DEPARTURE)
    if capacity == 0 or not dep.any():
        return Invalid("no delivery opportunity or no departure in the window")
    delay = log.delay[dep].astype(np.float64)
    return Metrics(
        t0,
        float(log.nbytes[dep].sum()) / capacity,
        float(delay.mean()),
        float(np.percentile(delay, 95)),
    )


def mahimahi_runs(d: Path) -> list[Run]:
    runs: list[Run] = []
    for f in sorted(d.glob("*.log")):
        m = LOG_NAME.match(f.name)
        if m is None or m["link"] not in BY_NAME:
            raise SystemExit(f"{f}: name it LINK.CONTROLLER.RUN.log, LINK one of {sorted(BY_NAME)}")
        try:
            result = metrics(mm.parse_mm_log(f.read_text()), BY_NAME[m["link"]])
        except ValueError as e:
            result = Invalid(f"not a Mahimahi log: {e}")
        runs.append(Run("mahimahi", m["link"], m["cc"], f.name, result))
    return runs


def _records(events: pa.Table) -> rec.Records:
    """The link tables of one episode's link_events rows, as Records hold them."""
    kind = pc.cast(events["event"], pa.string())

    def table(name: str) -> rec.Table:
        t = events.filter(pc.equal(kind, pa.scalar(name)))
        arr = np.zeros(t.num_rows, rec.LINK)
        for col in EVENTS[2:]:
            arr[col] = t[col].to_numpy()
        return arr

    def empty(dtype: np.dtype[np.void]) -> rec.Table:
        return np.zeros(0, dtype)

    return rec.Records(
        meta=empty(rec.META),
        link_arrival=table("arrival"),
        link_drop=table("drop"),
        link_depart=table("depart"),
        link_opp=empty(rec.OPP),
        link_stats=empty(rec.STATS),
        tcp_obs=empty(rec.OBS),
        agent_action=empty(rec.AGENT),
    )


def palaestra_runs(root: Path) -> list[Run]:
    aliases_file = root / "aliases.json"
    aliases: dict[str, str] = json.loads(aliases_file.read_text()) if aliases_file.exists() else {}
    runs: list[Run] = []
    for episodes in sorted(root.rglob("episodes.parquet")):
        rows = pq.read_table(
            episodes, columns=["episode_id", "spec_json", "status", "failure"]
        ).to_pylist()
        todo: dict[str, tuple[Link, str, EpisodeSpec, str]] = {}
        for r in rows:
            spec = EpisodeSpec.model_validate_json(r["spec_json"])
            if (link := link_of(spec)) is None:
                continue
            cc = (
                aliases.get(spec.cc.label, spec.cc.label)
                if isinstance(spec.cc, AgentCC)
                else spec.cc.name
            )
            source = f"episode {r['episode_id']} seed {spec.seed}"
            if r["status"] != "ok":
                runs.append(
                    Run("palaestra", link.name, cc, source, Invalid(f"failed: {r['failure']}"))
                )
            else:
                todo[r["episode_id"]] = (link, cc, spec, source)
        events_file = episodes.parent / "link_events.parquet"
        if todo and not events_file.exists():
            raise SystemExit(f"{events_file} is missing: run the campaign with link_events")
        if not todo:
            continue
        events = pq.read_table(events_file, columns=EVENTS)
        ids = pc.cast(events["episode_id"], pa.string())
        for eid, (link, cc, spec, source) in todo.items():
            mine = events.filter(pc.equal(ids, pa.scalar(eid)))
            log = mm.mm_log_from_records(
                _records(mine), DOWNLINK, spec.path.trace.timestamps(), spec.duration_ms
            )
            runs.append(Run("palaestra", link.name, cc, source, metrics(log, link)))
    return runs


def _cell(values: list[float], digits: int) -> str:
    """The mean over runs, and their range when the runs differ."""
    if not values:
        return "-"
    mean, lo, hi = (f"{v:.{digits}f}" for v in (np.mean(values), min(values), max(values)))
    return mean if lo == hi else f"{mean} ({lo} to {hi})"


def table(runs: list[Run], sides: tuple[str, ...]) -> str:
    groups: dict[tuple[str, str], dict[str, list[Run]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        groups[(r.link, r.controller)][r.side].append(r)
    rank = {c: i for i, c in enumerate(ORDER)}
    keys = sorted(groups, key=lambda k: (k[0], rank.get(k[1], len(ORDER)), k[1]))
    names = {"mahimahi": "Mahimahi", "palaestra": "palaestra"}
    metric_cols = [
        ("utilization", "utilization", 3),
        ("delay_mean_ms", "mean queueing delay (ms)", 2),
        ("delay_p95_ms", "95th percentile queueing delay (ms)", 2),
    ]
    head = ["link", "controller", *(f"runs, {names[s]}" for s in sides)]
    for _, title, _ in metric_cols:
        head += [f"{title}, {names[s]}" for s in sides]
        if len(sides) == 2:
            head.append(f"{title}, difference")
    head += [f"data start (ms), {names[s]}" for s in sides]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    left_out: list[str] = []
    for key in keys:
        ok = {s: [r.result for r in groups[key][s] if isinstance(r.result, Metrics)] for s in sides}
        for s in sides:
            left_out += [
                f"- {names[s]}, link {r.link}, {r.controller}, {r.source}: {r.result.reason}"
                for r in groups[key][s]
                if isinstance(r.result, Invalid)
            ]
        row = [*key, *(f"{len(ok[s])} of {len(groups[key][s])}" for s in sides)]
        for attr, _, digits in metric_cols:
            vals = {s: [float(getattr(m, attr)) for m in ok[s]] for s in sides}
            row += [_cell(vals[s], digits) for s in sides]
            if len(sides) == 2:
                a, b = (vals[s] for s in sides)
                row.append(f"{np.mean(a) - np.mean(b):+.{digits}f}" if a and b else "-")
        row += [_cell([float(m.t0_ms) for m in ok[s]], 0) for s in sides]
        lines.append("| " + " | ".join(row) + " |")
    windows = ", ".join(f"{k.name} [{k.window_s[0]}, {k.window_s[1]}) s" for k in LINKS)
    diff = "; the difference is Mahimahi minus palaestra" if len(sides) == 2 else ""
    lines += [
        "",
        f"Means over runs, with their range where the runs differ{diff}."
        f" Windows after each run's first data packet: {windows}.",
    ]
    if left_out:
        lines += ["", "Runs that gave no metrics:", "", *left_out]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mahimahi", type=Path, help="the team's LINK.CONTROLLER.RUN.log files")
    p.add_argument("--palaestra", type=Path, required=True, help="an unpacked crosscheck release")
    p.add_argument("--out", type=Path, help="also write the table here")
    a = p.parse_args()
    runs = palaestra_runs(a.palaestra)
    sides: tuple[str, ...] = ("palaestra",)
    if a.mahimahi is not None:
        runs = mahimahi_runs(a.mahimahi) + runs
        sides = ("mahimahi", "palaestra")
    title = "the original stack's Mahimahi and palaestra" if len(sides) == 2 else "palaestra"
    text = f"## Cross-check links: {title}\n\n" + table(runs, sides)
    print(text, end="")
    if a.out is not None:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
