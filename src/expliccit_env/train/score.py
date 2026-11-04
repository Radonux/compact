"""Scores of closed-loop runs, from ground truth, without PyTorch.

Each ok episode scores the mean over its steps of a net2 reward (`utility`,
the log utility by default). A controller's score on a link is the mean of
its episodes there, per condition (the cross traffic and random loss an
episode ran under) and over all conditions. Intervals are bootstrap 95%
intervals over links; a difference from a reference controller is paired
by link. Failed episodes and steps without a defined throughput share never
count, and an incomplete shard is an error, not a smaller sample.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from expliccit_env.spec import AgentCC
from expliccit_env.train.data import F32, link_key
from expliccit_env.train.spec import Net2Spec

LABELS = ("receive_rate_bps", "capacity_bps", "queue_delay_mean_ns", "base_rtt_ns", "drop_fraction")
RESAMPLES = 4000

type Key = tuple[str, str, str]  # controller, link, condition


def utility(labels: dict[str, F32], spec: Net2Spec) -> F32:
    """Per step: receive rate over capacity (its log, at least 0.01, for
    "log_utility"), minus the delay and loss terms, from ground truth. An
    empty window (no departures) costs no delay."""
    with np.errstate(divide="ignore", invalid="ignore"):
        used = labels["receive_rate_bps"] / labels["capacity_bps"]
        delay = np.nan_to_num(labels["queue_delay_mean_ns"] / labels["base_rtt_ns"])
    loss = np.nan_to_num(labels["drop_fraction"])
    gain = np.log(np.maximum(used, 0.01)) if spec.reward == "log_utility" else used
    r = gain - spec.delay_weight * delay - spec.loss_weight * loss
    return np.where(np.isfinite(used), r, np.nan).astype(np.float32)


def condition(spec_json: str) -> str:
    """The cross traffic and random loss of an episode, as a short label."""
    spec: dict[str, Any] = json.loads(spec_json)
    cross: list[dict[str, Any]] = spec.get("cross") or []
    loss: dict[str, float] = spec["path"].get("loss") or {}
    parts: list[str] = []
    observed: dict[str, Any] = spec.get("cc") or {}
    for c in cross:
        if c["kind"] != "tcp":
            what = f"UDP {c['rate_mbps']:g} Mbit/s"
        elif c["cc"] == observed or c["cc"] == observed.get("name"):
            # A copy of the observed flow's own controller: one label for
            # every controller, so copies pair across controllers.
            what = "itself"
        elif isinstance(c["cc"], str):
            what = c["cc"]
        else:  # an agent: named as in the episodes table
            what = AgentCC.model_validate(c["cc"]).label
        parts.append(f"{what} from {c['start_ms'] / 1000:g} s")
    if loss.get("downlink"):
        parts.append(f"loss {100 * loss['downlink']:g}%")
    if loss.get("uplink"):
        parts.append(f"ACK loss {100 * loss['uplink']:g}%")
    return ", ".join(parts) or "clean"


def episode_scores(dirs: list[Path], spec: Net2Spec) -> dict[Key, list[float]]:
    """Each ok episode's mean reward, grouped by controller, link and condition."""
    shards = [s for d in dirs for s in sorted((d / "shards").glob("*"))]
    if missing := [str(s) for s in shards if not (s / "DONE").exists()]:
        raise ValueError(f"incomplete shards: {missing}")
    out: dict[Key, list[float]] = defaultdict(list)
    for shard in shards:
        eps: dict[str, Key] = {}
        # pyarrow-stubs leave read_table partly untyped.
        cols = ["episode_id", "controller", "spec_json", "status"]
        table = pq.read_table(shard / "episodes.parquet", columns=cols)  # pyright: ignore[reportUnknownMemberType]
        for e in table.to_pylist():
            if e["status"] == "ok":
                eps[e["episode_id"]] = (
                    e["controller"],
                    link_key(e["spec_json"]),
                    condition(e["spec_json"]),
                )
        f = pq.ParquetFile(shard / "steps.parquet")
        columns = ["episode_id", *(f"label_{k}" for k in LABELS)]
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g, columns=columns)  # pyright: ignore[reportUnknownMemberType]
            ids, which = np.unique(
                t["episode_id"].to_numpy(zero_copy_only=False), return_inverse=True
            )
            labels = {
                k: t[f"label_{k}"].to_numpy(zero_copy_only=False).astype(np.float32) for k in LABELS
            }
            r = utility(labels, spec).astype(np.float64)
            ok = np.isfinite(r)
            total = np.bincount(which[ok], r[ok], minlength=ids.size)
            count = np.bincount(which[ok], minlength=ids.size)
            for eid, s, n in zip(ids.tolist(), total.tolist(), count.tolist(), strict=True):
                if eid in eps and n:
                    out[eps[eid]].append(s / n)
    return dict(out)


def failures(dirs: list[Path]) -> dict[str, int]:
    """Episodes that did not finish ok, per controller: never scored, always reported."""
    out: dict[str, int] = defaultdict(int)
    for shard in (s for d in dirs for s in sorted((d / "shards").glob("*"))):
        cols = ["controller", "status"]
        table = pq.read_table(shard / "episodes.parquet", columns=cols)  # pyright: ignore[reportUnknownMemberType]
        for e in table.to_pylist():
            if e["status"] != "ok":
                out[e["controller"]] += 1
    return dict(out)


def _interval(x: np.ndarray[Any, np.dtype[np.float64]], rng: np.random.Generator) -> dict[str, Any]:
    boot = x[rng.integers(0, x.size, (RESAMPLES, x.size))].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "links": int(x.size),
    }


def compare(
    scores: dict[Key, list[float]],
    reference: str | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Per controller, over all conditions and per condition: the mean over
    links with its interval and, when a reference is named, the paired
    difference from it."""
    alias = aliases or {}
    per: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (ctl, link, cond), v in scores.items():
        per[alias.get(ctl, ctl)][cond][link] = float(np.mean(v))
    for conds in per.values():
        links = {link for by_link in conds.values() for link in by_link}
        conds["all"] = {
            link: float(np.mean([c[link] for c in conds.values() if link in c])) for link in links
        }
    if reference is not None and reference not in per:
        raise ValueError(f"reference {reference!r} not among {sorted(per)}")
    rng = np.random.default_rng(0)
    result: dict[str, Any] = {}
    for name in sorted(per):
        result[name] = {}
        for cond, by_link in sorted(per[name].items()):
            entry = _interval(np.array(list(by_link.values())), rng)
            if reference is not None and name != reference:
                ref = per[reference].get(cond, {})
                shared = sorted(set(by_link) & set(ref))
                if shared:
                    entry["minus_reference"] = _interval(
                        np.array([by_link[k] - ref[k] for k in shared]), rng
                    )
            result[name][cond] = entry
    return {"reference": reference, "controllers": result}


def markdown(result: dict[str, Any]) -> str:
    """The comparison as a Markdown table, controllers by score over all conditions."""
    ctls: dict[str, dict[str, Any]] = result["controllers"]
    conds = sorted({c for v in ctls.values() for c in v if c != "all"})
    ref = result["reference"]

    def cell(e: dict[str, Any] | None) -> str:
        if e is None:
            return ""
        lo, hi = e["ci95"]
        return f"{e['mean']:+.3f} ({lo:+.3f} to {hi:+.3f})"

    head = ["controller", "all conditions", *conds]
    if ref is not None:
        head.append(f"minus {ref}, all conditions")
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, v in sorted(ctls.items(), key=lambda kv: -kv[1]["all"]["mean"]):
        line = [name, cell(v["all"]), *(f"{v[c]['mean']:+.3f}" if c in v else "" for c in conds)]
        if ref is not None:
            line.append(cell(v["all"].get("minus_reference")))
        rows.append("| " + " | ".join(line) + " |")
    failed: dict[str, int] = result.get("failed", {})
    note = ", ".join(f"{k} {v}" for k, v in sorted(failed.items())) if failed else "none"
    return "\n".join(rows) + f"\nFailed episodes, not scored: {note}.\n"
