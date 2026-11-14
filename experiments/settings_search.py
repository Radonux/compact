"""Rank the settings of a search over closed-loop training runs.

Each run is one `expliccit train ppo` output named GROUP-sSEED-TAG, where
GROUP names what is trained (for example ctl-cubic-k64) and TAG the point
of the search. Runs of one group and tag differ only by seed. A point's
score is the mean over its seeds of the score of the policy each run left:
the log utility on the validation links minus the fixed rule's, which is
the same for every point and so cancels between them. The score is also
given apart for the conditions with a competing flow and without one,
because a larger share raises it with one (docs/methodology.md, comparing
controllers). A run that left no scored policy counts for nothing, and a
point with such a run is listed without a rank.

Usage: python experiments/settings_search.py OUT RUN_DIR...
Writes OUT/settings_search.json and OUT/settings-search.md.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

NAME = re.compile(r"^(?P<group>.+)-s(?P<seed>\d+)-(?P<tag>[^-].*)$")
TAIL = 40  # iterations at the end over which the learner's step size is read


def competing(condition: str) -> bool:
    """Whether a condition's label (train/score.py) names a competing flow."""
    return " from " in condition


def run_row(run: Path) -> dict[str, Any] | None:
    """One run's numbers, or None when it left no scored policy."""
    m = NAME.match(run.name)
    history = run / "history.json"
    if m is None or not history.exists():
        return None
    hist = json.loads(history.read_text())
    if not hist or not (ev := hist[-1].get("eval")) or not ev["pairs"]:
        return None
    by = ev["minus_rule_by_condition"]
    with_flow = [v for c, v in by.items() if competing(c)]
    alone = [v for c, v in by.items() if not competing(c)]
    tail = [h["learner"] for h in hist[-TAIL:] if h.get("learner")]
    return {
        "group": m["group"],
        "tag": m["tag"],
        "seed": int(m["seed"]),
        "score": ev["minus_rule"],
        "alone": statistics.fmean(alone) if alone else None,
        "competing": statistics.fmean(with_flow) if with_flow else None,
        "evals": [h["eval"]["minus_rule"] for h in hist if h.get("eval")],
        "clip_fraction": statistics.fmean(t["clip_fraction"] for t in tail) if tail else None,
        "approx_kl": statistics.fmean(t["approx_kl"] for t in tail) if tail else None,
    }


def rank(rows: list[dict[str, Any]], seeds: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Per group, its points by mean score, best first. `seeds` is how many
    runs each (group, tag) started with; a point missing one has no rank."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["group"], r["tag"])].append(r)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (group, tag), n in seeds.items():
        rs = by.get((group, tag), [])
        full = len(rs) == n

        def mean(key: str, rs: list[dict[str, Any]] = rs, full: bool = full) -> float | None:
            vals = [r[key] for r in rs]
            return statistics.fmean(vals) if full and None not in vals else None

        scores = [r["score"] for r in rs]
        out[group].append(
            {
                "tag": tag,
                "seeds": len(rs),
                "started": n,
                "score": mean("score"),
                "spread": statistics.stdev(scores) if full and n > 1 else None,
                "alone": mean("alone"),
                "competing": mean("competing"),
                "clip_fraction": mean("clip_fraction"),
                "approx_kl": mean("approx_kl"),
                "scores": sorted(scores),
            }
        )
    for points in out.values():
        points.sort(key=lambda p: (p["score"] is None, -(p["score"] or 0.0), p["tag"]))
    return dict(out)


def _f(v: float | None, spec: str = "+.2f") -> str:
    return "n/a" if v is None else format(v, spec)


def summary(ranked: dict[str, Any]) -> str:
    lines = [
        "# The settings search",
        "",
        "Scores are differences in log utility from the fixed rule, for the",
        "policy each run left, as the mean over seeds. The rule is the same",
        "for every point, so it cancels between points. Spread is the",
        "standard deviation over seeds. Clipped and KL (per step) are the learner's",
        f"means over the last {TAIL} iterations.",
        "",
    ]
    for group, points in sorted(ranked.items()):
        lines += [
            f"## {group}",
            "",
            "| Point | Seeds | Score | Spread | Alone | Competing | Clipped | KL | Each seed |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in points:
            each = ", ".join(f"{s:+.2f}" for s in p["scores"])
            lines.append(
                f"| {p['tag']} | {p['seeds']} of {p['started']} | {_f(p['score'])} | "
                f"{_f(p['spread'], '.2f')} | {_f(p['alone'])} | {_f(p['competing'])} | "
                f"{_f(p['clip_fraction'], '.2f')} | {_f(p['approx_kl'], '.3f')} | {each} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(out: Path, runs: list[Path]) -> None:
    seeds: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for run in runs:
        if (m := NAME.match(run.name)) is None:
            raise SystemExit(f"{run.name}: not GROUP-sSEED-TAG")
        seeds[(m["group"], m["tag"])] += 1
        if (row := run_row(run)) is not None:
            rows.append(row)
    ranked = rank(rows, dict(seeds))
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings_search.json").write_text(json.dumps({"points": ranked, "runs": rows}))
    (out / "settings-search.md").write_text(summary(ranked))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
