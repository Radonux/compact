"""Each controller against BBRv3, on the same validation episodes.

Reads run directories named GROUP-sSEED whose RUN/ground/score holds the
run's own rollouts on the validation links (experiments/ground.py
rollouts; stock-bbr3-s1 is BBRv3 itself, rollouts --stock). Per link and
condition, a group's score is the mean over its seeds of the log utility
the trainer optimizes (trajectories.REWARD), and its share is the
observed flow's part of what all flows took from the bottleneck
(score.share). Every group's difference from BBRv3 is paired by link,
over the conditions without a competing flow and over those with one,
with a bootstrap 95% interval over links. A group is within the margin
in a kind of condition when the whole interval lies above -0.1, and
trained well when it is within in both (docs/methodology.md, comparing
controllers).
ponytail: seeds are averaged within a link before the bootstrap, so the
interval is over links; a bootstrap over seeds as well needs more seeds.

Usage: python experiments/versus_reference.py OUT RUN_DIR...
Writes OUT/versus_reference.json and OUT/versus-reference.md.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from expliccit_env.train import score
from expliccit_env.train.trajectories import REWARD

NAME = re.compile(r"^(?P<group>.+)-s(?P<seed>\d+)$")
MARGIN = 0.1
REFERENCE = "stock-bbr3"  # the group every other is set against

type Pair = tuple[str, str]  # link, condition


def competing(condition: str) -> bool:
    """Whether a condition's label (score.condition) names a competing flow."""
    return " from " in condition


def per_pair(eps: dict[score.Key, list[float]]) -> dict[Pair, float]:
    """One run's episodes, as the mean per link and condition."""
    out: dict[Pair, list[float]] = defaultdict(list)
    for (_, link, cond), v in eps.items():
        out[(link, cond)] += v
    return {k: statistics.fmean(v) for k, v in out.items()}


def over_seeds(runs: list[dict[Pair, float]]) -> dict[Pair, float]:
    """The mean over seeds, on the pairs every seed has."""
    keys = set.intersection(*(set(r) for r in runs))
    return {k: statistics.fmean(r[k] for r in runs) for k in keys}


def by_link(values: dict[Pair, float], want: bool) -> dict[str, float]:
    """Per link, the mean over its conditions with (want) or without a competing flow."""
    out: dict[str, list[float]] = defaultdict(list)
    for (link, cond), v in values.items():
        if competing(cond) == want:
            out[link].append(v)
    return {k: statistics.fmean(v) for k, v in out.items()}


def interval(x: list[float], rng: np.random.Generator) -> dict[str, Any]:
    a = np.asarray(x, np.float64)
    boot = a[rng.integers(0, a.size, (score.RESAMPLES, a.size))].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    verdict = "within" if lo > -MARGIN else "below" if hi < -MARGIN else "unresolved"
    return {"mean": float(a.mean()), "ci95": [lo, hi], "links": int(a.size), "verdict": verdict}


def compare(groups: dict[str, dict[str, dict[Pair, float]]]) -> dict[str, Any]:
    """groups: name -> {"score": ..., "share": ...} over seeds; each against
    REFERENCE."""
    rng = np.random.default_rng(0)
    ref = groups.get(REFERENCE)
    out: dict[str, Any] = {}
    for name in sorted(groups):
        row: dict[str, Any] = {}
        for kind, want in (("alone", False), ("competing", True)):
            own = {q: by_link(groups[name][q], want) for q in ("score", "share")}
            row[kind] = {q: statistics.fmean(v.values()) for q, v in own.items() if v}
            if ref is None or name == REFERENCE:
                continue
            theirs = by_link(ref["score"], want)
            links = sorted(set(own["score"]) & set(theirs))
            if links:
                diff = [own["score"][k] - theirs[k] for k in links]
                row[kind]["minus_bbr3"] = interval(diff, rng)
        verdicts = [row[k].get("minus_bbr3", {}).get("verdict") for k in ("alone", "competing")]
        row["trained_well"] = verdicts == ["within", "within"]
        out[name] = row
    return out


def text(result: dict[str, Any], seeds: dict[str, int]) -> str:
    lines = [
        "Log utility minus BBRv3's on the same validation episodes, the mean over "
        "seeds per link, paired by link, with a bootstrap 95% interval over links. "
        f"Within: the whole interval above -{MARGIN}. Trained well: within both alone "
        "and with a competing flow. Share: the observed flow's part of what all "
        "flows took, with a competing flow.\n",
        "| group | seeds | alone, minus BBRv3 | | with a competing flow, minus BBRv3 | "
        "| trained well | share |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def d(e: dict[str, Any] | None) -> tuple[str, str]:
        if e is None:
            return "", ""
        lo, hi = e["ci95"]
        return f"{e['mean']:+.3f} ({lo:+.2f} to {hi:+.2f})", e["verdict"]

    for name, r in result.items():
        a, c = r["alone"], r["competing"]
        ad, av = d(a.get("minus_bbr3"))
        cd, cv = d(c.get("minus_bbr3"))
        well = "" if name == REFERENCE else ("yes" if r["trained_well"] else "no")
        lines.append(
            f"| {name} | {seeds[name]} | {ad} | {av} | {cd} | {cv} | {well} "
            f"| {c.get('share', float('nan')):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(out: Path, runs: list[Path]) -> None:
    found: dict[str, list[dict[str, dict[Pair, float]]]] = defaultdict(list)
    for run in runs:
        if (m := NAME.match(run.name)) is None:
            raise SystemExit(f"{run.name}: not GROUP-sSEED")
        d = [run / "ground" / "score"]
        if not d[0].is_dir():  # without rollouts a run would read as not trained well
            raise SystemExit(f"{run}: no rollouts in ground/score")
        found[m["group"]].append(
            {
                "score": per_pair(score.episode_scores(d, REWARD)),
                "share": per_pair(score.episode_means(d, score.SHARE_LABELS, score.share)),
            }
        )
    groups = {
        g: {q: over_seeds([r[q] for r in rs]) for q in ("score", "share")}
        for g, rs in found.items()
    }
    result = compare(groups)
    seeds = {g: len(rs) for g, rs in found.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "versus_reference.json").write_text(
        json.dumps({"groups": result, "seeds": seeds}, indent=1) + "\n"
    )
    (out / "versus-reference.md").write_text(text(result, seeds))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
