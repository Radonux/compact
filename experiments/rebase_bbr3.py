"""Training runs' evaluations against BBRv3, from evaluations made against the rule.

A run made before PPOSpec.reference_cc pairs each evaluation with the
fixed rule only (history.json, minus_rule_by_pair). The rule's and
BBRv3's own rollouts on the same validation episodes (experiments/
ground.py rollouts, as in experiments/versus-bbr3.json) turn each pair's
difference from the rule into its difference from BBRv3: the policy's
score is the rule's plus the difference. Per evaluation, the difference
from BBRv3 over all conditions, and without and with a competing flow
with a bootstrap 95% interval over links and the verdict of
versus_reference. A run's best evaluation is the one highest over all
conditions (docs/methodology.md, comparing controllers).

Usage: python experiments/rebase_bbr3.py OUT --rule RULE_RUN --bbr3 BBR3_RUN RUN_DIR...
Writes OUT/rebase_bbr3.json and OUT/rebase-bbr3.md.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from versus_reference import Pair, by_link, interval, per_pair

from expliccit_env.train import score
from expliccit_env.train.trajectories import REWARD


def own(run: Path) -> dict[Pair, float]:
    """A run's scores per link and condition on its ground/score rollouts."""
    return per_pair(score.episode_scores([run / "ground" / "score"], REWARD))


def rebased(
    ev: dict[str, Any], rule: dict[Pair, float], bbr3: dict[Pair, float]
) -> dict[Pair, float]:
    """One evaluation's differences from BBRv3, per link and condition."""
    out = {}
    for link, cond, d in ev["minus_rule_by_pair"]:
        if (link, cond) in rule and (link, cond) in bbr3:
            out[(link, cond)] = d + rule[(link, cond)] - bbr3[(link, cond)]
    return out


def summary(diff: dict[Pair, float], rng: np.random.Generator) -> dict[str, Any]:
    row: dict[str, Any] = {"all": statistics.fmean(diff.values()), "pairs": len(diff)}
    for kind, want in (("alone", False), ("competing", True)):
        links = by_link(diff, want)
        if links:
            row[kind] = interval(list(links.values()), rng)
    row["trained_well"] = all(
        row.get(k, {}).get("verdict") == "within" for k in ("alone", "competing")
    )
    return row


def main(out: Path, rule_run: Path, bbr3_run: Path, runs: list[Path]) -> None:
    rule, bbr3 = own(rule_run), own(bbr3_run)
    rng = np.random.default_rng(0)
    result: dict[str, Any] = {}
    for run in runs:
        history = json.loads((run / "history.json").read_text())
        evals = []
        for h in history:
            ev = h.get("eval")
            if ev and ev.get("minus_rule_by_pair"):
                diff = rebased(ev, rule, bbr3)
                if diff:
                    evals.append({"iteration": h["iteration"]} | summary(diff, rng))
        if evals:
            best = max(evals, key=lambda e: e["all"])
            result[run.name] = {
                "start": evals[0],
                "best": best,
                "last": evals[-1],
                "evaluations": evals,
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rebase_bbr3.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "rebase-bbr3.md").write_text(text(result))


def text(result: dict[str, Any]) -> str:
    def cell(e: dict[str, Any], kind: str) -> str:
        if kind not in e:
            return ""
        lo, hi = e[kind]["ci95"]
        return f"{e[kind]['mean']:+.3f} ({lo:+.2f} to {hi:+.2f})"

    lines = [
        "Log utility minus BBRv3's on the validation episodes, per run: the start "
        "(iteration -1), the best evaluation over all conditions and the last.\n",
        "| run | start | best iteration | best, all | best, alone | best, with a competing flow "
        "| trained well | last, all |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(result.items()):
        b = r["best"]
        lines.append(
            f"| {name} | {r['start']['all']:+.3f} | {b['iteration']} | {b['all']:+.3f} "
            f"| {cell(b, 'alone')} | {cell(b, 'competing')} "
            f"| {'yes' if b['trained_well'] else 'no'} | {r['last']['all']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--rule", type=Path, required=True)
    p.add_argument("--bbr3", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    main(a.out, a.rule, a.bbr3, a.runs)
