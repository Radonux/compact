"""Rank the net1 runs of a grid (one experiments/net1_train.py output each).

Usage: python experiments/net1_grid.py OUT_DIR RUN_DIR...

Ranks by mean CRPS skill over the LCFs on the validation rows, the only
rows a choice may use, and shows each run's 90% coverage error there; then,
for the chosen run only, its test, held-out-controller and held-out-family
views next to the comparison net1. Writes OUT_DIR/summary.md.
"""

import json
import sys
from pathlib import Path
from typing import Any


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main(out: Path, runs: list[Path]) -> None:
    reps = {r.name: json.loads((r / "report.json").read_text()) for r in runs}

    def val(rep: dict[str, Any], key: str) -> float:
        v = rep["views"]["val"]["net1"]
        return mean([x[key] for x in v.values()])

    ranked = sorted(reps.items(), key=lambda kv: -val(kv[1], "skill"))
    head = "| run | dropout | width | CRPS skill | mean abs(coverage at 90% - 0.9) |"
    lines = ["Grid, ranked on the validation rows:", "", head, "|---|---|---|---|---|"]
    for name, rep in ranked:
        s = rep["spec"]
        err = mean([abs(x["c90"] - 0.9) for x in rep["views"]["val"]["net1"].values()])
        lines.append(
            f"| {name} | {s['dropout']} | {s['width']} | {val(rep, 'skill'):.3f} | {err:.3f} |"
        )
    best, rep = ranked[0]
    lcfs = list(rep["spec"]["lcfs"])
    note = f"Chosen: {best}. 90% coverage / R^2 per LCF; compare is the net1 given to --compare."
    lines += ["", note, "", "| view | model | " + " | ".join(lcfs) + " |"]
    lines.append("|" + "---|" * (len(lcfs) + 2))
    for view, models in rep["views"].items():
        for model, s in models.items():
            cells = [f"{s[k]['c90']:.2f} / {s[k]['r2']:.2f}" if k in s else "" for k in lcfs]
            who = "chosen" if model == "net1" else "compare"
            lines.append(f"| {view} | {who} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "chosen.txt").write_text(best + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
