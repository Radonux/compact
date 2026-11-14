"""The width curve of discovery controllers, and its knee by a fixed rule.

Reads `expliccit train ppo` run directories whose names end in
`-kK-sSEED`; the rest of the name is the curve the run belongs to (one per
teacher). Per run it reports the closed-loop score of the policy the run
started from and of the one it left: the paired mean difference in log
utility from the fixed rule on the validation links (history.json). It also
reports how far training moved the policy from its start (the RMS
difference between the two policies' log2 windows on the last iteration's
rollouts, from PPO's kl_to_start) and, where RUN/probe/probe.json exists
(experiments/ground.py probe), the R^2 of the decision regressed on the
latents on held-out rollouts.

The knee is the one-standard-error rule: the smallest width whose mean
score over seeds is within one standard error of the best width's mean. The
standard error is pooled over widths: the root of the mean within-width
variance over seeds, divided by the root of the seeds per width. Width 0
(a reference without an encoder) is listed and takes no part in the knee.
A curve with fewer than two seeds at some width has no knee.

Writes OUT/width_curve.json and OUT/width-curve.md.

Usage: python experiments/width_curve.py OUT RUN_DIR...
"""

import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

NAME = re.compile(r"^(?P<curve>.+)-k(?P<k>\d+)-s(?P<seed>\d+)$")


def run_row(run: Path) -> dict[str, Any] | None:
    """One run's numbers, or None when it left no scored policy."""
    m = NAME.match(run.name)
    history = run / "history.json"
    if m is None or not history.exists():
        return None
    hist = json.loads(history.read_text())
    scored = [h for h in hist if h.get("eval")]
    if not scored or scored[-1] is not hist[-1] or not hist[-1]["eval"]["pairs"]:
        return None  # the policy the run left has no score
    spec = json.loads((run / "ppo_spec.json").read_text())
    learner = hist[-1].get("learner") or {}
    kl = learner.get("kl_to_start")
    probe = run / "probe" / "probe.json"
    decision = json.loads(probe.read_text())["runs"][run.name] if probe.exists() else {}
    return {
        "curve": m["curve"],
        "k": int(m["k"]),
        "seed": int(m["seed"]),
        "iterations": hist[-1]["iteration"] + 1,
        "start": scored[0]["eval"]["minus_rule"] if scored[0]["iteration"] == -1 else math.nan,
        "end": hist[-1]["eval"]["minus_rule"],
        # kl_to_start is the mean of (mu - mu_start)^2 over 2 sigma^2.
        "moved": math.nan if kl is None else math.sqrt(2 * kl) * spec["noise_sigma"],
        "saturated": learner.get("u_saturated", math.nan),
        "decision_linear": decision.get("decision_linear", math.nan),
        "decision_quadratic": decision.get("decision_quadratic", math.nan),
    }


def knee(scores: dict[int, list[float]]) -> dict[str, Any]:
    """The one-standard-error knee of {width: scores over seeds}."""
    widths = sorted(k for k in scores if k > 0)
    means = {k: statistics.fmean(scores[k]) for k in widths}
    if not widths or min(len(scores[k]) for k in widths) < 2:
        return {"means": means, "knee": None, "standard_error": None, "best": None}
    seeds = min(len(scores[k]) for k in widths)
    se = math.sqrt(statistics.fmean(statistics.variance(scores[k]) for k in widths) / seeds)
    best = max(widths, key=lambda k: means[k])
    first = next(k for k in widths if means[k] >= means[best] - se)
    return {"means": means, "knee": first, "standard_error": se, "best": best}


def curves(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in sorted({r["curve"] for r in rows}):
        scores: dict[int, list[float]] = {}
        for r in rows:
            if r["curve"] == name:
                scores.setdefault(r["k"], []).append(r["end"])
        reference = scores.get(0)
        out[name] = knee(scores) | {
            "reference": None if not reference else statistics.fmean(reference),
            "seeds": {k: len(v) for k, v in sorted(scores.items())},
        }
    return out


def text(rows: list[dict[str, Any]], result: dict[str, dict[str, Any]], missing: list[str]) -> str:
    lines = [
        "Scores are the mean difference in log utility from the fixed rule,",
        "paired by link and condition, on the validation links. `moved` is the",
        "RMS difference in log2 of the window between a run's last policy and",
        "its start. The knee is the smallest width within one pooled standard",
        "error of the best width. It is not a count of concepts: one latent",
        "can carry the decision itself (docs/methodology.md).",
        "",
    ]
    for name, c in result.items():
        lines += [f"## {name}", ""]
        if c["knee"] is None:
            lines += ["No knee: some width has fewer than two scored seeds.", ""]
        else:
            lines += [
                f"Knee: width {c['knee']}. Best width {c['best']}, pooled standard error "
                f"{c['standard_error']:.3f}.",
                "",
            ]
        if c["reference"] is not None:
            lines += [f"The reference without an encoder: {c['reference']:+.3f}.", ""]
        lines += ["| width | seeds | mean score |", "|---|---|---|"]
        lines += [f"| {k} | {c['seeds'][k]} | {v:+.3f} |" for k, v in c["means"].items()]
        lines += [
            "",
            "| run | iterations | start | end | moved | saturated | decision R^2, linear "
            "| quadratic |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| k{r['k']} s{r['seed']} | {r['iterations']} | {r['start']:+.3f} | {r['end']:+.3f} "
            f"| {r['moved']:.3f} | {r['saturated']:.3f} | {r['decision_linear']:.3f} "
            f"| {r['decision_quadratic']:.3f} |"
            for r in sorted(rows, key=lambda r: (r["k"], r["seed"]))
            if r["curve"] == name
        ]
        lines.append("")
    if missing:
        lines += ["Runs that left no scored policy, and count for nothing:", ""]
        lines += [f"- {m}" for m in missing] + [""]
    return "\n".join(lines)


def main(out: Path, runs: list[Path]) -> None:
    found = [(r.name, run_row(r)) for r in runs]
    rows = [row for _, row in found if row is not None]
    missing = [name for name, row in found if row is None]
    result = curves(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "width_curve.json").write_text(
        json.dumps({"curves": result, "runs": rows, "missing": missing}, indent=1) + "\n"
    )
    (out / "width-curve.md").write_text(text(rows, result, missing))
    print(text(rows, result, missing))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
