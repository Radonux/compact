"""Summarize online net2 runs (one `expliccit train online` output each):
the evaluation log utility at each checkpoint and the training diagnostics
at the last iteration. A diagnostic, not the choice between runs, which is
made on all five validation families afterwards.

Usage: python experiments/net2_online_report.py OUT RUN_DIR...
"""

import json
import sys
from pathlib import Path

DIAGNOSTICS = ("critic", "q", "actor", "bc", "pi_mean", "pi_std", "sat", "ood")


def main(out: Path, runs: list[Path]) -> None:
    lines = [
        "Online net2 on net1 v4. Evaluation: mean log utility of the noise-free",
        "policy on the validation links' new-conditions family, at each checkpoint.",
        "Diagnostics: the learner's last-iteration means (q and actor are the",
        "critic's value scale; pi_mean and pi_std the actions taken; sat the",
        "share of saturated actions; ood the share outside the data).",
        "",
    ]
    for r in runs:
        hist = json.loads((r / "history.json").read_text())
        spec = json.loads((r / "online_spec.json").read_text())
        evals = [(h["iteration"], h["eval"]["log_utility"]) for h in hist if "eval" in h]
        hist = [h for h in hist if h["iteration"] >= 0]
        last = hist[-1]["losses"] if hist else {}
        wall = sum(h.get("wall_s", 0.0) for h in hist)
        lines += [
            f"## {r.name}: alpha {spec['net2']['alpha']}, exploration {spec['noise_sigma']}"
            f" held {spec['noise_hold_steps']} steps",
            "",
            f"{len(hist)} of {spec['iterations']} iterations in {wall / 3600:.1f} h; mean"
            f" reward of the last rollouts {hist[-1]['mean_reward_new']:.3f}."
            if hist
            else "No iteration.",
            "",
            "| checkpoint | log utility |",
            "|---|---|",
            *(
                f"| {'start (the rule)' if i < 0 else f'iteration {i + 1}'} | {v:+.3f} |"
                for i, v in evals
            ),
            "",
            "| " + " | ".join(DIAGNOSTICS) + " |",
            "|" + "---|" * len(DIAGNOSTICS),
            "| " + " | ".join(f"{last.get(k, float('nan')):.3f}" for k in DIAGNOSTICS) + " |",
            "",
        ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
