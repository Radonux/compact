"""Summarize net2 training runs (one `expliccit train ppo` output each, by
PPO, TD3+BC or DAgger): at each evaluation, the noise-free policy's mean
log utility and its difference from the fixed rule paired by link and
condition (iteration -1 is the starting policy); and per iteration the
learner's diagnostics, a DAgger student's against its teacher. A
diagnostic, not the choice between runs, which the gate makes on all
validation families afterwards (experiments/net2_ppo_gate_report.py).

Usage: python experiments/net2_ppo_report.py OUT RUN_DIR...
"""

import json
import sys
from pathlib import Path

DIAGNOSTICS = (
    ("u_mean", "u mean"),
    ("u_std", "u spread"),
    ("u_saturated", "saturated"),
    ("kl_to_rule", "KL to rule"),
    ("approx_kl", "KL step"),
    ("clip_fraction", "clipped"),
    ("value_loss", "value loss"),
    ("explained_variance", "expl. var."),
    ("mean_block_reward", "block reward"),
)
# algo "dagger": the student against its teacher on each iteration's new
# rollouts, before it learns from them, and where the teacher pushes.
DAGGER = (
    ("u_mean", "u mean"),
    ("u_std", "u spread"),
    ("teacher_push_before", "teacher pushes"),
    ("teacher_u_where_push_before", "teacher's u there"),
    ("u_where_push_before", "u there"),
    ("u_where_rule_before", "u elsewhere"),
    ("teacher_gap_before", "gap to teacher"),
    ("p_where_tcp_before", "p where TCP"),
    ("p_elsewhere_before", "p elsewhere"),
    ("q_where_copies_before", "q where copies"),
    ("q_elsewhere_before", "q elsewhere"),
    ("p_where_copies_before", "p where copies"),
    ("imitation_loss", "imitation loss"),
    ("mean_block_reward", "block reward"),
)


def main(out: Path, runs: list[Path]) -> None:
    lines = [
        "net2 by a learner of train/ppo.py. Evaluation: the noise-free policy on",
        "the validation links' new-conditions family, mean log utility and minus the fixed rule",
        "(paired by link and condition; iteration -1 is the starting policy).",
        "",
    ]
    for r in runs:
        hist = json.loads((r / "history.json").read_text()) if (r / "history.json").exists() else []
        lines += [f"## {r.name}", ""]
        if not hist:
            lines += ["No history (the run failed before its first iteration).", ""]
            continue
        lines += [
            "| iteration | log utility | minus rule | pairs | worst condition |",
            "|---|---|---|---|---|",
        ]
        for h in hist:
            e = h.get("eval")
            if not e:
                continue
            by = e.get("minus_rule_by_condition") or {}
            worst = min(by.items(), key=lambda kv: kv[1]) if by else ("", float("nan"))
            lines.append(
                f"| {h['iteration']} | {e['log_utility']:+.3f} | {e['minus_rule']:+.3f} "
                f"| {e['pairs']} | {worst[0]} {worst[1]:+.3f} |"
            )
        logs = [h["learner"] for h in hist if "learner" in h]
        cols = DAGGER if logs and "teacher_gap" in logs[0] else DIAGNOSTICS
        lines += ["", "| iteration | " + " | ".join(n for _, n in cols) + " | wall (min) |"]
        lines.append("|---" * (len(cols) + 2) + "|")
        for h in hist:
            g = h.get("learner")
            if g is None:
                continue
            cells = " | ".join(f"{g.get(k, float('nan')):.3g}" for k, _ in cols)
            lines.append(f"| {h['iteration']} | {cells} | {h.get('wall_s', 0) / 60:.1f} |")
        last = next((h for h in reversed(hist) if "eval" in h), None)
        if last:
            lines += ["", "Minus the rule by condition at the last evaluation:", ""]
            for c, v in last["eval"]["minus_rule_by_condition"].items():
                lines.append(f"- {c}: {v:+.3f}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
