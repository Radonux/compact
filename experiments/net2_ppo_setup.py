"""Everything the cells of an `expliccit train ppo` experiment share (PPO,
TD3+BC or DAgger): the fixed rule on a net1 (the start, and the baseline
every evaluation is paired with), the training paths, the validation set
evaluated during training, and a base PPOSpec each cell edits.

Usage: python experiments/net2_ppo_setup.py OUT --net1 NET1.pt --models DIR --u 0.25
       [--paths net2-train-xl]

Training paths: examples/net1_broad.py's net2-train set (600 links of their
own under 64 broad conditions), or with --paths net2-train-xl 9,600; the
learner over-weights competition on top (PPOSpec.compete). Evaluation
during training: the "new" family of experiments/heldout.py on the
validation links (Cubic from 0 s, BBRv1 from 4 s, two Cubic flows, 0.1% and
3% loss, ACK-path loss), paired by link and condition with the rule; the
choice between runs is made afterwards on all validation families.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--net1", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True, help="release model-net1v3")
    p.add_argument("--u", type=float, required=True, help="the rule's u, the starting point")
    p.add_argument("--paths", choices=["net2-train", "net2-train-xl"], default="net2-train")
    a = p.parse_args()
    out = a.out.resolve()
    run("experiments/rule_export.py", str(out / "rule"), str(a.net1), "--u", f"{a.u:g}")
    run(
        "examples/net1_broad.py",
        str(out / "paths"),
        "--set",
        a.paths,
        "--models",
        str(a.models),
    )
    run(
        "experiments/heldout.py", str(out / "eval"), "--split", "val", "--family", "new",
        "--duration-ms", "30000", "--seed", "11", "--stock", "bbr3",
    )  # fmt: skip
    spec = {
        "paths_from": str(out / "paths" / "campaign.json"),
        "eval_from": str(out / "eval" / "campaign.json"),
        "net1_from": str(a.net1.resolve()),
        "rule_policy": str(out / "rule" / f"rule-u{a.u:g}.onnx"),
        "u_rule": a.u,
        "input_bins": 32,
        "action_max": 3.0,
        "iterations": 10,
        "links_per_iteration": 8,
        "group": 4,
        "duration_ms": 30_000,
        "noise_sigma": 0.1,
        "tau_s": 3.0,
        "eval_every": 5,
        "keep_rollouts": False,
        "time_budget_s": 5 * 3600 + 15 * 60,
    }
    (out / "ppo-base.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
