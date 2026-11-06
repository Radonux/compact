"""Everything the cells of an `expliccit train online` experiment share:
the fixed rule's actor on a net1 as the starting point, the training paths,
the validation set evaluated during training, and a base OnlineSpec each
cell edits.

Usage: python experiments/net2_online_setup.py OUT --net1 NET1.pt --models DIR --u 0.125

Training paths: examples/net1_broad.py's net2-train set (600 links of their
own under 64 broad conditions, 30 s). Evaluation during training: the "new"
family of experiments/heldout.py on the validation links (Cubic from 0 s,
BBRv1 from 4 s, two Cubic flows, 0.1% and 3% loss, ACK-path loss), a
diagnostic only; the choice between runs is made afterwards on all five
validation families.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# TD3+BC with 40-step returns: with one-step returns, online TD3+BC on net1
# v3 drifted below its start (docs/results.md). alpha and the exploration
# width are each cell's.
NET2 = {
    "algo": "td3bc",
    "view": "bins",
    "input_bins": 32,
    "action": "bdp",
    "hidden": 256,
    "reward": "log_utility",
    "delay_weight": 0.5,
    "loss_weight": 4.0,
    "action_max": 3.0,
    "discount": 0.97,
    "n_step": 40,
    "reward_scale": 0.1,
    "alpha": 2.5,
    "batch_size": 512,
    "lr": 3e-4,
    "policy_noise": 0.05,
    "noise_clip": 0.1,
}


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--net1", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True, help="release model-net1v3")
    p.add_argument("--u", type=float, required=True, help="the rule's u, the starting point")
    a = p.parse_args()
    out = a.out.resolve()
    run("experiments/rule_export.py", str(out / "init"), str(a.net1), "--u", f"{a.u:g}")
    run(
        "examples/net1_broad.py",
        str(out / "paths"),
        "--set",
        "net2-train",
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
        "init_from": str(out / "init" / f"rule-u{a.u:g}-actor.pt"),
        "net2": NET2,
        "iterations": 40,
        "episodes_per_iteration": 32,
        "duration_ms": 30_000,
        "noise_sigma": 0.5,
        "noise_hold_steps": 20,
        "updates_per_iteration": 2000,
        "eval_every": 10,
        "replay_capacity": 1_000_000,
        "keep_rollouts": False,
        "time_budget_s": 5 * 3600 + 30 * 60,
        "eval_initial": True,
    }
    (out / "online-base.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
