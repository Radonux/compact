"""Export a trained gate net2 (train/ppo.py, student "gate" or "gate3")
with its switch sharpened: the policy pushes toward the share rule only
past a belief t in a stock TCP flow, p' = clamp((p - t) / (1 - t), 0, 1).
Copies lose more per unit of push than a flow against Cubic gains, so the
best push under an uncertain belief is below p. Threshold 0 is the policy
without sharpening. A "gate3" student's belief q in copies takes its own
threshold from --tq, or t without it.

Usage: python experiments/gate_threshold_export.py OUT RUN_DIR... --t 0.3 0.5 [--tq 0.5]
Each RUN_DIR is an `expliccit train ppo` output (ppo_spec.json, net1.pt,
actor.pt); writes OUT/<RUN_DIR name with -t<t> before its seed>.onnx, so
ppo-r5gate-k1-s2 at t = 0.5 becomes ppo-r5gate-k1-t0.5-s2.onnx, and with
--tq 0.6, ppo-r5gate-k1-t0.5q0.6-s2.onnx.
"""

import argparse
import re
from pathlib import Path

import torch

from expliccit_env.train import fit, ppo
from expliccit_env.train.export import to_onnx
from expliccit_env.train.recurrent import RecurrentActor

SEED = re.compile(r"^(?P<cand>.+)-s(?P<seed>[0-9]+)$")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("runs", type=Path, nargs="+")
    a.add_argument("--t", type=float, nargs="+", required=True)
    a.add_argument("--tq", type=float, nargs="+", default=[None])
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        spec = ppo.PPOSpec.parse((run / "ppo_spec.json").read_text())
        if spec.student not in ("gate", "gate3"):
            raise SystemExit(f"{run}: not a gate student")
        m = SEED.match(run.name)
        if m is None:
            raise SystemExit(f"{run}: name does not end in -s<seed>")
        net1, _ = fit.load_net1(run / "net1.pt")
        width = ppo._width(net1, spec)  # pyright: ignore[reportPrivateUsage]
        outputs = 2 if spec.student == "gate3" else 1
        actor = RecurrentActor(width, spec.hidden, memory=spec.memory, outputs=outputs)
        actor.load_state_dict(torch.load(run / "actor.pt", weights_only=True))
        for t in args.t:
            for tq in args.tq:
                update = {"gate_threshold": t, "gate_threshold_q": tq}
                graph = ppo.policy_graph(net1, actor, spec.model_copy(update=update))
                q = "" if tq is None else f"q{tq:g}"
                name = f"{m['cand']}-t{t:g}{q}-s{m['seed']}.onnx"
                (args.out / name).write_bytes(to_onnx(graph))
                print(name)


if __name__ == "__main__":
    main()
