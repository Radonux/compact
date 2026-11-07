"""Export several trained gate net2s (train/ppo.py, student "gate" or
"gate3") as one policy whose beliefs are the mean of theirs
(recurrent.EnsembleActor), with the switch sharpened at t as in
gate_threshold_export.py (and for "gate3", q at --tq and the copies policy
--copies). Seeds of one setting misfire on different links, so their mean
belief should cross a threshold less often where it should not.

Usage: python experiments/gate_ensemble_export.py OUT NAME RUN_DIR... --t 0.5
       [--tq 0] [--copies rule]
Each RUN_DIR is an `expliccit train ppo` output (ppo_spec.json, net1.pt,
actor.pt) of the same setting but its seed; writes OUT/NAME.onnx.
"""

import argparse
from pathlib import Path

import torch

from expliccit_env.train import fit, ppo
from expliccit_env.train.export import to_onnx
from expliccit_env.train.recurrent import EnsembleActor, RecurrentActor


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("name")
    a.add_argument("runs", type=Path, nargs="+")
    a.add_argument("--t", type=float, required=True)
    a.add_argument("--tq", type=float)
    a.add_argument("--copies", choices=["coop", "rule"])
    args = a.parse_args()
    specs = [ppo.PPOSpec.parse((r / "ppo_spec.json").read_text()) for r in args.runs]
    same = {s.model_copy(update={"seed": 0, "init_actor": None}).model_dump_json() for s in specs}
    if len(same) != 1 or specs[0].student not in ("gate", "gate3"):
        raise SystemExit("the runs must be gate students of one setting, differing only in seed")
    update: dict[str, object] = {"gate_threshold": args.t, "gate_threshold_q": args.tq}
    if args.copies is not None:
        update["gate3_copies"] = args.copies
    spec = specs[0].model_copy(update=update)
    net1, _ = fit.load_net1(args.runs[0] / "net1.pt")
    width = ppo._width(net1, spec)  # pyright: ignore[reportPrivateUsage]
    members: list[RecurrentActor] = []
    for r in args.runs:
        outputs = 2 if spec.student == "gate3" else 1
        actor = RecurrentActor(width, spec.hidden, memory=spec.memory, outputs=outputs)
        actor.load_state_dict(torch.load(r / "actor.pt", weights_only=True))
        members.append(actor)
    args.out.mkdir(parents=True, exist_ok=True)
    graph = ppo.policy_graph(net1, EnsembleActor(members), spec)
    (args.out / f"{args.name}.onnx").write_bytes(to_onnx(graph))
    print(f"{args.name}.onnx")


if __name__ == "__main__":
    main()
