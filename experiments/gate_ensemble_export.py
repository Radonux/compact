"""Export several trained "gate" net2s (train/ppo.py, student "gate") as one
policy whose belief is the mean of theirs (recurrent.EnsembleActor), with
the switch sharpened at t as in gate_threshold_export.py. Seeds of one
setting differ mostly in where they believe a stock TCP flow is present
when none is, so their mean belief should cross t less often among copies.

Usage: python experiments/gate_ensemble_export.py OUT NAME RUN_DIR... --t 0.5
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
    args = a.parse_args()
    specs = [ppo.PPOSpec.parse((r / "ppo_spec.json").read_text()) for r in args.runs]
    same = {s.model_copy(update={"seed": 0, "init_actor": None}).model_dump_json() for s in specs}
    if len(same) != 1 or specs[0].student != "gate":
        raise SystemExit("the runs must be gate students of one setting, differing only in seed")
    spec = specs[0].model_copy(update={"gate_threshold": args.t})
    net1, _ = fit.load_net1(args.runs[0] / "net1.pt")
    width = ppo._width(net1, spec)  # pyright: ignore[reportPrivateUsage]
    members: list[RecurrentActor] = []
    for r in args.runs:
        actor = RecurrentActor(width, spec.hidden, memory=spec.memory)
        actor.load_state_dict(torch.load(r / "actor.pt", weights_only=True))
        members.append(actor)
    args.out.mkdir(parents=True, exist_ok=True)
    graph = ppo.policy_graph(net1, EnsembleActor(members), spec)
    (args.out / f"{args.name}.onnx").write_bytes(to_onnx(graph))
    print(f"{args.name}.onnx")


if __name__ == "__main__":
    main()
