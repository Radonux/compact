"""Export several trained net2s of one setting (train/ppo.py) as one
policy on the mean of their outputs (recurrent.EnsembleActor): for gate
students ("gate", "gate3") the mean belief, with the switch sharpened at
--t as in gate_threshold_export.py (and for "gate3", q at --tq and the
copies policy --copies); for students that learn u (PPO, or DAgger "u"),
the mean u. Seeds of one setting misfire on different links, so their
mean should misfire less often.

Usage: python experiments/gate_ensemble_export.py OUT NAME RUN_DIR... [--t 0.5]
       [--tq 0] [--copies rule]
Each RUN_DIR is an `expliccit train ppo` output (ppo_spec.json, net1.pt,
actor.pt) of the same setting but its seed; writes OUT/NAME.onnx.

--net1 NET1.pt runs the actors on another net1 than the one they were
trained on, unchanged (no leakage, H4; repair without retraining net2,
H5). It must give the same outputs in the same bins: the same LCFs, bin
boundaries and input scaler.
"""

import argparse
from pathlib import Path

import torch

from expliccit_env.train import fit, ppo
from expliccit_env.train.export import to_onnx
from expliccit_env.train.recurrent import EnsembleActor, RecurrentActor


def _same_outputs(a: fit.Net1, b: fit.Net1) -> bool:
    """Whether b's output means what a's does: the same LCFs, bin
    boundaries and input scaler (seeds of one net1 training run)."""
    bounds = [getattr(n.net, "bounds", None) for n in (a, b)]
    return (
        a.lcfs == b.lcfs
        and bounds[0] is not None
        and bounds[1] is not None
        and torch.equal(bounds[0], bounds[1])
        and all(
            torch.equal(getattr(a.scaler, k), getattr(b.scaler, k))
            for k in ("cols", "lo", "hi", "center", "spread")
        )
    )


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("name")
    a.add_argument("runs", type=Path, nargs="+")
    a.add_argument("--t", type=float, default=0.0, help="gate students' threshold")
    a.add_argument("--tq", type=float)
    a.add_argument("--copies", choices=["coop", "rule"])
    a.add_argument("--net1", type=Path, help="run the actors on this net1 instead")
    args = a.parse_args()
    specs = [ppo.PPOSpec.parse((r / "ppo_spec.json").read_text()) for r in args.runs]
    free = {"seed": 0, "init_actor": None, "init_critic": None}
    same = {s.model_copy(update=free).model_dump_json() for s in specs}
    if len(same) != 1:
        raise SystemExit("the runs must be of one setting, differing only in seed and start")
    gate = specs[0].student in ("gate", "gate3")
    update: dict[str, object] = (
        {"gate_threshold": args.t, "gate_threshold_q": args.tq} if gate else {}
    )
    if args.copies is not None:
        update["gate3_copies"] = args.copies
    spec = specs[0].model_copy(update=update)
    net1, _ = fit.load_net1(args.runs[0] / "net1.pt")
    if args.net1 is not None:
        other, _ = fit.load_net1(args.net1)
        if not _same_outputs(net1, other):
            raise SystemExit(f"{args.net1}: not the same LCFs, bins and input scaler")
        net1 = other
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
