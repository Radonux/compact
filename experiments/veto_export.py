"""Export a net2 that pushes as a learned policy does unless it believes
copies of itself share the link: u = u_rule + (1 - q')(u_policy - u_rule),
where u_policy is a trained u student's (PPO) and q' is the mean copies
belief of trained "gate3" students past --tq (q' = clamp((q - t) / (1 - t),
0, 1)), as round 7v's veto did for a gate3 student's own push.

Usage: python experiments/veto_export.py OUT NAME --policy RUN_DIR --q RUN_DIR... [--tq 0]
Each RUN_DIR is an `expliccit train ppo` output (ppo_spec.json, net1.pt,
actor.pt) on the same net1 and inputs; writes OUT/NAME.onnx.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from expliccit_env.train import fit, ppo
from expliccit_env.train.export import to_onnx
from expliccit_env.train.recurrent import ConcatActor, EnsembleActor, RecurrentActor


class Veto(nn.Module):
    """A Held decision's head (train/recurrent.py): the concatenated outputs
    [u / max, 2p - 1, 2q - 1] -> u."""

    def __init__(self, u_rule: float, action_max: float, tq: float) -> None:
        super().__init__()
        self.u_rule, self.action_max, self.tq = u_rule, action_max, tq

    def forward(
        self, a: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None
    ) -> torch.Tensor:
        del z, traffic
        q = (a[:, 2] + 1) / 2
        q = torch.clamp((q - self.tq) / (1 - self.tq), 0.0, 1.0) if self.tq > 0 else q
        return self.u_rule + (1 - q) * (a[:, 0] * self.action_max - self.u_rule)


def _actor(run: Path, width: int, spec: ppo.PPOSpec, outputs: int) -> RecurrentActor:
    actor = RecurrentActor(width, spec.hidden, memory=spec.memory, outputs=outputs)
    actor.load_state_dict(torch.load(run / "actor.pt", weights_only=True))
    return actor


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("name")
    a.add_argument("--policy", type=Path, required=True, help="a u student's run")
    a.add_argument("--q", type=Path, nargs="+", required=True, help="gate3 students' runs")
    a.add_argument("--tq", type=float, default=0.0)
    args = a.parse_args()
    spec = ppo.PPOSpec.parse((args.policy / "ppo_spec.json").read_text())
    qspecs = [ppo.PPOSpec.parse((r / "ppo_spec.json").read_text()) for r in args.q]
    if spec.student != "u" or spec.base != "fixed" or any(s.student != "gate3" for s in qspecs):
        raise SystemExit("--policy must be a u student on the fixed base, --q gate3 students")
    keys = ("u_rule", "action_max", "view", "input_bins", "hidden", "memory")
    if any(getattr(s, k) != getattr(spec, k) for s in qspecs for k in keys):
        raise SystemExit(f"the runs must agree on {', '.join(keys)}")
    net1, _ = fit.load_net1(args.policy / "net1.pt")
    width = ppo._width(net1, spec)  # pyright: ignore[reportPrivateUsage]
    beliefs = EnsembleActor([_actor(r, width, s, 2) for r, s in zip(args.q, qspecs, strict=True)])
    actor = ConcatActor([_actor(args.policy, width, spec, 1), beliefs])
    veto = Veto(spec.u_rule, spec.action_max, args.tq)
    graph = ppo.policy_graph(net1, actor, spec, head=veto)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.onnx").write_bytes(to_onnx(graph))
    print(f"{args.name}.onnx")


if __name__ == "__main__":
    main()
