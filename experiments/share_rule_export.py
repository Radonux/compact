"""Export the share rule on a net1: the fixed rule's window, 2^u times the
believed BDP, with u raised when net1 believes the link is busy and this
flow gets a small share of it.

    u = u0 + k * busy * max(0, 1 - receive_rate / capacity)

from net1's mean beliefs (busy fraction, this flow's receive rate, the
capacity). With k = 0 it is the fixed rule. It acts on net1's output
alone, like any net2, and has two numbers to choose, so it can be searched
directly in closed loop on the validation links.

Usage: python experiments/share_rule_export.py OUT NET1.pt --u0 0.125 --k 0 0.5 1 2 [--oracle]
Writes OUT/share-u{u0}-k{k}.onnx for each k; --oracle writes
OUT/share-u{u0}-k{k}-oracle.onnx instead: the same rule on the true LCFs,
read from the guest's live truth into net1's bins (each mean is then the
true value's bin centre).

--flows N gives the rule the number of flows on the bottleneck, which is
not among the LCFs: the window aims at 1/N of the believed BDP, and the
rule pushes only while this flow gets less than 1/N of the capacity,

    u = u0 - log2(N) + k * busy * max(0, 1 - N * receive_rate / capacity)

written to OUT/share-u{u0}-k{k}-n{N}[-oracle].onnx. Without it the graph is
the one above, byte for byte.
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from expliccit_env.train import fit
from expliccit_env.train.export import to_onnx

ACTION_MAX = 3.0


class ShareRule(nn.Module):
    """net1's standardized means -> u / ACTION_MAX (the graph multiplies)."""

    cols: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, net1: fit.Net1, u0: float, k: float, flows: int | None = None) -> None:
        super().__init__()
        s = net1.lcf_scaler
        self.register_buffer("cols", s.cols.clone())
        self.register_buffer("center", s.center.clone())
        self.register_buffer("spread", s.spread.clone())
        lcfs = list(net1.lcfs)
        self.busy = lcfs.index("busy_fraction")
        self.recv = lcfs.index("receive_rate_bps")
        self.cap = lcfs.index("capacity_bps")
        self.u0, self.k, self.flows = u0, k, flows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phys = x.clone()
        phys[:, self.cols] = x[:, self.cols] * self.spread + self.center
        busy = torch.clamp(phys[:, self.busy], 0.0, 1.0)
        share = phys[:, self.recv] / torch.clamp(phys[:, self.cap], min=0.1)
        if self.flows is None:
            u = self.u0 + self.k * busy * torch.clamp(1.0 - share, 0.0, 1.0)
        else:
            u0 = self.u0 - math.log2(self.flows)
            u = u0 + self.k * busy * torch.clamp(1.0 - self.flows * share, 0.0, 1.0)
        return torch.clamp(u / ACTION_MAX, -1.0, 1.0).unsqueeze(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("--u0", type=float, required=True)
    p.add_argument("--k", type=float, nargs="+", required=True)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--flows", type=int, help="flows on the bottleneck, known to the rule")
    a = p.parse_args()
    net1, _ = fit.load_net1(a.net1)
    truth = frozenset(net1.lcfs) if a.oracle else frozenset[str]()
    a.out.mkdir(parents=True, exist_ok=True)
    for k in a.k:
        rule = ShareRule(net1, a.u0, k, a.flows).eval()
        graph = fit.policy_graph(net1, rule, ACTION_MAX, action="bdp", view="means", truth=truth)
        name = f"share-u{a.u0:g}-k{k:g}" + (f"-n{a.flows}" if a.flows else "")
        name += "-oracle" if a.oracle else ""
        (a.out / f"{name}.onnx").write_bytes(to_onnx(graph))
        print(name)


if __name__ == "__main__":
    main()
