"""Two rules on a net1's mean beliefs, each a net2 that acts on net1's
output alone.

ShareRule, the share rule: the fixed rule's window, 2^u times the believed
BDP, with u raised when net1 believes the link is busy and this flow gets
a small share of it (experiments/share_rule_export.py documents its
variants). CoopRule: the mode rule's cooperative window, which yields to
copies of itself.
"""

import math

import torch
from torch import nn

from expliccit_env.train import fit

ACTION_MAX = 3.0


class ShareRule(nn.Module):
    """net1's standardized means -> u / ACTION_MAX (the graph multiplies)."""

    cols: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(
        self, net1: fit.Net1, u0: float, k: float, flows: int | None = None, used: bool = False
    ) -> None:
        super().__init__()
        s = net1.lcf_scaler
        self.register_buffer("cols", s.cols.clone())
        self.register_buffer("center", s.center.clone())
        self.register_buffer("spread", s.spread.clone())
        lcfs = list(net1.lcfs)
        self.busy = lcfs.index("busy_fraction")
        self.recv = lcfs.index("receive_rate_bps")
        self.cap = lcfs.index("capacity_bps")
        self.u0, self.k, self.flows, self.used = u0, k, flows, used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phys = x.clone()
        phys[:, self.cols] = x[:, self.cols] * self.spread + self.center
        busy = torch.clamp(phys[:, self.busy], 0.0, 1.0)
        share = phys[:, self.recv] / torch.clamp(phys[:, self.cap], min=0.1)
        if self.flows is None:
            u = self.u0 + self.k * busy * torch.clamp(1.0 - share, 0.0, 1.0)
        elif self.used:
            u0 = self.u0 - math.log2(self.flows)
            fair = self.flows * share / torch.clamp(busy, min=0.05)
            u = u0 + self.k * torch.clamp(1.0 - fair, 0.0, 1.0)
        else:
            u0 = self.u0 - math.log2(self.flows)
            u = u0 + self.k * busy * torch.clamp(1.0 - self.flows * share, 0.0, 1.0)
        return torch.clamp(u / ACTION_MAX, -1.0, 1.0).unsqueeze(1)


class CoopRule(nn.Module):
    """The mode rule's cooperative window (experiments/mode_rule_export.py),
    without its state, on net1's standardized means: u = u0 + log2(share),
    share = (s + beta (1 - busy) + a) / (1 + a) in [0.001, 1], s this flow's
    share of the capacity. Alone it is the fixed rule (share = 1 at s = 1);
    among copies each settles at its share with a short queue.
    -> u / ACTION_MAX, as ShareRule."""

    cols: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, net1: fit.Net1, u0: float, beta: float = 0.3, a: float = 0.05) -> None:
        super().__init__()
        sc = net1.lcf_scaler
        self.register_buffer("cols", sc.cols.clone())
        self.register_buffer("center", sc.center.clone())
        self.register_buffer("spread", sc.spread.clone())
        lcfs = list(net1.lcfs)
        self.busy = lcfs.index("busy_fraction")
        self.recv = lcfs.index("receive_rate_bps")
        self.cap = lcfs.index("capacity_bps")
        self.u0, self.beta, self.a = u0, beta, a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phys = x.clone()
        phys[:, self.cols] = x[:, self.cols] * self.spread + self.center
        busy = torch.clamp(phys[:, self.busy], 0.0, 1.0)
        s = torch.clamp(phys[:, self.recv] / torch.clamp(phys[:, self.cap], min=0.1), 0.0, 1.0)
        share = torch.clamp((s + self.beta * (1.0 - busy) + self.a) / (1.0 + self.a), 1e-3, 1.0)
        u = self.u0 + torch.log2(share)
        return torch.clamp(u / ACTION_MAX, -1.0, 1.0).unsqueeze(1)
