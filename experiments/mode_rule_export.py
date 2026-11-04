"""Export the mode rule on a net1: a net2 with four numbers of memory that
yields to copies of itself and competes with a flow that fills the buffer.

Copies of the fixed rule fill the buffer as a Cubic flow does
(exp-share-copies-val-17), so the queue alone cannot tell them apart. The
mode rule's cooperative window follows its own share instead, so copies of
it keep the queue short and the buffer far from full; a full buffer then
means a flow that does not yield. From net1's mean beliefs, with s this
flow's share of the capacity:

    cooperative  share = (s + beta (1 - busy) + a) / (1 + a), at most 1
                 u = c + log2(share)
    compete      u = c + k * busy * (1 - s)   the share rule

Time is counted in units of the base RTT, at least ``unit_ms``. The rule
starts cooperating. An event, net1 believing the buffer fuller than ``phi``
(queue_pkts over queue_pkts plus free_buffer_pkts) or the queue longer than
``theta`` base RTTs, makes it compete for ``period`` units. Then it
cooperates and watches: from ``cool`` to ``cool + watch`` units later, a
mean believed fill (about one unit of memory) above ``phi_low`` is enough
to compete again. Copies compete and stop together, and the buffer then
empties, since all of them yield; a Cubic flow keeps it partly full. The
window falls by at most a factor 2^``fall`` per unit, so cooperating does
not collapse it in a deep queue.

Alone the cooperative window is the fixed rule's (share = 1 at s = 1).
Among n copies it settles with equal shares and the queue near
2^c (1 + n a) / (1 + a) - 1 base RTTs.

Usage: python experiments/mode_rule_export.py OUT NET1.pt --phi 0.6 0.8 --period 10 20
Writes OUT/mode-f{phi}-p{period}.onnx for each pair.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from expliccit_env.train import fit
from expliccit_env.train.export import to_onnx

ACTION_MAX = 3.0
# ponytail: the guest's step period, fixed at 10 ms; take it from the
# contract if steps ever come at another rate.
STEP_MS = 10.0
LCFS = (
    "receive_rate_bps",
    "capacity_bps",
    "busy_fraction",
    "queue_delay_mean_ns",
    "queue_pkts",
    "free_buffer_pkts",
    "one_way_delay_ns",
)


class ModeRule(nn.Module):
    """(net1's standardized means, (compete time left, time cooperating,
    mean fill, last u)) -> (u / ACTION_MAX, next state)."""

    state_dim = 4
    cols: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, net1: fit.Net1, p: argparse.Namespace, phi: float, period: float) -> None:
        super().__init__()
        sc = net1.lcf_scaler
        self.register_buffer("cols", sc.cols.clone())
        self.register_buffer("center", sc.center.clone())
        self.register_buffer("spread", sc.spread.clone())
        self.i = {n: list(net1.lcfs).index(n) for n in LCFS}
        self.p, self.phi, self.period = p, phi, period

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p, i, period = self.p, self.i, self.period
        b = x.clone()
        b[:, self.cols] = x[:, self.cols] * self.spread + self.center
        cap = torch.clamp(b[:, i["capacity_bps"]], min=0.1)
        s = torch.clamp(b[:, i["receive_rate_bps"]] / cap, 0.0, 1.0)
        busy = torch.clamp(b[:, i["busy_fraction"]], 0.0, 1.0)
        rtt = torch.clamp(2.0 * b[:, i["one_way_delay_ns"]], min=0.1)  # ms
        queue = torch.clamp(b[:, i["queue_delay_mean_ns"]], min=0.0) / rtt
        pkts = torch.clamp(b[:, i["queue_pkts"]], min=0.0)
        fill = pkts / torch.clamp(pkts + torch.clamp(b[:, i["free_buffer_pkts"]], min=0.0), min=1.0)
        event = (fill > self.phi) | (queue > p.theta)
        d = STEP_MS / torch.clamp(rtt, min=p.unit_ms)
        competing = state[:, 0] > 0
        r = torch.clamp(state[:, 0] - d, min=0.0)
        w = torch.where(competing, 0 * d, torch.clamp(state[:, 1] + d, max=p.cool + p.watch + 1))
        f = state[:, 2] + torch.clamp(d, max=1.0) * (fill - state[:, 2])
        watching = (w >= p.cool) & (w < p.cool + p.watch) & (f > p.phi_low)
        start = ~competing & (w >= p.cool) & (event | watching)
        r = torch.where(start, torch.full_like(d, period), r)
        share = torch.clamp((s + p.beta * (1.0 - busy) + p.a) / (1.0 + p.a), 1e-3, 1.0)
        u = torch.where(r > 0, p.c + p.k * busy * (1.0 - s), p.c + torch.log2(share))
        u = torch.maximum(u, state[:, 3] - p.fall * d)
        nxt = torch.stack([r, w, f, u], dim=1)
        return torch.clamp(u / ACTION_MAX, -1.0, 1.0).unsqueeze(1), nxt


def main() -> None:
    q = argparse.ArgumentParser()
    q.add_argument("out", type=Path)
    q.add_argument("net1", type=Path)
    q.add_argument("--phi", type=float, nargs="+", required=True, help="believed buffer fill")
    q.add_argument("--period", type=float, nargs="+", required=True, help="units")
    q.add_argument("--theta", type=float, default=1.0, help="queue / base RTT")
    q.add_argument("--c", type=float, default=0.125, help="the fixed rule's u")
    q.add_argument("--k", type=float, default=1.0, help="the share rule's k")
    q.add_argument("--a", type=float, default=0.05)
    q.add_argument("--beta", type=float, default=0.3, help="gain on the free capacity")
    q.add_argument("--cool", type=float, default=2.0, help="units")
    q.add_argument("--watch", type=float, default=10.0, help="units")
    q.add_argument("--phi-low", type=float, default=0.35, help="mean fill while watching")
    q.add_argument("--fall", type=float, default=1.0, help="log2 per unit")
    q.add_argument("--unit-ms", type=float, default=100.0)
    p = q.parse_args()
    net1, _ = fit.load_net1(p.net1)
    p.out.mkdir(parents=True, exist_ok=True)
    for phi in p.phi:
        for period in p.period:
            rule = ModeRule(net1, p, phi, period).eval()
            graph = fit.policy_graph(net1, rule, ACTION_MAX, action="bdp", view="means")
            name = f"mode-f{phi:g}-p{period:g}"
            (p.out / f"{name}.onnx").write_bytes(to_onnx(graph))
            print(name)


if __name__ == "__main__":
    main()
