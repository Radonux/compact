"""What net1 reports, and every way the rest of the system reads it.

net1 states one of three kinds of belief about each LCF:

- ``Point``: one value, as expliCCit's encoder does;
- ``Interval``: a centre with a lower and an upper bound;
- ``Bins``: a probability distribution over fixed bins of the quantity.

Its output z is the beliefs of all K LCFs side by side, in net1's target
units (standardized by its LCF scaler) for the first two kinds, and as
probabilities for the third. A kind knows how wide z is and how to read
it: the means in report units (which the window target and the rules
read), the standardized means (net2's "means" view), z with chosen LCFs
replaced by the truth (an oracle or a patched net1), and z with a wrong
belief injected (R2's faults). Everything downstream goes through these
methods; supporting a new kind means writing one more subclass, and no
other code asks which kind it has (``of`` is the one place that does).

Each method is written with ops the guest's tract runs, since all of them
can end up in an exported policy.
"""

# torch's own annotations leave many signatures partly unknown; every other
# strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from expliccit_env.train.models import BinNet1, IntervalNet1, Scaler

_BIG = 3.0e38  # |x| < _BIG is false for NaN and both infinities (no IsNaN in tract)


class Beliefs(nn.Module):
    """How to read one kind of net1 output z [n, width]."""

    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, k: int, width: int, lcf_scaler: Scaler) -> None:
        super().__init__()
        self.k, self.width = k, width
        self.register_buffer("center", lcf_scaler.center.clone())
        self.register_buffer("spread", lcf_scaler.spread.clone())

    def means(self, z: torch.Tensor) -> torch.Tensor:
        """[n, K] each belief's mean, in report units."""
        raise NotImplementedError

    def scaled_means(self, z: torch.Tensor) -> torch.Tensor:
        """[n, K] the means standardized as net1's targets are: the single
        values a point net1 would hand net2."""
        return (self.means(z) - self.center) / self.spread

    def patch(self, z: torch.Tensor, y: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        """z with LCF j's belief replaced by certainty of y[:, j] (report
        units) where take[j] and y[:, j] is a number; elsewhere net1's own
        belief stays."""
        raise NotImplementedError

    def fault(
        self, scale: Sequence[float], value: Sequence[float], pinned: Sequence[bool]
    ) -> nn.Module:
        """A module z -> z with each LCF's belief multiplied by scale[j], or
        pinned to value[j] where pinned[j], in report units: what a wrong
        encoder head would report (R2's injected faults)."""
        raise NotImplementedError

    def _taken(self, y: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        return take.unsqueeze(0) & (torch.abs(y) < _BIG)


class Point(Beliefs):
    """One value per LCF, standardized: z [n, K]."""

    def __init__(self, k: int, lcf_scaler: Scaler) -> None:
        super().__init__(k, k, lcf_scaler)

    def means(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.spread + self.center

    def scaled_means(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def patch(self, z: torch.Tensor, y: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        return torch.where(self._taken(y, take), (y - self.center) / self.spread, z)

    def fault(
        self, scale: Sequence[float], value: Sequence[float], pinned: Sequence[bool]
    ) -> nn.Module:
        return _ScaleOrPin(self, scale, value, pinned, parts=1)


class Interval(Beliefs):
    """A centre and two bounds per LCF, standardized: z [n, 3K], the K
    centres, then the K lower bounds, then the K upper bounds."""

    def __init__(self, k: int, lcf_scaler: Scaler) -> None:
        super().__init__(k, 3 * k, lcf_scaler)

    def means(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.k] * self.spread + self.center

    def scaled_means(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.k]

    def bounds(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[n, K] lower and upper bounds in report units."""
        k = self.k
        return (
            z[:, k : 2 * k] * self.spread + self.center,
            z[:, 2 * k :] * self.spread + self.center,
        )

    def patch(self, z: torch.Tensor, y: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        t = self._taken(y, take).repeat(1, 3)
        return torch.where(t, ((y - self.center) / self.spread).repeat(1, 3), z)

    def fault(
        self, scale: Sequence[float], value: Sequence[float], pinned: Sequence[bool]
    ) -> nn.Module:
        return _ScaleOrPin(self, scale, value, pinned, parts=3)


class Bins(Beliefs):
    """A distribution over B fixed bins per LCF: z [n, K * B], LCF j's
    probabilities in z[:, j * B : (j + 1) * B]. ``bounds`` [K, B + 1] are the
    bins' boundaries in report units (BinNet1's); a value below the first
    inner boundary falls in the first bin, one above the last in the last."""

    bounds: torch.Tensor
    mid: torch.Tensor

    def __init__(self, bounds: torch.Tensor, lcf_scaler: Scaler) -> None:
        k, b = bounds.shape[0], bounds.shape[1] - 1
        super().__init__(k, k * b, lcf_scaler)
        self.b = b
        self.register_buffer("bounds", bounds.detach().clone().float())
        self.register_buffer("mid", ((self.bounds[:, :-1] + self.bounds[:, 1:]) / 2))

    def probs(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, self.k, self.b)

    def means(self, z: torch.Tensor) -> torch.Tensor:
        return (self.probs(z) * self.mid).sum(dim=2)

    def bin_of(self, y: torch.Tensor) -> torch.Tensor:
        """[n, K] the bin index of each value (NaN's is meaningless)."""
        return (y.unsqueeze(2) >= self.bounds[:, 1:-1]).sum(dim=2)

    def patch(self, z: torch.Tensor, y: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        onehot = (self.bin_of(y).unsqueeze(2) == torch.arange(self.b)).to(z.dtype)
        t = self._taken(y, take).unsqueeze(2)
        return torch.where(t, onehot, self.probs(z)).reshape(-1, self.width)

    def fault(
        self, scale: Sequence[float], value: Sequence[float], pinned: Sequence[bool]
    ) -> nn.Module:
        """Each bin's mass moves to the bin holding its midpoint times the
        scale; a pinned LCF puts all its mass on the bin holding the value.
        One fixed matrix per LCF, so the fault is a matrix product."""
        with torch.no_grad():
            s = torch.tensor(scale, dtype=torch.float32).unsqueeze(1)
            v = torch.tensor(value, dtype=torch.float32).unsqueeze(1).expand(-1, self.b)
            pin = torch.tensor(pinned, dtype=torch.bool).unsqueeze(1)
            moved = torch.where(pin, v, self.mid * s)  # [K, B]: where each bin's mass goes
            to = (moved.unsqueeze(2) >= self.bounds[:, 1:-1].unsqueeze(1)).sum(dim=2)
            transfer = (to.unsqueeze(2) == torch.arange(self.b)).float()  # [K, B, B]
        return _Transfer(transfer)


def of(net: nn.Module, lcf_scaler: Scaler) -> Beliefs:
    """The kind of belief a net1 module reports."""
    k = int(lcf_scaler.center.numel())
    if isinstance(net, BinNet1):
        return Bins(net.bounds, lcf_scaler)
    if isinstance(net, IntervalNet1):
        return Interval(k, lcf_scaler)
    return Point(k, lcf_scaler)


class _ScaleOrPin(nn.Module):
    """A point or interval belief scaled or pinned in report units; an
    interval's bounds move with its centre."""

    scale: torch.Tensor
    value: torch.Tensor
    pinned: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(
        self,
        kind: Beliefs,
        scale: Sequence[float],
        value: Sequence[float],
        pinned: Sequence[bool],
        *,
        parts: int,
    ) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(list(scale) * parts, dtype=torch.float32))
        self.register_buffer("value", torch.tensor(list(value) * parts, dtype=torch.float32))
        self.register_buffer("pinned", torch.tensor(list(pinned) * parts, dtype=torch.bool))
        self.register_buffer("center", kind.center.repeat(parts))
        self.register_buffer("spread", kind.spread.repeat(parts))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = z * self.spread + self.center
        v = torch.where(self.pinned, self.value, v * self.scale)
        return (v - self.center) / self.spread


class _Transfer(nn.Module):
    """[n, K * B] distributions -> the same with LCF j's mass moved by
    transfer[j] ([B, B], rows summing to 1)."""

    transfer: torch.Tensor

    def __init__(self, transfer: torch.Tensor) -> None:
        super().__init__()
        self.k, self.b = transfer.shape[0], transfer.shape[1]
        self.register_buffer("transfer", transfer)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = z.reshape(-1, self.k, 1, self.b)
        return torch.matmul(p, self.transfer).reshape(-1, self.k * self.b)


class MeansView(nn.Module):
    """net2's "means" view: z -> the standardized means [n, K]."""

    def __init__(self, kind: Beliefs) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.kind.scaled_means(z)


class PoolBins(nn.Module):
    """[n, K * B] distributions -> [n, K * into]: each group of B / into
    adjacent bins summed into one."""

    def __init__(self, k: int, b: int, into: int) -> None:
        super().__init__()
        if b % into:
            raise ValueError(f"input_bins {into} does not divide net1's {b} bins")
        self.k, self.b, self.into = k, b, into

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.reshape(-1, self.k, self.into, self.b // self.into).sum(dim=3)
        return pooled.reshape(-1, self.k * self.into)
