"""Architectures by name, and the stages around them inside a policy graph.

A policy graph is: message -> Scaler (columns, non-finite values, scaling)
-> net1 -> LCF beliefs (scaled) -> net2 -> a. Everything a deployed policy
needs is a buffer or a parameter, so the exported ONNX graph is complete.

The "expliccit-v1" modules use expliCCit's layer layout and parameter names,
so its released checkpoints load into them unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

type F32 = npt.NDArray[np.float32]
_BIG = 3.0e38
# Inside the contract's |a| <= MAX_ACTION (policy.py), with room for float32.
_A_LIMIT = 7.99


class Scaler(nn.Module):
    """x -> (clip(x, lo, hi) - center) / spread on chosen columns, with any
    non-finite value replaced by 0 first (as expliCCit's environment does)."""

    cols: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    center: torch.Tensor
    spread: torch.Tensor

    def __init__(self, cols: tuple[int, ...], lo: F32, hi: F32, center: F32, spread: F32) -> None:
        super().__init__()
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long))
        for name, v in (("lo", lo), ("hi", hi), ("center", center), ("spread", spread)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))

    @staticmethod
    def fit(cols: tuple[int, ...], x: F32, kind: Literal["robust", "minmax", "standard"]) -> Scaler:
        """Fit on rows x (already restricted to ``cols``); non-finite values
        are ignored, and a column without any finite value maps to itself."""
        # An all-NaN column is expected (a label undefined on these rows):
        # it gets centre 0 and spread 1 below, so numpy's warnings are noise.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x64 = np.where(np.isfinite(x), x, np.nan).astype(np.float64)
            n = x64.shape[1]
            lo, hi = np.full(n, -_BIG), np.full(n, _BIG)
            if kind == "minmax":  # sklearn's MinMaxScaler, feature_range (0, 1)
                center = np.nanmin(x64, axis=0)
                spread = np.nanmax(x64, axis=0) - center
            else:
                if kind == "robust":
                    lo, hi = np.nanquantile(x64, 0.001, axis=0), np.nanquantile(x64, 0.999, axis=0)
                clipped = np.clip(x64, lo, hi)
                center, spread = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
        lo, hi = np.where(np.isfinite(lo), lo, -_BIG), np.where(np.isfinite(hi), hi, _BIG)
        f = np.float32
        return Scaler(cols, lo.astype(f), hi.astype(f), center.astype(f), spread.astype(f))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.index_select(1, self.cols)
        # |x| < 3e38 is false for NaN and both infinities, and exports as
        # Abs, Less, Where, which tract runs (IsInf and IsNaN do not analyse).
        x = torch.where(torch.abs(x) < _BIG, x, torch.zeros_like(x))
        return (torch.minimum(torch.maximum(x, self.lo), self.hi) - self.center) / self.spread

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.spread + self.center


class _Residual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear1, self.bn1 = nn.Linear(width, width), nn.BatchNorm1d(width)
        self.linear2, self.bn2 = nn.Linear(width, width), nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn2(self.linear2(torch.relu(self.bn1(self.linear1(x)))))
        return torch.relu(h + x)


class ExpliccitNet1(nn.Module):
    """expliCCit's encoder: a widening layer, two residual blocks, a tapering
    stack (w, w/2, w/4, w/8 with batch norm), one linear head per LCF."""

    def __init__(
        self, inputs: int, lcfs: int, width: int = 512, dropout: float = 0.3, per_lcf: int = 1
    ) -> None:
        super().__init__()
        w = width

        def layer(a: int, b: int, drop: bool) -> list[nn.Module]:
            return [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.ReLU(),
                *([nn.Dropout(dropout)] if drop else []),
            ]

        self.initial = nn.Sequential(*layer(inputs, w, True))
        self.res1, self.res2 = _Residual(w), _Residual(w)
        self.hidden = nn.Sequential(
            *layer(w, w // 2, True), *layer(w // 2, w // 4, True), *layer(w // 4, w // 8, False)
        )
        self.heads = nn.ModuleList([nn.Linear(w // 8, per_lcf) for _ in range(lcfs)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(self.res2(self.res1(self.initial(x))))
        return torch.cat([head(h) for head in self.heads], dim=1)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # relu(x) + log(1 + exp(-|x|)): stable, and only ops tract runs.
    return torch.relu(x) + torch.log(1 + torch.exp(-torch.abs(x)))


class IntervalNet1(nn.Module):
    """net1 reporting what the observations support: for each LCF a centre
    and an interval. The base network gives three numbers per LCF (m, a, b);
    the bounds are m - softplus(a) - c and m + softplus(b) + c, so they never
    cross the centre, and c is the conformal correction fitted on held-out
    paths. Output: the K centres, then the K lower bounds, then the K upper
    bounds, all in the LCF scaler's units."""

    widen: torch.Tensor

    def __init__(self, base: nn.Module, lcfs: int) -> None:
        super().__init__()
        self.base, self.k = base, lcfs
        self.register_buffer("widen", torch.zeros(lcfs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.base(x).reshape(-1, self.k, 3)
        m, a, b = o[:, :, 0], o[:, :, 1], o[:, :, 2]
        lo = torch.minimum(m - _softplus(a) - self.widen, m)
        hi = torch.maximum(m + _softplus(b) + self.widen, m)
        return torch.cat([m, lo, hi], dim=1)


class BinNet1(nn.Module):
    """net1 reporting, for each LCF, a probability distribution over fixed
    bins of that quantity, in report units. ``bounds[j]`` holds LCF j's
    B + 1 bin boundaries: bin i is [bounds[j, i], bounds[j, i + 1]). For
    classifying a value, the first bin also takes everything below and the
    last everything above; the outer boundaries only serve to read
    quantiles. Output: the K distributions in LCF order, each summing to 1.
    With several members (an ensemble, each trained on its own), the
    distribution is their average, which widens where they disagree.
    """

    bounds: torch.Tensor

    def __init__(self, members: Sequence[nn.Module], lcfs: int, bins: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.k, self.b = lcfs, bins
        self.register_buffer("bounds", torch.zeros(lcfs, bins + 1))

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        p = [torch.softmax(m(x).reshape(-1, self.k, self.b), dim=2) for m in self.members]
        return p[0] if len(p) == 1 else torch.stack(p).mean(dim=0)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """[n, K, B] log-probabilities."""
        if len(self.members) == 1:
            return torch.log_softmax(self.members[0](x).reshape(-1, self.k, self.b), dim=2)
        return torch.log(self._probs(x).clamp(min=1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._probs(x).reshape(-1, self.k * self.b)

    def bin_of(self, y: torch.Tensor) -> torch.Tensor:
        """[n, K] bin index of each value: the number of inner boundaries at
        or below it. A NaN's bin means nothing (mask it)."""
        inner = self.bounds[:, 1:-1].contiguous()
        return torch.searchsorted(inner, y.t().contiguous(), right=True).t()


class PoolBins(nn.Module):
    """[n, K * B] distributions -> [n, K * into]: each group of B / into
    adjacent bins summed into one."""

    def __init__(self, k: int, b: int, into: int) -> None:
        super().__init__()
        self.k, self.b, self.into = k, b, into

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.reshape(-1, self.k, self.into, self.b // self.into).sum(dim=3)
        return pooled.reshape(-1, self.k * self.into)


class BinMeans(nn.Module):
    """[n, K * B] distributions -> [n, K] means in report units, each bin's
    mass at its midpoint, as Net1.predict computes them."""

    mid: torch.Tensor

    def __init__(self, bounds: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mid", ((bounds[:, :-1] + bounds[:, 1:]) / 2).float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        k, b = self.mid.shape
        return (z.reshape(-1, k, b) * self.mid).sum(dim=2)


class ScaledMeans(nn.Module):
    """A distribution's means, standardized as net1's targets are: the
    single values a point net1 would hand net2, from the same estimator."""

    def __init__(self, means: BinMeans, scaler: Scaler) -> None:
        super().__init__()
        self.means, self.scaler = means, scaler

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.scaler(self.means(z))


# Bits per packet in the BDP: Mahimahi's 1500-byte packets.
PACKET_BITS = 1500 * 8


class WindowTarget(nn.Module):
    """log2 of the path's believed bandwidth-delay product in packets, from
    a distribution net1's output: its mean capacity (Mbit/s) times twice
    its mean one-way delay (ms). At least one packet."""

    def __init__(self, means: BinMeans, capacity: int, one_way: int) -> None:
        super().__init__()
        self.means, self.capacity, self.one_way = means, capacity, one_way

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.means(z)
        bdp = mean[:, self.capacity] * 1e6 * mean[:, self.one_way] * 2e-3 / PACKET_BITS
        return torch.log2(torch.clamp(bdp, min=1.0))


class ExpliccitActor(nn.Module):
    """LCF beliefs -> a in [-1, 1] (two hidden ReLU layers, tanh)."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )  # fmt: skip

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    """(LCF beliefs, a) -> Q."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=1)).squeeze(1)


class Value(nn.Module):
    """LCF beliefs -> V (implicit Q-learning's state value)."""

    def __init__(self, lcfs: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lcfs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(1)


NET1 = {"expliccit-v1": ExpliccitNet1}
NET2 = {"expliccit-v1": ExpliccitActor}


class PolicyGraph(nn.Module):
    """The deployed policy under contract sage-v1: message -> a.

    With ``as_deployed`` it repeats what expliCCit's deployed controller does
    between the networks (beliefs back to physical units, clamped at 0, the
    first six rounded to integers, scaled again) and rounds a to three
    decimals before the guest exponentiates it; otherwise net2 sees net1's
    output directly, as in training.

    ``take_from_truth`` makes chosen LCFs come from the contract's "truth"
    input instead of net1: all of them for a policy on the true LCFs (an
    oracle), one or a few for a patched net1. The truth goes through the
    same LCF scaler as net1's training targets, so a patched belief is what
    a perfect net1 would have said; for an interval net1, its centre and
    both bounds become the truth; for a distribution net1, all probability
    goes to the truth's bin, except where the truth is NaN, which keeps
    net1's own belief.

    ``view`` turns net1's output into net2's input (PoolBins or
    ScaledMeans; none passes it as it is). With a ``target``, net2's output u is the
    next window as a log2 multiple of the believed BDP, and the graph
    answers a = log2(BDP) + u - log2(cwnd) from the contract's "cwnd"
    input, kept within the contract's bound."""

    truth_cols: torch.Tensor
    truth_units: torch.Tensor
    from_truth: torch.Tensor
    truth_bounds: torch.Tensor
    rounded: torch.Tensor

    def __init__(
        self,
        scaler: Scaler,
        net1: nn.Module,
        lcf_scaler: Scaler,
        net2: nn.Module,
        action_max: float,
        *,
        as_deployed: bool = False,
    ) -> None:
        super().__init__()
        self.scaler, self.net1, self.lcf_scaler, self.net2 = scaler, net1, lcf_scaler, net2
        self.action_max = action_max
        self.as_deployed = as_deployed
        # Round a to this many decimals (as_deployed implies 3).
        self.action_decimals: int | None = None
        self.register_buffer("truth_cols", torch.zeros(0, dtype=torch.long))
        self.register_buffer("truth_units", torch.zeros(0))
        self.register_buffer("from_truth", torch.zeros(0, dtype=torch.bool))
        self.register_buffer("truth_bounds", torch.zeros(0))
        # With as_deployed, the LCFs rounded to integers (in net1's order);
        # empty: the first six, as expliCCit's deployed controller does.
        self.register_buffer("rounded", torch.zeros(0, dtype=torch.bool))
        self.view: nn.Module | None = None
        self.target: nn.Module | None = None  # WindowTarget, or a measured BDP (train/ppo.py)

    def take_from_truth(
        self,
        cols: list[int],
        units: list[float],
        mask: list[bool],
        reps: int,
        bounds: torch.Tensor | None = None,
    ) -> None:
        """Per LCF in net1's order: its column of the truth input, the factor
        to net1's target units, and whether to take it from the truth;
        ``reps`` is 3 for an interval net1 (centres, lower, upper), else 1;
        ``bounds`` are a distribution net1's bin boundaries."""
        self.truth_cols = torch.tensor(cols, dtype=torch.long)
        self.truth_units = torch.tensor(units, dtype=torch.float32)
        self.from_truth = torch.tensor(mask * reps, dtype=torch.bool)
        if bounds is not None:
            self.truth_bounds = bounds.clone()

    @property
    def uses_truth(self) -> bool:
        return bool(self.from_truth.any())

    @property
    def state_dim(self) -> int:
        """S of the contract's "state" input, 0 for a net2 without memory. A
        net2 with memory declares ``state_dim`` and maps (x, state) to
        (u, next state)."""
        return int(getattr(self.net2, "state_dim", 0))

    def forward(
        self,
        obs: torch.Tensor,
        truth: torch.Tensor | None = None,
        cwnd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = torch.zeros(obs.shape[0], self.state_dim) if self.state_dim else None
        return self.step(obs, state, truth, cwnd)[0]

    def step(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None,
        truth: torch.Tensor | None = None,
        cwnd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """a and, for a net2 with memory, its next state."""
        z = self.net1(self.scaler(obs))
        if truth is not None and self.uses_truth and self.truth_bounds.numel():
            y = truth.index_select(1, self.truth_cols) * self.truth_units
            k, b = self.truth_bounds.shape[0], self.truth_bounds.shape[1] - 1
            idx = (y.unsqueeze(2) >= self.truth_bounds[:, 1:-1]).sum(dim=2, keepdim=True)
            onehot = (idx == torch.arange(b)).to(z.dtype)
            take = self.from_truth.reshape(1, k, 1) & (torch.abs(y) < _BIG).unsqueeze(2)
            z = torch.where(take, onehot, z.reshape(-1, k, b)).reshape(-1, k * b)
        elif truth is not None and self.uses_truth:
            zt = self.lcf_scaler(truth.index_select(1, self.truth_cols) * self.truth_units)
            reps = self.from_truth.shape[0] // zt.shape[1]
            z = torch.where(self.from_truth, zt.repeat(1, reps), z)
        if self.as_deployed:
            c = torch.relu(self.lcf_scaler.inverse(z))
            mask = self.rounded if self.rounded.numel() else torch.arange(c.shape[1]) < 6
            c = torch.where(mask, torch.round(c), c)
            z = (c - self.lcf_scaler.center) / self.lcf_scaler.spread
        x = z if self.view is None else self.view(z)
        if state is None:
            a = self.net2(x) * self.action_max
        else:
            u, state = self.net2(x, state)
            a = u * self.action_max
        if self.target is not None:
            if cwnd is None:
                raise ValueError("a window target needs the cwnd input")
            a = self.target(z).unsqueeze(1) + a - torch.log2(torch.clamp(cwnd, min=1.0))
            a = torch.clamp(a, -_A_LIMIT, _A_LIMIT)
        decimals = 3 if self.as_deployed else self.action_decimals
        if decimals is None:
            return a, state
        scale = float(10**decimals)
        return torch.round(a * scale) / scale, state
