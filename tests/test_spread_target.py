"""SpreadTarget and IdleTarget (train/graph.py): window targets that probe
on net1's stated spread of capacity, or on its mean busy fraction."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train.beliefs import Bins  # noqa: E402
from expliccit_env.train.graph import IdleTarget, SpreadTarget, WindowTarget  # noqa: E402
from expliccit_env.train.models import Scaler  # noqa: E402

# Two LCFs (capacity in Mbit/s, one-way delay in ms), four bins each.
BOUNDS = torch.tensor([[1.0, 3.0, 5.0, 12.0, 20.0], [5.0, 15.0, 25.0, 35.0, 45.0]])
MID = (BOUNDS[:, :-1] + BOUNDS[:, 1:]) / 2  # capacity 2, 4, 8.5, 16


def _bins(bounds: torch.Tensor) -> Bins:
    return Bins(bounds, Scaler((0, 1), *([torch.zeros(2).numpy()] * 3), torch.ones(2).numpy()))


class _On(torch.nn.Module):
    """A target called on net1's beliefs alone, as the tests read it."""

    def __init__(self, target: torch.nn.Module) -> None:
        super().__init__()
        self.target = target

    def forward(self, z: torch.Tensor, traffic: torch.Tensor | None = None) -> torch.Tensor:
        return self.target(torch.zeros(z.shape[0], 78), z, traffic)


def _target(k: float, s0: float, *, gated: bool = False) -> tuple[_On, _On]:
    base = WindowTarget(_bins(BOUNDS), capacity=0, one_way=1)
    return _On(base), _On(SpreadTarget(base, k, s0, copies_gated=gated))


def _belief(capacity: list[float]) -> torch.Tensor:
    delay = [0.0, 1.0, 0.0, 0.0]
    return torch.tensor([capacity + delay])


def test_a_sure_capacity_leaves_the_target_unchanged() -> None:
    base, spread = _target(k=2.0, s0=0.0)
    z = _belief([0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(spread(z), base(z))


def test_k_zero_is_the_window_target() -> None:
    base, spread = _target(k=0.0, s0=0.2)
    z = _belief([0.25, 0.25, 0.25, 0.25])
    assert torch.allclose(spread(z), base(z))


def test_the_spread_past_s0_raises_the_target_by_k_times_it() -> None:
    base, spread = _target(k=2.0, s0=0.2)
    z = _belief([0.5, 0.0, 0.0, 0.5])  # log2 capacity 1 or 4: sd 1.5
    expected = base(z) + 2.0 * (1.5 - 0.2)
    assert torch.allclose(spread(z), expected, atol=1e-5)
    # Below s0 the spread adds nothing.
    _, high = _target(k=2.0, s0=2.0)
    assert torch.allclose(high(z), base(z))
    assert math.isclose(float(MID[0, 0]), 2.0) and math.isclose(float(MID[0, 3]), 16.0)


def test_idle_raises_the_target_by_k_times_the_believed_idle_share() -> None:
    # The second LCF read as a busy fraction: bins 0-0.25 ... 0.75-1.
    bounds = torch.tensor([[1.0, 3.0, 5.0, 12.0, 20.0], [0.0, 0.25, 0.5, 0.75, 1.0]])
    window = WindowTarget(_bins(bounds), capacity=0, one_way=1)
    base, idle = _On(window), _On(IdleTarget(window, busy=1, k=2.0, i0=0.05))
    busy = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # busy 0.875
    half = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])  # busy 0.375
    assert torch.allclose(idle(busy), base(busy) + 2.0 * (1 - 0.875 - 0.05))
    assert torch.allclose(idle(half), base(half) + 2.0 * (1 - 0.375 - 0.05))


def test_a_copies_gated_spread_lifts_by_one_minus_p_copies() -> None:
    base, spread = _target(k=2.0, s0=0.2, gated=True)
    z = _belief([0.5, 0.0, 0.0, 0.5]).repeat(3, 1)
    head = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])
    lift = 2.0 * (1.5 - 0.2)
    # No other traffic: the full lift; copies: none; P(copies) 0.25: 0.75 of it.
    want = base(z) + lift * torch.tensor([1.0, 0.0, 0.75])
    assert torch.allclose(spread(z, head), want, atol=1e-5)
    with pytest.raises(ValueError, match="traffic head"):
        spread(z)
