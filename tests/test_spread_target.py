"""SpreadTarget (train/models.py): the window target that reads net1's
stated spread of capacity."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train.models import BinMeans, SpreadTarget, WindowTarget  # noqa: E402

# Two LCFs (capacity in Mbit/s, one-way delay in ms), four bins each.
BOUNDS = torch.tensor([[1.0, 3.0, 5.0, 12.0, 20.0], [5.0, 15.0, 25.0, 35.0, 45.0]])
MID = (BOUNDS[:, :-1] + BOUNDS[:, 1:]) / 2  # capacity 2, 4, 8.5, 16


def _target(k: float, s0: float) -> tuple[WindowTarget, SpreadTarget]:
    base = WindowTarget(BinMeans(BOUNDS), capacity=0, one_way=1)
    return base, SpreadTarget(base, k, s0)


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
