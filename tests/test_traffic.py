"""net1's traffic head and the wrappers that put it into a policy
(train/traffic.py)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train.traffic import (  # noqa: E402
    CLASSES,
    Beliefs,
    HeadQ,
    TrafficHead,
    WithTraffic,
    fit_head,
)


def test_the_head_learns_separable_classes() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, len(CLASSES), 3000)
    x = (rng.normal(size=(3000, 5)) + 3 * np.eye(5)[y]).astype(np.float32)
    head = fit_head(x, y.astype(np.int64), epochs=5)
    with torch.no_grad():
        p = head(torch.from_numpy(x))
    assert torch.allclose(p.sum(1), torch.ones(3000), atol=1e-5)
    assert (p.argmax(1).numpy() == y).mean() > 0.9


class _Echo(torch.nn.Module):
    """A gate that returns what it was given, to read HeadQ's arguments."""

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, a], dim=1)


def test_head_q_takes_q_from_the_head_and_hides_its_columns() -> None:
    beliefs = torch.tensor([[0.1, 0.2]])
    head = torch.tensor([[0.2, 0.7, 0.1]])  # none, copies, stock
    out = HeadQ(_Echo())(torch.cat([beliefs, head], dim=1), torch.tensor([[0.5, -1.0]]))
    # The gate sees the beliefs alone, p as given and q = 2 P(copies) - 1.
    assert torch.allclose(out, torch.tensor([[0.1, 0.2, 0.5, 0.4]]))


def test_with_traffic_appends_the_head_and_beliefs_strip_it() -> None:
    head = TrafficHead(4)
    net1 = torch.nn.Linear(4, 6)
    x = torch.randn(3, 4)
    z = WithTraffic(net1, head)(x)
    assert z.shape == (3, 6 + len(CLASSES))
    assert torch.allclose(Beliefs(None)(z), net1(x))
    assert torch.allclose(Beliefs(torch.nn.Identity())(z), net1(x))
