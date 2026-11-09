"""train/attribution.py: exact Shapley values of a decision over its LCFs."""

from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train.attribution import coalitions, leave_one_out, shapley  # noqa: E402


def _mlp(d: int, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Tanh()
    ).eval()


def _brute(h, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """The definition, summed over permutations' prefixes the slow way."""
    k = x.shape[0]
    phi = torch.zeros(k, dtype=torch.float64)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for r in range(k):
            for subset in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                z = ref.clone()
                z[list(subset)] = x[list(subset)]
                zi = z.clone()
                zi[i] = x[i]
                with torch.no_grad():
                    gain = h(zi.unsqueeze(0)).item() - h(z.unsqueeze(0)).item()
                phi[i] += w * gain
    return phi


def test_coalitions_enumerate_every_subset() -> None:
    c = coalitions(3)
    assert c.shape == (8, 3)
    assert {tuple(r.tolist()) for r in c} == set(itertools.product([False, True], repeat=3))


def test_values_match_the_definition_and_sum_to_the_change() -> None:
    h = _mlp(5)
    x, ref = torch.randn(4, 5), torch.randn(5)
    phi = shapley(lambda z: h(z).squeeze(1), x, ref)
    for row in range(4):
        assert torch.allclose(phi[row], _brute(h, x[row], ref), atol=1e-5)
    with torch.no_grad():
        change = (h(x) - h(ref.unsqueeze(0))).squeeze(1).to(torch.float64)
    assert torch.allclose(phi.sum(1), change, atol=1e-5)


def test_an_additive_decision_gets_its_own_terms() -> None:
    w = torch.tensor([0.5, -2.0, 0.0, 1.5])
    x, ref = torch.randn(3, 4), torch.randn(4)

    def h(z: torch.Tensor) -> torch.Tensor:
        return z @ w

    phi = shapley(h, x, ref)
    assert torch.allclose(phi, ((x - ref) * w).to(torch.float64), atol=1e-6)
    assert torch.allclose(leave_one_out(h, x, ref), phi, atol=1e-6)
    assert torch.all(phi[:, 2] == 0)  # an ignored LCF gets exactly 0


def test_groups_attribute_a_block_of_columns_to_one_player() -> None:
    h = _mlp(6, seed=1)
    x, ref = torch.randn(2, 6), torch.randn(6)
    phi = shapley(lambda z: h(z).squeeze(1), x, ref, groups=[[0, 1, 2], [3, 4], [5]])
    assert phi.shape == (2, 3)
    with torch.no_grad():
        change = (h(x) - h(ref.unsqueeze(0))).squeeze(1).to(torch.float64)
    assert torch.allclose(phi.sum(1), change, atol=1e-5)
    with pytest.raises(ValueError, match="exactly one player"):
        shapley(lambda z: h(z).squeeze(1), x, ref, groups=[[0, 1], [1, 2, 3, 4, 5]])
