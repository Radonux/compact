"""Exact attribution of a decision to the LCFs it was computed from.

net2 is a function h of net1's output alone, so a decision can be split
among the LCFs exactly: the Shapley value of LCF i is its marginal effect
h(z_{S+i}) - h(z_S) averaged over every set S of the other LCFs, weighted
|S|! (k - |S| - 1)! / k!, where z_S takes the LCFs in S from the decision
and the rest from a reference input (the mean over training data). The
values sum to h(x) - h(reference) for every decision (efficiency), and
an LCF the network ignores gets exactly 0. It takes 2^k evaluations of h
per decision: 512 for nine LCFs, 128 for seven.

A player is a group of input columns: one column per LCF for a net1 that
reports single values, or an LCF's whole block of bins for one that
reports distributions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

type Decide = Callable[[torch.Tensor], torch.Tensor]  # [n, d] inputs -> [n] actions
CHUNK = 65_536  # inputs per call of h


def coalitions(k: int) -> torch.Tensor:
    """[2^k, k] bool: row s holds the players whose bit is set in s."""
    if not 1 <= k <= 16:
        raise ValueError(f"{k} players: exact Shapley takes 2^k evaluations; use 1 to 16")
    s = torch.arange(2**k).unsqueeze(1)
    return (s >> torch.arange(k)) & 1 == 1


def _columns(groups: Sequence[Sequence[int]], d: int) -> torch.Tensor:
    """[k, d] bool: player j owns these input columns."""
    owner = torch.zeros(len(groups), d, dtype=torch.bool)
    for j, cols in enumerate(groups):
        owner[j, list(cols)] = True
    if owner.sum(0).ne(1).any():
        raise ValueError("every input column belongs to exactly one player")
    return owner


def _evaluate(h: Decide, x: torch.Tensor, ref: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """[m, 2^k]: h on every coalition's input for every decision."""
    m, d = x.shape
    member = coalitions(owner.shape[0]).to(torch.float32) @ owner.to(torch.float32) > 0.5
    z = torch.where(member.unsqueeze(0), x.unsqueeze(1), ref.view(1, 1, d)).reshape(-1, d)
    with torch.no_grad():
        out = torch.cat([h(z[i : i + CHUNK]).reshape(-1) for i in range(0, z.shape[0], CHUNK)])
    return out.reshape(m, -1)


def shapley(
    h: Decide, x: torch.Tensor, ref: torch.Tensor, groups: Sequence[Sequence[int]] | None = None
) -> torch.Tensor:
    """[m, k] Shapley values of each decision's players, for inputs x [m, d]
    against the reference ref [d]; groups default to one column each."""
    d = x.shape[1]
    owner = _columns(groups if groups is not None else [[c] for c in range(d)], d)
    k = owner.shape[0]
    values = _evaluate(h, x.to(torch.float32), ref.to(torch.float32), owner).to(torch.float64)
    sets = coalitions(k)
    size = sets.sum(1)
    weight = torch.tensor(
        [math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k) for s in range(k)],
        dtype=torch.float64,
    )
    phi = torch.zeros(x.shape[0], k, dtype=torch.float64)
    index = torch.arange(2**k)
    for i in range(k):
        without = index[~sets[:, i]]
        gain = values[:, without | (1 << i)] - values[:, without]
        phi[:, i] = gain @ weight[size[without]]
    return phi


def leave_one_out(
    h: Decide, x: torch.Tensor, ref: torch.Tensor, groups: Sequence[Sequence[int]] | None = None
) -> torch.Tensor:
    """[m, k]: h(x) minus h with one player set to the reference, the
    intervention an operator can run on the deployed function. It equals
    the Shapley value when h is additive; the gap measures interaction."""
    d = x.shape[1]
    owner = _columns(groups if groups is not None else [[c] for c in range(d)], d)
    xs = x.to(torch.float32)
    with torch.no_grad():
        full = h(xs).reshape(-1)
        out: list[torch.Tensor] = []
        for j in range(owner.shape[0]):
            z = torch.where(owner[j].unsqueeze(0), ref.to(torch.float32).unsqueeze(0), xs)
            out.append(full - h(z).reshape(-1))
    return torch.stack(out, dim=1).to(torch.float64)
