"""A recurrent net2 that decides once per interval and holds its decision.

net2 reads net1's output (pooled bins, or means) and its own previous u
through a GRU, and answers u: the window is 2^u times the believed BDP. It
decides once per clamp(min RTT, 10 ms, 50 ms) and holds u in between,
aiming each step's window at that step's believed BDP. All of this lives
in the policy graph's state (``Held``, a decision of graph.PolicyGraph), so
the guest runs it as any stateful policy (contract sage-v1,
guest/src/step.rs).

The guest adds exploration noise to every step's action. The graph takes
the u that the decision step's noise produced as the decision it made (it
reads it from the next step's cwnd) and holds that u for the rest of the
block. Every step's action is computed from the sender's actual window, so
later steps' noise only jitters the window and never accumulates. The
learner (ppo.py) replays the same counter from the recorded steps
(``decision_steps``) and reads each decision's realized u the same way, so
nothing beyond the steps table is needed.
"""

# torch's own annotations leave many signatures partly unknown; every other
# strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import math
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.train.graph import Seen

# The step message's last column: the kernel's min RTT in ms (step.rs).
OBS_MIN_RTT: Final = 77
STEP_MS: Final = 10.0  # the guest's step interval
BLOCK_MIN, BLOCK_MAX = 1.0, 5.0  # steps per decision: 10 to 50 ms
_A_LIMIT: Final = 7.99  # the contract's |a| <= 8
_RTT_CAP_MS: Final = 1000.0  # an infinite min RTT counts as this


def block_steps(min_rtt_ms: torch.Tensor) -> torch.Tensor:
    """Steps per decision: min RTT in 10 ms steps, rounded, within 1 to 5;
    a min RTT that is not a number counts as 0 (one step). Written with
    IsNaN and Clip, which the guest's tract runs (isfinite exports an Or it
    does not)."""
    m = torch.where(torch.isnan(min_rtt_ms), torch.zeros_like(min_rtt_ms), min_rtt_ms)
    m = torch.clamp(m, 0.0, _RTT_CAP_MS)
    return torch.clamp(torch.floor((m + STEP_MS / 2) / STEP_MS), BLOCK_MIN, BLOCK_MAX)


def decision_steps(min_rtt_ms: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
    """The steps of one trajectory at which a Held decision decides, from its
    steps' min RTT column, as the graph's counter runs (in float32, as the
    graph computes it)."""
    m = np.asarray(min_rtt_ms, np.float32)
    m = np.clip(np.where(np.isnan(m), np.float32(0), m), 0, _RTT_CAP_MS).astype(np.float32)
    half, step = np.float32(STEP_MS / 2), np.float32(STEP_MS)
    length = np.clip(np.floor((m + half) / step), BLOCK_MIN, BLOCK_MAX)
    out: list[int] = []
    counter = 0.0
    for i, n in enumerate(length.tolist()):
        if counter < 0.5:
            out.append(i)
            counter = n - 1
        else:
            counter -= 1
    return np.asarray(out, np.int64)


class RecurrentActor(nn.Module):
    """GRU over [net1's view, previous u / action_max] -> mean u / action_max
    in [-1, 1] (for a gate student, 2p - 1 for each belief p). ``cell`` is
    one step of the same GRU, written out so the exported graph needs only
    matrix products and activations."""

    def __init__(
        self, width: int, hidden: int = 64, *, memory: bool = True, outputs: int = 1
    ) -> None:
        super().__init__()
        # memory False: every decision starts from a zero hidden state, the
        # same network without memory (an ablation). outputs > 1: a vector
        # per decision (a "gate3" student's two beliefs), [..., outputs].
        self.width, self.hidden, self.memory, self.outputs = width, hidden, memory, outputs
        self.gru = nn.GRU(width + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def clone_rule(self, u: float, action_max: float) -> None:
        """Answer u whatever the input: a zero head with bias atanh(u / max),
        the fixed rule, so training starts from it."""
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(math.atanh(u / action_max))

    def _out(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(-1) if self.outputs == 1 else y

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return self._out(torch.tanh(self.head(h)))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        """``sequence`` before its tanh: [B, T] head outputs ([B, T, outputs])."""
        if not self.memory:
            b, t, f = x.shape
            h, _ = self.gru(x.reshape(b * t, 1, f))
            return self._out(self.head(h).reshape(b, t, self.outputs))
        h, _ = self.gru(x)
        return self._out(self.head(h))

    def cell(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if not self.memory:
            h = torch.zeros_like(h)
        w_ih, w_hh, b_ih, b_hh = (
            cast(torch.Tensor, getattr(self.gru, n))
            for n in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")
        )
        gi = x @ w_ih.t() + b_ih
        gh = h @ w_hh.t() + b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, width + 1] decisions' inputs -> [B, T] mean u / max, from a
        zero hidden state."""
        return torch.tanh(self.pre(x))


class HeldActor(Protocol):
    """What Held needs of its actor: one GRU step and its output."""

    hidden: int

    def cell(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor: ...

    def mean(self, h: torch.Tensor) -> torch.Tensor: ...


class EnsembleActor(nn.Module):
    """Several RecurrentActors on the same inputs, each with its own part of
    the hidden state; the output is the mean of theirs (for a gate
    student, the mean of their beliefs, since p = (a + 1) / 2)."""

    def __init__(self, members: list[RecurrentActor]) -> None:
        super().__init__()
        if not members or any(not m.memory for m in members):
            raise ValueError("an ensemble needs one or more actors with memory")
        self.members = nn.ModuleList(members)
        self.sizes = [m.hidden for m in members]
        self.hidden = sum(self.sizes)

    def _parts(self, h: torch.Tensor) -> list[tuple[RecurrentActor, torch.Tensor]]:
        hs = torch.split(h, self.sizes, dim=-1)
        return [(cast(RecurrentActor, m), hi) for m, hi in zip(self.members, hs, strict=True)]

    def cell(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([m.cell(x, hi) for m, hi in self._parts(h)], dim=-1)

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.mean(hi) for m, hi in self._parts(h)]).mean(dim=0)


class ConcatActor(nn.Module):
    """Actors on the same inputs, each with its own part of the hidden
    state; the output is theirs side by side, [B, sum of outputs] (a
    policy's u next to another's beliefs, for a gate that combines them)."""

    def __init__(self, members: list[RecurrentActor | EnsembleActor]) -> None:
        super().__init__()
        if not members:
            raise ValueError("a concatenation needs one or more actors")
        self.members = nn.ModuleList(members)
        self.sizes = [m.hidden for m in members]
        self.hidden = sum(self.sizes)

    def _parts(self, h: torch.Tensor) -> list[tuple[HeldActor, torch.Tensor]]:
        hs = torch.split(h, self.sizes, dim=-1)
        return [(cast(HeldActor, m), hi) for m, hi in zip(self.members, hs, strict=True)]

    def cell(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([m.cell(x, hi) for m, hi in self._parts(h)], dim=-1)

    def mean(self, h: torch.Tensor) -> torch.Tensor:
        outs = [m.mean(hi) for m, hi in self._parts(h)]
        return torch.cat([o.unsqueeze(-1) if o.dim() == 1 else o for o in outs], dim=-1)


class Held(nn.Module):
    """A PolicyGraph decision (graph.py) for a recurrent net2 that decides
    once per block and holds its decision.

    A decision holds its realized u (the u it asked for, plus the guest's
    exploration noise on that step) for the block, and every step aims the
    window at 2^u times the believed BDP of that step, as the fixed rule
    does. Holding the window itself would freeze the BDP it was computed
    from: copies of the rule cloned that way queued more than copies of the
    rule and lost 0.12 in log utility to them (exp-net2-clone-copies-107).

    ``head`` maps the actor's output a in [-1, 1], net1's beliefs z and the
    traffic head's probabilities to u (ppo.Head: a times action_max, a
    residual on a rule, or a gate between rules).

    State, [1, hidden + 5]: the GRU's hidden state, the previous decision's
    realized u, the steps left in the block, the window of this step (log2
    packets), the believed BDP at the decision (log2 packets), and 1 on
    the step after a decision (when its realized u is read)."""

    def __init__(
        self,
        actor: RecurrentActor | EnsembleActor | ConcatActor,
        head: nn.Module,
        action_max: float,
    ) -> None:
        super().__init__()
        self.actor, self.head, self.action_max = actor, head, action_max

    @property
    def held(self) -> HeldActor:
        return cast(HeldActor, self.actor)

    @property
    def state_dim(self) -> int:
        return self.held.hidden + 5

    def forward(self, seen: Seen, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        cwnd, lb, obs = seen.cwnd, seen.lb, seen.obs
        if state is None or cwnd is None or lb is None:
            raise ValueError("a held net2 needs its state, a window target and the cwnd input")
        lc = torch.log2(torch.clamp(cwnd, min=1.0)).squeeze(1)
        n = self.held.hidden
        h = state[:, :n]
        prev_u, left, _, lb_dec, pending = (state[:, n + i] for i in range(5))
        captured = pending > 0.5
        prev_u = torch.where(captured, lc - lb_dec, prev_u)
        decide = left < 0.5
        inp = torch.cat([seen.x, (prev_u / self.action_max).unsqueeze(1)], dim=1)
        h_new = self.held.cell(inp, h)
        u = self.head(self.held.mean(h_new), seen.z, seen.traffic)
        h = torch.where(decide.unsqueeze(1), h_new, h)
        window = torch.where(decide, lb + u, lb + prev_u)
        a = torch.clamp(window - lc, -_A_LIMIT, _A_LIMIT).unsqueeze(1)
        left = torch.where(decide, block_steps(obs[:, OBS_MIN_RTT]) - 1, left - 1)
        lb_dec = torch.where(decide, lb, lb_dec)
        nxt = torch.stack([prev_u, left, window, lb_dec, decide.to(obs.dtype)], dim=1)
        return a, torch.cat([h, nxt], dim=1)
