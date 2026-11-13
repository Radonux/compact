"""The closed-loop learners of the recurrent net2 (ppo.py): PPO, TD3+BC
and DAgger, one interface (``actor``, ``critic``, ``update``), the head
of actor.py in each, and how a run's learner starts (handoff.py)."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
import math
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.train.actors import new_actor, policy_u
from expliccit_env.train.handoff import init_paths
from expliccit_env.train.recurrent import RecurrentActor
from expliccit_env.train.spec import PPOSpec
from expliccit_env.train.trajectories import Trajectory

type F32 = npt.NDArray[np.float32]


def mean_u(spec: PPOSpec, trs: list[Trajectory], a: torch.Tensor) -> torch.Tensor:
    """policy_u at each padded decision of ``trs``, from the actor's padded
    output a in [-1, 1] and each trajectory's rules' u."""

    def padded(xs: list[F32]) -> torch.Tensor:
        return pad([x[:, None] for x in xs])[0].squeeze(-1)

    if spec.student.kind != "u":
        share = padded([field_of(tr.share) for tr in trs])
        coop = padded([field_of(tr.coop) for tr in trs]) if spec.student.kind == "gate3" else None
        return policy_u(spec, a, share, coop)
    if all(tr.base is None for tr in trs):
        return policy_u(spec, a)
    base = [tr.base if tr.base is not None else np.zeros(tr.u.size, np.float32) for tr in trs]
    return policy_u(spec, a, base=padded(base))


class ValueNet(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c).squeeze(-1)


def advantages(trs: list[Trajectory], values: list[F32], lam: float) -> tuple[list[F32], list[F32]]:
    """GAE over each trajectory's decisions (the last bootstraps on its own
    value: episodes are cut, not ended), then each group's mean advantage
    at the same decision index taken off. Returns (advantages, returns)."""
    adv, ret = [], []
    for tr, v in zip(trs, values, strict=True):
        nxt = np.append(v[1:], v[-1])
        delta = tr.reward + tr.gamma * nxt - v
        a = np.zeros_like(delta)
        run = 0.0
        for i in range(delta.size - 1, -1, -1):
            run = float(delta[i]) + float(tr.gamma[i]) * lam * run
            a[i] = run
        adv.append(a)
        ret.append((a + v).astype(np.float32))
    by: dict[tuple[int, int], list[int]] = {}
    for i, tr in enumerate(trs):
        by.setdefault(tr.group, []).append(i)
    centred = [a.copy() for a in adv]
    for members in by.values():
        if len(members) < 2:
            continue
        n = min(adv[i].size for i in members)
        mean = np.mean([adv[i][:n] for i in members], axis=0)
        for i in members:
            centred[i][:n] -= mean
    return centred, ret


def pad(xs: list[F32]) -> tuple[torch.Tensor, torch.Tensor]:
    t = max(x.shape[0] for x in xs)
    out = np.zeros((len(xs), t, *xs[0].shape[1:]), np.float32)
    mask = np.zeros((len(xs), t), np.float32)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = 1
    return torch.from_numpy(out), torch.from_numpy(mask)


def value_step(
    critic: ValueNet,
    opt: torch.optim.Optimizer,
    batch: list[Trajectory],
    returns: list[F32],
    max_grad_norm: float,
) -> float:
    """One gradient step of the value network toward the returns."""
    c, cmask = pad([tr.critic for tr in batch])
    g, _ = pad([r[:, None] for r in returns])
    vl = (((critic(c) - g.squeeze(-1)) ** 2) * cmask).sum() / cmask.sum()
    opt.zero_grad()
    vl.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    opt.step()
    return vl.item()


def explained_variance(returns: list[F32], values: list[F32]) -> float:
    allret, allv = np.concatenate(returns), np.concatenate(values)
    return 1.0 - float(np.var(allret - allv)) / max(float(np.var(allret)), 1e-12)


class PPOLearner:
    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = new_actor(width, spec)
        self.critic = ValueNet(critic_width, spec.critic_hidden)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=spec.critic_lr)
        # The policy it started from, frozen (make_learner sets it after
        # loading any saved actor): the anchor's reference and kl_to_start's.
        self.start = copy.deepcopy(self.actor).requires_grad_(False)

    def _means(self, actor: RecurrentActor, trs: list[Trajectory]) -> list[F32]:
        with torch.no_grad():
            x, mask = pad([tr.inp for tr in trs])
            mu = mean_u(self.spec, trs, actor.sequence(x)).numpy()
        return [mu[i, : int(mask[i].sum())] for i in range(len(trs))]

    def means(self, trs: list[Trajectory]) -> list[F32]:
        return self._means(self.actor, trs)

    def values(self, trs: list[Trajectory]) -> list[F32]:
        with torch.no_grad():
            return [self.critic(torch.from_numpy(tr.critic)).numpy() for tr in trs]

    def update(
        self, trs: list[Trajectory], rng: np.random.Generator, *, actor: bool = True
    ) -> dict[str, float]:
        s = self.spec
        v_old = self.values(trs)
        mu_old = self.means(trs)
        adv, ret = advantages(trs, v_old, s.gae_lambda)
        flat = np.concatenate(adv)
        scale = float(flat.std()) + 1e-8
        adv = [((a - flat.mean()) / scale).astype(np.float32) for a in adv]
        explained = explained_variance(ret, v_old)
        logs: dict[str, list[float]] = {}
        var = 2 * s.noise_sigma**2
        for _ in range(s.epochs):
            order = rng.permutation(len(trs))
            for b in range(0, len(trs), s.batch_trajectories):
                idx = order[b : b + s.batch_trajectories]
                x, mask = pad([trs[i].inp for i in idx])
                u, _ = pad([trs[i].u[:, None] for i in idx])
                a, _ = pad([adv[i][:, None] for i in idx])
                m0, _ = pad([mu_old[i][:, None] for i in idx])
                u, a, m0 = u.squeeze(-1), a.squeeze(-1), m0.squeeze(-1)
                batch = [trs[i] for i in idx]
                mu = mean_u(s, batch, self.actor.sequence(x))
                ratio = torch.exp(((u - m0) ** 2 - (u - mu) ** 2) / var)
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - s.clip, 1 + s.clip) * a)
                loss = -(surr * mask).sum() / mask.sum()
                if s.anchor > 0:
                    with torch.no_grad():
                        m_start = mean_u(s, batch, self.start.sequence(x))
                    kl = (((mu - m_start) ** 2 / var) * mask).sum() / mask.sum()
                    loss = loss + s.anchor * kl
                if actor:
                    self.opt_a.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
                    self.opt_a.step()
                vl = value_step(
                    self.critic, self.opt_c, batch, [ret[i] for i in idx], s.max_grad_norm
                )
                with torch.no_grad():
                    w = mask.bool()
                    logs.setdefault("policy_loss", []).append(float(loss))
                    logs.setdefault("value_loss", []).append(vl)
                    logs.setdefault("clip_fraction", []).append(
                        float(((ratio - 1).abs() > s.clip)[w].float().mean())
                    )
                    logs.setdefault("approx_kl", []).append(float(((mu - m0) ** 2 / var)[w].mean()))
        mu_all = np.concatenate(self.means(trs))
        mu_start = np.concatenate(self._means(self.start, trs))
        return {k: float(np.mean(v)) for k, v in logs.items()} | {
            "explained_variance": explained,
            "kl_to_start": float(np.mean((mu_all - mu_start) ** 2) / var),
            "u_mean": float(mu_all.mean()),
            "u_std": float(mu_all.std()),
            "u_saturated": float((np.abs(mu_all) >= 0.99 * s.action_max).mean()),
            "kl_to_rule": float(np.mean((mu_all - s.u_rule) ** 2) / var),
            "u_realized_std": float(np.concatenate([tr.u for tr in trs]).std()),
            "mean_block_reward": float(np.concatenate([tr.reward for tr in trs]).mean()),
            "decisions": int(mu_all.size),
            "trajectories": len(trs),
        }


class TD3Learner:
    """TD3+BC with the same interface as PPOLearner (``actor``, ``critic`` and
    ``update``). A transition is one decision to the next of one flow; the
    last decision of a flow has no successor and is not learned from."""

    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = new_actor(width, spec)
        self.actor_target = copy.deepcopy(self.actor)
        self.q = nn.ModuleList(ValueNet(critic_width + 1, spec.critic_hidden) for _ in range(2))
        self.q_target = copy.deepcopy(self.q)
        self.critic = self.q  # saved as critic.pt by run_ppo
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.q.parameters(), lr=spec.critic_lr)
        self.replay: list[list[Trajectory]] = []
        self.steps = 0

    def _qs(self, qs: nn.ModuleList, c: torch.Tensor, u: torch.Tensor) -> list[torch.Tensor]:
        cu = torch.cat([c, (u / self.spec.action_max)[..., None]], dim=-1)
        return [q(cu) for q in qs]

    def update(
        self, trs: list[Trajectory], rng: np.random.Generator, *, actor: bool = True
    ) -> dict[str, float]:
        s = self.spec
        self.replay = [*self.replay, trs][-s.replay_iterations :]
        pool = [tr for batch in self.replay for tr in batch]
        logs: dict[str, list[float]] = {}
        for _ in range(s.updates_per_iteration):
            idx = rng.choice(len(pool), size=min(s.batch_trajectories, len(pool)), replace=False)
            batch = [pool[i] for i in idx]
            x, mask = pad([tr.inp for tr in batch])
            c, _ = pad([tr.critic for tr in batch])
            u, _ = pad([tr.u[:, None] for tr in batch])
            r, _ = pad([tr.reward[:, None] for tr in batch])
            g, _ = pad([tr.gamma[:, None] for tr in batch])
            u, r, g = u.squeeze(-1), r.squeeze(-1), g.squeeze(-1)
            m = mask[:, 1:]  # the decisions with a successor
            with torch.no_grad():
                mu_next = mean_u(s, batch, self.actor_target.sequence(x))[:, 1:]
                noise = (torch.randn_like(mu_next) * s.policy_noise).clamp(
                    -s.noise_clip, s.noise_clip
                )
                mu_next = (mu_next + noise).clamp(-s.action_max, s.action_max)
                q_next = torch.min(*self._qs(self.q_target, c[:, 1:], mu_next))
                y = r[:, :-1] + g[:, :-1] * q_next
            q1, q2 = self._qs(self.q, c[:, :-1], u[:, :-1])
            critic_loss = ((((q1 - y) ** 2 + (q2 - y) ** 2) * m).sum()) / m.sum()
            self.opt_c.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), s.max_grad_norm)
            self.opt_c.step()
            logs.setdefault("critic_loss", []).append(critic_loss.item())
            self.steps += 1
            if not actor or self.steps % s.policy_delay:
                continue
            mu = mean_u(s, batch, self.actor.sequence(x))
            q = self._qs(self.q, c, mu)[0]
            bc = (((mu - u) ** 2) * mask).sum() / mask.sum() / (2 * s.noise_sigma**2)
            actor_loss = -s.alpha * (q * mask).sum() / mask.sum() + bc
            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
            self.opt_a.step()
            with torch.no_grad():
                for net, target in ((self.actor, self.actor_target), (self.q, self.q_target)):
                    for w, wt in zip(net.parameters(), target.parameters(), strict=True):
                        wt.lerp_(w, s.target_tau)
            logs.setdefault("actor_q", []).append(((q * mask).sum() / mask.sum()).item())
            logs.setdefault("bc", []).append(bc.item())
        with torch.no_grad():
            x, mask = pad([tr.inp for tr in trs])
            mu_all = mean_u(s, trs, self.actor.sequence(x))[mask.bool()].numpy()
        return {k: float(np.mean(v)) for k, v in logs.items()} | {
            "u_mean": float(mu_all.mean()),
            "u_std": float(mu_all.std()),
            "u_saturated": float((np.abs(mu_all) >= 0.99 * s.action_max).mean()),
            "kl_to_rule": float(np.mean((mu_all - s.u_rule) ** 2) / (2 * s.noise_sigma**2)),
            "u_realized_std": float(np.concatenate([tr.u for tr in trs]).std()),
            "mean_block_reward": float(np.concatenate([tr.reward for tr in trs]).mean()),
            "decisions": int(mu_all.size),
            "trajectories": len(trs),
            "replay_trajectories": len(pool),
        }


class DAggerLearner:
    """algos "dagger" and "clone" with the same interface as PPOLearner.
    Its value network learns, as PPO's does, the return of the policy whose
    rollouts it sees (the student's, or for "clone" the teacher's), so that
    PPO can continue from both (init_from)."""

    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = new_actor(width, spec)
        self.critic = ValueNet(critic_width, spec.critic_hidden)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=spec.critic_lr)
        # Its own draws, so the imitation's batches are those it drew before
        # it had a value network.
        self.value_rng = np.random.default_rng(spec.seed + 1)
        self.replay: list[list[Trajectory]] = []

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        """The actor's head outputs on recorded decisions: reading the
        recorded previous u where the actor set it ("dagger"), its own where
        a teacher did ("clone")."""
        return self.actor.free(x) if self.spec.algo == "clone" else self.actor.pre(x)

    def _u(self, trs: list[Trajectory], pre: torch.Tensor) -> torch.Tensor:
        """The student's u as deployed, padded like ``pre`` (the actor's head
        outputs)."""
        return mean_u(self.spec, trs, torch.tanh(pre))

    def _values(self, trs: list[Trajectory]) -> dict[str, float]:
        """PPO's value regression (epochs of minibatches) on this
        iteration's rollouts, before and after."""
        s = self.spec
        with torch.no_grad():
            v_old = [self.critic(torch.from_numpy(tr.critic)).numpy() for tr in trs]
        _, ret = advantages(trs, v_old, s.gae_lambda)
        losses: list[float] = []
        for _ in range(s.epochs):
            order = self.value_rng.permutation(len(trs))
            for b in range(0, len(trs), s.batch_trajectories):
                idx = order[b : b + s.batch_trajectories]
                batch, rs = [trs[i] for i in idx], [ret[i] for i in idx]
                losses.append(value_step(self.critic, self.opt_c, batch, rs, s.max_grad_norm))
        return {
            "value_loss": float(np.mean(losses)),
            "explained_variance": explained_variance(ret, v_old),
        }

    def _gaps(self, trs: list[Trajectory]) -> dict[str, float]:
        """The actor against the teacher on ``trs``: mean |u - teacher|,
        the actor's mean u where the teacher pushes and where not, its mean
        belief p of a stock TCP flow where there is one and where not, and
        for "gate3" its mean belief q of copies (and p) where only copies
        share the link, and q elsewhere."""
        with torch.no_grad():
            x, mask = pad([tr.inp for tr in trs])
            pre = self._pre(x)
            w = mask.bool()
            mu, pq = self._u(trs, pre)[w].numpy(), torch.sigmoid(2 * pre)[w].numpy()
        p = pq if pq.ndim == 1 else pq[:, 0]
        y = np.concatenate([field_of(tr.teacher) for tr in trs])
        tcp = np.concatenate([field_of(tr.tcp) for tr in trs]) > 0.5
        push = y != np.float32(self.spec.u_rule)
        extra: dict[str, float] = {}
        if pq.ndim == 2:  # gate3: the belief in copies, where copies only are active and not
            q = pq[:, 1]
            cop = (np.concatenate([field_of(tr.copies) for tr in trs]) > 0.5) & ~tcp
            extra = {
                "q_where_copies": float(q[cop].mean()) if cop.any() else math.nan,
                "q_elsewhere": float(q[~cop].mean()) if (~cop).any() else math.nan,
                "p_where_copies": float(p[cop].mean()) if cop.any() else math.nan,
            }
        return extra | {
            "teacher_gap": float(np.abs(mu - y).mean()),
            "u_where_push": float(mu[push].mean()) if push.any() else math.nan,
            "u_where_rule": float(mu[~push].mean()) if (~push).any() else math.nan,
            "teacher_push": float(push.mean()),
            "teacher_u_where_push": float(y[push].mean()) if push.any() else math.nan,
            "p_where_tcp": float(p[tcp].mean()) if tcp.any() else math.nan,
            "p_elsewhere": float(p[~tcp].mean()) if (~tcp).any() else math.nan,
        }

    def _loss(self, batch: list[Trajectory], pre: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.spec.student.kind != "u":
            tcp, _ = pad([field_of(tr.tcp)[:, None] for tr in batch])
            if self.spec.student.kind == "gate3":
                cop, _ = pad([field_of(tr.copies)[:, None] for tr in batch])
                y = torch.cat([tcp, cop], dim=-1)
                m = mask.unsqueeze(-1)
            else:
                y, m = tcp.squeeze(-1), mask
            ce = nn.functional.binary_cross_entropy_with_logits(2 * pre, y, reduction="none")
            return (ce * m).sum() / m.expand_as(ce).sum()
        y, _ = pad([field_of(tr.teacher)[:, None] for tr in batch])
        return (((self._u(batch, pre) - y.squeeze(-1)) ** 2) * mask).sum() / mask.sum()

    def update(
        self, trs: list[Trajectory], rng: np.random.Generator, *, actor: bool = True
    ) -> dict[str, float]:
        s = self.spec
        before = self._gaps(trs)  # on this iteration's rollouts, before learning from them
        self.replay = [*self.replay, trs][-s.replay_iterations :]
        pool = [tr for batch in self.replay for tr in batch]
        losses: list[float] = []
        for _ in range(s.updates_per_iteration if actor else 0):
            idx = rng.choice(len(pool), size=min(s.batch_trajectories, len(pool)), replace=False)
            batch = [pool[i] for i in idx]
            x, mask = pad([tr.inp for tr in batch])
            loss = self._loss(batch, self._pre(x), mask)
            self.opt_a.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
            self.opt_a.step()
            losses.append(loss.item())
        after = self._gaps(trs)
        with torch.no_grad():
            x, mask = pad([tr.inp for tr in trs])
            mu_all = self._u(trs, self._pre(x))[mask.bool()].numpy()
        return (
            {f"{k}_before": v for k, v in before.items()}
            | after
            | self._values(trs)
            | {
                "imitation_loss": float(np.mean(losses)) if losses else math.nan,
                "u_mean": float(mu_all.mean()),
                "u_std": float(mu_all.std()),
                "u_saturated": float((np.abs(mu_all) >= 0.99 * s.action_max).mean()),
                "kl_to_rule": float(np.mean((mu_all - s.u_rule) ** 2) / (2 * s.noise_sigma**2)),
                "u_realized_std": float(np.concatenate([tr.u for tr in trs]).std()),
                "mean_block_reward": float(np.concatenate([tr.reward for tr in trs]).mean()),
                "decisions": int(mu_all.size),
                "trajectories": len(trs),
                "replay_trajectories": len(pool),
            }
        )


def field_of(x: F32 | None) -> F32:
    if x is None:
        raise ValueError(
            "a trajectory lacks what its learner reads: the teacher's labels (algo"
            ' "dagger") or the rules\' u (a gate student)'
        )
    return x


LEARNERS: Final = {
    "ppo": PPOLearner,
    "td3bc": TD3Learner,
    "dagger": DAggerLearner,
    "clone": DAggerLearner,
}


def make_learner(
    spec: PPOSpec, width: int, critic_width: int, source: PPOSpec | None = None
) -> PPOLearner | TD3Learner | DAggerLearner:
    """The spec's learner, started from what init_paths names."""
    learner = LEARNERS[spec.algo](width, critic_width, spec)
    actor_pt, critic_pt = init_paths(spec, source)
    if actor_pt is not None:
        state = torch.load(actor_pt, weights_only=True)
        learner.actor.load_state_dict(state)
        if isinstance(learner, TD3Learner):
            learner.actor_target.load_state_dict(state)
    if critic_pt is not None and isinstance(learner, PPOLearner):
        learner.critic.load_state_dict(torch.load(critic_pt, weights_only=True))
    if isinstance(learner, PPOLearner):
        learner.start = copy.deepcopy(learner.actor).requires_grad_(False)
    return learner
