"""Recurrent PPO for net2 on a frozen net1, starting from the fixed rule.

The actor (recurrent.py) reads net1's pooled bins and its own previous
action through a GRU and holds each decision for clamp(min RTT, 10, 50 ms).
The value network sees the true LCFs with net1's output during training
only and is never exported. Each iteration exports the policy, runs each
of a few training links K times (the same seed, so the same path and loss
draws, with different exploration noise), and updates on those rollouts.
The reward is the per-step log utility from ground truth (score.utility),
per flow. Copies of the policy on one bottleneck (self-play) each learn
from their own, or, with copies_reward "team", from their mean: copies are
scored all running the policy, and a flow that learns from its own utility
gains by pushing harder than its copies, which lowers what they all get.
Returns are discounted by elapsed time, gamma =
exp(-dt / tau), and each decision's advantage has its group's mean at the
same decision taken off, on top of the value function: link-to-link spread
dominates the returns, and the K rollouts of a link share it.
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import nn

from expliccit_env.policy import OBS_DIM
from expliccit_env.spec import (
    AgentCC,
    Condition,
    EpisodeSpec,
    OnnxPolicy,
    PathSpec,
    StockName,
    TcpCross,
    UdpCross,
)
from expliccit_env.train import fit, score
from expliccit_env.train.export import to_onnx
from expliccit_env.train.online import _arms, _run  # pyright: ignore[reportPrivateUsage]
from expliccit_env.train.recurrent import (
    OBS_MIN_RTT,
    HeldPolicyGraph,
    RecurrentActor,
    decision_steps,
)
from expliccit_env.train.spec import LCF_UNITS, Net2Spec
from expliccit_env.uml import UmlBuild

type F32 = npt.NDArray[np.float32]

# Per-step reward: the validation tables' log utility (delay weight 0.5,
# loss weight 4).
_REWARD: Net2Spec = Net2Spec(reward="log_utility")
_STEP_S: Final = 0.01  # the guest's step interval
_UTILITY_LABELS = ("receive_rate_bps", "capacity_bps", "queue_delay_mean_ns", "base_rtt_ns")


class PPOSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_version: Literal[1] = 1
    paths_from: Path  # training arms: a campaign's sweep and conditions
    eval_from: Path | None = None  # validation arms, every path under every condition
    net1_from: Path  # a training run's net1.pt, kept frozen
    # The fixed rule's exported policy, run once on the evaluation arms so
    # every evaluation is paired with it by link and condition.
    rule_policy: Path | None = None
    u_rule: float = 0.125  # the start: the actor answers this u
    view: Literal["bins", "means"] = "bins"  # "means": the ablation without distributions
    input_bins: Annotated[int, Field(ge=2, le=256)] | None = 32
    action_max: Annotated[float, Field(gt=0, le=8)] = 3.0
    hidden: PositiveInt = 64
    memory: bool = True  # False: the ablation without memory
    critic_hidden: PositiveInt = 256
    critic_truth: bool = True  # False: the ablation with a plain critic
    # Iterations that update only the value network, so the actor does not
    # follow a blank critic.
    critic_warmup: Annotated[int, Field(ge=0)] = 3
    iterations: PositiveInt = 40
    links_per_iteration: PositiveInt = 8
    group: PositiveInt = 4  # K rollouts per link
    duration_ms: PositiveInt = 30_000
    noise_sigma: Annotated[float, Field(gt=0, le=2)] = 0.1  # exploration, log2 of the window
    tau_s: Annotated[float, Field(gt=0)] = 3.0
    gae_lambda: Annotated[float, Field(ge=0, le=1)] = 0.95
    lr: Annotated[float, Field(gt=0)] = 3e-4
    critic_lr: Annotated[float, Field(gt=0)] = 1e-3
    clip: Annotated[float, Field(gt=0, lt=1)] = 0.2
    epochs: PositiveInt = 4
    batch_trajectories: PositiveInt = 16
    max_grad_norm: Annotated[float, Field(gt=0)] = 0.5
    # Share of links whose condition is replaced by competition: Cubic or
    # BBR from 0 s, or 2 to 4 copies of the policy itself.
    compete: Annotated[float, Field(ge=0, le=1)] = 0.5
    copies: tuple[int, ...] = (2, 3, 4)
    # What a flow learns from where several flows run the policy (copies):
    # its own utility, or the mean of theirs at each step. A flow facing
    # other controllers always learns from its own.
    copies_reward: Literal["own", "team"] = "own"
    eval_every: PositiveInt = 5
    seed: Annotated[int, Field(ge=0)] = 0
    keep_rollouts: bool = False
    time_budget_s: PositiveInt | None = None

    @classmethod
    def parse(cls, text: str) -> PPOSpec:
        return cls.model_validate_json(text)


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One flow's decisions in one rollout."""

    group: tuple[int, int]  # (link of the iteration, flow: 0 observed, i + 1 cross i)
    inp: F32  # [T, width + 1] the actor's inputs: net1's view, previous u / max
    critic: F32  # [T, C] the value network's inputs
    u: F32  # [T] realized u of each decision
    reward: F32  # [T] the block's summed per-step reward after the decision
    gamma: F32  # [T] exp(-dt / tau) to the next decision


def _agent(policy: Path, sigma: float, noise_seed: int | None) -> AgentCC:
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    return AgentCC(
        policy=OnnxPolicy(path=policy, sha256=sha), noise_sigma=sigma, noise_seed=noise_seed
    )


def _competition(
    rng: np.random.Generator, copies: tuple[int, ...], copy: AgentCC | None
) -> list[TcpCross | UdpCross]:
    """Cubic or BBR from 0 s, or n - 1 copies of the policy (n in copies)."""
    choices = ["cubic", "bbr", *(f"copies{n}" for n in copies if copy is not None)]
    pick = choices[int(rng.integers(len(choices)))]
    if not pick.startswith("copies"):
        stock: StockName = "cubic" if pick == "cubic" else "bbr"
        return [TcpCross(cc=stock, start_ms=0)]
    assert copy is not None
    n = int(pick.removeprefix("copies"))
    return [TcpCross(cc=copy, start_ms=0) for _ in range(n - 1)]


def _group_episodes(
    arm: tuple[PathSpec, Condition],
    policy: Path,
    spec: PPOSpec,
    rng: np.random.Generator,
    compete: bool,
) -> list[EpisodeSpec]:
    """K rollouts of one link: the same seed, different noise seeds."""
    path, cond = arm
    seed, base = int(rng.integers(0, 2**63)), int(rng.integers(0, 2**62))
    template = _agent(policy, spec.noise_sigma, None)
    cross = _competition(rng, spec.copies, template) if compete else cond.flows(path.trace)

    def own(c: TcpCross | UdpCross, noise: int) -> TcpCross | UdpCross:
        if isinstance(c, TcpCross) and isinstance(c.cc, AgentCC):
            return c.model_copy(update={"cc": c.cc.model_copy(update={"noise_seed": noise})})
        return c

    return [
        EpisodeSpec(
            path=path.model_copy(update={"loss": cond.loss}),
            cross=[own(c, base + k + 7919 * (i + 1)) for i, c in enumerate(cross)],
            cc=template.model_copy(update={"noise_seed": base + k}),
            duration_ms=spec.duration_ms,
            seed=seed,
            steps=True,
        )
        for k in range(spec.group)
    ]


def _batch(
    arms: list[tuple[PathSpec, Condition]], policy: Path, spec: PPOSpec, rng: np.random.Generator
) -> tuple[list[EpisodeSpec], dict[str, int]]:
    """One iteration's rollouts: links_per_iteration links, K each, and each
    episode's link index (its group)."""
    eps: list[EpisodeSpec] = []
    groups: dict[str, int] = {}
    for j in range(spec.links_per_iteration):
        arm = arms[int(rng.integers(len(arms)))]
        for e in _group_episodes(arm, policy, spec, rng, bool(rng.random() < spec.compete)):
            groups[e.episode_id()] = j
            eps.append(e)
    return eps, groups


def _views(net1: fit.Net1, spec: PPOSpec) -> tuple[nn.Module | None, nn.Module]:
    view = fit._view(net1, spec.view, spec.input_bins if spec.view == "bins" else None)  # pyright: ignore[reportPrivateUsage]
    return view, fit.window_target(net1)


def _width(net1: fit.Net1, spec: PPOSpec) -> int:
    return fit.net2_width(
        net1,
        Net2Spec(
            view=spec.view,
            input_bins=spec.input_bins if spec.view == "bins" else None,
            action="bdp",
        ),
    )


def _read(path: Path, eids: set[str], lcfs: tuple[str, ...]) -> pa.Table | None:
    if not path.exists():
        return None
    labels = {f"label_{n}" for n in (*_UTILITY_LABELS, "drop_fraction", *lcfs)}
    names = pq.read_schema(path).names
    cols = [c for c in names if c in {"episode_id", "flow", "step", "t_ns", "obs", "cwnd"} | labels]
    t = pq.read_table(path, columns=cols)
    keep = np.isin(np.asarray(t["episode_id"].to_pylist()), list(eids))
    return t.filter(pa.array(keep))


def _labels(t: pa.Table) -> dict[str, npt.NDArray[np.float32]]:
    return {
        c.removeprefix("label_"): np.asarray(t[c].to_numpy(zero_copy_only=False), np.float32)
        for c in t.column_names
        if c.startswith("label_")
    }


def _by_step(t: pa.Table) -> pa.Table:
    order = np.argsort(np.asarray(t["step"].to_numpy(zero_copy_only=False)), kind="stable")
    return t.take(pa.array(order, pa.int64()))


def team_utility(utilities: list[npt.NDArray[np.float32]]) -> list[npt.NDArray[np.float32]]:
    """Each flow's per-step utility replaced by the flows' mean at the same
    step (copies start together, so step k is the same instant for all);
    a step only some flows reached averages those."""
    n = max(u.size for u in utilities)
    stack = np.stack([np.pad(u, (0, n - u.size), constant_values=np.nan) for u in utilities])
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    return [mean[: u.size] for u in utilities]


@torch.no_grad()
def _trajectory(
    t: pa.Table,
    utility: npt.NDArray[np.float32],
    net1: fit.Net1,
    *,
    view: nn.Module | None,
    target: nn.Module,
    spec: PPOSpec,
    group: tuple[int, int],
) -> Trajectory | None:
    """One flow's decisions; ``t`` is its steps in step order and
    ``utility`` the reward of each of those steps."""
    n = t.num_rows
    flat = cast("pa.FixedSizeListArray[Any, Any]", t["obs"].combine_chunks()).flatten()
    obs = np.asarray(flat, np.float32).reshape(n, OBS_DIM)
    cwnd = np.asarray(t["cwnd"].to_numpy(zero_copy_only=False), np.float32)
    t_s = np.asarray(t["t_ns"].to_numpy(zero_copy_only=False), np.float64) / 1e9
    labels = _labels(t)
    d = decision_steps(obs[:, OBS_MIN_RTT])
    d = d[d + 1 < n]  # a decision's realized window is the next step's
    if d.size < 2:
        return None
    z = torch.from_numpy(net1.beliefs(obs[d]))
    x = (z if view is None else view(z)).numpy()
    lb = target(z).numpy()
    u = (np.log2(np.maximum(cwnd[d + 1], 1.0)) - lb).astype(np.float32)
    prev = np.concatenate([[0.0], u[:-1]]).astype(np.float32)
    # Each step's reward weighs 1 - exp(-10 ms / tau), so a return is a
    # discounted mean utility, of order 1 whatever tau is.
    r = utility * -math.expm1(-_STEP_S / spec.tau_s)
    ends = np.append(d[1:], n - 1)
    cum = np.concatenate([[0.0], np.cumsum(r)])
    reward = (cum[ends + 1] - cum[d + 1]).astype(np.float32)  # steps d + 1 .. next decision
    gamma = np.exp(-(t_s[ends] - t_s[d]) / spec.tau_s).astype(np.float32)
    frac = (np.arange(d.size) / d.size).astype(np.float32)[:, None]
    inp = np.concatenate([x, (prev / spec.action_max)[:, None]], axis=1).astype(np.float32)
    parts = [x, (prev / spec.action_max)[:, None], frac]
    if spec.critic_truth:
        y = np.stack([labels[k][d] * np.float32(LCF_UNITS[k][0]) for k in net1.lcfs], axis=1)
        ys = net1.lcf_scaler(torch.from_numpy(y)).numpy()
        parts.insert(0, np.where(np.isfinite(ys), ys, 0.0).astype(np.float32))
    critic = np.concatenate(parts, axis=1).astype(np.float32)
    return Trajectory(group, inp, critic, u, reward, gamma)


def _trajectories(
    rollouts: Path, groups: dict[str, int], net1: fit.Net1, spec: PPOSpec
) -> list[Trajectory]:
    """Every agent flow of every ok rollout episode."""
    view, target = _views(net1, spec)
    out: list[Trajectory] = []
    flows_of: dict[str, list[tuple[int, pa.Table]]] = {}
    for shard in sorted(rollouts.glob("shards/*")):
        eps = pq.read_table(shard / "episodes.parquet", columns=["episode_id", "status"])
        ok = {
            str(e)
            for e, s in zip(eps["episode_id"].to_pylist(), eps["status"].to_pylist(), strict=True)
            if s == "ok"
        }
        for name, flow in (("steps.parquet", False), ("cross_steps.parquet", True)):
            t = _read(shard / name, ok, net1.lcfs)
            if t is None or t.num_rows == 0:
                continue
            eids = np.asarray(t["episode_id"].to_pylist())
            flows = np.asarray(t["flow"].to_pylist()) + 1 if flow else np.zeros(len(eids), int)
            for e in np.unique(eids):
                for f in np.unique(flows[eids == e]):
                    rows = np.flatnonzero((eids == e) & (flows == f))
                    flows_of.setdefault(str(e), []).append(
                        (int(f), _by_step(t.take(pa.array(rows, pa.int64()))))
                    )
    # Only agent flows record steps, so an episode with several is copies.
    for e, fl in flows_of.items():
        utils = [np.nan_to_num(score.utility(_labels(t), _REWARD)) for _, t in fl]
        if spec.copies_reward == "team" and len(fl) > 1:
            utils = team_utility(utils)
        for (f, t), u in zip(fl, utils, strict=True):
            tr = _trajectory(t, u, net1, view=view, target=target, spec=spec, group=(groups[e], f))
            if tr is not None:
                out.append(tr)
    return out


class _Critic(nn.Module):
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


def _advantages(
    trs: list[Trajectory], values: list[F32], lam: float
) -> tuple[list[F32], list[F32]]:
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


def _pad(xs: list[F32]) -> tuple[torch.Tensor, torch.Tensor]:
    t = max(x.shape[0] for x in xs)
    out = np.zeros((len(xs), t, *xs[0].shape[1:]), np.float32)
    mask = np.zeros((len(xs), t), np.float32)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = 1
    return torch.from_numpy(out), torch.from_numpy(mask)


class _Learner:
    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = RecurrentActor(width, spec.hidden, memory=spec.memory)
        self.actor.clone_rule(spec.u_rule, spec.action_max)
        self.critic = _Critic(critic_width, spec.critic_hidden)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=spec.critic_lr)

    def means(self, trs: list[Trajectory]) -> list[F32]:
        with torch.no_grad():
            x, mask = _pad([tr.inp for tr in trs])
            mu = (self.actor.sequence(x) * self.spec.action_max).numpy()
        return [mu[i, : int(mask[i].sum())] for i in range(len(trs))]

    def values(self, trs: list[Trajectory]) -> list[F32]:
        with torch.no_grad():
            return [self.critic(torch.from_numpy(tr.critic)).numpy() for tr in trs]

    def update(
        self, trs: list[Trajectory], rng: np.random.Generator, *, actor: bool = True
    ) -> dict[str, float]:
        s = self.spec
        v_old = self.values(trs)
        mu_old = self.means(trs)
        adv, ret = _advantages(trs, v_old, s.gae_lambda)
        flat = np.concatenate(adv)
        scale = float(flat.std()) + 1e-8
        adv = [((a - flat.mean()) / scale).astype(np.float32) for a in adv]
        allret, allv = np.concatenate(ret), np.concatenate(v_old)
        explained = 1.0 - float(np.var(allret - allv)) / max(float(np.var(allret)), 1e-12)
        logs: dict[str, list[float]] = {}
        var = 2 * s.noise_sigma**2
        for _ in range(s.epochs):
            order = rng.permutation(len(trs))
            for b in range(0, len(trs), s.batch_trajectories):
                idx = order[b : b + s.batch_trajectories]
                x, mask = _pad([trs[i].inp for i in idx])
                u, _ = _pad([trs[i].u[:, None] for i in idx])
                a, _ = _pad([adv[i][:, None] for i in idx])
                m0, _ = _pad([mu_old[i][:, None] for i in idx])
                u, a, m0 = u.squeeze(-1), a.squeeze(-1), m0.squeeze(-1)
                mu = self.actor.sequence(x) * s.action_max
                ratio = torch.exp(((u - m0) ** 2 - (u - mu) ** 2) / var)
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - s.clip, 1 + s.clip) * a)
                loss = -(surr * mask).sum() / mask.sum()
                if actor:
                    self.opt_a.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
                    self.opt_a.step()
                c, cmask = _pad([trs[i].critic for i in idx])
                g, _ = _pad([ret[i][:, None] for i in idx])
                vl = (((self.critic(c) - g.squeeze(-1)) ** 2) * cmask).sum() / cmask.sum()
                self.opt_c.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), s.max_grad_norm)
                self.opt_c.step()
                with torch.no_grad():
                    w = mask.bool()
                    logs.setdefault("policy_loss", []).append(float(loss))
                    logs.setdefault("value_loss", []).append(float(vl))
                    logs.setdefault("clip_fraction", []).append(
                        float(((ratio - 1).abs() > s.clip)[w].float().mean())
                    )
                    logs.setdefault("approx_kl", []).append(float(((mu - m0) ** 2 / var)[w].mean()))
        mu_all = np.concatenate(self.means(trs))
        return {k: float(np.mean(v)) for k, v in logs.items()} | {
            "explained_variance": explained,
            "u_mean": float(mu_all.mean()),
            "u_std": float(mu_all.std()),
            "u_saturated": float((np.abs(mu_all) >= 0.99 * s.action_max).mean()),
            "kl_to_rule": float(np.mean((mu_all - s.u_rule) ** 2) / var),
            "u_realized_std": float(np.concatenate([tr.u for tr in trs]).std()),
            "mean_block_reward": float(np.concatenate([tr.reward for tr in trs]).mean()),
            "decisions": int(mu_all.size),
            "trajectories": len(trs),
        }


def policy_graph(net1: fit.Net1, actor: RecurrentActor, spec: PPOSpec) -> HeldPolicyGraph:
    base = fit.policy_graph(
        net1,
        actor,
        spec.action_max,
        action="bdp",
        view=spec.view,
        input_bins=spec.input_bins if spec.view == "bins" else None,
    )
    return HeldPolicyGraph(base, actor).eval()


def _export(path: Path, net1: fit.Net1, actor: RecurrentActor, spec: PPOSpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_onnx(policy_graph(net1, actor, spec)))
    actor.train()
    return path


def _scores(d: Path) -> dict[tuple[str, str], float]:
    """Mean log utility per (link, condition) of an evaluation run."""
    per = score.episode_scores([d], _REWARD)
    return {(k[1], k[2]): float(np.mean(v)) for k, v in per.items()}


def _evaluate(
    d: Path,
    policy: Path,
    arms: list[tuple[PathSpec, Condition]],
    spec: PPOSpec,
    run: Any,
) -> dict[tuple[str, str], float]:
    cc = AgentCC(
        policy=OnnxPolicy(path=policy, sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    )
    eps = [
        EpisodeSpec(
            path=p.model_copy(update={"loss": c.loss}),
            cross=c.flows(p.trace),
            cc=cc,
            duration_ms=spec.duration_ms,
            seed=7 + i,
            steps=True,
        )
        for i, (p, c) in enumerate(arms)
    ]
    run(d, eps)
    out = _scores(d)
    if not spec.keep_rollouts:
        shutil.rmtree(d / "shards", ignore_errors=True)
    return out


def _paired(
    pol: dict[tuple[str, str], float], rule: dict[tuple[str, str], float]
) -> dict[str, Any]:
    keys = sorted(pol.keys() & rule.keys())
    diff = np.array([pol[k] - rule[k] for k in keys])
    by: dict[str, list[float]] = {}
    for k, v in zip(keys, diff.tolist(), strict=True):
        by.setdefault(k[1], []).append(v)
    return {
        "log_utility": float(np.mean([pol[k] for k in keys])) if keys else math.nan,
        "minus_rule": float(diff.mean()) if keys else math.nan,
        "pairs": len(keys),
        "minus_rule_by_condition": {c: float(np.mean(v)) for c, v in sorted(by.items())},
    }


def run_ppo(
    spec: PPOSpec, out: Path, build: UmlBuild, *, workers: int, scratch: Path | None = None
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ppo_spec.json").write_text(spec.model_dump_json(indent=2))
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    net1, net1_spec = fit.load_net1(spec.net1_from)
    fit.save_net1(net1, net1_spec, out / "net1.pt")
    width = _width(net1, spec)
    critic_width = width + 2 + (len(net1.lcfs) if spec.critic_truth else 0)
    learner = _Learner(width, critic_width, spec)
    train_arms = _arms(spec.paths_from, rng)
    eval_arms = _arms(spec.eval_from, rng) if spec.eval_from else []

    def run(d: Path, eps: list[EpisodeSpec]) -> dict[str, Any]:
        return _run(d, eps, build, workers, scratch)

    history: list[dict[str, Any]] = []
    rule_scores: dict[tuple[str, str], float] = {}
    if eval_arms and spec.rule_policy is not None:
        rule_scores = _evaluate(out / "rule-eval", spec.rule_policy, eval_arms, spec, run)
    if eval_arms:
        start = _export(out / "initial" / "policy.onnx", net1, learner.actor, spec)
        pol = _evaluate(out / "initial" / "eval", start, eval_arms, spec, run)
        history.append({"iteration": -1, "eval": _paired(pol, rule_scores)})
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    began_all, slowest = time.monotonic(), 0.0
    for it in range(spec.iterations):
        began = time.monotonic()
        if spec.time_budget_s is not None and began - began_all + slowest > spec.time_budget_s:
            break
        it_dir = out / f"iter{it:03d}"
        policy = _export(it_dir / "policy.onnx", net1, learner.actor, spec)
        eps, groups = _batch(train_arms, policy, spec, rng)
        summary = next(iter(run(it_dir / "rollouts", eps).values()))
        trs = _trajectories(it_dir / "rollouts", groups, net1, spec)
        entry: dict[str, Any] = {"iteration": it, "rollouts": summary}
        if trs:
            entry["learner"] = learner.update(trs, rng, actor=it >= spec.critic_warmup)
        if not spec.keep_rollouts:
            shutil.rmtree(it_dir / "rollouts" / "shards", ignore_errors=True)
        if eval_arms and ((it + 1) % spec.eval_every == 0 or it + 1 == spec.iterations):
            updated = _export(it_dir / "policy-updated.onnx", net1, learner.actor, spec)
            pol = _evaluate(it_dir / "eval", updated, eval_arms, spec, run)
            entry["eval"] = _paired(pol, rule_scores)
            torch.save(learner.actor.state_dict(), it_dir / "actor.pt")
        entry["wall_s"] = time.monotonic() - began
        slowest = max(slowest, entry["wall_s"])
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    torch.save(learner.actor.state_dict(), out / "actor.pt")
    torch.save(learner.critic.state_dict(), out / "critic.pt")
    _export(out / "policy.onnx", net1, learner.actor, spec)
    return {"iterations": history}
