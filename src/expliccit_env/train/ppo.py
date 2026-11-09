"""A recurrent net2 trained in closed loop on a frozen net1, by PPO,
TD3+BC or DAgger (PPOSpec.algo).

The actor (recurrent.py) reads net1's output (pooled bins, or means) and
its own previous u through a GRU, and holds each decision for
clamp(min RTT, 10 ms, 50 ms). It answers u, the window's log2 ratio to the
believed BDP, and starts as the fixed rule (u_rule everywhere) unless the
spec sets base or init_actor. Each iteration exports the policy, runs each
of a few training links K times (the same seed, so the same path and loss
draws, with different exploration noise), and learns from those rollouts.
A share of the links (compete) gets competition in place of its
condition: Cubic or BBRv1, or copies of the policy itself (PPOSpec
competition sets when the stock flows start and how many there are).

- "ppo": the clipped policy gradient on each iteration's rollouts. The
  reward is the per-step log utility from ground truth (score.utility),
  per flow. Copies of the policy each learn from their own utility or,
  with copies_reward "team", from the copies' mean: a copy that learns
  from its own gains by pushing harder than the others, which lowers what
  they all get. Returns are discounted by elapsed time, gamma =
  exp(-dt / tau), and each decision's advantage has its group's mean at the
  same decision taken off, on top of the value function: the spread
  between links dominates the returns, and the K rollouts of a link share
  it.
- "td3bc": TD3+BC off a replay of the last few iterations' rollouts, with
  the same reward.
- "dagger": DAgger from a privileged teacher, a rule that knows which
  other flows share the link. Every decision the actor makes is labelled
  with the teacher's u, so it learns from a label at each decision rather
  than from a return. The student (PPOSpec.student) learns u itself ("u"),
  its belief that a stock TCP flow shares the link ("gate"), or that
  belief and its belief that copies do ("gate3"); a gate student's u moves
  from the fixed rule toward the teacher's rules by its beliefs.

The critics (PPO's value network, TD3+BC's Q networks) see the actor's
inputs, the true LCFs (critic_truth) and, with critic_flows, how many
flows of each kind share the link. They are used in training only and
never exported.
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
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
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from torch import nn

from expliccit_env.policy import OBS_DIM
from expliccit_env.spec import (
    MAX_CROSS,
    STOCK_NAMES,
    AgentCC,
    Condition,
    EpisodeSpec,
    OnnxPolicy,
    PathSpec,
    PeriodicLevels,
    StockName,
    TcpCross,
    Trace,
    UdpCross,
)
from expliccit_env.train import fit, rollouts, score
from expliccit_env.train.export import to_onnx
from expliccit_env.train.graph import Front, MeasuredBdp, PolicyGraph
from expliccit_env.train.recurrent import (
    OBS_MIN_RTT,
    ConcatActor,
    EnsembleActor,
    Held,
    RecurrentActor,
    decision_steps,
)
from expliccit_env.train.share import ACTION_MAX as SHARE_ACTION_MAX
from expliccit_env.train.share import CoopRule, ShareRule
from expliccit_env.train.spec import LCF_UNITS, Net2Spec
from expliccit_env.train.traffic import CLASSES
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
    # The fixed rule's u: where the actor starts, and what the teachers and
    # gate students fall back to.
    u_rule: float = 0.125
    view: Literal["bins", "means"] = "bins"  # "means": the ablation without distributions
    input_bins: Annotated[int, Field(ge=2, le=256)] | None = 32
    action_max: Annotated[float, Field(gt=0, le=8)] = 3.0
    hidden: PositiveInt = 64
    memory: bool = True  # False: the ablation without memory
    critic_hidden: PositiveInt = 256
    critic_truth: bool = True  # False: the ablation with a plain critic
    # Iterations before the actor's first update, so it does not follow a
    # blank critic; the critics learn alone ("dagger" only gathers labels).
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
    # BBRv1 from 0 s, or n copies of the policy in all, n drawn from copies.
    compete: Annotated[float, Field(ge=0, le=1)] = 0.5
    # A share of the links where the flow runs alone (the link's random
    # loss kept, no cross flows), drawn before compete; the rest keep the
    # paths' own conditions. With compete high, a flow is alone on few
    # training links (round 10: 8%) but on about half the evaluation's, and
    # a policy that pushes everywhere is the optimum of the wrong mixture.
    alone: Annotated[float, Field(ge=0, le=1)] = 0.0
    # A share of the links, drawn after compete, whose traffic leaves the
    # flow unable to see the capacity part of the time (_info): a stock flow
    # that leaves, UDP that switches on and off, or a stock TCP flow that
    # does. Where the flow keeps the link busy (96% of validation steps for
    # the fixed rule alone), net1's beliefs are narrow and a probe rarely
    # pays; these links are where it can.
    info: Annotated[float, Field(ge=0, le=1)] = 0.0
    # A share of the links, drawn on its own, whose capacity alternates
    # between half and twice the link's mean, each level held 1 to 5 s
    # (uniform), so the capacity often rises above what the flow last saw.
    rise: Annotated[float, Field(ge=0, le=1)] = 0.0
    copies: tuple[int, ...] = (2, 3, 4)
    # The share of competition links that hold copies rather than a stock
    # flow; None keeps the original draw (copies as likely as each of Cubic
    # and BBRv1, one in four or two in four). Where copies and a stock flow
    # look alike to net1, this share is the prior the shared policy weighs.
    copies_share: Annotated[float, Field(ge=0, le=1)] | None = None
    # When the competing stock flows start. "at-start": one Cubic or BBRv1
    # flow from 0 s. "broad": that flow starts at a uniform time in the
    # first 10 s, and half the time a second one (Cubic, BBRv1 or Reno)
    # joins at its own such time, as the evaluation's competition does
    # (BBRv1 from 4 s, Cubic from 2 s and 10 s). "diverse": as "broad", but
    # each stock flow is any controller except the held-out BBRv3 and
    # Westwood, the common ones more often (DIVERSE_WEIGHT). "dynamic": as
    # "diverse", but each stock flow starts at 0 s or anywhere in the first
    # 60% of the episode and a third of them leave early (_window), as in
    # net1's broad data, so longer episodes hold arrivals and departures.
    # Copies start at 0 s in all four.
    competition: Literal["at-start", "broad", "diverse", "dynamic"] = "at-start"
    # Hold each exploration draw for this many steps (10 ms each), so that a
    # push lasting several RTTs, and the others' response to it, can be
    # sampled; 1 draws anew at every step.
    noise_hold_steps: Annotated[int, Field(ge=1, le=1000)] = 1
    # What a flow learns from where several flows run the policy (copies):
    # its own utility, or the mean of theirs at each step. A flow facing
    # other controllers always learns from its own.
    copies_reward: Literal["own", "team"] = "own"
    # The value network also sees, at each decision, how many other copies
    # of the policy, stock TCP flows and UDP flows share the link then:
    # privileged, like the true LCFs, and never shown to the actor.
    critic_flows: bool = False
    # The learner. "ppo": the clipped policy gradient on the rollouts of
    # each iteration. "td3bc": TD3+BC (Fujimoto and Gu, 2021) off a replay
    # of the last replay_iterations iterations' rollouts: twin Q networks
    # on the value network's inputs and the action, and the recurrent actor
    # following the gradient of Q at each state, weighted alpha, against its
    # squared distance to the actions taken in units of the exploration
    # noise, (mu - u)^2 / (2 sigma^2). A return here is a discounted mean
    # utility, so Q needs no rescaling (TD3+BC's alpha / mean |Q| would
    # blow up where utilities are near 0), and each iteration moves the
    # actor by about alpha sigma^2 dQ/du: small steps around the data.
    # "dagger": DAgger (Ross, Gordon and Bagnell, 2011) with a privileged
    # teacher. The actor acts in every rollout; each of its decisions is
    # labelled with the teacher's u, and the actor learns from the labels
    # of the last replay_iterations iterations (updates_per_iteration
    # batches). For students "u" and "gate" the teacher is the share rule
    # (share_k) while a stock TCP flow shares the link and the fixed rule
    # (u_rule) otherwise; "gate3" has its own teacher (below). The teacher
    # knows the other flows, which the actor must infer from net1's output.
    algo: Literal["ppo", "td3bc", "dagger"] = "ppo"
    # What a "dagger" student learns. "u": u itself, by squared error on the
    # teacher's u. "gate": only the probability p that a stock TCP flow
    # shares the link, by cross-entropy on that label at every decision;
    # its u is u_rule + p (share rule's u - u_rule), the teacher's u with
    # its switch replaced by the student's belief. It starts at p = 0.02.
    # "gate3": two beliefs, p of a stock TCP flow and q of copies of this
    # policy, each by cross-entropy on its label; the teacher is the share
    # rule with stock TCP, the mode rule's cooperative window with copies
    # only (train/share.py CoopRule), and the fixed rule otherwise. Its u
    # is u_rule + p (share - u_rule) + (1 - p) q (coop - u_rule); both
    # beliefs start at 0.02.
    student: Literal["u", "gate", "gate3"] = "u"
    # A gate student's policy ("gate" or "gate3") acts on each belief p
    # only past this threshold t: p' = clamp((p - t) / (1 - t), 0, 1), as
    # copies lose more per unit of push than a flow against Cubic gains. Set
    # at export (experiments/gate_threshold_export.py), or here so that the
    # rollouts DAgger labels come from the policy that would be deployed;
    # the beliefs themselves are learned without it.
    gate_threshold: Annotated[float, Field(ge=0, lt=1)] = 0.0
    # A "gate3" student's threshold for q, its belief in copies, where it
    # differs from p's; None applies gate_threshold to both.
    gate_threshold_q: Annotated[float, Field(ge=0, lt=1)] | None = None
    # What a "gate3" student does where it believes copies share the link.
    # "coop": moves toward the cooperative window by q. "rule": q vetoes
    # the push instead, u = u_rule + p (1 - q) (share - u_rule), so taking
    # a buffer-filling flow for copies costs at most the push, while the
    # cooperative window yields to it. The beliefs train the same either
    # way (cross-entropy on their labels); this sets the policy that acts.
    gate3_copies: Literal["coop", "rule"] = "coop"
    # Start the actor from a saved state (an earlier run's actor.pt) instead
    # of the rule; its width and hidden size must match this spec's.
    init_actor: Path | None = None
    # Start PPO's value network from an earlier run's critic.pt, to continue
    # training across runs.
    init_critic: Path | None = None
    # Continue an earlier run (its output directory), the handoff from a
    # taught stage to an RL one: the actor from its actor.pt and, for
    # "ppo", the value network from its critic.pt where it has one (a
    # "dagger" run trains one on its rollouts, so PPO does not start from a
    # blank critic). The run must have used the same net1, and the same
    # view, input_bins, hidden, memory, student, base, inputs and
    # action_max, so the actor computes the same function here; the value
    # network also needs the same critic_truth, critic_flows and
    # critic_hidden. In place of init_actor and init_critic.
    init_from: Path | None = None
    # "ppo" only: the weight of a penalty on the policy's distance from the
    # one it started from, the KL divergence between their Gaussians
    # (mu - mu_start)^2 / (2 sigma^2) per decision. A taught start is worth
    # keeping until the value network can say what improves on it: on the
    # true LCFs, PPO lost its start's lead of 0.04 within 80 iterations
    # (round 19). 0 leaves PPO's objective unchanged.
    anchor: Annotated[float, Field(ge=0)] = 0.0
    # "dagger" only: a run directory whose actor reads the true LCFs
    # (inputs "truth", student "u", base "fixed", the same net1), the
    # teacher in place of the rules. At each decision of a rollout it reads
    # the truth's bins and the window the student actually set, and the
    # student learns the window it would set, as u on the student's own
    # believed (or, for "raw", measured) BDP: the student learns to act as
    # if it knew the path, from what net1 says about it.
    teacher_from: Path | None = None
    # What the actor's output is added to. "fixed": nothing; the actor starts
    # as the fixed rule, u_rule everywhere. "share": the share rule on net1's
    # mean beliefs, u = u_rule + share_k * busy * max(0, 1 - share)
    # (train/share.py); the actor starts at 0, so training starts from the
    # share rule and learns a residual on it.
    base: Literal["fixed", "share"] = "fixed"
    # What the actor reads. "net1": net1's beliefs (the split). "truth": the
    # oracle (arm A1): net1's output with all probability on the true
    # LCF's bin, from the guest's live truth, so the actor and the window
    # read the true path state. "raw": the black box (arm A4): the
    # measurements net1 reads, through net1's input scaler, with the window
    # aimed at a measured BDP (BBR's: the maximum delivery rate over the
    # last 2 s times the minimum RTT) instead of net1's. net1 then serves
    # only as that scaler and the evaluation's rule.
    inputs: Literal["net1", "truth", "raw"] = "net1"
    share_k: Annotated[float, Field(ge=0)] = 0.5
    alpha: Annotated[float, Field(gt=0)] = 20.0
    updates_per_iteration: PositiveInt = 200
    replay_iterations: PositiveInt = 8
    target_tau: Annotated[float, Field(gt=0, le=1)] = 0.005
    policy_noise: Annotated[float, Field(ge=0)] = 0.05  # target smoothing, log2 of the window
    noise_clip: Annotated[float, Field(ge=0)] = 0.1
    policy_delay: PositiveInt = 2
    eval_every: PositiveInt = 5
    seed: Annotated[int, Field(ge=0)] = 0
    keep_rollouts: bool = False
    time_budget_s: PositiveInt | None = None

    @model_validator(mode="after")
    def _dagger_from_the_rule(self) -> PPOSpec:
        if self.algo == "dagger" and self.base != "fixed":
            raise ValueError('algo "dagger" learns the whole u: base must be "fixed"')
        if self.student != "u" and (self.algo == "td3bc" or self.base != "fixed"):
            raise ValueError(
                f'student "{self.student}" is taught by algo "dagger" and continued by "ppo",'
                ' on base "fixed"'
            )
        if self.init_from is not None and (self.init_actor or self.init_critic):
            raise ValueError("init_from replaces init_actor and init_critic: set one or the other")
        if self.anchor > 0 and self.algo != "ppo":
            raise ValueError('anchor is a penalty in algo "ppo" only')
        if self.teacher_from is not None and (self.algo != "dagger" or self.student != "u"):
            raise ValueError('teacher_from teaches u to a student "u" of algo "dagger" only')
        if self.gate_threshold_q is not None and self.student != "gate3":
            raise ValueError('gate_threshold_q is the threshold of student "gate3" only')
        if self.gate3_copies != "coop" and self.student != "gate3":
            raise ValueError('gate3_copies is a choice of student "gate3" only')
        if self.init_critic is not None and self.algo != "ppo":
            raise ValueError('init_critic continues algo "ppo" only')
        if self.alone + self.compete + self.info > 1:
            raise ValueError("alone, compete and info are shares of the links: at most 1 together")
        if self.inputs != "net1" and (self.algo == "td3bc" or self.base != "fixed"):
            raise ValueError(f'inputs "{self.inputs}" is an arm of algo "ppo" on base "fixed"')
        if (
            self.inputs != "net1"
            and self.algo == "dagger"
            and (self.inputs != "raw" or self.student != "u")
        ):
            raise ValueError('only the black box ("raw") learns from the teacher, and only u')
        return self

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
    base: F32 | None = None  # [T] the rule's u the actor's output adds to (base "share")
    teacher: F32 | None = None  # [T] the privileged teacher's u (algo "dagger")
    tcp: F32 | None = None  # [T] 1 while a stock TCP flow is active (algo "dagger")
    share: F32 | None = None  # [T] the share rule's u (algo "dagger", gate students)
    copies: F32 | None = None  # [T] 1 while a copy of the policy is active (algo "dagger")
    coop: F32 | None = None  # [T] the cooperative window's u (algo "dagger", gate students)


def _agent(policy: Path, sigma: float, noise_seed: int | None, hold: int = 1) -> AgentCC:
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    return AgentCC(
        policy=OnnxPolicy(path=policy, sha256=sha),
        noise_sigma=sigma,
        noise_seed=noise_seed,
        noise_hold_steps=hold,
    )


BROAD_START_MS: Final = 10_000
BROAD_SECOND: Final[tuple[StockName, ...]] = ("cubic", "bbr", "reno")
HELD_OUT: Final[frozenset[StockName]] = frozenset({"bbr3", "westwood"})
# The Internet's common controllers more often, as examples/net1_broad.py.
DIVERSE_WEIGHT: Final[dict[StockName, float]] = {
    n: {"cubic": 6.0, "bbr": 4.0, "reno": 2.0}.get(n, 1.0) for n in STOCK_NAMES if n not in HELD_OUT
}


def _diverse(rng: np.random.Generator) -> StockName:
    names = list(DIVERSE_WEIGHT)
    w = np.array([DIVERSE_WEIGHT[n] for n in names])
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _window(rng: np.random.Generator, duration_ms: int) -> TcpCross:
    """A stock flow's lifetime under competition "dynamic", as
    examples/net1_broad.py's window: from 0 s (35%) or a uniform time in
    [1 s, 60% of the episode], and 30% of the time leaving at a uniform
    time at least 5 s later; the controller is drawn from DIVERSE_WEIGHT."""
    cc = _diverse(rng)
    start = 0 if rng.random() < 0.35 else int(rng.uniform(1000, 0.6 * duration_ms))
    if rng.random() < 0.3 and start + 5000 < duration_ms:
        return TcpCross(cc=cc, start_ms=start, stop_ms=int(rng.uniform(start + 5000, duration_ms)))
    return TcpCross(cc=cc, start_ms=start)


def _competition(
    rng: np.random.Generator,
    copies: tuple[int, ...],
    copy: AgentCC | None,
    competition: Literal["at-start", "broad", "diverse", "dynamic"] = "at-start",
    duration_ms: int = 30_000,
    *,
    copies_share: float | None = None,
) -> list[TcpCross | UdpCross]:
    """Cubic or BBR, or n - 1 copies of the policy (n in copies) from 0 s;
    with competition "broad", stock flows start within BROAD_START_MS and a
    second one joins half the time; with "diverse", as "broad" with each
    stock flow drawn from DIVERSE_WEIGHT; with "dynamic", each stock flow
    gets a lifetime anywhere in the episode (_window)."""
    if copies_share is None:
        choices = ["cubic", "bbr", *(f"copies{n}" for n in copies if copy is not None)]
        pick = choices[int(rng.integers(len(choices)))]
    elif copy is not None and copies and rng.random() < copies_share:
        pick = f"copies{copies[int(rng.integers(len(copies)))]}"
    else:
        pick = ("cubic", "bbr")[int(rng.integers(2))]
    if not pick.startswith("copies"):
        stock: StockName = "cubic" if pick == "cubic" else "bbr"
        if competition == "at-start":
            return [TcpCross(cc=stock, start_ms=0)]
        if competition == "dynamic":
            first = _window(rng, duration_ms)
            return [first, _window(rng, duration_ms)] if rng.random() < 0.5 else [first]
        if competition == "diverse":
            stock = _diverse(rng)
        flows: list[TcpCross | UdpCross] = [
            TcpCross(cc=stock, start_ms=int(rng.integers(0, BROAD_START_MS + 1)))
        ]
        if rng.random() < 0.5:
            second = (
                _diverse(rng)
                if competition == "diverse"
                else BROAD_SECOND[int(rng.integers(len(BROAD_SECOND)))]
            )
            flows.append(TcpCross(cc=second, start_ms=int(rng.integers(0, BROAD_START_MS + 1))))
        return flows
    assert copy is not None
    n = int(pick.removeprefix("copies"))
    return [TcpCross(cc=copy, start_ms=0) for _ in range(n - 1)]


def _info(rng: np.random.Generator, trace: Trace, duration_ms: int) -> list[TcpCross | UdpCross]:
    """Traffic under which the flow cannot see the capacity part of the
    time, one of three alike: a stock flow (DIVERSE_WEIGHT) from 0 s that
    leaves at a uniform time in [20%, 60%] of the episode; UDP at a uniform
    0.3 to 0.7 of the link's mean capacity, on and off for uniform 1 to 4 s
    each; or a stock TCP flow on and off for uniform 2 to 6 s each."""
    kind = int(rng.integers(3))
    if kind == 0:
        stop = int(rng.uniform(0.2, 0.6) * duration_ms)
        return [TcpCross(cc=_diverse(rng), start_ms=0, stop_ms=max(stop, 1))]
    lo, hi = (1000, 4000) if kind == 1 else (2000, 6000)
    on, off = int(rng.uniform(lo, hi)), int(rng.uniform(lo, hi))
    starts = range(int(rng.uniform(0, off)), duration_ms - 1, on + off)
    windows = [(t, min(t + on, duration_ms)) for t in starts][:MAX_CROSS]
    if kind == 1:
        rate = round(float(rng.uniform(0.3, 0.7)) * _mean_mbps(trace), 3)
        return [UdpCross(rate_mbps=rate, start_ms=a, stop_ms=b) for a, b in windows]
    cc = _diverse(rng)
    return [TcpCross(cc=cc, start_ms=a, stop_ms=b) for a, b in windows]


def _mean_mbps(trace: Trace) -> float:
    ts = trace.timestamps()  # one 1500-byte packet per entry, in ms
    return 12.0 * ts.size / float(ts[-1])


def _rising(rng: np.random.Generator, path: PathSpec) -> PathSpec:
    """The link with its capacity alternating between half and twice its
    mean, each level held a uniform 1 to 5 s."""
    m = _mean_mbps(path.trace)
    trace = PeriodicLevels(levels_mbps=[m / 2, 2 * m], level_ms=int(rng.uniform(1000, 5000)))
    return path.model_copy(update={"trace": trace})


def _group_episodes(
    arm: tuple[PathSpec, Condition],
    policy: Path,
    spec: PPOSpec,
    rng: np.random.Generator,
    draw: float,
) -> list[EpisodeSpec]:
    """K rollouts of one link: the same seed, different noise seeds. The
    uniform ``draw`` picks the link's traffic: alone below spec.alone, then
    competition (_competition) for the next spec.compete, else the paths'
    own condition, unless spec.info takes the next share (_info). Then
    spec.rise of the links get a rising capacity (_rising)."""
    path, cond = arm
    seed, base = int(rng.integers(0, 2**63)), int(rng.integers(0, 2**62))
    template = _agent(policy, spec.noise_sigma, None, spec.noise_hold_steps)
    if spec.rise and rng.random() < spec.rise:
        path = _rising(rng, path)
    if draw < spec.alone:
        cross: list[TcpCross | UdpCross] = []
    elif draw < spec.alone + spec.compete:
        cross = _competition(
            rng,
            spec.copies,
            template,
            spec.competition,
            spec.duration_ms,
            copies_share=spec.copies_share,
        )
    elif draw < spec.alone + spec.compete + spec.info:
        cross = _info(rng, path.trace, spec.duration_ms)
    else:
        cross = cond.flows(path.trace)

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


# The kinds of other flow on the link, in the critic's counts.
_RIVALS: Final = ("copy", "tcp", "udp")


def _rivals(e: EpisodeSpec) -> npt.NDArray[np.float64]:
    """An episode's cross flows as rows: kind (index in _RIVALS), start
    and stop in seconds."""

    def kind(c: TcpCross | UdpCross) -> int:
        if isinstance(c, UdpCross):
            return 2
        return 0 if isinstance(c.cc, AgentCC) else 1

    rows = [
        (kind(c), c.start_ms / 1000, (e.duration_ms if c.stop_ms is None else c.stop_ms) / 1000)
        for c in e.cross
    ]
    return np.array(rows, np.float64).reshape(-1, 3)


def _batch(
    arms: list[tuple[PathSpec, Condition]], policy: Path, spec: PPOSpec, rng: np.random.Generator
) -> tuple[list[EpisodeSpec], dict[str, int], dict[str, npt.NDArray[np.float64]]]:
    """One iteration's rollouts: links_per_iteration links, K each, each
    episode's link index (its group), and its cross flows (_rivals)."""
    eps: list[EpisodeSpec] = []
    groups: dict[str, int] = {}
    rivals: dict[str, npt.NDArray[np.float64]] = {}
    for j in range(spec.links_per_iteration):
        arm = arms[int(rng.integers(len(arms)))]
        for e in _group_episodes(arm, policy, spec, rng, float(rng.random())):
            groups[e.episode_id()] = j
            rivals[e.episode_id()] = _rivals(e)
            eps.append(e)
    return eps, groups, rivals


P0: Final = 0.02  # a gate student's beliefs at the start


def policy_u(
    spec: PPOSpec,
    a: torch.Tensor,
    share: torch.Tensor | None = None,
    coop: torch.Tensor | None = None,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """The policy's mean u from its actor's output a in [-1, 1]: the one
    head every learner trains and the exported graph runs, so a policy
    taught by one learner continues under another unchanged. Student "u":
    a times action_max, plus the base rule's u where one is given. "gate"
    and "gate3": beliefs (a + 1) / 2 past their thresholds (a [..., 2] for
    "gate3"), mixing the share rule's u and the cooperative window's."""
    if spec.student == "u":
        u = a * spec.action_max
        return u if base is None else u + base
    if share is None or (spec.student == "gate3" and coop is None):
        raise ValueError(f'student "{spec.student}" needs the rules\' u at each decision')
    t = spec.gate_threshold
    if spec.student == "gate":
        return gated_u(_sharpen((a + 1) / 2, t), share, spec.u_rule)
    assert coop is not None
    tq = t if spec.gate_threshold_q is None else spec.gate_threshold_q
    p, q = _sharpen((a[..., 0] + 1) / 2, t), _sharpen((a[..., 1] + 1) / 2, tq)
    return gated3_u(p, q, share, coop, spec.u_rule, copies=spec.gate3_copies)


def gated_u(p: torch.Tensor, share: torch.Tensor, u_rule: float) -> torch.Tensor:
    """A "gate" student's u: the fixed rule's, moved toward the share
    rule's by the believed probability p of a stock TCP flow."""
    return u_rule + p * (share - u_rule)


def gated3_u(
    p: torch.Tensor,
    q: torch.Tensor,
    share: torch.Tensor,
    coop: torch.Tensor,
    u_rule: float,
    *,
    copies: Literal["coop", "rule"] = "coop",
) -> torch.Tensor:
    """A "gate3" student's u: toward the share rule by p, the belief in a
    stock TCP flow, and otherwise toward the cooperative window by q, the
    belief in copies; with copies "rule", q vetoes the push instead."""
    if copies == "rule":
        return u_rule + p * (1 - q) * (share - u_rule)
    return u_rule + p * (share - u_rule) + (1 - p) * q * (coop - u_rule)


def _sharpen(p: torch.Tensor, t: float) -> torch.Tensor:
    return torch.clamp((p - t) / (1 - t), 0.0, 1.0) if t > 0 else p


class Rules(nn.Module):
    """net1's beliefs z -> [n, 2]: the share rule's u and the cooperative
    window's u (train/share.py), both on net1's standardized means, from any
    kind of net1. What the gate students mix, and what the rule teachers
    read."""

    def __init__(self, net1: fit.Net1, spec: PPOSpec) -> None:
        super().__init__()
        self.kind = net1.kind
        self.share = ShareRule(net1, spec.u_rule, spec.share_k)
        self.coop = CoopRule(net1, spec.u_rule)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.kind.scaled_means(z)
        return torch.cat([self.share(m), self.coop(m)], dim=1) * SHARE_ACTION_MAX


class Head(nn.Module):
    """A Held decision's head (recurrent.py): the actor's output a in [-1, 1]
    (with net1's beliefs z and the traffic head's probabilities) -> u, by
    policy_u, so the exported graph sets the u every learner trained. With
    ``traffic_q`` (or ``traffic_p``), a "gate3" student's belief in copies
    (or in a stock flow) comes from the traffic head instead
    (train/traffic.py)."""

    def __init__(
        self,
        net1: fit.Net1,
        spec: PPOSpec,
        *,
        traffic_q: bool = False,
        traffic_p: bool = False,
    ) -> None:
        super().__init__()
        if (traffic_q or traffic_p) and spec.student != "gate3":
            raise ValueError('the traffic head stands in for a "gate3" student\'s beliefs only')
        self.spec, self.traffic_q, self.traffic_p = spec, traffic_q, traffic_p
        needs = spec.student != "u" or spec.base == "share"
        self.rules = Rules(net1, spec) if needs else None

    def forward(
        self, a: torch.Tensor, z: torch.Tensor, traffic: torch.Tensor | None
    ) -> torch.Tensor:
        if self.traffic_q or self.traffic_p:
            if traffic is None:
                raise ValueError("this head reads the traffic head's probabilities")
            p = 2 * traffic[:, CLASSES.index("stock")] - 1 if self.traffic_p else a[:, 0]
            a = torch.stack([p, 2 * traffic[:, CLASSES.index("copies")] - 1], dim=1)
        if self.rules is None:
            return policy_u(self.spec, a)
        r = self.rules(z)
        if self.spec.student == "u":  # base "share": a residual on the share rule
            return policy_u(self.spec, a, base=r[:, 0])
        return policy_u(self.spec, a, r[:, 0], r[:, 1])


def teacher3_u(share_u: F32, coop_u: F32, counts: F32, u_rule: float) -> F32:
    """Student "gate3"'s teacher: the share rule's u while a stock TCP flow
    is active, else the cooperative window's while a copy is, else the
    fixed rule's."""
    rest = np.where(counts[:, 0] > 0, coop_u, np.float32(u_rule))
    return np.where(counts[:, 1] > 0, share_u, rest).astype(np.float32)


def teacher_u(share_u: F32, counts: F32, u_rule: float) -> F32:
    """The teacher of students "u" and "gate" at each decision: the share
    rule's u while a stock TCP flow is active (counts from ``active``), the
    fixed rule's otherwise."""
    return np.where(counts[:, 1] > 0, share_u, np.float32(u_rule)).astype(np.float32)


def _mean_u(spec: PPOSpec, trs: list[Trajectory], a: torch.Tensor) -> torch.Tensor:
    """policy_u at each padded decision of ``trs``, from the actor's padded
    output a in [-1, 1] and each trajectory's rules' u."""

    def padded(xs: list[F32]) -> torch.Tensor:
        return _pad([x[:, None] for x in xs])[0].squeeze(-1)

    if spec.student != "u":
        share = padded([_field(tr.share) for tr in trs])
        coop = padded([_field(tr.coop) for tr in trs]) if spec.student == "gate3" else None
        return policy_u(spec, a, share, coop)
    if all(tr.base is None for tr in trs):
        return policy_u(spec, a)
    base = [tr.base if tr.base is not None else np.zeros(tr.u.size, np.float32) for tr in trs]
    return policy_u(spec, a, base=padded(base))


def front(net1: fit.Net1, spec: PPOSpec, *, traffic: nn.Module | None = None) -> Front:
    """The actor's front end (graph.Front), shared by training and the
    exported graph: net1's beliefs ("net1"), the truth's ("truth", an
    oracle), or the measurements alone ("raw", the black box)."""
    inputs: fit.Inputs = (
        "true_lcfs" if spec.inputs == "truth" else "raw" if spec.inputs == "raw" else "net1"
    )
    return fit.front(
        net1,
        inputs=inputs,
        view=spec.view,
        input_bins=spec.input_bins if spec.view == "bins" else None,
        traffic=traffic,
    )


def target(net1: fit.Net1, spec: PPOSpec) -> nn.Module:
    """The BDP the window is aimed at: net1's belief, or BBR's estimate for
    the black box."""
    return MeasuredBdp() if spec.inputs == "raw" else fit.window_target(net1)


def _width(net1: fit.Net1, spec: PPOSpec) -> int:
    return fit.input_width(front(net1, spec))


# What an actor's function depends on besides its weights, and what a value
# network's inputs do: a run continued from another (init_from) must match.
_ACTOR_FIELDS: Final = (
    "view",
    "input_bins",
    "hidden",
    "memory",
    "student",
    "base",
    "inputs",
    "action_max",
)
_CRITIC_FIELDS: Final = ("critic_truth", "critic_flows", "critic_hidden")


def _same_net1(a: fit.Net1, b: fit.Net1) -> bool:
    def parts(n: fit.Net1) -> list[dict[str, torch.Tensor]]:
        return [n.scaler.state_dict(), n.lcf_scaler.state_dict(), n.net.state_dict()]

    return a.lcfs == b.lcfs and all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
        for x, y in zip(parts(a), parts(b), strict=True)
    )


def earlier_run(run: Path, net1: fit.Net1) -> PPOSpec:
    """The spec of an earlier run (its output directory), which must have
    used this net1."""
    spec = PPOSpec.parse((run / "ppo_spec.json").read_text())
    if not _same_net1(fit.load_net1(run / "net1.pt")[0], net1):
        raise ValueError(f"{run} used another net1 than {spec.net1_from}'s")
    return spec


def _mismatch(a: PPOSpec, b: PPOSpec, fields: tuple[str, ...]) -> list[str]:
    return [f for f in fields if getattr(a, f) != getattr(b, f)]


@dataclass(frozen=True, slots=True)
class _PolicyTeacher:
    """A recurrent actor on the true LCFs (PPOSpec.teacher_from) as
    DAgger's teacher."""

    actor: RecurrentActor
    view: nn.Module | None
    action_max: float

    @staticmethod
    def load(run: Path, net1: fit.Net1) -> _PolicyTeacher:
        t = earlier_run(run, net1)
        if t.inputs != "truth" or t.student != "u" or t.base != "fixed":
            raise ValueError(f'{run}: a teacher reads the truth (inputs "truth"), student "u"')
        actor = RecurrentActor(_width(net1, t), t.hidden, memory=t.memory)
        actor.load_state_dict(torch.load(run / "actor.pt", weights_only=True))
        view = fit.view_of(net1, t.view, t.input_bins if t.view == "bins" else None)
        return _PolicyTeacher(actor.eval(), view, t.action_max)

    @torch.no_grad()
    def window(self, net1: fit.Net1, z: torch.Tensor, y: F32, set_window: F32) -> F32:
        """[T] log2 of the window the teacher would set at each decision of
        one flow, given net1's output z, the true LCFs y (report units) and
        the window the flow set after each decision (log2 packets): it reads
        the truth's bins and, as its previous u, the flow's last window on
        the true BDP."""
        zt = fit.oracle(net1, z, y)
        x = zt if self.view is None else self.view(zt)
        lb = fit.window_target(net1).bdp(zt).numpy()
        prev = np.concatenate([[0.0], (set_window - lb)[:-1]]).astype(np.float32)
        inp = torch.cat([x, torch.from_numpy(prev / self.action_max).unsqueeze(1)], dim=1)
        u = self.actor.sequence(inp.unsqueeze(0))[0].numpy() * self.action_max
        return (lb + u).astype(np.float32)


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
    front: Front,
    target: nn.Module,
    rules: Rules,
    spec: PPOSpec,
    group: tuple[int, int],
    rivals: npt.NDArray[np.float64],
    teacher: _PolicyTeacher | None = None,
) -> Trajectory | None:
    """One flow's decisions; ``t`` is its steps in step order, ``utility``
    the reward of each of those steps, ``rivals`` its episode's cross
    flows (_rivals) and ``teacher`` DAgger's teacher where it is a policy
    on the truth (PPOSpec.teacher_from). The actor's inputs and the window
    target come from the exported graph's own front end and target."""
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
    y = np.stack([labels[k][d] * np.float32(LCF_UNITS[k][0]) for k in net1.lcfs], axis=1)
    o = torch.from_numpy(obs[d])
    z, xt, traffic = front(o, torch.from_numpy(y) if front.uses_truth else None)
    x = xt.numpy()
    lb = target(o, z, traffic).numpy()
    u = (np.log2(np.maximum(cwnd[d + 1], 1.0)) - lb).astype(np.float32)
    # The rules' u on the actor's beliefs, as its head reads them in the graph.
    ruled = rules(z).numpy().astype(np.float32) if spec.inputs != "raw" else None
    base = ruled[:, 0] if ruled is not None and spec.base == "share" else None
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
        ys = net1.lcf_scaler(torch.from_numpy(y)).numpy()
        parts.insert(0, np.where(np.isfinite(ys), ys, 0.0).astype(np.float32))
    counts = active(rivals, d * _STEP_S)
    if spec.critic_flows:
        parts.append(counts)
    critic = np.concatenate(parts, axis=1).astype(np.float32)
    if spec.algo != "dagger" and spec.student == "u":
        return Trajectory(group, inp, critic, u, reward, gamma, base)
    # The teachers read net1's beliefs, also for the black box ("raw"), whose
    # label is then the teacher's window on its own measured BDP.
    z1 = z if spec.inputs == "net1" else torch.from_numpy(net1.beliefs(obs[d]))
    if ruled is None or spec.inputs != "net1":
        ruled = rules(z1).numpy().astype(np.float32)
    share_u, coop_u = ruled[:, 0], ruled[:, 1]
    if spec.algo != "dagger":  # a gate student continued by PPO
        return Trajectory(group, inp, critic, u, reward, gamma, share=share_u, coop=coop_u)
    tcp = (counts[:, 1] > 0).astype(np.float32)
    copies = (counts[:, 0] > 0).astype(np.float32)
    if teacher is not None:  # the window it would set, on the student's BDP
        label = (teacher.window(net1, z1, y, u + lb) - lb).astype(np.float32)
    else:
        label = (
            teacher3_u(share_u, coop_u, counts, spec.u_rule)
            if spec.student == "gate3" and spec.gate3_copies == "coop"
            else teacher_u(share_u, counts, spec.u_rule)
        )
        if spec.inputs == "raw":
            label = (label + fit.window_target(net1).bdp(z1).numpy() - lb).astype(np.float32)
    return Trajectory(
        group, inp, critic, u, reward, gamma, base, label, tcp, share_u, copies, coop_u
    )


def active(rivals: npt.NDArray[np.float64], times: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """At each time (s into the episode), how many cross flows of each
    kind in _RIVALS are active, over 4."""
    on = (rivals[None, :, 1] <= times[:, None]) & (times[:, None] < rivals[None, :, 2])
    kinds = rivals[:, 0].astype(int)
    counts = [(on & (kinds == k)).sum(axis=1) for k in range(len(_RIVALS))]
    return (np.stack(counts, axis=1) / 4).astype(np.float32)


def _trajectories(
    rollouts: Path,
    groups: dict[str, int],
    net1: fit.Net1,
    spec: PPOSpec,
    rivals: dict[str, npt.NDArray[np.float64]],
    *,
    teacher: _PolicyTeacher | None = None,
) -> list[Trajectory]:
    """Every agent flow of every ok rollout episode."""
    fe, tgt, rules = front(net1, spec), target(net1, spec), Rules(net1, spec).eval()
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
            tr = _trajectory(
                t,
                u,
                net1,
                front=fe,
                target=tgt,
                rules=rules,
                spec=spec,
                group=(groups[e], f),
                rivals=rivals[e],
                teacher=teacher,
            )
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


def _new_actor(width: int, spec: PPOSpec) -> RecurrentActor:
    """The spec's actor as it starts: the fixed rule (or, on base "share",
    the share rule), or a gate student believing P0 everywhere."""
    actor = RecurrentActor(
        width, spec.hidden, memory=spec.memory, outputs=2 if spec.student == "gate3" else 1
    )
    if spec.student != "u":  # p = (tanh(head) + 1) / 2 = P0 everywhere
        actor.clone_rule(2 * P0 - 1, 1.0)
    else:
        actor.clone_rule(spec.u_rule if spec.base == "fixed" else 0.0, spec.action_max)
    return actor


def _value_step(
    critic: _Critic,
    opt: torch.optim.Optimizer,
    batch: list[Trajectory],
    returns: list[F32],
    max_grad_norm: float,
) -> float:
    """One gradient step of the value network toward the returns."""
    c, cmask = _pad([tr.critic for tr in batch])
    g, _ = _pad([r[:, None] for r in returns])
    vl = (((critic(c) - g.squeeze(-1)) ** 2) * cmask).sum() / cmask.sum()
    opt.zero_grad()
    vl.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    opt.step()
    return vl.item()


def _explained(returns: list[F32], values: list[F32]) -> float:
    allret, allv = np.concatenate(returns), np.concatenate(values)
    return 1.0 - float(np.var(allret - allv)) / max(float(np.var(allret)), 1e-12)


class _Learner:
    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = _new_actor(width, spec)
        self.critic = _Critic(critic_width, spec.critic_hidden)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=spec.critic_lr)
        # The policy it started from, frozen (_make_learner sets it after
        # loading any saved actor): the anchor's reference and kl_to_start's.
        self.start = copy.deepcopy(self.actor).requires_grad_(False)

    def _means(self, actor: RecurrentActor, trs: list[Trajectory]) -> list[F32]:
        with torch.no_grad():
            x, mask = _pad([tr.inp for tr in trs])
            mu = _mean_u(self.spec, trs, actor.sequence(x)).numpy()
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
        adv, ret = _advantages(trs, v_old, s.gae_lambda)
        flat = np.concatenate(adv)
        scale = float(flat.std()) + 1e-8
        adv = [((a - flat.mean()) / scale).astype(np.float32) for a in adv]
        explained = _explained(ret, v_old)
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
                batch = [trs[i] for i in idx]
                mu = _mean_u(s, batch, self.actor.sequence(x))
                ratio = torch.exp(((u - m0) ** 2 - (u - mu) ** 2) / var)
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - s.clip, 1 + s.clip) * a)
                loss = -(surr * mask).sum() / mask.sum()
                if s.anchor > 0:
                    with torch.no_grad():
                        m_start = _mean_u(s, batch, self.start.sequence(x))
                    kl = (((mu - m_start) ** 2 / var) * mask).sum() / mask.sum()
                    loss = loss + s.anchor * kl
                if actor:
                    self.opt_a.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
                    self.opt_a.step()
                vl = _value_step(
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


class _TD3Learner:
    """TD3+BC with the same interface as _Learner (``actor``, ``critic`` and
    ``update``). A transition is one decision to the next of one flow; the
    last decision of a flow has no successor and is not learned from."""

    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = _new_actor(width, spec)
        self.actor_target = copy.deepcopy(self.actor)
        self.q = nn.ModuleList(_Critic(critic_width + 1, spec.critic_hidden) for _ in range(2))
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
            x, mask = _pad([tr.inp for tr in batch])
            c, _ = _pad([tr.critic for tr in batch])
            u, _ = _pad([tr.u[:, None] for tr in batch])
            r, _ = _pad([tr.reward[:, None] for tr in batch])
            g, _ = _pad([tr.gamma[:, None] for tr in batch])
            u, r, g = u.squeeze(-1), r.squeeze(-1), g.squeeze(-1)
            m = mask[:, 1:]  # the decisions with a successor
            with torch.no_grad():
                mu_next = _mean_u(s, batch, self.actor_target.sequence(x))[:, 1:]
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
            mu = _mean_u(s, batch, self.actor.sequence(x))
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
            x, mask = _pad([tr.inp for tr in trs])
            mu_all = _mean_u(s, trs, self.actor.sequence(x))[mask.bool()].numpy()
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


class _DAggerLearner:
    """algo "dagger" with the same interface as _Learner. Its value network
    learns, as PPO's does, the return of the policy whose rollouts it sees
    (the student's), so that PPO can continue from both (init_from)."""

    def __init__(self, width: int, critic_width: int, spec: PPOSpec) -> None:
        self.spec = spec
        self.actor = _new_actor(width, spec)
        self.critic = _Critic(critic_width, spec.critic_hidden)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=spec.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=spec.critic_lr)
        # Its own draws, so the imitation's batches are those it drew before
        # it had a value network.
        self.value_rng = np.random.default_rng(spec.seed + 1)
        self.replay: list[list[Trajectory]] = []

    def _u(self, trs: list[Trajectory], pre: torch.Tensor) -> torch.Tensor:
        """The student's u as deployed, padded like ``pre`` (the actor's head
        outputs)."""
        return _mean_u(self.spec, trs, torch.tanh(pre))

    def _values(self, trs: list[Trajectory]) -> dict[str, float]:
        """PPO's value regression (epochs of minibatches) on this
        iteration's rollouts, before and after."""
        s = self.spec
        with torch.no_grad():
            v_old = [self.critic(torch.from_numpy(tr.critic)).numpy() for tr in trs]
        _, ret = _advantages(trs, v_old, s.gae_lambda)
        losses: list[float] = []
        for _ in range(s.epochs):
            order = self.value_rng.permutation(len(trs))
            for b in range(0, len(trs), s.batch_trajectories):
                idx = order[b : b + s.batch_trajectories]
                batch, rs = [trs[i] for i in idx], [ret[i] for i in idx]
                losses.append(_value_step(self.critic, self.opt_c, batch, rs, s.max_grad_norm))
        return {"value_loss": float(np.mean(losses)), "explained_variance": _explained(ret, v_old)}

    def _gaps(self, trs: list[Trajectory]) -> dict[str, float]:
        """The actor against the teacher on ``trs``: mean |u - teacher|,
        the actor's mean u where the teacher pushes and where not, its mean
        belief p of a stock TCP flow where there is one and where not, and
        for "gate3" its mean belief q of copies (and p) where only copies
        share the link, and q elsewhere."""
        with torch.no_grad():
            x, mask = _pad([tr.inp for tr in trs])
            pre = self.actor.pre(x)
            w = mask.bool()
            mu, pq = self._u(trs, pre)[w].numpy(), torch.sigmoid(2 * pre)[w].numpy()
        p = pq if pq.ndim == 1 else pq[:, 0]
        y = np.concatenate([_field(tr.teacher) for tr in trs])
        tcp = np.concatenate([_field(tr.tcp) for tr in trs]) > 0.5
        push = y != np.float32(self.spec.u_rule)
        extra: dict[str, float] = {}
        if pq.ndim == 2:  # gate3: the belief in copies, where copies only are active and not
            q = pq[:, 1]
            cop = (np.concatenate([_field(tr.copies) for tr in trs]) > 0.5) & ~tcp
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
        if self.spec.student != "u":
            tcp, _ = _pad([_field(tr.tcp)[:, None] for tr in batch])
            if self.spec.student == "gate3":
                cop, _ = _pad([_field(tr.copies)[:, None] for tr in batch])
                y = torch.cat([tcp, cop], dim=-1)
                m = mask.unsqueeze(-1)
            else:
                y, m = tcp.squeeze(-1), mask
            ce = nn.functional.binary_cross_entropy_with_logits(2 * pre, y, reduction="none")
            return (ce * m).sum() / m.expand_as(ce).sum()
        y, _ = _pad([_field(tr.teacher)[:, None] for tr in batch])
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
            x, mask = _pad([tr.inp for tr in batch])
            loss = self._loss(batch, self.actor.pre(x), mask)
            self.opt_a.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), s.max_grad_norm)
            self.opt_a.step()
            losses.append(loss.item())
        after = self._gaps(trs)
        with torch.no_grad():
            x, mask = _pad([tr.inp for tr in trs])
            mu_all = self._u(trs, self.actor.pre(x))[mask.bool()].numpy()
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


def _field(x: F32 | None) -> F32:
    if x is None:
        raise ValueError(
            "a trajectory lacks what its learner reads: the teacher's labels (algo"
            ' "dagger") or the rules\' u (a gate student)'
        )
    return x


_LEARNERS: Final = {"ppo": _Learner, "td3bc": _TD3Learner, "dagger": _DAggerLearner}


def _init_paths(spec: PPOSpec, source: PPOSpec | None) -> tuple[Path | None, Path | None]:
    """The saved actor and value network the run starts from: init_actor and
    init_critic, or init_from's (``source`` is that run's spec), its value
    network only for "ppo" and where it has one that reads the same inputs."""
    if spec.init_from is None:
        return spec.init_actor, spec.init_critic
    if source is None:
        raise ValueError("init_from needs the earlier run's spec (earlier_run)")
    if bad := _mismatch(spec, source, _ACTOR_FIELDS):
        raise ValueError(f"init_from {spec.init_from} differs in {', '.join(bad)}")
    critic = spec.init_from / "critic.pt"
    keep = (
        spec.algo == "ppo"
        and source.algo != "td3bc"
        and not _mismatch(spec, source, _CRITIC_FIELDS)
        and bool(torch.load(critic, weights_only=True))  # an older "dagger" run saved none
    )
    return spec.init_from / "actor.pt", critic if keep else None


def _make_learner(
    spec: PPOSpec, width: int, critic_width: int, source: PPOSpec | None = None
) -> _Learner | _TD3Learner | _DAggerLearner:
    """The spec's learner, started from what _init_paths names."""
    learner = _LEARNERS[spec.algo](width, critic_width, spec)
    actor_pt, critic_pt = _init_paths(spec, source)
    if actor_pt is not None:
        state = torch.load(actor_pt, weights_only=True)
        learner.actor.load_state_dict(state)
        if isinstance(learner, _TD3Learner):
            learner.actor_target.load_state_dict(state)
    if critic_pt is not None and isinstance(learner, _Learner):
        learner.critic.load_state_dict(torch.load(critic_pt, weights_only=True))
    if isinstance(learner, _Learner):
        learner.start = copy.deepcopy(learner.actor).requires_grad_(False)
    return learner


def policy_graph(
    net1: fit.Net1,
    actor: RecurrentActor | EnsembleActor | ConcatActor,
    spec: PPOSpec,
    truth: frozenset[str] = frozenset(),
    *,
    traffic: nn.Module | None = None,
    head: nn.Module | None = None,
) -> PolicyGraph:
    """The exported net2: the spec's front end (the LCFs in ``truth`` from
    the guest's live truth instead of net1, all of them for inputs
    "truth"; ``traffic`` a traffic head beside net1), its window target,
    and a Held decision with the spec's Head (or ``head``)."""
    f = front(net1, spec, traffic=traffic)
    if truth:
        if spec.inputs != "net1":
            raise ValueError(f'inputs "{spec.inputs}" take no LCF from the truth one by one')
        f = fit.front(
            net1,
            view=spec.view,
            input_bins=spec.input_bins if spec.view == "bins" else None,
            truth=truth,
            traffic=traffic,
        )
    decide = Held(actor, head if head is not None else Head(net1, spec), spec.action_max)
    columns = fit.truth_columns(net1) if f.uses_truth else None
    return PolicyGraph(f, decide, target(net1, spec), columns).eval()


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
    critic_width = (
        width
        + 2
        + (len(net1.lcfs) if spec.critic_truth else 0)
        + (len(_RIVALS) if spec.critic_flows else 0)
    )
    source = None if spec.init_from is None else earlier_run(spec.init_from, net1)
    learner = _make_learner(spec, width, critic_width, source)
    teacher = None if spec.teacher_from is None else _PolicyTeacher.load(spec.teacher_from, net1)
    train_arms = rollouts.arms(spec.paths_from, rng)
    eval_arms = rollouts.arms(spec.eval_from, rng) if spec.eval_from else []

    def run(d: Path, eps: list[EpisodeSpec]) -> dict[str, Any]:
        return rollouts.run(d, eps, build, workers, scratch)

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
        eps, groups, rivals = _batch(train_arms, policy, spec, rng)
        summary = next(iter(run(it_dir / "rollouts", eps).values()))
        trs = _trajectories(it_dir / "rollouts", groups, net1, spec, rivals, teacher=teacher)
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
    started = _init_paths(spec, source)
    return {
        "iterations": history,
        "started_from": {
            k: None if v is None else str(v)
            for k, v in zip(("actor", "critic"), started, strict=True)
        },
    }
