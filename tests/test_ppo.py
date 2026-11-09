"""The recurrent, decision-holding net2 and its PPO learner (train/recurrent.py,
train/ppo.py)."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.policy import OBS_DIM  # noqa: E402
from expliccit_env.spec import AgentCC, EpisodeSpec, OnnxPolicy, TcpCross  # noqa: E402
from expliccit_env.train import data, fit, ppo  # noqa: E402
from expliccit_env.train.export import to_onnx  # noqa: E402
from expliccit_env.train.graph import MAX_DELIVERED, MIN_RTT_MS, PolicyGraph  # noqa: E402
from expliccit_env.train.recurrent import OBS_MIN_RTT, RecurrentActor, decision_steps  # noqa: E402

from .test_train import _bdp_net1  # noqa: E402


def test_decisions_come_once_per_min_rtt_within_10_to_50_ms() -> None:
    rtt = np.array([30.0] * 6 + [np.nan] * 2 + [200.0] * 5 + [4.0] * 3, np.float32)
    # A decision fixes its block from the min RTT it reads: 30 ms, 3 steps;
    # not finite, 1; 200 ms, 5 (50 ms at most); 4 ms, 1 (10 ms at least).
    assert decision_steps(rtt).tolist() == [0, 3, 6, 7, 8, 13, 14, 15]


@torch.no_grad()
def _lb(
    net1: fit.Net1, spec: ppo.PPOSpec, obs: np.ndarray, y: np.ndarray | None = None
) -> torch.Tensor:
    """log2 of the BDP the spec's actor aims at, through its front end."""
    front, o = ppo.front(net1, spec), torch.from_numpy(obs)
    z, _, traffic = front(o, torch.from_numpy(y) if front.uses_truth and y is not None else None)
    return ppo.target(net1, spec)(o, z, traffic)


def _graph(u: float, memory: bool = True) -> tuple[PolicyGraph, fit.Net1, np.ndarray]:
    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=u,
        input_bins=4,
        hidden=8,
        memory=memory,
    )
    actor = RecurrentActor(ppo._width(net1, spec), spec.hidden, memory=memory)  # pyright: ignore[reportPrivateUsage]
    actor.clone_rule(u, spec.action_max)
    return ppo.policy_graph(net1, actor, spec), net1, steps.obs[steps.episode == 0][:60]


@pytest.mark.parametrize("memory", [True, False])
def test_held_graph_decides_where_the_learner_reads_decisions(memory: bool) -> None:
    graph, _, obs = _graph(0.25, memory)
    obs = obs.copy()
    obs[:, OBS_MIN_RTT] = np.where(np.arange(len(obs)) < 30, 30.0, 12.0)
    rng = np.random.default_rng(0)
    state = torch.zeros(1, graph.state_dim)
    cwnd, decided, windows, beliefs = 10.0, [], [], []
    with torch.no_grad():
        for row in obs:
            o = torch.from_numpy(row[None, :])
            a, state = graph.step(o, state, cwnd=torch.tensor([[cwnd]]))
            assert state is not None
            decided.append(bool(state[0, -1] > 0.5))
            windows.append(float(a) + np.log2(cwnd))
            beliefs.append(float(graph.target(o, graph.beliefs(o), None)))  # pyright: ignore[reportOptionalCall]
            cwnd = cwnd * 2 ** (float(a) + 0.1 * rng.standard_normal())  # the guest's noise
    d = decision_steps(obs[:, OBS_MIN_RTT])
    assert np.flatnonzero(decided).tolist() == d.tolist()
    # The cloned rule asks for 2^0.25 times the believed BDP at each decision ...
    for k in d:
        assert windows[k] == pytest.approx(beliefs[k] + 0.25, abs=1e-4)
    # ... and holds the u the decision's noise made until the next one: the
    # window follows each step's believed BDP, as the rule's does.
    for k, nxt in itertools.pairwise(d):
        held = (np.array(windows) - np.array(beliefs))[k + 1 : nxt]
        assert np.ptp(held) < 1e-4 if len(held) > 1 else True


def test_a_residual_on_the_share_rule_starts_as_the_share_rule() -> None:
    from expliccit_env.train.share import ACTION_MAX, ShareRule  # noqa: PLC0415

    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        input_bins=4,
        hidden=8,
        base="share",
        share_k=2.0,
    )
    actor = RecurrentActor(ppo._width(net1, spec), spec.hidden)  # pyright: ignore[reportPrivateUsage]
    actor.clone_rule(0.0, spec.action_max)
    graph = ppo.policy_graph(net1, actor, spec)
    view = fit.view_of(net1, "means")
    assert view is not None
    rule = ShareRule(net1, 0.25, 2.0)
    obs = steps.obs[steps.episode == 0][:40].copy()
    obs[:, OBS_MIN_RTT] = 30.0
    state, cwnd = torch.zeros(1, graph.state_dim), 10.0
    with torch.no_grad():
        for row in obs:
            o = torch.from_numpy(row[None, :])
            a, state = graph.step(o, state, cwnd=torch.tensor([[cwnd]]))
            assert state is not None
            z = graph.beliefs(o)
            if state[0, -1] > 0.5:  # a decision
                want = float(graph.target(o, z, None)) + float(rule(view(z))) * ACTION_MAX  # pyright: ignore[reportOptionalCall]
                assert float(a) + np.log2(cwnd) == pytest.approx(want, abs=1e-4)
            cwnd = cwnd * 2 ** float(a)


@pytest.mark.parametrize(
    ("p", "t", "acted"), [(0.75, 0.0, 0.75), (0.75, 0.5, 0.5), (0.3, 0.5, 0.0)]
)
def test_a_gate_graph_moves_u_toward_the_share_rule_by_its_belief(
    p: float, t: float, acted: float
) -> None:
    from expliccit_env.train.share import ACTION_MAX, ShareRule  # noqa: PLC0415

    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        input_bins=4,
        hidden=8,
        algo="dagger",
        student="gate",
        share_k=2.0,
        gate_threshold=t,
    )
    actor = RecurrentActor(ppo._width(net1, spec), spec.hidden)  # pyright: ignore[reportPrivateUsage]
    actor.clone_rule(2 * p - 1, 1.0)  # (a + 1) / 2 = p on any input
    graph = ppo.policy_graph(net1, actor, spec)
    view = fit.view_of(net1, "means")
    assert view is not None
    rule = ShareRule(net1, 0.25, 2.0)
    obs = steps.obs[steps.episode == 0][:40].copy()
    obs[:, OBS_MIN_RTT] = 30.0
    state, cwnd = torch.zeros(1, graph.state_dim), 10.0
    with torch.no_grad():
        for row in obs:
            o = torch.from_numpy(row[None, :])
            a, state = graph.step(o, state, cwnd=torch.tensor([[cwnd]]))
            assert state is not None
            z = graph.beliefs(o)
            if state[0, -1] > 0.5:  # a decision
                share = float(rule(view(z))) * ACTION_MAX  # pyright: ignore[reportOptionalCall]
                want = float(graph.target(o, z, None)) + 0.25 + acted * (share - 0.25)  # pyright: ignore[reportOptionalCall]
                assert float(a) + np.log2(cwnd) == pytest.approx(want, abs=1e-4)
            cwnd = cwnd * 2 ** float(a)


def test_an_ensemble_gate_graph_acts_on_its_members_mean_belief() -> None:
    from expliccit_env.train.recurrent import EnsembleActor  # noqa: PLC0415
    from expliccit_env.train.share import ACTION_MAX, ShareRule  # noqa: PLC0415

    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        input_bins=4,
        hidden=8,
        algo="dagger",
        student="gate",
        share_k=2.0,
        gate_threshold=0.5,
    )
    members = []
    beliefs = (0.2, 0.9, 0.97)
    acted = (np.mean(beliefs) - 0.5) / 0.5  # the mean belief, past t = 0.5
    for p in beliefs:
        actor = RecurrentActor(ppo._width(net1, spec), spec.hidden)  # pyright: ignore[reportPrivateUsage]
        actor.clone_rule(2 * p - 1, 1.0)
        members.append(actor)
    graph = ppo.policy_graph(net1, EnsembleActor(members), spec)
    assert graph.state_dim == 3 * 8 + 5
    view = fit.view_of(net1, "means")
    assert view is not None
    rule = ShareRule(net1, 0.25, 2.0)
    obs = steps.obs[steps.episode == 0][:40].copy()
    obs[:, OBS_MIN_RTT] = 30.0
    state, cwnd = torch.zeros(1, graph.state_dim), 10.0
    with torch.no_grad():
        for row in obs:
            o = torch.from_numpy(row[None, :])
            a, state = graph.step(o, state, cwnd=torch.tensor([[cwnd]]))
            assert state is not None
            z = graph.beliefs(o)
            if state[0, -1] > 0.5:  # a decision
                share = float(rule(view(z))) * ACTION_MAX  # pyright: ignore[reportOptionalCall]
                want = float(graph.target(o, z, None)) + 0.25 + acted * (share - 0.25)  # pyright: ignore[reportOptionalCall]
                assert float(a) + np.log2(cwnd) == pytest.approx(want, abs=1e-4)
            cwnd = cwnd * 2 ** float(a)


@pytest.mark.parametrize(
    ("p", "q", "t", "tq", "copies"),
    [
        (0.9, 0.2, 0.5, None, "coop"),
        (0.2, 0.9, 0.5, None, "coop"),
        (0.7, 0.8, 0.0, None, "coop"),
        (0.6, 0.7, 0.3, 0.6, "coop"),
        (0.9, 0.7, 0.5, None, "rule"),
    ],
)
def test_a_gate3_graph_mixes_the_rules_by_its_beliefs(
    p: float, q: float, t: float, tq: float | None, copies: Literal["coop", "rule"]
) -> None:
    from expliccit_env.train.share import ACTION_MAX, CoopRule, ShareRule  # noqa: PLC0415

    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        input_bins=4,
        hidden=8,
        algo="dagger",
        student="gate3",
        share_k=2.0,
        gate_threshold=t,
        gate_threshold_q=tq,
        gate3_copies=copies,
    )
    actor = RecurrentActor(ppo._width(net1, spec), spec.hidden, outputs=2)  # pyright: ignore[reportPrivateUsage]
    with torch.no_grad():
        actor.head.weight.zero_()
        actor.head.bias.copy_(torch.tensor([np.arctanh(2 * p - 1), np.arctanh(2 * q - 1)]))
    graph = ppo.policy_graph(net1, actor, spec)

    def sharp(x: float, t: float) -> float:
        return min(max((x - t) / (1 - t), 0.0), 1.0) if t > 0 else x

    ps, qs = sharp(p, t), sharp(q, t if tq is None else tq)
    view = fit.view_of(net1, "means")
    assert view is not None
    share_rule, coop_rule = ShareRule(net1, 0.25, 2.0), CoopRule(net1, 0.25)
    obs = steps.obs[steps.episode == 0][:40].copy()
    obs[:, OBS_MIN_RTT] = 30.0
    state, cwnd = torch.zeros(1, graph.state_dim), 10.0
    with torch.no_grad():
        for row in obs:
            o = torch.from_numpy(row[None, :])
            a, state = graph.step(o, state, cwnd=torch.tensor([[cwnd]]))
            assert state is not None
            z = graph.beliefs(o)
            if state[0, -1] > 0.5:  # a decision
                sh = float(share_rule(view(z))) * ACTION_MAX  # pyright: ignore[reportOptionalCall]
                co = float(coop_rule(view(z))) * ACTION_MAX  # pyright: ignore[reportOptionalCall]
                if copies == "rule":
                    u = 0.25 + ps * (1 - qs) * (sh - 0.25)
                else:
                    u = 0.25 + ps * (sh - 0.25) + (1 - ps) * qs * (co - 0.25)
                want = float(graph.target(o, z, None)) + u  # pyright: ignore[reportOptionalCall]
                assert float(a) + np.log2(cwnd) == pytest.approx(want, abs=1e-4)
            cwnd = cwnd * 2 ** float(a)


def test_cloned_actor_answers_the_rule_on_any_input() -> None:
    actor = RecurrentActor(12, 8)
    actor.clone_rule(-0.4, 3.0)
    mu = actor.sequence(torch.randn(3, 20, 13)) * 3.0
    assert torch.allclose(mu, torch.full_like(mu, -0.4), atol=1e-6)


def test_held_graph_exports_with_state_and_cwnd() -> None:
    graph, _, _ = _graph(0.125)
    m = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in m.graph.input] == ["obs", "state", "cwnd"]
    assert [o.name for o in m.graph.output] == ["a", "next_state"]


def test_the_black_box_aims_at_the_measured_bdp_and_reads_no_belief() -> None:
    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"), net1_from=Path("unused"), u_rule=0.25, hidden=8, inputs="raw"
    )
    width = ppo._width(net1, spec)  # pyright: ignore[reportPrivateUsage]
    assert width == net1.scaler.cols.numel()
    actor = RecurrentActor(width, spec.hidden)
    actor.clone_rule(0.25, spec.action_max)
    graph = ppo.policy_graph(net1, actor, spec)
    obs = steps.obs[steps.episode == 0][:5].copy()
    obs[:, MAX_DELIVERED] = 0.5  # 50 Mbit/s
    obs[:, MIN_RTT_MS] = 24.0
    bdp = np.log2(50e6 * 0.024 / 12_000)  # 100 packets
    for p in net1.net.parameters():  # no belief reaches the black box
        p.data.fill_(float("nan"))
    with torch.no_grad():
        a, _ = graph.step(
            torch.from_numpy(obs[:1]), torch.zeros(1, graph.state_dim), cwnd=torch.tensor([[10.0]])
        )
    assert float(a) + np.log2(10.0) == pytest.approx(bdp + 0.25, abs=1e-4)
    m = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in m.graph.input] == ["obs", "state", "cwnd"]
    # The guest's tract cannot run IsInf (round 19's first try).
    assert "IsInf" not in {n.op_type for n in m.graph.node}
    with pytest.raises(ValueError, match="an arm of algo"):
        ppo.PPOSpec(paths_from=Path("u"), net1_from=Path("u"), inputs="raw", base="share")


def test_the_oracle_reads_the_live_truth_as_its_trajectories_do() -> None:
    steps, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        input_bins=4,
        hidden=8,
        inputs="truth",
    )
    actor = RecurrentActor(ppo._width(net1, spec), spec.hidden)  # pyright: ignore[reportPrivateUsage]
    actor.clone_rule(0.25, spec.action_max)
    graph = ppo.policy_graph(net1, actor, spec)
    obs = steps.obs[steps.episode == 0][:1]
    # 80 Mbit/s and 30 ms one way, where net1 was trained near 50 and 20.
    truth = np.array([[np.nan, np.nan, 80e6, np.nan, np.nan, 30e6, np.nan]], np.float32)
    with torch.no_grad():
        a, _ = graph.step(
            torch.from_numpy(obs),
            torch.zeros(1, graph.state_dim),
            torch.from_numpy(truth),
            torch.tensor([[10.0]]),
        )
    y = truth * np.array([ppo.LCF_UNITS[k][0] for k in net1.lcfs], np.float32)
    lb = float(_lb(net1, spec, obs, y))
    assert float(a) + np.log2(10.0) == pytest.approx(lb + 0.25, abs=1e-4)
    believed = float(fit.window_target(net1).bdp(torch.from_numpy(net1.beliefs(obs))))
    assert abs(believed - lb) > 0.2  # the truth's bins, not net1's belief
    m = onnx.load_from_string(to_onnx(graph))
    assert [i.name for i in m.graph.input] == ["obs", "state", "truth", "cwnd"]
    # One LCF from the truth, the rest from net1 (a patched net1).
    net1_spec = spec.model_copy(update={"inputs": "net1"})
    one = ppo.policy_graph(net1, actor, net1_spec, frozenset({"capacity_bps"}))
    assert one.front.take is not None
    assert one.front.take.tolist() == [n == "capacity_bps" for n in net1.lcfs]
    assert not ppo.policy_graph(net1, actor, net1_spec).uses_truth


def _flow_table(steps: data.Steps, net1: fit.Net1, n: int = 80) -> tuple[pa.Table, np.ndarray]:
    """Episode 0's first n steps as a rollout's step table, window 20."""
    obs = steps.obs[steps.episode == 0][:n]
    cols = {
        "step": pa.array(np.arange(n, dtype=np.int32)),
        "t_ns": pa.array((np.arange(n) * 10_000_000).astype(np.uint64)),
        "obs": pa.FixedSizeListArray.from_arrays(pa.array(obs.reshape(-1)), OBS_DIM),
        "cwnd": pa.array(np.full(n, 20, np.uint32)),
        **{
            f"label_{k}": pa.array(steps.labels[k][steps.episode == 0][:n].astype(np.float64))
            for k in net1.lcfs
        },
    }
    return pa.table(cols), obs


def test_the_black_box_learns_the_teachers_window_on_its_own_bdp() -> None:
    steps, net1 = _bdp_net1()
    table, obs = _flow_table(steps, net1)
    n = obs.shape[0]
    rivals = np.zeros((0, 3))

    def teacher(inputs: Literal["net1", "raw"]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spec = ppo.PPOSpec(
            paths_from=Path("u"),
            net1_from=Path("u"),
            u_rule=0.25,
            input_bins=4,
            algo="dagger",
            share_k=0.0,
            inputs=inputs,
        )
        tr = ppo._trajectory(  # pyright: ignore[reportPrivateUsage]
            table,
            np.zeros(n, np.float32),
            net1,
            front=ppo.front(net1, spec),
            target=ppo.target(net1, spec),
            rules=ppo.Rules(net1, spec),
            spec=spec,
            group=(0, 0),
            rivals=rivals,
        )
        assert tr is not None and tr.teacher is not None
        d = decision_steps(obs[:, OBS_MIN_RTT])
        d = d[d + 1 < n]
        return tr.teacher, _lb(net1, spec, obs[d]).numpy(), d

    on_net1, _, _ = teacher("net1")
    raw, lb_raw, d = teacher("raw")
    lb_net1 = fit.window_target(net1).bdp(torch.from_numpy(net1.beliefs(obs[d]))).numpy()
    # The fixed rule's window (share_k 0), as u on net1's BDP and on the measured one.
    assert np.allclose(on_net1, 0.25)
    assert np.allclose(raw, 0.25 + lb_net1 - lb_raw, atol=1e-5)
    with pytest.raises(ValueError, match="nothing to learn"):
        ppo.PPOSpec(paths_from=Path("u"), net1_from=Path("u"), algo="dagger", inputs="truth")


def _traj(group: tuple[int, int], reward: np.ndarray) -> ppo.Trajectory:
    n = reward.size
    z = np.zeros((n, 3), np.float32)
    return ppo.Trajectory(group, z, z, np.zeros(n, np.float32), reward, np.full(n, 0.9, np.float32))


def test_group_mean_advantage_is_taken_off_at_each_decision() -> None:
    a = _traj((0, 0), np.array([1, 1, 1, 1], np.float32))
    b = _traj((0, 0), np.array([3, 3, 3, 3], np.float32))
    c = _traj((1, 0), np.array([5, 5, 5, 5], np.float32))  # alone in its group
    v = [np.zeros(4, np.float32)] * 3
    adv, ret = ppo._advantages([a, b, c], v, 1.0)  # pyright: ignore[reportPrivateUsage]
    assert np.allclose(adv[0] + adv[1], 0, atol=1e-5) and np.all(adv[1] > 0)
    assert np.allclose(adv[2], ret[2], atol=1e-5) and np.all(ret[1] > ret[0])


def test_team_utility_is_the_copies_mean_at_each_step() -> None:
    a = np.array([1.0, 2.0, 3.0], np.float32)
    b = np.array([3.0, 4.0], np.float32)  # this copy's episode ended a step early
    ta, tb = ppo.team_utility([a, b])
    assert ta.tolist() == [2.0, 3.0, 3.0] and tb.tolist() == [2.0, 3.0]


def test_the_critic_counts_the_flows_active_at_each_decision() -> None:
    # A copy for the whole episode, Cubic from 8 s, UDP from 5 to 15 s.
    rivals = np.array([[0, 0.0, 30.0], [1, 8.0, 30.0], [2, 5.0, 15.0]])
    counts = ppo.active(rivals, np.array([1.0, 6.0, 10.0, 20.0]))
    assert (counts * 4).tolist() == [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]


def test_td3bc_moves_the_actor_toward_the_better_action() -> None:
    # Reward peaks at u = 0.5; the cloned rule starts at 0.25 and its data
    # explore around it, so the critic's gradient points up.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        hidden=8,
        critic_hidden=32,
        algo="td3bc",
        updates_per_iteration=150,
        batch_trajectories=8,
        lr=3e-3,
        critic_lr=3e-3,
    )
    learner = ppo._TD3Learner(5, 4, spec)  # pyright: ignore[reportPrivateUsage]

    def traj(k: int) -> ppo.Trajectory:
        n = 40
        u = (0.25 + spec.noise_sigma * rng.standard_normal(n)).astype(np.float32)
        inp = rng.standard_normal((n, 6)).astype(np.float32)  # features and previous u
        critic = rng.standard_normal((n, 4)).astype(np.float32)
        reward = (-((u - 0.5) ** 2)).astype(np.float32)
        return ppo.Trajectory((k, 0), inp, critic, u, reward, np.full(n, 0.5, np.float32))

    logs = {}
    for it in range(4):
        logs = learner.update([traj(k) for k in range(16)], rng, actor=it > 0)
    # It moves up, but stays within a few sigma of the data it learned from.
    assert 0.26 < logs["u_mean"] < 0.25 + 2 * spec.noise_sigma
    assert np.isfinite(logs["critic_loss"])


def test_the_dagger_teacher_pushes_only_while_stock_tcp_is_active() -> None:
    share_u = np.array([0.5, 0.6, 0.7, 0.8], np.float32)
    # copies only, Cubic and a copy, UDP only, Cubic alone (counts over 4)
    counts = np.array([[2, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]], np.float32) / 4
    assert np.allclose(ppo.teacher_u(share_u, counts, 0.25), [0.25, 0.6, 0.25, 0.8])


def test_dagger_needs_the_fixed_base() -> None:
    with pytest.raises(ValueError, match="base"):
        ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), algo="dagger", base="share")


def test_dagger_learns_to_push_where_the_teacher_does() -> None:
    # The teacher pushes to 0.75 where input 0 is positive and holds the
    # rule's 0.25 elsewhere; the cloned actor starts at 0.25 everywhere.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        hidden=8,
        algo="dagger",
        updates_per_iteration=100,
        batch_trajectories=8,
        lr=3e-3,
    )
    learner = ppo._DAggerLearner(5, 6, spec)  # pyright: ignore[reportPrivateUsage]
    first = learner.update([_dagger_traj(k, rng) for k in range(16)], rng)
    logs = first
    for _ in range(3):
        logs = learner.update([_dagger_traj(k, rng) for k in range(16)], rng)
    assert first["teacher_gap_before"] > 0.2  # the clone: 0.25 everywhere
    assert logs["teacher_gap_before"] < 0.1  # on rollouts it has not learned from
    assert logs["u_where_push"] > 0.6 and logs["u_where_rule"] < 0.4


def _dagger_traj(k: int, rng: np.random.Generator) -> ppo.Trajectory:
    """A stock TCP flow is active where input 0 is positive; there the
    teacher pushes to the share rule's 0.75, elsewhere it holds 0.25."""
    n = 40
    inp = rng.standard_normal((n, 6)).astype(np.float32)
    tcp = (inp[:, 0] > 0).astype(np.float32)
    share = np.full(n, 0.75, np.float32)
    teacher = np.where(tcp > 0, share, 0.25).astype(np.float32)
    z = np.zeros(n, np.float32)
    return ppo.Trajectory((k, 0), inp, inp, z, z, z, None, teacher, tcp, share)


def test_a_gate_student_starts_near_the_rule_and_learns_which_flows_share() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        hidden=8,
        algo="dagger",
        student="gate",
        updates_per_iteration=100,
        batch_trajectories=8,
        lr=3e-3,
    )
    learner = ppo._DAggerLearner(5, 6, spec)  # pyright: ignore[reportPrivateUsage]
    first = learner.update([_dagger_traj(k, rng) for k in range(16)], rng)
    assert abs(first["p_where_tcp_before"] - ppo.P0) < 1e-4
    assert abs(first["u_where_rule_before"] - (0.25 + ppo.P0 * 0.5)) < 1e-4
    logs = first
    for _ in range(3):
        logs = learner.update([_dagger_traj(k, rng) for k in range(16)], rng)
    assert logs["p_where_tcp_before"] > 0.8 and logs["p_elsewhere_before"] < 0.2
    assert logs["u_where_push_before"] > 0.6 and logs["u_where_rule_before"] < 0.4


@pytest.mark.parametrize("algo", ["ppo", "td3bc", "dagger"])
def test_a_learner_can_start_from_a_saved_actor(tmp_path: Path, algo: str) -> None:
    saved = RecurrentActor(5, 8)
    saved.clone_rule(0.6, 3.0)
    torch.save(saved.state_dict(), tmp_path / "actor.pt")
    spec = ppo.PPOSpec(
        paths_from=Path("x"),
        net1_from=Path("x"),
        hidden=8,
        algo=algo,  # pyright: ignore[reportArgumentType]
        init_actor=tmp_path / "actor.pt",
    )
    learner = ppo._make_learner(spec, 5, 4)  # pyright: ignore[reportPrivateUsage]
    x = torch.randn(2, 7, 6)
    assert torch.allclose(learner.actor.sequence(x), saved.sequence(x))
    if isinstance(learner, ppo._TD3Learner):  # pyright: ignore[reportPrivateUsage]
        assert torch.allclose(learner.actor_target.sequence(x), saved.sequence(x))


def test_the_gate3_teacher_pushes_with_tcp_and_cooperates_with_copies_only() -> None:
    share = np.array([0.6, 0.6, 0.6, 0.6], np.float32)
    coop = np.array([-0.5, -0.5, -0.5, -0.5], np.float32)
    # copies only, Cubic and a copy, UDP only, nothing (counts over 4)
    counts = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32) / 4
    got = ppo.teacher3_u(share, coop, counts, 0.25)
    assert np.allclose(got, [-0.5, 0.6, 0.25, 0.25])


def test_a_gate3_student_learns_both_beliefs() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = ppo.PPOSpec(
        paths_from=Path("unused"),
        net1_from=Path("unused"),
        u_rule=0.25,
        hidden=8,
        algo="dagger",
        student="gate3",
        updates_per_iteration=100,
        batch_trajectories=8,
        lr=3e-3,
    )
    learner = ppo._DAggerLearner(5, 6, spec)  # pyright: ignore[reportPrivateUsage]

    def traj(k: int) -> ppo.Trajectory:
        n = 40
        inp = rng.standard_normal((n, 6)).astype(np.float32)
        tcp = (inp[:, 0] > 0).astype(np.float32)
        copies = (inp[:, 1] > 0).astype(np.float32)
        share, coop = np.full(n, 0.75, np.float32), np.full(n, -0.25, np.float32)
        counts = np.stack([copies, tcp, 0 * tcp], axis=1) / 4
        teacher = ppo.teacher3_u(share, coop, counts, 0.25)
        z = np.zeros(n, np.float32)
        return ppo.Trajectory((k, 0), inp, inp, z, z, z, None, teacher, tcp, share, copies, coop)

    first = learner.update([traj(k) for k in range(16)], rng)
    assert abs(first["q_where_copies_before"] - ppo.P0) < 1e-4
    logs = first
    for _ in range(4):
        logs = learner.update([traj(k) for k in range(16)], rng)
    assert logs["p_where_tcp_before"] > 0.8 and logs["p_elsewhere_before"] < 0.2
    assert logs["q_where_copies_before"] > 0.8 and logs["q_elsewhere_before"] < 0.5
    assert logs["teacher_gap_before"] < 0.15


def test_a_gate_student_is_taught_by_dagger_and_continued_by_ppo_only() -> None:
    ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), student="gate")
    for algo, base in (("td3bc", "fixed"), ("ppo", "share")):
        with pytest.raises(ValueError, match="continued"):
            ppo.PPOSpec(
                paths_from=Path("x"),
                net1_from=Path("x"),
                student="gate",
                algo=algo,  # pyright: ignore[reportArgumentType]
                base=base,  # pyright: ignore[reportArgumentType]
            )


def test_only_a_gate3_student_chooses_what_to_do_with_copies() -> None:
    with pytest.raises(ValueError, match="gate3_copies"):
        ppo.PPOSpec(
            paths_from=Path("x"),
            net1_from=Path("x"),
            algo="dagger",
            student="gate",
            gate3_copies="rule",
        )


def test_only_a_gate3_student_has_a_threshold_for_copies() -> None:
    with pytest.raises(ValueError, match="gate_threshold_q"):
        ppo.PPOSpec(
            paths_from=Path("x"),
            net1_from=Path("x"),
            algo="dagger",
            student="gate",
            gate_threshold_q=0.5,
        )


def test_noise_seed_is_left_out_unless_set(tmp_path: Path) -> None:
    pol = tmp_path / "p.onnx"
    pol.write_bytes(b"x")
    cc = AgentCC(policy=OnnxPolicy(path=pol, sha256="0" * 64), noise_sigma=0.1)
    assert "noise_seed" not in json.loads(cc.model_dump_json())
    seeded = cc.model_copy(update={"noise_seed": 5})
    assert json.loads(seeded.model_dump_json())["noise_seed"] == 5
    path = {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24], "level_ms": 1000},
        "delay": {"mahak_compat_mrtt_ms": 10},
        "buffer": {"packets": 100},
    }
    one, two = (
        EpisodeSpec.model_validate({"path": path, "cc": c, "seed": 3}) for c in (cc, seeded)
    )
    assert one.episode_id() != two.episode_id()


@pytest.mark.uml
@pytest.mark.parametrize(
    "case",
    [
        ("own", False, "ppo", "fixed", "u"),
        ("team", True, "ppo", "share", "u"),
        ("team", True, "td3bc", "share", "u"),
        ("team", False, "dagger", "fixed", "u"),
        ("team", False, "dagger", "fixed", "gate"),
        ("team", False, "dagger", "fixed", "gate3"),
    ],
)
def test_ppo_loop_runs_an_iteration_with_copies(
    build: object, tmp_path: Path, case: tuple[str, bool, str, str, str]
) -> None:
    reward, flows, algo, base, student = case
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    spec = _tiny_loop(tmp_path).model_copy(
        update={
            "copies_reward": reward,
            "critic_flows": flows,
            "algo": algo,
            "base": base,
            "student": student,
        }
    )
    report = ppo.run_ppo(spec, tmp_path / "out", build, workers=2)
    initial, entry = report["iterations"]
    assert entry["rollouts"]["ok"] == 2
    # Each rollout's flows (the observed one and, with copies, a cross agent).
    assert entry["learner"]["trajectories"] >= 2 and entry["learner"]["decisions"] > 100
    assert entry["learner"]["kl_to_rule"] >= 0 and initial["eval"]["pairs"] == 0
    assert (tmp_path / "out" / "policy.onnx").exists()


def _tiny_loop(tmp_path: Path) -> ppo.PPOSpec:
    """One iteration on one 24 Mbit/s link against a copy, net1 _bdp_net1."""
    _, net1 = _bdp_net1()
    fit.save_net1(net1, fit.Net1Spec(width=64, bins=16, lcfs=net1.lcfs), tmp_path / "net1.pt")  # pyright: ignore[reportArgumentType]
    path = {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24], "level_ms": 1000},
        "delay": {"mahak_compat_mrtt_ms": 10},
        "buffer": {"packets": 100},
    }
    camp = {
        "name": "p",
        "ccs": [{"kind": "stock", "name": "cubic"}],
        "sweep": {"kind": "paths", "paths": [path]},
        "conditions": [{}],
    }
    (tmp_path / "paths.json").write_text(json.dumps(camp))
    return ppo.PPOSpec(
        paths_from=tmp_path / "paths.json",
        eval_from=tmp_path / "paths.json",
        net1_from=tmp_path / "net1.pt",
        input_bins=4,
        hidden=8,
        critic_hidden=16,
        iterations=1,
        links_per_iteration=1,
        group=2,
        duration_ms=3000,
        compete=1.0,
        copies=(2,),
        epochs=1,
        critic_warmup=0,
        copies_reward="team",
        updates_per_iteration=5,
    )


@pytest.mark.uml
def test_a_taught_gate3_student_hands_over_to_ppo(build: object, tmp_path: Path) -> None:
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    taught = _tiny_loop(tmp_path).model_copy(update={"algo": "dagger", "student": "gate3"})
    first = ppo.run_ppo(taught, tmp_path / "taught", build, workers=2)
    assert np.isfinite(first["iterations"][-1]["learner"]["explained_variance"])
    rl = taught.model_copy(update={"algo": "ppo", "init_from": tmp_path / "taught", "anchor": 1.0})
    report = ppo.run_ppo(rl, tmp_path / "rl", build, workers=2)
    assert report["started_from"] == {
        "actor": str(tmp_path / "taught" / "actor.pt"),
        "critic": str(tmp_path / "taught" / "critic.pt"),
    }
    _, entry = report["iterations"]
    assert entry["rollouts"]["ok"] == 2 and entry["learner"]["kl_to_start"] >= 0
    # Its first policy is the taught one, graph for graph.
    start = (tmp_path / "rl" / "initial" / "policy.onnx").read_bytes()
    assert start == (tmp_path / "taught" / "policy.onnx").read_bytes()


def test_broad_competition_starts_stock_flows_late_and_adds_a_second() -> None:
    rng = np.random.default_rng(3)
    copy = AgentCC(policy=OnnxPolicy(path=Path("p.onnx"), sha256="0" * 64))
    draws = [ppo._competition(rng, (2, 4), copy, "broad") for _ in range(400)]  # pyright: ignore[reportPrivateUsage]
    stock = [d for d in draws if not any(isinstance(c.cc, AgentCC) for c in d)]  # pyright: ignore[reportAttributeAccessIssue]
    copies = [d for d in draws if d not in stock]
    assert stock and copies
    assert all(c.start_ms == 0 for d in copies for c in d)  # pyright: ignore[reportAttributeAccessIssue]
    starts = [c.start_ms for d in stock for c in d]  # pyright: ignore[reportAttributeAccessIssue]
    assert min(starts) >= 0 and max(starts) <= ppo.BROAD_START_MS and len(set(starts)) > 50
    assert {len(d) for d in stock} == {1, 2}
    assert {d[0].cc for d in stock} == {"cubic", "bbr"}  # pyright: ignore[reportAttributeAccessIssue]


def test_at_start_competition_starts_every_flow_at_zero() -> None:
    rng = np.random.default_rng(3)
    copy = AgentCC(policy=OnnxPolicy(path=Path("p.onnx"), sha256="0" * 64))
    draws = [ppo._competition(rng, (2, 4), copy) for _ in range(200)]  # pyright: ignore[reportPrivateUsage]
    assert all(c.start_ms == 0 for d in draws for c in d)  # pyright: ignore[reportAttributeAccessIssue]
    assert all(len(d) == 1 for d in draws if not isinstance(d[0].cc, AgentCC))  # pyright: ignore[reportAttributeAccessIssue]


def test_diverse_competition_draws_every_controller_but_the_held_out() -> None:
    rng = np.random.default_rng(5)
    copy = AgentCC(policy=OnnxPolicy(path=Path("p.onnx"), sha256="0" * 64))
    draws = [ppo._competition(rng, (2, 4), copy, "diverse") for _ in range(3000)]  # pyright: ignore[reportPrivateUsage]
    stock = [c.cc for d in draws for c in d if not isinstance(c.cc, AgentCC)]  # pyright: ignore[reportAttributeAccessIssue]
    assert set(stock) == set(ppo.DIVERSE_WEIGHT)
    assert not set(stock) & ppo.HELD_OUT
    assert stock.count("cubic") > stock.count("vegas")


def test_dynamic_competition_spreads_arrivals_and_departures_over_the_episode() -> None:
    rng = np.random.default_rng(7)
    copy = AgentCC(policy=OnnxPolicy(path=Path("p.onnx"), sha256="0" * 64))
    draws = [ppo._competition(rng, (2, 4), copy, "dynamic", 120_000) for _ in range(3000)]  # pyright: ignore[reportPrivateUsage]
    flows = [c for d in draws for c in d if isinstance(c, TcpCross)]
    copies = [c for c in flows if isinstance(c.cc, AgentCC)]
    stock = [c for c in flows if not isinstance(c.cc, AgentCC)]
    assert all(c.start_ms == 0 and c.stop_ms is None for c in copies)
    assert not {str(c.cc) for c in stock} & ppo.HELD_OUT
    starts = np.array([c.start_ms for c in stock])
    assert 0.25 < (starts == 0).mean() < 0.45
    assert 60_000 < starts.max() <= 72_000
    stops = [(c.start_ms, c.stop_ms) for c in stock if c.stop_ms is not None]
    assert 0.2 < len(stops) / len(stock) < 0.4
    assert all(a + 5000 <= b <= 120_000 for a, b in stops)


def test_alone_links_have_no_cross_flows_and_shares_add_up(tmp_path: Path) -> None:
    pol = tmp_path / "p.onnx"
    pol.write_bytes(b"x")
    with pytest.raises(ValueError, match="at most 1"):
        ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), alone=0.6, compete=0.5)
    spec = ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), alone=0.4, compete=0.4)
    path = ppo.PathSpec.model_validate(
        {
            "trace": {"kind": "periodic_levels", "levels_mbps": [10.0], "level_ms": 2000},
            "delay": {"mahak_compat_mrtt_ms": 20},
            "buffer": {"packets": 100},
        }
    )
    cond = ppo.Condition.model_validate({"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]})
    rng = np.random.default_rng(1)
    eps = {d: ppo._group_episodes((path, cond), pol, spec, rng, d) for d in (0.1, 0.5, 0.9)}  # pyright: ignore[reportPrivateUsage]
    assert all(e.cross == [] for e in eps[0.1])
    assert all(e.cross for e in eps[0.5])
    assert all([c.cc for c in e.cross] == ["cubic"] for e in eps[0.9])  # pyright: ignore[reportAttributeAccessIssue]


def test_info_traffic_hides_the_capacity_part_of_the_time() -> None:
    trace = ppo.PeriodicLevels(levels_mbps=[10.0, 20.0], level_ms=3000)
    rng = np.random.default_rng(5)
    draws = [ppo._info(rng, trace, 30_000) for _ in range(300)]  # pyright: ignore[reportPrivateUsage]
    kinds = {"leave": 0, "udp": 0, "tcp": 0}
    for flows in draws:
        assert 1 <= len(flows) <= ppo.MAX_CROSS
        for f in flows:
            assert f.stop_ms is not None and 0 <= f.start_ms < f.stop_ms <= 30_000
        if isinstance(flows[0], ppo.UdpCross):
            kinds["udp"] += 1
            # 0.3 to 0.7 of the mean capacity, 15 Mbit/s here.
            assert all(4.5 <= f.rate_mbps <= 10.5 for f in flows)  # pyright: ignore[reportAttributeAccessIssue]
        elif len(flows) == 1 and flows[0].start_ms == 0 and flows[0].stop_ms <= 18_000:
            kinds["leave"] += 1
        else:
            kinds["tcp"] += 1
            assert len(flows) > 1
    assert all(n > 60 for n in kinds.values())


def test_info_and_rise_shares_draw_nothing_when_off(tmp_path: Path) -> None:
    pol = tmp_path / "p.onnx"
    pol.write_bytes(b"x")
    with pytest.raises(ValueError, match="at most 1"):
        ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), alone=0.5, compete=0.3, info=0.3)
    path = ppo.PathSpec.model_validate(
        {
            "trace": {"kind": "periodic_levels", "levels_mbps": [10.0], "level_ms": 2000},
            "delay": {"mahak_compat_mrtt_ms": 20},
            "buffer": {"packets": 100},
        }
    )
    cond = ppo.Condition()
    off = ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), alone=0.4, compete=0.4)
    a = ppo._group_episodes((path, cond), pol, off, np.random.default_rng(3), 0.9)  # pyright: ignore[reportPrivateUsage]
    b = ppo._group_episodes((path, cond), pol, off, np.random.default_rng(3), 0.9)  # pyright: ignore[reportPrivateUsage]
    assert a == b and all(e.path.trace == path.trace and e.cross == [] for e in a)
    on = off.model_copy(update={"compete": 0.2, "info": 0.4, "rise": 1.0})
    eps = ppo._group_episodes((path, cond), pol, on, np.random.default_rng(3), 0.7)  # pyright: ignore[reportPrivateUsage]
    trace = eps[0].path.trace
    assert isinstance(trace, ppo.PeriodicLevels)
    assert trace.levels_mbps == pytest.approx([5.0, 20.0], rel=1e-3)  # the trace's packet mean
    assert 1000 <= trace.level_ms <= 5000
    assert all(e.cross for e in eps)


def test_copies_share_sets_how_often_competition_is_copies() -> None:
    copy = AgentCC(policy=OnnxPolicy(path=Path("p.onnx"), sha256="0" * 64))
    rng = np.random.default_rng(11)
    draws = [
        ppo._competition(rng, (2, 4), copy, "diverse", 30_000, copies_share=0.8)
        for _ in range(4000)
    ]  # pyright: ignore[reportPrivateUsage]
    share = np.mean([isinstance(d[0].cc, AgentCC) for d in draws])  # pyright: ignore[reportAttributeAccessIssue]
    assert 0.77 < share < 0.83
    old = np.random.default_rng(11)
    new = np.random.default_rng(11)
    a = [ppo._competition(old, (2, 4), copy, "diverse") for _ in range(50)]  # pyright: ignore[reportPrivateUsage]
    b = [
        ppo._competition(new, (2, 4), copy, "diverse", 30_000, copies_share=None) for _ in range(50)
    ]  # pyright: ignore[reportPrivateUsage]
    assert a == b


def test_exploration_noise_is_held_and_a_critic_needs_ppo(tmp_path: Path) -> None:
    pol = tmp_path / "p.onnx"
    pol.write_bytes(b"x")
    assert ppo._agent(pol, 0.2, None, 20).noise_hold_steps == 20  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="init_critic"):
        ppo.PPOSpec(
            paths_from=Path("x"), net1_from=Path("x"), algo="td3bc", init_critic=Path("c.pt")
        )


# The handoff from a taught stage to an RL one (init_from, anchor,
# teacher_from).


def _save_run(
    d: Path,
    spec: ppo.PPOSpec,
    net1: fit.Net1,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None = None,
) -> Path:
    """What run_ppo leaves in its output directory, for init_from."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "ppo_spec.json").write_text(spec.model_dump_json())
    fit.save_net1(net1, fit.Net1Spec(width=64, bins=16, lcfs=net1.lcfs), d / "net1.pt")  # pyright: ignore[reportArgumentType]
    torch.save(actor.state_dict(), d / "actor.pt")
    torch.save({} if critic is None else critic.state_dict(), d / "critic.pt")
    return d


def _gate3_traj(k: int, rng: np.random.Generator, width: int = 6) -> ppo.Trajectory:
    """Rules' u fixed (share 0.75, cooperative window -0.25); reward the
    realized u, so PPO pushes the mean up."""
    n = 40
    inp = rng.standard_normal((n, width)).astype(np.float32)
    u = (0.4 + 0.2 * rng.standard_normal(n)).astype(np.float32)
    share, coop = np.full(n, 0.75, np.float32), np.full(n, -0.25, np.float32)
    return ppo.Trajectory(
        (k, 0), inp, inp, u, u, np.full(n, 0.5, np.float32), share=share, coop=coop
    )


def test_a_taught_gate3_student_continues_under_ppo_as_it_was_taught(tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    _, net1 = _bdp_net1()
    common = {
        "paths_from": Path("x"),
        "net1_from": Path("x"),
        "u_rule": 0.25,
        "input_bins": 4,
        "hidden": 8,
        "critic_hidden": 16,
        "student": "gate3",
        "gate_threshold": 0.3,
        "noise_sigma": 0.2,
    }
    taught_spec = ppo.PPOSpec.model_validate(common | {"algo": "dagger"})
    w = ppo._width(net1, taught_spec)  # pyright: ignore[reportPrivateUsage]
    taught = ppo._DAggerLearner(w, w + 1, taught_spec)  # pyright: ignore[reportPrivateUsage]
    with torch.no_grad():  # beliefs p = 0.7, q = 0.6
        taught.actor.head.bias.copy_(torch.tensor([np.arctanh(0.4), np.arctanh(0.2)]))
    run = _save_run(tmp_path / "taught", taught_spec, net1, taught.actor, taught.critic)
    spec = ppo.PPOSpec.model_validate(common | {"algo": "ppo", "init_from": run, "lr": 3e-3})
    learner = ppo._make_learner(spec, w, w + 1, ppo.earlier_run(run, net1))  # pyright: ignore[reportPrivateUsage]
    assert isinstance(learner, ppo._Learner)  # pyright: ignore[reportPrivateUsage]
    trs = [_gate3_traj(k, rng, w + 1) for k in range(16)]
    x, _ = ppo._pad([tr.inp for tr in trs])  # pyright: ignore[reportPrivateUsage]
    with torch.no_grad():
        was = taught._u(trs, taught.actor.pre(x))  # pyright: ignore[reportPrivateUsage]
    # The deployed mix, as the exported graph's: p and q past the 0.3 threshold.
    p, q = (0.7 - 0.3) / 0.7, (0.6 - 0.3) / 0.7
    want = 0.25 + p * 0.5 + (1 - p) * q * -0.5
    mu = np.concatenate(learner.means(trs))
    assert np.allclose(mu, want, atol=1e-5)
    assert np.allclose(
        mu,
        np.concatenate([w[: tr.u.size] for w, tr in zip(was.numpy(), trs, strict=True)]),
        atol=1e-6,
    )
    # The policy it exports is the taught one, graph for graph.
    assert to_onnx(ppo.policy_graph(net1, learner.actor, spec)) == to_onnx(
        ppo.policy_graph(net1, taught.actor, taught_spec)
    )
    logs = {}
    for _ in range(3):
        logs = learner.update(trs, rng)
    assert logs["u_mean"] > want + 0.01 and logs["kl_to_start"] > 0


def test_dagger_learns_the_students_value_for_ppo_to_continue_from(tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    _, net1 = _bdp_net1()
    common = {
        "paths_from": Path("x"),
        "net1_from": Path("x"),
        "u_rule": 0.25,
        "hidden": 8,
        "critic_hidden": 32,
        "critic_lr": 3e-3,
        "updates_per_iteration": 20,
        "batch_trajectories": 8,
        "epochs": 8,
    }
    spec = ppo.PPOSpec.model_validate(common | {"algo": "dagger"})
    learner = ppo._DAggerLearner(5, 6, spec)  # pyright: ignore[reportPrivateUsage]

    def traj(k: int) -> ppo.Trajectory:
        # The reward follows critic input 0, so a value network can learn it.
        tr = _dagger_traj(k, rng)
        reward = (0.5 + 0.5 * np.tanh(tr.critic[:, 0])).astype(np.float32)
        return ppo.Trajectory(
            tr.group,
            tr.inp,
            tr.critic,
            tr.u,
            reward,
            np.full(reward.size, 0.5, np.float32),
            None,
            tr.teacher,
            tr.tcp,
            tr.share,
        )

    first = learner.update([traj(k) for k in range(16)], rng)
    logs = first
    for _ in range(6):
        logs = learner.update([traj(k) for k in range(16)], rng)
    assert logs["explained_variance"] > 0.5 > first["explained_variance"]
    run = _save_run(tmp_path / "taught", spec, net1, learner.actor, learner.critic)
    rl = ppo.PPOSpec.model_validate(common | {"algo": "ppo", "init_from": run})
    cont = ppo._make_learner(rl, 5, 6, ppo.earlier_run(run, net1))  # pyright: ignore[reportPrivateUsage]
    c = torch.randn(3, 6)
    assert torch.equal(cont.critic(c), learner.critic(c))
    x = torch.randn(2, 7, 6)
    assert torch.equal(cont.actor.sequence(x), learner.actor.sequence(x))
    # A value network reading other inputs is left behind: PPO's starts blank.
    other = rl.model_copy(update={"critic_flows": True})
    blank = ppo._make_learner(other, 5, 6, ppo.earlier_run(run, net1))  # pyright: ignore[reportPrivateUsage]
    assert not torch.equal(blank.critic(c), learner.critic(c))


def test_init_from_refuses_another_net1_or_another_actor(tmp_path: Path) -> None:
    _, net1 = _bdp_net1()
    spec = ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), hidden=8, algo="dagger")
    run = _save_run(tmp_path / "taught", spec, net1, RecurrentActor(5, 8))
    other = fit.Net1(net1.scaler, copy.deepcopy(net1.net), net1.lcf_scaler, net1.lcfs)
    with torch.no_grad():
        next(other.net.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="another net1"):
        ppo.earlier_run(run, other)
    wider = spec.model_copy(update={"algo": "ppo", "hidden": 16, "init_from": run})
    with pytest.raises(ValueError, match="differs in hidden"):
        ppo._make_learner(wider, 5, 6, ppo.earlier_run(run, net1))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="init_from replaces"):
        ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), init_from=run, init_actor=run)
    with pytest.raises(ValueError, match="anchor"):
        ppo.PPOSpec(paths_from=Path("x"), net1_from=Path("x"), algo="dagger", anchor=1.0)


def test_the_anchor_keeps_ppo_nearer_the_policy_it_started_from() -> None:
    def moved(anchor: float) -> float:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        spec = ppo.PPOSpec(
            paths_from=Path("x"),
            net1_from=Path("x"),
            u_rule=0.25,
            hidden=8,
            critic_hidden=16,
            noise_sigma=0.2,
            lr=3e-3,
            anchor=anchor,
        )
        learner = ppo._make_learner(spec, 5, 6)  # pyright: ignore[reportPrivateUsage]
        trs = [_gate3_traj(k, rng) for k in range(16)]  # reward = realized u
        logs = {}
        for _ in range(4):
            logs = learner.update(trs, rng)
        return logs["kl_to_start"]

    free, held = moved(0.0), moved(20.0)
    assert free > 0 and held < 0.5 * free


def test_a_policy_teacher_labels_the_window_it_would_set_on_the_students_bdp(
    tmp_path: Path,
) -> None:
    steps, net1 = _bdp_net1()
    common = {"paths_from": Path("x"), "net1_from": Path("x"), "input_bins": 4, "hidden": 8}
    oracle = ppo.PPOSpec.model_validate(common | {"inputs": "truth"})
    actor = RecurrentActor(ppo._width(net1, oracle), 8)  # pyright: ignore[reportPrivateUsage]
    actor.clone_rule(0.5, oracle.action_max)
    d = _save_run(tmp_path / "oracle", oracle, net1, actor)
    teacher = ppo._PolicyTeacher.load(d, net1)  # pyright: ignore[reportPrivateUsage]
    table, obs = _flow_table(steps, net1)
    labels = {"net1": 0.0, "raw": 0.0}
    for inputs in labels:
        spec = ppo.PPOSpec.model_validate(
            common | {"algo": "dagger", "inputs": inputs, "teacher_from": d}
        )
        tr = ppo._trajectory(  # pyright: ignore[reportPrivateUsage]
            table,
            np.zeros(obs.shape[0], np.float32),
            net1,
            front=ppo.front(net1, spec),
            target=ppo.target(net1, spec),
            rules=ppo.Rules(net1, spec),
            spec=spec,
            group=(0, 0),
            rivals=np.zeros((0, 3)),
            teacher=teacher,
        )
        assert tr is not None and tr.teacher is not None
        dec = decision_steps(obs[:, OBS_MIN_RTT])
        dec = dec[dec + 1 < obs.shape[0]]
        y = np.stack(
            [steps.labels[k][steps.episode == 0][dec] * ppo.LCF_UNITS[k][0] for k in net1.lcfs],
            axis=1,
        ).astype(np.float32)
        z = torch.from_numpy(net1.beliefs(obs[dec]))
        true_bdp = fit.window_target(net1).bdp(fit.oracle(net1, z, y)).numpy()
        own = _lb(net1, spec, obs[dec]).numpy()  # the student's believed or measured BDP
        assert np.allclose(tr.teacher, 0.5 + true_bdp - own, atol=1e-4)
        assert np.abs(true_bdp - own).max() > 0.05  # the frames differ
    with pytest.raises(ValueError, match="teacher_from"):
        ppo.PPOSpec.model_validate(common | {"algo": "ppo", "teacher_from": d})
    plain = ppo.PPOSpec.model_validate(common)
    with pytest.raises(ValueError, match="reads the truth"):
        ppo._PolicyTeacher.load(_save_run(tmp_path / "plain", plain, net1, actor), net1)  # pyright: ignore[reportPrivateUsage]
