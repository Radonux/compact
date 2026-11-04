"""The recurrent, decision-holding net2 and its PPO learner (train/recurrent.py,
train/ppo.py)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.spec import AgentCC, EpisodeSpec, OnnxPolicy  # noqa: E402
from expliccit_env.train import fit, ppo  # noqa: E402
from expliccit_env.train.export import to_onnx  # noqa: E402
from expliccit_env.train.recurrent import OBS_MIN_RTT, RecurrentActor, decision_steps  # noqa: E402

from .test_train import _bdp_net1  # noqa: E402


def test_decisions_come_once_per_min_rtt_within_10_to_50_ms() -> None:
    rtt = np.array([30.0] * 6 + [np.nan] * 2 + [200.0] * 5 + [4.0] * 3, np.float32)
    # A decision fixes its block from the min RTT it reads: 30 ms, 3 steps;
    # not finite, 1; 200 ms, 5 (50 ms at most); 4 ms, 1 (10 ms at least).
    assert decision_steps(rtt).tolist() == [0, 3, 6, 7, 8, 13, 14, 15]


def _graph(u: float, memory: bool = True) -> tuple[ppo.HeldPolicyGraph, fit.Net1, np.ndarray]:
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
            beliefs.append(float(graph.target(graph.net1(graph.scaler(o)))))  # pyright: ignore[reportOptionalCall]
            cwnd = cwnd * 2 ** (float(a) + 0.1 * rng.standard_normal())  # the guest's noise
    d = decision_steps(obs[:, OBS_MIN_RTT])
    assert np.flatnonzero(decided).tolist() == d.tolist()
    # The cloned rule asks for 2^0.25 times the believed BDP at each decision ...
    for k in d:
        assert windows[k] == pytest.approx(beliefs[k] + 0.25, abs=1e-4)
    # ... and holds the window the decision's noise made until the next one.
    for k, nxt in itertools.pairwise(d):
        held = windows[k + 1 : nxt]
        assert np.ptp(held) < 1e-4 if len(held) > 1 else True


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
def test_ppo_loop_runs_an_iteration_with_copies(build: object, tmp_path: Path) -> None:
    from expliccit_env.uml import UmlBuild  # noqa: PLC0415

    assert isinstance(build, UmlBuild)
    _, net1 = _bdp_net1()
    fit.save_net1(net1, fit.Net1Spec(width=64, bins=16, lcfs=net1.lcfs), tmp_path / "net1.pt")
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
    spec = ppo.PPOSpec(
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
    )
    report = ppo.run_ppo(spec, tmp_path / "out", build, workers=2)
    initial, entry = report["iterations"]
    assert entry["rollouts"]["ok"] == 2
    # Each rollout's flows (the observed one and, with copies, a cross agent).
    assert entry["learner"]["trajectories"] >= 2 and entry["learner"]["decisions"] > 100
    assert entry["learner"]["kl_to_rule"] >= 0 and initial["eval"]["pairs"] == 0
    assert (tmp_path / "out" / "policy.onnx").exists()
