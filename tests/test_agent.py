"""Closed-loop sage-v1 policies driving tcp_agent in UML (needs EXPLICCIT_UML_DIR)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import OBS_DIM
from expliccit_env.records import Records, load_records
from expliccit_env.spec import (
    AgentCC,
    DropTail,
    EpisodeSpec,
    MahakMrtt,
    MahakStep,
    OnnxPolicy,
    PathSpec,
)
from expliccit_env.uml import UmlBuild
from tests import onnx_policies

pytestmark = pytest.mark.uml


def agent_spec(tmp: Path, policy: bytes, mode: str, period_ms: int = 10) -> EpisodeSpec:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "policy.onnx"
    path.write_bytes(policy)
    cc = AgentCC(
        mode=mode,  # type: ignore[arg-type]
        policy=OnnxPolicy(path=path, sha256=hashlib.sha256(policy).hexdigest()),
        period_ms=period_ms,
    )
    return EpisodeSpec(
        path=PathSpec(
            trace=MahakStep(bw_mbps=20, step=1.0),
            delay=MahakMrtt(mahak_compat_mrtt_ms=10),
            buffer=DropTail(packets=200),
        ),
        cc=cc,
        duration_ms=3000,
        seed=3,
    )


def run_raw(build: UmlBuild, tmp: Path, spec: EpisodeSpec) -> tuple[EpisodeOk, Records]:
    outcome = run_episode(spec, build, tmp / "work", keep_raw=True)
    assert isinstance(outcome, EpisodeOk), outcome.row
    return outcome, load_records(tmp / "work" / outcome.episode_id / "raw")


def sage_window(actions: list[float]) -> list[int]:
    """Sage's accumulator: from 10 packets, times 2^a, rounded up, in [4, 20000]."""
    precise, out = 10.0, []
    for a in actions:
        precise = min(precise * 2.0 ** float(np.float32(a)), 2147483647.0)
        cwnd = min(math.ceil(precise), 2147483647)
        if cwnd < 4:
            cwnd, precise = 4, 4.0
        if cwnd > 20000:
            cwnd, precise = 20000, 20000.0
        out.append(cwnd)
    return out


def steps_of(outcome: EpisodeOk) -> pa.Table:
    return outcome.tables["steps"]


def test_hold_agent_follows_sage_window(build: UmlBuild, tmp_path: Path) -> None:
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(0.1), "hold")
    )
    acts, steps = rec.agent_action, steps_of(outcome)
    n = steps.num_rows
    assert 250 <= n <= 300  # about one step per 10 ms over 3 s
    assert acts.size == n == outcome.row["agent_actions"]
    assert acts["cwnd"].tolist() == sage_window([0.1] * n)
    assert set(steps["applied_action"].to_numpy().tolist()) == {float(np.float32(0.1))}
    # The next step reports the action taken as Sage's column 76.
    assert np.all(steps["action"].to_numpy()[:-1] == 0.1)
    assert np.all(acts["pacing_rate"][5:] > 0)  # paced once srtt is known
    # Hold keeps the socket at exactly what the policy set.
    obs = rec.tcp_obs
    k = np.searchsorted(acts["t_ns"], obs["t_ns"], side="right") - 1
    seen = obs["snd_cwnd"][1:][k[1:] >= 0]
    want = acts["cwnd"][k[1:][k[1:] >= 0]]
    assert np.mean(seen == want) > 0.95


def test_pure_agent_paces_every_delivery_as_sages_kernel(build: UmlBuild, tmp_path: Path) -> None:
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(0.05), "pure")
    )
    acts = rec.agent_action
    # The window is Sage's accumulator, as under hold; the kernel paces.
    assert acts["cwnd"].tolist() == sage_window([0.05] * acts.size)
    assert set(acts["pacing_rate"].tolist()) == {0}
    # Each arriving segment sees the rate the previous delivery set:
    # max(cwnd, packets_out) * MTU (1448 + 52) * 1e6 / srtt, rounded down.
    obs = rec.tcp_obs
    obs = obs[(obs["srtt_us"] > 0) & (obs["packets_out"] <= obs["snd_cwnd"])][10:]
    want = obs["snd_cwnd"].astype(np.uint64) * 1500 * 1_000_000 // obs["srtt_us"]
    assert np.mean(obs["pacing_rate"] == want) > 0.95
    assert outcome.row["utilization"] > 0.5


def test_cubic_agent_reads_orca_statistics_every_period(build: UmlBuild, tmp_path: Path) -> None:
    # a = 1e-7 * the period in us: 0.002 for Orca's 20 ms, a window change
    # too small to round to anything at these windows.
    spec = agent_spec(tmp_path / "t", onnx_policies.on_cc_info(3, 1e-7), "cubic", 20)
    outcome, _ = run_raw(build, tmp_path / "t", spec)
    period = steps_of(outcome)["applied_action"].to_numpy()[1:] / 1e-7
    assert 120 <= period.size <= 150  # one step per 20 ms over 3 s
    assert period.min() >= 19_999 and np.median(period) < 21_000
    # a = 1e-9 * the mean rate sample: about the 20 Mbit/s link's rate.
    spec = agent_spec(tmp_path / "r", onnx_policies.on_cc_info(1, 1e-9), "cubic", 20)
    outcome, _ = run_raw(build, tmp_path / "r", spec)
    rate = steps_of(outcome)["applied_action"].to_numpy()[50:] / 1e-9 * 8
    assert 0.8 * 20e6 < np.median(rate) < 1.05 * 20e6


def test_rate_agent_paces_at_its_rate_each_half_rtt(build: UmlBuild, tmp_path: Path) -> None:
    # a = 1e-7 * the window's length in us: about 0.001, so the rate stays
    # near its start, 6 Mbit/s, and a reveals each step's length.
    spec = agent_spec(tmp_path / "p", onnx_policies.on_cc_info(0, 1e-7, 7), "rate")
    outcome, rec = run_raw(build, tmp_path / "p", spec)
    acts, obs = rec.agent_action, rec.tcp_obs
    assert set(acts["cwnd"].tolist()) == {20000}
    # From 6 Mbit/s, times 2^a with a about 0.001 a step: at most 1.3 times it in 3 s.
    assert acts["pacing_rate"].min() >= 6e6 / 8 and acts["pacing_rate"].max() < 1.3 * 6e6 / 8
    # The kernel paces at the rate the last action wrote, from the next
    # delivery on: each arriving segment sees the action that stood when the
    # one before it arrived.
    k = np.searchsorted(acts["t_ns"], obs["t_ns"], side="right") - 1
    seen, wrote = obs["pacing_rate"][1:][k[:-1] >= 0], acts["pacing_rate"][k[:-1][k[:-1] >= 0]]
    assert np.mean(seen[5:] == wrote[5:]) > 0.95
    # Each step lasts half the last one's mean RTT.
    period_us = steps_of(outcome)["applied_action"].to_numpy()[5:] / 1e-7
    srtt_us = np.median(obs["srtt_us"][obs["srtt_us"] > 0])
    assert 0.35 * srtt_us < np.median(period_us) < 0.75 * srtt_us
    # The window's mean RTT is the flow's RTT.
    spec = agent_spec(tmp_path / "m", onnx_policies.on_cc_info(5, 1e-7, 7), "rate")
    outcome, rec = run_raw(build, tmp_path / "m", spec)
    mean_us = steps_of(outcome)["applied_action"].to_numpy()[5:] / 1e-7
    srtt_us = np.median(rec.tcp_obs["srtt_us"][rec.tcp_obs["srtt_us"] > 0])
    assert 0.8 * srtt_us < np.median(mean_us) < 1.25 * srtt_us


def test_cc_info_needs_the_cubic_agent(build: UmlBuild, tmp_path: Path) -> None:
    outcome = run_episode(
        agent_spec(tmp_path, onnx_policies.on_cc_info(0, 0.0), "hold"), build, tmp_path / "work"
    )
    assert not isinstance(outcome, EpisodeOk)
    assert "cc_info needs cc agent_cubic or agent_rate" in str(outcome.row.get("failure"))


def test_policy_reads_the_observation(build: UmlBuild, tmp_path: Path) -> None:
    # a = 0.5 r - 0.35 with r = Sage's column 70 (max delivered rate times min
    # RTT over the window). Every action must be that function of the very
    # message the step table recorded, and the window must react to it.
    w = np.zeros(OBS_DIM)
    w[70] = 0.5
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.linear(w, -0.35), "hold")
    )
    steps = steps_of(outcome)
    obs = np.asarray(steps["obs"].combine_chunks().flatten()).reshape(-1, OBS_DIM)
    want = np.float32(0.5) * obs[:, 70].astype(np.float32) + np.float32(-0.35)
    assert np.array_equal(steps["applied_action"].to_numpy(), want.astype(np.float64))
    cwnd = rec.agent_action["cwnd"]
    assert len(set(cwnd.tolist())) > 5 and cwnd[-1] > cwnd.min()


def test_recurrent_state_round_trips(build: UmlBuild, tmp_path: Path) -> None:
    outcome, _ = run_raw(build, tmp_path, agent_spec(tmp_path, onnx_policies.counter(1e-3), "hold"))
    applied = steps_of(outcome)["applied_action"].to_numpy()
    expect = np.array(
        [float(np.float32(np.float32(k) * np.float32(1e-3))) for k in range(applied.size)]
    )
    assert np.array_equal(applied, expect)


def test_cubic_agent_at_zero_leaves_the_window_to_cubic(build: UmlBuild, tmp_path: Path) -> None:
    outcome, _ = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.constant(0.0), "cubic")
    )
    steps = steps_of(outcome)
    read, wrote = steps["cwnd"].to_numpy(), steps["cwnd_set"].to_numpy()
    # 2^0 times the kernel's window: each action sets the window it read.
    assert np.array_equal(wrote, np.maximum(4, read))
    # Cubic grew it from 10, past the path's BDP (33 packets).
    assert wrote.max() > 40


def test_bad_policy_output_fails_the_episode(build: UmlBuild, tmp_path: Path) -> None:
    outcome = run_episode(
        agent_spec(tmp_path, onnx_policies.constant(20.0), "hold"), build, tmp_path / "w"
    )
    assert not isinstance(outcome, EpisodeOk)
    assert "exceeds" in str(outcome.row["failure"])


def test_exploration_noise_is_seeded_and_has_the_requested_spread(
    build: UmlBuild, tmp_path: Path
) -> None:
    base = agent_spec(tmp_path, onnx_policies.constant(0.0), "hold")
    noisy = base.model_copy(update={"cc": base.cc.model_copy(update={"noise_sigma": 0.1})})
    runs = [
        steps_of(run_raw(build, tmp_path / name, spec)[0])["applied_action"].to_numpy()
        for name, spec in (
            ("a", noisy),
            ("b", noisy),
            ("c", noisy.model_copy(update={"seed": 4})),
        )
    ]
    a, b, c = runs
    assert np.array_equal(a, b)  # same episode seed, same noise
    assert not np.array_equal(a[:50], c[:50])
    assert abs(float(a.mean())) < 0.02 and 0.08 < float(a.std()) < 0.12


def test_held_noise_keeps_each_draw_for_its_block(build: UmlBuild, tmp_path: Path) -> None:
    base = agent_spec(tmp_path, onnx_policies.constant(0.0), "hold")
    noisy = base.model_copy(update={"cc": base.cc.model_copy(update={"noise_sigma": 0.1})})
    held = noisy.model_copy(update={"cc": noisy.cc.model_copy(update={"noise_hold_steps": 5})})
    every, kept = (
        steps_of(run_raw(build, tmp_path / name, spec)[0])["applied_action"].to_numpy()
        for name, spec in (("every", noisy), ("held", held))
    )
    n = min(every.size, kept.size)
    assert n > 100
    # The stream is drawn every step either way; a block keeps its first draw.
    assert np.array_equal(kept[:n], every[(np.arange(n) // 5) * 5])


def test_a_policy_reading_cwnd_sets_an_absolute_window(build: UmlBuild, tmp_path: Path) -> None:
    outcome, rec = run_raw(
        build, tmp_path, agent_spec(tmp_path, onnx_policies.window_target(100.0), "hold")
    )
    cwnd = rec.agent_action["cwnd"]
    assert cwnd.size > 100 and np.all(cwnd[2:] == 100)
    read = steps_of(outcome)["cwnd"].to_numpy()  # the input the policy saw
    assert read[0] == 10 and np.all(read[3:] == 100)
