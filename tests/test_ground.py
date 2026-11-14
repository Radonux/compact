"""Grounding (experiments/ground.py): the fits, the library, and a
controller's latents on its own rollouts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.labels import WINDOWED, WINDOWS_MS, label_name  # noqa: E402
from expliccit_env.policy import OBS_DIM  # noqa: E402
from expliccit_env.train import actors, data, fit, lcfs  # noqa: E402
from expliccit_env.train.recurrent import OBS_MIN_RTT, decision_steps  # noqa: E402
from expliccit_env.train.spec import Bins, Net1Spec, PPOSpec  # noqa: E402

from .test_train import EXPLICCIT_LCFS, _noisy  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "ground.py"
_spec = importlib.util.spec_from_file_location("ground", _PATH)
assert _spec is not None and _spec.loader is not None
ground: Any = importlib.util.module_from_spec(_spec)
sys.modules["ground"] = ground
_spec.loader.exec_module(ground)


def test_ground_probe_scores_linear_and_quadratic_relations() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 3))
    y = np.stack([z @ [1.0, -2.0, 0.5] + 3, z[:, 0] * z[:, 1], rng.normal(size=4000)], axis=1)
    y[::7, 0] = np.nan  # undefined rows are left out, not scored as errors
    fit, score = slice(0, 3000), slice(3000, None)
    linear = ground.r2(z[fit], y[fit], z[score], y[score])
    assert linear[0] > 0.999 and linear[1] < 0.1 and linear[2] < 0.05
    q = ground.quadratic(z)
    assert q.shape == (4000, 9)
    assert ground.r2(q[fit], y[fit], q[score], y[score])[1] > 0.999
    # A variable that is sometimes undefined: its indicator carries that.
    b = z.copy()
    b[z[:, 2] > 1, 2] = np.nan
    b_fit, b_score = ground.filled(b[fit], b[score])
    assert b_fit.shape[1] == 4 and np.isfinite(b_score).all()
    target = np.isnan(b[:, 2:3]).astype(np.float64)
    assert ground.r2(b_fit, target[fit], b_score, target[score])[0] > 0.999


def test_ground_selection_finds_the_features_that_span_the_latents() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6000, 6))
    gap = rng.random(6000) < 0.2  # feature 4 is undefined on a fifth of the rows
    cols = [np.ones(6000), *x.T, (gap - gap.mean()) / gap.std()]
    cols[5] = np.where(gap, 0.0, x[:, 4])
    design = np.stack(cols, axis=1)
    lib = ground.Library(
        names=tuple(f"f{i}" for i in range(6)),
        variable=tuple(f"v{i // 2}" for i in range(6)),
        columns=((1,), (2,), (3,), (4,), (5, 7), (6,)),
        same_as={},
        fit=design[:4000],
        score=design[4000:],
        base=(),
        base_fit=np.zeros((4000, 0)),
        base_score=np.zeros((2000, 0)),
    )
    # Two latents: mixes of features 1 and 4, where 4 also acts by being undefined.
    z = np.stack([2 * x[:, 1] + cols[5] + 3 * gap, x[:, 1] - cols[5]], axis=1)
    z += 0.01 * rng.normal(size=z.shape)
    m = ground.Moments.of(lib, z[:4000], z[4000:])
    order = ground.greedy(lib, m, list(range(6)), 3)
    assert set(order[:2]) == {1, 4}
    assert m.r2(ground._columns(lib, order[:2])).min() > 0.999
    assert m.r2(ground._columns(lib, order[:1])).min() < 0.9
    sel = ground._selection(lib, (z[:4000], z[4000:]))
    assert sel["formulas"] == [] and sel["formula_path"] == sel["hidden_path"]
    assert sel["flexible"]["picked"] > 0.99
    assert sel["library"] > 0.999 and set(sel["without"]) >= {"v0", "v2"}
    assert sel["without"]["v0"][-1]["mean"] < 0.9  # no substitute for feature 1


def test_the_grounding_library_holds_every_stored_label_and_composite() -> None:
    rng = np.random.default_rng(2)
    stored = {c for name in lcfs.LCF_UNITS for c in lcfs.columns([name])}
    labels = lcfs.Labels({c: rng.uniform(1, 1e7, 50).astype(np.float32) for c in stored})
    names, b = ground.base_variables(SimpleNamespace(labels=labels))
    # Every label in every window it has, and every composite: nothing is listed twice.
    assert set(names) == set(lcfs.LCF_UNITS) and b.shape == (50, len(names))
    assert {label_name(n, w) for n in WINDOWED for w in WINDOWS_MS} <= set(names) | {
        label_name("drop_fraction", w) for w in WINDOWS_MS
    }
    # A label column that no LCF names still joins.
    labels["made_up_bytes"] = np.full(50, 2000.0, np.float32)
    assert "made_up_bytes" in ground.base_variables(SimpleNamespace(labels=labels))[0]
    # With only one window stored, the other window's composites are left out.
    short = lcfs.Labels({k: v for k, v in labels.items() if not k.endswith("_10ms")})
    assert not any(
        n.endswith("_10ms") for n in ground.base_variables(SimpleNamespace(labels=short))[0]
    )
    # Taking a label out takes out both windows and what is computed from it.
    assert ground.sources("capacity_bps_10ms") == {"capacity_bps"}
    assert ground.sources("fair_share_bps_10ms") == {"capacity_bps", "active_flows"}
    assert ground.sources("sender_cwnd") == {"sender_cwnd"}


def _steps(n_episodes: int, length: int, rng: np.random.Generator) -> Any:
    """Episodes of ``length`` steps with two random-walk variables."""
    n = n_episodes * length
    obs = np.zeros((n, OBS_DIM), np.float32)
    obs[:, OBS_MIN_RTT] = 20.0  # a decision every second step
    return SimpleNamespace(
        obs=obs,
        episode=np.repeat(np.arange(n_episodes), length),
        step=np.tile(np.arange(length), n_episodes),
    )


def test_decisions_are_the_graphs_and_leave_out_each_episodes_last_step() -> None:
    s = _steps(3, 11, np.random.default_rng(0))
    d = ground.decisions(s)
    per = decision_steps(s.obs[:11, OBS_MIN_RTT])
    assert per.tolist() == [0, 2, 4, 6, 8, 10]
    assert d.tolist() == [e * 11 + k for e in range(3) for k in (0, 2, 4, 6, 8)]


def test_the_library_finds_a_windowed_feature_and_drops_repeats() -> None:
    rng = np.random.default_rng(3)
    s = _steps(40, 200, rng)
    n = s.episode.size
    v = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    v = np.concatenate([v, v[:, :1] * 2 + 1], axis=1)  # variable c repeats a
    names = ["a", "b", "c"]
    idx = ground.decisions(s)
    feats, x = ground.features(names, v, s, idx)
    assert x.shape == (idx.size, 3 * 8) and feats[:3] == names
    assert "mean_240ms(b)" in feats and "change_60ms(a)" in feats
    # A window stops at the episode's start: its first step has no change yet.
    first = np.flatnonzero(s.step[idx] == 0)
    assert np.isnan(x[first, feats.index("change_60ms(a)")]).all()
    assert np.array_equal(x[first, feats.index("mean_960ms(b)")], v[idx[first], 1])
    half = idx.size // 2
    lib = ground.Library.of(feats, 3, x[:half], x[half:])
    assert lib.same_as["c"] == "a" and "c" not in lib.names
    assert lib.variable[lib.names.index("mean_240ms(b)")] == "b"
    # A latent that is the 240 ms mean of b: the selection finds that feature.
    z = np.nan_to_num(x[:, [feats.index("mean_240ms(b)")]])
    m = ground.Moments.of(lib, z[:half], z[half:])
    (first_pick,) = ground.greedy(lib, m, list(range(len(lib.names))), 1)
    assert lib.names[first_pick] == "mean_240ms(b)"
    assert m.r2(ground._columns(lib, [first_pick])).min() > 0.999
    # On a subset of the rows, the moments are those of that subset alone.
    keep_f, keep_s = np.arange(half) % 2 == 0, np.arange(idx.size - half) % 3 == 0
    part = lib.rows(keep_f, keep_s)
    sub = ground.Moments.of(part, z[:half][keep_f], z[half:][keep_s])
    assert sub.r2(ground._columns(part, [first_pick])).min() > 0.999


def _campaign(d: Path, s: data.Steps) -> None:
    """Steps as a campaign with step tables stores them."""
    d.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_id": list(s.episode_ids),
                "status": ["ok"] * len(s.episode_ids),
                "spec_json": [json.dumps({"path": {"bw": i}}) for i in range(len(s.episode_ids))],
                "controller": ["agent:x"] * len(s.episode_ids),
            }
        ),
        d / "episodes.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_id": [s.episode_ids[e] for e in s.episode],
                "step": s.step,
                "obs": pa.FixedSizeListArray.from_arrays(pa.array(s.obs.ravel()), OBS_DIM),
                "action": s.action,
                "reward": s.reward,
                "cwnd": s.cwnd,
            }
            | {f"label_{k}": v for k, v in s.labels.items()}
        ),
        d / "steps.parquet",
    )


def test_a_controllers_latents_are_grounded_on_its_own_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ground, "FLEX_STEPS", 20)  # its value is not under test here
    torch.manual_seed(0)
    steps = _noisy(n_episodes=12)
    net1_spec = Net1Spec(
        width=32, epochs=1, batch_size=256, lcfs=EXPLICCIT_LCFS, output=Bins(bins=8)
    )
    net1 = fit.fit_net1(net1_spec, steps, seed=0)[0]
    spec = PPOSpec.model_validate(
        {
            "paths_from": "unused",
            "net1_from": "unused",
            "u_rule": 0.25,
            "hidden": 8,
            "inputs": "raw",
            "latents": 2,
            "encoder_hidden": 16,
            "action_max": 8.0,
        }
    )
    runs = []
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):  # b is a copy of a
        run = tmp_path / name
        run.mkdir()
        torch.manual_seed(seed)
        actor = actors.new_actor(actors.width(net1, spec), spec)
        torch.save(actor.state_dict(), run / "actor.pt")
        fit.save_net1(net1, net1_spec, run / "net1.pt")
        (run / "ppo_spec.json").write_text(spec.model_dump_json())
        for kind in ("fit", "score"):
            _campaign(run / "ground" / kind / "shards" / "0", steps)
        runs.append(run)
    ctl = ground.Controller.load(runs[0])
    z = ctl.encode(steps.obs[:5])
    with torch.no_grad():
        view = actors.front(net1, spec)(torch.from_numpy(steps.obs[:5]), None)[1]
        want = actors.new_actor(actors.width(net1, spec), spec)
        want.load_state_dict(torch.load(runs[0] / "actor.pt", weights_only=True))
        assert np.allclose(z, want.encode(view).numpy(), atol=1e-6) and z.shape == (5, 2)

    ground.probe(tmp_path / "probe", runs)
    result = json.loads((tmp_path / "probe" / "probe.json").read_text())
    assert set(result["runs"]) == {"a", "b", "c"}
    own = result["runs"]["a"]
    assert (
        own["k"] == 2 and own["fit"] > 100 and len(own["read_linear"]) == len(result["variables"])
    )
    assert np.isfinite(own["decision_linear"]) and np.isfinite(own["decision_quadratic"])
    assert "capacity_bps" in result["variables"] and "sender_cwnd" in result["variables"]
    # The copy's latents are the original's; another seed's are not.
    pair = {(p["from"], p["to"]): p["linear"] for p in result["seeds"]}
    assert pair["a", "b"] > 0.999 and pair["a", "c"] < pair["a", "b"]
    assert "action collapse" in (tmp_path / "probe" / "summary.md").read_text()

    ground.select(tmp_path / "select", runs)
    sel = json.loads((tmp_path / "select" / "select.json").read_text())
    assert sel["pooled"]["2"]["runs"] == 3 and len(sel["runs"]["a"]["path"]) >= 1
    assert sel["runs"]["a"]["path"] == sel["runs"]["b"]["path"]
    assert {t["from"] for t in sel["transfer"]} == {"a", "b", "c"}

    # A run without a narrow middle is refused.
    wide = tmp_path / "wide"
    wide.mkdir()
    (wide / "ppo_spec.json").write_text(spec.model_copy(update={"latents": None}).model_dump_json())
    with pytest.raises(SystemExit, match="not a discovery controller"):
        ground.Controller.load(wide)


def _plain_library(x: np.ndarray, fit: int) -> ground.Library:
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    n = x.shape[1]
    return ground.Library(
        names=tuple(f"f{i}" for i in range(n)),
        variable=tuple(f"v{i}" for i in range(n)),
        columns=tuple((i + 1,) for i in range(n)),
        same_as={},
        fit=design[:fit],
        score=design[fit:],
        base=(),
        base_fit=np.zeros((fit, 0)),
        base_score=np.zeros((len(x) - fit, 0)),
    )


def test_pruning_puts_a_feature_that_stood_in_for_a_mix_last() -> None:
    rng = np.random.default_rng(2)
    a, b, junk = rng.normal(size=(3, 6000))
    proxy = a + b + 0.5 * rng.normal(size=6000)  # the best single feature, not a concept
    lib = _plain_library(np.stack([proxy, a, b, junk], axis=1), 4000)
    z = np.stack([a + b, a + 0.8 * b], axis=1) + 0.05 * rng.normal(size=(6000, 2))
    m = ground.Moments.of(lib, z[:4000], z[4000:])
    forward = ground.greedy(lib, m, [0, 1, 2, 3], 3)
    assert forward[0] == 0
    assert set(ground.pruned(lib, m, forward)[:2]) == {1, 2}


def test_the_flexible_fit_sees_a_variable_the_latent_holds_in_another_form() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6000, 3))
    lib = _plain_library(x, 4000)
    z = np.minimum(x[:, 0], x[:, 1])[:, None] ** 2  # no linear fit follows this
    m = ground.Moments.of(lib, z[:4000], z[4000:])
    assert m.r2(ground._columns(lib, [0, 1, 2]))[0] < 0.5
    assert ground.flexible(lib.fit[:, 1:3], z[:4000], lib.score[:, 1:3], z[4000:]) > 0.95
    assert ground.flexible(lib.fit[:, 3:], z[:4000], lib.score[:, 3:], z[4000:]) < 0.05


def test_the_formula_search_rebuilds_a_composite_from_its_labels() -> None:
    rng = np.random.default_rng(4)
    n = 6000
    raw = {
        "capacity_bps": rng.uniform(5e6, 1e8, n),
        "base_rtt_ns": rng.uniform(1e7, 2e8, n),
        "queue_bytes": rng.uniform(0, 3e5, n),
        "busy_fraction": rng.uniform(0, 1, n),
        "active_flows": rng.integers(1, 6, n).astype(np.float64),
    }
    names = list(raw)
    # The library's own scale, as base_variables gives it.
    x = np.stack([np.log1p(raw[v] * ground._factor(v)) for v in names], axis=1)
    lib = ground.Library.of(names, len(names), x[:4000], x[4000:])
    planted = {
        # HPCC's normalized inflight, q / (C R) + U: a sum no product of powers writes.
        "busy_fraction + queue_bytes/(base_rtt_ns*capacity_bps)": raw["busy_fraction"]
        + 8 * raw["queue_bytes"] / (raw["capacity_bps"] * raw["base_rtt_ns"] * 1e-9),
        # C U, what the link sent: a product with a fraction, which log(1 + U) does not give.
        "busy_fraction*capacity_bps": raw["capacity_bps"] * raw["busy_fraction"],
    }
    for name, truth in planted.items():
        held = np.tanh(2 * (truth - truth.mean()) / truth.std())  # one latent, squashed
        z = (held + 0.1 * rng.normal(size=n))[:, None]
        wider, made = ground.formulas(lib, (z[:4000], z[4000:]), names)
        assert made[0] == name
        m = ground.Moments.of(wider, z[:4000], z[4000:])
        first = ground.greedy(wider, m, list(range(len(wider.names))), 1)[0]
        ranks = ground._places(np.stack([wider.score[:, wider.columns[first][0]], truth[4000:]], 1))
        assert abs(np.mean(ranks[:, 0] * ranks[:, 1])) > 0.999
