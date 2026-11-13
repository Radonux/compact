"""The codec's positive control: on data made from a known number of
hidden factors, the score must level off at that number of latents."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "codec.py"
_spec = importlib.util.spec_from_file_location("codec", _PATH)
assert _spec is not None and _spec.loader is not None
codec = importlib.util.module_from_spec(_spec)
sys.modules["codec"] = codec
_spec.loader.exec_module(codec)

FACTORS = 3


def _synthetic(cfg: codec.Config, episodes: int = 600, length: int = 40) -> codec.Data:
    """Each episode has FACTORS hidden constants s. A row's four targets
    are a full-rank map of tanh(s), so the future is decided by s alone.
    The encoder sees a noisy nonlinear mix of s in 20 columns."""
    rng = np.random.default_rng(0)
    n = episodes * length
    s = np.repeat(rng.normal(size=(episodes, FACTORS)), length, axis=0)
    lc = rng.normal(scale=0.3, size=n).cumsum()
    mix = rng.normal(size=(FACTORS, 20))
    x = np.tanh(s @ mix) + 0.05 * rng.normal(size=(n, 20))
    out = rng.normal(size=(FACTORS, 4))
    tgt = np.tanh(s) @ out
    episode = np.repeat(np.arange(episodes), length)
    reach = max(cfg.horizons)
    rows = np.arange(n - reach)
    rows = rows[episode[rows + reach] == episode[rows]]
    split = np.where(episode[rows] % 5 == 0, "val", "train")
    return codec.Data(
        x=torch.from_numpy(x.astype(np.float32)),
        tgt=torch.from_numpy(((tgt - tgt.mean(0)) / tgt.std(0)).astype(np.float32)),
        lc=torch.from_numpy(lc.astype(np.float32)),
        ctl=torch.from_numpy(episode % 2),
        a_scale=torch.tensor([[[0.0] * len(cfg.horizons), [1.0] * len(cfg.horizons)]] * 2),
        anchors={
            "train": torch.from_numpy(rows[split == "train"]),
            "val": torch.from_numpy(rows[split == "val"]),
            "test": torch.from_numpy(rows[:0]),
        },
    )


def test_the_score_levels_off_at_the_number_of_hidden_factors() -> None:
    cfg = codec.Config(
        k=6, width=64, depth=2, dec_width=64, dec_depth=1, horizons=(1, 2), epochs=30,
        batch=512, lr=3e-3, warmup_steps=50, patience=30,
    )  # fmt: skip
    d = _synthetic(cfg)
    model, _ = codec.fit(d, cfg, log=lambda _: None)
    score = [
        float(codec.r2(model, d, d.anchors["val"], cfg, keep).mean()) for keep in range(cfg.k + 1)
    ]
    assert score[0] < 0.05  # nothing is predictable without a latent
    assert score[FACTORS] > 0.9
    assert score[FACTORS - 1] < score[FACTORS] - 0.05  # one latent short costs
    assert score[cfg.k] - score[FACTORS] < 0.03  # more latents add nothing


def test_a_horizon_sees_only_the_actions_up_to_it() -> None:
    cfg = codec.Config(k=2, width=16, depth=1, dec_width=16, dec_depth=1, horizons=(1, 2, 4))
    model = codec.Codec(5, 3, cfg).eval()
    z = torch.randn(4, 2)
    a = torch.randn(4, 3)
    b = a.clone()
    b[:, 2] += 1.0  # change only the last horizon's action
    p, q = model.decode(z, a), model.decode(z, b)
    assert torch.equal(p[:, :2], q[:, :2])
    assert not torch.equal(p[:, 2], q[:, 2])


def _load_ground() -> Any:
    spec = importlib.util.spec_from_file_location("ground", _PATH.with_name("ground.py"))
    assert spec is not None and spec.loader is not None
    ground = importlib.util.module_from_spec(spec)
    sys.modules["ground"] = ground
    spec.loader.exec_module(ground)
    return ground


def test_ground_probe_scores_linear_and_quadratic_relations() -> None:
    ground = _load_ground()
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
    ground = sys.modules.get("ground") or _load_ground()
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
    )
    # Two latents: mixes of features 1 and 4, where 4 also acts by being undefined.
    z = np.stack([2 * x[:, 1] + cols[5] + 3 * gap, x[:, 1] - cols[5]], axis=1)
    z += 0.01 * rng.normal(size=z.shape)
    m = ground.Moments.of(lib, z[:4000], z[4000:])
    order = ground.greedy(lib, m, list(range(6)), 3)
    assert set(order[:2]) == {1, 4}
    assert m.r2(ground._columns(lib, order[:2])).min() > 0.999
    assert m.r2(ground._columns(lib, order[:1])).min() < 0.9
    sel = ground._selection(lib, m)
    assert sel["library"] > 0.999 and set(sel["without"]) >= {"v0", "v2"}
    assert sel["without"]["v0"][-1]["mean"] < 0.9  # no substitute for feature 1


def test_action_check_scores_a_controller_only_on_what_its_inputs_hold() -> None:
    """Controller 0 acts on variable a and controller 1 on variable b.
    Inputs that hold only a must predict the first and not the second."""
    if "ground" not in sys.modules:
        _load_ground()
    spec = importlib.util.spec_from_file_location("actions", _PATH.with_name("actions.py"))
    assert spec is not None and spec.loader is not None
    actions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actions)

    rng = np.random.default_rng(0)

    def rows(n: int) -> tuple[Any, Any, Any]:
        v = rng.normal(size=(n, 2))
        ctl = rng.integers(0, 2, size=n)
        move = np.tanh(np.where(ctl == 0, v[:, 0], v[:, 1]))
        y = np.stack([move * h for h in actions.HORIZONS], axis=1)
        return v, y + 0.05 * rng.normal(size=y.shape), ctl

    (v_fit, y_fit, c_fit), (v, y, c) = rows(6000), rows(3000)
    group = np.arange(6000) // 20

    def score(cols: list[int]) -> Any:
        return actions.fit_scores(
            (v_fit[:, cols], v[:, cols]), (y_fit, y), (c_fit, c), group=group, n_ctl=2, seed=0
        )

    only_a, both, neither = score([0]), score([0, 1]), score([])
    assert only_a[0].min() > 0.9 and only_a[1].max() < 0.1
    assert both.min() > 0.9
    assert abs(neither).max() < 0.05


def test_a_student_of_the_actions_needs_what_its_controllers_act_on() -> None:
    """Two controllers move their windows by different hidden factors.
    Two latents predict both; one latent cannot."""
    base = codec.Config(
        targets="action", nested=False, width=64, depth=2, dec_width=64, dec_depth=1
    )
    base = base.model_copy(update={"horizons": (1, 2), "epochs": 12, "batch": 256, "lr": 3e-3})
    rng = np.random.default_rng(0)
    episodes, length = 600, 12
    n = episodes * length
    s = np.repeat(rng.normal(size=(episodes, 2)), length, axis=0)
    episode = np.repeat(np.arange(episodes), length)
    ctl = episode % 2
    step = s[np.arange(n), ctl]  # controller c moves by factor c each row
    lc = np.concatenate([[0.0], step[:-1]]).cumsum()
    x = np.tanh(s @ rng.normal(size=(2, 10))) + 0.05 * rng.normal(size=(n, 10))
    rows = np.arange(n - 2)
    rows = rows[episode[rows + 2] == episode[rows]]
    val = (episode[rows] // 2) % 5 == 0
    change = lc[rows[:, None] + np.array([1, 2])] - lc[rows][:, None]
    scale = [[change[ctl[rows] == c].mean(0), change[ctl[rows] == c].std(0)] for c in (0, 1)]
    d = codec.Data(
        x=torch.from_numpy(x.astype(np.float32)),
        tgt=torch.zeros(n, 1),
        lc=torch.from_numpy(lc.astype(np.float32)),
        ctl=torch.from_numpy(ctl),
        a_scale=torch.from_numpy(np.array(scale, np.float32)),
        anchors={
            "train": torch.from_numpy(rows[~val]),
            "val": torch.from_numpy(rows[val]),
            "test": torch.from_numpy(rows[:0]),
        },
    )
    score = {}
    for k in (1, 2):
        cfg = base.model_copy(update={"k": k})
        model, _ = codec.fit(d, cfg, log=lambda _: None)
        score[k] = float(codec.r2(model, d, d.anchors["val"], cfg, k).mean())
    assert score[2] > 0.9
    assert score[1] < 0.7


def test_one_latent_has_no_other_to_correlate_with() -> None:
    cfg = codec.Config(k=1, nested=False, width=16, depth=1, dec_width=16, dec_depth=1)
    cfg = cfg.model_copy(update={"horizons": (1, 2, 4)})
    d = _synthetic(cfg, episodes=20)
    stats = codec._latents(codec.Codec.of(d, cfg), d, d.anchors["val"], cfg)
    assert stats["max_abs_corr"] == [0.0]
