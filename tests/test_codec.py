"""The codec's positive control: on data made from a known number of
hidden factors, the score must level off at that number of latents."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_ground_probe_scores_linear_and_quadratic_relations() -> None:
    spec = importlib.util.spec_from_file_location("ground", _PATH.with_name("ground.py"))
    assert spec is not None and spec.loader is not None
    ground = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ground)
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
