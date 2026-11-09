"""experiments/probing.py: how a window responds where the capacity is unseen."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "probing.py"
_spec = importlib.util.spec_from_file_location("probing", _PATH)
assert _spec is not None and _spec.loader is not None
probing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probing)


def test_slopes_recover_both_dependences() -> None:
    rng = np.random.default_rng(0)
    idle, spread = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    u = 0.25 + 0.8 * idle - 0.3 * spread
    a, b = probing.slopes(u, idle, spread)
    assert a == pytest.approx(0.8) and b == pytest.approx(-0.3)


def test_a_constant_window_has_no_slope() -> None:
    rng = np.random.default_rng(1)
    idle, spread = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
    a, b = probing.slopes(np.full(100, 0.25), idle, spread)
    assert abs(a) < 1e-9 and abs(b) < 1e-9
