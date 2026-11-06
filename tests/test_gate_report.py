"""The learned net2 gate's decision rule (experiments/net2_ppo_gate_report.py),
on synthetic per-link scores."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
_PATH = Path(__file__).resolve().parents[1] / "experiments" / "net2_ppo_gate_report.py"
_spec = importlib.util.spec_from_file_location("net2_ppo_gate_report", _PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["net2_ppo_gate_report"] = gate
_spec.loader.exec_module(gate)

LINKS = [f"l{i}" for i in range(12)]


def _scores(gain: dict[str, float], copies: dict[str, float]) -> dict[str, object]:
    """Every family with the rule at 0 and each policy at its candidate's
    gain on every link and condition (plus small per-link noise); new-20
    has a competition condition and a lone one."""
    rng = np.random.default_rng(1)

    def camp(conds: list[str], shift: dict[str, float]) -> dict[str, object]:
        ctl: dict[str, object] = {"rule-v5": {c: dict.fromkeys(LINKS, 0.0) for c in conds}}
        for cand, g in shift.items():
            for s in (1, 2):
                ctl[f"ppo-{cand}-s{s}"] = {
                    c: {k: g + 0.01 * rng.standard_normal() for k in LINKS} for c in conds
                }
        return ctl

    out = {f: camp(["clean"], gain) for f in gate.FAMILIES}
    out["new-20"] = camp(["cubic from 0 s", "loss 3%"], gain)
    out["cubic-first-60"] = camp(["cubic from 0 s"], gain)
    out[gate.COPIES] = camp(["itself from 0 s"], copies)
    return out


def test_the_gate_keeps_the_best_mean_and_passes_only_a_clear_win() -> None:
    cands, kept = gate.decide(_scores({"a": 0.1, "b": -0.05}, {"a": 0.0, "b": 0.0}), set())
    assert kept == "a" and cands["a"].passes() and not cands["b"].passes()
    assert cands["a"].five_family_mean == pytest.approx(0.1, abs=0.01)
    assert set(cands["a"].per_seed) == {"ppo-a-s1", "ppo-a-s2"}


def test_the_gate_fails_on_worse_copies_and_never_keeps_a_failed_candidate() -> None:
    cands, kept = gate.decide(_scores({"a": 0.1}, {"a": -0.2}), set())
    assert kept == "a" and not cands["a"].passes()  # copies worse beyond their interval
    _, kept = gate.decide(_scores({"a": 0.1, "b": 0.05}, {"a": 0.0, "b": 0.0}), {"ppo-a-s2"})
    assert kept == "b"


def test_a_gain_inside_the_noise_does_not_pass() -> None:
    cands, kept = gate.decide(_scores({"a": 0.002}, {"a": 0.0}), set())
    assert kept == "a" and not cands["a"].passes()
