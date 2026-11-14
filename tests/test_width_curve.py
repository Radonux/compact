"""experiments/width_curve.py: the width curve and its knee."""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments" / "width_curve.py"
    spec = importlib.util.spec_from_file_location("width_curve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wc = _load()


def test_the_knee_is_the_smallest_width_within_one_standard_error_of_the_best() -> None:
    scores = {0: [9.0, 9.0], 1: [-0.5, -0.3], 2: [-0.12, -0.08], 4: [-0.05, -0.01], 8: [0.0, 0.0]}
    k = wc.knee(scores)
    # Pooled variance (0.02 + 0.0008 + 0.0008 + 0) / 4 over 2 seeds.
    assert k["standard_error"] == pytest.approx(math.sqrt(0.0054 / 2))
    assert k["best"] == 8 and k["knee"] == 4  # width 2 is 0.1 below, the error is 0.052
    assert 0 not in k["means"]  # the reference takes no part
    assert wc.knee({1: [0.0, 0.0], 2: [0.0, 0.0]})["knee"] == 1
    assert wc.knee({1: [0.0], 2: [0.1, 0.2]})["knee"] is None


def _run(root: Path, name: str, history: list[dict[str, object]]) -> Path:
    run = root / name
    run.mkdir()
    (run / "ppo_spec.json").write_text(json.dumps({"noise_sigma": 0.2}))
    (run / "history.json").write_text(json.dumps(history))
    return run


def _eval(score: float, pairs: int = 126) -> dict[str, object]:
    return {"minus_rule": score, "pairs": pairs}


def test_a_run_counts_only_when_the_policy_it_left_is_scored(tmp_path: Path) -> None:
    done = [
        {"iteration": -1, "eval": _eval(-0.2)},
        {"iteration": 0, "learner": {"kl_to_start": 0.1}},
        {"iteration": 1, "learner": {"kl_to_start": 0.5, "u_saturated": 0.0}, "eval": _eval(0.1)},
    ]
    runs = [
        _run(tmp_path, "ctl-cubic-k2-s1", done),
        _run(tmp_path, "ctl-cubic-k2-s2", done),
        _run(tmp_path, "ctl-cubic-k1-s1", done[:2]),  # ended without a score
        _run(tmp_path, "ctl-cubic-k1-s2", [done[0], {"iteration": 0, "eval": _eval(0.0, 0)}]),
        tmp_path / "ctl-cubic-k4-s1",  # never ran
    ]
    probe = runs[0] / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text(
        json.dumps({"runs": {"ctl-cubic-k2-s1": {"decision_linear": 0.9}}})
    )
    wc.main(tmp_path / "out", runs)
    result = json.loads((tmp_path / "out" / "width_curve.json").read_text())
    assert sorted(result["missing"]) == ["ctl-cubic-k1-s1", "ctl-cubic-k1-s2", "ctl-cubic-k4-s1"]
    first, second = result["runs"]
    assert (first["start"], first["end"], first["iterations"]) == (-0.2, 0.1, 2)
    assert first["moved"] == pytest.approx(0.2)  # sqrt(2 * 0.5) sigma
    assert first["decision_linear"] == 0.9 and math.isnan(second["decision_linear"])
    assert result["curves"]["ctl-cubic"]["knee"] == 2
    assert "Knee: width 2" in (tmp_path / "out" / "width-curve.md").read_text()
