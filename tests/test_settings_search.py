"""experiments/settings_search.py: the ranking of a search's points."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments" / "settings_search.py"
    spec = importlib.util.spec_from_file_location("settings_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ss = _load()


def _run(root: Path, name: str, alone: float, flow: float, scored: bool = True) -> Path:
    run = root / name
    run.mkdir()
    ev = {
        "pairs": 2,
        "minus_rule": (alone + flow) / 2,
        "minus_rule_by_condition": {"loss 3%": alone, "cubic from 0 s": flow},
    }
    learner = {"clip_fraction": 0.2, "approx_kl": 0.01}
    hist = [{"iteration": 0, "learner": learner}, {"iteration": 1, "learner": learner}]
    if scored:
        hist[-1]["eval"] = ev
    (run / "history.json").write_text(json.dumps(hist))
    return run


def test_points_rank_by_mean_score_and_split_by_kind_of_condition(tmp_path: Path) -> None:
    runs = [
        _run(tmp_path, "ctl-cubic-k64-s1-a", -0.4, -0.2),
        _run(tmp_path, "ctl-cubic-k64-s2-a", -0.2, -0.4),
        _run(tmp_path, "ctl-cubic-k64-s1-b", -0.1, -0.1),
        _run(tmp_path, "ctl-cubic-k64-s2-b", -0.1, -0.3),
    ]
    ss.main(tmp_path / "out", runs)
    points = json.loads((tmp_path / "out" / "settings_search.json").read_text())["points"]
    a, b = points["ctl-cubic-k64"][1], points["ctl-cubic-k64"][0]
    assert (b["tag"], a["tag"]) == ("b", "a")
    assert a["score"] == pytest.approx(-0.3) and a["spread"] == pytest.approx(0.0)
    assert b["alone"] == pytest.approx(-0.1) and b["competing"] == pytest.approx(-0.2)
    assert "| b | 2 of 2 |" in (tmp_path / "out" / "settings-search.md").read_text()


def test_a_point_with_a_run_that_left_no_score_has_no_rank(tmp_path: Path) -> None:
    runs = [
        _run(tmp_path, "g-s1-a", 0.5, 0.5),
        _run(tmp_path, "g-s2-a", 0.5, 0.5, scored=False),
        _run(tmp_path, "g-s1-b", -1.0, -1.0),
        _run(tmp_path, "g-s2-b", -1.0, -1.0),
    ]
    ss.main(tmp_path / "out", runs)
    points = json.loads((tmp_path / "out" / "settings_search.json").read_text())["points"]["g"]
    assert [p["tag"] for p in points] == ["b", "a"]
    assert points[1]["score"] is None and points[1]["seeds"] == 1
