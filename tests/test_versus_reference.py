"""experiments/versus_reference.py: each width against its teacher's reference."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from .test_score import shard, spec_json


def _load() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments" / "versus_reference.py"
    spec = importlib.util.spec_from_file_location("versus_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vr = _load()


def _run(root: Path, name: str, alone: float, flow: float, cross: float) -> Path:
    cubic = [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]
    eps = [(f"{name}{i}", "P", spec_json(i), "ok", [alone]) for i in range(4)]
    eps += [(f"{name}x{i}", "P", spec_json(i, cross=cubic), "ok", [flow]) for i in range(4)]
    shard(root / name / "ground" / "score", "00000-of-00001", eps, cross=cross)
    return root / name


def test_a_width_is_set_against_its_teachers_reference_by_link(tmp_path: Path) -> None:
    runs = [
        _run(tmp_path, "ref-cubic-s1", 0.5, 0.5, 0.5),
        _run(tmp_path, "ref-cubic-s2", 0.5, 0.5, 0.5),
        _run(tmp_path, "ctl-cubic-k7-s1", 0.5, 0.25, 0.75),
        _run(tmp_path, "ctl-cubic-k7-s2", 0.5, 0.25, 0.75),
    ]
    vr.main(tmp_path / "out", runs)
    result = json.loads((tmp_path / "out" / "versus_reference.json").read_text())
    k7 = result["groups"]["ctl-cubic-k7"]
    assert k7["alone"]["score_minus_reference"]["mean"] == 0.0
    assert k7["alone"]["score_minus_reference"]["verdict"] == "comparable"
    # log(0.25) - log(0.5) on every link, below the margin.
    flow = k7["competing"]["score_minus_reference"]
    assert abs(flow["mean"] + 0.693) < 1e-3 and flow["verdict"] == "not comparable"
    # 0.25 / (0.25 + 0.75) against 0.5 / (0.5 + 0.5).
    assert abs(k7["competing"]["share"] - 0.25) < 1e-6
    assert abs(k7["competing"]["share_minus_reference"]["mean"] + 0.25) < 1e-6
    assert "score_minus_reference" not in result["groups"]["ref-cubic"]["alone"]
    assert result["seeds"] == {"ref-cubic": 2, "ctl-cubic-k7": 2}
    assert "| ctl-cubic-k7 | 2 |" in (tmp_path / "out" / "versus-reference.md").read_text()
