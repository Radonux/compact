"""experiments/rebase_bbr3.py: evaluations against the rule set against BBRv3."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

from expliccit_env.train import data

from .test_score import shard, spec_json


def _load() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments" / "rebase_bbr3.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rebase_bbr3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = _load()


def _run(root: Path, name: str, alone: float, flow: float) -> Path:
    cubic = [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]
    eps = [(f"{name}{i}", "P", spec_json(i), "ok", [alone]) for i in range(4)]
    eps += [(f"{name}x{i}", "P", spec_json(i, cross=cubic), "ok", [flow]) for i in range(4)]
    shard(root / name / "ground" / "score", "00000-of-00001", eps)
    return root / name


def test_an_evaluation_against_the_rule_is_set_against_bbr3(tmp_path: Path) -> None:
    rule = _run(tmp_path, "rule-s1", 0.5, 0.5)
    bbr3 = _run(tmp_path, "stock-bbr3-s1", 0.5, 0.25)
    links = [data.link_key(spec_json(i)) for i in range(4)]
    cond = "cubic from 0 s"
    run = tmp_path / "ctl-cubic-k7-s1-x"
    run.mkdir()
    pairs = [[k, "clean", 0.0] for k in links]
    worse = [[k, "clean", -0.5] for k in links] + [[k, cond, 0.0] for k in links]
    history = [
        {"iteration": -1, "eval": {"minus_rule_by_pair": pairs + [[k, cond, 0.0] for k in links]}},
        {"iteration": 19, "eval": {"minus_rule_by_pair": worse}},
    ]
    (run / "history.json").write_text(json.dumps(history))
    rb.main(tmp_path / "out", rule, bbr3, [run])
    r = json.loads((tmp_path / "out" / "rebase_bbr3.json").read_text())[run.name]
    # The start equals the rule: alone level with BBRv3, with a flow log(0.5 / 0.25) above.
    assert r["start"]["alone"]["mean"] == 0.0
    assert math.isclose(r["start"]["competing"]["mean"], math.log(2), rel_tol=1e-4)
    assert r["best"]["iteration"] == -1 and r["best"]["trained_well"]
    assert r["last"]["alone"]["verdict"] == "below"
    assert "| ctl-cubic-k7-s1-x |" in (tmp_path / "out" / "rebase-bbr3.md").read_text()
