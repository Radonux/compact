"""The experiment file parser behind .github/workflows/experiment.yml."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("experiment", ROOT / "tools/ci/experiment.py")
assert _spec is not None and _spec.loader is not None
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def parse(**fields: object):
    return experiment.Experiment.model_validate_json(json.dumps({"run": "true"} | fields))


def test_cells_are_shards_or_the_product_of_the_matrix_axes() -> None:
    assert parse(shards=3).cells() == [{"shard": k, "shards": 3} for k in range(3)]
    m = parse(matrix={"seed": [0, 1], "lr": [0.1, 0.2, 0.3]})
    assert len(m.cells()) == 6
    assert m.cells()[4] == {"seed": 1, "lr": 0.2}
    env = experiment.environment(m, 4)
    assert (env["SEED"], env["LR"], env["CELL"], env["CELLS"]) == ("1", "0.2", "4", "6")
    assert experiment.label(parse(shards=3).cells()[1]) == "shard 1/3"


@pytest.mark.parametrize(
    "fields",
    [
        {},  # neither shards nor matrix
        {"shards": 2, "matrix": {"seed": [0]}},  # both
        {"shards": 257},  # beyond one run's 256 jobs
        {"matrix": {"a": list(range(16)), "b": list(range(17))}},  # 272 cells
        {"matrix": {"shard": [0, 1]}},  # reserved axis
        {"matrix": {"seed": []}},  # empty axis
        {"shards": 2, "inputs": {"data": "https://example.org/x"}},  # not a release or a run
        {"shards": 2, "timeout_minutes": 361},  # beyond GitHub's 6 h
        {"shards": 2, "unknown": 1},
    ],
)
def test_invalid_experiments_are_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse(**fields)


def test_the_repository_experiments_parse() -> None:
    files = sorted((ROOT / "experiments").glob("*.json"))
    assert files
    for f in files:
        assert experiment.load(str(f))[0] == f.stem


def test_an_inline_experiment_is_named_by_its_name_field() -> None:
    assert experiment.load('{"name": "adhoc-1", "shards": 1, "run": "true"}')[0] == "adhoc-1"
    with pytest.raises(SystemExit):
        experiment.load('{"shards": 1, "run": "true"}')


@pytest.mark.skipif(shutil.which("zstd") is None, reason="needs zstd (GitHub's runners have it)")
def test_unpack_joins_each_packed_result_and_moves_other_files(tmp_path: Path) -> None:
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.txt").write_text(name * 100)
        pack = 'tar -C "$DIR" -cf - . | zstd -q | split -b 64 -d -a 3 - "$NAME.tar.zst.part-"'
        env = os.environ | {"DIR": str(tmp_path / name), "NAME": name}
        subprocess.run(["bash", "-euo", "pipefail", "-c", pack], cwd=src, env=env, check=True)
    assert len(list(src.glob("a.tar.zst.part-*"))) > 1  # the test packs in several parts
    (src / "NOTES.md").write_text("notes")
    experiment.unpack(src, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["NOTES.md", "a.txt", "b.txt"]
    assert (dest / "a.txt").read_text() == "a" * 100
    assert not list(src.iterdir())
