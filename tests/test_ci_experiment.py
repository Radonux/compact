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
    assert (parse(shards=3).parallel, parse(shards=3, parallel=5).parallel) == (20, 5)


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
        {"shards": 2, "parallel": 0},
        {"shards": 2, "parallel": 21},  # beyond the workflow's 20
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


def test_a_download_retries_into_a_clean_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    into, calls = tmp_path / "in", []

    def run(cmd: list[str], check: bool) -> subprocess.CompletedProcess[bytes]:
        calls.append(sorted(p.name for p in into.iterdir()))
        (into / "partial").write_text("x")
        return subprocess.CompletedProcess(cmd, 0 if len(calls) == 3 else 1)

    monkeypatch.setattr(experiment.subprocess, "run", run)
    monkeypatch.setattr(experiment.time, "sleep", lambda _: None)
    experiment.download(["gh", "release", "download", "x"], into)
    assert calls == [[], [], []]


@pytest.mark.skipif(shutil.which("zstd") is None, reason="needs zstd (GitHub's runners have it)")
def test_a_release_input_can_be_some_paths_of_its_packed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, dist, bin_ = tmp_path / "out", tmp_path / "dist", tmp_path / "bin"
    for d in ("keep/a", "keep/b", "drop"):
        (out / d).mkdir(parents=True)
        (out / d / "x.json").write_text(d * 50)
    dist.mkdir()
    bin_.mkdir()
    pack = 'tar -C "$OUT" -cf - . | zstd -q | split -b 64 -d -a 3 - "$DIST/r.tar.zst.part-"'
    env = os.environ | {"OUT": str(out), "DIST": str(dist)}
    subprocess.run(["bash", "-euo", "pipefail", "-c", pack], env=env, check=True)
    assert len(list(dist.iterdir())) > 1
    # gh, as the release view and download the command runs: the parts in dist.
    gh = bin_ / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        'if [ "$2" = view ]; then (cd "$DIST" && ls -r; echo NOTES.md); exit 0; fi\n'
        'while [ "$1" != --pattern ]; do shift; done; cat "$DIST/$2"\n'
    )
    gh.chmod(0o755)
    into = tmp_path / "in"
    monkeypatch.setenv("DIST", str(dist))
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    experiment.download(experiment.release_download("o/r", "exp-x-1:keep/a,keep/b/", into), into)
    found = sorted(str(p.relative_to(into)) for p in into.rglob("*.json"))
    assert found == ["keep/a/x.json", "keep/b/x.json"]
    assert experiment.release_download("o/r", "exp-x-1", into)[:4] == [
        "gh",
        "release",
        "download",
        "exp-x-1",
    ]
    assert parse(shards=1, inputs={"hit": "release:exp-x-1:keep/a,keep/b"}).inputs


def test_a_download_gives_up_after_four_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment.subprocess, "run", lambda cmd, check: subprocess.CompletedProcess(cmd, 1)
    )
    monkeypatch.setattr(experiment.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit, match="failed 4 times"):
        experiment.download(["gh", "x"], tmp_path / "in")
