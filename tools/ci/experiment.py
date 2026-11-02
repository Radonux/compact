"""Experiments in GitHub Actions (.github/workflows/experiment.yml).

An experiment is one JSON file under experiments/, or the same JSON given
inline to the workflow (it then needs a "name"), parsed here once. The
workflow runs it in four steps: plan (expand the cells, run `prepare` once),
one job per cell (`run`; at most 256 cells, 20 at a time, 6 h each), collect
(merge every cell's output, run `collect`, pack the result), and publish
(the result as a GitHub release, the long-term record of every run).

    {
      "name": "smoke",              # the release tag's NAME; default: the file's stem
      "title": "a readable release title; default: NAME",
      "description": "one line, for the release notes",
      "uml": true,                  # cells get a built UML tree in $UML
      "train": false,               # install the trainer's extra (PyTorch)
      "inputs": {"data": "release:TAG", "net1": "run:RUN_ID/result"},
      "prepare": "bash, once, before the cells; writes under $OUT",
      "shards": 20,                 # or "matrix": {"seed": [0, 1], "lr": [0.001, 0.0003]}
      "run": "bash, once per cell",
      "collect": "bash, once, after every cell succeeded; reads and writes $OUT",
      "timeout_minutes": 360
    }

Every command runs under `bash -euo pipefail` from the repository root, with
OUT (the shared output tree: what prepare wrote, then every cell's output),
IN (one directory per input), UML (when "uml" is set), WORKERS (cores),
CELL and CELLS (index and count), and SHARD and SHARDS or each matrix axis
in upper case. A cell writes only under paths no other cell writes (a
campaign shard writes shards/K-of-N), so the outputs merge by copying.

Inputs: "release:TAG" is every asset of a release of this repository;
"run:RUN_ID/NAME" is an artifact of an earlier workflow run. A packed
result (NAME.tar.zst.part-*) is unpacked into IN/<input>.

Usage: python tools/ci/experiment.py plan|prepare|fetch|run|collect|pack FILE_OR_JSON [CELL]
"""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO = Path(__file__).resolve().parents[2]
MAX_CELLS = 256  # GitHub's limit on the jobs one matrix generates
PART_BYTES = "1900M"  # release assets must be under 2 GiB each
RESERVED = {"out", "in", "uml", "workers", "cell", "cells", "shard", "shards"}

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Ref = Annotated[str, Field(pattern=r"^(release:[A-Za-z0-9._/-]+|run:[0-9]+/[A-Za-z0-9._-]+)$")]
Scalar = str | int | float | bool


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Slug | None = None
    title: Annotated[str, Field(max_length=120, pattern=r"^[^\n]*$")] = ""
    description: str = ""
    uml: bool = False
    train: bool = False
    inputs: dict[Name, Ref] = Field(default_factory=dict[Name, Ref])
    prepare: str = ""
    shards: Annotated[int, Field(ge=1, le=MAX_CELLS)] | None = None
    matrix: dict[Name, Annotated[list[Scalar], Field(min_length=1)]] | None = None
    run: Annotated[str, Field(min_length=1)]
    collect: str = ""
    timeout_minutes: Annotated[int, Field(ge=1, le=360)] = 360

    @model_validator(mode="after")
    def _cells(self) -> Self:
        if (self.shards is None) == (self.matrix is None):
            raise ValueError("give exactly one of shards and matrix")
        if self.matrix is not None and (clash := RESERVED & set(self.matrix)):
            raise ValueError(f"matrix axes {sorted(clash)} are reserved")
        if (n := len(self.cells())) > MAX_CELLS:
            raise ValueError(f"{n} cells; a run holds at most {MAX_CELLS}")
        return self

    def cells(self) -> list[dict[str, Scalar]]:
        """Each cell's variables: SHARD and SHARDS, or one value per matrix axis."""
        if self.shards is not None:
            return [{"shard": k, "shards": self.shards} for k in range(self.shards)]
        assert self.matrix is not None
        axes = list(self.matrix)
        return [dict(zip(axes, v, strict=True)) for v in itertools.product(*self.matrix.values())]


def load(source: str) -> tuple[str, Experiment]:
    """The experiment and its name, from a file or from inline JSON."""
    if source.lstrip().startswith("{"):
        exp = Experiment.model_validate_json(source)
        if exp.name is None:
            raise SystemExit("an inline experiment needs a name")
        return exp.name, exp
    exp = Experiment.model_validate_json(Path(source).read_text())
    return exp.name or Path(source).stem, exp


def label(cell: dict[str, Scalar]) -> str:
    if "shards" in cell:
        return f"shard {cell['shard']}/{cell['shards']}"
    return " ".join(f"{k}={v}" for k, v in cell.items())


def environment(exp: Experiment, cell: int | None = None) -> dict[str, str]:
    env = os.environ | {
        "OUT": str(REPO / "out"),
        "IN": str(REPO / "in"),
        "WORKERS": str(os.cpu_count() or 1),
        "CELLS": str(len(exp.cells())),
    }
    if exp.uml:
        env["UML"] = str(REPO / "build/uml")
    if cell is not None:
        env |= {"CELL": str(cell)} | {k.upper(): str(v) for k, v in exp.cells()[cell].items()}
    return env


def bash(command: str, env: dict[str, str]) -> None:
    if command:
        (REPO / "out").mkdir(exist_ok=True)
        subprocess.run(["bash", "-euo", "pipefail", "-c", command], cwd=REPO, env=env, check=True)


def plan(source: str, name: str, exp: Experiment) -> None:
    """Print the plan as GITHUB_OUTPUT lines, and record what ran in OUT."""
    cells = [{"index": i, "label": label(c)} for i, c in enumerate(exp.cells())]
    flags = {
        "uml": exp.uml,
        "train": exp.train,
        "inputs": bool(exp.inputs),
        "prepare": bool(exp.prepare),
    }
    print(f"name={name}")
    print(f"title={exp.title or name}")
    print("matrix=" + json.dumps({"cell": cells}, separators=(",", ":")))
    print(f"timeout={exp.timeout_minutes}")
    for k, v in flags.items():
        print(f"{k}={str(v).lower()}")
    out = REPO / "out"
    out.mkdir(exist_ok=True)
    server, repo, run_id = (
        os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    record = {
        "file": "inline" if source.lstrip().startswith("{") else source,
        "experiment": exp.model_dump(mode="json"),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "run": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
    }
    (out / "experiment.json").write_text(json.dumps(record, indent=1) + "\n")


def unpack(src: Path, dest: Path) -> None:
    """Move src's files into dest, joining and unpacking any packed result."""
    dest.mkdir(parents=True, exist_ok=True)
    parts = sorted(src.glob("*.tar.zst.part-*"))
    if parts:
        script = 'cat "$@" | zstd -dcq | tar -x -C "$DEST"'
        subprocess.run(
            ["bash", "-c", script, "_", *map(str, parts)],
            env={**os.environ, "DEST": str(dest)},
            check=True,
        )
    for f in src.iterdir():
        if f not in parts:
            f.rename(dest / f.name)


def fetch(exp: Experiment) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    for name, ref in exp.inputs.items():
        tmp = REPO / "in" / f".{name}"
        tmp.mkdir(parents=True, exist_ok=True)
        kind, _, what = ref.partition(":")
        if kind == "release":
            cmd = ["gh", "release", "download", what, "--dir", str(tmp), "--repo", repo]
        else:
            run_id, _, artifact = what.partition("/")
            cmd = [
                "gh",
                "run",
                "download",
                run_id,
                "--name",
                artifact,
                "--dir",
                str(tmp),
                "--repo",
                repo,
            ]
        subprocess.run(cmd, check=True)
        unpack(tmp, REPO / "in" / name)
        tmp.rmdir()


def pack(name: str) -> None:
    """dist/: OUT as NAME.tar.zst in parts under 2 GiB, and the release notes."""
    dist = REPO / "dist"
    dist.mkdir(exist_ok=True)
    split = 'split -b "$PART" -d -a 3 - "dist/$NAME.tar.zst.part-"'
    script = f"tar -C out -cf - . | zstd -T0 -q -10 | {split}"
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=REPO,
        env={**os.environ, "NAME": name, "PART": PART_BYTES},
        check=True,
    )
    record = json.loads((REPO / "out/experiment.json").read_text())
    summary = REPO / "out/summary.md"
    notes = [
        record["experiment"]["description"] or name,
        "",
        f"Experiment `{record['file']}` at commit {record['commit']}, run {record['run']}.",
        f"Unpack: `cat {name}.tar.zst.part-* | zstd -dc | tar -x`.",
        "",
        "```json",
        json.dumps(record["experiment"], indent=1),
        "```",
    ]
    if summary.exists():
        notes += ["", summary.read_text()[:100_000]]
    (dist / "notes.md").write_text("\n".join(notes) + "\n")


def main(argv: list[str]) -> None:
    verb, source = argv[0], argv[1]
    name, exp = load(source)
    match verb:
        case "plan":
            plan(source, name, exp)
        case "prepare":
            bash(exp.prepare, environment(exp))
        case "fetch":
            fetch(exp)
        case "run":
            bash(exp.run, environment(exp, int(argv[2])))
        case "collect":
            bash(exp.collect, environment(exp))
        case "pack":
            pack(name)
        case _:
            raise SystemExit(f"unknown verb {verb}")


if __name__ == "__main__":
    main(sys.argv[1:])
