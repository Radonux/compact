"""Grounding: relate a trained codec's latents to ground-truth variables.

Usage: python experiments/ground.py probe OUT --data DIR... --runs RUN_DIR...

probe is the first, linear look, before any formula search. Each RUN_DIR is
a codec run (experiments/codec.py: config.json and model.pt). For every
run, on anchors of the validation links, with fits made on anchors of the
training links:

  read    R^2 of each base variable predicted from the latents, by a
          linear fit and by a quadratic one (the latents, their squares and
          pairwise products): which variables the latents carry
  write   R^2 of each latent predicted linearly from all base variables:
          how much of a latent the vocabulary can express at all
  seeds   for each pair of runs of one width, the mean R^2 of one run's
          latents predicted from the other's, linear and quadratic: whether
          the seeds found the same subspace

The base variables are the label columns of the step data, each on a
logarithmic scale (signed log1p of megabits per second, milliseconds,
kilobytes or packets). A linear probe is a lower bound: the decoder is not
linear, so a latent may hold a variable in a form these fits miss.

Writes OUT/probe.json and OUT/summary.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from codec import Codec, Config, Data, _batch, build

from expliccit_env.train import data as steps_data

F64 = npt.NDArray[np.float64]

# Rows used: fits on FIT anchors of the training links, scores on SCORE
# anchors of the validation links. ponytail: a fixed subsample, all anchors
# if a fit ever needs them.
FIT, SCORE = 300_000, 200_000
_SCALE = {"_bps": 1e-6, "_ns": 1e-6, "_bytes": 1e-3}


def base_variables(s: steps_data.Steps) -> tuple[list[str], F64]:
    """Every label that is ever defined, on a logarithmic scale; NaN where
    it is undefined."""
    names = sorted(n for n, v in s.labels.items() if np.isfinite(v).any())
    cols = []
    for n in names:
        scale = next((v for suffix, v in _SCALE.items() if n.endswith(suffix)), 1.0)
        v = s.labels[n].astype(np.float64) * scale
        cols.append(np.sign(v) * np.log1p(np.abs(v)))
    b = np.stack(cols, axis=1)
    b[~np.isfinite(b)] = np.nan
    return names, b


def quadratic(z: F64) -> F64:
    """The columns of z, their squares and pairwise products."""
    i, j = np.triu_indices(z.shape[1])
    return np.concatenate([z, z[:, i] * z[:, j]], axis=1)


def r2(x_fit: F64, y_fit: F64, x: F64, y: F64) -> F64:
    """Held-out R^2 of a least-squares fit of each column of y on x, over
    the rows where that column is defined. NaN for a column that is
    undefined or constant on either side."""
    out = np.full(y.shape[1], np.nan)
    a_fit = np.concatenate([np.ones((x_fit.shape[0], 1)), x_fit], axis=1)
    a = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    for c in range(y.shape[1]):
        m_fit, m = np.isfinite(y_fit[:, c]), np.isfinite(y[:, c])
        if m_fit.sum() <= a.shape[1] or not m.any() or np.var(y[m, c]) == 0:
            continue
        w, *_ = np.linalg.lstsq(a_fit[m_fit], y_fit[m_fit, c], rcond=None)
        out[c] = 1 - np.mean((a[m] @ w - y[m, c]) ** 2) / np.var(y[m, c])
    return out


def filled(b_fit: F64, b: F64) -> tuple[F64, F64]:
    """Undefined entries set to the fit rows' mean, with one indicator
    column per variable that is ever undefined, so a linear fit can use
    'undefined' as a value."""
    mean = np.nan_to_num(np.nanmean(b_fit, axis=0))
    gaps = np.isnan(b_fit).any(axis=0) | np.isnan(b).any(axis=0)

    def fill(v: F64) -> F64:
        return np.concatenate([np.where(np.isnan(v), mean, v), np.isnan(v)[:, gaps]], axis=1)

    return fill(b_fit), fill(b)


def latents(run: Path, d: Data, idx: torch.Tensor) -> tuple[Config, F64]:
    cfg = Config.model_validate_json((run / "config.json").read_text())
    model = Codec(d.x.shape[1] * cfg.frames, d.tgt.shape[1], cfg)
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True)["state"])
    model.eval()
    with torch.no_grad():
        z = torch.cat([model.encode(_batch(d, i, cfg)[0]) for i in idx.split(65_536)])
    return cfg, z.numpy().astype(np.float64)


def probe(out: Path, dirs: list[Path], runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    first = Config.model_validate_json((runs[0] / "config.json").read_text())
    s = steps_data.load(
        dirs,
        seed=first.split_seed,
        val=first.val_fraction,
        test=first.test_fraction,
        split_by="link",
    )
    d, _ = build(s, first)
    names, b = base_variables(s)
    gen = torch.Generator().manual_seed(0)

    def sample(split: str, n: int) -> torch.Tensor:
        a = d.anchors[split]
        return a[torch.randperm(a.numel(), generator=gen)[:n]].sort().values

    fit, score = sample("train", FIT), sample("val", SCORE)
    b_fit, b_score = b[fit.numpy()], b[score.numpy()]
    bf_fit, bf_score = filled(b_fit, b_score)
    result: dict[str, Any] = {"variables": names, "fit": int(fit.numel()), "runs": {}}
    z_of: dict[str, tuple[int, F64, F64]] = {}
    for run in runs:
        cfg, z_fit = latents(run, d, fit)
        if (cfg.frames, cfg.stride, cfg.horizons, cfg.split_seed) != (
            first.frames,
            first.stride,
            first.horizons,
            first.split_seed,
        ):
            raise SystemExit(f"{run}: its anchors differ from those of {runs[0]}")
        _, z_score = latents(run, d, score)
        z_of[run.name] = (cfg.k, z_fit, z_score)
        result["runs"][run.name] = {
            "k": cfg.k,
            "read_linear": r2(z_fit, b_fit, z_score, b_score).tolist(),
            "read_quadratic": r2(quadratic(z_fit), b_fit, quadratic(z_score), b_score).tolist(),
            "write_linear": r2(bf_fit, z_fit, bf_score, z_score).tolist(),
        }
    result["seeds"] = [
        {
            "from": a,
            "to": c,
            "k": z_of[a][0],
            "linear": float(np.mean(r2(z_of[a][1], z_of[c][1], z_of[a][2], z_of[c][2]))),
            "quadratic": float(
                np.mean(r2(quadratic(z_of[a][1]), z_of[c][1], quadratic(z_of[a][2]), z_of[c][2]))
            ),
        }
        for a, c in itertools.permutations(z_of, 2)
        if z_of[a][0] == z_of[c][0]
    ]
    (out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.md").write_text(summary(result))


def summary(result: dict[str, Any]) -> str:
    """Per width, the mean over its runs. Latents of two runs do not
    correspond, so a latent's score is reduced within its run first."""
    runs = list(result["runs"].values())
    widths = sorted({r["k"] for r in runs})

    def cells(score: Any) -> str:
        """One table cell per width: the mean of score(run) over its runs."""
        out = []
        for k in widths:
            v = np.array([score(r) for r in runs if r["k"] == k], dtype=np.float64)
            out.append("n/a" if np.isnan(v).all() else f"{np.nanmean(v):.3f}")
        return " | ".join(out) + " |\n"

    head = " | ".join(f"{k} latents" for k in widths) + " |\n"
    rule = "|---" * (len(widths) + 1) + "|\n"
    lines = ["R^2 on validation links, mean over runs.\n\n"]
    for key, title in (("read_linear", "linear"), ("read_quadratic", "quadratic")):
        lines += [f"Base variable from the latents, {title}:\n\n| variable | " + head, rule]
        for i, name in enumerate(result["variables"]):
            lines.append(f"| {name} | " + cells(lambda r, i=i, key=key: r[key][i]))
        lines.append("\n")
    lines += ["Latents from all base variables, linear:\n\n| | " + head, rule]
    lines.append("| mean over latents | " + cells(lambda r: np.mean(r["write_linear"])))
    lines.append("| worst latent | " + cells(lambda r: np.min(r["write_linear"])))
    lines += ["\nOne run's latents from another's, mean over pairs:\n\n| | " + head, rule]
    for key in ("linear", "quadratic"):
        row = []
        for k in widths:
            v = [p[key] for p in result["seeds"] if p["k"] == k]
            row.append(f"{np.mean(v):.3f}" if v else "n/a")
        lines.append(f"| {key} | " + " | ".join(row) + " |\n")
    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("out", type=Path)
    pr.add_argument("--data", type=Path, nargs="+", required=True)
    pr.add_argument("--runs", type=Path, nargs="+", required=True)
    a = p.parse_args()
    probe(a.out, a.data, a.runs)


if __name__ == "__main__":
    main()
