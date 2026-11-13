"""Which concepts hold what known controllers act on: an offline check.

Usage:
  python experiments/actions.py OUT --data DIR... --runs RUN_DIR... --select SELECT_JSON

For each set of inputs, one small network predicts, from the inputs at a
step and the identity of the controller that logged it, how that
controller changes its window over the next 60, 240 and 960 ms. The same
network, training and rows serve every set, so the sets differ only in
what they tell it. Scores are R^2 on anchors of the validation links, per
controller and horizon, around that controller's own mean change.

The sets, each alone and each with the sender's own window added (a
controller knows its window; no concept has to state it):

  id        nothing but the controller's identity: the floor
  latents   the k latents of each codec run (RUN_DIR, experiments/codec.py)
  truth     the first k features of the ground-truth path that
            experiments/ground.py select found for the runs of width k
            pooled (SELECT_JSON, its select.json)
  lcfs      expliCCit's seven hand-picked concepts
  sender    the sender's smoothed RTT, delivery rate and loss rate
  message   the whole step message the codec's encoder reads: the ceiling
            for a predictor that sees one step

This is a proxy and supports no claim about control. The controllers have
memory and the predictor sees one step, and predicting logged actions is
not acting in the closed loop (docs/methodology.md, step 5).

Writes OUT/actions.json and OUT/summary.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from codec import _mlp
from ground import (
    EXPLICCIT_LCFS,
    Library,
    _latents_of,
    _setup,
    base_variables,
    library,
    sender_variables,
)

from expliccit_env.train import data as steps_data

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

# Horizons in rows of the data (60 ms each at the data's stride of 6).
HORIZONS = (1, 4, 16)
SEEDS = (0, 1, 2)
SENDER = ("sender_srtt", "sender_delivery_rate", "sender_loss_rate")
# ponytail: one fixed network and schedule for every set, a search per set
# if two sets ever score close enough for the choice to matter.
WIDTH, DEPTH, EPOCHS, BATCH, LR = 256, 2, 20, 2048, 1e-3


def fit_scores(
    x: tuple[F64, F64],
    y: tuple[F64, F64],
    ctl: tuple[I64, I64],
    *,
    group: I64,
    n_ctl: int,
    seed: int,
) -> F64:
    """R^2 [controller, horizon] on the score rows of a network fitted on
    the fit rows. Each pair is (fit rows, score rows); y is NaN where a
    horizon is undefined. A tenth of the fit rows' groups (episodes) is
    held out to choose the epoch. Targets are scaled per controller, so
    every controller weighs the same in the loss."""
    torch.manual_seed(seed)
    mean = np.stack([np.nanmean(y[0][ctl[0] == c], 0) for c in range(n_ctl)])
    std = np.stack([np.nanstd(y[0][ctl[0] == c], 0) for c in range(n_ctl)])
    std = np.where(std > 1e-9, std, np.nan)  # a controller that never moves its window

    def tensors(i: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.concatenate([np.eye(n_ctl)[ctl[i]], x[i]], axis=1)
        target = (y[i] - mean[ctl[i]]) / std[ctl[i]]
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(
            target.astype(np.float32)
        )

    x_fit, y_fit = tensors(0)
    x_score, y_score = tensors(1)
    held = torch.from_numpy(group % 10 == 0)
    model = _mlp(x_fit.shape[1], WIDTH, DEPTH, len(HORIZONS), 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        m = torch.isfinite(target)
        return ((pred - target.nan_to_num()) ** 2 * m).sum() / m.sum().clamp(min=1)

    rows = torch.nonzero(~held).flatten()
    best, state = float("inf"), model.state_dict()
    for _ in range(EPOCHS):
        model.train()
        for batch in rows[torch.randperm(rows.numel())].split(BATCH):
            opt.zero_grad()
            loss(model(x_fit[batch]), y_fit[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            now = float(loss(model(x_fit[held]), y_fit[held])) if held.any() else 0.0
        if now < best:
            best, state = now, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        err = ((model(x_score) - y_score) ** 2).numpy().astype(np.float64)
    target = y_score.numpy().astype(np.float64)
    out = np.full((n_ctl, len(HORIZONS)), np.nan)
    for c in range(n_ctl):
        own = ctl[1] == c
        if own.any():
            # The target is scaled by the fit rows; score against its
            # variance on the score rows themselves.
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c] = 1 - np.nanmean(err[own], 0) / np.nanvar(target[own], 0)
    out[~np.isfinite(out)] = np.nan
    return out


def controllers(dirs: list[Path], s: steps_data.Steps) -> tuple[list[str], I64]:
    """The controllers' names, and for each row the index of its own."""
    of: dict[str, str] = {}
    for shard in sorted({p.parent for d in dirs for p in d.rglob("steps.parquet")}):
        # pyarrow-stubs leave read_table partly untyped.
        eps = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
            shard / "episodes.parquet", columns=["episode_id", "controller"]
        )
        ids, own = eps["episode_id"].to_pylist(), eps["controller"].to_pylist()
        of.update((str(e), str(c)) for e, c in zip(ids, own, strict=True))
    names = sorted({of[e] for e in s.episode_ids})
    per_episode = np.array([names.index(of[e]) for e in s.episode_ids], dtype=np.int64)
    return names, per_episode[s.episode]


def columns(lib: Library, names: tuple[str, ...]) -> tuple[F64, F64]:
    """The library's columns for these features, on the fit and score rows.
    A feature the library left out as a repeat is read from its twin."""
    index = {n: i for i, n in enumerate(lib.names)}
    cols = [c for n in names for c in lib.columns[index[lib.same_as.get(n, n)]]]
    return lib.fit[:, cols], lib.score[:, cols]


def run(out: Path, dirs: list[Path], runs: list[Path], select: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s, d, first, fit, score = _setup(dirs, runs)
    i_fit, i_score = fit.numpy(), score.numpy()
    labels, b = base_variables(s)
    sender, m = sender_variables(s)
    lib = library(
        [*labels, *sender], np.concatenate([b, m], 1), s, i_fit, i_score, stride=first.stride
    )
    names, ctl = controllers(dirs, s)
    lc = d.lc.numpy().astype(np.float64)

    def change(idx: I64) -> F64:
        return np.clip(lc[idx[:, None] + np.array(HORIZONS)[None]] - lc[idx][:, None], -8, 8)

    y, c = (change(i_fit), change(i_score)), (ctl[i_fit], ctl[i_score])
    group = s.episode[i_fit]
    z_of = _latents_of(runs, d, first, fit, score)
    picked = json.loads(select.read_text())["pooled"]
    window = columns(lib, ("sender_cwnd",))
    none = (np.zeros((i_fit.size, 0)), np.zeros((i_score.size, 0)))

    # (set, width or None, the codec run or None, inputs, model seeds)
    sets: list[tuple[str, int | None, str | None, tuple[F64, F64], tuple[int, ...]]] = [
        ("id", None, None, none, SEEDS),
        ("lcfs", None, None, columns(lib, EXPLICCIT_LCFS), SEEDS),
        ("sender", None, None, columns(lib, SENDER), SEEDS),
        ("message", None, None, (d.x[fit].numpy(), d.x[score].numpy()), SEEDS),
    ]
    for k in sorted({k for k, _, _ in z_of.values()}):
        path = tuple(p["feature"] for p in picked[str(k)]["hidden_path"][:k])
        sets.append(("truth", k, None, columns(lib, path), SEEDS))
    sets += [("latents", k, name, (zf, zs), (0,)) for name, (k, zf, zs) in z_of.items()]

    rows: list[dict[str, Any]] = []
    for name, k, codec_run, x, seeds in sets:
        for with_window in (False, True) if name not in ("id", "message") else (False,):
            xs = (np.concatenate([x[0], window[0]], 1), np.concatenate([x[1], window[1]], 1))
            for seed in seeds:
                r2 = fit_scores(
                    xs if with_window else x, y, c, group=group, n_ctl=len(names), seed=seed
                )
                rows.append(
                    {
                        "set": name,
                        "k": k,
                        "codec": codec_run,
                        "window": with_window,
                        "inputs": int(x[0].shape[1]) + int(with_window),
                        "seed": seed,
                        "r2": r2.tolist(),
                    }
                )
                print(name, k, codec_run, with_window, seed, np.nanmean(r2, 0).round(3), flush=True)
    result = {
        "controllers": names,
        "horizons_ms": [h * first.stride * 10 for h in HORIZONS],
        "fit": int(i_fit.size),
        "score": int(i_score.size),
        "score_rows": np.bincount(c[1], minlength=len(names)).tolist(),
        "truth": {k: [p["feature"] for p in v["hidden_path"][: int(k)]] for k, v in picked.items()},
        "rows": rows,
    }
    (out / "actions.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.md").write_text(summary(result))


def summary(result: dict[str, Any]) -> str:
    """Per set: the mean over its fits (model seeds, or codec runs) of the
    R^2, averaged over controllers; and per controller at each horizon."""
    groups: dict[tuple[str, int | None, bool], list[F64]] = {}
    inputs: dict[tuple[str, int | None, bool], int] = {}
    for r in result["rows"]:
        key = (r["set"], r["k"], r["window"])
        groups.setdefault(key, []).append(np.array(r["r2"], dtype=np.float64))
        inputs[key] = r["inputs"]

    def label(key: tuple[str, int | None, bool]) -> str:
        name, k, window = key
        return name + (f" {k}" if k is not None else "") + (" + window" if window else "")

    ms = result["horizons_ms"]
    lines = [
        "R^2 of the predicted change of the window on validation links, per controller "
        "around its own mean change, then averaged over controllers. Mean and standard "
        "deviation over the fits of a set (three model seeds, or one per codec run). "
        "Columns counts the network's inputs besides the controller's identity: a "
        "feature that is sometimes undefined has a second column saying so. "
        "A proxy: no claim about control.\n\n",
        "| inputs | columns | fits | " + " | ".join(f"{h} ms" for h in ms) + " |\n",
        "|---|---|---|" + "---|" * len(ms) + "\n",
    ]
    for key, fits in groups.items():
        per_fit = np.stack([np.nanmean(f, axis=0) for f in fits])
        cells = " | ".join(
            f"{a:.3f} ± {b:.3f}" for a, b in zip(per_fit.mean(0), per_fit.std(0), strict=True)
        )
        lines.append(f"| {label(key)} | {inputs[key]} | {len(fits)} | {cells} |\n")
    for j, h in enumerate(ms):
        lines += [
            f"\nPer controller, {h} ms (mean over the fits of a set):\n\n",
            "| inputs | " + " | ".join(result["controllers"]) + " |\n",
            "|---|" + "---|" * len(result["controllers"]) + "\n",
        ]
        for key, fits in groups.items():
            v = np.nanmean(np.stack(fits)[:, :, j], axis=0)
            cells = " | ".join("n/a" if np.isnan(a) else f"{a:.3f}" for a in v)
            lines.append(f"| {label(key)} | {cells} |\n")
    lines.append("\nThe ground-truth features of `truth k`:\n\n")
    lines += [f"- {k}: {', '.join(v)}\n" for k, v in result["truth"].items()]
    rows = ", ".join(
        f"{n} {r}" for n, r in zip(result["controllers"], result["score_rows"], strict=True)
    )
    lines.append(f"\nValidation anchors per controller: {rows}.\n")
    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--select", type=Path, required=True)
    a = p.parse_args()
    run(a.out, a.data, a.runs, a.select)


if __name__ == "__main__":
    main()
