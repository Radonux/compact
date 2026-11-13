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

Usage: python experiments/ground.py select OUT --data DIR... --runs RUN_DIR...

select names the few variables that span the latents. The library is every
base variable (the labels, and four sender measurements: window, smoothed
RTT, delivery rate, loss rate) under eight operators: its value now, its
mean, minimum and maximum over the last 240 ms and 960 ms, and its change
over 60 ms. Sums of logarithms are products and ratios, so a linear fit on
this library is a fit by products of powers. Features are added one at a
time, each the one that most raises the R^2 of a joint linear fit of all
latents on the training anchors, and every size is scored on the
validation anchors. Reported per run, and per width with the latents of
its runs pooled:

  path     the features in the order added, with the mean and the worst
           latent's R^2 at each size
  bounds   the R^2 with the whole library, and with expliCCit's seven
           hand-picked concepts alone
  hidden   the path and the bound when the sender's four measurements are
           taken out, so only ground truth is left. The latents are
           computed from the sender's message, so they can repeat what it
           measured; this path asks what they hold of the path's state.
  without  for each base variable among the first PICK features: the path
           when that variable is taken out of the library, which is the
           vocabulary sensitivity check. Taking out a variable that was
           never picked cannot change the path.
  transfer the R^2 of one run's latents on another run's first PICK
           features

Writes OUT/select.json and OUT/summary.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from codec import DELIVERY, LOST, SRTT, Codec, Config, Data, _batch, build

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


def _setup(
    dirs: list[Path], runs: list[Path]
) -> tuple[steps_data.Steps, Data, Config, torch.Tensor, torch.Tensor]:
    """The data, and the anchors fits are made and scored on."""
    first = Config.model_validate_json((runs[0] / "config.json").read_text())
    s = steps_data.load(
        dirs,
        seed=first.split_seed,
        val=first.val_fraction,
        test=first.test_fraction,
        split_by="link",
    )
    d, _ = build(s, first)
    gen = torch.Generator().manual_seed(0)

    def sample(split: str, n: int) -> torch.Tensor:
        a = d.anchors[split]
        return a[torch.randperm(a.numel(), generator=gen)[:n]].sort().values

    return s, d, first, sample("train", FIT), sample("val", SCORE)


def _latents_of(
    runs: list[Path], d: Data, first: Config, fit: torch.Tensor, score: torch.Tensor
) -> dict[str, tuple[int, F64, F64]]:
    """Run name -> (width, latents on the fit anchors, on the score anchors)."""
    out = {}
    for run in runs:
        cfg, z_fit = latents(run, d, fit)
        if (cfg.frames, cfg.stride, cfg.horizons, cfg.split_seed) != (
            first.frames,
            first.stride,
            first.horizons,
            first.split_seed,
        ):
            raise SystemExit(f"{run}: its anchors differ from those of {runs[0]}")
        out[run.name] = (cfg.k, z_fit, latents(run, d, score)[1])
    return out


def probe(out: Path, dirs: list[Path], runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s, d, first, fit, score = _setup(dirs, runs)
    names, b = base_variables(s)
    b_fit, b_score = b[fit.numpy()], b[score.numpy()]
    bf_fit, bf_score = filled(b_fit, b_score)
    result: dict[str, Any] = {"variables": names, "fit": int(fit.numel()), "runs": {}}
    z_of = _latents_of(runs, d, first, fit, score)
    for run, (k, z_fit, z_score) in z_of.items():
        result["runs"][run] = {
            "k": k,
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


# The library's windows, in rows of the data (one row is 60 ms at stride 6).
WINDOWS = (4, 16)
# Features in a reported path, and the leading ones the checks are made on.
STEPS, PICK = 16, 8
EXPLICCIT_LCFS = (
    "queue_delay_mean_ns",
    "queue_pkts",
    "capacity_bps",
    "free_buffer_pkts",
    "busy_fraction",
    "one_way_delay_ns",
    "receive_rate_bps",
)


def sender_variables(s: steps_data.Steps) -> tuple[list[str], F64]:
    """Four measurements the sender has, on the codec's own scales."""
    o = s.obs.astype(np.float64)
    cols = [
        np.log2(np.maximum(s.cwnd.astype(np.float64), 1)),
        np.log(np.maximum(o[:, SRTT], 1e-4)),
        np.log1p(100 * np.maximum(o[:, DELIVERY], 0)),
        np.log1p(100 * np.maximum(o[:, LOST], 0)),
    ]
    v = np.stack(cols, axis=1)
    v[~np.isfinite(v)] = np.nan
    return ["sender_cwnd", "sender_srtt", "sender_delivery_rate", "sender_loss_rate"], v


@dataclass(frozen=True, slots=True)
class Library:
    """A design matrix whose columns are standardized on the fit rows.
    Column 0 is the constant. A feature owns one or two columns: its value
    (0, the fit mean, where undefined) and, if it is ever undefined, an
    indicator of that."""

    names: tuple[str, ...]  # feature -> "operator(variable)", or the variable
    variable: tuple[str, ...]  # feature -> its base variable
    columns: tuple[tuple[int, ...], ...]  # feature -> its columns
    same_as: dict[str, str]  # a feature left out -> the kept one it repeats
    fit: F64
    score: F64


def _history(v: F64, s: steps_data.Steps, idx: npt.NDArray[np.int64], stride: int) -> F64:
    """[anchor, age in rows, variable]: the variables at the anchor and at
    the rows before it in its own episode; NaN beyond the episode's start."""
    age = np.arange(max(WINDOWS))
    past = idx[:, None] - age[None]
    row = np.maximum(past, 0)
    ok = (past >= 0) & (s.episode[row] == s.episode[idx][:, None])
    ok &= s.step[idx][:, None] - s.step[row] == age[None] * stride
    h = v[row]
    h[~ok] = np.nan
    return h


def _operators(h: F64, stride: int) -> list[tuple[str, F64]]:
    out = [("", h[:, 0]), (f"change_{stride * 10}ms", h[:, 0] - h[:, 1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined row
        for w in WINDOWS:
            for op, f in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
                out.append((f"{op}_{w * stride * 10}ms", f(h[:, :w], axis=1)))
    return out


def library(
    names: list[str],
    v: F64,
    s: steps_data.Steps,
    fit: npt.NDArray[np.int64],
    score: npt.NDArray[np.int64],
    *,
    stride: int,
) -> Library:
    """Every variable under every operator, less the features that are
    constant, almost never defined, or repeat an earlier one."""
    ops_fit, ops_score = (
        _operators(_history(v, s, fit, stride), stride),
        _operators(_history(v, s, score, stride), stride),
    )
    kept: list[tuple[str, str, F64, F64]] = []  # name, variable, fit value, score value
    same_as: dict[str, str] = {}
    for (op, a), (_, b) in zip(ops_fit, ops_score, strict=True):
        for j, var in enumerate(names):
            name = f"{op}({var})" if op else var
            x = a[:, j]
            if np.isfinite(x).mean() < 0.01 or np.nanstd(x) < 1e-9:
                continue
            mean, std = np.nanmean(x), np.nanstd(x)
            xf = np.nan_to_num((x - mean) / std)
            twin = next(
                (n for n, _, y, _ in kept if abs(np.mean(xf * np.nan_to_num(y))) > 0.999), None
            )
            if twin is not None:
                same_as[name] = twin
            else:
                kept.append((name, var, (x - mean) / std, (b[:, j] - mean) / std))
    cols_fit, cols_score, columns = [np.ones(fit.size)], [np.ones(score.size)], []
    for _, _, x, y in kept:
        own = [len(cols_fit)]
        cols_fit.append(np.nan_to_num(x))
        cols_score.append(np.nan_to_num(y))
        gap = np.isnan(x)
        if 0 < gap.mean() < 1:
            m, sd = gap.mean(), gap.std()
            own.append(len(cols_fit))
            cols_fit.append((gap - m) / sd)
            cols_score.append((np.isnan(y) - m) / sd)
        columns.append(tuple(own))
    return Library(
        names=tuple(k[0] for k in kept),
        variable=tuple(k[1] for k in kept),
        columns=tuple(columns),
        same_as=same_as,
        fit=np.stack(cols_fit, axis=1),
        score=np.stack(cols_score, axis=1),
    )


@dataclass(frozen=True, slots=True)
class Moments:
    """What a least-squares fit of z on columns of a library needs."""

    g_fit: F64  # X'X on the fit rows
    c_fit: F64  # X'z
    g_score: F64
    c_score: F64
    zz_score: F64  # per latent, the mean of z squared on the score rows
    var_score: F64  # per latent, its variance on the score rows

    @staticmethod
    def of(lib: Library, z_fit: F64, z_score: F64) -> Moments:
        mean, std = z_fit.mean(0), z_fit.std(0) + 1e-12
        a, b = (z_fit - mean) / std, (z_score - mean) / std
        return Moments(
            g_fit=lib.fit.T @ lib.fit / len(a),
            c_fit=lib.fit.T @ a / len(a),
            g_score=lib.score.T @ lib.score / len(b),
            c_score=lib.score.T @ b / len(b),
            zz_score=(b**2).mean(0),
            var_score=b.var(0),
        )

    def _weights(self, cols: list[int]) -> F64:
        g = self.g_fit[np.ix_(cols, cols)] + 1e-8 * np.eye(len(cols))
        return np.linalg.solve(g, self.c_fit[cols])

    def explained(self, cols: list[int]) -> float:
        """Sum over latents of the variance the fit explains on the fit rows."""
        return float(np.sum(self.c_fit[cols] * self._weights(cols)))

    def r2(self, cols: list[int]) -> F64:
        """Per latent, on the score rows, of the fit made on the fit rows."""
        w = self._weights(cols)
        g = self.g_score[np.ix_(cols, cols)]
        resid = self.zz_score - 2 * np.sum(w * self.c_score[cols], 0) + np.sum(w * (g @ w), 0)
        return 1 - resid / self.var_score


def _columns(lib: Library, features: list[int]) -> list[int]:
    return [0, *(c for f in features for c in lib.columns[f])]


def greedy(lib: Library, m: Moments, allowed: list[int], steps: int) -> list[int]:
    """Features in the order a forward selection adds them.
    ponytail: greedy forward selection, a group lasso path if the picks
    prove unstable between seeds."""
    picked: list[int] = []
    for _ in range(min(steps, len(allowed))):
        rest = [f for f in allowed if f not in picked]
        picked.append(max(rest, key=lambda f: m.explained(_columns(lib, [*picked, f]))))
    return picked


def _path(lib: Library, m: Moments, allowed: list[int]) -> list[dict[str, Any]]:
    order = greedy(lib, m, allowed, STEPS)
    out = []
    for n in range(1, len(order) + 1):
        r = m.r2(_columns(lib, order[:n]))
        out.append(
            {"feature": lib.names[order[n - 1]], "mean": float(r.mean()), "worst": float(r.min())}
        )
    return out


def _selection(lib: Library, m: Moments) -> dict[str, Any]:
    everything = list(range(len(lib.names)))
    path = _path(lib, m, everything)
    index = {n: i for i, n in enumerate(lib.names)}
    lcfs = [index[n] for n in EXPLICCIT_LCFS if n in index]
    used = dict.fromkeys(lib.variable[index[p["feature"]]] for p in path[:PICK])
    hidden = [f for f in everything if not lib.variable[f].startswith("sender_")]
    return {
        "path": path,
        "library": float(m.r2(_columns(lib, everything)).mean()),
        "hidden_path": _path(lib, m, hidden),
        "hidden_library": float(m.r2(_columns(lib, hidden)).mean()),
        "expliccit_lcfs": float(m.r2(_columns(lib, lcfs)).mean()),
        "without": {
            var: _path(lib, m, [f for f in everything if lib.variable[f] != var])[:PICK]
            for var in used
        },
    }


def select(out: Path, dirs: list[Path], runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s, d, first, fit, score = _setup(dirs, runs)
    labels, b = base_variables(s)
    sender, m = sender_variables(s)
    lib = library(
        [*labels, *sender],
        np.concatenate([b, m], 1),
        s,
        fit.numpy(),
        score.numpy(),
        stride=first.stride,
    )
    z_of = _latents_of(runs, d, first, fit, score)
    moments = {run: Moments.of(lib, zf, zs) for run, (_, zf, zs) in z_of.items()}
    index = {n: i for i, n in enumerate(lib.names)}
    result: dict[str, Any] = {
        "features": lib.names,
        "same_as": lib.same_as,
        "runs": {run: {"k": z_of[run][0]} | _selection(lib, moments[run]) for run in z_of},
        "pooled": {},
        "transfer": [],
    }
    for k in sorted({k for k, _, _ in z_of.values()}):
        same = [v for v in z_of.values() if v[0] == k]
        pooled = Moments.of(
            lib, np.concatenate([v[1] for v in same], 1), np.concatenate([v[2] for v in same], 1)
        )
        result["pooled"][str(k)] = {"runs": len(same)} | _selection(lib, pooled)
    for a, c in itertools.permutations(z_of, 2):
        if z_of[a][0] == z_of[c][0]:
            picks = [index[p["feature"]] for p in result["runs"][a]["path"][:PICK]]
            r = float(moments[c].r2(_columns(lib, picks)).mean())
            result["transfer"].append({"from": a, "to": c, "k": z_of[a][0], "mean": r})
    (out / "select.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.md").write_text(select_summary(result))


def select_summary(result: dict[str, Any]) -> str:
    lines = [
        "R^2 of a joint linear fit of the latents, on validation links. Pooled: the "
        "latents of every run of one width together.\n"
    ]
    for k, sel in result["pooled"].items():
        lines += [
            f"\n## {k} latents, {sel['runs']} runs pooled\n\n",
            f"Whole library: {sel['library']:.3f}. expliCCit's seven concepts: "
            f"{sel['expliccit_lcfs']:.3f}.\n\n",
        ]
        hidden = (
            "\nThe library without the sender's measurements "
            f"(all of it: {sel['hidden_library']:.3f}):\n\n"
        )
        for intro, key in (("", "path"), (hidden, "hidden_path")):
            lines.append(
                intro + "| size | feature added | mean | worst latent |\n|---|---|---|---|\n"
            )
            for n, p in enumerate(sel[key], 1):
                lines.append(f"| {n} | {p['feature']} | {p['mean']:.3f} | {p['worst']:.3f} |\n")
        lines += [
            f"\nWith one base variable taken out of the library (first {PICK} features):\n\n",
            f"| taken out | mean at {PICK} | features |\n|---|---|---|\n",
        ]
        for var, path in sel["without"].items():
            names = ", ".join(p["feature"] for p in path)
            lines.append(f"| {var} | {path[-1]['mean']:.3f} | {names} |\n")
        lines.append(
            f"\nEach run alone, its first {PICK} features:\n\n"
            "| run | mean | features |\n|---|---|---|\n"
        )
        for run, r in result["runs"].items():
            if str(r["k"]) == k:
                path = r["path"][:PICK]
                names = ", ".join(p["feature"] for p in path)
                lines.append(f"| {run} | {path[-1]['mean']:.3f} | {names} |\n")
        t = [x["mean"] for x in result["transfer"] if str(x["k"]) == k]
        if t:
            lines.append(
                f"\nOne run's first {PICK} features fitted to another run's latents: "
                f"mean {np.mean(t):.3f}, lowest {np.min(t):.3f}.\n"
            )
    plain = {a: b for a, b in result["same_as"].items() if "(" not in a}
    if result["same_as"]:
        lines.append(
            f"\n{len(result['same_as'])} features were left out because they repeat another "
            "(correlation above 0.999; select.json lists them). Those without an operator:\n\n"
        )
        lines += [f"- {a} repeats {b}\n" for a, b in plain.items()]
    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)
    pr = sub.add_parser("probe")
    se = sub.add_parser("select")
    for q in (pr, se):
        q.add_argument("out", type=Path)
        q.add_argument("--data", type=Path, nargs="+", required=True)
        q.add_argument("--runs", type=Path, nargs="+", required=True)
    a = p.parse_args()
    (probe if a.verb == "probe" else select)(a.out, a.data, a.runs)


if __name__ == "__main__":
    main()
