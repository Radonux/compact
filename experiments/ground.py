"""Grounding: relate a trained controller's latents to ground truth.

The controllers are discovery controllers (`train ppo` with
PPOSpec.latents: an encoder in net1's place, a decoder in net2's). Each
is grounded on its own rollouts: the steps of episodes in which it drove.

Usage: python experiments/ground.py rollouts RUN_DIR --fit CAMPAIGN.json
           --score CAMPAIGN.json --build UML [--workers N] [--links N]

rollouts runs RUN_DIR's policy.onnx in the closed loop, without
exploration noise, and keeps every step with its labels: on N (path,
condition) pairs of the --fit campaign (the training links; the same
pairs and seeds for every run) into RUN_DIR/ground/fit, and on every pair
of the --score campaign (the validation links) into RUN_DIR/ground/score.
Fits are made on the first and scored on the second.

Usage: python experiments/ground.py probe OUT --runs RUN_DIR...

The rows are the controllers' decisions (one per clamp(min RTT, 10 ms,
50 ms), as the policy graph counts them). A latent is the encoder's output
on that step's measurements, so it is computed here exactly as the guest
computed it. probe is the first, linear look, before any formula search.
For every run, on its own rollouts:

  decision  R^2 of the decision u (log2 of the window the controller set)
            predicted from the latents alone, by a linear fit and by a
            quadratic one. This is the check for action collapse: near 1
            at width 1 or 2, the encoder computes the decision itself and
            the width counts nothing.
  read      R^2 of each base variable predicted from the latents, linear
            and quadratic (the latents, their squares and pairwise
            products): which variables the latents carry
  write     R^2 of each latent predicted linearly from all base variables:
            how much of a latent the vocabulary can express at all
  spread    each latent's standard deviation. Every fit standardizes the
            latents, so one that hardly moves counts as much as the rest.

and for each pair of runs of one width, on the rollouts of every run given:

  seeds     the mean R^2 of one run's latents predicted from the other's,
            linear and quadratic: whether the seeds found the same subspace

The base variables are the label columns of the step data and the
composites of them, each on a logarithmic scale (signed log1p of megabits
per second, milliseconds or packets). A linear probe is a lower bound: the
decoder is not linear, so a latent may hold a variable in a form these
fits miss.

Writes OUT/probe.json and OUT/summary.md.

Usage: python experiments/ground.py select OUT --runs RUN_DIR...

select names the few variables that span the latents. The base variables
are every label the step data stores, in each window it is stored in
(labels.py), every composite of them (train/lcfs.py), and four sender
measurements (window, smoothed RTT, delivery rate, loss rate). Nothing is
listed here: a label or composite added there joins the library. The
library is every base variable under eight operators: its value now, its
mean, minimum and maximum over the last 240 ms and 960 ms, and its change
over 60 ms. Sums of logarithms are products and ratios, so a linear fit on
this library is a fit by products of powers. STEPS features are added one
at a time, each the one that most raises the R^2 of a joint linear fit of
all latents on the fit rollouts. They are then put in order by taking them
away again, each time the one the fit loses least without, so a feature
that stood in for a mix of later ones falls to the end. Every size is
scored on the score rollouts. Reported per run on its own rollouts, and per width with the
latents of its runs pooled, on the rollouts of every run given (give the
runs of one teacher together):

  path     the features in the order added, with the mean and the worst
           latent's R^2 at each size
  bounds   the R^2 with the whole library, and with expliCCit's seven
           hand-picked concepts alone
  flexible the R^2 of a small network, in place of the linear fit, on the
           first PICK features, on the first PICK of the hidden path, and
           on the whole library. An encoder may hold a variable in any
           form, and the decoder can undo it, so the linear numbers are
           lower bounds. The features are enough when the network on them
           reaches the network on the whole library; what that one misses
           is not a function of the library at all (the sender's
           estimation error, or a variable nobody listed).
  hidden   the path and the bound when the sender's four measurements are
           taken out, so only ground truth is left. The latents are
           computed from the sender's message, so they can repeat what it
           measured; this path asks what they hold of the path's state.
  without  for each label behind the first PICK features: the path
           when that label is taken out of the library in every window,
           with every composite computed from it, which is the vocabulary
           sensitivity check. Taking out a label that no picked feature
           uses cannot change the path.
  transfer the R^2 of one run's latents on another run's first PICK
           features

Writes OUT/select.json and OUT/summary.md.
"""

# torch's own annotations leave many signatures partly unknown; every other
# strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from expliccit_env.labels import WINDOWS_MS
from expliccit_env.policy import OBS_COLUMNS
from expliccit_env.spec import AgentCC, EpisodeSpec, OnnxPolicy
from expliccit_env.train import actors, fit, lcfs, rollouts
from expliccit_env.train import data as steps_data
from expliccit_env.train.graph import FixedWindow
from expliccit_env.train.recurrent import OBS_MIN_RTT, decision_steps
from expliccit_env.train.spec import PPOSpec
from expliccit_env.uml import UmlBuild

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

# Decisions used, over all runs: fits on FIT of the fit rollouts, scores on
# SCORE of the score rollouts. ponytail: a fixed subsample, every decision
# if a fit ever needs them.
FIT, SCORE = 300_000, 200_000
FLEX_STEPS = 1500  # gradient steps of the flexible fit
_SCALE = {"_bps": 1e-6, "_ns": 1e-6, "_bytes": 1e-3}
SRTT, DELIVERY, LOST = (
    OBS_COLUMNS.index(c) for c in ("srtt_x100ms", "delivery_rate_x100mbps", "lost_x100mbps")
)
# The library's windows and its difference, in 10 ms steps.
WINDOWS, CHANGE = (24, 96), 6
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
SENDER = ("sender_cwnd", "sender_srtt", "sender_delivery_rate", "sender_loss_rate")


def _label(name: str) -> str:
    """A label column's name without its window."""
    return next(
        (name.removesuffix(f"_{w}ms") for w in WINDOWS_MS if name.endswith(f"_{w}ms")), name
    )


def sources(variable: str) -> frozenset[str]:
    """The labels, whatever the window, a base variable is read or computed
    from; a sender measurement is its own source."""
    return frozenset(_label(c) for c in lcfs.columns([variable]))


def _factor(name: str) -> float:
    if name in lcfs.LCF_UNITS:
        return lcfs.LCF_UNITS[name][0]
    return next((v for suffix, v in _SCALE.items() if _label(name).endswith(suffix)), 1.0)


def base_variables(s: steps_data.Steps) -> tuple[list[str], F64]:
    """Every stored label and every composite the stored labels give, on a
    logarithmic scale; NaN where it is undefined. The names depend on the
    stored columns alone, so every rollout of one guest gives the same."""
    stored = set(s.labels)
    made = {n for n in lcfs.LCF_UNITS if n not in stored and lcfs.columns([n]) <= stored}
    names = sorted(stored | made)
    cols = []
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for n in names:
            v = s.labels[n].astype(np.float64) * _factor(n)
            v = np.sign(v) * np.log1p(np.abs(v))
            v[~np.isfinite(v)] = np.nan
            cols.append(v)
    return names, np.stack(cols, axis=1)


def sender_variables(s: steps_data.Steps) -> F64:
    """SENDER: four measurements the sender has, on logarithmic scales."""
    o = s.obs.astype(np.float64)
    cols = [
        np.log2(np.maximum(s.cwnd.astype(np.float64), 1)),
        np.log(np.maximum(o[:, SRTT], 1e-4)),
        np.log1p(100 * np.maximum(o[:, DELIVERY], 0)),
        np.log1p(100 * np.maximum(o[:, LOST], 0)),
    ]
    v = np.stack(cols, axis=1)
    v[~np.isfinite(v)] = np.nan
    return v


def decisions(s: steps_data.Steps) -> I64:
    """The rows at which the controller decided, as the policy graph counts
    them; an episode's last step is left out, because a decision's window
    is read from the step after it."""
    starts = np.flatnonzero(np.r_[True, s.episode[1:] != s.episode[:-1]])
    out = []
    for a, b in zip(starts, np.r_[starts[1:], s.episode.size], strict=True):
        d = decision_steps(s.obs[a:b, OBS_MIN_RTT])
        out.append(a + d[d + 1 < b - a])
    return np.concatenate(out)


def _history(v: F64, s: steps_data.Steps, idx: I64) -> F64:
    """[anchor, age in steps, variable]: the variables at the anchor and at
    the steps before it in its own episode; NaN beyond the episode's start
    and where a step is missing."""
    age = np.arange(max(WINDOWS))
    past = idx[:, None] - age[None]
    row = np.maximum(past, 0)
    ok = (past >= 0) & (s.episode[row] == s.episode[idx][:, None])
    ok &= s.step[idx][:, None] - s.step[row] == age[None]
    h = v[row]
    h[~ok] = np.nan
    return h


def _operators(h: F64) -> list[tuple[str, F64]]:
    out = [("", h[:, 0]), (f"change_{CHANGE * 10}ms", h[:, 0] - h[:, CHANGE])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined row
        for w in WINDOWS:
            for op, f in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
                out.append((f"{op}_{w * 10}ms", f(h[:, :w], axis=1)))
    return out


def features(names: list[str], v: F64, s: steps_data.Steps, idx: I64) -> tuple[list[str], F64]:
    """Every variable under every operator at the anchors: the features'
    names, operator by operator (the variables themselves first), and
    their values [anchor, feature]."""
    # A few thousand anchors at a time: the history is anchors x 96 x variables.
    parts = [_operators(_history(v, s, i)) for i in np.array_split(idx, idx.size // 4096 + 1)]
    ops = [op for op, _ in parts[0]]
    x = np.concatenate([np.concatenate([a for _, a in part], axis=1) for part in parts])
    return [f"{op}({n})" if op else n for op in ops for n in names], x


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

    @staticmethod
    def of(names: list[str], variables: int, fit: F64, score: F64) -> Library:
        """The features (``features``: ``variables`` base variables under
        each operator) less those that are constant, almost never defined,
        or repeat an earlier one."""
        kept: list[tuple[int, F64]] = []  # feature, its standardized fit values (0 undefined)
        same_as: dict[str, str] = {}
        cols_fit, cols_score, columns = [np.ones(len(fit))], [np.ones(len(score))], []
        for j, name in enumerate(names):
            x = fit[:, j]
            if np.isfinite(x).mean() < 0.01 or np.nanstd(x) < 1e-9:
                continue
            mean, std = np.nanmean(x), np.nanstd(x)
            xf = np.nan_to_num((x - mean) / std)
            twin = next((i for i, y in kept if abs(np.mean(xf * y)) > 0.999), None)
            if twin is not None:
                same_as[name] = names[twin]
                continue
            kept.append((j, xf))
            own = [len(cols_fit)]
            cols_fit.append(xf)
            cols_score.append(np.nan_to_num((score[:, j] - mean) / std))
            gap = np.isnan(x)
            if 0 < gap.mean() < 1:
                m, sd = gap.mean(), gap.std()
                own.append(len(cols_fit))
                cols_fit.append((gap - m) / sd)
                cols_score.append((np.isnan(score[:, j]) - m) / sd)
            columns.append(tuple(own))
        return Library(
            names=tuple(names[j] for j, _ in kept),
            variable=tuple(names[j % variables] for j, _ in kept),
            columns=tuple(columns),
            same_as=same_as,
            fit=np.stack(cols_fit, axis=1),
            score=np.stack(cols_score, axis=1),
        )


@dataclass(frozen=True, slots=True)
class Controller:
    """A trained discovery controller: its width and its encoder."""

    name: str
    k: int
    encode: Callable[[F32], F64]  # [n, 78] step messages -> [n, k] latents

    @staticmethod
    def load(run: Path) -> Controller:
        spec = PPOSpec.model_validate_json((run / "ppo_spec.json").read_text())
        if spec.latents is None:
            raise SystemExit(f"{run}: not a discovery controller (PPOSpec.latents is unset)")
        net1, _ = fit.load_net1(run / "net1.pt")
        front = actors.front(net1, spec)
        actor = actors.new_actor(actors.width(net1, spec), spec)
        actor.load_state_dict(torch.load(run / "actor.pt", weights_only=True))
        actor.eval()

        @torch.no_grad()
        def encode(obs: F32) -> F64:
            # The view the exported graph gives the encoder (graph.Front).
            z = [actor.encode(front(o, None)[1]) for o in torch.from_numpy(obs).split(65_536)]
            return torch.cat(z).numpy().astype(np.float64)

        return Controller(run.name, spec.latents, encode)


@dataclass(frozen=True, slots=True)
class Rows:
    """Decisions of every run's rollouts of one kind (fit or score)."""

    owner: I64  # [a] the run (its index) whose controller drove
    obs: F32  # [a, 78] the step message at the decision
    u: F64  # [a] the decision: log2 of the window set, over FixedWindow's
    x: F64  # [a, F] every feature of the library, unscaled

    def own(self, run: int) -> npt.NDArray[np.bool_]:
        return self.owner == run


def _rows(runs: list[Path], kind: str, budget: int) -> tuple[list[str], int, Rows]:
    """(feature names, base variables, the rows) of each run's ground/kind."""
    rng = np.random.default_rng(0)
    names: list[str] = []
    variables = 0
    parts: list[tuple[I64, F32, F64, F64]] = []
    for r, run in enumerate(runs):
        s = steps_data.load([run / "ground" / kind], seed=0, val=0.0, test=0.0)
        labels, b = base_variables(s)
        d = decisions(s)
        d = np.sort(rng.choice(d, min(d.size, budget // len(runs)), replace=False))
        base = [*labels, *SENDER]
        own, x = features(base, np.concatenate([b, sender_variables(s)], axis=1), s, d)
        if names and own != names:
            raise SystemExit(f"{run}: its step data holds other labels than {runs[0]}'s")
        names, variables = own, len(base)
        u = np.log2(np.maximum(s.cwnd[d + 1].astype(np.float64), 1)) - FixedWindow.LOG2_PACKETS
        parts.append((np.full(d.size, r), s.obs[d], u, x))
    rows = Rows(
        owner=np.concatenate([p[0] for p in parts]),
        obs=np.concatenate([p[1] for p in parts]),
        u=np.concatenate([p[2] for p in parts]),
        x=np.concatenate([p[3] for p in parts]),
    )
    return names, variables, rows


def _setup(runs: list[Path]) -> tuple[list[Controller], list[str], int, Rows, Rows]:
    names_of = [r.name for r in runs]
    if len(set(names_of)) != len(runs):
        raise SystemExit(f"run directories must differ in name: {names_of}")
    names, variables, fit_rows = _rows(runs, "fit", FIT)
    _, _, score_rows = _rows(runs, "score", SCORE)
    return [Controller.load(r) for r in runs], names, variables, fit_rows, score_rows


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


def probe(out: Path, runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ctls, names, variables, fr, sr = _setup(runs)
    b_fit, b_score = fr.x[:, :variables], sr.x[:, :variables]  # the variables themselves
    bf_fit, bf_score = filled(b_fit, b_score)
    z_of = {c.name: (c.encode(fr.obs), c.encode(sr.obs)) for c in ctls}
    result: dict[str, Any] = {"variables": names[:variables], "runs": {}}
    for r, c in enumerate(ctls):
        f, s = fr.own(r), sr.own(r)
        z_fit, z_score = z_of[c.name][0][f], z_of[c.name][1][s]
        u_fit, u_score = fr.u[f][:, None], sr.u[s][:, None]
        result["runs"][c.name] = {
            "k": c.k,
            "fit": int(f.sum()),
            "score": int(s.sum()),
            "latent_std": z_fit.std(0).tolist(),
            "decision_linear": float(r2(z_fit, u_fit, z_score, u_score)[0]),
            "decision_quadratic": float(
                r2(quadratic(z_fit), u_fit, quadratic(z_score), u_score)[0]
            ),
            "read_linear": r2(z_fit, b_fit[f], z_score, b_score[s]).tolist(),
            "read_quadratic": r2(
                quadratic(z_fit), b_fit[f], quadratic(z_score), b_score[s]
            ).tolist(),
            "write_linear": r2(bf_fit[f], z_fit, bf_score[s], z_score).tolist(),
        }
    result["seeds"] = [
        {
            "from": a.name,
            "to": c.name,
            "k": a.k,
            "linear": float(
                np.mean(r2(z_of[a.name][0], z_of[c.name][0], z_of[a.name][1], z_of[c.name][1]))
            ),
            "quadratic": float(
                np.mean(
                    r2(
                        quadratic(z_of[a.name][0]),
                        z_of[c.name][0],
                        quadratic(z_of[a.name][1]),
                        z_of[c.name][1],
                    )
                )
            ),
        }
        for a, c in itertools.permutations(ctls, 2)
        if a.k == c.k
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
    lines = [
        "R^2 on the validation links, each controller on its own rollouts, mean over runs.\n\n",
        "The decision u from the latents alone. Near 1 at width 1 or 2 is action "
        "collapse:\n\n| | " + head,
        rule,
    ]
    for key in ("linear", "quadratic"):
        lines.append(f"| {key} | " + cells(lambda r, key=key: r[f"decision_{key}"]))
    lines.append("\n")
    for key, title in (("read_linear", "linear"), ("read_quadratic", "quadratic")):
        lines += [f"Base variable from the latents, {title}:\n\n| variable | " + head, rule]
        for i, name in enumerate(result["variables"]):
            lines.append(f"| {name} | " + cells(lambda r, i=i, key=key: r[key][i]))
        lines.append("\n")
    lines += ["Latents from all base variables, linear:\n\n| | " + head, rule]
    lines.append("| mean over latents | " + cells(lambda r: np.mean(r["write_linear"])))
    lines.append("| worst latent | " + cells(lambda r: np.min(r["write_linear"])))
    lines.append(
        "| smallest latent's standard deviation over the largest | "
        + cells(lambda r: np.min(r["latent_std"]) / np.max(r["latent_std"]))
    )
    lines += [
        "\nOne run's latents from another's, on every run's rollouts, mean over pairs:\n\n| | "
        + head,
        rule,
    ]
    for key in ("linear", "quadratic"):
        row = []
        for k in widths:
            v = [p[key] for p in result["seeds"] if p["k"] == k]
            row.append(f"{np.mean(v):.3f}" if v else "n/a")
        lines.append(f"| {key} | " + " | ".join(row) + " |\n")
    return "".join(lines)


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
    def of(
        lib: Library,
        z_fit: F64,
        z_score: F64,
        fit: npt.NDArray[np.bool_] | None = None,
        score: npt.NDArray[np.bool_] | None = None,
    ) -> Moments:
        """On the library's rows, or on those ``fit`` and ``score`` keep
        (z_fit and z_score hold these rows alone)."""
        x = lib.fit if fit is None else lib.fit[fit]
        y = lib.score if score is None else lib.score[score]
        mean, std = z_fit.mean(0), z_fit.std(0) + 1e-12
        a, b = (z_fit - mean) / std, (z_score - mean) / std
        return Moments(
            g_fit=x.T @ x / len(a),
            c_fit=x.T @ a / len(a),
            g_score=y.T @ y / len(b),
            c_score=y.T @ b / len(b),
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
    ponytail: the criterion is a linear fit, so a variable that acts only
    through a product with another is picked late; a selection scored by
    the flexible fit if PICK features prove too few for it."""
    picked: list[int] = []
    for _ in range(min(steps, len(allowed))):
        rest = [f for f in allowed if f not in picked]
        picked.append(max(rest, key=lambda f: m.explained(_columns(lib, [*picked, f]))))
    return picked


def pruned(lib: Library, m: Moments, order: list[int]) -> list[int]:
    """``order`` again, by backward elimination: the last is the feature the
    fit on all of them loses least without, and so on down. Forward
    selection alone keeps a first pick that stood in for a mix of later ones."""
    keep, gone = list(order), []
    while len(keep) > 1:
        least = max(keep, key=lambda f: m.explained(_columns(lib, [g for g in keep if g != f])))
        keep.remove(least)
        gone.append(least)
    return [*keep, *reversed(gone)]


def flexible(x_fit: F64, z_fit: F64, x: F64, z: F64) -> float:
    """The mean over latents of the held-out R^2 of a small network fitted
    to z on x. It can overfit the few links of the fit rows, so it is a
    lower bound on what any function of x explains.
    ponytail: one fixed network, a tuned one if its numbers decide a claim."""
    torch.manual_seed(0)
    mean, std = z_fit.mean(0), z_fit.std(0) + 1e-12
    a, b, t, v = (
        torch.tensor(q, dtype=torch.float32)
        for q in (x_fit, (z_fit - mean) / std, x, (z - mean) / std)
    )
    net = torch.nn.Sequential(
        torch.nn.Linear(a.shape[1], 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, b.shape[1]),
    )
    opt = torch.optim.Adam(net.parameters())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=FLEX_STEPS)
    for _ in range(FLEX_STEPS):
        i = torch.randint(len(a), (2048,))
        loss = ((net(a[i]) - b[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return float((1 - ((net(t) - v) ** 2).mean(0) / v.var(0)).mean())


def _path(lib: Library, m: Moments, allowed: list[int]) -> list[dict[str, Any]]:
    order = pruned(lib, m, greedy(lib, m, allowed, STEPS))
    out = []
    for n in range(1, len(order) + 1):
        r = m.r2(_columns(lib, order[:n]))
        out.append(
            {"feature": lib.names[order[n - 1]], "mean": float(r.mean()), "worst": float(r.min())}
        )
    return out


def _selection(lib: Library, m: Moments, flex: Callable[[list[int]], float]) -> dict[str, Any]:
    """``flex``: the flexible fit's R^2 on these columns of the library."""
    everything = list(range(len(lib.names)))
    path = _path(lib, m, everything)
    index = {n: i for i, n in enumerate(lib.names)}

    def picked(path: list[dict[str, Any]]) -> list[int]:
        return _columns(lib, [index[p["feature"]] for p in path[:PICK]])[1:]

    lcfs = [index[n] for n in EXPLICCIT_LCFS if n in index]
    of = [sources(v) for v in lib.variable]
    used = sorted(set().union(*(of[index[p["feature"]]] for p in path[:PICK])))
    hidden = [f for f in everything if not lib.variable[f].startswith("sender_")]
    hidden_path = _path(lib, m, hidden)
    return {
        "path": path,
        "library": float(m.r2(_columns(lib, everything)).mean()),
        "flexible": {
            "picked": flex(picked(path)),
            "hidden_picked": flex(picked(hidden_path)),
            "library": flex(list(range(1, lib.fit.shape[1]))),
        },
        "hidden_path": hidden_path,
        "hidden_library": float(m.r2(_columns(lib, hidden)).mean()),
        "expliccit_lcfs": float(m.r2(_columns(lib, lcfs)).mean()),
        "without": {
            var: _path(lib, m, [f for f in everything if var not in of[f]])[:PICK] for var in used
        },
    }


def select(out: Path, runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ctls, names, variables, fr, sr = _setup(runs)
    lib = Library.of(names, variables, fr.x, sr.x)
    z_of = {c.name: (c.encode(fr.obs), c.encode(sr.obs)) for c in ctls}
    moments = {
        c.name: Moments.of(
            lib, z_of[c.name][0][fr.own(r)], z_of[c.name][1][sr.own(r)], fr.own(r), sr.own(r)
        )
        for r, c in enumerate(ctls)
    }
    index = {n: i for i, n in enumerate(lib.names)}

    def flex(
        z: tuple[F64, F64], fit: Any = slice(None), score: Any = slice(None)
    ) -> Callable[[list[int]], float]:
        x, y = lib.fit[fit], lib.score[score]
        return lambda cols: flexible(x[:, cols], z[0], y[:, cols], z[1])

    def own(r: int, c: Controller) -> dict[str, Any]:
        f, s = fr.own(r), sr.own(r)
        z = z_of[c.name]
        return _selection(lib, moments[c.name], flex((z[0][f], z[1][s]), f, s))

    result: dict[str, Any] = {
        "features": lib.names,
        "same_as": lib.same_as,
        "runs": {c.name: {"k": c.k} | own(r, c) for r, c in enumerate(ctls)},
        "pooled": {},
        "transfer": [],
    }
    for k in sorted({c.k for c in ctls}):
        same = [z_of[c.name] for c in ctls if c.k == k]
        z = (np.concatenate([v[0] for v in same], 1), np.concatenate([v[1] for v in same], 1))
        result["pooled"][str(k)] = {"runs": len(same)} | _selection(
            lib, Moments.of(lib, *z), flex(z)
        )
    for a, c in itertools.permutations(ctls, 2):
        if a.k == c.k:
            picks = [index[p["feature"]] for p in result["runs"][a.name]["path"][:PICK]]
            r = float(moments[c.name].r2(_columns(lib, picks)).mean())
            result["transfer"].append({"from": a.name, "to": c.name, "k": a.k, "mean": r})
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
            f"A small network in place of the linear fit: {sel['flexible']['picked']:.3f} on "
            f"the first {PICK} features, {sel['flexible']['hidden_picked']:.3f} on the first "
            f"{PICK} without the sender's measurements, {sel['flexible']['library']:.3f} on "
            "the whole library.\n\n",
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


def drive(
    run: Path, fit_from: Path, score_from: Path, build: UmlBuild, *, workers: int, links: int
) -> None:
    """The run's controller in the closed loop, every step kept."""
    spec = PPOSpec.model_validate_json((run / "ppo_spec.json").read_text())
    policy = run / "policy.onnx"
    cc = AgentCC(
        policy=OnnxPolicy(path=policy, sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    )
    for kind, source, n in (("fit", fit_from, links), ("score", score_from, None)):
        # The same pairs and seeds for every run: both generators start alike.
        arms = rollouts.arms(source, np.random.default_rng(0))
        if n is not None and n < len(arms):
            pick = np.random.default_rng(0).choice(len(arms), n, replace=False)
            arms = [arms[i] for i in sorted(pick.tolist())]
        episodes = [
            EpisodeSpec(
                path=p.model_copy(update={"loss": c.loss}),
                cross=c.flows(p.trace),
                cc=cc,
                duration_ms=spec.duration_ms,
                seed=7 + i,
                steps=True,
            )
            for i, (p, c) in enumerate(arms)
        ]
        done = rollouts.run(run / "ground" / kind, episodes, build, workers, None)
        print(f"{run.name} {kind}: {json.dumps(done, default=float)}")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)
    ro = sub.add_parser("rollouts")
    ro.add_argument("run", type=Path)
    ro.add_argument("--fit", type=Path, required=True)
    ro.add_argument("--score", type=Path, required=True)
    ro.add_argument("--build", type=Path, required=True)
    ro.add_argument("--workers", type=int, default=1)
    ro.add_argument("--links", type=int, default=96)
    for verb in ("probe", "select"):
        q = sub.add_parser(verb)
        q.add_argument("out", type=Path)
        q.add_argument("--runs", type=Path, nargs="+", required=True)
    a = p.parse_args()
    if a.verb == "rollouts":
        drive(a.run, a.fit, a.score, UmlBuild.at(a.build), workers=a.workers, links=a.links)
    else:
        (probe if a.verb == "probe" else select)(a.out, a.runs)


if __name__ == "__main__":
    main()
