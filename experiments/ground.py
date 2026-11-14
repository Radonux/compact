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
            products): which variables the latents carry. Quadratic
            fits are left out (n/a) above WIDE latents.
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

select names the few variables that span the latents beyond the sender's
own window. The window is what the controller set a moment ago, so it
predicts the latents and the next decision without saying anything about
the path, and so does every quantity the window sets (the flow's packets
in flight, its rate). A concept is a variable of the path that says more
about the latents or the decision than the window's own history does. So
every fit below holds the window's features (OWN under the eight
operators) and a path lists what is added to them; "own" is the R^2 of
the window's features alone. A variable the window sets adds nothing to
them; one the window sets only in part (the queue) adds what it tells of
the path. The base variables
are every label the step data stores, in each window it is stored in
(labels.py), every composite of them (train/lcfs.py), and four sender
measurements (window, smoothed RTT, delivery rate, loss rate). Nothing is
listed here: a label or composite added there joins the library. The
library is every base variable under eight operators: its value now, its
mean, minimum and maximum over the last 240 ms and 960 ms, and its change
over 60 ms.

Every fit is made on places: a latent and a feature are each replaced by
their rank among the fit rows (``ranked``). An encoder may hold a variable
in any monotone form, since the decoder can undo it, and x, 1 / x and
log x are one variable. On places none of that changes a fit, and a
feature that orders the rows as another does is left out as a repeat.

STEPS features are added one at a time, each the one that most raises the
R^2 of a joint linear fit of all latents on the fit rollouts. They are
then put in order by taking them away again, each time the one the fit
loses least without, so a feature that stood in for a mix of later ones
falls to the end. Every size is scored on the score rollouts. Reported
per run on its own rollouts, and per width with the latents of its runs
pooled, on the rollouts of every run given (give the runs of one teacher
together):

  path     the features in the order added, with the mean and the worst
           latent's R^2 at each size
  bounds   the R^2 with the whole library, and with expliCCit's seven
           hand-picked concepts alone
  hidden   the path and the bound when the sender's four measurements are
           taken out, so only ground truth is left. The latents are
           computed from the sender's message, so they can repeat what it
           measured; this path asks what they hold of the path's state.
  formulas the hidden path again, with formulas built from the stored
           labels behind it allowed too (``formulas``): products of two
           or three labels, and sums of two terms of one dimension. A
           linear fit on places writes neither, and they are the form of
           the literature's composites, so a concept nobody listed can
           still be written in closed form.
  flexible the R^2 of a small network, in place of the linear fit, on the
           first PICK features of each path, and on the whole library. A
           mix of concepts through a nonlinear form is not linear even on
           places, so the linear numbers are lower bounds. The features
           are enough when the network on them reaches the network on the
           whole library; what that one misses is not a function of the
           library at all (the sender's estimation error, or a variable
           nobody listed).
  without  for each label behind the first PICK features: the path
           when that label is taken out of the library in every window,
           with every composite computed from it, which is the vocabulary
           sensitivity check. Taking out a label that no picked feature
           uses cannot change the path.
  transfer the R^2 of one run's latents on another run's first PICK
           features
  decision per run, all of the above with the run's decision in place of
           its latents: the change it makes, log2 of the window it set
           less log2 of the window in force. An encoder trained end to
           end can pass the decision itself through one latent, so the
           width of the latents counts no concepts. What the decision
           reads of the path's state is asked directly here, and its path
           is the order in which concepts are tried in retraining.

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
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from expliccit_env.labels import WINDOWS_MS
from expliccit_env.mahimahi import PACKET_SIZE
from expliccit_env.policy import OBS_COLUMNS
from expliccit_env.spec import AgentCC, EpisodeSpec, OnnxPolicy, StockCC
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
# The formula search: formulas kept, rows they are screened on (and those of
# these the first, coarse pass uses), factors in a
# sum's two terms together, the width above which the screen reads the
# latents linearly, and the rank correlation at which two formulas are one.
FORMULAS, SCREEN, COARSE, TERMS, WIDE, ALIKE = 32, 5000, 2000, 4, 16, 0.98
PAIRS, LABELS = 4, 12
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
OWN = "sender_cwnd"  # what the controller itself set: held in every fit of select


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


def ranked(ref: F64, x: F64) -> F64:
    """Per column, the place of x's values among ref's, from -0.5 to 0.5;
    equal values share a place and NaN stays NaN. A latent may hold a
    variable in any monotone form, and so may a feature (x, 1 / x and
    log x are one variable), so every fit is made on places: a monotone
    change of a column changes none of them. The ranks themselves and not
    their normal scores: on the stress test's mixes of seven concepts the
    scores lost a sixth of the linear fit, since many labels sit at a
    bound for much of the time and the scores stretch the rest."""
    out = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        r = np.sort(ref[np.isfinite(ref[:, j]), j])
        ok = np.isfinite(x[:, j])
        if r.size:
            v = x[ok, j]
            out[ok, j] = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (
                2 * r.size
            ) - 0.5
    return out


@dataclass(frozen=True, slots=True)
class Library:
    """A design matrix of the features' places (``ranked``) among the fit
    rows, standardized.
    Column 0 is the constant. A feature owns one or two columns: its value
    (0, the fit mean, where undefined) and, if it is ever undefined, an
    indicator of that."""

    names: tuple[str, ...]  # feature -> "operator(variable)", or the variable
    variable: tuple[str, ...]  # feature -> its base variable
    columns: tuple[tuple[int, ...], ...]  # feature -> its columns
    same_as: dict[str, str]  # a feature left out -> the kept one it repeats
    fit: F64
    score: F64
    base: tuple[str, ...]  # the base variables
    base_fit: F64  # [fit row, base variable] its value now, on its log scale, unscaled
    base_score: F64

    @staticmethod
    def of(names: list[str], variables: int, fit: F64, score: F64) -> Library:
        """The features (``features``: ``variables`` base variables under
        each operator) less those that are constant, almost never defined,
        or repeat an earlier one."""
        empty = Library(
            names=(),
            variable=(),
            columns=(),
            same_as={},
            fit=np.ones((len(fit), 1)),
            score=np.ones((len(score), 1)),
            base=tuple(names[:variables]),
            base_fit=fit[:, :variables].copy(),  # not a view: the caller may drop fit
            base_score=score[:, :variables].copy(),
        )
        return empty.plus(names, [names[j % variables] for j in range(len(names))], fit, score)

    def plus(self, names: list[str], variable: list[str], fit: F64, score: F64) -> Library:
        """With these features too, less those that are constant, almost
        never defined, or repeat a feature already here."""
        have = self.fit[:, [c[0] for c in self.columns]]  # the features' values, standardized
        kept = list(self.names)
        fresh: list[tuple[str, F64]] = []
        same_as = dict(self.same_as)
        cols_fit, cols_score = list(self.fit.T), list(self.score.T)
        columns, variables = list(self.columns), list(self.variable)
        for j, name in enumerate(names):
            raw = np.where(np.isfinite(fit[:, j]), fit[:, j], np.nan)
            if np.isfinite(raw).mean() < 0.01 or np.nanstd(raw) < 1e-9:
                continue
            x = ranked(raw[:, None], raw[:, None])[:, 0]
            mean, std = np.nanmean(x), np.nanstd(x)
            xf = np.nan_to_num((x - mean) / std)
            alike = np.abs(xf @ have) / len(xf) > 0.999
            twin = self.names[int(np.argmax(alike))] if alike.any() else None
            twin = twin or next((n for n, y in fresh if abs(np.mean(xf * y)) > 0.999), None)
            if twin is not None:
                same_as[name] = twin
                continue
            kept.append(name)
            fresh.append((name, xf))
            variables.append(variable[j])
            own = [len(cols_fit)]
            cols_fit.append(xf)
            other = ranked(raw[:, None], score[:, [j]])[:, 0]
            cols_score.append(np.nan_to_num((other - mean) / std))
            gap = np.isnan(x)
            if 0 < gap.mean() < 1:
                m, sd = gap.mean(), gap.std()
                own.append(len(cols_fit))
                cols_fit.append((gap - m) / sd)
                cols_score.append((np.isnan(other) - m) / sd)
            columns.append(tuple(own))
        return replace(
            self,
            names=tuple(kept),
            variable=tuple(variables),
            columns=tuple(columns),
            same_as=same_as,
            fit=np.stack(cols_fit, axis=1),
            score=np.stack(cols_score, axis=1),
        )

    def rows(self, fit: npt.NDArray[np.bool_], score: npt.NDArray[np.bool_]) -> Library:
        """The same library on some of its rows."""
        return replace(
            self,
            fit=self.fit[fit],
            score=self.score[score],
            base_fit=self.base_fit[fit],
            base_score=self.base_score[score],
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
    change: F64  # [a] the change it makes: u less the same of the window in force
    x: F64  # [a, F] every feature of the library, unscaled

    def own(self, run: int) -> npt.NDArray[np.bool_]:
        return self.owner == run


def _rows(runs: list[Path], kind: str, budget: int) -> tuple[list[str], int, Rows]:
    """(feature names, base variables, the rows) of each run's ground/kind."""
    rng = np.random.default_rng(0)
    names: list[str] = []
    variables = 0
    parts: list[tuple[I64, F32, F64, F64, F64]] = []
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
        w = np.log2(np.maximum(s.cwnd.astype(np.float64), 1)) - FixedWindow.LOG2_PACKETS
        parts.append((np.full(d.size, r), s.obs[d], w[d + 1], x, w[d + 1] - w[d]))
    rows = Rows(
        owner=np.concatenate([p[0] for p in parts]),
        obs=np.concatenate([p[1] for p in parts]),
        u=np.concatenate([p[2] for p in parts]),
        x=np.concatenate([p[3] for p in parts]),
        change=np.concatenate([p[4] for p in parts]),
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


def r2_quadratic(x_fit: F64, y_fit: F64, x: F64, y: F64) -> F64:
    """r2 on the quadratic terms of x. NaN for every column when x is wider
    than WIDE: the terms then number in the thousands, and a fit on them
    takes more memory and time than the answer is worth."""
    if x.shape[1] > WIDE:
        return np.full(y.shape[1], np.nan)
    return r2(quadratic(x_fit), y_fit, quadratic(x), y)


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
            "decision_quadratic": float(r2_quadratic(z_fit, u_fit, z_score, u_score)[0]),
            "read_linear": r2(z_fit, b_fit[f], z_score, b_score[s]).tolist(),
            "read_quadratic": r2_quadratic(z_fit, b_fit[f], z_score, b_score[s]).tolist(),
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
                    r2_quadratic(z_of[a.name][0], z_of[c.name][0], z_of[a.name][1], z_of[c.name][1])
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
            v = np.array([p[key] for p in result["seeds"] if p["k"] == k])
            row.append("n/a" if np.isnan(v).all() else f"{np.nanmean(v):.3f}")
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
    def of(lib: Library, z_fit: F64, z_score: F64) -> Moments:
        """Of the latents' places among the fit rows, like the library's."""
        x, y = lib.fit, lib.score
        a, b = ranked(z_fit, z_fit), ranked(z_fit, z_score)
        mean, std = a.mean(0), a.std(0) + 1e-12
        a, b = (a - mean) / std, (b - mean) / std
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


def _window(lib: Library) -> list[int]:
    """The features of the sender's own window."""
    return [f for f, v in enumerate(lib.variable) if v == OWN]


def greedy(
    lib: Library, m: Moments, allowed: list[int], steps: int, held: Sequence[int] = ()
) -> list[int]:
    """Features in the order a forward selection adds them to ``held``.
    ponytail: the criterion is a linear fit, so a variable that acts only
    through a product with another is picked late; a selection scored by
    the flexible fit if PICK features prove too few for it."""
    picked: list[int] = []
    for _ in range(min(steps, len(allowed))):
        rest = [f for f in allowed if f not in picked]
        picked.append(max(rest, key=lambda f: m.explained(_columns(lib, [*held, *picked, f]))))
    return picked


def pruned(lib: Library, m: Moments, order: list[int], held: Sequence[int] = ()) -> list[int]:
    """``order`` again, by backward elimination: the last is the feature the
    fit on all of them and ``held`` loses least without, and so on down. Forward
    selection alone keeps a first pick that stood in for a mix of later ones."""
    keep, gone = list(order), []
    while len(keep) > 1:
        least = max(
            keep, key=lambda f: m.explained(_columns(lib, [*held, *(g for g in keep if g != f)]))
        )
        keep.remove(least)
        gone.append(least)
    return [*keep, *reversed(gone)]


def flexible(x_fit: F64, z_fit: F64, x: F64, z: F64) -> float:
    """The mean over latents of the held-out R^2 of a small network fitted
    to z on x. It can overfit the few links of the fit rows, so it is a
    lower bound on what any function of x explains.
    ponytail: one fixed network, a tuned one if its numbers decide a claim."""
    torch.manual_seed(0)
    zf, zs = ranked(z_fit, z_fit), ranked(z_fit, z)  # as Moments scores them
    scale = zf.std(0) + 1e-12
    a, b, t, v = (torch.tensor(q, dtype=torch.float32) for q in (x_fit, zf / scale, x, zs / scale))
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
    """The features of ``allowed`` added to the sender's own window, in
    order, each with the R^2 of the window's features and those so far."""
    held = _window(lib)
    allowed = [f for f in allowed if f not in held]
    order = pruned(lib, m, greedy(lib, m, allowed, STEPS, held), held)
    out = []
    for n in range(1, len(order) + 1):
        r = m.r2(_columns(lib, [*held, *order[:n]]))
        out.append(
            {"feature": lib.names[order[n - 1]], "mean": float(r.mean()), "worst": float(r.min())}
        )
    return out


# A stored label's factor to SI units and its dimension (bits, seconds). A
# packet counts as PACKET_SIZE bytes, as in the composites.
_SI: dict[str, tuple[float, tuple[int, int]]] = {
    "_bps": (1.0, (1, -1)),
    "_ns": (1e-9, (0, 1)),
    "_bytes": (8.0, (1, 0)),
    "_pkts": (8.0 * PACKET_SIZE, (1, 0)),
}
type Monomial = tuple[tuple[int, int], ...]  # (variable, exponent 1 or -1), ...
type Sum = tuple[Monomial, int, Monomial]  # m1 + sign * m2; sign 0 is m1 alone


def _si(label: str) -> tuple[float, tuple[int, int]]:
    return next((v for suffix, v in _SI.items() if label.endswith(suffix)), (1.0, (0, 0)))


def _sums(dims: list[tuple[int, int]], most: int) -> list[Sum]:
    """Every product of two or three of the variables or their inverses,
    and every sum and difference of two terms of one dimension, a term
    being 1 or a product of up to three, with ``most`` factors at most in
    the two together."""
    by: dict[tuple[int, int], list[Monomial]] = {}
    out: list[Sum] = []
    for r in range(min(most, 3) + 1):
        for vs in itertools.combinations(range(len(dims)), r):
            for es in itertools.product((1, -1), repeat=r):
                m = tuple(zip(vs, es, strict=True))
                d0, d1 = (sum(e * dims[v][i] for v, e in m) for i in (0, 1))
                by.setdefault((d0, d1), []).append(m)
                if r > 1:
                    out.append((m, 0, ()))
    for group in by.values():
        for a, b in itertools.combinations(group, 2):
            if len(a) + len(b) <= most:
                out += [(a, 1, b), (a, -1, b)]
    return out


def _written(f: Sum, names: list[str]) -> str:
    def term(m: Monomial) -> str:
        top = "*".join(names[v] for v, e in m if e > 0) or "1"
        under = [names[v] for v, e in m if e < 0]
        if len(under) > 1:
            return f"{top}/({'*'.join(under)})"
        return top + "".join("/" + u for u in under)

    a, sign, b = f
    return term(a) + (f" {'+' if sign > 0 else '-'} {term(b)}" if sign else "")


def _values(sums: list[Sum], raw: F64) -> F64:
    """[row, sum] from the variables' values [row, variable]."""
    made: dict[Monomial, F64] = {}

    def product(m: Monomial) -> F64:
        if m not in made:
            out = np.ones(len(raw))
            for v, e in m:
                out = out * raw[:, v] if e > 0 else out / raw[:, v]
            made[m] = out
        return made[m]

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([product(a) + sign * product(b) for a, sign, b in sums], axis=1)


def _places(x: F64) -> F64:
    """``ranked`` among x's own rows, standardized, an undefined entry in the middle."""
    r = np.nan_to_num(ranked(x, x))
    return r / (r.std(0) + 1e-12)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A formula over some labels of one window."""

    formula: Sum
    labels: tuple[str, ...]  # the variables of the formula, without the window
    at: tuple[int, ...]  # their columns in the library's base
    name: str

    def cost(self) -> tuple[int, int, int]:
        """Factors, terms and inverses: fewer is simpler, in that order."""
        a, sign, b = self.formula
        return len(a) + len(b), abs(sign), sum(e < 0 for _, e in a + b)


def formulas(  # noqa: PLR0915  (one search, read top to bottom)
    lib: Library, z: tuple[F64, F64], picked: list[str]
) -> tuple[Library, list[str]]:
    """(the library with formulas the latents suggest, their names).

    A linear fit on places writes no product and no sum of two variables.
    This adds both, over stored labels of one window at a time (``_sums``).
    Two terms are added only if they have one dimension in SI units, so
    no coefficient is fitted. That is the form of the literature's
    composites (lcfs.COMPOSITES).

    The search has two rounds. The first tries every formula of two
    factors over all the stored labels: a difference of two variables
    that are nearly equal (arrival rate less capacity) shows in no linear
    fit on either, so the picked features cannot be trusted to lead to
    it. The second tries formulas of up to TERMS factors over LABELS
    labels: those of the first round's best PAIRS formulas, then those
    behind ``picked``, in its order.

    FORMULAS of them are kept: those the latents hold best. A formula is
    scored by how well the latents' places predict its own, with their
    squares and products while those are few: what a decoder
    can read from the latents is what they hold, however they hold it.
    The sender's own window is taken out of both first (a linear fit on
    its features), so a formula for the window itself scores nothing.
    Of formulas that order the rows alike (rank correlation above ALIKE)
    the one with the fewest factors, then the fewest inverses, is kept.
    ponytail: two terms with coefficient 1, PySR on the picked labels if
    the flexible fit shows a form beyond this."""
    index = {n: i for i, n in enumerate(lib.names)}
    stored = sorted(
        {_label(n) for n in lib.base}
        - set(lcfs.COMPOSITES)
        - {n for n in map(_label, lib.base) if n.startswith("sender_")}
    )
    some = np.random.default_rng(0).choice(len(z[0]), min(len(z[0]), SCREEN), replace=False)
    zs = _places(z[0][some])
    design = zs if zs.shape[1] > WIDE else quadratic(zs)
    # The constant and the sender's own window, taken out of the latents and
    # of every formula before one is scored on the other.
    held = np.linalg.qr(lib.fit[some][:, _columns(lib, _window(lib))])[0]
    few = min(len(some), COARSE)
    held_few = np.linalg.qr(held[:few])[0].astype(np.float32)
    q, _ = np.linalg.qr(design - held @ (held.T @ design))
    coarse = np.linalg.qr(design[:few] - held_few @ (held_few.T @ design[:few]))[0]
    coarse = coarse.astype(np.float32)
    places = np.linspace(-0.5, 0.5, few, dtype=np.float32)

    def raw(x: F64, c: _Candidate) -> F64:
        """The candidate's labels in SI units, from the library's log scale."""
        at = list(c.at)
        k = np.array([_si(n)[0] / _factor(lib.base[i]) for n, i in zip(c.labels, at, strict=True)])
        return np.sign(x[:, at]) * np.expm1(np.abs(x[:, at])) * k

    def screened(
        labels: list[str], most: int, windows: list[int]
    ) -> list[tuple[float, _Candidate]]:
        """(-score, formula) on some of the rows, with equal and undefined
        values in no particular place: there are thousands of formulas."""
        out: list[tuple[float, _Candidate]] = []
        sums = _sums([_si(n)[1] for n in labels], most)
        for w in windows:
            names = [f"{n}_{w}ms" if f"{n}_{w}ms" in lib.base else n for n in labels]
            at = tuple(lib.base.index(n) for n in names)
            whole = _Candidate(((), 0, ()), tuple(labels), at, "")
            r = raw(lib.base_fit[some[:few]], whole)
            for lo in range(0, len(sums), 1024):
                part = sums[lo : lo + 1024]
                v = _values(part, r).astype(np.float32)
                ok = np.isfinite(v)
                g = np.empty_like(v)
                np.put_along_axis(g, np.argsort(np.where(ok, v, 0), axis=0), places[:, None], 0)
                rest = g - held_few @ (held_few.T @ g)
                score = np.sum((coarse.T @ rest) ** 2, axis=0) / np.sum(g**2, axis=0)
                score[ok.mean(0) < 0.5] = 0.0
                out += [
                    (-float(c), _Candidate(f, tuple(labels), at, _written(f, names)))
                    for c, f in zip(score, part, strict=True)
                ]
        return sorted(out, key=lambda t: (t[0], t[1].cost(), t[1].name))

    def variables(c: _Candidate) -> list[str]:
        a, _, b = c.formula
        return [c.labels[v] for v, _ in a + b]

    used = {lib.variable[index[f]] for f in picked if f in index}
    pairs = screened(stored, 2, list(WINDOWS_MS))
    leads = [n for _, c in pairs[:PAIRS] for n in variables(c)]
    leads += [n for f in picked if f in index for n in sorted(sources(lib.variable[index[f]]))]
    labels = sorted(list(dict.fromkeys(n for n in leads if n in stored))[:LABELS])
    windows = [
        w
        for w in WINDOWS_MS
        if w == WINDOWS_MS[0]
        or any(v.endswith(f"_{w}ms") for v in used)
        or any(c.name.count(f"_{w}ms") for _, c in pairs[:PAIRS])
    ]
    first = sorted(
        {c.name: (s, c) for s, c in pairs + screened(labels, TERMS, windows)}.values(),
        key=lambda t: (t[0], t[1].cost(), t[1].name),
    )

    def exact(c: _Candidate) -> F64:
        return _places(_values([c.formula], raw(lib.base_fit[some], c)))[:, 0]

    def beyond(g: F64) -> float:
        return float(np.sum((q.T @ (g - held @ (held.T @ g))) ** 2) / len(some))

    found = sorted(
        ((-round(beyond(exact(c)), 2), *c.cost(), c.name), c) for _, c in first[: 16 * FORMULAS]
    )
    kept: list[_Candidate] = []
    ranks: list[F64] = []
    for _, c in found:
        g = exact(c)
        if all(abs(np.mean(g * h)) < ALIKE for h in ranks):
            kept.append(c)
            ranks.append(g)
        if len(kept) == FORMULAS:
            break
    if not kept:
        return lib, []
    fit, score = (
        np.concatenate([_values([c.formula], raw(x, c)) for c in kept], axis=1)
        for x in (lib.base_fit, lib.base_score)
    )
    names = [c.name for c in kept]
    wider = lib.plus(names, names, fit, score)
    return wider, [n for n in names if n in wider.names]


def _selection(lib: Library, z: tuple[F64, F64]) -> dict[str, Any]:
    """Of latents ``z`` on the library's rows."""
    m = Moments.of(lib, *z)
    everything = list(range(len(lib.names)))
    path = _path(lib, m, everything)
    index = {n: i for i, n in enumerate(lib.names)}

    def flex(lib: Library, path: list[dict[str, Any]]) -> float:
        index = {n: i for i, n in enumerate(lib.names)}
        cols = _columns(lib, [*_window(lib), *(index[p["feature"]] for p in path[:PICK])])[1:]
        if not cols:
            return 0.0
        return flexible(lib.fit[:, cols], z[0], lib.score[:, cols], z[1])

    lcfs = [*_window(lib), *(index[n] for n in EXPLICCIT_LCFS if n in index)]
    of = [sources(v) for v in lib.variable]
    used = sorted(set().union(*(of[index[p["feature"]]] for p in path[:PICK])))
    hidden = [
        f for f in everything if f in _window(lib) or not lib.variable[f].startswith("sender_")
    ]
    hidden_path = _path(lib, m, hidden)
    wider, made = formulas(lib, z, [p["feature"] for p in hidden_path])
    again = {n: i for i, n in enumerate(wider.names)}
    formula_path = _path(
        wider,
        Moments.of(wider, *z),
        [again[lib.names[f]] for f in hidden] + [again[n] for n in made],
    )
    return {
        "path": path,
        "own": float(m.r2(_columns(lib, _window(lib))).mean()),
        "library": float(m.r2(_columns(lib, everything)).mean()),
        "flexible": {
            "own": flex(lib, []),
            "picked": flex(lib, path),
            "hidden_picked": flex(lib, hidden_path),
            "formula_picked": flex(wider, formula_path),
            "library": flexible(lib.fit[:, 1:], z[0], lib.score[:, 1:], z[1]),
        },
        "hidden_path": hidden_path,
        "hidden_library": float(m.r2(_columns(lib, hidden)).mean()),
        "formula_path": formula_path,
        "formulas": made,
        "expliccit_lcfs": float(m.r2(_columns(lib, lcfs)).mean()),
        "without": {
            var: _path(lib, m, [f for f in everything if var not in of[f]])[:PICK] for var in used
        },
    }


def select(out: Path, runs: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ctls, names, variables, fr, sr = _setup(runs)
    lib = Library.of(names, variables, fr.x, sr.x)
    # The library holds the features now. The rows' own copy is a gigabyte
    # or more that nothing below reads.
    fr, sr = (replace(q, x=np.empty((len(q.u), 0))) for q in (fr, sr))
    z_of = {c.name: (c.encode(fr.obs), c.encode(sr.obs)) for c in ctls}
    index = {n: i for i, n in enumerate(lib.names)}
    result: dict[str, Any] = {
        "features": lib.names,
        "same_as": lib.same_as,
        "runs": {},
        "pooled": {},
        "transfer": [],
    }
    # Each run on its own rollouts, one library of its rows at a time.
    moments: dict[str, Moments] = {}
    for r, c in enumerate(ctls):
        own = lib.rows(fr.own(r), sr.own(r))
        z = (z_of[c.name][0][fr.own(r)], z_of[c.name][1][sr.own(r)])
        u = (fr.change[fr.own(r)][:, None], sr.change[sr.own(r)][:, None])
        result["runs"][c.name] = {"k": c.k} | _selection(own, z) | {"decision": _selection(own, u)}
        moments[c.name] = Moments.of(own, *z)
        print(f"select: {c.name}", file=sys.stderr, flush=True)
    for k in sorted({c.k for c in ctls}):
        same = [z_of[c.name] for c in ctls if c.k == k]
        z = (np.concatenate([v[0] for v in same], 1), np.concatenate([v[1] for v in same], 1))
        result["pooled"][str(k)] = {"runs": len(same)} | _selection(lib, z)
        print(f"select: pooled {k}", file=sys.stderr, flush=True)
    for a, c in itertools.permutations(ctls, 2):
        if a.k == c.k:
            picks = [index[p["feature"]] for p in result["runs"][a.name]["path"][:PICK]]
            picks += _window(lib)
            r = float(moments[c.name].r2(_columns(lib, picks)).mean())
            result["transfer"].append({"from": a.name, "to": c.name, "k": a.k, "mean": r})
    (out / "select.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.md").write_text(select_summary(result))


def select_summary(result: dict[str, Any]) -> str:
    lines = [
        "R^2 of a joint linear fit of the latents, on validation links. Every fit holds "
        "the features of the sender's own window; a path lists what is added to them. "
        "Pooled: the latents of every run of one width together.\n"
    ]
    for k, sel in result["pooled"].items():
        lines += [
            f"\n## {k} latents, {sel['runs']} runs pooled\n\n",
            f"The sender's own window alone: {sel['own']:.3f}. Whole library: "
            f"{sel['library']:.3f}. expliCCit's seven concepts: "
            f"{sel['expliccit_lcfs']:.3f}.\n\n",
            f"A small network in place of the linear fit: {sel['flexible']['own']:.3f} on "
            f"the window alone, {sel['flexible']['picked']:.3f} on "
            f"the first {PICK} features, {sel['flexible']['hidden_picked']:.3f} on the first "
            f"{PICK} without the sender's measurements, {sel['flexible']['library']:.3f} on "
            "the whole library.\n\n",
        ]
        hidden = (
            "\nThe library without the sender's measurements "
            f"(all of it: {sel['hidden_library']:.3f}):\n\n"
        )
        sums = (
            "\nGround truth again, with the sums of labels that the latents hold best "
            f"allowed too (a small network on the first {PICK}: "
            f"{sel['flexible']['formula_picked']:.3f}):\n\n"
        )
        for intro, key in (("", "path"), (hidden, "hidden_path"), (sums, "formula_path")):
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
        lines.append(
            f"\nEach run's decision (the change it makes to the window) in place of its "
            f"latents, its first {PICK} features of ground truth (linear, and a small "
            "network on them):\n\n"
            "| run | window alone | linear | network on the window | network "
            "| network on the library | features |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for run, r in result["runs"].items():
            if str(r["k"]) == k:
                d = r["decision"]
                path = d["formula_path"][:PICK]
                names = ", ".join(p["feature"] for p in path)
                lines.append(
                    f"| {run} | {d['own']:.3f} | {path[-1]['mean']:.3f} "
                    f"| {d['flexible']['own']:.3f} | {d['flexible']['formula_picked']:.3f} "
                    f"| {d['flexible']['library']:.3f} | {names} |\n"
                )
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


def drive(  # noqa: PLR0917  (one caller per verb, each naming what it passes)
    out: Path,
    ccs: Sequence[AgentCC | StockCC],
    duration_ms: int,
    fit_from: Path,
    score_from: Path,
    build: UmlBuild,
    *,
    workers: int,
    links: int,
) -> None:
    """The controllers, in turn, in the closed loop, every step kept, into
    OUT/ground."""
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
                cc=ccs[i % len(ccs)],
                duration_ms=duration_ms,
                seed=7 + i,
                steps=True,
            )
            for i, (p, c) in enumerate(arms)
        ]
        done = rollouts.run(out / "ground" / kind, episodes, build, workers, None)
        print(f"{out.name} {kind}: {json.dumps(done, default=float)}")


def _own(run: Path) -> tuple[AgentCC, int]:
    """(a run's exported policy, its episodes' length)."""
    spec = PPOSpec.model_validate_json((run / "ppo_spec.json").read_text())
    policy = run / "policy.onnx"
    sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    return AgentCC(policy=OnnxPolicy(path=policy, sha256=sha)), spec.duration_ms


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
        cc, duration_ms = _own(a.run)
        drive(
            a.run,
            [cc],
            duration_ms,
            a.fit,
            a.score,
            UmlBuild.at(a.build),
            workers=a.workers,
            links=a.links,
        )
    else:
        (probe if a.verb == "probe" else select)(a.out, a.runs)


if __name__ == "__main__":
    main()
