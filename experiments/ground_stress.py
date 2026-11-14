"""Stress test of grounding: latents planted with a known answer.

Usage: python experiments/ground_stress.py rollouts OUT --fit CAMPAIGN.json
           --score CAMPAIGN.json --build UML [--workers N] [--links N]
       python experiments/ground_stress.py run OUT

rollouts runs Cubic and BBRv3 in turn on N (path, condition) pairs of the
--fit campaign and on every pair of the --score campaign, every step
kept, into OUT/ground. No trained controller takes part: the steps only
supply ground truth that moves as it does under real controllers.

run plants latents on those steps and puts them through ground.py's
selection, as if an encoder had learned exactly the planted concepts. It
answers one question: if training found real concepts, does grounding
give them back in closed form? A planted case is

  concepts  features of the library that the latents are made from
  form      how a latent holds them: on the library's log scale ("log"),
            as a power of the plain value ("value"), through a tanh ("squash"), as a
            random invertible mix of them ("mix"), or as a mix through a
            tanh ("mix-squash"). A decoder can undo each of these.
  noise     white noise added to each latent, in its standard deviations

and the groups are

  composite   each literature composite (train/lcfs.py) alone
  seven       expliCCit's seven hand-picked concepts together, in 7
              latents and in 10
  random      random sets of 2 and of 4 ground-truth variables
  operator    one variable under a windowed operator
  off-grammar formulas the library cannot write (a minimum, a clipped
              ratio, a weighted sum), to see what is reported then
  null        latents with no relation to the path: nothing may be found
  degenerate  a latent that hardly moves, and a latent held three times
  sender      what the sender measures, grounded in ground truth alone

Each case is selected twice. "curated" allows every ground-truth feature,
so the planted concepts are in the vocabulary and must be picked.
"labels" takes every composite out, so a planted composite has to be
rebuilt from the stored labels: its closed form has to be found, not
looked up. The sender's measurements are never allowed.

Per case and regime, on the score rollouts:

  limit     the R^2 the noise leaves to explain. Every fit is made on
            places (ground.ranked), so this is the squared rank
            correlation of a latent with itself before the noise.
  linear    R^2 of the joint linear fit on the first c features (c is the
            number of concepts planted) and on the first PICK
  flexible  R^2 of a small network on the first PICK features, and on
            every allowed feature (run only where the linear fit falls
            short of the limit)
  span      the worst planted concept's R^2 from the first c features,
            and from the first PICK: whether the picks span the concepts
  named     the worst planted concept's best rank correlation with one
            picked feature: 1 when some pick is the concept up to a
            monotone change, which is all a decoder cares about
  formulas  named and linear again, with the sums ground.py builds from
            the picked labels allowed too

Writes OUT/stress.json and OUT/summary.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ground
import numpy as np
import numpy.typing as npt
from ground import PICK, STEPS, Library, Moments

from expliccit_env.spec import StockCC
from expliccit_env.train import lcfs
from expliccit_env.uml import UmlBuild

type F64 = npt.NDArray[np.float64]

TEACHERS = ("cubic", "bbr3")
NOISE = (0.0, 0.3, 1.0)
NAMED, SPANNED, ENOUGH = 0.99, 0.98, 0.02  # the verdicts' thresholds


@dataclass(frozen=True, slots=True)
class Case:
    group: str
    title: str
    concepts: tuple[str, ...]  # features planted; none for a custom latent
    make: Callable[[F64, np.random.Generator], F64]  # concepts [n, c], standardized -> latents
    noise: float


def _value(c: F64) -> F64:
    return np.sign(c) * np.expm1(np.abs(c))


def _mixer(k: int, squash: bool) -> Callable[[F64, np.random.Generator], F64]:
    def make(c: F64, rng: np.random.Generator) -> F64:
        z = c @ rng.normal(size=(c.shape[1], k)) / np.sqrt(c.shape[1])
        return np.tanh(z) if squash else z

    return make


FORMS: dict[str, Callable[[F64, np.random.Generator], F64]] = {
    "log": lambda c, _: c,
    "value": lambda c, _: np.exp(c),
    "squash": lambda c, _: np.tanh(1.5 * c),
}


def cases(names: list[str], variables: int) -> list[Case]:
    """names: the library's features; the first ``variables`` are the base
    variables at their current value."""
    out: list[Case] = []
    truth = [n for n in names[:variables] if not n.startswith("sender_")]
    for name in lcfs.COMPOSITES:
        for form, make in FORMS.items():
            out += [Case("composite", f"{name}, {form}", (name,), make, s) for s in NOISE[:2]]
    seven = tuple(n for n in ground.EXPLICCIT_LCFS if n in names)
    for k in (len(seven), 10):
        for squash in (False, True):
            form = "mix-squash" if squash else "mix"
            out += [
                Case("seven", f"the seven in {k} latents, {form}", seven, _mixer(k, squash), s)
                for s in NOISE
            ]
    rng = np.random.default_rng(0)
    for c in (2, 4):
        for _ in range(6):
            pick = tuple(str(n) for n in rng.choice(truth, c, replace=False))
            out.append(Case("random", f"{', '.join(pick)}, mix-squash", pick, _mixer(c, True), 0.3))
    for name in (
        "mean_960ms(queue_delay_mean_ns)",
        "max_240ms(load_factor)",
        "min_960ms(available_capacity_bps)",
        "change_60ms(queue_pkts)",
    ):
        out += [Case("operator", f"{name}, {f}", (name,), m, 0.3) for f, m in FORMS.items()]
    return out


def custom(get: Callable[[str], F64]) -> list[tuple[str, str, tuple[str, ...], F64, float]]:
    """(group, title, the variables it is made of, latents, noise) of the
    cases that are not a form of library features. ``get``: a base
    variable in its reported unit (Mbit/s, ms, packets), 0 where undefined."""
    cap, rtt, qd = get("capacity_bps"), get("base_rtt_ns"), get("queue_delay_mean_ns")
    queue, buffer = get("queue_pkts"), get("buffer_pkts")
    n = len(cap)
    rng = np.random.default_rng(1)
    # One draw per link: capacity, base RTT and buffer name the link.
    _, link = np.unique(np.stack([cap, rtt, buffer], 1).round(6), axis=0, return_inverse=True)
    bdp = get("bdp_pkts")
    live = np.log1p(bdp)[:, None]
    off = [
        ("min(BDP, buffer)", ("bdp_pkts", "buffer_pkts"), np.log1p(np.minimum(bdp, buffer))),
        (
            "excess demand over capacity times base RTT, clipped",
            ("arrival_rate_bps", "capacity_bps", "base_rtt_ns"),
            np.clip(get("arrival_rate_bps") / np.maximum(cap, 1e-9) - 1, -1, 2) * rtt,
        ),
        (
            "1 + queueing delay / (2 base RTT), the utility's delay term",
            ("queue_delay_mean_ns", "base_rtt_ns"),
            np.log(1 + 0.5 * qd / np.maximum(rtt, 1e-9)),
        ),
        (
            "queue / (queue + buffer / 2)",
            ("queue_pkts", "buffer_pkts"),
            queue / (queue + 0.5 * buffer + 1e-9),
        ),
        (
            "sqrt(queueing delay * base RTT) / (1 + 100 drop fraction)",
            ("queue_delay_mean_ns", "base_rtt_ns", "drop_fraction"),
            np.sqrt(qd * rtt) / (1 + 100 * get("drop_fraction")),
        ),
    ]
    return [
        *(("off-grammar", t, v, z[:, None], s) for t, v, z in off for s in NOISE[:2]),
        ("null", "white noise", (), rng.normal(size=(n, 2)), 0.0),
        ("null", "one random number per link", (), rng.normal(size=link.max() + 1)[link, None], 0),
        (
            "degenerate",
            "BDP beside a latent that hardly moves",
            ("bdp_pkts",),
            np.concatenate([live, 1e-7 * rng.normal(size=(n, 1))], 1),
            0.0,
        ),
        (
            "degenerate",
            "BDP held three times",
            ("bdp_pkts",),
            np.concatenate([live, -live, np.tanh(live - live.mean())], 1),
            0.3,
        ),
        *(
            ("sender", f"the sender's {m}", (), get(m)[:, None], 0.0)
            for m in ("sender_srtt", "max_960ms(sender_delivery_rate)", "sender_loss_rate")
        ),
    ]


def named(features: F64, concepts: F64) -> float:
    """The worst concept's best absolute rank correlation with one feature."""
    f, c = ground._places(features), ground._places(concepts)
    return float(np.abs(f.T @ c / len(f)).max(0).min())


@dataclass(frozen=True, slots=True)
class Planted:
    """One case's latents and its concepts, on the fit and the score rows."""

    z: tuple[F64, F64]
    concepts: tuple[F64, F64]  # [n, c]; no column for a case without concepts
    limit: float


def regime(lib: Library, p: Planted, allowed: list[int], *, formulas: bool) -> dict[str, Any]:
    """The selection on ``allowed`` features of ``lib``, scored against the
    planted concepts."""
    m = Moments.of(lib, *p.z)
    order = ground.pruned(lib, m, ground.greedy(lib, m, allowed, STEPS))
    c = max(p.concepts[0].shape[1], 1)

    def cols(n: int) -> list[int]:
        return ground._columns(lib, order[:n])[1:]

    def linear(n: int) -> float:
        return float(m.r2([0, *cols(n)]).mean())

    def span(n: int) -> float | None:
        if not p.concepts[0].shape[1]:
            return None
        r = ground.r2(lib.fit[:, cols(n)], p.concepts[0], lib.score[:, cols(n)], p.concepts[1])
        return float(np.nanmin(r))

    out: dict[str, Any] = {
        "features": [lib.names[f] for f in order[:PICK]],
        "linear": [linear(c), linear(PICK)],
        "span": [span(c), span(PICK)],
        "named": named(lib.score[:, cols(PICK)], p.concepts[1]) if p.concepts[0].shape[1] else None,
    }
    if out["linear"][1] < p.limit - ENOUGH:
        everything = ground._columns(lib, allowed)[1:]
        out["flexible"] = [
            ground.flexible(lib.fit[:, q], p.z[0], lib.score[:, q], p.z[1])
            for q in (cols(PICK), everything)
        ]
    if formulas:
        wider, made = ground.formulas(lib, p.z, [lib.names[f] for f in order[:PICK]])
        w = Moments.of(wider, *p.z)
        index = {n: i for i, n in enumerate(wider.names)}
        again = [index[lib.names[f]] for f in allowed if lib.names[f] in index]
        again += [index[n] for n in made if n in index]
        order2 = ground.pruned(wider, w, ground.greedy(wider, w, again, STEPS))
        q = ground._columns(wider, order2[:PICK])[1:]
        out["formulas"] = {
            "features": [wider.names[f] for f in order2[:PICK]],
            "linear": [float(w.r2(ground._columns(wider, order2[:n])).mean()) for n in (c, PICK)],
            "named": named(wider.score[:, q], p.concepts[1]) if p.concepts[0].shape[1] else None,
        }
    return out


def run(out: Path) -> None:
    names, variables, fr = ground._rows([out], "fit", ground.FIT)
    _, _, sr = ground._rows([out], "score", ground.SCORE)
    lib = Library.of(names, variables, fr.x, sr.x)
    col = {n: i for i, n in enumerate(names)}
    mean = np.nan_to_num(np.nanmean(fr.x, axis=0))
    std = np.nanstd(fr.x, axis=0) + 1e-12

    def concept(x: F64, picks: tuple[str, ...]) -> F64:
        i = [col[n] for n in picks]
        return np.nan_to_num((x[:, i] - mean[i]) / std[i])

    def get(x: F64) -> Callable[[str], F64]:
        return lambda n: _value(np.nan_to_num(x[:, col[n]]))

    truth = [f for f, v in enumerate(lib.variable) if not v.startswith("sender_")]
    # A library of the stored labels alone: no composite to pick or to repeat.
    keep = [i for i, n in enumerate(names[:variables]) if _is_label(n)]
    at = [op * variables + i for op in range(len(names) // variables) for i in keep]
    bare = Library.of([names[i] for i in at], len(keep), fr.x[:, at], sr.x[:, at])
    plan: list[tuple[str, str, tuple[str, ...], tuple[F64, F64], float]] = []
    for i, c in enumerate(cases(names, variables)):
        z = tuple(
            c.make(concept(x, c.concepts), np.random.default_rng(100 + i)) for x in (fr.x, sr.x)
        )
        plan.append((c.group, c.title, c.concepts, (z[0], z[1]), c.noise))
    for (g, t, v, zf, s), (_, _, _, zs, _) in zip(
        custom(get(fr.x)), custom(get(sr.x)), strict=True
    ):
        plan.append((g, t, v, (zf, zs), s))
    results = []
    for i, (group, title, concepts, (zf, zs), noise) in enumerate(plan):
        rng = np.random.default_rng(1000 + i)
        scale = zf.std(0)
        z = (
            zf + noise * scale * rng.normal(size=zf.shape),
            zs + noise * scale * rng.normal(size=zs.shape),
        )
        # What the noise leaves: each latent's rank correlation with itself before the noise.
        limit = float(np.mean(np.mean(ground._places(zs) * ground._places(z[1]), axis=0) ** 2))
        p = Planted(z, (concept(fr.x, concepts), concept(sr.x, concepts)), limit)
        row = {
            "group": group,
            "case": title,
            "concepts": concepts,
            "noise": noise,
            "limit": limit,
            "curated": regime(lib, p, truth, formulas=False),
            "labels": regime(bare, p, list(range(len(bare.names))), formulas=True),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    result = {"fit": len(fr.x), "score": len(sr.x), "cases": results}
    (out / "stress.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "summary.md").write_text(summary(result))


def _is_label(variable: str) -> bool:
    return not variable.startswith("sender_") and ground._label(variable) not in lcfs.COMPOSITES


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def summary(result: dict[str, Any]) -> str:
    lines = [
        f"Planted latents through the selection: {result['fit']} decisions to fit, "
        f"{result['score']} to score. Columns are explained in ground_stress.py.\n\n"
        "## Verdicts\n\n"
        f"picked: the first c features span every planted concept (R^2 of {SPANNED} or "
        f"more). named: some picked feature is each concept up to a monotone change (rank "
        f"correlation of {NAMED} or more). enough: the fit on {PICK} features, linear or "
        f"flexible, is within {ENOUGH} of the limit.\n\n"
        "| group | cases | curated: picked | curated: enough | labels: named | "
        "labels with formulas: named | labels with formulas: enough |\n"
        "|---|---|---|---|---|---|---|\n"
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in result["cases"]:
        groups.setdefault(c["group"], []).append(c)

    def enough(c: dict[str, Any], r: dict[str, Any]) -> bool:
        return max(r["linear"][1], *r.get("flexible", [-1.0])) >= c["limit"] - ENOUGH

    def count(rows: list[dict[str, Any]], ok: Callable[[dict[str, Any]], bool | None]) -> str:
        v = [ok(c) for c in rows]
        v = [x for x in v if x is not None]
        return f"{sum(v)} of {len(v)}" if v else "n/a"

    def over(key: str, least: float) -> Callable[[dict[str, Any]], bool | None]:
        def ok(r: dict[str, Any]) -> bool | None:
            v = r[key][0] if isinstance(r[key], list) else r[key]
            return None if v is None else v >= least

        return ok

    for g, rows in groups.items():
        lines.append(
            f"| {g} | {len(rows)} | "
            + " | ".join(
                [
                    count(rows, lambda c: over("span", SPANNED)(c["curated"])),
                    count(rows, lambda c: enough(c, c["curated"])),
                    count(rows, lambda c: over("named", NAMED)(c["labels"])),
                    count(rows, lambda c: over("named", NAMED)(c["labels"]["formulas"])),
                    count(
                        rows,
                        lambda c: (
                            max(c["labels"]["formulas"]["linear"][1], enough(c, c["labels"]))
                            >= c["limit"] - ENOUGH
                        ),
                    ),
                ]
            )
            + " |\n"
        )
    for g, rows in groups.items():
        lines.append(
            f"\n## {g}\n\n| case | noise | limit | regime | linear at c, {PICK} | flexible on "
            f"{PICK}, all | span at c, {PICK} | named | features |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for c in rows:
            for key, r in (
                ("curated", c["curated"]),
                ("labels", c["labels"]),
                ("labels with formulas", c["labels"]["formulas"]),
            ):
                lines.append(
                    f"| {c['case']} | {c['noise']} | {c['limit']:.3f} | {key} | "
                    + ", ".join(_f(v) for v in r["linear"])
                    + " | "
                    + (", ".join(_f(v) for v in r["flexible"]) if "flexible" in r else "")
                    + " | "
                    + (", ".join(_f(v) for v in r["span"]) if "span" in r else "")
                    + f" | {_f(r['named'])} | {', '.join(r['features'][:4])} |\n"
                )
    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)
    ro = sub.add_parser("rollouts")
    ro.add_argument("out", type=Path)
    ro.add_argument("--fit", type=Path, required=True)
    ro.add_argument("--score", type=Path, required=True)
    ro.add_argument("--build", type=Path, required=True)
    ro.add_argument("--workers", type=int, default=1)
    ro.add_argument("--links", type=int, default=96)
    sub.add_parser("run").add_argument("out", type=Path)
    a = p.parse_args()
    if a.verb == "rollouts":
        ground.drive(
            a.out,
            [StockCC(name=t) for t in TEACHERS],
            30_000,
            a.fit,
            a.score,
            UmlBuild.at(a.build),
            workers=a.workers,
            links=a.links,
        )
    else:
        run(a.out)


if __name__ == "__main__":
    main()
