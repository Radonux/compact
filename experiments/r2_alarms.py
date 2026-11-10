"""Score alarms that flag wrong net1 beliefs without ground truth (research item R2).

Usage: python experiments/r2_alarms.py OUT --net1 NAME=NET1.pt [--net1 ...]
           --calib DIR... --negatives DIR... [--positives NAME=DIR ...]
           [--stride N] [--max-episodes N]

Each alarm is a statistic of one step, computed from net1's distributions
(their means, in report units) and the sender's own measurements only. Its
episode statistic is the mean over the episode's steps where it is defined,
and it fires on an episode when that mean is above its threshold: the 95th
percentile over the calibration episodes (a nominal false-alarm rate of 5%),
set per alarm and per net1. The alarms, and the LCFs each implicates:

  I1   receive rate at most capacity times busy fraction:
       max(0, log(receive / (capacity * busy))); receive, capacity, busy
  I2   mean queueing delay equal to queue / capacity (1500-byte packets,
       12 ms per packet at 1 Mbit/s): |log(delay / (12 * queue / capacity))|,
       both floored at 1 ms; queue delay, queue, capacity
  I3   queue plus free buffer (the buffer size) constant over the episode:
       |log(sum / the episode's median sum)|; queue, free buffer
  H    the entropy of net1's distribution over its bins, over log(bins); its LCF
  C1   believed receive rate against the delivery rate measured three
       smoothed RTTs later (by the trace clock, in the same episode):
       |log(believed / measured)|; receive, capacity
  C2+  twice the believed one-way delay above the min RTT measured three
       smoothed RTTs later: max(0, log(2 * delay / min RTT)); one-way delay
  C2-  the reverse, max(0, log(min RTT / (2 * delay))); one-way delay

Rates floor at 0.1 Mbit/s and delays at 1 ms, so idle links and empty
queues give 0 rather than a division by zero.

Ground truth, for scoring only: net1 is wrong about an LCF in an episode
when the truth lies outside its central 90% set in more than half of the
steps where the truth is defined. The test is on bins, as net1 states bins
and as fit's coverage is: a step counts as the probability that fit's
randomized probability integral transform (uniform over the truth's bin)
falls outside the central 90%, so no draw is made. A value past the last
bin falls in the last bin, which takes everything above.

Sets: calibration is the --calib campaigns' validation links, the negatives
the --negatives campaigns' test links (split by link with net1 v4's split:
seed 1, 15% each), both without the episodes that have a Cubic flow from
0 s or a held-out observed controller (BBRv3, Westwood, threshold-0.3).
Those test-link episodes are the positive families "cubic from 0 s" and
"held-out controller". Each --positives campaign is a family of its own,
whole. --max-episodes caps each set (the first episodes in shard order).

--fault NAME=LCF*F (or LCF=V; several joined by commas) says family NAME
ran a policy whose net1 had that injected fault (rule_export.py --fault):
its beliefs are recomputed with the same fault, so the alarms see what the
policy believed.

Detection rate of an alarm for an LCF it implicates: the share of a family's
episodes wrong about the LCF where it fires. False-alarm rate: the share of
negatives right about the LCF where it fires. AUC: the episode statistic, a
family's wrong episodes against the right negatives. "any" fires when an
alarm implicating the LCF fires; its statistic is the largest ratio of an
alarm's statistic to its threshold. Writes OUT/summary.md and OUT/r2.csv.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from net1_train import held

from expliccit_env.policy import OBS_COLUMNS
from expliccit_env.train import data, fit
from expliccit_env.train.models import BinNet1
from expliccit_env.train.spec import EXPLICCIT_LCFS

type F = npt.NDArray[np.float64]
type Ints = npt.NDArray[np.int64]

QD, Q, CAP, FREE, BUSY, OWD, RECV = range(7)
SHORT = ("queue delay", "queue", "capacity", "free buffer", "busy", "one-way delay", "receive")
T, SRTT, DELIVERY, MIN_RTT = (
    OBS_COLUMNS.index(c) for c in ("t_s", "srtt_x100ms", "delivery_rate_x100mbps", "min_rtt_ms")
)
RATE_FLOOR = 0.1  # Mbit/s
DELAY_FLOOR = 1.0  # ms
RTTS_LATER = 3
TAIL = 0.05  # each side of the central 90% set
NOMINAL = 95  # percentile of the calibration episodes
BATCH = 32_768
ALARMS: dict[str, tuple[int, ...]] = {
    "I1": (RECV, CAP, BUSY),
    "I2": (QD, Q, CAP),
    "I3": (Q, FREE),
    **{f"H {SHORT[j]}": (j,) for j in range(7)},
    "C1": (RECV, CAP),
    "C2+": (OWD,),
    "C2-": (OWD,),
}
CALIB, NEG, CUBIC0, HELD = "calibration", "negatives", "cubic from 0 s", "held-out controller"


def beliefs(
    net1: fit.Net1, obs: npt.NDArray[np.float32], y: F, fault: torch.nn.Module | None = None
) -> tuple[F, F, F]:
    """Per row and LCF: net1's mean, its normalized entropy, and the
    probability that the truth is inside its central 90% set (NaN where the
    truth is undefined): the share of the truth's bin, in CDF terms, that
    lies between the 5% and 95% levels. ``fault`` moves the distributions'
    mass first, as in the faulted policy."""
    assert isinstance(net1.net, BinNet1)
    b = net1.net.bounds.numpy().astype(np.float64)
    mid = (b[:, :-1] + b[:, 1:]) / 2
    parts: list[tuple[F, F, F]] = []
    for i in range(0, len(obs), BATCH):
        p = net1.distributions(obs[i : i + BATCH])
        if fault is not None:
            with torch.no_grad():
                p = fault(torch.from_numpy(p.reshape(len(p), -1))).numpy().reshape(p.shape)
        p = p.astype(np.float64)
        yy = y[i : i + BATCH]
        mean = (p * mid).sum(axis=2)
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=2) / np.log(p.shape[2])
        k = np.stack(
            [np.searchsorted(b[j, 1:-1], np.nan_to_num(yy[:, j]), "right") for j in range(len(b))],
            axis=1,
        )[:, :, None]
        upto = np.take_along_axis(np.cumsum(p, axis=2), k, axis=2)[:, :, 0]
        below = upto - np.take_along_axis(p, k, axis=2)[:, :, 0]
        mass = upto - below
        overlap = np.clip(np.minimum(upto, 1 - TAIL) - np.maximum(below, TAIL), 0, None)
        point = ((below >= TAIL) & (below <= 1 - TAIL)).astype(np.float64)
        inside = np.where(mass > 1e-12, overlap / np.maximum(mass, 1e-12), point)
        parts.append((mean, ent, np.where(np.isfinite(yy), inside, np.nan)))
    mean, ent, inside = (np.concatenate(z) for z in zip(*parts, strict=True))
    return mean, ent, inside


def log_ratio(a: F, b: F, floor: float) -> F:
    return np.log(np.maximum(a, floor) / np.maximum(b, floor))


def step_stats(m: F, ent: F, obs: npt.NDArray[np.float32], seg: Ints) -> F:
    """[rows, alarms]: each alarm's statistic per step (NaN where undefined).
    ``seg`` numbers the episodes, rows in episode then time order."""
    o = obs.astype(np.float64)
    o[~np.isfinite(o)] = np.nan
    # The row three smoothed RTTs later in the same episode (srtt in 100 ms).
    key = seg * 1e4 + o[:, T]
    assert np.all(np.diff(key) >= 0), "rows must be in episode then time order"
    later = np.searchsorted(key, key + RTTS_LATER * 0.1 * o[:, SRTT], "left")
    j = np.minimum(later, len(key) - 1)
    ok = (later < len(key)) & (seg[j] == seg) & (o[:, SRTT] > 0)
    measured = np.where(ok, o[j, DELIVERY] * 100, np.nan)
    min_rtt = np.where(ok & (o[j, MIN_RTT] > 0), o[j, MIN_RTT], np.nan)
    total = m[:, Q] + m[:, FREE]
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    median = np.repeat(
        [np.median(x) for x in np.split(total, starts[1:])], np.diff([*starts, len(seg)])
    )
    base = 2 * m[:, OWD]
    stat = {
        "I1": np.maximum(0, log_ratio(m[:, RECV], m[:, CAP] * m[:, BUSY], RATE_FLOOR)),
        "I2": np.abs(
            log_ratio(m[:, QD], 12 * m[:, Q] / np.maximum(m[:, CAP], RATE_FLOOR), DELAY_FLOOR)
        ),
        "I3": np.abs(log_ratio(total, median, 1.0)),
        **{f"H {SHORT[j]}": ent[:, j] for j in range(7)},
        "C1": np.abs(log_ratio(m[:, RECV], measured, RATE_FLOOR)),
        "C2+": np.maximum(0, log_ratio(base, min_rtt, DELAY_FLOOR)),
        "C2-": np.maximum(0, log_ratio(min_rtt, base, DELAY_FLOOR)),
    }
    return np.stack([stat[a] for a in ALARMS], axis=1)


def episode_means(x: F, seg: Ints, n: int) -> F:
    """[n, columns]: each episode's mean of each column over its finite rows."""
    out = np.full((n, x.shape[1]), np.nan)
    for c in range(x.shape[1]):
        ok = np.isfinite(x[:, c])
        count = np.bincount(seg[ok], minlength=n)
        total = np.bincount(seg[ok], x[ok, c], minlength=n)
        out[:, c] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def cubic_from_start(spec_json: str) -> bool:
    cross: list[dict[str, Any]] = json.loads(spec_json).get("cross") or []
    return any(c["kind"] == "tcp" and c["cc"] == "cubic" and c["start_ms"] == 0 for c in cross)


def member(feeds: set[str], split: str, cubic0: bool, held_cc: bool) -> set[str]:
    """The sets an episode belongs to, of those its campaign feeds."""
    rule = {
        CALIB: split == "val" and not (cubic0 or held_cc),
        NEG: split == "test" and not (cubic0 or held_cc),
        CUBIC0: split == "test" and cubic0,
        HELD: split == "test" and held_cc,
    }
    return {s for s in feeds if rule.get(s, True)}


def collect(
    a: argparse.Namespace,
    nets: dict[str, fit.Net1],
    feeds: dict[Path, set[str]],
    faults: dict[str, dict[str, torch.nn.Module]],
) -> tuple[list[frozenset[str]], dict[str, F], dict[str, F]]:
    """Per episode: its sets, then per net1 its episode statistics [E, alarms]
    and whether it is wrong about each LCF [E, 7] (1, 0 or NaN)."""
    held_ccs, cap = set(held()), a.max_episodes or float("inf")
    count: Counter[str] = Counter()
    sets: list[frozenset[str]] = []
    stats: dict[str, list[F]] = {m: [] for m in nets}
    wrong: dict[str, list[F]] = {m: [] for m in nets}
    for shard, feed in feeds.items():
        if all(count[s] >= cap for s in feed):
            continue
        steps = data.load([shard], seed=1, val=0.15, test=0.15, split_by="link", stride=a.stride)
        t = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "spec_json", "controller"]
        )
        meta = {
            e: (s, c) for e, s, c in zip(*(t[k].to_pylist() for k in t.column_names), strict=True)
        }
        first = np.unique(steps.episode, return_index=True)[1]
        chosen: dict[int, frozenset[str]] = {}
        for e, row in enumerate(first):
            spec_json, ctl = meta[steps.episode_ids[e]]
            got = member(feed, str(steps.split[row]), cubic_from_start(spec_json), ctl in held_ccs)
            if got := {s for s in got if count[s] < cap}:
                count.update(got)
                chosen[e] = frozenset(got)
        if not chosen:
            continue
        eps = np.array(sorted(chosen))
        rows = np.isin(steps.episode, eps)
        seg = np.searchsorted(eps, steps.episode[rows])
        obs, y = steps.obs[rows], fit.targets(steps, EXPLICCIT_LCFS)[rows].astype(np.float64)
        faulted = [f for f in feed if f in faults]
        if len(faulted) > 1:
            raise SystemExit(f"{shard} feeds two faulted families {faulted}")
        for name, net1 in nets.items():
            fault = faults[faulted[0]][name] if faulted else None
            mean, ent, inside = beliefs(net1, obs, y, fault)
            stats[name].append(episode_means(step_stats(mean, ent, obs, seg), seg, len(eps)))
            w = episode_means(1 - inside, seg, len(eps))
            wrong[name].append(np.where(np.isnan(w), np.nan, w > 0.5))
        sets += [chosen[e] for e in eps]
        print(f"{shard}: {len(eps)} episodes; so far {dict(count)}", flush=True)
    return (
        sets,
        {m: np.concatenate(v) for m, v in stats.items()},
        {m: np.concatenate(v) for m, v in wrong.items()},
    )


def auc(pos: F, neg: F) -> float:
    if not (pos.size and neg.size):
        return float("nan")
    labels = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    return float(roc_auc_score(labels, np.r_[pos, neg]))


def score(
    sets: list[frozenset[str]], stats: F, wrong: F, families: list[str]
) -> list[dict[str, Any]]:
    """One row per LCF, alarm implicating it (and "any") and set."""
    member_of = {s: np.array([s in x for x in sets]) for s in (CALIB, NEG, *families)}
    if not member_of[CALIB].any():
        raise SystemExit("no calibration episodes")
    thr = np.array(
        [np.nanpercentile(stats[member_of[CALIB], i], NOMINAL) for i in range(len(ALARMS))]
    )
    ratio = stats / np.maximum(thr, 1e-12)
    out: list[dict[str, Any]] = []
    for j, lcf in enumerate(EXPLICCIT_LCFS):
        cols = [i for i, imp in enumerate(ALARMS.values()) if j in imp]
        alarms = [(list(ALARMS)[i], ratio[:, i], thr[i]) for i in cols]
        alarms.append(("any", np.fmax.reduce(ratio[:, cols], axis=1), float("nan")))
        for name, r, threshold in alarms:
            fires, defined = r > 1, np.isfinite(r)
            right_neg = member_of[NEG] & defined & (wrong[:, j] == 0)
            for s in (NEG, *families):
                w = member_of[s] & defined & (wrong[:, j] == 1)
                ok = member_of[s] & defined & (wrong[:, j] == 0)
                out.append(
                    {
                        "lcf": lcf,
                        "alarm": name,
                        "set": s,
                        "threshold": threshold,
                        "n_wrong": int(w.sum()),
                        "fired_wrong": int((w & fires).sum()),
                        "n_right": int(ok.sum()),
                        "fired_right": int((ok & fires).sum()),
                        "auc": auc(r[w], r[right_neg]),
                    }
                )
    return out


def rate(fired: int, n: int, auc: float | None = None) -> str:
    if not n:
        return "(n=0)"
    return f"{fired / n:.2f} (n={n})" + ("" if auc is None else f" AUC {auc:.2f}")


def summary(
    sets: list[frozenset[str]],
    wrong: dict[str, F],
    rows: dict[str, list[dict[str, Any]]],
    families: list[str],
    note: str,
) -> str:
    models = list(rows)
    lines = [
        "# R2: alarms on wrong net1 beliefs, without ground truth",
        "",
        note,
        "",
        "```text",
        __doc__[__doc__.index("  I1") : __doc__.index("Ground truth")].rstrip(),
        "```",
        "",
        "Episodes per set, and the share of them each net1 is wrong about, per LCF"
        f" ({' / '.join(models)}; wrong: the truth outside the central 90% set in"
        " more than half of the episode's steps):",
        "",
        "| set | episodes | " + " | ".join(SHORT) + " |",
        "|" + "---|" * (len(SHORT) + 2),
    ]
    for s in (CALIB, NEG, *families):
        m = np.array([s in x for x in sets])
        cells = [" / ".join(f"{np.nanmean(wrong[k][m, j]):.2f}" for k in models) for j in range(7)]
        lines.append(f"| {s} | {int(m.sum())} | " + " | ".join(cells) + " |")
    for model, rs in rows.items():
        lines += [
            "",
            f"## net1 {model}",
            "",
            "Negatives: false-alarm rate among the negatives right about the LCF. Each"
            " family: detection rate among its episodes wrong about the LCF, then the"
            " AUC of the statistic, those episodes against the right negatives.",
            "",
            "| LCF | alarm | threshold | negatives | " + " | ".join(families) + " |",
            "|" + "---|" * (len(families) + 4),
        ]
        by = {(r["lcf"], r["alarm"], r["set"]): r for r in rs}
        for lcf, short in zip(EXPLICCIT_LCFS, SHORT, strict=True):
            for alarm in dict.fromkeys(r["alarm"] for r in rs if r["lcf"] == lcf):
                neg = by[lcf, alarm, NEG]
                cells = [
                    rate(f["fired_wrong"], f["n_wrong"], f["auc"])
                    for f in (by[lcf, alarm, s] for s in families)
                ]
                thr = "" if alarm == "any" else f"{neg['threshold']:.3g}"
                lines.append(
                    f"| {short} | {alarm} | {thr} | {rate(neg['fired_right'], neg['n_right'])} | "
                    + " | ".join(cells)
                    + " |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--net1", action="append", required=True, help="NAME=NET1.pt")
    p.add_argument("--calib", type=Path, nargs="+", required=True)
    p.add_argument("--negatives", type=Path, nargs="+", required=True)
    p.add_argument("--positives", action="append", default=[], help="NAME=DIR, a family whole")
    p.add_argument("--stride", type=int, default=1, help="keep steps whose number it divides")
    p.add_argument("--max-episodes", type=int, help="at most this many episodes per set")
    p.add_argument("--fault", action="append", default=[], help="NAME=LCF*F[,LCF*F]")
    a = p.parse_args()
    t0 = time.time()
    nets = {k: fit.load_net1(Path(v))[0] for k, v in (x.split("=", 1) for x in a.net1)}
    if bad := [k for k, n in nets.items() if n.lcfs != EXPLICCIT_LCFS]:
        raise SystemExit(f"net1 {bad} does not report expliCCit's seven LCFs")
    positives = [x.split("=", 1) for x in a.positives]
    families = [CUBIC0, HELD, *(name for name, _ in positives)]
    if len(set(families)) < len(families) or {CALIB, NEG} & set(families):
        raise SystemExit(f"family names must be distinct and not {CALIB} or {NEG}")
    feeds: dict[Path, set[str]] = {}
    for dirs, names in (
        (a.calib, {CALIB}),
        (a.negatives, {NEG, CUBIC0, HELD}),
        *(([Path(d)], {name}) for name, d in positives),
    ):
        for d in dirs:
            found = sorted({f.parent for f in d.rglob("steps.parquet")})
            if not found:
                raise SystemExit(f"no steps.parquet under {d}")
            for shard in found:
                feeds.setdefault(shard, set()).update(names)
    faults: dict[str, dict[str, torch.nn.Module]] = {}
    for x in a.fault:
        fam, texts = x.split("=", 1)
        if fam not in {name for name, _ in positives}:
            raise SystemExit(f"--fault {x}: {fam} is not a --positives family")
        faults[fam] = {}
        for m, net1 in nets.items():
            spec = fit.parse_faults(texts.split(","), net1.lcfs)
            faults[fam][m] = net1.kind.fault(
                [spec[n][1] if spec.get(n, ("", 0.0))[0] == "scale" else 1.0 for n in net1.lcfs],
                [spec[n][1] if spec.get(n, ("", 0.0))[0] == "pin" else 0.0 for n in net1.lcfs],
                [spec.get(n, ("", 0.0))[0] == "pin" for n in net1.lcfs],
            )
    sets, stats, wrong = collect(a, nets, feeds, faults)
    rows = {m: score(sets, stats[m], wrong[m], families) for m in nets}
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "r2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["model", *rows[next(iter(rows))][0]])
        w.writeheader()
        w.writerows({"model": m, **r} for m, rs in rows.items() for r in rs)
    note = (
        f"Episode statistic: the mean of the per-step statistic; threshold: its {NOMINAL}th"
        f" percentile over the calibration episodes, per alarm and net1. Steps: those"
        f" stored whose step number --stride {a.stride} divides. Episodes per set:"
        f" {f'at most {a.max_episodes}' if a.max_episodes else 'all'}. Calibration and"
        " negatives are the validation and test links of the training campaign, without"
        " the two positive families drawn from its test links."
        " experiments/r2_alarms.py defines each alarm."
    )
    (a.out / "summary.md").write_text(summary(sets, wrong, rows, families, note))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
