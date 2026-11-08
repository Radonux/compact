"""The gate a learned net2 must pass on validation before its one run on
the test links: each candidate's policies (one per training seed, named
ppo-CANDIDATE-sSEED, from `expliccit train ppo` by PPO, TD3+BC or DAgger)
against the fixed rule (the policy named RULE, by default rule-v5, the
fixed rule on net1 v5), paired by link, on every validation family and as
copies.

The rule, fixed before the first gate ran (experiments/net2-ppo-gate.json):

- A controller's score on a link is the mean, over the link's conditions
  in a campaign, of each episode's mean log utility per 10 ms step. A
  candidate's difference on a link is the mean over its seeds of the
  policy's score minus the rule's.
- Kept: the candidate with the highest five-family mean of that difference
  (training-20, training-60, new-20, deep-long-20, cubic-first-60; the
  copies are shown apart). A candidate with a failed episode is never kept.
- The gate passes when the kept candidate beats the rule in the competition
  families, cubic-first-60 and new-20's competition conditions pooled
  (Cubic from 0 s, BBRv1 from 4 s, two Cubic flows), each with its 95%
  interval over links above 0. It must also be at or above the rule on the
  five-family mean, and not worse with copies beyond its interval (the
  interval's upper end at or above 0). Otherwise net2 stays the fixed rule
  and the share rule.

Usage: python experiments/net2_ppo_gate_report.py OUT [RULE]
Writes OUT/summary.md and OUT/gate.json.
"""

import dataclasses
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expliccit_env.train import score
from expliccit_env.train.spec import Net2Spec

RULE = "rule-v5"  # the default fixed rule
FAMILIES = ("training-20", "training-60", "new-20", "deep-long-20", "cubic-first-60")
COPIES = "copies-30s"
COMPETITION = ("cubic from 0 s", "bbr from 4 s", "cubic from 2 s, cubic from 10 s")
POLICY = re.compile(r"^ppo-(?P<cand>.+)-s(?P<seed>[0-9]+)$")
RESAMPLES = 4000

# campaign -> controller -> condition -> link -> score
type Scores = dict[str, dict[str, dict[str, dict[str, float]]]]


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    lo: float
    hi: float
    links: int

    def __str__(self) -> str:
        return f"{self.mean:+.3f} ({self.lo:+.3f} to {self.hi:+.3f})"


@dataclass(frozen=True, slots=True)
class Candidate:
    families: dict[str, Interval]
    five_family_mean: float
    competition_new: Interval  # new-20's competition conditions, pooled
    copies: Interval | None
    per_seed: dict[str, float]  # policy name -> its five-family mean
    failed: bool

    def passes(self) -> bool:
        return (
            self.families["cubic-first-60"].lo > 0
            and self.competition_new.lo > 0
            and self.five_family_mean >= 0
            and (self.copies is None or self.copies.hi >= 0)
        )


def _interval(d: list[float], rng: np.random.Generator) -> Interval:
    x = np.asarray(d, np.float64)
    boot = x[rng.integers(0, x.size, (RESAMPLES, x.size))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return Interval(float(x.mean()), float(lo), float(hi), x.size)


def difference(
    by_cond: dict[str, dict[str, dict[str, float]]],
    seeds: list[str],
    rule_name: str,
    conditions: tuple[str, ...] = (),
) -> list[float]:
    """Per link, the mean over seeds of policy minus rule, each the mean
    over ``conditions`` (all of the campaign's when empty)."""

    def link_scores(ctl: str) -> dict[str, float]:
        conds = [c for c in by_cond.get(ctl, {}) if not conditions or c in conditions]
        links = {k for c in conds for k in by_cond[ctl][c]}
        return {
            k: float(np.mean([by_cond[ctl][c][k] for c in conds if k in by_cond[ctl][c]]))
            for k in links
        }

    rule = link_scores(rule_name)
    per_seed = [link_scores(s) for s in seeds]
    shared = sorted(set(rule).intersection(*per_seed))
    return [float(np.mean([p[k] - rule[k] for p in per_seed])) for k in shared]


def decide(
    scores: Scores, failed: set[str], rule: str = RULE
) -> tuple[dict[str, Candidate], str | None]:
    """Every candidate's differences, and the one kept (the rule above)."""
    rng = np.random.default_rng(0)
    groups: dict[str, list[str]] = defaultdict(list)
    for n in sorted({n for camp in scores.values() for n in camp}):
        if m := POLICY.match(n):
            groups[m["cand"]].append(n)
    cands: dict[str, Candidate] = {}
    for cand, seeds in sorted(groups.items()):
        fam = {f: _interval(difference(scores[f], seeds, rule), rng) for f in FAMILIES}
        cands[cand] = Candidate(
            families=fam,
            five_family_mean=float(np.mean([v.mean for v in fam.values()])),
            competition_new=_interval(difference(scores["new-20"], seeds, rule, COMPETITION), rng),
            copies=_interval(difference(scores[COPIES], seeds, rule), rng)
            if COPIES in scores
            else None,
            per_seed={
                s: float(np.mean([np.mean(difference(scores[f], [s], rule)) for f in FAMILIES]))
                for s in seeds
            },
            failed=any(s in failed for s in seeds),
        )
    ok = [c for c, r in cands.items() if not r.failed]
    return cands, max(ok, key=lambda c: cands[c].five_family_mean) if ok else None


def load(out: Path) -> tuple[Scores, set[str]]:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    scores: Scores = {}
    failed: set[str] = set()
    for c in sorted(out.glob("c-*")):
        per: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        eps = score.episode_scores([c], Net2Spec(reward="log_utility"))
        for (ctl, link, cond), v in eps.items():
            per[aliases.get(ctl, ctl)][cond][link] = float(np.mean(v))
        scores[c.name[2:]] = per
        failed |= {aliases.get(k, k) for k, n in score.failures([c]).items() if n}
    return scores, failed


def markdown(cands: dict[str, Candidate], kept: str | None, rule: str = RULE) -> str:
    head = [*FAMILIES, "five-family mean", "new-20, competition", "copies", "per seed"]
    lines = [
        f"Learned net2 against the fixed rule ({rule}), validation links: per link the mean over",
        "seeds of policy minus rule, then the mean over links with its 95% interval.",
        "A * marks a candidate with a failed episode.",
        "",
        "| candidate | " + " | ".join(head) + " |",
        "|" + "---|" * (len(head) + 1),
    ]
    for c, r in cands.items():
        seeds = ", ".join(f"{s.rsplit('-s', 1)[1]}: {v:+.3f}" for s, v in r.per_seed.items())
        cells = [str(r.families[f]) for f in FAMILIES]
        cells += [f"{r.five_family_mean:+.3f}", str(r.competition_new), str(r.copies or "-"), seeds]
        lines.append(f"| {c}{' *' if r.failed else ''} | " + " | ".join(cells) + " |")
    passes = kept is not None and cands[kept].passes()
    verdict = "passes" if passes else "does not pass: net2 stays the fixed rule and the share rule"
    lines += ["", f"Kept: {kept}. The gate {verdict}."]
    return "\n".join(lines) + "\n"


def main(out: Path, rule: str = RULE) -> None:
    cands, kept = decide(*load(out), rule)
    text = markdown(cands, kept, rule)
    (out / "summary.md").write_text(text)
    gate = {
        "kept": kept,
        "passes": kept is not None and cands[kept].passes(),
        "candidates": {c: dataclasses.asdict(r) for c, r in cands.items()},
    }
    (out / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(text, end="")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else RULE)
