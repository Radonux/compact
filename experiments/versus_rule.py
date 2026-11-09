"""Every controller minus a reference policy, paired by link, per campaign
and per condition.

A controller's score on a link is each episode's mean log utility per 10 ms
step, averaged over the link's episodes in that condition (or over all the
campaign's conditions, for the campaign's row); the difference is taken per
link and its mean shown with a 95% bootstrap interval over links, as
experiments/net2_ppo_gate_report.py does for the five families. A UDP flow
at a share of each link's capacity has a different rate on each link, so
conditions are grouped with their UDP rates left out.

Usage: python experiments/versus_rule.py OUT REFERENCE
Reads OUT/c-* and OUT/aliases.json; writes OUT/versus-REFERENCE.md.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from net2_ppo_gate_report import _interval, difference, load  # pyright: ignore[reportPrivateUsage]


def group(condition: str) -> str:
    """A condition's label with its UDP rates left out."""
    return re.sub(r"UDP [0-9.e+-]+ Mbit/s", "UDP", condition)


def main(out: Path, reference: str) -> None:
    scores, failed = load(out)
    rng = np.random.default_rng(0)
    names = sorted({n for camp in scores.values() for n in camp} - {reference})
    lines = [
        f"Each controller minus {reference}, paired by link: the mean over links with its 95%",
        "interval, per campaign (all its conditions) and per condition. A * marks a controller",
        "with a failed episode.",
        "",
        "| campaign | condition | "
        + " | ".join(f"{n}{' *' if n in failed else ''}" for n in names)
        + " |",
        "|" + "---|" * (len(names) + 2),
    ]
    for camp, by_ctl in scores.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for c in sorted({c for ctl in by_ctl.values() for c in ctl}):
            groups[group(c)].append(c)
        rows = [("all", ())] + [(g, tuple(cs)) for g, cs in groups.items()] * (len(groups) > 1)
        for label, sel in rows:
            cells = []
            for n in names:
                d = difference(by_ctl, [n], reference, sel) if n in by_ctl else []
                cells.append(str(_interval(d, rng)) if d else "")
            short = label if len(label) <= 60 else label[:57] + "..."
            lines.append(f"| {camp} | {short} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out / f"versus-{reference}.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
