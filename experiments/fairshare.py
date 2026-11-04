"""Write campaigns where the number of flows on the bottleneck is constant
and known: on each of one split's links, the observed flow alone (20 s),
with a Cubic flow from 0 s (60 s), and as 2 or 4 copies starting together
(30 s). A policy can be given per number of flows, so a rule that is told
the number (experiments/share_rule_export.py --flows) runs with the right
one in each episode; all of a name's policies score as one controller.

Usage: python experiments/fairshare.py OUT --split val --seed 11 --stock bbr3
           --policy NAME=PATH            (the same policy for every episode)
           --policy NAME@N=PATH          (the policy for episodes with N flows)

Plans OUT/c-alone-20, OUT/c-cubic-first-60 and OUT/c-copies-30s (for
`expliccit campaign run`) and writes OUT/aliases.json (agent:SHA12 -> NAME).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from heldout import controllers, split_links

from expliccit_env import campaign
from expliccit_env.spec import EpisodeSpec, PathSpec

# (campaign, seconds, flows, cross flows given the observed controller)
CONDITIONS = (
    ("alone-20", 20, 1, lambda own: []),
    ("cubic-first-60", 60, 2, lambda own: ["cubic"]),
    ("copies-30s", 30, 2, lambda own: [own]),
    ("copies-30s", 30, 4, lambda own: [own] * 3),
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stock", default="bbr3")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH or NAME@N=PATH")
    a = p.parse_args()
    stock, aliases = controllers(a.stock, [])
    # name -> flows (None for every number) -> agent cc
    by_name: dict[str, dict[int | None, dict[str, Any]]] = {}
    for item in a.policy:
        key, path = item.split("=", 1)
        name, _, n = key.partition("@")
        (cc,), alias = controllers("", [f"{name}={path}"])
        # Episodes are scored by policy hash, so one file under two names
        # would merge two controllers.
        if clash := {h: aliases[h] for h in alias if aliases.get(h, name) != name}:
            raise SystemExit(f"{name}: same policy as {clash}")
        aliases |= alias
        by_name.setdefault(name, {})[int(n) if n else None] = cc
    links = list(split_links(a.split).values())
    rng = np.random.default_rng(a.seed)
    plans: dict[str, list[EpisodeSpec]] = {}
    for link in links:
        path = PathSpec.model_validate(link)
        for camp, seconds, flows, cross in CONDITIONS:
            ccs = [(c, c["name"]) for c in stock]
            for name, per in by_name.items():
                cc = per.get(flows, per.get(None))
                if cc is None:
                    raise SystemExit(f"{name} has no policy for {flows} flows")
                ccs.append((cc, cc))
            for cc, own in ccs:
                plans.setdefault(camp, []).append(
                    EpisodeSpec.model_validate(
                        {
                            "path": path.model_dump(),
                            "cc": cc,
                            "cross": [{"kind": "tcp", "cc": c, "start_ms": 0} for c in cross(own)],
                            "duration_ms": seconds * 1000,
                            "seed": int(rng.integers(0, 2**63)),
                            "steps": True,
                        }
                    )
                )
    for camp, episodes in plans.items():
        campaign.write_episodes(a.out / f"c-{camp}", episodes, {"name": f"c-{camp}"})
        print(f"c-{camp}: {len(episodes)} episodes")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")


if __name__ == "__main__":
    main()
