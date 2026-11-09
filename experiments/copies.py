"""Write a campaign of copies: on each of one split's links, n flows of the
same controller start together, for each controller and each n. The
observed flow is one of them, so its labels and log utility are those of
any copy; every flow's bytes through the bottleneck are in the episodes
table (flow_bytes), for fairness.

Usage: python experiments/copies.py OUT --split val --duration-ms 30000 --seed 11 --n 2 4
           --stock bbr3,cubic --policy NAME=PATH[@MODE] ...

Plans OUT/c-copies-{seconds}s (for `expliccit campaign run`) and writes
OUT/aliases.json (agent:SHA12 -> NAME).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from heldout import controllers, split_links

from expliccit_env import campaign
from expliccit_env.spec import EpisodeSpec, PathSpec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--duration-ms", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, nargs="+", required=True, help="flows per episode")
    p.add_argument("--stock", default="bbr3")
    p.add_argument(
        "--policy",
        action="append",
        default=[],
        help="NAME=PATH[@MODE] of an ONNX policy (mode hold, pure or cubic; default hold)",
    )
    a = p.parse_args()
    links = list(split_links(a.split).values())
    ccs, aliases = controllers(a.stock, a.policy)
    rng = np.random.default_rng(a.seed)
    episodes = []
    for link in links:
        path = PathSpec.model_validate(link)
        for cc in ccs:
            for n in a.n:
                own = cc["name"] if cc["kind"] == "stock" else cc
                episodes.append(
                    EpisodeSpec.model_validate(
                        {
                            "path": path.model_dump(),
                            "cc": cc,
                            "cross": [{"kind": "tcp", "cc": own, "start_ms": 0}] * (n - 1),
                            "duration_ms": a.duration_ms,
                            "seed": int(rng.integers(0, 2**63)),
                            "steps": True,
                        }
                    )
                )
    name = f"c-copies-{a.duration_ms // 1000}s"
    campaign.write_episodes(a.out / name, episodes, {"name": name})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(
        f"{len(episodes)} episodes: {len(links)} {a.split} links, {len(ccs)} controllers, n {a.n}"
    )


if __name__ == "__main__":
    main()
