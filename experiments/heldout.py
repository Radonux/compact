"""Write a campaign on the held-out links of examples/campaign-net1-varied.json.

The links are split as net1 v3 and every net2 run on it were: by link
(the path without its loss), seed 1, 15% validation, 15% test. The campaign
runs the given policies and stock controllers on one split's links under
that campaign's three conditions, every one with the same episode seeds.

Usage: python experiments/heldout.py OUT_DIR --split test --duration-ms 60000
           --seed 13 --stock all --policy NAME=PATH ...

Writes OUT_DIR/campaign.json and OUT_DIR/aliases.json (agent:SHA12 -> NAME,
for `expliccit score --alias`).
"""

import argparse
import hashlib
import json
from pathlib import Path

from expliccit_env.spec import STOCK_NAMES, CampaignSpec
from expliccit_env.train import data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/campaign-net1-varied.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--duration-ms", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stock", default="all", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of an ONNX policy")
    a = p.parse_args()
    source = json.loads(SOURCE.read_text())
    # The split hashes each planned episode's canonical spec, defaults included.
    paths: dict[str, dict[str, object]] = {}
    for e in CampaignSpec.parse(SOURCE.read_text()).episodes():
        sj = e.canonical_json()
        if data._split_of(data.link_key(sj), 1, 0.15, 0.15) == a.split:  # pyright: ignore[reportPrivateUsage]
            path = json.loads(sj)["path"]
            paths.setdefault(data.link_key(sj), {k: path[k] for k in ("trace", "delay", "buffer")})
    stock = list(STOCK_NAMES) if a.stock == "all" else a.stock.split(",")
    ccs: list[dict[str, object]] = [{"kind": "stock", "name": n} for n in stock]
    aliases = {}
    for name, path in (x.split("=", 1) for x in a.policy):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ccs.append(
            {
                "kind": "agent",
                "mode": "hold",
                "policy": {"path": str(Path(path).resolve()), "sha256": sha},
            }
        )
        aliases[f"agent:{sha[:12]}"] = name
    campaign = {
        "name": f"{a.split}-{a.duration_ms // 1000}s",
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": list(paths.values())},
        "conditions": source["conditions"],
        "duration_ms": a.duration_ms,
        "seed": a.seed,
        "steps": True,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(campaign, indent=1) + "\n")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    n = len(source["conditions"])
    print(f"{len(paths)} {a.split} links, {len(ccs)} controllers, {n} conditions")


if __name__ == "__main__":
    main()
