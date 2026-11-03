"""Write a campaign on the held-out links of examples/campaign-net1-varied.json.

The links are split as net1 v3 and every net2 run on it were: by link
(the path without its loss), seed 1, 15% validation, 15% test. The campaign
runs the given policies and stock controllers on one split's links under
one family of conditions, in one campaign. Episode seeds are drawn in order
from the campaign seed, so changing the controller list changes every
episode's seed; the pairing is by link and condition.

Families: "training" is that campaign's three conditions (clean, 1% loss,
3 Mbit/s UDP plus a Cubic flow from 8 s). The others are conditions it never
had: "new" (a Cubic flow from 0 s, a BBRv1 flow from 4 s, Cubic flows from
2 s and 10 s, 0.1% and 3% random loss, 1% loss on the ACK path),
"cubic-first" (a Cubic flow from 0 s alone), and "deep-long" (the training
conditions on the same links with a 2.5 times longer one-way delay, at most
80 ms, and a buffer of 4 to 16 BDP drawn with seed 31), and
"held-competitors" (a BBRv3 flow from 0 s, a Westwood flow from 0 s: the
competitors net1 v4's training data held out).

Usage: python experiments/heldout.py OUT_DIR --split test --duration-ms 60000
           --seed 13 [--family training] --stock all --policy NAME=PATH ...

Writes OUT_DIR/campaign.json and OUT_DIR/aliases.json (agent:SHA12 -> NAME,
for `expliccit score --alias`).
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from expliccit_env.spec import STOCK_NAMES, CampaignSpec
from expliccit_env.train import data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/campaign-net1-varied.json"
HELD_COMPETITORS: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": cc, "start_ms": 0}]} for cc in ("bbr3", "westwood")
]
NEW: list[dict[str, Any]] = [
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": 0}]},
    {"cross": [{"kind": "tcp", "cc": "bbr", "start_ms": 4000}]},
    {"cross": [{"kind": "tcp", "cc": "cubic", "start_ms": t} for t in (2000, 10000)]},
    {"loss": {"downlink": 0.001}},
    {"loss": {"downlink": 0.03}},
    {"loss": {"uplink": 0.01}},
]


def deep_long(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each link with a 2.5 times longer one-way delay (at most 80 ms) and a
    buffer of 4 to 16 BDP (log-uniform, seed 31, clipped to 20 to 8000)."""
    rng = np.random.default_rng(31)
    out = []
    for p in paths:
        ow = min(int(p["delay"]["mahak_compat_mrtt_ms"] * 2.5), 80)
        bdp = p["trace"]["levels_mbps"][0] * 1e6 * 2 * ow / 1e3 / 12000
        buf = int(np.clip(bdp * np.exp(rng.uniform(np.log(4), np.log(16))), 20, 8000))
        out.append(p | {"delay": {"mahak_compat_mrtt_ms": ow}, "buffer": {"packets": buf}})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--duration-ms", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--family",
        default="training",
        choices=["training", "new", "cubic-first", "deep-long", "held-competitors"],
    )
    p.add_argument("--stock", default="all", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of an ONNX policy")
    a = p.parse_args()
    source = json.loads(SOURCE.read_text())
    # The split hashes each planned episode's canonical spec, defaults included.
    paths: dict[str, dict[str, Any]] = {}
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
    family = a.family
    links = list(paths.values())
    conditions = source["conditions"]
    if family == "deep-long":
        links = deep_long(links)
    elif family == "held-competitors":
        conditions = HELD_COMPETITORS
    elif family != "training":
        conditions = NEW if family == "new" else NEW[:1]
    campaign = {
        "name": f"{a.split}-{family}-{a.duration_ms // 1000}s",
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": links},
        "conditions": conditions,
        "duration_ms": a.duration_ms,
        "seed": a.seed,
        "steps": True,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(campaign, indent=1) + "\n")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(f"{len(links)} {a.split} links, {len(ccs)} controllers, {len(conditions)} conditions")


if __name__ == "__main__":
    main()
