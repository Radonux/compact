"""Plan a congestion-control benchmark: ccBench's two suites as Abbasloo
defines them (github.com/Soheil-ab/ccBench at 49b5ada, solo_runall.sh and
multi_runall.sh; "Internet Congestion Control Benchmarking", arXiv
2307.10054), and suites ccBench lacks, scored by experiments/ccbench_report.py.

ccBench's suites, on one bottleneck of B Mbit/s, one-way delay D ms each way
and a drop-tail buffer, with BDP = D * B // 6 packets as ccBench computes it:

    solo      one flow for 30 s; B in {12, 24, 48, 96, 192}, D in {5, 10, 20,
              40, 80}, buffer in {1/2, 1, 2, 4, 8, 16} BDP; a constant link,
              and the same link raised 2x or 4x (B < 100 only) or lowered 2x
              or 4x from 7 s to 14 s (ccBench's "-7s" traces, which start 10 s
              into its runs, after its setup time). 690 runs per controller.
    friendly  a Cubic flow from 0 s and the controller from 5 s, 125 s in
              all; constant links, buffer in {1, 2, 4, 8, 16} BDP. 125 runs.
              Here the observed flow is the Cubic one and the controller is
              cross flow 0, since a cross flow is what can start late.

Suites ccBench lacks (constant links, buffer in {1, 4} BDP):

    loss         solo with i.i.d. data loss of 0.1%, 1% or 3%.
    copies       2, 4 or 8 flows of the controller starting together, 60 s.
    competitors  friendly with BBRv1 or BBRv3 in place of Cubic.

Usage: python experiments/ccbench.py OUT --suite solo friendly ... --seed 1
           --stock all --policy NAME=PATH ...
Plans OUT/c-<suite> for each suite and writes OUT/aliases.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from heldout import controllers, keep_only

from expliccit_env import campaign
from expliccit_env.spec import EpisodeSpec

BANDWIDTHS = (12, 24, 48, 96, 192)
DELAYS = (5, 10, 20, 40, 80)
SOLO_BUFFERS = (0.5, 1, 2, 4, 8, 16)
FRIENDLY_BUFFERS = (1, 2, 4, 8, 16)
SMALL_BUFFERS = (1, 4)
STEP_MS = 7000  # ccBench's -7s traces: changed for 7 s from 7 s


def bdp(bw: int, delay: int) -> int:
    return delay * bw // 6  # packets over the RTT 2 D, as solo_runall.sh rounds it


def buffer(bw: int, delay: int, mult: float) -> int:
    b = bdp(bw, delay)
    return b // 2 if mult == 0.5 else int(mult) * b


def path(
    bw: int, delay: int, packets: int, levels: list[float], loss: float = 0.0
) -> dict[str, Any]:
    return {
        "trace": {"kind": "periodic_levels", "levels_mbps": levels, "level_ms": STEP_MS},
        "delay": {"mahak_compat_mrtt_ms": delay},
        "buffer": {"packets": packets},
        "loss": {"downlink": loss},
    }


def solo_links() -> list[tuple[dict[str, Any], str]]:
    """ccBench1's links and their trace label."""
    out = []
    for bw in BANDWIDTHS:
        steps = [("", [bw])]
        steps += [(f"{s}x-up", [bw, bw * s, bw, bw, bw]) for s in (2, 4) if bw < 100]
        steps += [(f"{s}x-down", [bw, bw / s, bw, bw, bw]) for s in (2, 4)]
        for delay in DELAYS:
            for mult in SOLO_BUFFERS:
                for label, levels in steps:
                    out.append((path(bw, delay, buffer(bw, delay, mult), levels), label))
    return out


def constant_links(buffers: tuple[float, ...], loss: float = 0.0) -> list[dict[str, Any]]:
    return [
        path(bw, d, buffer(bw, d, m), [bw], loss)
        for bw in BANDWIDTHS
        for d in DELAYS
        for m in buffers
    ]


def episodes(suite: str, ccs: list[dict[str, Any]], rng: np.random.Generator) -> list[EpisodeSpec]:
    def spec(cc: Any, p: dict[str, Any], seconds: int, cross: list[dict[str, Any]]) -> EpisodeSpec:
        return EpisodeSpec.model_validate(
            {
                "path": p,
                "cc": cc,
                "cross": cross,
                "duration_ms": seconds * 1000,
                "seed": int(rng.integers(0, 2**63)),
            }
        )

    own = [c["name"] if c["kind"] == "stock" else c for c in ccs]
    out: list[EpisodeSpec] = []
    if suite == "solo":
        out = [spec(c, p, 30, []) for p, _ in solo_links() for c in ccs]
    elif suite == "loss":
        out = [
            spec(c, p, 30, [])
            for loss in (0.001, 0.01, 0.03)
            for p in constant_links(SMALL_BUFFERS, loss)
            for c in ccs
        ]
    elif suite in ("friendly", "competitors"):
        rivals = ["cubic"] if suite == "friendly" else ["bbr", "bbr3"]
        buffers = FRIENDLY_BUFFERS if suite == "friendly" else SMALL_BUFFERS
        out = [
            spec(
                {"kind": "stock", "name": rival},
                p,
                125,
                [{"kind": "tcp", "cc": o, "start_ms": 5000}],
            )
            for rival in rivals
            for p in constant_links(buffers)
            for o in own
        ]
    elif suite == "copies":
        out = [
            spec(c, p, 60, [{"kind": "tcp", "cc": o, "start_ms": 0}] * (n - 1))
            for n in (2, 4, 8)
            for p in constant_links(SMALL_BUFFERS)
            for c, o in zip(ccs, own, strict=True)
        ]
    else:
        raise SystemExit(f"unknown suite {suite}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument(
        "--suite",
        nargs="+",
        required=True,
        choices=["solo", "friendly", "loss", "copies", "competitors"],
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stock", default="all", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of an ONNX policy")
    p.add_argument("--only", default="", help="write only these controllers' episodes (NAMEs)")
    a = p.parse_args()
    ccs, aliases = controllers(a.stock, a.policy)
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    for suite in a.suite:
        eps = keep_only(episodes(suite, ccs, rng), a.only, aliases)
        campaign.write_episodes(a.out / f"c-{suite}", eps, {"name": f"c-{suite}"})
        print(f"c-{suite}: {len(eps)} episodes")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")


if __name__ == "__main__":
    main()
