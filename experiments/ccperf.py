"""Plan the experiments of ccperf's benchmark that this emulator can express,
scored by experiments/ccperf_report.py. ccperf is Vielhaus's benchmark
(gitlab.com/ccperf/ccperf at 18d1620, benchmark.py and paths.py, MIT), with
the semantics of its ns-3 backend (gitlab.com/ccperf/ccperf-ns3 at 7c1709f:
ns-3 simulation code, which does not carry over to a UML kernel, so only
what it sets up is taken). Every run lasts ccperf's 30 s, on one drop-tail
bottleneck.

ccperf's paths (capacity, base RTT, loss, buffer); the default one is scaled
by the number of flows K, as ccperf scales it:

    default                    16 K Mbit/s   15 ms   0       20 K packets
    India to AWS India         100.42        54      0       173
    AWS Korea to China         77.72         102     0.06%   94
    AWS California to Mexico   114.68        90      0       450
    WINDS satellite            42            800     0.74%   2800

The experiments, on the default path unless named otherwise:

    single_flow            one flow on each path
    synchronous_fairness   5 flows on each path; 2 and 10 on the default
    start_time_fairness    2, 5 or 10 flows, the first at 0 s, the others
                           uniform in [0, 5) s
    loss_resilience        loss of 0.1, 0.5, 1, 2, 3, 4 or 5%
    deep_buffers           buffer of 1, 2, 4, 8 or 16 BDP (20 packets each)
    shallow_buffers        buffer of 1, 2, 5, 10 or 20 packets; 5 flows, 10
    rate_step              16 Mbit/s up to 2x or 4x at 10 s, or down from it
    rate_pulse             2x or 4x for 5 s, then 16 Mbit/s for 5 s, repeated
    rate_random_walk       capacity a bounded correlated random walk (from
                           8.5 Mbit/s, steps of 0.9 times the last plus
                           N(0, 0.5^2), clipped to 1 to 16) every 0.1 or 1 s
    replay_rate_trace      ccperf's three Poisson rate traces (0.57, 2.64 and
                           3.04 Mbit/s as named) on its three dynamic paths
    reno_fairness,         2, 4 or 10 flows, half (rounded down) the
    cubic_fairness,        controller and the rest the competitor
    vegas_fairness

ccperf's benchmark has 121 experiments per controller; these are 51 of them
plus 2. Not expressible yet, each for a missing emulator feature: RTT
fairness (8, per-flow RTT), the RTT and loss steps and pulses (10) and the
random walks of delay and loss (4, time-varying delay and loss), scalability
(13, CoDel, and a delay change for 8 of them), AQM (5) and ECN (12), ACK
compression (5), policing (5), UDP background (4, Poisson UDP and
rate-limited TCP), and TCP on-off (4, the observed flow cannot stop before
the episode ends). Scalability with drop-tail in place of CoDel does not
test scalability: its buffer of one BDP at the higher rate lets a loss-based
controller fill it before the step (Cubic used 0.999 of a step from 16 to
512 Mbit/s in a local check), so it is left out.

Divergences from ccperf, beyond the kernel (Linux 7.0 against ns-3's models)
and the link (Mahimahi's 1500-byte delivery opportunities):

- Flows are greedy for the whole run. ccperf sends a fixed size, sized to end
  near 15 s on the default path, and measures each flow until it ends.
- Random loss drops data packets before the bottleneck queue. ccperf's ns-3
  error model drops packets after they cross the bottleneck.
- Synchronous fairness with 2 and 10 flows on the default path: ccperf's
  benchmark.py builds these runs but never adds them. They are added here.
- Each experiment runs --repeats times with fresh seeds, start times and
  walks. ccperf runs each once, with seed 42.

Usage: python experiments/ccperf.py OUT --seed 1 --repeats 5 --stock all
           --policy NAME=PATH ...
Plans OUT/p-<experiment> for each experiment, with variants.json (episode id
to variant and number of flows under test), and writes OUT/aliases.json.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from heldout import controllers

from expliccit_env import campaign
from expliccit_env.spec import EpisodeSpec

SECONDS = 30
BW, RTT, PACKETS = 16, 15, 20  # the default path for one flow; BDP = 20 packets
STATIC = {  # capacity (Mbit/s), base RTT (ms), buffer (packets), loss
    "default": (BW, RTT, PACKETS, 0.0),
    "india_to_aws_india": (100.42, 54, 173, 0.0),
    "aws_korea_to_china": (77.72, 102, 94, 0.0006),
    "aws_california_to_mexico": (114.68, 90, 450, 0.0),
    "winds_satellite": (42.0, 800, 2800, 0.0074),
}
# ccperf's rate traces (resources/traces at 18d1620, from Pantheon's
# observatory), copied unchanged: trace -> base RTT (ms), buffer, loss
TRACES = Path(__file__).resolve().parent / "traces" / "ccperf"
REPLAY = {
    "0.57mbps-poisson.trace": (56, 14, 0.0477),
    "2.64mbps-poisson.trace": (176, 130, 0.0),
    "3.04mbps-poisson.trace": (260, 426, 0.0),
}

# (experiment, variant, path, start of each flow in ms, flows under test, competitor)
type Run = tuple[str, str, dict[str, Any], list[int], int, str | None]


def path(
    levels: Sequence[float],
    rtt: int,
    packets: int,
    loss: float = 0.0,
    level_ms: int = SECONDS * 1000,
) -> dict[str, Any]:
    return {
        "trace": {"kind": "periodic_levels", "levels_mbps": list(levels), "level_ms": level_ms},
        "delay": {"base_rtt_ms": rtt},
        "buffer": {"packets": packets},
        "loss": {"downlink": loss},
    }


def static(
    name: str, k: int = 1, packets: int | None = None, loss: float | None = None
) -> dict[str, Any]:
    bw, rtt, q, p = STATIC[name]
    if name == "default":
        bw, q = bw * k, q * k
    return path([bw], rtt, q if packets is None else packets, p if loss is None else loss)


def random_walk(rng: np.random.Generator, interval_ms: int) -> list[float]:
    """ccperf's bounded correlated random walk of the capacity in Mbit/s, one
    value per interval (CorrelatedRandomWalk in ccperf-ns3)."""
    lo, hi, rho, sigma = 1.0, 16.0, 0.9, 0.5
    x, step, out = (lo + hi) / 2, 0.0, []
    for _ in range(SECONDS * 1000 // interval_ms):
        out.append(round(x, 3))
        step = rho * step + rng.normal(0.0, sigma)
        x = min(max(x + step, lo), hi)
    return out


def trace_file(name: str) -> dict[str, Any]:
    file = TRACES / name
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"kind": "mahimahi_file", "path": str(file), "sha256": sha}


def one(exp: str, variant: str, p: dict[str, Any]) -> Run:
    return (exp, variant, p, [0], 1, None)


def shared(rng: np.random.Generator) -> list[Run]:
    """ccperf's experiments with several flows."""
    out: list[Run] = [
        ("synchronous_fairness", f"default, {k} flows", static("default", k), [0] * k, k, None)
        for k in (2, 10)
    ]
    out += [
        ("synchronous_fairness", f"{n}, 5 flows", static(n, 5), [0] * 5, 5, None) for n in STATIC
    ]
    for k in (2, 5, 10):
        starts = [0] + [10 * round(s * 100) for s in rng.uniform(0, SECONDS / 6, k - 1)]
        out.append(("start_time_fairness", f"{k} flows", static("default", k), starts, k, None))
    out.append(
        ("shallow_buffers", "5 flows, 10 packets", static("default", 5, 10), [0] * 5, 5, None)
    )
    for rival in ("reno", "cubic", "vegas"):
        for k in (2, 4, 10):
            out.append(
                (f"{rival}_fairness", f"{k} flows", static("default", k), [0] * k, k // 2, rival)
            )
    return out


def alone(rng: np.random.Generator) -> list[Run]:
    """ccperf's experiments with one flow."""
    out = [one("single_flow", n, static(n)) for n in STATIC]
    for loss in (0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05):
        out.append(one("loss_resilience", f"{loss:.1%}", static("default", loss=loss)))
    for m in (1, 2, 4, 8, 16):
        out.append(one("deep_buffers", f"{m} BDP", static("default", packets=PACKETS * m)))
    for q in (1, 2, PACKETS // 4, PACKETS // 2, PACKETS):
        out.append(one("shallow_buffers", f"{q} packets", static("default", packets=q)))
    for f in (2, 4):
        for direction, levels in (("up", [BW, BW * f, BW * f]), ("down", [BW * f, BW, BW])):
            p = path(levels, RTT, PACKETS, level_ms=SECONDS * 1000 // 3)
            out.append(one("rate_step", f"{direction} {f}x", p))
        p = path([BW * f, BW] * 3, RTT, PACKETS, level_ms=SECONDS * 1000 // 6)
        out.append(one("rate_pulse", f"{f}x", p))
    for interval in (100, 1000):
        p = path(random_walk(rng, interval), RTT, PACKETS, level_ms=interval)
        out.append(one("rate_random_walk", f"every {interval} ms", p))
    for name, (rtt, q, loss) in REPLAY.items():
        p = path([1.0], rtt, q, loss) | {"trace": trace_file(name)}
        out.append(one("replay_rate_trace", name.removesuffix(".trace"), p))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--stock", default="all", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of an ONNX policy")
    a = p.parse_args()
    ccs, aliases = controllers(a.stock, a.policy)
    rng = np.random.default_rng(a.seed)
    eps: dict[str, list[EpisodeSpec]] = defaultdict(list)
    variants: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    for _ in range(a.repeats):
        for exp, variant, pth, starts, n_test, rival in alone(rng) + shared(rng):
            for c in ccs:
                own = c["name"] if c["kind"] == "stock" else c
                cross = [
                    {"kind": "tcp", "cc": own if i < n_test else rival, "start_ms": s}
                    for i, s in enumerate(starts)
                ][1:]
                spec = EpisodeSpec.model_validate(
                    {
                        "path": pth,
                        "cc": c,
                        "cross": cross,
                        "duration_ms": SECONDS * 1000,
                        "seed": int(rng.integers(0, 2**63)),
                    }
                )
                eps[exp].append(spec)
                variants[exp][spec.episode_id()] = (variant, n_test)
    for exp, specs in eps.items():
        campaign.write_episodes(a.out / f"p-{exp}", specs, {"name": f"p-{exp}"})
        (a.out / f"p-{exp}" / "variants.json").write_text(json.dumps(variants[exp]) + "\n")
        print(f"p-{exp}: {len(specs)} episodes")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")


if __name__ == "__main__":
    main()
