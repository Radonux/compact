"""What one decision of each controller costs (research item R6), and what
explaining one of expliCCit's decisions costs.

Each NAME=PATH@RATE is a policy graph and how many times a second the
guest runs it: once per 10 ms step for the step contract's policies
(expliCCit, Sage, the rules, the learned net2), once per 20 ms for Orca's,
once per half RTT for Aurora's (100 a second at a 20 ms RTT).
tools/policy-bench times each with the guest's tract on one core, and the
CPU share is the median times the rate. For expliCCit's released models,
in torch on one thread: net1 on one state message, the actor on its
beliefs, and the exact Shapley values of the decision over its 7 LCFs
(train/attribution.py: 2^7 = 128 actor evaluations in one batch), each
timed over CALLS decisions after 200 to warm up.

Usage: python experiments/overhead.py OUT BENCH NAME=PATH@RATE ...
           --expliccit MODELS [--calls N]
Writes OUT/overhead.md and OUT/overhead.json.
"""

import argparse
import json
import platform
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from expliccit_env.train import fit
from expliccit_env.train.attribution import shapley

WARM = 200
BENCH_LINE = re.compile(r"median ([\d.]+) us, p90 ([\d.]+) us, p99 ([\d.]+) us")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def bench(binary: Path, graph: Path, calls: int) -> tuple[float, float, float]:
    out = subprocess.run(
        [str(binary), str(graph), str(calls)], capture_output=True, text=True, check=True
    ).stdout
    m = BENCH_LINE.search(out)
    if m is None:
        raise SystemExit(f"policy-bench printed no timing for {graph}: {out!r}")
    med, p90, p99 = (float(x) for x in m.groups())
    return med, p90, p99


def timed(f: Callable[[], None], calls: int) -> tuple[float, float]:
    """Median and p99 in us of f() over calls, after WARM calls."""
    times: list[float] = []
    for k in range(WARM + calls):
        t = time.perf_counter()
        f()
        if k >= WARM:
            times.append((time.perf_counter() - t) * 1e6)
    q = np.quantile(times, [0.5, 0.99])
    return float(q[0]), float(q[1])


def explain_costs(models: Path, calls: int) -> dict[str, tuple[float, float]]:
    torch.set_num_threads(1)
    net1, actor = fit.load_expliccit(models)
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 1, (calls + WARM, 78)).astype(np.float32)
    z = torch.from_numpy(net1.beliefs(obs))
    ref = z.mean(0)
    rows = iter(range(10**9))

    def one_obs() -> None:
        i = next(rows) % len(obs)
        net1.beliefs(obs[i : i + 1])

    def act() -> None:
        i = next(rows) % len(z)
        with torch.no_grad():
            actor(z[i : i + 1])

    def explain() -> None:
        i = next(rows) % len(z)
        with torch.no_grad():
            shapley(lambda x: actor(x).reshape(-1), z[i : i + 1], ref)

    return {
        "net1 on one message": timed(one_obs, calls),
        "actor on its beliefs": timed(act, calls),
        "exact Shapley over 7 LCFs": timed(explain, calls),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("bench", type=Path, help="tools/policy-bench's binary")
    p.add_argument("graphs", nargs="+", help="NAME=PATH@RATE (decisions a second)")
    p.add_argument("--expliccit", type=Path, required=True, help="expliCCit's released models")
    p.add_argument("--calls", type=int, default=20_000)
    a = p.parse_args()
    rows = []
    for g in a.graphs:
        name, rest = g.split("=", 1)
        path, rate = rest.rsplit("@", 1)
        med, p90, p99 = bench(a.bench, Path(path), a.calls)
        rows.append(
            {
                "name": name,
                "rate": float(rate),
                "median_us": med,
                "p90_us": p90,
                "p99_us": p99,
                "cpu_share": med * float(rate) / 1e6,
            }
        )
    explain = explain_costs(a.expliccit, min(a.calls, 5_000))
    cpu = cpu_model()
    lines = [
        "# Cost of a decision",
        "",
        f"On one core of {cpu}. A policy graph runs in the guest's tract"
        " (tools/policy-bench, inputs fixed, the first 1000 calls left out); the CPU share"
        " is the median times the rate at which the guest runs it.",
        "",
        "| Policy | Runs a second | Median (us) | p90 (us) | p99 (us) | CPU share |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['rate']:g} | {r['median_us']:.1f} | {r['p90_us']:.1f} |"
            f" {r['p99_us']:.1f} | {100 * r['cpu_share']:.2f}% |"
        )
    lines += [
        "",
        "expliCCit's released models in torch on one thread, per decision:",
        "",
        "| Step | Median (us) | p99 (us) |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {m:.1f} | {q:.1f} |" for k, (m, q) in explain.items()]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "overhead.md").write_text("\n".join(lines) + "\n")
    (a.out / "overhead.json").write_text(
        json.dumps({"cpu": cpu, "graphs": rows, "expliccit_torch": explain}, indent=1) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
