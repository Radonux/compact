"""Plot net1's beliefs against the truth over one episode (research item
R4.2): for each LCF, the true value at each step, the belief's mean and its
central 90% interval, against time.

Picks, among CAMPAIGN's ok episodes of --controller under --condition
(score.condition's label), a typical one: the episode with the median
variation of capacity (episode row capacity_cv; --pick max takes the most
variable, which on cellular traces is often a link that is mostly out).
Prints the
episode, its link and, per LCF over the episode, the truth's mean, the
mean absolute error of the belief's mean and the coverage of the 90%
interval. Lines break where no step was recorded for over 0.2 s.

Needs matplotlib (not a project dependency):
    uv run --with matplotlib python experiments/lcf_timeseries.py OUT.png NET1.pt CAMPAIGN
        [--controller cubic] [--condition clean] [--pick median|max]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from expliccit_env.policy import OBS_DIM
from expliccit_env.train.fit import load_net1
from expliccit_env.train.score import condition
from expliccit_env.train.spec import LCF_UNITS

GAP_S = 0.2
NAMES = {
    "queue_delay_mean_ns": "mean queueing delay",
    "queue_pkts": "queue",
    "capacity_bps": "capacity",
    "free_buffer_pkts": "free buffer",
    "busy_fraction": "busy fraction",
    "one_way_delay_ns": "one-way delay",
    "receive_rate_bps": "receive rate",
}


def pick(campaign: Path, controller: str, cond: str, how: str) -> tuple[Path, str, str]:
    found: list[tuple[float, str, Path, str]] = []
    for f in sorted(campaign.glob("shards/*/episodes.parquet")):
        rows = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
            f, columns=["episode_id", "status", "controller", "spec_json", "capacity_cv"]
        ).to_pylist()
        for e in rows:
            if e["status"] != "ok" or e["controller"] != controller:
                continue
            if condition(e["spec_json"]) != cond:
                continue
            cv = float(e["capacity_cv"] or 0.0)
            found.append((cv, str(e["episode_id"]), f.parent, str(e["spec_json"])))
    if not found:
        raise SystemExit(f"no ok {controller} episode under {cond!r} in {campaign}")
    found.sort()
    _, eid, shard, spec_json = found[-1] if how == "max" else found[(len(found) - 1) // 2]
    return shard, eid, spec_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("campaign", type=Path)
    p.add_argument("--controller", default="cubic")
    p.add_argument("--condition", default="clean")
    p.add_argument("--pick", choices=["median", "max"], default="median")
    a = p.parse_args()
    shard, eid, spec_json = pick(a.campaign, a.controller, a.condition, a.pick)
    t = pq.read_table(shard / "steps.parquet")  # pyright: ignore[reportUnknownMemberType]
    t = t.filter(pc.equal(t["episode_id"], eid))  # pyright: ignore[reportUnknownMemberType]
    secs = np.asarray(t["t_ns"].to_numpy(), np.float64) / 1e9
    obs = np.asarray(t["obs"].combine_chunks().flatten(), np.float32).reshape(-1, OBS_DIM)
    net1, _ = load_net1(a.net1)
    mean = net1.predict(obs)
    lo, hi = net1.bounds(obs, 0.9)
    lcfs = list(net1.lcfs)
    fig, axes = plt.subplots(len(lcfs), 1, figsize=(8, 1.6 * len(lcfs)), sharex=True)
    path = json.loads(spec_json)["path"]
    print(f"episode {eid} ({a.controller}, {a.condition}); path {json.dumps(path)}")
    print("LCF | truth's mean | mean absolute error | 90% coverage")
    gap = np.concatenate([[False], np.diff(secs) > GAP_S])  # a step after a gap

    def broken(y: np.ndarray) -> np.ndarray:
        return np.where(gap, np.nan, y)

    for i, (n, ax) in enumerate(zip(lcfs, axes, strict=True)):
        scale, unit = LCF_UNITS[n]
        truth = np.asarray(t[f"label_{n}"].to_numpy(zero_copy_only=False), np.float64) * scale
        ok = np.isfinite(truth)
        mae = np.mean(np.abs(mean[ok, i] - truth[ok]))
        cover = np.mean((truth[ok] >= lo[ok, i]) & (truth[ok] <= hi[ok, i]))
        print(
            f"{NAMES.get(n, n)} | {np.mean(truth[ok]):.3g} {unit} | {mae:.3g} {unit} | {cover:.3f}"
        )
        ax.fill_between(
            secs, broken(lo[:, i]), broken(hi[:, i]), color="tab:blue", alpha=0.25, lw=0
        )
        ax.plot(secs, broken(mean[:, i]), color="tab:blue", lw=0.8, label="belief: mean, 90%")
        ax.plot(secs, broken(truth), color="black", lw=0.8, label="truth")
        ax.set_ylabel(f"{NAMES.get(n, n)}\n({unit})" if unit else NAMES.get(n, n), fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
