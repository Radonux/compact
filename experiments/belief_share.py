"""net1's capacity belief on a shared link.

For each controller and number of copies n in a copies campaign (n = 1 is
the flow alone), the median over the observed flow's steps after its
first 5 s of net1's believed capacity (its distribution's mean) over the
link's capacity and over the fair share (the link's capacity over n), with
the flow's own receive rate over the link's capacity beside them. It tests
whether net1 believes a shared link smaller than it is, nearer the flow's
share: what a window on the believed BDP needs among copies, and what a
window on the true capacity lacks.

Usage: python experiments/belief_share.py OUT NET1.pt
Reads every c-* campaign under OUT, whose steps tables keep the
observations, and OUT/aliases.json; writes OUT/summary.md.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit, score

type F64 = npt.NDArray[np.float64]
SETTLE_S = 5.0


def copies_of(condition: str) -> int:
    """Flows on the link: the observed one and each "itself" cross flow."""
    return 1 + condition.count("itself")


def medians(belief: F64, capacity: F64, receive: F64, n: int) -> dict[str, float]:
    """Median ratios over the steps where all three are finite and the
    capacity positive (Mbit/s throughout)."""
    ok = np.isfinite(belief) & np.isfinite(capacity) & np.isfinite(receive) & (capacity > 0)
    b, c, r = belief[ok], capacity[ok], receive[ok]
    return {
        "steps": int(ok.sum()),
        "belief_over_capacity": float(np.median(b / c)),
        "belief_over_share": float(np.median(b / (c / n))),
        "receive_over_capacity": float(np.median(r / c)),
    }


def main(out: Path, net1_path: Path) -> None:
    net1, _ = fit.load_net1(net1_path)
    cap = net1.lcfs.index("capacity_bps")
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    acc: dict[tuple[str, int], list[F64]] = defaultdict(lambda: [np.zeros(0)] * 3)
    for shard in sorted(out.glob("c-*/shards/*")):
        eps = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "controller", "spec_json", "status"]
        )
        which = {
            e["episode_id"]: (
                aliases.get(e["controller"], e["controller"]),
                copies_of(score.condition(e["spec_json"])),
            )
            for e in eps.to_pylist()
            if e["status"] == "ok"
        }
        cols = ["episode_id", "t_ns", "obs", "label_capacity_bps", "label_receive_rate_bps"]
        t = pq.read_table(shard / "steps.parquet", columns=cols)
        ids = np.asarray(t["episode_id"].to_pylist())
        ts = np.asarray(t["t_ns"].to_numpy(), np.float64) / 1e9
        obs = np.asarray(t["obs"].combine_chunks().flatten(), np.float32).reshape(-1, OBS_DIM)
        truth = np.asarray(t["label_capacity_bps"].to_numpy(zero_copy_only=False), np.float64) / 1e6
        recv = (
            np.asarray(t["label_receive_rate_bps"].to_numpy(zero_copy_only=False), np.float64) / 1e6
        )
        for eid in np.unique(ids):
            if eid not in which:
                continue
            rows = np.flatnonzero(ids == eid)
            rows = rows[ts[rows] >= ts[rows].min() + SETTLE_S]
            if rows.size == 0:
                continue
            belief = net1.predict(obs[rows])[:, cap].astype(np.float64)
            parts = acc[which[eid]]
            acc[which[eid]] = [
                np.concatenate([parts[0], belief]),
                np.concatenate([parts[1], truth[rows]]),
                np.concatenate([parts[2], recv[rows]]),
            ]
    lines = [
        "net1's believed capacity on a shared link: medians over the observed flow's steps",
        "after its first 5 s; the fair share is the link's capacity over the number of flows.",
        "",
        "| controller | flows | steps | belief / capacity | belief / fair share"
        " | receive rate / capacity |",
        "|---|---|---|---|---|---|",
    ]
    for (ctl, n), (b, c, r) in sorted(acc.items()):
        m = medians(b, c, r, n)
        lines.append(
            f"| {ctl} | {n} | {m['steps']} | {m['belief_over_capacity']:.2f} | "
            f"{m['belief_over_share']:.2f} | {m['receive_over_capacity']:.2f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
