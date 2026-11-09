"""Does a controller probe where it cannot see the capacity?

A probe here is a window above net1's believed bandwidth-delay product,
held where the link is not busy, so that a rise in capacity shows. Per
step, u is log2 of the window over the believed BDP (the fixed rule's u is
its constant; a probing policy's u rises off a busy link). Against it:
how idle net1 believes the link is (1 minus its mean busy fraction), the
spread of its log2 capacity belief, and the true busy fraction.

Usage:
    python experiments/probing.py shard DIR NET1.pt
        For every shard under DIR whose steps keep the observations, write
        probe.parquet beside them: every 10th step after the first 5 s.
        Run it in a cell before experiments/slim_steps.py drops them.
    python experiments/probing.py report OUT
        Reads every probe.parquet under OUT/c-* and OUT/aliases.json and
        writes OUT/probing.md: per campaign and controller, mean u where
        the link is busy (true busy fraction at least 0.95) and where it
        is not, and the slope of u on net1's believed idle share and on
        its spread (least squares, both at once).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit

type F64 = npt.NDArray[np.float64]
SETTLE_S = 5.0
STRIDE = 10
BUSY = 0.95


def shard(directory: Path, net1_path: Path) -> None:
    net1, _ = fit.load_net1(net1_path)
    target = fit.window_target(net1)
    busy_i, cap_i = net1.lcfs.index("busy_fraction"), net1.lcfs.index("capacity_bps")
    mid = target.means.mid
    log_mid = torch.log2(torch.clamp(mid[cap_i], min=1e-3))
    for steps in sorted(directory.rglob("steps.parquet")):
        names = pq.read_schema(steps).names
        if "obs" not in names:
            continue
        t = pq.read_table(
            steps, columns=["episode_id", "t_ns", "obs", "cwnd_set", "cwnd", "label_busy_fraction"]
        )
        ids = np.asarray(t["episode_id"].to_pylist())
        ts = np.asarray(t["t_ns"].to_numpy(), np.float64) / 1e9
        keep = np.zeros(len(ids), bool)
        for e in np.unique(ids):
            rows = np.flatnonzero(ids == e)
            rows = rows[ts[rows] >= ts[rows].min() + SETTLE_S][::STRIDE]
            keep[rows] = True
        rows = np.flatnonzero(keep)
        obs = np.asarray(t["obs"].combine_chunks().flatten(), np.float32).reshape(-1, OBS_DIM)
        window = np.asarray(t["cwnd_set"].to_numpy(zero_copy_only=False), np.float64)
        stock = np.asarray(t["cwnd"].to_numpy(zero_copy_only=False), np.float64)
        window = np.where(np.isfinite(window) & (window > 0), window, stock)[rows]
        with torch.no_grad():
            z = torch.from_numpy(net1.beliefs(obs[rows]))
            lb = target(z).numpy().astype(np.float64)
            p = z.reshape(len(rows), *mid.shape)
            idle = 1 - (p[:, busy_i] * mid[busy_i]).sum(dim=1).numpy()
            pc = p[:, cap_i]
            m = (pc * log_mid).sum(dim=1, keepdim=True)
            spread = torch.sqrt(torch.clamp((pc * (log_mid - m) ** 2).sum(dim=1), min=0)).numpy()
        out = pa.table(
            {
                "episode_id": ids[rows],
                "u": np.log2(np.maximum(window, 1.0)) - lb,
                "idle_belief": idle.astype(np.float64),
                "spread": spread.astype(np.float64),
                "busy": t["label_busy_fraction"].to_numpy()[rows].astype(np.float64),
            }
        )
        pq.write_table(out, steps.with_name("probe.parquet"))


def slopes(u: F64, idle: F64, spread: F64) -> tuple[float, float]:
    """Least-squares slopes of u on the believed idle share and the spread."""
    x = np.column_stack([np.ones_like(u), idle, spread])
    coef, *_ = np.linalg.lstsq(x, u, rcond=None)
    return float(coef[1]), float(coef[2])


def report(out: Path) -> None:
    aliases: dict[str, str] = json.loads((out / "aliases.json").read_text())
    rows: dict[tuple[str, str], list[pa.Table]] = defaultdict(list)
    for probe in sorted(out.glob("c-*/shards/*/probe.parquet")):
        eps = pq.read_table(
            probe.with_name("episodes.parquet"), columns=["episode_id", "controller", "status"]
        ).to_pylist()
        ctl = {e["episode_id"]: e["controller"] for e in eps if e["status"] == "ok"}
        t = pq.read_table(probe)
        ids = t["episode_id"].to_pylist()
        for c in sorted({ctl[i] for i in ids if i in ctl}):
            mask = pa.array([ctl.get(i) == c for i in ids])
            rows[(probe.parents[2].name[2:], aliases.get(c, c))].append(t.filter(mask))
    lines = [
        "u is log2 of the window over net1's believed BDP. Busy: true busy fraction at least",
        f"{BUSY}. Slopes: least squares of u on net1's believed idle share and on the spread",
        "of its log2 capacity belief, together. A controller that probes where it cannot",
        "see the capacity has a higher u off a busy link and a positive slope.",
        "",
        "| campaign | controller | steps | u busy | u not busy | share not busy | slope on idle"
        " | slope on spread |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (camp, name), parts in sorted(rows.items()):
        t = pa.concat_tables(parts)
        u, busy = t["u"].to_numpy(), t["busy"].to_numpy()
        ok = np.isfinite(u) & np.isfinite(busy)
        u, busy = u[ok], busy[ok]
        idle, spread = t["idle_belief"].to_numpy()[ok], t["spread"].to_numpy()[ok]
        off = busy < BUSY
        a, b = slopes(u, idle, spread) if u.size > 2 else (float("nan"), float("nan"))
        lines.append(
            f"| {camp} | {name} | {u.size} | {u[~off].mean():+.3f} | "
            f"{u[off].mean() if off.any() else float('nan'):+.3f} | {off.mean():.3f} | "
            f"{a:+.3f} | {b:+.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "probing.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["shard", d, n]:
            shard(Path(d), Path(n))
        case ["report", o]:
            report(Path(o))
        case _:
            raise SystemExit(__doc__)
