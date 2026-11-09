"""Attribute each decision of expliCCit's released policy to its LCFs.

Reads every steps.parquet under DATA whose observations are kept, runs
expliCCit's released net1 on them, and splits each decision of its actor,
a = h(c), into exact Shapley values over the LCFs
(src/expliccit_env/train/attribution.py), against the mean LCF vector over
REFERENCE's steps (default: DATA's). Also the intervention an operator
can run on the deployed function: a minus h with one LCF set to the
reference (leave one out), which equals the Shapley value where h is
additive. With --as-deployed the LCFs are first transformed as its
deployed controller does (beliefs clamped at 0, the first six rounded),
so the values explain the function that ran; the action's own rounding to
three decimals is left out.

Usage: python experiments/attribution.py OUT DATA... --models MODELS
           [--reference DIR ...] [--stride N]
Writes OUT/attribution.parquet (per step: episode, time, action, each
LCF's value and its leave-one-out effect) and OUT/attribution.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit
from expliccit_env.train.attribution import leave_one_out, shapley

TOP = 3  # Fig. 5 shows the largest three LCFs and "Others"


def observations(dirs: list[Path], stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, ts, obs = [], [], []
    for d in dirs:
        for f in sorted(d.rglob("steps.parquet")):
            if "obs" not in pq.read_schema(f).names:
                continue
            t = pq.read_table(f, columns=["episode_id", "t_ns", "obs"])
            o = np.asarray(t["obs"].combine_chunks().flatten(), np.float32).reshape(-1, OBS_DIM)
            ids.append(np.asarray(t["episode_id"].to_pylist())[::stride])
            ts.append(np.asarray(t["t_ns"].to_numpy(), np.int64)[::stride])
            obs.append(o[::stride])
    if not obs:
        raise SystemExit("no steps with observations under the data directories")
    return np.concatenate(ids), np.concatenate(ts), np.concatenate(obs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("data", type=Path, nargs="+")
    p.add_argument("--models", type=Path, required=True, help="expliCCit's released models")
    p.add_argument("--reference", type=Path, nargs="*", default=[])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--as-deployed", action="store_true")
    a = p.parse_args()
    net1, actor = fit.load_expliccit(a.models)
    ids, ts, obs = observations(a.data, a.stride)
    _, _, ref_obs = observations(a.reference or a.data, a.stride)

    def beliefs(o: np.ndarray) -> torch.Tensor:
        z = torch.from_numpy(net1.beliefs(o))
        if not a.as_deployed:
            return z
        with torch.no_grad():  # as graph.Deployed does
            v = torch.relu(net1.lcf_scaler.inverse(z))
            v = torch.where(torch.arange(v.shape[1]) < 6, torch.round(v), v)
            return (v - net1.lcf_scaler.center) / net1.lcf_scaler.spread

    c = beliefs(obs)
    ref = beliefs(ref_obs).mean(0)

    def h(z: torch.Tensor) -> torch.Tensor:
        return actor(z).reshape(-1)

    phi = shapley(h, c, ref)
    loo = leave_one_out(h, c, ref)
    with torch.no_grad():
        action = h(c).to(torch.float64)
        base = h(ref.unsqueeze(0)).item()
    gap = (phi.sum(1) - (action - base)).abs().max().item()
    names = list(net1.lcfs)
    a.out.mkdir(parents=True, exist_ok=True)
    columns = {"episode_id": ids, "t_ns": ts, "action": action.numpy()}
    for i, n in enumerate(names):
        columns[f"phi_{n}"] = phi[:, i].numpy()
        columns[f"loo_{n}"] = loo[:, i].numpy()
    pq.write_table(pa.table(columns), a.out / "attribution.parquet")

    mag = phi.abs()
    share = mag / mag.sum(1, keepdim=True).clamp(min=1e-12)
    top = share.sort(dim=1, descending=True).values[:, :TOP].sum(1)
    lines = [
        f"expliCCit's released policy on {len(ids)} steps: exact Shapley values over its"
        f" {len(names)} LCFs against the mean LCF vector of the reference steps"
        f" (h at the reference: {base:+.4f}).",
        f"Efficiency: largest |sum of values - (a - h(reference))| = {gap:.2e}.",
        f"The largest {TOP} LCFs hold {top.mean().item():.3f} of each decision's total |value|"
        f" on average; Others hold {1 - top.mean().item():.3f}.",
        "",
        "| LCF | mean share of |value| | mean value | mean leave-one-out"
        " | corr(value, leave-one-out) |",
        "|---|---|---|---|---|",
    ]
    for i, n in enumerate(names):
        corr = (
            np.corrcoef(phi[:, i].numpy(), loo[:, i].numpy())[0, 1] if phi.shape[0] > 2 else np.nan
        )
        lines.append(
            f"| {n} | {share[:, i].mean().item():.3f} | {phi[:, i].mean().item():+.4f} |"
            f" {loo[:, i].mean().item():+.4f} | {corr:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (a.out / "attribution.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
