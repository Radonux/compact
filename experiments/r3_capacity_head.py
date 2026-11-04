"""Repair net1 v3 by retraining its capacity head alone (research item R3).

Usage: python experiments/r3_capacity_head.py OUT --net1 NET1.pt --data DIR... \
           [--ood NAME=DIR ...] [--compare NET1.pt] [--bins refit|keep] \
           [--epochs N] [--lr LR ...] [--seed 0] [--max-rows N] [--u 0.125 0.25]

Each member of net1's ensemble (models.ExpliccitNet1) ends in one linear
head per LCF, from its last 64 features to that LCF's bin logits. This
retrains the capacity head of every member, and nothing else, on the rows
net1 v4 was trained on (experiments/net1_train.py: every 12th step, split by
link, BBRv3, Westwood and threshold-0.3 held out of training), with net1's
own objective for that LCF, the log score of the true bin (fit._loss), and
its own optimizer and early stopping on the validation rows' capacity log
score (fit._train, with net1's spec but for --epochs). Each member's head
is fitted once per learning rate in --lr (default: net1's) from the same
start, and keeps the fit with the best validation log score. The capacity
head is a layer of its own, so no rows of a shared layer need masking:
every other parameter has requires_grad False, and the rest of each member
runs as at inference (BatchNorm's running statistics, no dropout), so its
features are computed once per row.

--bins refit (the default) refits capacity's bin boundaries on the training
rows as fit_net1 does and starts the head afresh: net1 v3's boundaries span
4.5 to 213 Mbit/s, and the broad data's capacities about 1 to 300. --bins
keep keeps net1 v3's boundaries and starts from its head. --max-rows keeps
at most N rows of each split, for a quick check.

Reported for net1 v3 (before) and the repaired net1 (after), on the
validation and test rows, the held-out controllers' test rows, the
validation rows of episodes with a competing TCP flow, and each --ood
dataset whole: per LCF the 90% coverage, R^2 of the mean, CRPS and log
score, and capacity's median belief (its mean) over the truth; --compare
reports another net1 (net1 v4) on the same rows. The other six LCFs'
distributions must be bit-identical on every view, or the run fails.

Writes OUT/net1.pt, OUT/rule-u{u}.onnx (experiments/rule_export.py),
OUT/report.json (with the training time and CPU threads) and OUT/summary.md.
"""

import argparse
import copy
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch
from net1_train import STRIDE, controllers, held
from torch import nn

from expliccit_env.train import data, fit
from expliccit_env.train.models import BinNet1, ExpliccitNet1
from expliccit_env.train.spec import LCF_UNITS, Net1Spec

ROOT = Path(__file__).resolve().parents[1]
CAP = "capacity_bps"
CHUNK = 1 << 17  # rows per forward pass
type Rows = npt.NDArray[np.bool_]
type Tensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def competing_tcp(dirs: list[Path], steps: data.Steps) -> Rows:
    """Per row: whether its episode has a competing TCP flow."""
    of: dict[str, bool] = {}
    for d in dirs:
        for f in d.rglob("episodes.parquet"):
            t = pq.read_table(f, columns=["episode_id", "spec_json"])
            for e, s in zip(t["episode_id"].to_pylist(), t["spec_json"].to_pylist(), strict=True):
                of[e] = any(x["kind"] == "tcp" for x in json.loads(s).get("cross") or [])
    return np.array([of[e] for e in steps.episode_ids])[steps.episode]


@torch.no_grad()
def features(net1: fit.Net1, member: ExpliccitNet1, obs: npt.NDArray[np.float32], rows: Rows):
    """The member's last features, what its heads read, on the chosen rows.
    ponytail: holds them in memory, 256 bytes a row (0.9 GB for net1 v4's
    3.6 million training rows); stream them per batch if the data outgrows it."""
    body = nn.Sequential(member.initial, member.res1, member.res2, member.hidden).eval()
    idx = np.flatnonzero(rows)
    out = torch.empty(idx.size, member.heads[0].in_features)
    for i in range(0, idx.size, CHUNK):
        out[i : i + CHUNK] = body(net1.scaler(torch.from_numpy(obs[idx[i : i + CHUNK]])))
    return out


def retrain(
    net1: fit.Net1,
    spec: Net1Spec,
    steps: data.Steps,
    split: npt.NDArray[np.str_],
    a: argparse.Namespace,
) -> dict[str, Any]:
    """Retrain every member's capacity head of ``net1`` in place."""
    net = net1.net
    assert isinstance(net, BinNet1)
    c = net1.lcfs.index(CAP)
    y = fit.targets(steps, net1.lcfs)[:, c : c + 1]
    tr, va = split == "train", split == "val"
    if a.bins == "refit":
        net.bounds[c] = torch.from_numpy(fit._bin_bounds(y[tr], net.b)[0])  # pyright: ignore[reportPrivateUsage]
    net.requires_grad_(False)
    t0 = time.perf_counter()
    members: list[dict[str, Any]] = []
    for i, member in enumerate(net.members):
        assert isinstance(member, ExpliccitNet1)
        head = member.heads[c]
        assert isinstance(head, nn.Linear) and head.out_features == net.b
        head.requires_grad_(True)
        if a.bins == "refit":
            torch.manual_seed(a.seed + i)
            head.reset_parameters()
        one = BinNet1([head], 1, net.b)  # the head alone: capacity's bins from the features
        one.bounds.copy_(net.bounds[c : c + 1])

        def tensors(rows: Rows, member: ExpliccitNet1 = member, one: BinNet1 = one) -> Tensors:
            with torch.no_grad():
                yb = one.bin_of(torch.from_numpy(np.nan_to_num(y[rows]))).float()
            m = torch.from_numpy(np.isfinite(y[rows]).astype(np.float32))
            return features(net1, member, steps.obs, rows), yb, m

        t1 = time.perf_counter()
        train, val = tensors(tr), tensors(va)
        log: dict[str, Any] = {"features_s": time.perf_counter() - t1}
        log["val_log_score_start"] = fit._val_loss(one, spec, *val)  # pyright: ignore[reportPrivateUsage]
        start, fits = copy.deepcopy(head.state_dict()), []
        for lr in a.lr or [spec.lr]:
            t2 = time.perf_counter()
            head.load_state_dict(start)
            s = Net1Spec.model_validate(spec.model_dump() | {"lr": lr})
            history = fit._train(one, s, train, val, a.seed + i)  # pyright: ignore[reportPrivateUsage]
            run = {"lr": lr, "fit_s": time.perf_counter() - t2, "val_log_score": history}
            fits.append((min(history), run, copy.deepcopy(head.state_dict())))
        best = min(fits, key=lambda f: f[0])
        head.load_state_dict(best[2])
        members.append(log | {"chosen_lr": best[1]["lr"], "fits": [f[1] for f in fits]})
        del train, val
    net.eval()
    return {
        "train_s": time.perf_counter() - t0,
        "torch_threads": torch.get_num_threads(),
        "members": members,
    }


def compare(models: dict[str, fit.Net1], steps: data.Steps, rows: Rows) -> dict[str, Any]:
    """Whether the LCFs other than capacity have bit-identical distributions
    before and after, and each net1's capacity belief (its distribution's
    mean) over the truth, the median over these rows."""
    obs = steps.obs[rows]
    truth = steps.labels[CAP][rows] * np.float32(LCF_UNITS[CAP][0])
    same, means = True, {k: list[npt.NDArray[np.float32]]() for k in models}
    others = models["before"].lcfs.index(CAP)  # delete capacity, compare the rest
    for i in range(0, obs.shape[0], CHUNK):
        p = {k: n1.distributions(obs[i : i + CHUNK]) for k, n1 in models.items()}
        same &= np.array_equal(np.delete(p["before"], others, 1), np.delete(p["after"], others, 1))
        for k, n1 in models.items():
            c = n1.lcfs.index(CAP)
            b = n1.net.bounds[c].numpy()
            means[k].append(p[k][:, c] @ ((b[:-1] + b[1:]) / 2))
    ok = np.isfinite(truth) & (truth > 0)
    belief = {
        k: float(np.median(np.concatenate(m)[ok] / truth[ok])) if ok.any() else float("nan")
        for k, m in means.items()
    }
    return {"others_identical": bool(same), "capacity_belief_over_truth_median": belief}


def brief(rep: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        k: {
            "c90": v["distribution"]["coverage"]["0.9"],
            "r2": v["net1"].get("r2", float("nan")),
            "crps": v["distribution"]["crps"],
            "crps_skill": v["distribution"]["crps_skill"],
            "log_score": v["distribution"]["log_loss"],
        }
        for k, v in rep.items()
        if "distribution" in v
    }


def report(
    models: dict[str, fit.Net1], steps: data.Steps, rows: Rows, keep: npt.NDArray[np.str_]
) -> dict[str, Any]:
    """Each net1 on the rows marked; ``keep`` holds the other rows' splits
    (training rows give the forecast without measurements)."""
    out = {"rows": int(rows.sum())} | compare(models, steps, rows)
    if not out["others_identical"]:
        raise RuntimeError("an LCF other than capacity changed, though its weights were frozen")
    view = dataclasses.replace(steps, split=np.where(rows, "test", keep).astype("<U5"))
    for name, n1 in models.items():
        out[name] = brief(fit.report_net1(n1, view, "test"))
    return out


def summary(rep: dict[str, Any], lcfs: list[str]) -> str:
    t = rep["training"]
    m = t["members"]
    fits = [(f, f["lr"] == x["chosen_lr"]) for x in m for f in x["fits"]]
    lrs = ", ".join(f"{f['lr']:g}" for f in m[0]["fits"])
    lines = [
        f"net1 v3 ({rep['net1']}) with its capacity head retrained (bins: {rep['bins']}),"
        f" {rep['rows']['train']} training rows, seed {rep['seed']}.",
        "",
        f"Training took {t['train_s']:.1f} s of wall time with {t['torch_threads']} CPU threads:"
        f" {sum(x['features_s'] for x in m):.1f} s for the frozen features,"
        f" {sum(f['fit_s'] for f, _ in fits):.1f} s fitting the heads at learning rates {lrs},"
        f" of which {sum(f['fit_s'] for f, chosen in fits if chosen):.1f} s for the fits kept"
        f" (learning rate per member, by validation log score:"
        f" {', '.join(f'{x["chosen_lr"]:g}' for x in m)}).",
        "",
        "Capacity, net1 v3 -> repaired (in parentheses the net1 given to --compare): 90%"
        " coverage, R^2 of the mean, CRPS (Mbit/s), log score (nats; it depends on the bins),"
        " and the median over rows of the mean belief over the truth.",
        "",
        "| view | rows | c90 | R^2 | CRPS | log score | belief / truth |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, v in rep["views"].items():
        r = {m: s[CAP] for m, s in v.items() if m in ("before", "after", "compare")}
        r = {m: s | {"ratio": v["capacity_belief_over_truth_median"][m]} for m, s in r.items()}
        cells = [
            f"{r['before'][k]:.{d}f} -> {r['after'][k]:.{d}f}"
            + (f" ({r['compare'][k]:.{d}f})" if "compare" in r else "")
            for k, d in (("c90", 3), ("r2", 3), ("crps", 2), ("log_score", 2), ("ratio", 2))
        ]
        lines.append(f"| {name} | {v['rows']} | " + " | ".join(cells) + " |")
    others = [k for k in lcfs if k != CAP]
    lines += [
        "",
        "The other LCFs, bit-identical before and after on every view (checked):"
        " 90% coverage / R^2.",
        "",
        "| view | " + " | ".join(others) + " |",
        "|" + "---|" * (len(others) + 1),
    ]
    for name, v in rep["views"].items():
        cells = [f"{v['after'][k]['c90']:.2f} / {v['after'][k]['r2']:.2f}" for k in others]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def on_training_data(
    models: dict[str, fit.Net1], spec: Net1Spec, a: argparse.Namespace
) -> dict[str, Any]:
    """Retrain on the training rows and report on their views; the data is
    freed when this returns, before the held-out sets load."""
    t0 = time.perf_counter()
    steps = data.load(a.data, seed=1, val=0.15, test=0.15, stride=STRIDE, split_by="link")
    split = np.where(np.isin(controllers(a.data, steps), list(held())), "hold", steps.split)
    if a.max_rows:
        rng = np.random.default_rng(a.seed)
        for s in ("train", "val", "test", "hold"):
            idx = np.flatnonzero(split == s)
            split[rng.choice(idx, max(0, idx.size - a.max_rows), replace=False)] = "drop"
    rep: dict[str, Any] = {"load_s": time.perf_counter() - t0}
    rep["rows"] = {k: int((split == k).sum()) for k in ("train", "val", "test", "hold")}
    before, after = models["before"], models["after"]
    c = before.lcfs.index(CAP)
    rep["bounds_before"] = before.net.bounds[c].tolist()
    rep["training"] = retrain(after, spec, steps, split, a)
    rep["bounds_after"] = after.net.bounds[c].tolist()
    old = before.net.state_dict()
    rep["changed"] = [k for k, v in after.net.state_dict().items() if not torch.equal(v, old[k])]
    keep = np.where(split == "train", "train", "other")
    views: dict[str, Rows] = {
        "val": split == "val",
        "test": split == "test",
        "test_held": (split == "hold") & (steps.split == "test"),
        "val_competing_tcp": (split == "val") & competing_tcp(a.data, steps),
    }
    rep["views"] = {name: report(models, steps, rows, keep) for name, rows in views.items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--net1", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--ood", action="append", default=[], help="NAME=DIR, reported whole")
    p.add_argument("--compare", type=Path, help="another net1.pt to report on the same views")
    p.add_argument("--bins", choices=("refit", "keep"), default="refit")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float, nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--u", type=float, nargs="+", default=[0.125, 0.25])
    a = p.parse_args()
    ood = [x.split("=", 1) for x in a.ood]
    t0 = time.perf_counter()
    before, spec = fit.load_net1(a.net1)
    models = {"before": before, "after": fit.load_net1(a.net1)[0]}
    models |= {"compare": fit.load_net1(a.compare)[0]} if a.compare else {}
    if a.epochs:
        spec = Net1Spec.model_validate(spec.model_dump() | {"epochs": a.epochs})
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"net1": str(a.net1), "compare": str(a.compare or "")}
    rep |= {"bins": a.bins, "seed": a.seed} | on_training_data(models, spec, a)
    fit.save_net1(models["after"], spec, a.out / "net1.pt")
    for name, d in ood:
        o = data.load([Path(d)], seed=1, val=0.0, test=1.0, stride=STRIDE, split_by="link")
        rep["views"][f"ood_{name}"] = report(models, o, o.rows("test"), o.split)
    export = [sys.executable, str(ROOT / "experiments/rule_export.py"), str(a.out)]
    subprocess.run([*export, str(a.out / "net1.pt"), "--u", *map(str, a.u)], check=True)
    rep["spec"] = spec.model_dump(mode="json")
    (a.out / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    (a.out / "summary.md").write_text(summary(rep, list(spec.lcfs)))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
