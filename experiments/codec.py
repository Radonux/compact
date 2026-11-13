"""The codec: what a sender's measurements say about its own future, squeezed
through k latents.

An encoder maps one step's message (net1's observation columns) to k
latents z. A decoder predicts, from z and the window changes the flow then
makes, what the flow measures 60 ms to about 1 s later (smoothed RTT,
delivery rate, loss) and, optionally, the terms of its utility there. The
decoder sees nothing of the step but z, so z must hold whatever state of
the path decides the flow's future under its own actions. That is the
state a controller needs, and no teacher and no label defines it
(docs/methodology.md).

With "targets": "action" the same network is a student of the logged
controllers instead: the decoder reads z and the identity of the
controller that logged the step, and predicts how that controller changes
its window up to each horizon. Then z must hold what those controllers
act on, as far as one step's message shows it.

The latents are ordered (nested dropout, Rippel et al., ICML 2014): each
example keeps a random prefix of z, so one model gives the score at every
width 0..k and latent i matters more than latent i+1. A model with
"nested": false is the same network at one fixed width, for comparison.

Usage:
  uv run python experiments/codec.py train OUT --data DIR... [--set KEY=VALUE]...
  uv run python experiments/codec.py test RUN_DIR --data DIR...

train writes config.json, history.json, metrics.json (validation links
only) and model.pt under OUT. test scores a trained run once on the test
links and writes test.json beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from expliccit_env.policy import EXPLICCIT_OBS_COLS
from expliccit_env.train import data as steps_data

# The flow's own measurements a decoder predicts, by column of the message.
SRTT, DELIVERY, LOST = 2, 7, 67
OBS_TARGETS = ("log_srtt", "log1p_delivery_mbps", "log1p_lost_mbps")
UTILITY_TARGETS = ("log_share", "queue_delay_over_base_rtt", "drop_fraction")
STEP_MS = 10


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k: Annotated[int, Field(ge=1, le=256)] = 16
    nested: bool = True
    # Share of examples that keep every latent; the rest keep a uniform
    # prefix of 0..k. Ignored when nested is false.
    p_full: Annotated[float, Field(ge=0, le=1)] = 0.25
    targets: Literal["obs", "obs+utility", "action"] = "obs"
    # Horizons in rows of the data (one row is `stride` steps of 10 ms).
    horizons: tuple[Annotated[int, Field(ge=1)], ...] = (1, 2, 4, 8, 16)
    stride: Annotated[int, Field(ge=1)] = 6
    # The encoder reads this many consecutive rows, ending at the anchor.
    frames: Annotated[int, Field(ge=1)] = 1
    width: Annotated[int, Field(ge=8)] = 256
    depth: Annotated[int, Field(ge=1)] = 3
    dec_width: Annotated[int, Field(ge=8)] = 256
    dec_depth: Annotated[int, Field(ge=1)] = 2
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.0
    lr: Annotated[float, Field(gt=0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0)] = 1e-4
    batch: Annotated[int, Field(ge=1)] = 2048
    epochs: Annotated[int, Field(ge=1)] = 20
    warmup_steps: Annotated[int, Field(ge=0)] = 500
    patience: Annotated[int, Field(ge=1)] = 5
    # Train on this many anchors drawn from the training links (all if None).
    anchors: Annotated[int, Field(ge=1)] | None = None
    seed: int = 0
    # The split of links; one value for every run that is compared.
    split_seed: int = 0
    val_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.15
    test_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.15
    # Stop training before an epoch would cross it; the run still reports.
    time_budget_s: Annotated[float, Field(gt=0)] | None = None

    @property
    def target_names(self) -> tuple[str, ...]:
        if self.targets == "action":
            return ("window_change",)
        return OBS_TARGETS + (UTILITY_TARGETS if self.targets == "obs+utility" else ())


@dataclass(frozen=True, slots=True)
class Data:
    """Rows in episode order. An anchor i is a row whose horizons i+h and
    whose earlier frames all lie in its own episode; x[i] and the frames
    before it are the encoder's input, tgt[i+h] the decoder's target (NaN
    where undefined) and lc[i+h]-lc[i] its action. For a student of the
    actions that change is the target, scaled per controller by a_scale."""

    x: Tensor  # [n, f], standardized on the training anchors
    tgt: Tensor  # [n, t], standardized on the training anchors
    lc: Tensor  # [n], log2 of the window
    ctl: Tensor  # [n], the index of the controller that logged the row
    a_scale: Tensor  # [controllers, 2, h]: mean and deviation of the window's change
    anchors: dict[str, Tensor]  # split -> row indices


def _signed_log1p(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.log1p(np.abs(v))


def _targets(s: steps_data.Steps, cfg: Config) -> np.ndarray:
    o = s.obs
    cols = [
        np.log(np.maximum(o[:, SRTT], 1e-4)),
        np.log1p(100 * np.maximum(o[:, DELIVERY], 0)),
        np.log1p(100 * np.maximum(o[:, LOST], 0)),
    ]
    if cfg.targets == "obs+utility":
        lab = s.labels
        with np.errstate(divide="ignore", invalid="ignore"):
            share = lab["receive_rate_bps"] / lab["capacity_bps"]
            cols += [
                np.log(np.maximum(share, 0.01)),
                lab["queue_delay_mean_ns"] / lab["base_rtt_ns"],
                lab["drop_fraction"],
            ]
    t = np.stack(cols, axis=1).astype(np.float32)
    t[~np.isfinite(t)] = np.nan
    return t


def build(s: steps_data.Steps, cfg: Config) -> tuple[Data, dict[str, Any]]:
    """The arrays a run trains on, and the statistics that standardize them."""
    reach = max(cfg.horizons)
    n = s.episode.size
    i = np.arange(n - reach)
    ok = (s.episode[i + reach] == s.episode[i]) & (
        s.step[i + reach] - s.step[i] == reach * cfg.stride
    )
    back = cfg.frames - 1
    past = np.maximum(i - back, 0)
    ok &= (i >= back) & (s.episode[past] == s.episode[i])
    ok &= s.step[i] - s.step[past] == back * cfg.stride
    rows = i[ok]
    anchors = {sp: rows[s.split[rows] == sp] for sp in ("train", "val", "test")}
    if cfg.anchors is not None and anchors["train"].size > cfg.anchors:
        rng = np.random.default_rng(cfg.split_seed)
        anchors["train"] = np.sort(rng.choice(anchors["train"], cfg.anchors, replace=False))
    x = _signed_log1p(np.nan_to_num(s.obs[:, EXPLICCIT_OBS_COLS], posinf=1e6, neginf=-1e6))
    tgt = _targets(s, cfg)
    tr = anchors["train"]
    lc = np.log2(np.maximum(s.cwnd, 1)).astype(np.float32)
    names = sorted(set(s.controllers))
    ctl = np.array([names.index(c) for c in s.controllers], np.int64)[s.episode]
    change = np.clip(lc[tr[:, None] + np.array(cfg.horizons)[None]] - lc[tr][:, None], -8, 8)
    own = [change[ctl[tr] == c] for c in range(len(names))]
    # A controller that never moves its window has no scale: its targets
    # become NaN and leave the loss.
    a_scale = np.stack(
        [np.stack([o.mean(0), np.where(o.std(0) > 1e-9, o.std(0), np.nan)]) for o in own]
    )
    x_mean, x_std = x[tr].mean(0), x[tr].std(0) + 1e-6
    t_mean, t_std = np.nanmean(tgt[tr], 0), np.nanstd(tgt[tr], 0) + 1e-6
    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "t_mean": t_mean.tolist(),
        "t_std": t_std.tolist(),
        "anchors": {k: int(v.size) for k, v in anchors.items()},
        "controllers": names,
    }
    return Data(
        x=torch.from_numpy(((x - x_mean) / x_std).astype(np.float32)),
        tgt=torch.from_numpy(((tgt - t_mean) / t_std).astype(np.float32)),
        lc=torch.from_numpy(lc),
        ctl=torch.from_numpy(ctl),
        a_scale=torch.from_numpy(a_scale.astype(np.float32)),
        anchors={k: torch.from_numpy(v) for k, v in anchors.items()},
    ), stats


class _Block(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, h: Tensor) -> Tensor:
        return h + self.net(h)


def _mlp(n_in: int, width: int, depth: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, width),
        *(_Block(width, dropout) for _ in range(depth)),
        nn.LayerNorm(width),
        nn.SiLU(),
        nn.Linear(width, n_out),
    )


class Codec(nn.Module):
    def __init__(self, n_in: int, n_targets: int, cfg: Config, n_ctl: int = 0) -> None:
        """n_ctl > 0 makes a student of the actions: the decoder then reads
        the controller's identity where the codec reads the actions."""
        super().__init__()
        self.k, self.h, self.n_ctl = cfg.k, len(cfg.horizons), n_ctl
        self.encoder = _mlp(n_in, cfg.width, cfg.depth, cfg.k, cfg.dropout)
        # Unit-variance latents without a learned scale, so a latent cannot
        # hide from the prefix by shrinking.
        self.norm = nn.BatchNorm1d(cfg.k, affine=False)
        side = (n_ctl or self.h) + self.h
        self.decoder = _mlp(cfg.k + side, cfg.dec_width, cfg.dec_depth, n_targets, 0.0)
        self.register_buffer("causal", torch.tril(torch.ones(self.h, self.h)))
        self.register_buffer("which", torch.eye(self.h))

    @classmethod
    def of(cls, d: Data, cfg: Config) -> Codec:
        if cfg.targets == "action":
            return cls(d.x.shape[1] * cfg.frames, 1, cfg, d.a_scale.shape[0])
        return cls(d.x.shape[1] * cfg.frames, d.tgt.shape[1], cfg)

    def encode(self, x: Tensor) -> Tensor:
        return self.norm(self.encoder(x))

    def decode(self, z: Tensor, a: Tensor) -> Tensor:
        """Predictions [b, h, t] from latents [b, k] and actions [b, h]: the
        window's log2 change up to each horizon. Horizon j sees only the
        actions up to j. For a student of the actions, a [b] is the index
        of each example's controller."""
        b = z.shape[0]
        zz = z[:, None, :].expand(b, self.h, self.k)
        if self.n_ctl:
            one = nn.functional.one_hot(a, self.n_ctl).to(z.dtype)
            aa = one[:, None, :].expand(b, self.h, self.n_ctl)
        else:
            aa = a[:, None, :] * self.causal[None]
        hh = self.which[None].expand(b, self.h, self.h)
        return self.decoder(torch.cat([zz, aa, hh], dim=2))

    def forward(self, x: Tensor, a: Tensor, keep: Tensor) -> Tensor:
        """keep [b]: how many leading latents each example keeps."""
        z = self.encode(x)
        mask = torch.arange(self.k, device=z.device)[None] < keep[:, None]
        return self.decode(z * mask, a)


def _batch(d: Data, idx: Tensor, cfg: Config) -> tuple[Tensor, Tensor, Tensor]:
    j = idx[:, None] + torch.tensor(cfg.horizons)[None]
    a = (d.lc[j] - d.lc[idx][:, None]).clamp(-8, 8)
    x = d.x[idx[:, None] - torch.arange(cfg.frames)[None]].flatten(1)
    if cfg.targets == "action":
        scale = d.a_scale[d.ctl[idx]]
        return x, d.ctl[idx], ((a - scale[:, 0]) / scale[:, 1])[:, :, None]
    return x, a, d.tgt[j]


def _still(d: Data, idx: Tensor, cfg: Config, y: Tensor) -> Tensor:
    """Persistence: each target keeps its value at the anchor; for a
    student of the actions, the window does not change."""
    if cfg.targets == "action":
        scale = d.a_scale[d.ctl[idx]]
        return (-scale[:, 0] / scale[:, 1])[:, :, None]
    return d.tgt[idx][:, None, :].expand_as(y)


def _loss(pred: Tensor, y: Tensor) -> Tensor:
    m = torch.isfinite(y)
    return nn.functional.huber_loss(pred[m], y[m])


def _keep(n: int, cfg: Config, gen: torch.Generator) -> Tensor:
    if not cfg.nested:
        return torch.full((n,), cfg.k)
    keep = torch.randint(0, cfg.k + 1, (n,), generator=gen)
    full = torch.rand(n, generator=gen) < cfg.p_full
    return torch.where(full, torch.full_like(keep, cfg.k), keep)


@torch.no_grad()
def errors(model: Codec | None, d: Data, idx: Tensor, cfg: Config, keep: int = 0) -> np.ndarray:
    """[2, h, t]: the mean squared error on the anchors idx, and each
    target's variance there. With a model, its predictions from `keep`
    leading latents; without one, persistence: each target keeps the value
    it has at the anchor."""
    targets = 1 if cfg.targets == "action" else d.tgt.shape[1]
    n = torch.zeros(len(cfg.horizons), targets, dtype=torch.float64)
    sy, syy, se = n.clone(), n.clone(), n.clone()
    if model is not None:
        model.eval()
    for s in range(0, idx.numel(), 65536):
        x, a, y = _batch(d, idx[s : s + 65536], cfg)
        if model is None:
            pred = _still(d, idx[s : s + 65536], cfg, y)
        else:
            pred = model(x, a, torch.full((x.shape[0],), keep))
        m = torch.isfinite(y) & torch.isfinite(pred)
        y0 = torch.where(m, y, torch.zeros_like(y)).double()
        n += m.sum(0)
        sy += y0.sum(0)
        syy += (y0**2).sum(0)
        se += (torch.where(m, pred - y, torch.zeros_like(y)).double() ** 2).sum(0)
    n = n.clamp(min=1)
    return torch.stack([se / n, (syy - sy**2 / n) / n]).numpy()


def r2(model: Codec, d: Data, idx: Tensor, cfg: Config, keep: int) -> np.ndarray:
    """R^2 [h, t] of the predictions from `keep` leading latents."""
    mse, var = errors(model, d, idx, cfg, keep)
    return 1 - mse / np.maximum(var, 1e-12)


def curve(model: Codec, d: Data, idx: Tensor, cfg: Config) -> dict[str, Any]:
    """The scores at each width the model was trained to work at. `score`
    is the mean R^2 over horizons and targets. `skill` is the mean of
    1 - MSE / persistence's MSE: 0 for a codec that only copies the
    present, 1 for a perfect forecast."""
    still, var = errors(None, d, idx, cfg)
    widths = range(cfg.k + 1) if cfg.nested else (cfg.k,)
    rows = []
    for keep in widths:
        mse, _ = errors(model, d, idx, cfg, keep)
        rows.append(
            {
                "keep": keep,
                "score": float((1 - mse / np.maximum(var, 1e-12)).mean()),
                "skill": float((1 - mse / np.maximum(still, 1e-12)).mean()),
                "r2": (1 - mse / np.maximum(var, 1e-12)).round(4).tolist(),
                "skill_by": (1 - mse / np.maximum(still, 1e-12)).round(4).tolist(),
            }
        )
    return {
        "persistence_r2": (1 - still / np.maximum(var, 1e-12)).round(4).tolist(),
        "widths": rows,
    }


def fit(d: Data, cfg: Config, log: Any = print) -> tuple[Codec, list[dict[str, float]]]:
    """Train on the training anchors; keep the weights with the lowest
    validation loss (over the same random prefixes every epoch)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = Codec.of(d, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr, va = d.anchors["train"], d.anchors["val"]
    per_epoch = max(1, tr.numel() // cfg.batch)
    total = per_epoch * cfg.epochs

    def rate(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup_steps) / max(1, total)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    va = va[torch.randperm(va.numel(), generator=torch.Generator().manual_seed(0))[:200_000]]
    va_keep = _keep(va.numel(), cfg, torch.Generator().manual_seed(0))
    history: list[dict[str, float]] = []
    best, best_state, since, t0 = math.inf, model.state_dict(), 0, time.monotonic()
    for epoch in range(cfg.epochs):
        model.train()
        perm = tr[torch.randperm(tr.numel(), generator=gen)]
        run = 0.0
        for b in range(per_epoch):
            x, a, y = _batch(d, perm[b * cfg.batch : (b + 1) * cfg.batch], cfg)
            loss = _loss(model(x, a, _keep(x.shape[0], cfg, gen)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        model.eval()
        with torch.no_grad():
            v = 0.0
            for s in range(0, va.numel(), 65536):
                x, a, y = _batch(d, va[s : s + 65536], cfg)
                v += _loss(model(x, a, va_keep[s : s + 65536]), y).item() * x.shape[0]
            v /= va.numel()
        elapsed = time.monotonic() - t0
        history.append(
            {"epoch": epoch, "train": run / per_epoch, "val": v, "lr": sched.get_last_lr()[0]}
            | {"s": elapsed}
        )
        log(json.dumps(history[-1]))
        if v < best:
            best, since = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            since += 1
        over = cfg.time_budget_s is not None and elapsed * (epoch + 2) / (epoch + 1) > (
            cfg.time_budget_s
        )
        if since >= cfg.patience or over:
            break
    model.load_state_dict(best_state)
    return model, history


def _load(dirs: list[Path], cfg: Config) -> tuple[Data, dict[str, Any]]:
    s = steps_data.load(
        dirs,
        seed=cfg.split_seed,
        val=cfg.val_fraction,
        test=cfg.test_fraction,
        split_by="link",
    )
    return build(s, cfg)


def _latents(model: Codec, d: Data, idx: Tensor, cfg: Config) -> dict[str, Any]:
    """How the latents behave on held-out anchors: their spread, and how
    far each is from independent of the others."""
    with torch.no_grad():
        model.eval()
        z = model.encode(_batch(d, idx[:200_000], cfg)[0])
    c = torch.corrcoef(z.T).nan_to_num()
    off = c - torch.diag(torch.diag(c))
    return {"std": z.std(0).tolist(), "max_abs_corr": off.abs().max(0).values.tolist()}


def train(out: Path, dirs: list[Path], cfg: Config) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d, stats = _load(dirs, cfg)
    (out / "config.json").write_text(cfg.model_dump_json(indent=1) + "\n")
    model, history = fit(d, cfg)
    va = d.anchors["val"]
    metrics = {
        "targets": cfg.target_names,
        "horizons_ms": [h * cfg.stride * STEP_MS for h in cfg.horizons],
        "anchors": stats["anchors"],
        "epochs_run": len(history),
        "best_val_loss": min(h["val"] for h in history),
        "train_loss_at_best": min(history, key=lambda h: h["val"])["train"],
        "seconds": history[-1]["s"],
        "val": curve(model, d, va, cfg),
        "latents": _latents(model, d, va, cfg),
    }
    (out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    torch.save({"state": model.state_dict(), "stats": stats}, out / "model.pt")


def test(run: Path, dirs: list[Path]) -> None:
    """Score a trained run on the test links. Run it once per model, after
    every choice was made on validation."""
    cfg = Config.model_validate_json((run / "config.json").read_text())
    d, _ = _load(dirs, cfg)
    model = Codec.of(d, cfg)
    saved = torch.load(run / "model.pt", weights_only=True)
    model.load_state_dict(saved["state"])
    result = {"test": curve(model, d, d.anchors["test"], cfg)}
    (run / "test.json").write_text(json.dumps(result, indent=1) + "\n")


def _overrides(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pairs:
        key, _, value = p.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("train")
    t.add_argument("out", type=Path)
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    e = sub.add_parser("test")
    e.add_argument("run", type=Path)
    e.add_argument("--data", type=Path, nargs="+", required=True)
    a = p.parse_args()
    if a.verb == "train":
        train(a.out, a.data, Config.model_validate(_overrides(a.set)))
    else:
        test(a.run, a.data)


if __name__ == "__main__":
    main()
