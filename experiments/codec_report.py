"""Collect codec runs (experiments/codec.py) into one table.

Usage: uv run python experiments/codec_report.py OUT RUN_DIR...

Each RUN_DIR holds config.json and metrics.json. Writes OUT/summary.md and
OUT/runs.json: every run's settings, its validation loss, and its
validation skill over persistence at selected widths.
Runs are ordered by validation loss, the quantity the search minimizes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WIDTHS = (1, 2, 3, 4, 6, 8, 10, 12, 16)
SETTINGS = ("targets", "nested", "k", "width", "depth", "lr", "weight_decay", "batch", "seed")


def _row(run: Path) -> dict[str, Any] | None:
    try:
        cfg = json.loads((run / "config.json").read_text())
        m = json.loads((run / "metrics.json").read_text())
    except FileNotFoundError:
        return None
    by = {c["keep"]: c for c in m["val"]["widths"]}
    still = m["val"]["persistence_r2"]
    row: dict[str, Any] = {"run": run.name} | {k: cfg[k] for k in SETTINGS}
    row |= {
        "val_loss": m["best_val_loss"],
        "train_loss": m["train_loss_at_best"],
        "epochs": m["epochs_run"],
        "minutes": m["seconds"] / 60,
        "persistence": sum(map(sum, still)) / sum(map(len, still)),
        "score": by[cfg["k"]]["score"],
        "skill": by[cfg["k"]]["skill"],
    }
    row |= {f"skill{w}": by[w]["skill"] if w in by else None for w in WIDTHS}
    return row


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 0.01 or abs(v) >= 100 else f"{v:.4f}"
    return str(v)


def main() -> None:
    out, runs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    rows = [r for r in map(_row, runs) if r is not None]
    missing = len(runs) - len(rows)
    rows.sort(key=lambda r: r["val_loss"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [
        f"{len(rows)} runs, ordered by validation loss; {missing} without metrics.",
        "All numbers are on the validation links. `score` is the mean R^2 over",
        "horizons and targets with every latent kept, and `persistence` the same",
        "for predicting that nothing changes. `skill` is the mean of 1 - MSE /",
        "persistence's MSE: 0 for copying the present, 1 for a perfect forecast.",
        "`skillN` is the skill of the first N latents of a nested model (or of the",
        "model itself if N is its fixed width).",
        "",
    ]
    if rows:
        cols = list(rows[0])
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(_cell(r[c]) for c in cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
