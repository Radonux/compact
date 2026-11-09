"""Score net1's traffic head on links it was not trained on.

Reads the c-* campaigns under DATA as experiments/competitor_info.py does
(the observed flow one controller throughout, its observations kept),
applies a head trained elsewhere (experiments/traffic_head.py) and writes
OUT/summary.md: the balanced accuracy, the confusions between copies and
a stock flow, and per set of other senders the share of steps called each
class and the mean P(copies), which a policy's copies belief reads.

Usage: python experiments/traffic_head_val.py OUT DATA NET1.pt HEAD.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from competitor_info import balanced_accuracy, load

from expliccit_env.train import fit
from expliccit_env.train.traffic import CLASSES, load_head


def main(out: Path, data: Path, net1_path: Path, head_path: Path) -> None:
    net1, _ = fit.load_net1(net1_path)
    raw, _, y, link, who = load(data, net1)
    with torch.no_grad():
        p = load_head(head_path)(torch.from_numpy(raw)).numpy()
    pred = p.argmax(1)
    copies, stock = CLASSES.index("copies"), CLASSES.index("stock")
    lines = [
        f"net1's traffic head on {len(np.unique(link))} links it was not trained on:"
        f" balanced accuracy {balanced_accuracy(pred, y):.3f}, copies called stock"
        f" {(pred[y == copies] == stock).mean():.3f}, stock called copies"
        f" {(pred[y == stock] == copies).mean():.3f}.",
        "",
        "| other senders | class | steps | called none | called copies | called stock"
        " | mean P(copies) |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in sorted(set(who.tolist())):
        m = who == w
        share = np.bincount(pred[m], minlength=len(CLASSES)) / m.sum()
        lines.append(
            f"| {w} | {CLASSES[int(y[m][0])]} | {int(m.sum())} | "
            + " | ".join(f"{s:.3f}" for s in share)
            + f" | {p[m, copies].mean():.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
