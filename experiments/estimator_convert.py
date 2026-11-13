"""Rewrite a saved net1 whose spec is in the earlier flat form (intervals,
coverage, bins, members) into the form Net1Spec reads (output), leaving
the weights and scalers untouched, and check that it loads.

Usage: python experiments/estimator_convert.py SOURCE.pt OUT.pt
"""

import sys
from pathlib import Path

import torch

from expliccit_env.train import fit

FLAT = ("intervals", "coverage", "bins", "members")


def main(source: Path, out: Path) -> None:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    spec = dict(saved["spec"])
    if "output" not in spec:
        flat = {k: spec.pop(k) for k in FLAT}
        if flat["intervals"]:
            spec["output"] = {"kind": "interval", "coverage": flat["coverage"]}
        elif flat["bins"]:
            spec["output"] = {"kind": "bins", "bins": flat["bins"], "members": flat["members"]}
        else:
            spec["output"] = {"kind": "point"}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**saved, "spec": spec}, out)
    net1, parsed = fit.load_net1(out)
    print(f"LCFs: {', '.join(net1.lcfs)}")
    print()
    print(f"Output: {parsed.output.model_dump_json()}. Measurements read: {fit.raw_width(net1)}.")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
