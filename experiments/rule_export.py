"""Export the fixed rule on a net1: net2 always answers u, so the next window
is 2^u times net1's believed bandwidth-delay product.

Usage: python experiments/rule_export.py OUT NET1.pt --u -0.25 0 0.25 [--oracle]

For each u, writes OUT/rule-u{u}.onnx, where net2 reads net1's distributions
pooled to 32 bins, and OUT/rule-u{u}-actor.pt, the same net2 as an actor (a
starting point for online training). --oracle writes OUT/rule-u{u}-oracle.onnx
instead: the same rule on the true LCFs, read from the guest's live truth.
The actor's last layer is zero and its bias atanh(u / 3), 3 being action_max.
"""

import argparse
import math
from pathlib import Path

import torch

from expliccit_env.train import fit
from expliccit_env.train.export import to_onnx
from expliccit_env.train.models import ExpliccitActor

ACTION_MAX = 3.0
BINS = 32
HIDDEN = 256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("--u", type=float, nargs="+", required=True)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    net1, _ = fit.load_net1(a.net1)
    truth = frozenset(net1.lcfs) if a.oracle else frozenset[str]()
    a.out.mkdir(parents=True, exist_ok=True)
    for u in a.u:
        torch.manual_seed(0)  # the hidden layers do not reach the output; same bytes each run
        actor = ExpliccitActor(len(net1.lcfs) * BINS, HIDDEN)
        last = actor.net[4]
        assert isinstance(last, torch.nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(math.atanh(u / ACTION_MAX))
        graph = fit.policy_graph(
            net1, actor.eval(), ACTION_MAX, action="bdp", truth=truth, input_bins=BINS
        )
        name = f"rule-u{u:g}" + ("-oracle" if a.oracle else "")
        (a.out / f"{name}.onnx").write_bytes(to_onnx(graph))
        if not a.oracle:
            torch.save(actor.state_dict(), a.out / f"{name}-actor.pt")
        print(name)


if __name__ == "__main__":
    main()
