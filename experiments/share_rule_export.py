"""Export the share rule on a net1: the fixed rule's window, 2^u times the
believed BDP, with u raised when net1 believes the link is busy and this
flow gets a small share of it.

    u = u0 + k * busy * max(0, 1 - receive_rate / capacity)

from net1's mean beliefs (busy fraction, this flow's receive rate, the
capacity). With k = 0 it is the fixed rule. It acts on net1's output
alone, like any net2, and has two numbers to choose, so it can be searched
directly in closed loop on the validation links.

Usage: python experiments/share_rule_export.py OUT NET1.pt --u0 0.125 --k 0 0.5 1 2 [--oracle]
Writes OUT/share-u{u0}-k{k}.onnx for each k; --oracle writes
OUT/share-u{u0}-k{k}-oracle.onnx instead: the same rule on the true LCFs,
read from the guest's live truth into net1's bins (each mean is then the
true value's bin centre).

--flows N gives the rule the number of flows on the bottleneck, which is
not among the LCFs: the window aims at 1/N of the believed BDP, and the
rule pushes only while this flow gets less than 1/N of the capacity,

    u = u0 - log2(N) + k * busy * max(0, 1 - N * receive_rate / capacity)

written to OUT/share-u{u0}-k{k}-n{N}[-oracle].onnx. Without it the graph is
the one above, byte for byte. --used (with --flows) compares this flow's
share with 1/N of the capacity in use instead of the whole capacity,

    u = u0 - log2(N) + k * max(0, 1 - N * receive_rate / (busy * capacity))

so a flow alone, or one of N copies at its fair share, never pushes;
written to OUT/share-u{u0}-k{k}-n{N}-used[-oracle].onnx.
"""

import argparse
from pathlib import Path

from expliccit_env.train import fit
from expliccit_env.train.export import to_onnx
from expliccit_env.train.share import ACTION_MAX, ShareRule


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("--u0", type=float, required=True)
    p.add_argument("--k", type=float, nargs="+", required=True)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--flows", type=int, help="flows on the bottleneck, known to the rule")
    p.add_argument("--used", action="store_true", help="with --flows: share of the capacity in use")
    a = p.parse_args()
    if a.used and not a.flows:
        p.error("--used needs --flows")
    net1, _ = fit.load_net1(a.net1)
    truth = frozenset(net1.lcfs) if a.oracle else frozenset[str]()
    a.out.mkdir(parents=True, exist_ok=True)
    for k in a.k:
        rule = ShareRule(net1, a.u0, k, a.flows, a.used).eval()
        graph = fit.policy_graph(net1, rule, ACTION_MAX, action="bdp", view="means", truth=truth)
        name = f"share-u{a.u0:g}-k{k:g}" + (f"-n{a.flows}" if a.flows else "")
        name += "-used" if a.used else ""
        name += "-oracle" if a.oracle else ""
        (a.out / f"{name}.onnx").write_bytes(to_onnx(graph))
        print(name)


if __name__ == "__main__":
    main()
