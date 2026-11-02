"""Write the net1 v3 dataset's campaigns (examples/campaign-net1-varied.json,
every controller on 180 links; examples/campaign-net1-links.json, 3 of them
on each of 1,050 more) and the four delay-threshold policies
(examples/policies/threshold-*.onnx).

Run from the repository root: uv run python examples/net1_varied.py

The policies make net1's data cover states the kernel's controllers rarely
reach: each grows the window by 2^0.01 per 10 ms step until min RTT over
smoothed RTT (message column 66) falls below theta, then shrinks it by
2^0.05 per step, so each aims at a different standing queue: (1/theta - 1)
base RTTs, from 0.11 (theta 0.9) to 9 (theta 0.1). A buffer here holds 0.25
to 4 base RTTs of queue, so theta 0.3 backs off only on the largest buffers
and theta 0.1 never does: both keep the queue full and lose most of their
packets, states no kernel controller reaches.

Every link is sampled with its own capacity, delay and buffer, so a
held-out link is a new combination. Each runs under three conditions
(clean, cross traffic, 1% random loss), and training with split_by "link"
keeps all of a link's episodes in one split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from expliccit_env.policy import OBS_DIM

ROOT = Path(__file__).parent
STOCK = ("cubic", "bbr", "bbr3", "reno", "vegas", "bic", "cdg", "highspeed", "htcp", "hybla",
         "illinois", "lp", "nv", "scalable", "veno", "westwood", "yeah")  # fmt: skip


def threshold(theta: float) -> bytes:
    """a = -0.05 if obs[66] < theta else +0.01 (NaN counts as above)."""
    init = {
        "col": np.array([66], np.int64),
        "theta": np.array([[theta]], np.float32),
        "down": np.array([[-0.05]], np.float32),
        "up": np.array([[0.01]], np.float32),
    }
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["obs", "col"], ["x"], axis=1),
            helper.make_node("Less", ["x", "theta"], ["low"]),
            helper.make_node("Where", ["low", "down", "up"], ["a"]),
        ],
        "threshold",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS_DIM])],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1])],
        initializer=[numpy_helper.from_array(v, k) for k, v in init.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def paths(n: int, rng: np.random.Generator) -> list[dict[str, object]]:
    out = []
    for _ in range(n):
        b = float(np.round(np.exp(rng.uniform(np.log(8), np.log(100))), 1))
        shape = rng.integers(3)
        levels = [b] if shape == 0 else [b, b * 2 if shape == 1 else round(b / 2, 1)]
        level_ms = int(rng.choice([3000, 7000, 15000])) if shape else 1000
        one_way = int(rng.integers(5, 51))
        bdp = b * 1e6 * 2 * one_way / 1e3 / (1500 * 8)
        buf = int(np.clip(bdp * np.exp(rng.uniform(np.log(0.25), np.log(4))), 20, 2000))
        out.append({
            "trace": {"kind": "periodic_levels", "levels_mbps": levels, "level_ms": level_ms},
            "delay": {"mahak_compat_mrtt_ms": one_way},
            "buffer": {"packets": buf},
        })  # fmt: skip
    return out


def main() -> None:
    rng = np.random.default_rng(2026)
    ccs: list[dict[str, object]] = [{"kind": "stock", "name": n} for n in STOCK]
    for theta in (0.9, 0.6, 0.3, 0.1):
        path = ROOT / "policies" / f"threshold-{theta}.onnx"
        path.write_bytes(threshold(theta))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        ccs.append({
            "kind": "agent", "mode": "hold", "noise_sigma": 0.01,
            "policy": {"path": f"examples/policies/{path.name}", "sha256": sha},
        })  # fmt: skip
    campaign = {
        "name": "net1-varied",
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": paths(180, rng)},
        "conditions": [
            {},
            {"cross": [
                {"kind": "udp_cbr", "rate_mbps": 3, "start_ms": 5000, "stop_ms": 15000},
                {"kind": "tcp", "cc": "cubic", "start_ms": 8000},
            ]},
            {"loss": {"downlink": 0.01}},
        ],
        "duration_ms": 20000,
        "seed": 2026,
        "steps": True,
    }  # fmt: skip
    (ROOT / "campaign-net1-varied.json").write_text(json.dumps(campaign, indent=1) + "\n")
    # Breadth: 1,050 more links, each with 3 of the 21 controllers. With every
    # controller on 180 links, net1 learned to recognize a training link
    # (its min RTT nearly names it) and recall its capacity and buffer;
    # many independently drawn links leave no such shortcut.
    links = campaign | {
        "name": "net1-links",
        "sweep": {"kind": "paths", "paths": paths(1050, np.random.default_rng(2027))},
        "ccs_per_point": 3,
        "seed": 2027,
    }
    (ROOT / "campaign-net1-links.json").write_text(json.dumps(links, indent=1) + "\n")


if __name__ == "__main__":
    main()
