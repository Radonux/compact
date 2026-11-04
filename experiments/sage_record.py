"""Record the sage-v1 messages of one Cubic episode, for checking Sage's
converted policy against TensorFlow (experiments/sage_weights.py and
sage_export.py --check). The episode is tests/test_sage.py's: 24 Mbit/s
stepping to 12, base RTT 40 ms, 80-packet buffer, 14 s, seed 9.

Usage: python experiments/sage_record.py OUT.npy --build UML_DIR
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from expliccit_env.episode import EpisodeOk, run_episode
from expliccit_env.policy import OBS_DIM
from expliccit_env.spec import EpisodeSpec
from expliccit_env.uml import UmlBuild

SPEC = {
    "path": {
        "trace": {"kind": "mahak_step", "bw_mbps": 24, "step": 0.5},
        "delay": {"base_rtt_ms": 40},
        "buffer": {"packets": 80},
    },
    "cc": {"kind": "stock", "name": "cubic"},
    "duration_ms": 14_000,
    "seed": 9,
    "steps": True,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--build", type=Path, required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_episode(EpisodeSpec.parse(json.dumps(SPEC)), UmlBuild.at(a.build), Path(tmp))
    if not isinstance(out, EpisodeOk):
        raise SystemExit(f"episode failed: {out.row.get('failure')}")
    obs = np.asarray(out.tables["steps"]["obs"].combine_chunks().flatten(), np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, obs.reshape(-1, OBS_DIM))
    print(f"{obs.size // OBS_DIM} steps")


if __name__ == "__main__":
    main()
