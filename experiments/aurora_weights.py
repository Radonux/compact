"""Run a trained Aurora with PCC-RL's own code on a stream of monitor
intervals, and write the policy's weights by name, for
experiments/aurora_export.py and its --check.

Aurora (Jay, Rotman, Godfrey, Schapira and Tamar, ICML 2019; PCC-RL at
github.com/PCCproject/PCC-RL, 64bea6e, Apache-2.0) turns each monitor
interval into three features with src/common/sender_obs.py
(SenderMonitorInterval: sent latency inflation, latency ratio, send ratio),
keeps the last 10 intervals, oldest first, in its SenderHistory, and asks
the policy src/udt-plugins/testing/loaded_agent.py loads from the
SavedModel stable_solve.py writes (signature serving_default, input ob,
output act: the Gaussian's mean). This runs those pieces from PCC-RL's
source, the SavedModel under TensorFlow's v1 compatibility layer. Each
interval is built as PCC-RL's simulator builds it (network_sim.py,
get_run_data): one window for sending and receiving, packets of 1500
bytes, and every RTT sample.

Needs a TensorFlow 2 environment (the one experiments/sage_weights.py uses
works): ENV/bin/python experiments/aurora_weights.py PCC_RL MODEL STREAM.npz OUT
    PCC_RL      the repository at 64bea6e
    MODEL       the SavedModel directory (experiments/aurora-train.json)
    STREAM.npz  per interval: dur_s, sent, acked, lost (packets), and the
                RTT samples (rtt_s, flat, split by rtt_count)
    OUT         gets weights.npz, obs.npy (the policy's inputs) and act.npy
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

pcc, model, stream_path, out = (Path(a).resolve() for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)
tf1 = tf.compat.v1
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1  # loaded_agent.py is TensorFlow 1 code
sys.path.insert(0, str(pcc / "src"))
sys.path.insert(0, str(pcc / "src" / "udt-plugins" / "testing"))
from common import sender_obs  # noqa: E402
from loaded_agent import LoadedModel  # noqa: E402

FEATURES = ["sent latency inflation", "latency ratio", "send ratio"]
PACKET = 1500

s = np.load(stream_path)
bounds = np.concatenate([[0], np.cumsum(s["rtt_count"])])
history = sender_obs.SenderHistory(10, FEATURES, 0)
agent = LoadedModel(str(model))
obs, act = [], []
for k in range(len(s["dur_s"])):
    mi = sender_obs.SenderMonitorInterval(
        0,
        bytes_sent=float(s["sent"][k]) * PACKET,
        bytes_acked=float(s["acked"][k]) * PACKET,
        bytes_lost=float(s["lost"][k]) * PACKET,
        send_start=0.0,
        send_end=float(s["dur_s"][k]),
        recv_start=0.0,
        recv_end=float(s["dur_s"][k]),
        rtt_samples=[float(r) for r in s["rtt_s"][bounds[k] : bounds[k + 1]]],
        packet_size=PACKET,
    )
    history.step(mi)
    x = history.as_array()
    obs.append(x)
    act.append(float(np.asarray(agent.act(x.reshape(1, -1))["act"]).reshape(-1)[0]))
weights = {v.name.split(":")[0]: agent.sess.run(v) for v in tf1.global_variables()}
np.savez(out / "weights.npz", **weights)
np.save(out / "obs.npy", np.array(obs))
np.save(out / "act.npy", np.array(act))
print(f"{len(weights)} arrays: {sorted(weights)}; {len(act)} intervals")
