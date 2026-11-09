"""Run Orca's released policy with Orca's own code on a stream of its
server's messages, and write the actor's weights by name, for
experiments/orca_export.py and its --check.

Orca (Abbasloo, Yen and Chao, SIGCOMM 2020; github.com/Soheil-ab/Orca at
cafaf9a, MIT-licensed) evaluates with rl-module/d5.py --eval: every 20 ms its
server (src/orca-server-mahimahi.cc) writes a message of 15 values to shared
memory; envwrapper.py's TCP_Env_Wrapper.get_state turns it into 7 features
(use_normalizer false in params_base.json); d5.py keeps the last 10 states,
oldest first, starting from zeros; agent.py's Actor (dense 256, batch norm
without scale, leaky ReLU, twice, then a dense tanh output) maps them to an
action without noise; and map_action sends int(100 * 4^a), the percent the
server multiplies the window by. This runs those three pieces from Orca's
source: get_state reads the messages through a stand-in for the shared
memory, the Actor is built under TensorFlow's v1 compatibility layer and
restores models/model.ckpt-1283529 (the file train_dir/learner0 holds).

Needs a TensorFlow 2 environment (the one experiments/sage_weights.py uses
works): ENV/bin/python experiments/orca_weights.py ORCA_CHECKOUT RAW.npy OUT
    RAW.npy  float64 [N, 15], the server's messages after slow start, in its
             order: delay, thr, cnt, time_delta, target, cwnd, pacing_rate,
             lost_rate, srtt_ms, snd_ssthresh, packets_out, retrans_out,
             max_packets_out, mss, min_rtt
    OUT      gets weights.npz, raw.npy, states.npy (the actor's inputs),
             actions.npy and percents.npy
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf

orca, raw_path, out = (Path(a).resolve() for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)  # Orca's utils.py logs to ./rl_logging on import
tf1 = tf.compat.v1
tf1.disable_v2_behavior()
# Orca's modules are TensorFlow 1 code with imports this check never uses.
sys.modules["tensorflow"] = tf1
sys.modules["gym"] = types.ModuleType("gym")
ipc = types.ModuleType("sysv_ipc")
ipc.ExistentialError = OSError  # pyright: ignore[reportAttributeAccessIssue]
sys.modules["sysv_ipc"] = ipc
sys.path.insert(0, str(orca / "rl-module"))
from agent import Actor  # noqa: E402
from envwrapper import TCP_Env_Wrapper  # noqa: E402
from utils import Params  # noqa: E402

raw = np.load(raw_path).astype(np.float64)
if raw.ndim != 2 or raw.shape[1] != 15:
    raise SystemExit(f"{raw_path}: want [N, 15], got {raw.shape}")


class Shm:
    """The server's shared memory: each read returns the next message."""

    def __init__(self) -> None:
        self.i = 0

    def read(self) -> bytes:
        msg = f"{self.i % 1000} " + " ".join(f"{v:.7f}" for v in raw[self.i]) + "\0"
        self.i += 1
        return msg.encode()


params = Params(str(orca / "params_base.json"))
config = types.SimpleNamespace(load=None, eval=True, task=0)
env = TCP_Env_Wrapper("TCP", params, config, False, Shm(), None, use_normalizer=False)
s_dim = params.dict["state_dim"]
states = np.zeros((len(raw), s_dim * params.dict["rec_dim"]))
buf = np.zeros(s_dim * params.dict["rec_dim"])
for k in range(len(raw)):
    s, _, _, ok = env.get_state(evaluation=True)
    if not ok:
        raise SystemExit(f"message {k}: get_state refused it")
    buf = np.concatenate((buf[s_dim:], s))  # d5.py: from zeros, oldest first
    states[k] = buf

s0 = tf1.placeholder(tf.float32, [None, states.shape[1]])
is_training = tf1.placeholder(tf.bool)
action = Actor(states.shape[1], 1, h1_shape=256, h2_shape=256).build(s0, is_training)
variables = tf1.global_variables(scope="actor")
with tf1.Session() as sess:
    tf1.train.Saver(var_list=variables).restore(sess, str(orca / "models" / "model.ckpt-1283529"))
    weights = {v.name.split(":")[0]: sess.run(v) for v in variables}  # Python 3.8
    actions = sess.run(action, {s0: states, is_training: False})[:, 0]
percents = np.array([env.map_action(a) for a in actions])
np.savez(out / "weights.npz", **weights)
np.save(out / "raw.npy", raw)
np.save(out / "states.npy", states)
np.save(out / "actions.npy", actions.astype(np.float32))
np.save(out / "percents.npy", percents)
print(f"{len(weights)} arrays; {len(raw)} steps; percents {percents.min()} to {percents.max()}")
