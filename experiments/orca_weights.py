"""Read Orca's released checkpoint with Orca's own actor code and write its
weights by name, for experiments/orca_export.py, plus the actions Orca's
actor takes on given states, for its --check.

Orca (Abbasloo, Yen and Chao, SIGCOMM 2020; github.com/Soheil-ab/Orca at
cafaf9a, MIT-licensed) evaluates with rl-module/d5.py, which builds
rl-module/agent.py's Actor (dense 256, batch norm without scale, leaky
ReLU, twice, then a dense tanh output) on 7 features times 10 steps of
history and restores train_dir/learner0/model.ckpt-1283529 (the same file
as models/). At evaluation it runs the actor with is_training False and no
noise. This builds that Actor, from Orca's own source, under TensorFlow's
v1 compatibility layer, restores the actor's variables, and runs it.

Needs a TensorFlow 2 environment (the one experiments/sage_weights.py uses
works): ENV/bin/python experiments/orca_weights.py ORCA_CHECKOUT STATES.npy OUT
    STATES.npy  float32 [N, 70], Orca's state histories (oldest step first)
    OUT         gets weights.npz, states.npy and actions.npy
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

orca, states_path, out = (Path(a).resolve() for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)  # Orca's utils.py logs to ./rl_logging on import
tf1 = tf.compat.v1
tf1.disable_v2_behavior()
# agent.py is TensorFlow 1 code: give it the v1 API under its own name.
sys.modules["tensorflow"] = tf1
sys.path.insert(0, str(orca / "rl-module"))
from agent import Actor  # noqa: E402

CKPT = orca / "models" / "model.ckpt-1283529"
states = np.load(states_path).astype(np.float32)
if states.ndim != 2 or states.shape[1] != 70:
    raise SystemExit(f"{states_path}: want [N, 70], got {states.shape}")

s0 = tf1.placeholder(tf.float32, [None, 70])
is_training = tf1.placeholder(tf.bool)
action = Actor(70, 1, h1_shape=256, h2_shape=256).build(s0, is_training)
variables = tf1.global_variables(scope="actor")
with tf1.Session() as sess:
    tf1.train.Saver(var_list=variables).restore(sess, str(CKPT))
    weights = {v.name.split(":")[0]: sess.run(v) for v in variables}  # Python 3.8
    actions = sess.run(action, {s0: states, is_training: False})
np.savez(out / "weights.npz", **weights)
np.save(out / "states.npy", states)
np.save(out / "actions.npy", actions.astype(np.float32))
print(f"{len(weights)} arrays: {sorted(weights)}; {len(states)} actions")
