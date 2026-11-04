"""Read Sage's released checkpoint with Sage's own stack and write its
weights by module path, for experiments/sage_export.py, plus the actions
Sage's evaluation actor takes on a recorded episode, for its --check.

It builds the policy as sage_rl/rl_module/tcpactor_evalcrr.py does
(make_networks with nw_type 1909: ControlNetwork, GRU, ControlNetwork,
LayerNormAndResidualMLP, a five-component Gaussian mixture; then the
mixture's mean, clipped to the action bounds), restores
models/ckpt-1 into it, and feeds it the recorded message at
config-rl-eval.yaml's obs_cols with NaN and infinity set to 0, from a zero
state, as envwapper.py does.

Needs Python 3.8 with Sage's pinned stack, for example:
    uv venv --python 3.8 ENV && uv pip install --python ENV/bin/python \
        tensorflow-cpu==2.4.1 tensorflow-probability==0.12.1 dm-sonnet==2.0.0 \
        dm-tree==0.1.5 numpy==1.19.5 dm-env==1.3 absl-py ruamel.yaml protobuf==3.20.3 \
        pillow gym "matplotlib<3.4"
    ENV/bin/python -m ensurepip && ENV/bin/python -m pip install --no-deps dm-acme==0.2.0
(dm-acme's metadata names a git:// URL that uv rejects, hence pip.)

Usage: ENV/bin/python experiments/sage_weights.py SAGE_CHECKOUT OBS.npy OUT
    SAGE_CHECKOUT  github.com/Soheil-ab/sage at eedd281 (tests/sage_ref.py)
    OBS.npy        float32 [N, 78], sage-v1 messages of one episode
    OUT            gets weights.npz, actions.npy, obs.npy and obs69.npy
"""

import sys
from pathlib import Path

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import specs
from acme.tf import networks
from acme.tf import utils as tf2_utils
from ruamel.yaml import YAML

sage, obs_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sage)
from sage_rl.offline.unplug_networks import ControlNetwork  # noqa: E402

cfg = YAML().load(Path(f"{sage}/sage_rl/rl_module/config-rl-eval.yaml").read_text())
oc, ts = cfg["offline_config"], cfg["tcpspec"]
amax = float(ts["action_max"])
action_spec = specs.BoundedArray((1,), np.float32, minimum=[-amax], maximum=[amax])
obs_spec = specs.BoundedArray((ts["obs_dim"],), np.float32, minimum=[-1e6], maximum=[1e6])
# make_networks(nw_type=1909) with tcpactor_evalcrr's arguments (act_fn left at "tanh").
head = networks.MultivariateGaussianMixture(num_components=5, num_dimensions=1)
policy = snt.DeepRNN(
    [
        ControlNetwork(proprio_encoder_size=oc["p_enc_size"], activation=tf.nn.tanh),
        snt.GRU(oc["policy_lstm_size"]),
        ControlNetwork(proprio_encoder_size=oc["p_enc_size"], activation=tf.nn.tanh),
        networks.LayerNormAndResidualMLP(
            hidden_size=oc["p_mlp_size"], num_blocks=oc["p_mlp_depth"]
        ),
        head,
    ]
)
tf2_utils.create_variables(network=policy, input_spec=[obs_spec])
ck = tf.train.Checkpoint(policy=policy)
status = ck.restore(tf.train.latest_checkpoint(f"{sage}/models"))
status.assert_existing_objects_matched()
eval_policy = snt.DeepRNN([policy, networks.StochasticMeanHead(), networks.ClipToSpec(action_spec)])

cols = list(ts["obs_cols"])
obs = np.load(obs_path)[:, cols].astype(np.float64)
obs[~np.isfinite(obs)] = 0.0  # envwapper.check_values
obs = obs.astype(np.float32)
state = eval_policy.initial_state(1)
acts = []
for o in obs:
    a, state = eval_policy(tf.constant(o[None]), state)
    acts.append(float(a.numpy()[0, 0]))
np.save(f"{out}/actions.npy", np.array(acts, np.float32))
np.save(f"{out}/obs69.npy", obs)
np.save(f"{out}/obs.npy", np.load(obs_path))  # the messages, for sage_export.py --check
enc1, gru, enc2, neck, mix = policy._layers


def lin(m):
    return {"w": m.w.numpy(), "b": m.b.numpy()}


def ln(m):
    return {"scale": m.scale.numpy(), "offset": m.offset.numpy()}


def enc(e):
    lin_, ln_ = e._proprio_encoder._network._layers[:2]
    return {
        **{f"lin_{k}": v for k, v in lin(lin_).items()},
        **{f"ln_{k}": v for k, v in ln(ln_).items()},
    }


W = {}
for name, e in (("enc1", enc1), ("enc2", enc2)):
    W.update({f"{name}.{k}": v for k, v in enc(e).items()})
W.update({"gru.w_i": gru._w_i.numpy(), "gru.w_h": gru._w_h.numpy(), "gru.b": gru.b.numpy()})
first, *blocks = neck._module._layers
W.update({f"neck0.{k}": v for k, v in lin(first._layers[0]).items()})
for i, blk in enumerate(blocks):
    l0, l1 = blk._layer._layers
    W.update({f"block{i}.l0.{k}": v for k, v in lin(l0).items()})
    W.update({f"block{i}.l1.{k}": v for k, v in lin(l1).items()})
    W.update({f"block{i}.ln.{k}": v for k, v in ln(blk._layer_norm).items()})
for n in ("logit", "loc", "scale"):
    W.update({f"mix.{n}.{k}": v for k, v in lin(getattr(mix, f"_{n}_layer")).items()})
np.savez(f"{out}/weights.npz", **W)
for k, v in W.items():
    print(k, v.shape)
print("actions", np.round(acts[:10], 4), "range", min(acts), max(acts), "n", len(acts))
