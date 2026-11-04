"""Export Sage's released policy under contract sage-v1, rewritten in torch.

Sage (Yen, Abbasloo et al., SIGCOMM 2023; code and model MIT-licensed,
github.com/Soheil-ab/sage at eedd281) deploys a recurrent actor built with
sonnet 2 and acme 0.2 (sage_rl/offline/offline_agent.py, make_networks with
nw_type 1909, and rl_module/tcpactor_evalcrr.py):

    enc(x)  = tanh(LayerNorm(x W + b))                       (ControlNetwork)
    h       = GRU_1024(enc1(obs69), h)                       (snt.GRU)
    y       = neck(enc2(h)), neck = linear, then two blocks
              LayerNorm(l1(relu(l0(y))) + y)                 (LayerNormAndResidualMLP)
    action  = clip(sum softmax(logits(y)) * locs(y), -2, 2)  (mixture mean, ClipToSpec)

and the sender multiplies its window by 2^round(action, 3). The 69 inputs
are sage.cc's message at its config's obs_cols (policy.SAGE_OBS_COLS), with
NaN and infinite values set to 0 (envwapper.check_values); there is no
normalizer. So under sage-v1, a = round(action, 3) and the GRU's state is the
contract's state.

The weights come from experiments/sage_weights.py, which reads the released
checkpoint with Sage's own TensorFlow stack; it also records the actions
TensorFlow takes on a recorded episode, which --check compares with this
graph.

Usage: python experiments/sage_export.py WEIGHTS.npz OUT.onnx [--check REF_DIR]
"""

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from expliccit_env.policy import OBS_DIM, SAGE_OBS_COLS

HIDDEN = 1024
ACTION_MAX = 2.0  # config-rl-eval.yaml, tcpspec.action_max
EPS = 1e-5  # snt.LayerNorm


class _Lin(nn.Module):
    def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(w), requires_grad=False)
        self.b = nn.Parameter(torch.from_numpy(b), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b


class _Norm(nn.Module):
    def __init__(self, scale: np.ndarray, offset: np.ndarray) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.from_numpy(scale), requires_grad=False)
        self.offset = nn.Parameter(torch.from_numpy(offset), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + EPS) * self.scale + self.offset


class Sage(nn.Module):
    """(obs [B, 78], GRU state [B, 1024]) -> (a [B, 1], next state)."""

    state_dim = HIDDEN
    cols: torch.Tensor

    def __init__(self, w: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.register_buffer("cols", torch.tensor(SAGE_OBS_COLS, dtype=torch.long))
        self.enc1 = _Lin(w["enc1.lin_w"], w["enc1.lin_b"])
        self.norm1 = _Norm(w["enc1.ln_scale"], w["enc1.ln_offset"])
        self.w_i = nn.Parameter(torch.from_numpy(w["gru.w_i"]), requires_grad=False)
        self.w_h = nn.Parameter(torch.from_numpy(w["gru.w_h"]), requires_grad=False)
        self.gru_b = nn.Parameter(torch.from_numpy(w["gru.b"]), requires_grad=False)
        self.enc2 = _Lin(w["enc2.lin_w"], w["enc2.lin_b"])
        self.norm2 = _Norm(w["enc2.ln_scale"], w["enc2.ln_offset"])
        self.neck = _Lin(w["neck0.w"], w["neck0.b"])
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    _Lin(w[f"block{i}.l0.w"], w[f"block{i}.l0.b"]),
                    _Lin(w[f"block{i}.l1.w"], w[f"block{i}.l1.b"]),
                    _Norm(w[f"block{i}.ln.scale"], w[f"block{i}.ln.offset"]),
                ]
            )
            for i in range(2)
        )
        self.logit = _Lin(w["mix.logit.w"], w["mix.logit.b"])
        self.loc = _Lin(w["mix.loc.w"], w["mix.loc.b"])

    def action(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sage's clipped action before rounding, and the next state."""
        x = obs.index_select(1, self.cols)
        # NaN fails every comparison, and tract has no IsNaN or IsInf.
        x = torch.where(torch.abs(x) < 1e30, x, torch.zeros_like(x))
        e = torch.tanh(self.norm1(self.enc1(x)))
        gx = e @ self.w_i
        n = HIDDEN
        zr = torch.sigmoid(gx[:, : 2 * n] + h @ self.w_h[:, : 2 * n] + self.gru_b[: 2 * n])
        z, r = zr[:, :n], zr[:, n:]
        cand = torch.tanh(gx[:, 2 * n :] + (r * h) @ self.w_h[:, 2 * n :] + self.gru_b[2 * n :])
        h = (1 - z) * h + z * cand
        y = self.neck(torch.tanh(self.norm2(self.enc2(h))))
        for l0, l1, norm in self.blocks:  # pyright: ignore[reportGeneralTypeIssues]
            y = norm(l1(torch.relu(l0(y))) + y)
        mean = (torch.softmax(self.logit(y), dim=1) * self.loc(y)).sum(dim=1, keepdim=True)
        return torch.clamp(mean, -ACTION_MAX, ACTION_MAX), h

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, h = self.action(obs, state)
        return torch.round(a * 1000.0) / 1000.0, h


def to_onnx(model: Sage) -> bytes:
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "sage.onnx"
        torch.onnx.export(
            model.eval(),
            (torch.zeros(1, OBS_DIM), torch.zeros(1, HIDDEN)),
            path,
            input_names=["obs", "state"],
            output_names=["a", "next_state"],
            opset_version=17,
            dynamo=False,
        )
        raw = path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(raw))
    return raw


def check(model: Sage, raw: bytes, ref: Path) -> None:
    """Our graph against TensorFlow's actions on the recorded episode, step
    by step from a zero state, in torch and in the exported ONNX."""
    from onnx.reference import ReferenceEvaluator  # noqa: PLC0415

    obs = np.load(ref / "obs.npy").astype(np.float32)
    want = np.load(ref / "actions.npy")
    run = ReferenceEvaluator(onnx.load_from_string(raw))
    h = torch.zeros(1, HIDDEN)
    s = np.zeros((1, HIDDEN), np.float32)
    got, onnx_a = [], []
    with torch.no_grad():
        for o in obs:
            a, h = model.action(torch.from_numpy(o[None]), h)
            got.append(float(a[0, 0]))
            ao, s = run.run(None, {"obs": o[None], "state": s})
            onnx_a.append(float(ao[0, 0]))
    err = np.abs(np.array(got) - want)
    # In thousandths, as the sender reads them (2^round(action, 3)).
    same = np.rint(np.array(onnx_a) * 1000) == np.rint(np.round(want.astype(np.float64), 3) * 1000)
    print(f"{len(want)} steps: torch against TensorFlow, max |difference| {err.max():.2e}")
    print(f"exported a equal to TensorFlow's rounded action at {same.mean():.4f} of steps")
    assert err.max() < 1e-4, err.max()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--check", type=Path, help="sage_weights.py's output directory")
    a = p.parse_args()
    w = dict(np.load(a.weights))
    model = Sage(w).eval()
    raw = to_onnx(model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(raw)
    if a.check:
        check(model, raw, a.check)


if __name__ == "__main__":
    main()
