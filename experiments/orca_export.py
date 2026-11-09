"""Export Orca's released policy as an ONNX graph for an agent_cubic agent
with a 20 ms period, rewritten in torch from Orca's own code.

Orca (Abbasloo, Yen and Chao, SIGCOMM 2020; github.com/Soheil-ab/Orca at
cafaf9a, MIT-licensed) runs Cubic and, every 20 ms, rescales Cubic's window
(src/orca-server-mahimahi.cc, rl-module/envwrapper.py, d5.py, agent.py):

- until slow start ends (the first report with ssthresh < cwnd), the window
  is multiplied by 1.1 and the agent is not asked;
- after it, the 7 features of each report are, with max_bw the largest thr
  seen so far: thr / max_bw, min(pacing_rate / max_bw, 10),
  5 * lost_rate / max_bw (all three 0 while max_bw is 0), cnt / cwnd,
  time_delta (s), min_rtt / srtt, and min(1, 1.25 * min_rtt / srtt);
- the actor reads the last 10 feature vectors, oldest first, from zeros;
- the window becomes int(int(100 * 4^a) * cwnd / 100), at least 4.

Here the guest provides the report: TCP_CC_INFO's statistics (cc_info: cnt,
the mean rate sample thr, lost bytes, the period, Orca's min RTT), and
TCP_INFO's pacing rate, smoothed RTT and ssthresh (message columns 6, 2
and 8) and window (cwnd). The graph's state is the history (70), max_bw
and whether slow start has ended. Its output a is log2 of Orca's ratio,
1.1 or int(100 * 4^a) / 100, which the agent_cubic agent applies to the
kernel's window and rounds down, as Orca's server does. Run it as
NAME=orca.onnx@cubic:20.

Usage:
    python experiments/orca_export.py stream RAW.npy
        (a synthetic stream of Orca server messages, for the check)
    python experiments/orca_export.py export REF OUT.onnx
        (REF: experiments/orca_weights.py's output on RAW.npy; checks the
        graph against Orca's own code on that stream, then writes OUT.onnx)
"""

import argparse
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn

from expliccit_env.policy import OBS_DIM

FEATURES, HISTORY = 7, 10
STATE = FEATURES * HISTORY + 2  # history, max_bw, slow start ended
EPS = 1e-3  # tf.layers.batch_normalization
SLOPE = 0.2  # tf.nn.leaky_relu
SLOW_START = math.log2(1.1)
PACING_COL, SRTT_COL, SSTHRESH_COL = 6, 2, 8  # step.rs: Mbit/s / 100, us / 1e5, packets


class Orca(nn.Module):
    """(obs, cwnd, cc_info, state) -> (a, next state)."""

    def __init__(self, w: dict[str, np.ndarray]) -> None:
        super().__init__()
        layers = []
        for fc, bn in (("fc1", "batch_normalization"), ("fc2", "batch_normalization_1")):
            scale = 1 / np.sqrt(w[f"actor/{bn}/moving_variance"] + EPS)
            kernel = w[f"actor/{fc}/kernel"] * scale
            bias = (w[f"actor/{fc}/bias"] - w[f"actor/{bn}/moving_mean"]) * scale
            layers.append((kernel, bias + w[f"actor/{bn}/beta"]))
        layers.append((w["actor/dense/kernel"], w["actor/dense/bias"]))
        self.w = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.asarray(m, np.float32)), requires_grad=False)
            for layer in layers
            for m in layer
        )
        # Columns are picked with Gather. The message's three are gathered
        # at once and split by a product with the identity (they are always
        # finite): tract 0.22's optimizer fails (pass PushSliceUp) on several
        # one-column Gathers of one input.
        for name, idx in {
            "i_hist": range(70),
            "i_keep": range(FEATURES, 70),
            "i_bw": [70],
            "i_ended": [71],
            "i_msg": [SRTT_COL, PACING_COL, SSTHRESH_COL],
            **{f"i_cc{i}": [i] for i in range(5)},
        }.items():
            self.register_buffer(name, torch.tensor(list(idx), dtype=torch.long))
        self.register_buffer("eye3", torch.eye(3))

    def col(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x.index_select(1, getattr(self, name))

    def actor(self, x: torch.Tensor) -> torch.Tensor:
        w = list(self.w)
        h = nn.functional.leaky_relu(x @ w[0] + w[1], SLOPE)
        h = nn.functional.leaky_relu(h @ w[2] + w[3], SLOPE)
        return torch.tanh(h @ w[4] + w[5])

    def forward(
        self, obs: torch.Tensor, cwnd: torch.Tensor, cc: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist, max_bw = self.col(state, "i_hist"), self.col(state, "i_bw")
        msg = self.col(obs, "i_msg")
        srtt_col, pacing_col, ssthresh = (msg @ self.eye3[:, i : i + 1] for i in range(3))
        ended = torch.maximum(self.col(state, "i_ended"), (ssthresh < cwnd).float())
        samples, thr, lost, period_us, min_rtt_us = (self.col(cc, f"i_cc{i}") for i in range(5))
        dt = period_us / 1e6
        pacing = pacing_col * 1.25e7
        srtt = srtt_col * 100.0
        min_rtt = min_rtt_us / 1000.0
        bw = torch.maximum(max_bw, thr)
        safe_bw = torch.where(bw > 0, bw, torch.ones_like(bw))
        known = (bw > 0).float()
        safe_dt = torch.where(dt > 0, dt, torch.ones_like(dt))
        safe_srtt = torch.where(srtt > 0, srtt, torch.ones_like(srtt))
        ratio = min_rtt / safe_srtt
        f = torch.cat(
            [
                known * thr / safe_bw,
                known * torch.clamp(pacing / safe_bw, max=10.0),
                known * 5.0 * (lost / safe_dt) / safe_bw,
                samples / cwnd,
                dt,
                ratio,
                torch.where(1.25 * min_rtt < srtt, 1.25 * ratio, torch.ones_like(ratio)),
            ],
            dim=1,
        )
        new_hist = torch.cat([self.col(hist, "i_keep"), f], dim=1)
        percent = torch.floor(100.0 * torch.exp(self.actor(new_hist) * math.log(4.0)))
        a = torch.where(
            ended > 0, torch.log2(percent / 100.0), torch.full_like(percent, SLOW_START)
        )
        hist = torch.where(ended > 0, new_hist, hist)
        max_bw = torch.where(ended > 0, bw, max_bw)
        return a, torch.cat([hist, max_bw, ended], dim=1)


def stream(path: Path, n: int = 600, seed: int = 0) -> None:
    """Orca server messages (orca-server-mahimahi.cc's 15 values) after slow
    start: a link stepping between capacities, a wandering window, losses
    now and then, and RTTs on both sides of 1.25 times the minimum."""
    rng = np.random.default_rng(seed)
    cap = np.repeat(rng.uniform(1e6, 1.2e7, n // 100 + 1), 100)[:n]
    cwnd = np.clip(np.cumsum(rng.normal(0, 20, n)) + 300, 10, 4000).round()
    thr = cap * rng.uniform(0.5, 1.0, n)
    pacing = thr * rng.uniform(0.8, 3.0, n)
    lost_rate = np.where(rng.uniform(size=n) < 0.15, thr * rng.uniform(0, 0.05, n), 0.0)
    srtt = rng.uniform(10, 200, n)
    min_rtt = srtt * rng.uniform(0.3, 1.0, n)
    dt = rng.integers(19_900, 21_500, n) / 1e6
    cnt = (cwnd * rng.uniform(0.05, 1.0, n)).round()
    raw = np.stack(
        [srtt, thr, cnt, dt, np.zeros(n), cwnd, pacing, lost_rate, srtt, cwnd // 2,
         cwnd, np.zeros(n), cwnd, np.full(n, 1448.0), min_rtt], axis=1,
    )  # fmt: skip
    np.save(path, raw)


def guest_inputs(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the guest would give the graph for each message: the message
    columns in step.rs's units, the window, and TCP_CC_INFO."""
    n = len(raw)
    obs = np.zeros((n, OBS_DIM))
    obs[:, SRTT_COL] = raw[:, 8] / 100.0
    obs[:, PACING_COL] = raw[:, 6] / 1.25e7
    obs[:, SSTHRESH_COL] = raw[:, 9]
    cc = np.stack(
        [raw[:, 2], raw[:, 1], raw[:, 7] * raw[:, 3], raw[:, 3] * 1e6, raw[:, 14] * 1000], 1
    )
    return obs, raw[:, 5:6], cc


def to_onnx(model: Orca) -> bytes:
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "orca.onnx"
        torch.onnx.export(
            model.eval(),
            (torch.zeros(1, OBS_DIM), torch.ones(1, 1), torch.zeros(1, 5), torch.zeros(1, STATE)),
            path,
            input_names=["obs", "cwnd", "cc_info", "state"],
            output_names=["a", "next_state"],
            opset_version=17,
            dynamo=False,
        )
        raw = path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(raw))
    return raw


def check(raw_onnx: bytes, ref: Path) -> None:
    """The graph against Orca's own code (orca_weights.py) on the same
    messages, step by step, after 5 slow-start steps it must answer 1.1 to."""
    raw = np.load(ref / "raw.npy")
    want_states, want_pct = np.load(ref / "states.npy"), np.load(ref / "percents.npy")
    obs, cwnd, cc = guest_inputs(raw)
    ss_obs = obs[:5].copy()
    ss_obs[:, SSTHRESH_COL] = 2**31 - 1  # slow start: ssthresh not yet below cwnd
    obs, cwnd, cc = np.vstack([ss_obs, obs]), np.vstack([cwnd[:5], cwnd]), np.vstack([cc[:5], cc])
    sess = ReferenceEvaluator(onnx.load_from_string(raw_onnx))
    state = np.zeros((1, STATE), np.float32)
    got_a, got_hist = [], []
    for k in range(len(obs)):
        feed = {
            "obs": obs[k : k + 1].astype(np.float32),
            "cwnd": cwnd[k : k + 1].astype(np.float32),
            "cc_info": cc[k : k + 1].astype(np.float32),
            "state": state,
        }
        a, state = sess.run(None, feed)  # pyright: ignore[reportUnknownVariableType]
        got_a.append(float(a[0, 0]))
        got_hist.append(state[0, :70].copy())
    ss = np.array(got_a[:5])
    assert np.allclose(ss, SLOW_START, atol=1e-6), f"slow start answered {ss}"
    assert not np.any(got_hist[4]), "slow start must not record history"
    got_pct = np.round(100 * 2.0 ** np.array(got_a[5:])).astype(int)
    hist_err = np.abs(np.array(got_hist[5:]) - want_states).max()
    same = int(np.sum(got_pct == want_pct))
    print(f"slow start: {len(ss)} steps of 1.1, no history")
    print(f"actor inputs: max |ours - Orca's| {hist_err:.3g}")
    print(f"window percent equal to Orca's at {same} of {len(want_pct)} steps", end="")
    bad = np.flatnonzero(got_pct != want_pct)
    if bad.size:
        print(f"; elsewhere off by at most {np.abs(got_pct - want_pct).max()}")
    else:
        print()
    if hist_err > 1e-3 or len(want_pct) - same > len(want_pct) // 100:
        raise SystemExit("the graph does not follow Orca's code")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream")
    s.add_argument("raw", type=Path)
    e = sub.add_parser("export")
    e.add_argument("ref", type=Path)
    e.add_argument("out", type=Path)
    a = p.parse_args()
    if a.cmd == "stream":
        stream(a.raw)
        return
    w = dict(np.load(a.ref / "weights.npz"))
    raw = to_onnx(Orca(w))
    check(raw, a.ref)
    a.out.write_bytes(raw)


if __name__ == "__main__":
    main()
