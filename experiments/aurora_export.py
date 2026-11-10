"""Export a trained Aurora as an ONNX graph for an agent_rate agent,
rewritten in torch from PCC-RL's own code.

Aurora (Jay, Rotman, Godfrey, Schapira and Tamar, ICML 2019; PCC-RL at
github.com/PCCproject/PCC-RL, 64bea6e, Apache-2.0) acts once per monitor
interval. Its features (src/common/sender_obs.py), from an interval of
length d with packets sent s, acknowledged k and lost, and RTT samples:

- sent latency inflation: (mean of the second half of the samples less the
  mean of the first half) / d, 0 with fewer than two samples or d = 0;
- latency ratio: the mean RTT over the smallest mean RTT of any interval so
  far, 1 before there is one;
- send ratio: (8 * 1500 * s / d) / (8 * 1500 * (k - 1) / d), 1 when the
  received rate is not positive or the ratio is 1000 or more.

The policy reads the last 10 intervals, oldest first, from 10 empty ones
(0, 1, 1), through a [32, 16] tanh MLP, and its action a is the Gaussian's
mean; the rate is multiplied by 1 + 0.025 a, or divided by 1 - 0.025 a when
a < 0 (network_sim.py, DELTA_SCALE in src/common/config.py). Here the
guest gives the interval as cc_info (agent_rate's TCP_CC_INFO: length,
packets sent, delivered and lost, sample count, mean RTT and the halves'
difference, us) and the graph outputs log2 of that factor; the state is the
history (30) and the smallest mean RTT. The rate's start (6 Mbit/s) and
bounds (0.5 to 300 Mbit/s) are the guest's, PCC-RL's deployment's. Run it
as NAME=aurora.onnx@rate.

--blind exports the same policy with the RTT samples PCC-Uspace's
deployment hands it (deep-learning branch, 2c4853a,
src/pcc/rate_control/pcc_python_rc.cpp): the interval's first and last
sample only, each truncated to whole seconds by PyLong_FromLong. Under 1 s
both are 0, so the policy sees no latency at all: inflation 0 and latency
ratio 1. The guest gives no first and last sample, so they are taken as
the mean less and plus half the halves' difference; that is exact while
every RTT is under 1 s and approximate above it.

Usage:
    python experiments/aurora_export.py stream STREAM.npz
        (a synthetic stream of monitor intervals, for the check)
    python experiments/aurora_export.py export REF OUT.onnx [--blind]
        (REF: experiments/aurora_weights.py's output on STREAM.npz, which
        must sit in REF; checks the graph against PCC-RL's own code on that
        stream, then writes OUT.onnx)
"""

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn

from expliccit_env.policy import OBS_DIM

HISTORY, FEATURES = 10, 3
STATE = HISTORY * FEATURES + 1  # history, smallest mean RTT (s)
DELTA_SCALE = 0.025  # src/common/config.py
PACKET = 1500.0
EMPTY = (0.0, 1.0, 1.0)  # an empty interval's features
SRTT_COL = 2  # step.rs: us / 1e5, always finite
POLICY = ("pi/model/pi_fc0", "pi/model/pi_fc1", "pi/model/pi")  # stable-baselines' names


class Aurora(nn.Module):
    """(obs, cc_info, state) -> (a, next state)."""

    def __init__(self, w: dict[str, np.ndarray], blind: bool = False) -> None:
        super().__init__()
        self.blind = blind
        missing = [f"{p}/{x}" for p in POLICY for x in "wb" if f"{p}/{x}" not in w]
        if missing:
            raise SystemExit(f"weights lack {missing}; they hold {sorted(w)}")
        self.w = nn.ParameterList(
            nn.Parameter(
                torch.from_numpy(np.asarray(w[f"{p}/{x}"], np.float32)), requires_grad=False
            )
            for p in POLICY
            for x in "wb"
        )
        # Columns are taken by products with constant matrices: tract 0.22's
        # optimizer fails on several one-column Gathers of one input, and
        # cc_info and the state are always finite.
        eye7, eye_s = torch.eye(7), torch.eye(STATE)
        shift = torch.zeros(HISTORY * FEATURES, HISTORY * FEATURES)
        for i in range(FEATURES, HISTORY * FEATURES):
            shift[i, i - FEATURES] = 1.0
        place = torch.zeros(FEATURES, HISTORY * FEATURES)
        for i in range(FEATURES):
            place[i, (HISTORY - 1) * FEATURES + i] = 1.0
        self.register_buffer("cc_cols", eye7)
        self.register_buffer("hist_cols", eye_s[:, : HISTORY * FEATURES].contiguous())
        self.register_buffer("min_col", eye_s[:, HISTORY * FEATURES :].contiguous())
        self.register_buffer("shift", shift)
        self.register_buffer("place", place)
        self.register_buffer("out_hist", eye_s[: HISTORY * FEATURES].contiguous())
        self.register_buffer("out_min", eye_s[HISTORY * FEATURES :].contiguous())
        self.register_buffer("i_srtt", torch.tensor([SRTT_COL]))

    def actor(self, x: torch.Tensor) -> torch.Tensor:
        w = list(self.w)
        h = torch.tanh(x @ w[0] + w[1])
        h = torch.tanh(h @ w[2] + w[3])
        return h @ w[4] + w[5]

    def forward(
        self, obs: torch.Tensor, cc: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dur_us, sent, acked, _lost, n, mean_us, rise_us = (
            cc @ self.cc_cols[:, i : i + 1] for i in range(7)
        )
        hist, low = state @ self.hist_cols, state @ self.min_col
        dur = dur_us / 1e6
        some = (dur > 0).float()
        safe_dur = torch.where(dur > 0, dur, torch.ones_like(dur))
        if self.blind:  # [first, last], whole seconds; [0, 0] with no sample
            first = torch.floor((mean_us - 0.5 * rise_us) / 1e6) * (n > 0).float()
            last = torch.floor((mean_us + 0.5 * rise_us) / 1e6) * (n > 0).float()
            avg, rise = 0.5 * (first + last), last - first
        else:
            avg = torch.where(n > 0, mean_us / 1e6, torch.zeros_like(mean_us))
            rise = rise_us / 1e6
        inflation = some * rise / safe_dur
        low = torch.where(avg > 0, torch.where(low > 0, torch.minimum(low, avg), avg), low)
        safe_low = torch.where(low > 0, low, torch.ones_like(low))
        latency_ratio = torch.where(low > 0, avg / safe_low, torch.ones_like(avg))
        send_rate = some * 8.0 * PACKET * sent / safe_dur
        recv_rate = some * 8.0 * PACKET * (acked - 1.0) / safe_dur
        ok = (recv_rate > 0) & (send_rate < 1000.0 * recv_rate)
        safe_recv = torch.where(recv_rate > 0, recv_rate, torch.ones_like(recv_rate))
        send_ratio = torch.where(ok, send_rate / safe_recv, torch.ones_like(send_rate))
        f = torch.cat([inflation, latency_ratio, send_ratio], dim=1)
        hist = hist @ self.shift + f @ self.place
        a = self.actor(hist) * DELTA_SCALE
        factor = torch.where(a >= 0, 1.0 + a, 1.0 / (1.0 - a))
        # obs is the contract's; Aurora does not read it (0 times a finite column).
        out = torch.log2(factor) + 0.0 * obs.index_select(1, self.i_srtt)
        return out, hist @ self.out_hist + low @ self.out_min


def initial_state() -> np.ndarray:
    return np.array([list(EMPTY) * HISTORY + [0.0]], np.float32)


def stream(path: Path, n: int = 600, seed: int = 0) -> None:
    """Monitor intervals as the simulator would see them, over every branch
    of the features: no samples, one, many; nothing acknowledged, one
    packet, many; a rising and a falling RTT; and lengths of 10 to 600 ms."""
    rng = np.random.default_rng(seed)
    dur_us = rng.integers(10_000, 600_000, n)
    sent = rng.integers(0, 400, n)
    acked = np.where(
        rng.uniform(size=n) < 0.1,
        rng.integers(0, 2, n),
        (sent * rng.uniform(0.4, 1.2, n)).astype(int),
    )
    lost = rng.poisson(0.5, n)
    count = np.where(rng.uniform(size=n) < 0.1, rng.integers(0, 2, n), rng.integers(2, 80, n))
    base = np.clip(np.cumsum(rng.normal(0, 2_000, n)) + 150_000, 20_000, 800_000)
    rtt = [
        (base[k] + np.cumsum(rng.normal(rng.uniform(-300, 300), 500, count[k])))
        .clip(1_000)
        .astype(np.int64)
        for k in range(n)
    ]
    np.savez(
        path,
        dur_s=dur_us / 1e6,
        dur_us=dur_us,
        sent=sent,
        acked=acked,
        lost=lost,
        rtt_count=count,
        rtt_us=np.concatenate(rtt),
        rtt_s=np.concatenate(rtt) / 1e6,
    )


def guest_inputs(s: dict[str, np.ndarray]) -> np.ndarray:
    """agent_rate's TCP_CC_INFO for each interval, with the module's integer
    arithmetic: the mean and each half's mean rounded down, in us."""
    bounds = np.concatenate([[0], np.cumsum(s["rtt_count"])])
    rows = []
    for k in range(len(s["dur_us"])):
        r = s["rtt_us"][bounds[k] : bounds[k + 1]].astype(np.int64)
        half = len(r) // 2
        mean = int(r.sum()) // len(r) if len(r) else 0
        rise = int(r[half:].sum()) // (len(r) - half) - int(r[:half].sum()) // half if half else 0
        rows.append([s["dur_us"][k], s["sent"][k], s["acked"][k], s["lost"][k], len(r), mean, rise])
    return np.array(rows, np.float64)


def to_onnx(model: Aurora) -> bytes:
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "aurora.onnx"
        torch.onnx.export(
            model.eval(),
            (torch.zeros(1, OBS_DIM), torch.zeros(1, 7), torch.from_numpy(initial_state())),
            path,
            input_names=["obs", "cc_info", "state"],
            output_names=["a", "next_state"],
            opset_version=17,
            dynamo=False,
        )
        raw = path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(raw))
    return raw


def check(raw_onnx: bytes, ref: Path, blind: bool) -> None:
    """The graph against PCC-RL's own code (aurora_weights.py) on the same
    intervals, step by step: the policy's inputs and the rate factor."""
    s = dict(np.load(ref / "stream.npz"))
    tag = "_blind" if blind else ""
    want_obs, want_act = np.load(ref / f"obs{tag}.npy"), np.load(ref / f"act{tag}.npy")
    want = np.where(want_act >= 0, 1 + DELTA_SCALE * want_act, 1 / (1 - DELTA_SCALE * want_act))
    cc = guest_inputs(s)
    sess = ReferenceEvaluator(onnx.load_from_string(raw_onnx))
    state = initial_state()
    got, hist = [], []
    for k in range(len(cc)):
        feed = {
            "obs": np.zeros((1, OBS_DIM), np.float32),
            "cc_info": cc[k : k + 1].astype(np.float32),
            "state": state,
        }
        a, state = sess.run(None, feed)  # pyright: ignore[reportUnknownVariableType]
        got.append(2.0 ** float(a[0, 0]))
        hist.append(state[0, : HISTORY * FEATURES].copy())
    got_f, hist_a = np.array(got), np.array(hist)
    obs_err = np.abs(hist_a - want_obs) / np.maximum(1.0, np.abs(want_obs))
    rel = np.abs(got_f - want) / want
    print(
        f"{'blind: ' if blind else ''}policy inputs: max relative difference"
        f" from PCC-RL's {obs_err.max():.3g}"
    )
    print(
        f"rate factor: max relative difference from PCC-RL's {rel.max():.3g},",
        f"over {len(rel)} intervals",
    )
    if obs_err.max() > 1e-3 or rel.max() > 1e-4:
        raise SystemExit("the graph does not follow PCC-RL's code")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream")
    s.add_argument("stream", type=Path)
    e = sub.add_parser("export")
    e.add_argument("ref", type=Path)
    e.add_argument("out", type=Path)
    e.add_argument("--blind", action="store_true", help="PCC-Uspace's RTT samples")
    a = p.parse_args()
    if a.cmd == "stream":
        stream(a.stream)
        return
    raw = to_onnx(Aurora(dict(np.load(a.ref / "weights.npz")), a.blind))
    check(raw, a.ref, a.blind)
    a.out.write_bytes(raw)


if __name__ == "__main__":
    main()
