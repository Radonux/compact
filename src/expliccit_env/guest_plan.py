"""The plan handed to the guest runner (``plan.json`` plus trace files).

Mirrors ``guest/src/plan.rs``, which re-validates everything because the
guest does not trust its input either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

PLAN_VERSION: Final = 7
AGENT_CCS: Final = ("agent", "agent_reno")
U32: Final = 2**32
MAX_TRACE_BYTES: Final = 65_000


@dataclass(frozen=True, slots=True)
class LinkPlan:
    """One direction: delay line plus trace-driven droptail link."""

    delay_ns: int
    trace: npt.NDArray[np.uint8]  # opportunities per ms (mahimahi.dense_counts)
    limit_pkts: int
    limit_bytes: int
    loss: int = 0  # i.i.d. drop probability times 2^32

    def __post_init__(self) -> None:
        if not 0 <= self.loss < U32:
            raise ValueError("loss must be a u32 (probability times 2^32)")
        if self.delay_ns < 0:
            raise ValueError("delay_ns must be nonnegative")
        if not (2 <= self.trace.size <= MAX_TRACE_BYTES) or self.trace[-1] == 0:
            raise ValueError(f"trace must have 2..{MAX_TRACE_BYTES} entries and end nonzero")
        if (
            self.limit_pkts < 0
            or self.limit_bytes < 0
            or (self.limit_pkts == 0 and self.limit_bytes == 0)
        ):
            raise ValueError("droptail needs a positive packet or byte limit")


@dataclass(frozen=True, slots=True)
class AgentPlan:
    """An ONNX policy (sage-v1 contract, see policy.py) run every step, with
    Gaussian exploration noise of ``noise_sigma`` added to each action, each
    draw kept for ``noise_hold`` steps."""

    policy_onnx: bytes
    noise_sigma: float = 0.0
    noise_seed: int = 0
    noise_hold: int = 1

    def __post_init__(self) -> None:
        if not self.policy_onnx:
            raise ValueError("agent needs a policy")
        if not 0.0 <= self.noise_sigma <= 2.0 or not 0 <= self.noise_seed < 2**64:
            raise ValueError("noise_sigma must be in [0, 2] and noise_seed a u64")
        if not 1 <= self.noise_hold <= 1000:
            raise ValueError("noise_hold must be in [1, 1000]")


@dataclass(frozen=True, slots=True)
class TcpCrossPlan:
    cc: str
    start_ms: int
    stop_ms: int
    # The flow's own policy when cc is an agent controller.
    agent: AgentPlan | None = None
    # Its ACKs wait this much longer than the path's (sch_flowdelay).
    extra_delay_ns: int = 0

    def __post_init__(self) -> None:
        if (self.cc in AGENT_CCS) != (self.agent is not None):
            raise ValueError("a cross flow with cc agent/agent_reno and an agent plan go together")
        if not 0 <= self.extra_delay_ns <= 10**10:
            raise ValueError("extra_delay_ns must be in [0, 10 s]")


@dataclass(frozen=True, slots=True)
class UdpCrossPlan:
    rate_bps: int
    start_ms: int
    stop_ms: int


type CrossPlan = TcpCrossPlan | UdpCrossPlan


@dataclass(frozen=True, slots=True)
class GuestPlan:
    duration_ms: int
    cc: str
    downlink: LinkPlan
    uplink: LinkPlan
    start_offset_ms: int = 10
    obs_mark: int = 7
    ring_mib: int = 64
    sysctls: tuple[tuple[str, str], ...] = ()
    agent: AgentPlan | None = None
    cross: tuple[CrossPlan, ...] = ()
    # Record Sage's steps (steps.bin); always on under a policy.
    steps: bool = False

    def __post_init__(self) -> None:
        if (self.cc in AGENT_CCS) != (self.agent is not None):
            raise ValueError("cc agent/agent_reno and an agent plan go together")

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        links: dict[str, dict[str, int | str]] = {}
        for name, link in (("downlink", self.downlink), ("uplink", self.uplink)):
            trace_file = f"{name}.trace"
            (directory / trace_file).write_bytes(link.trace.tobytes())
            links[name] = {
                "delay_ns": link.delay_ns,
                "trace_file": trace_file,
                "limit_pkts": link.limit_pkts,
                "limit_bytes": link.limit_bytes,
                "loss": link.loss,
            }
        plan: dict[str, object] = {
            "plan_version": PLAN_VERSION,
            "duration_ms": self.duration_ms,
            "start_offset_ms": self.start_offset_ms,
            "cc": self.cc,
            "obs_mark": self.obs_mark,
            "ring_mib": self.ring_mib,
            **links,
            "sysctls": [list(s) for s in self.sysctls],
            "agent": None,
            "cross": [_cross_json(c, directory) for c in self.cross],
            "steps": self.steps,
        }
        if self.agent is not None:
            plan["agent"] = _agent_json(self.agent, directory)
        (directory / "plan.json").write_text(json.dumps(plan, indent=2))


def _agent_json(a: AgentPlan, directory: Path) -> dict[str, object]:
    """The plan's agent section; the policy is written once per distinct
    file, named by its hash, so many flows can share one."""
    name = f"policy-{hashlib.sha256(a.policy_onnx).hexdigest()[:16]}.onnx"
    (directory / name).write_bytes(a.policy_onnx)
    return {
        "policy_file": name,
        "noise_sigma": a.noise_sigma,
        "noise_seed": a.noise_seed,
        "noise_hold": a.noise_hold,
    }


def _cross_json(c: CrossPlan, directory: Path) -> dict[str, object]:
    match c:
        case TcpCrossPlan(cc, start, stop, agent, extra):
            return {
                "kind": "tcp",
                "cc": cc,
                "start_ms": start,
                "stop_ms": stop,
                "extra_delay_ns": extra,
                "agent": None if agent is None else _agent_json(agent, directory),
            }
        case UdpCrossPlan(rate, start, stop):
            return {"kind": "udp_cbr", "rate_bps": rate, "start_ms": start, "stop_ms": stop}
