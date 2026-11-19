"""The LCFs net1 can learn, by name: label columns (labels.labels_at),
composites, each a fixed formula over label columns of one window, and
fitted formulas over the grounding library's features (library.py), each
written with its features in brackets.

A composite is never stored. ``Labels`` computes it from the stored columns
the first time it is asked for, so adding one needs no new data, and an LCF
set that holds a composite beside the columns it is made of is redundant by
construction: choose the set with that in mind. A windowed name follows
labels.label_name: plain for the first window, ``<name>_<w>ms`` otherwise.

A formula is written once and runs on both kinds of column: NumPy arrays
(the stored labels) and torch tensors (the guest's live truth inside an
exported policy, graph.TruthColumns), so a composite of the truth is the
composite of the labels.

Where a composite comes from is in docs/concepts.md.

A fitted formula reads features over the steps before each step, so it
needs whole episodes: its features stored when the episode was recorded
(dataset.ShardWriter), or ``Labels`` given every step in order. It has no
unit, and an exported policy cannot compute it from the live truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import numpy.typing as npt

from expliccit_env.labels import WINDOWED, WINDOWS_MS, label_name
from expliccit_env.mahimahi import PACKET_SIZE

if TYPE_CHECKING:
    import torch

type F32 = npt.NDArray[np.float32]
# A label column of the composite's window. Typed as the array it is in
# training; in a policy graph the same formulas get torch tensors.
type Get = Callable[[str], F32]

# Label columns, with the factor to the units net1's errors are reported in.
_COLUMNS: Final = {
    "queue_delay_mean_ns": (1e-6, "ms"),
    "queue_delay_ns": (1e-6, "ms"),
    "ack_queue_delay_mean_ns": (1e-6, "ms"),
    "queue_pkts": (1.0, "pkts"),
    "queue_bytes": (1.0, "bytes"),
    "queue_min_pkts": (1.0, "pkts"),
    "flow_queue_pkts": (1.0, "pkts"),
    "capacity_bps": (1e-6, "Mbit/s"),
    "capacity_ahead_bps": (1e-6, "Mbit/s"),
    "free_buffer_pkts": (1.0, "pkts"),
    "buffer_pkts": (1.0, "pkts"),
    "buffer_fill": (1.0, ""),
    "busy_fraction": (1.0, ""),
    "base_rtt_ns": (1e-6, "ms"),
    "one_way_delay_ns": (1e-6, "ms"),
    "receive_rate_bps": (1e-6, "Mbit/s"),
    "arrival_rate_bps": (1e-6, "Mbit/s"),
    "flow_arrival_rate_bps": (1e-6, "Mbit/s"),
    "active_flows": (1.0, "flows"),
    "random_loss_rate": (1.0, ""),
    "random_drop_fraction": (1.0, ""),
    "cross_traffic_share": (1.0, ""),
}


def _where(keep: npt.NDArray[np.bool_], x: F32 | torch.Tensor, other: float) -> F32:
    """x where ``keep`` holds, else ``other``."""
    if isinstance(x, np.ndarray):
        return np.where(keep, x, np.float32(other)).astype(np.float32)
    import torch  # noqa: PLC0415  (only a policy graph passes tensors)

    return cast("F32", torch.where(cast("torch.Tensor", keep), x, torch.full_like(x, other)))


def _ratio(a: F32, b: F32) -> F32:
    """a / b, NaN where b is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _where(b > 0, a / b, float("nan"))


def _no_nan(x: F32) -> F32:
    """x, 0 where it is NaN. Not nan_to_num: its IsInf is an op the guest's
    tract cannot run."""
    return _where(x == x, x, 0.0)  # noqa: PLR0124  (NaN alone differs from itself)


def _used(g: Get) -> F32:
    """The busy fraction, 0 where the link offered nothing (there it is NaN)."""
    return _no_nan(g("busy_fraction"))


def _queue_in_bdps(g: Get) -> F32:
    """The queue as a fraction of the bandwidth-delay product."""
    return _ratio(g("queue_bytes") * 8, g("capacity_bps") * g("base_rtt_ns") * np.float32(1e-9))


# name -> (formula, factor to report units, unit). C is capacity_bps, U the
# busy fraction, y the arrival rate, q the queue, R the base RTT.
COMPOSITES: Final[dict[str, tuple[Callable[[Get], F32], float, str]]] = {
    # Used = C U, what the link sent (RFC 5136; HPCC's txRate).
    "departure_rate_bps": (lambda g: g("capacity_bps") * _used(g), 1e-6, "Mbit/s"),
    # A = C (1 - U), the capacity no flow used (RFC 5136).
    "available_capacity_bps": (
        lambda g: g("capacity_bps") * (1 - _used(g)),
        1e-6,
        "Mbit/s",
    ),
    # y - C: positive while the queue grows (XCP's spare bandwidth, negated).
    "excess_demand_bps": (lambda g: g("arrival_rate_bps") - g("capacity_bps"), 1e-6, "Mbit/s"),
    # z = y / C, which keeps growing under overload where U stops at 1 (the
    # OSU scheme's load factor with the target at capacity). TIMELY's queue
    # growth rate (y - C) / C is this less 1.
    "load_factor": (lambda g: _ratio(g("arrival_rate_bps"), g("capacity_bps")), 1.0, ""),
    # C R in 1504-byte packets: the data in flight on a full link with an
    # empty queue (CCmatic).
    "bdp_pkts": (
        lambda g: g("capacity_bps") * g("base_rtt_ns") * np.float32(1e-9 / (8 * PACKET_SIZE)),
        1.0,
        "pkts",
    ),
    # q / mu, the wait of a packet that arrives now (PIE), mu the rate out.
    "littles_delay_ns": (
        lambda g: _ratio(
            g("queue_bytes") * np.float32(8e9), g("capacity_bps") * g("busy_fraction")
        ),
        1e-6,
        "ms",
    ),
    # U_j = q / (C R) + txRate / C: 1 on a full link with an empty queue (HPCC).
    "normalized_inflight": (lambda g: _queue_in_bdps(g) + g("busy_fraction"), 1.0, ""),
    # (q + C R) y / (C^2 R): PowerTCP's power over its value on a link that is
    # just full.
    "normalized_power": (
        lambda g: (1 + _queue_in_bdps(g)) * _ratio(g("arrival_rate_bps"), g("capacity_bps")),
        1.0,
        "",
    ),
    # The other flows' arrival rate (Dovrolis 2001's cross-traffic rate).
    "cross_arrival_rate_bps": (
        lambda g: g("arrival_rate_bps") - g("flow_arrival_rate_bps"),
        1e-6,
        "Mbit/s",
    ),
    # C / N, the processor-sharing rate (RCP).
    "fair_share_bps": (lambda g: _ratio(g("capacity_bps"), g("active_flows")), 1e-6, "Mbit/s"),
    # The flow's packets into the bottleneck over those out of it: above 1
    # while its queue or its losses build (Aurora's sending ratio, at the link).
    "sending_ratio": (
        lambda g: _ratio(g("flow_arrival_rate_bps"), g("receive_rate_bps")),
        1.0,
        "",
    ),
    # (R + queueing delay) / R (Aurora's latency ratio, Remy's RTT ratio, on
    # the true delays).
    "rtt_ratio": (
        lambda g: 1 + _ratio(_no_nan(g("queue_delay_mean_ns")), g("base_rtt_ns")),
        1.0,
        "",
    ),
}


def _names(name: str, windowed: bool) -> list[tuple[str, int]]:
    """(column name, window) for each window the name comes in."""
    return [(label_name(name, w), w) for w in (WINDOWS_MS if windowed else WINDOWS_MS[:1])]


# Every LCF name -> (factor, unit).
LCF_UNITS: Final[dict[str, tuple[float, str]]] = {
    column: unit for name, unit in _COLUMNS.items() for column, _ in _names(name, name in WINDOWED)
} | {
    column: (factor, unit)
    for name, (_, factor, unit) in COMPOSITES.items()
    for column, _ in _names(name, True)
}
_COMPOSITE_OF: Final = {
    column: (COMPOSITES[name][0], w) for name in COMPOSITES for column, w in _names(name, True)
}


def _getter(get: Get, window_ms: int) -> Get:
    return lambda name: get(label_name(name, window_ms) if name in WINDOWED else name)


def columns(lcfs: Iterable[str]) -> set[str]:
    """The stored label columns these LCFs are read or computed from."""
    out: set[str] = set()

    def note(name: str) -> F32:
        out.update(columns([name]))
        return np.zeros(1, np.float32)

    for name in lcfs:
        if is_formula(name):
            from expliccit_env.train import library  # noqa: PLC0415  (it reads LCF_UNITS)

            for f in library.features_of(name):
                out.update(columns([library.parse(f)[1]]))
        elif name in _COMPOSITE_OF:
            formula, w = _COMPOSITE_OF[name]
            formula(_getter(note, w))
        else:
            out.add(name)
    return out


def is_formula(name: str) -> bool:
    """Whether an LCF is a fitted formula: its features are in brackets."""
    return "[" in name


def units(name: str) -> tuple[float, str]:
    """An LCF's factor to its report units, and the units; a fitted formula
    is reported as it is, without a unit."""
    return (1.0, "") if is_formula(name) else LCF_UNITS[name]


def known(name: str) -> bool:
    """Whether a name is an LCF: a label column, a composite, or a fitted
    formula that ``library.evaluate`` reads (a feature it reads must still
    be in the data)."""
    if not is_formula(name):
        return name in LCF_UNITS
    from expliccit_env.train import library  # noqa: PLC0415  (it reads LCF_UNITS)

    try:
        for f in library.features_of(name):
            library.parse(f)
        library.evaluate(name, lambda _: np.ones(1))
    except (ValueError, SyntaxError):
        return False
    return True


def value(name: str, get: Get) -> F32:
    """An LCF from label columns: its own column, or a composite's formula
    over the columns of its window."""
    if name not in _COMPOSITE_OF:
        return get(name)
    formula, w = _COMPOSITE_OF[name]
    return formula(_getter(get, w))


class Labels(dict[str, F32]):
    """Label columns (and features stored with strided steps) by name; a
    composite or a fitted formula is computed on first use and kept.
    ``rows``, the rows' episode and step when they are every step of whole
    episodes in order, lets a formula compute the features it reads."""

    def __init__(
        self,
        columns: Mapping[str, F32] | None = None,
        rows: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]] | None = None,
    ) -> None:
        super().__init__(columns or {})
        self.rows = rows

    def __missing__(self, name: str) -> F32:
        if is_formula(name):
            from expliccit_env.train import library  # noqa: PLC0415  (it reads LCF_UNITS)

            out = library.evaluate(name, self.feature).astype(np.float32)
        elif name in _COMPOSITE_OF:
            out = value(name, self.__getitem__).astype(np.float32)
        else:
            raise KeyError(f"no label {name}: not in the data, and not a composite or formula")
        self[name] = out
        return out

    def feature(self, name: str) -> npt.NDArray[np.float64]:
        """A library feature in physical units: stored, or computed from
        whole episodes."""
        from expliccit_env.train import library  # noqa: PLC0415  (it reads LCF_UNITS)

        op, variable = library.parse(name)
        if op == "":  # a label or composite, in its physical units
            return library.physical(library.scaled(variable, self[variable]))
        if name in self:  # stored with strided steps, or computed before
            return self[name].astype(np.float64)
        if self.rows is None or np.any(
            (np.diff(self.rows[1]) != 1) & (np.diff(self.rows[0]) == 0)
        ):  # a step missing within an episode: a strided table
            raise KeyError(
                f"no feature {name}: not stored with the steps, and the rows are not every "
                "step of whole episodes (store it with campaign run --features)"
            )
        found = library.values([name], self.__getitem__, *self.rows)[name]
        self[name] = found.astype(np.float32)
        return found
