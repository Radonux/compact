"""experiments/belief_share.py: copies per episode and the median ratios."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parent.parent / "experiments" / "belief_share.py"
_spec = importlib.util.spec_from_file_location("belief_share", _PATH)
assert _spec is not None and _spec.loader is not None
belief_share = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(belief_share)


def test_copies_count_the_observed_flow_and_each_copy() -> None:
    assert belief_share.copies_of("clean") == 1
    assert belief_share.copies_of("itself from 0 s") == 2
    assert belief_share.copies_of("itself from 0 s, itself from 0 s, itself from 0 s") == 4


def test_ratios_are_medians_over_finite_steps() -> None:
    belief = np.array([25.0, 30.0, np.nan, 20.0])
    capacity = np.array([50.0, 50.0, 50.0, 50.0])
    receive = np.array([24.0, 26.0, 25.0, 25.0])
    m = belief_share.medians(belief, capacity, receive, 2)
    assert m["steps"] == 3
    assert m["belief_over_capacity"] == 0.5  # 25 / 50
    assert m["belief_over_share"] == 1.0  # 25 / 25
    assert m["receive_over_capacity"] == 0.5
