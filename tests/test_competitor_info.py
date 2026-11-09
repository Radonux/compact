"""experiments/competitor_info.py: the class of an episode's other traffic."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "competitor_info.py"
_spec = importlib.util.spec_from_file_location("competitor_info", _PATH)
assert _spec is not None and _spec.loader is not None
competitor_info = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(competitor_info)


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("clean", "none"),
        ("loss 3%", "none"),
        ("ACK loss 1%", "none"),
        ("itself from 0 s", "copies"),
        ("itself from 0 s, itself from 0 s, itself from 0 s", "copies"),
        ("cubic from 0 s", "stock"),
        ("bbr from 4 s", "stock"),
        ("cubic from 2 s, cubic from 10 s", "stock"),
        ("UDP 3 Mbit/s from 5 s", "none"),
        ("UDP 3 Mbit/s from 5 s, cubic from 8 s", "stock"),
    ],
)
def test_other_traffic_classes(condition: str, expected: str) -> None:
    assert competitor_info.CLASSES[competitor_info.traffic(condition)] == expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("clean", "clean"),
        ("loss 3%", "loss 3%"),
        ("itself from 0 s, itself from 0 s", "itself"),
        ("cubic from 2 s, cubic from 10 s", "cubic"),
        ("UDP 3 Mbit/s from 5 s, cubic from 8 s", "UDP 3 Mbit/s, cubic"),
    ],
)
def test_senders_name_the_other_traffic_without_start_times(condition: str, expected: str) -> None:
    assert competitor_info.senders(condition) == expected
