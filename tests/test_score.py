"""Scores of closed-loop runs from ground truth (expliccit_env.train.score)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from expliccit_env.train import score
from expliccit_env.train.spec import Net2Spec

LOG = Net2Spec(reward="log_utility", delay_weight=0.5, loss_weight=4.0)


def spec_json(link: int, loss: float = 0.0, cross: list[dict[str, object]] | None = None) -> str:
    path = {
        "trace": {"levels_mbps": [10.0 * (link + 1)]},
        "loss": {"downlink": loss, "uplink": 0.0},
    }
    return json.dumps({"path": path, "cross": cross or [], "cc": {"kind": "stock"}, "seed": 1})


def shard(
    root: Path, name: str, episodes: list[tuple[str, str, str, str, list[float]]], done: bool = True
) -> None:
    """episodes: id, controller, spec_json, status, and each step's share of capacity."""
    d = root / "shards" / name
    d.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                k: [e[i] for e in episodes]
                for i, k in enumerate(["episode_id", "controller", "spec_json", "status"])
            }
        ),
        d / "episodes.parquet",
    )
    rows = [(e[0], s) for e in episodes for s in e[4]]
    n = len(rows)
    pq.write_table(
        pa.table(
            {
                "episode_id": [r[0] for r in rows],
                "label_receive_rate_bps": np.array([r[1] * 10e6 for r in rows], np.float32),
                "label_capacity_bps": np.full(n, 10e6, np.float32),
                "label_queue_delay_mean_ns": np.zeros(n, np.float32),
                "label_base_rtt_ns": np.full(n, 40e6, np.float32),
                "label_drop_fraction": np.zeros(n, np.float32),
            }
        ),
        d / "steps.parquet",
    )
    if done:
        (d / "DONE").touch()


def test_scores_average_episodes_per_link_and_condition_and_pair_by_link(tmp_path: Path) -> None:
    cubic = [{"kind": "tcp", "cc": "cubic", "start_ms": 8000}]
    shard(tmp_path, "00000-of-00001", [
        ("a0", "A", spec_json(0), "ok", [1.0, 1.0]),
        ("a1", "A", spec_json(1), "ok", [0.5, 0.5]),
        ("a2", "A", spec_json(0, cross=cubic), "ok", [0.25]),
        ("b0", "B", spec_json(0), "ok", [0.5, 0.5]),
        ("b1", "B", spec_json(1), "ok", [0.5]),
        ("bx", "B", spec_json(1), "failed", [0.01]),  # never counts
        ("b2", "B", spec_json(0, cross=cubic), "ok", [0.5]),
    ])  # fmt: skip
    result = score.compare(
        score.episode_scores([tmp_path], LOG), reference="B", aliases={"A": "rule"}
    )
    a, b = result["controllers"]["rule"], result["controllers"]["B"]
    assert set(a) == {"all", "clean", "cubic from 8 s"}
    assert a["clean"]["mean"] == pytest.approx((0.0 + np.log(0.5)) / 2)
    assert b["clean"]["mean"] == pytest.approx(np.log(0.5))
    assert a["clean"]["minus_reference"]["mean"] == pytest.approx(-np.log(0.5) / 2)
    # Over all conditions a link is the mean of its conditions: link 0 is (0 + log 0.25) / 2.
    assert a["all"]["links"] == 2
    assert a["all"]["mean"] == pytest.approx(((0.0 + np.log(0.25)) / 2 + np.log(0.5)) / 2)
    assert "minus_reference" not in b["all"]
    result["failed"] = score.failures([tmp_path])
    assert result["failed"] == {"B": 1}
    table = score.markdown(result)
    assert table.rstrip().endswith("Failed episodes, not scored: B 1.")
    assert table.splitlines()[0].startswith(
        "| controller | all conditions | clean | cubic from 8 s | minus B"
    )


def test_condition_labels_name_cross_traffic_and_loss() -> None:
    udp = {"kind": "udp_cbr", "rate_mbps": 3, "start_ms": 5000}
    assert score.condition(spec_json(0)) == "clean"
    assert score.condition(spec_json(0, loss=0.01)) == "loss 1%"
    assert score.condition(spec_json(0, cross=[udp])) == "UDP 3 Mbit/s from 5 s"
    policy = {"path": "p.onnx", "sha256": "ab" * 32}
    agent = {"kind": "tcp", "cc": {"kind": "agent", "policy": policy}, "start_ms": 0}
    assert score.condition(spec_json(0, cross=[agent])) == "agent:abababababab from 0 s"


def test_an_incomplete_shard_is_an_error_not_a_smaller_sample(tmp_path: Path) -> None:
    shard(tmp_path, "00000-of-00002", [("a0", "A", spec_json(0), "ok", [1.0])])
    shard(tmp_path, "00001-of-00002", [("a1", "A", spec_json(1), "ok", [1.0])], done=False)
    with pytest.raises(ValueError, match="incomplete"):
        score.episode_scores([tmp_path], LOG)
