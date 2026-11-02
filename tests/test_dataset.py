"""Dataset shards (expliccit_env.dataset.ShardWriter)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from expliccit_env.dataset import ShardWriter
from expliccit_env.episode import EpisodeOk


def test_a_step_stride_keeps_every_kth_step_and_leaves_other_tables_whole(tmp_path: Path) -> None:
    steps = pa.table({"episode_id": ["e"] * 10, "step": list(range(10))})
    obs = pa.table({"episode_id": ["e"] * 10, "t": list(range(10))})
    with ShardWriter(tmp_path / "s", frozenset({"steps", "obs"}), step_stride=4) as w:
        w.add(EpisodeOk("e", {"episode_id": "e", "status": "ok"}, {"steps": steps, "obs": obs}))
    assert pq.read_table(tmp_path / "s/steps.parquet")["step"].to_pylist() == [0, 4, 8]
    assert pq.read_table(tmp_path / "s/obs.parquet").num_rows == 10
    with pytest.raises(ValueError):
        ShardWriter(tmp_path / "t", step_stride=0)
