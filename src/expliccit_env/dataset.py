"""Versioned dataset layout. One campaign directory holds:

    campaign.json            the CampaignSpec plus provenance of the plan
    episodes.jsonl           the expanded EpisodeSpecs, one per line, in order
    shards/<k>/              output of shard k (written by one worker process)
        episodes.parquet     one row per episode: parameters, status or failure,
                             metrics, provenance
        obs.parquet          sender observations (TCP_OBS), all episodes
        labels.parquet       ground-truth labels aligned 1:1 with obs rows
        link_events.parquet  bottleneck arrivals, drops, departures
        link_opp.parquet     processed delivery opportunities per ms
        agent_action.parquet actions accepted by tcp_agent (agent episodes)
        steps.parquet        Sage's steps with their labels, reward and action
                             (episodes that record steps)
        cross_steps.parquet  the same for every cross flow a policy drives,
                             with its index in the spec's cross list (flow)
        DONE                 written last; a shard without it is incomplete

Tables share ``episode_id``. Files are written as ``*.tmp`` and renamed on
close, so readers never see a half-written file; rerunning a shard replaces
it. Compression is zstd; each episode is its own row group.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from expliccit_env.episode import EpisodeOk, EpisodeOutcome

DATASET_VERSION: Final = 7
# "steps" exists only for episodes that record them (spec.steps or a policy),
# "cross_steps" only for episodes with an agent cross flow; "flow_seconds"
# (each flow's bytes and queueing delay per second) for every episode.
TABLES: Final = (
    "obs",
    "labels",
    "link_events",
    "link_opp",
    "agent_action",
    "steps",
    "cross_steps",
    "flow_seconds",
)


class ShardWriter:
    def __init__(
        self, directory: Path, tables: frozenset[str] = frozenset(TABLES), step_stride: int = 1
    ) -> None:
        """``tables``: which per-episode tables to keep (episodes.parquet is
        always written). Dropping link_events and link_opp saves most of the
        space, but labels can then never be recomputed for those episodes.
        ``step_stride``: keep only the steps whose number is a multiple of it,
        which shrinks estimator data (each step's message already summarizes
        the steps before it) but leaves no consecutive steps for net2."""
        if step_stride < 1:
            raise ValueError(f"step_stride {step_stride} must be at least 1")
        if not tables <= frozenset(TABLES):
            raise ValueError(f"unknown tables {sorted(tables - frozenset(TABLES))}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self.dir = directory
        self.tables = tables
        self.step_stride = step_stride
        self.rows: list[dict[str, object]] = []
        self.writers: dict[str, pq.ParquetWriter] = {}

    def __enter__(self) -> Self:
        return self

    def add(self, outcome: EpisodeOutcome) -> None:
        self.rows.append(outcome.row)
        if isinstance(outcome, EpisodeOk):
            for name, table in outcome.tables.items():
                assert name in TABLES, name
                if name not in self.tables:
                    continue
                if name in ("steps", "cross_steps") and self.step_stride > 1:
                    step = np.asarray(table["step"].to_numpy(), np.int64)
                    table = table.take(np.flatnonzero(step % self.step_stride == 0))  # noqa: PLW2901
                if name not in self.writers:
                    self.writers[name] = pq.ParquetWriter(
                        self.dir / f"{name}.parquet.tmp", table.schema, compression="zstd"
                    )
                self.writers[name].write_table(table, row_group_size=max(1, table.num_rows))

    def close(self) -> None:
        for name, w in self.writers.items():
            w.close()
            (self.dir / f"{name}.parquet.tmp").rename(self.dir / f"{name}.parquet")
        if self.rows:
            # Failed rows lack the metric columns: take the union of keys, in
            # first-seen order, and let missing values be null.
            keys = list(dict.fromkeys(k for row in self.rows for k in row))
            table = pa.Table.from_pydict({k: [row.get(k) for row in self.rows] for k in keys})
            # pyarrow-stubs leave write_table partly untyped.
            pq.write_table(table, self.dir / "episodes.parquet", compression="zstd")  # pyright: ignore[reportUnknownMemberType]
        (self.dir / "DONE").write_text(
            json.dumps({"dataset_version": DATASET_VERSION, "episodes": len(self.rows)})
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
            return
        for w in self.writers.values():
            w.close()
