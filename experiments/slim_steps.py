"""Keep only what scoring and the window and label plots read in every
steps.parquet under a directory, in place: the observation vectors are
most of a steps table and no evaluation reads them.

Usage: python experiments/slim_steps.py DIR
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

KEEP = ("episode_id", "step", "t_ns", "applied_action", "cwnd_set", "cwnd", "reward", "done")


def slim(path: Path) -> None:
    src = pq.ParquetFile(path)
    cols = [c for c in src.schema_arrow.names if c in KEEP or c.startswith("label_")]
    tmp = path.with_suffix(".slim")
    writer = None
    for g in range(src.metadata.num_row_groups):
        part = src.read_row_group(g, columns=cols)  # pyright: ignore[reportUnknownMemberType]
        writer = writer or pq.ParquetWriter(tmp, part.schema)
        writer.write_table(part)
    if writer is None:  # no rows: an empty table with the kept columns
        pq.write_table(src.read(columns=cols), tmp)  # pyright: ignore[reportUnknownMemberType]
    else:
        writer.close()
    tmp.replace(path)


if __name__ == "__main__":
    for f in sorted(Path(sys.argv[1]).rglob("steps.parquet")):
        slim(f)
