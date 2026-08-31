#!/usr/bin/env python3
"""Produce a copy of a LeRobot v3 dataset with the gripper removed from the
`observation.state` INPUT only (action is left untouched).

The cart7 datasets store `observation.state` as a 7-dim vector:

    [x, y, z, roll, pitch, yaw, gripper]

ACT auto-derives its `observation.state` input shape from the dataset, so a
7-dim dataset trains a policy that *sees* the current gripper value. To train
ACT with no gripper state as input (while still commanding the gripper via the
7-dim `action`), this script writes a new dataset with `observation.state`
sliced down to indices 0:6 — i.e. the gripper column dropped.

`action`, videos, tasks and every other column are copied verbatim, so the
policy still regresses a 7-dim action including the gripper command.

It rewrites everything that hard-codes the state width:
  * data/**/*.parquet                      observation.state column 7 -> 6
  * meta/info.json                         feature shape + names
  * meta/stats.json                        9 stat arrays 7 -> 6
  * meta/episodes/**/*.parquet             stats/observation.state/* 7 -> 6

Usage:
    python drop_gripper_state.py SRC_DATASET DST_DATASET
    # defaults below if no args given
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Indices of observation.state to KEEP (x, y, z, roll, pitch, yaw) — drop gripper.
KEEP = slice(0, 6)
KEEP_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
STATE_KEY = "observation.state"
# Per-feature stat fields that are one value *per state dimension* (so must be
# sliced). `count` is a single scalar per feature and is left untouched.
DIM_STAT_FIELDS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")

DEFAULT_SRC = Path("/home/batur/Coding/data/datasets/real/pickplace_cart7_v2_angleaxis")
DEFAULT_DST = Path("/home/batur/Coding/data/datasets/real/pickplace_cart7_v2_angleaxis_nogrip")


def slice_data_parquet(path: Path) -> None:
    """Slice the fixed-size observation.state column of one data parquet file."""
    table = pq.read_table(path)
    idx = table.schema.get_field_index(STATE_KEY)
    if idx < 0:
        raise KeyError(f"{STATE_KEY} not found in {path}")

    arr = np.asarray(table.column(STATE_KEY).to_pylist(), dtype=np.float32)
    if arr.shape[1] == len(KEEP_NAMES):
        return  # already sliced
    sliced = np.ascontiguousarray(arr[:, KEEP])

    new_col = pa.FixedSizeListArray.from_arrays(
        pa.array(sliced.reshape(-1), type=pa.float32()), len(KEEP_NAMES)
    )
    new_field = pa.field(STATE_KEY, pa.list_(pa.float32(), len(KEEP_NAMES)))
    table = table.set_column(idx, new_field, new_col)
    pq.write_table(table, path)


def slice_episodes_parquet(path: Path) -> None:
    """Slice the stats/observation.state/* list columns of one episodes file."""
    table = pq.read_table(path)
    for field in DIM_STAT_FIELDS:
        col_name = f"stats/{STATE_KEY}/{field}"
        idx = table.schema.get_field_index(col_name)
        if idx < 0:
            continue
        rows = table.column(col_name).to_pylist()  # list[list[float]], per episode
        if rows and len(rows[0]) == len(KEEP_NAMES):
            continue  # already sliced
        sliced = [row[KEEP] for row in rows]
        new_col = pa.array(sliced, type=table.schema.field(idx).type)
        table = table.set_column(idx, table.schema.field(idx), new_col)
    pq.write_table(table, path)


def rewrite_info_json(path: Path) -> None:
    info = json.loads(path.read_text())
    feat = info["features"][STATE_KEY]
    feat["shape"] = [len(KEEP_NAMES)]
    feat["names"] = list(KEEP_NAMES)
    path.write_text(json.dumps(info, indent=4) + "\n")


def rewrite_stats_json(path: Path) -> None:
    stats = json.loads(path.read_text())
    s = stats[STATE_KEY]
    for field in DIM_STAT_FIELDS:
        if field in s and isinstance(s[field], list):
            s[field] = s[field][KEEP]
    path.write_text(json.dumps(stats, indent=4) + "\n")


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DST

    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"No meta/info.json under {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    print(f"Copying {src}  ->  {dst}")
    shutil.copytree(src, dst)

    data_files = sorted((dst / "data").rglob("*.parquet"))
    print(f"Slicing {len(data_files)} data parquet file(s)...")
    for f in data_files:
        slice_data_parquet(f)

    ep_dir = dst / "meta" / "episodes"
    ep_files = sorted(ep_dir.rglob("*.parquet")) if ep_dir.exists() else []
    print(f"Slicing {len(ep_files)} episodes parquet file(s)...")
    for f in ep_files:
        slice_episodes_parquet(f)

    print("Rewriting meta/info.json and meta/stats.json...")
    rewrite_info_json(dst / "meta" / "info.json")
    stats_json = dst / "meta" / "stats.json"
    if stats_json.exists():
        rewrite_stats_json(stats_json)

    print(f"Done. observation.state is now {len(KEEP_NAMES)}-dim {KEEP_NAMES} "
          f"(gripper dropped from input); action unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
