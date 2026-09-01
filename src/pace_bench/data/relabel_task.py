#!/usr/bin/env python3
"""Relabel the (single) task / instruction string of a LeRobot v3.0 dataset.

The task string lives in two places in a LeRobot v3.0 dataset:
  * meta/tasks.parquet              -- the string column (mapped to task_index)
  * meta/episodes/**/*.parquet      -- the per-episode `tasks` list column
The per-frame data shards only store the integer `task_index`, so they are
left untouched.

Usage:
    python relabel_task.py <dataset_dir> "<new task string>"

Safety:
  * asserts the dataset has exactly ONE task,
  * validates every episode-meta file before writing anything,
  * copies the whole meta/ dir to meta_backup_<timestamp>/ first,
  * rewrites parquet via pyarrow, preserving each file's exact schema,
  * re-reads and verifies afterwards.
Re-runnable: it relabels whatever string is currently set, so you can run it
again with different wording at any time.
"""
import sys
import os
import glob
import shutil
import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python relabel_task.py <dataset_dir> "<new task string>"')
    base = os.path.abspath(sys.argv[1])
    new = sys.argv[2]
    meta = os.path.join(base, "meta")
    tasks_path = os.path.join(meta, "tasks.parquet")
    if not os.path.isfile(tasks_path):
        sys.exit(f"not a LeRobot dataset (no meta/tasks.parquet): {base}")

    # ---- read + validate tasks.parquet --------------------------------------
    ttab = pq.read_table(tasks_path)
    str_cols = [f.name for f in ttab.schema
                if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    if len(str_cols) != 1:
        sys.exit(f"tasks.parquet: expected exactly 1 string column, got {str_cols}")
    task_col = str_cols[0]
    cur = ttab.column(task_col).to_pylist()
    if len(cur) != 1:
        sys.exit(f"expected exactly 1 task, found {len(cur)}: {cur}")
    old = cur[0]

    print(f"dataset             : {base}")
    print(f"current task string : {old!r}")
    print(f"new task string     : {new!r}")
    if old == new:
        print("already set; nothing to do.")
        return

    # ---- read + validate every episode-meta file ----------------------------
    ep_files = sorted(glob.glob(os.path.join(meta, "episodes", "**", "*.parquet"),
                                recursive=True))
    ep_tabs = []
    for ep in ep_files:
        tab = pq.read_table(ep)
        if "tasks" not in tab.schema.names:
            sys.exit(f"{ep}: no 'tasks' column")
        bad = [v for v in tab.column("tasks").to_pylist() if v != [old]]
        if bad:
            sys.exit(f"{ep}: unexpected task value(s) {bad[:3]} -- aborting (multi-task?)")
        ep_tabs.append((ep, tab))
    print(f"episode-meta files  : {len(ep_files)}")

    # ---- backup meta/ -------------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(base, f"meta_backup_{stamp}")
    shutil.copytree(meta, bak)
    print(f"backed up meta/     -> {bak}")

    # ---- rewrite tasks.parquet ---------------------------------------------
    i = ttab.schema.get_field_index(task_col)
    field = ttab.schema.field(task_col)
    ttab2 = ttab.set_column(i, field, pa.array([new] * ttab.num_rows, type=field.type))
    pq.write_table(ttab2, tasks_path)
    print(f"updated             : {tasks_path}")

    # ---- rewrite episode-meta 'tasks' columns -------------------------------
    for ep, tab in ep_tabs:
        i = tab.schema.get_field_index("tasks")
        field = tab.schema.field("tasks")
        new_col = pa.array([[new]] * tab.num_rows, type=field.type)
        pq.write_table(tab.set_column(i, field, new_col), ep)
        print(f"updated             : {ep}  ({tab.num_rows} episodes)")

    # ---- verify -------------------------------------------------------------
    print("--- verify ---")
    ok = True
    t2 = pd.read_parquet(tasks_path)
    print(f"tasks.parquet -> index={list(t2.index)}  columns={list(t2.columns)}")
    if list(t2.index) != [new]:
        ok = False
        print("  MISMATCH in tasks.parquet")
    for ep in ep_files:
        vals = {str(x[0]) for x in pd.read_parquet(ep, columns=["tasks"])["tasks"]}
        if vals != {new}:
            ok = False
            print(f"  MISMATCH in {ep}: {vals}")
    if ok:
        print("OK -- task string updated everywhere.")
    else:
        print(f"VERIFICATION FAILED -- restore meta/ from: {bak}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
