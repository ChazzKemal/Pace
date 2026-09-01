#!/usr/bin/env python3
"""Write a copy of a LeRobot v3 dataset with named observation columns removed.

LeRobot classifies **every** ``observation.*`` key as a policy input
(``lerobot/utils/feature_utils.py:170``: ``elif key.startswith(OBS_STR)``), and
``DatasetConfig`` has no exclusion flag -- ``rename_map`` only renames, and it
refuses to run without a pretrained checkpoint. So a recording that carries
bookkeeping columns beside the robot state trains a policy that consumes them,
and the only place to fix that is the dataset.

The case this was written for: ``stackcups_20260829_merged`` carries

  * ``observation.timestamps.wall``           absolute epoch seconds
  * ``observation.timestamps.camera_header``  ditto, camera stamp
  * ``observation.timestamps.d405_header``    ditto, wrist camera stamp
  * ``observation.state.cartesian``           byte-identical to observation.state

The three timestamps are the harmful ones. They are monotonic across the whole
recording session, so during training they are an episode index and a within-
episode clock -- a shortcut a policy is free to take instead of looking at the
images. And they are *absolute*: the dataset's std is 6.5e3 seconds, so a
deployment one day after recording feeds the state encoder a value 13 sigma
outside anything it saw, on a channel it may well have learned to trust.
``observation.state.cartesian`` is merely redundant, but dropping it makes the
input set identical to the pickplace datasets', which is what lets the two tasks'
arms be compared.

Videos are hardlinked, not copied: they are unchanged and are 98% of the bytes.
Falls back to a copy across filesystems.

Usage:
    python drop_observation_features.py SRC DST \\
        --drop observation.timestamps.wall observation.state.cartesian ...
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq


def link_tree(src: Path, dst: Path) -> None:
    """Hardlink every file under `src` into `dst`, copying if that is refused."""
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def write_json(path: Path, payload: dict) -> None:
    """Replace `path`, breaking any hardlink first.

    Every file here arrives as a hardlink to the source dataset, and writing to a
    hardlink writes to *both* names. Doing that to info.json and stats.json
    silently stripped the four features from the source as well -- the copy looked
    right and the original was quietly gutted. Unlink, then create.
    """
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(payload, indent=4))


def rewrite_parquet(dst: Path, src_root: Path, dst_root: Path, drop: set[str]) -> list[str]:
    """Break the hardlink at `dst`, restore the source bytes, then drop columns."""
    dst.unlink()
    shutil.copy2(src_root / dst.relative_to(dst_root), dst)
    return drop_columns(dst, drop)


def drop_columns(path: Path, drop: set[str]) -> list[str]:
    """Rewrite one parquet file without the named columns. Returns what it removed."""
    table = pq.read_table(path)
    present = [c for c in table.column_names if c in drop]
    if not present:
        return []
    pq.write_table(table.drop_columns(present), path)
    return present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="source dataset root")
    ap.add_argument("dst", type=Path, help="destination dataset root (must not exist)")
    ap.add_argument("--drop", nargs="+", required=True, metavar="KEY", help="feature keys to remove")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    drop = set(args.drop)

    if not (src / "meta" / "info.json").exists():
        sys.exit(f"not a LeRobot dataset: {src}")
    if dst.exists():
        sys.exit(f"{dst} already exists -- refusing to overwrite")

    # Fail before writing anything if a name is wrong: a typo would otherwise
    # produce a dataset that looks clean and still carries the column.
    info = json.loads((src / "meta" / "info.json").read_text())
    missing = drop - set(info["features"])
    if missing:
        sys.exit(f"not features of this dataset: {sorted(missing)}\nhas: {sorted(info['features'])}")
    kept_obs = [k for k in info["features"] if k.startswith("observation.") and k not in drop]
    print(f"dropping {sorted(drop)}\nkeeping  {kept_obs}")

    link_tree(src, dst)

    # Every write below goes through write_json / rewrite_parquet, both of which
    # unlink before creating. Nothing in this function may write to a path it did
    # not first unlink -- see write_json's docstring for what that cost once.

    written: list[Path] = []

    # info.json -- the feature list is what the policy factory reads.
    info["features"] = {k: v for k, v in info["features"].items() if k not in drop}
    write_json(dst / "meta" / "info.json", info)
    written.append(dst / "meta" / "info.json")

    # stats.json -- one top-level entry per feature.
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        removed = [k for k in drop if k in stats]
        for key in removed:
            stats.pop(key)
        write_json(stats_path, stats)
        written.append(stats_path)
        print(f"stats.json: removed {sorted(removed)}")

    # data parquets -- the columns themselves.
    for path in sorted((dst / "data").rglob("*.parquet")):
        print(f"{path.relative_to(dst)}: removed {rewrite_parquet(path, src, dst, drop)}")
        written.append(path)

    # per-episode meta parquets -- per-feature stats live in `stats/<key>/<field>`.
    prefixes = tuple(f"stats/{k}/" for k in drop)
    for path in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        cols = {c for c in pq.read_schema(path).names if c.startswith(prefixes)}
        n = len(rewrite_parquet(path, src, dst, cols))
        written.append(path)
        print(f"{path.relative_to(dst)}: removed {n} stat columns")

    # The point of the unlinking above: prove nothing this script wrote is still
    # an alias for a source file. Only the written set is checked -- videos and
    # tasks.parquet are *meant* to stay hardlinked, that being the saving.
    shared = [p.relative_to(dst) for p in written if p.stat().st_nlink > 1]
    if shared:
        raise SystemExit(
            f"BUG: these still share an inode with {src} and have corrupted it: {shared}"
        )
    print(f"link check: all {len(written)} rewritten files are unshared")

    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
