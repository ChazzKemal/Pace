#!/usr/bin/env python3
"""In-place rename of ``observation.state.cartesian`` -> ``observation.state``
in a LeRobot v3 dataset.

LeRobot's ``--rename_map`` CLI flag renames keys at sample-load time but does
NOT propagate to ``dataset.features``, so the policy factory builds its
input_features from the unrenamed key and the pretrained ACT checkpoint cannot
load (VAE encoder pos_enc shape mismatch). Renaming inside the dataset is the
robust workaround.

Touches:
* ``meta/info.json`` -- key in the ``features`` dict.
* ``meta/stats.json`` -- top-level key.
* ``meta/episodes/chunk-XXX/file-XXX.parquet`` -- columns whose name starts
  with ``stats/observation.state.cartesian/``.
* ``data/chunk-XXX/file-XXX.parquet`` -- the column itself.

Videos are untouched (their on-disk path uses ``video_key``, which is only for
``observation.images.*`` features, not the state column).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

OLD = "observation.state.cartesian"
NEW = "observation.state"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset root dir")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"not a LeRobot dataset: {root}")

    # info.json
    info = json.loads(info_path.read_text())
    if OLD not in info["features"]:
        sys.exit(f"feature {OLD!r} not present in info.json (has: {list(info['features'])})")
    if NEW in info["features"]:
        sys.exit(f"feature {NEW!r} already present in info.json -- refusing to clobber")
    info["features"] = {(NEW if k == OLD else k): v for k, v in info["features"].items()}
    info_path.write_text(json.dumps(info, indent=4))
    print(f"info.json: {OLD!r} -> {NEW!r}")

    # stats.json
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    if OLD in stats:
        stats[NEW] = stats.pop(OLD)
        stats_path.write_text(json.dumps(stats, indent=4))
        print(f"stats.json: {OLD!r} -> {NEW!r}")

    # meta/episodes/*.parquet (per-episode stats live in columns named
    # 'stats/<feature_key>/<min|max|mean|std|count>')
    ep_dir = root / "meta" / "episodes"
    for p in sorted(ep_dir.rglob("file-*.parquet")):
        df = pd.read_parquet(p)
        old_prefix = f"stats/{OLD}/"
        new_prefix = f"stats/{NEW}/"
        renames = {c: new_prefix + c[len(old_prefix):]
                   for c in df.columns if c.startswith(old_prefix)}
        if renames:
            df = df.rename(columns=renames)
            df.to_parquet(p)
            print(f"episodes {p.name}: renamed {len(renames)} stats columns")

    # data/*.parquet
    data_dir = root / "data"
    for p in sorted(data_dir.rglob("file-*.parquet")):
        df = pd.read_parquet(p)
        if OLD in df.columns:
            df = df.rename(columns={OLD: NEW})
            df.to_parquet(p)
            print(f"data {p.name}: renamed column")

    print(f"\nDone. {root.name} now exposes proprio as {NEW!r}.")


if __name__ == "__main__":
    main()
