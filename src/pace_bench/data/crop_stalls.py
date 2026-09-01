#!/usr/bin/env python3
"""Create a stall-free copy of a LeRobot v3 dataset.

A frame is "held" (stalling) when the commanded end-effector position barely
moves from the previous frame:

    ||action_pos[i] - action_pos[i-1]|| <= motion_eps

This is the exact definition ``compute_speed_schedule_drop_holds()`` in
``17_replay_dataset.py`` uses (position channels of ``action``, motion_eps
default 1e-6). Frame 0 of every episode is always kept as an anchor.

Unlike that helper -- which only *skips* held frames while computing a replay
speed schedule -- this script physically *removes* them: held parquet rows are
dropped and every camera video stream is re-encoded without the held frames.

The cropped dataset is rebuilt through the LeRobot API, so parquet indices,
episode metadata, per-feature statistics and the re-encoded videos are all
consistent by construction. Replay timestamps are re-derived contiguously at
the original fps (the stalls simply vanish from the timeline).

Both LeRobot v3 storage layouts are handled transparently:

* one-file-per-episode  -- each episode has its own data parquet + mp4;
* packed (multi-episode) -- many episodes share a single data parquet and a
  single mp4 per camera. Episode boundaries are taken from the
  ``dataset_from_index`` / ``dataset_to_index`` columns of ``meta/episodes``
  and the per-row ``episode_index`` column of the data parquet.

Each episode's rows are sliced out by ``episode_index``; each video file is
decoded with a single forward pass (episodes are stored contiguously and in
order), so a packed dataset is never fully loaded into RAM.

Usage
-----
    python3 crop_stalls.py \
        --src data/datasets/real/infinite_task_trimmed \
        --dst data/datasets/real/infinite_task_trimmed_nostall

    # smoke-test on the first 2 episodes:
    python3 crop_stalls.py --src ... --dst ... --limit-episodes 2
"""

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

# Default threshold: matches 17_replay_dataset.py compute_speed_schedule_drop_holds.
MOTION_EPS = 1e-6


def held_mask(actions: np.ndarray, motion_eps: float) -> np.ndarray:
    """Boolean mask: True where a frame is a zero-motion hold (frame 0 never)."""
    n = len(actions)
    mask = np.zeros(n, dtype=bool)
    if n >= 2:
        step = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
        mask[1:] = step <= float(motion_eps)
    return mask


class SeqVideoReader:
    """Forward-only sequential reader over a single mp4 file.

    ``take(start, n, keep_set)`` advances to absolute frame index ``start``,
    decodes the next ``n`` frames, and returns ``{local_index: rgb_frame}`` for
    the local indices (0..n-1) present in ``keep_set``. Successive calls must
    request non-decreasing ``start`` values -- which holds because episodes are
    stored contiguously and in order within a video file -- so every frame in
    the file is decoded at most once and only kept frames are retained in RAM.
    """

    def __init__(self, path: Path):
        self.path = path
        self._container = av.open(str(path))
        self._gen = self._container.decode(video=0)
        self._pos = 0  # absolute index of the next frame the generator yields

    def take(self, start: int, n: int, keep_set: set[int]) -> dict[int, np.ndarray]:
        """Return kept RGB frames of the slice [start, start+n) of this file."""
        if start < self._pos:
            raise RuntimeError(
                f"non-monotonic read on {self.path}: asked for frame {start} "
                f"but the reader is already at {self._pos}"
            )
        while self._pos < start:
            next(self._gen)
            self._pos += 1
        kept: dict[int, np.ndarray] = {}
        for local in range(n):
            try:
                fr = next(self._gen)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{self.path}: ran out of frames at absolute index "
                    f"{self._pos} (needed up to {start + n})"
                ) from exc
            if local in keep_set:
                kept[local] = fr.to_ndarray(format="rgb24")
            self._pos += 1
        return kept

    def close(self) -> None:
        self._container.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="source LeRobot v3 dataset dir")
    ap.add_argument("--dst", required=True, help="destination dir (must not exist)")
    ap.add_argument("--motion-eps", type=float, default=MOTION_EPS,
                    help=f"hold threshold on action xyz step (default {MOTION_EPS})")
    ap.add_argument("--limit-episodes", type=int, default=None,
                    help="only process the first N episodes (smoke test)")
    ap.add_argument("--keep-features", default=None,
                    help="comma-separated feature keys to keep in the cropped "
                         "dataset (default: keep every non-bookkeeping feature). "
                         "Use this to drop features a policy must not ingest -- "
                         "e.g. LeRobot classifies every 'observation.*' key as a "
                         "STATE input, so leftover 'observation.timestamps.*' or "
                         "redundant sub-states would otherwise be fed to the model.")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"not a LeRobot dataset: {src}")
    if dst.exists():
        sys.exit(f"destination already exists, refusing to overwrite: {dst}")

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(info["fps"])

    # Features for the new dataset: everything except the bookkeeping columns
    # LeRobot manages itself (timestamp/frame_index/episode_index/index/
    # task_index). Shapes -> tuples so validate_frame's shape compare passes.
    features: dict = {}
    for key, ft in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        if ft["dtype"] == "video":
            features[key] = {
                "dtype": "video",
                "shape": tuple(ft["shape"]),
                "names": ft.get("names", ["height", "width", "channels"]),
            }
        else:
            features[key] = {
                "dtype": ft["dtype"],
                "shape": tuple(ft["shape"]),
                "names": ft.get("names"),
            }

    # Optionally restrict the cropped dataset to a subset of features.
    if args.keep_features:
        wanted = {k.strip() for k in args.keep_features.split(",") if k.strip()}
        missing = wanted - set(features)
        if missing:
            sys.exit(
                f"--keep-features: unknown feature(s) {sorted(missing)}; "
                f"available: {sorted(features)}"
            )
        features = {k: v for k, v in features.items() if k in wanted}

    # Re-derive video keys from the (possibly filtered) feature set.
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    # Episode metadata, indexed by episode. Carries the per-episode row range
    # (dataset_from_index/dataset_to_index) and the data/video file indices.
    ep_meta = pd.concat(
        [pd.read_parquet(p) for p in
         sorted((src / "meta" / "episodes").rglob("file-*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index").sort_index()

    n_episodes = len(ep_meta)
    if args.limit_episodes is not None:
        n_episodes = min(n_episodes, args.limit_episodes)

    out = LeRobotDataset.create(
        repo_id=dst.name,
        fps=fps,
        features=features,
        root=dst,
        robot_type=info.get("robot_type"),
        use_videos=bool(video_keys),
    )
    # Match a one-parquet + one-mp4 per episode layout. LeRobot v3.0 otherwise
    # packs many episodes into a shared file up to the size limits. A
    # sub-kilobyte cap is below any single episode, so every save_episode()
    # flushes to a fresh file (file-000, file-001, ...). The cosmetic size
    # fields in info.json are restored to the source values once done.
    out.meta.update_chunk_settings(
        data_files_size_in_mb=1e-3, video_files_size_in_mb=1e-3,
    )

    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    # Non-video, non-default feature columns to copy straight through.
    scalar_keys = [k for k in features if k not in video_keys]

    # Base absolute frame index of each video file: the smallest
    # dataset_from_index among the episodes that share it. Episode E's frames
    # occupy [dataset_from_index - base, dataset_to_index - base) of that file.
    # (For a one-file-per-episode dataset the base is just the episode's own
    # dataset_from_index, so the slice starts at 0.)
    video_file_base: dict = {}
    for vkey in video_keys:
        for e in ep_meta.index:
            r = ep_meta.loc[e]
            fkey = (vkey,
                    int(r[f"videos/{vkey}/chunk_index"]),
                    int(r[f"videos/{vkey}/file_index"]))
            gfrom = int(r["dataset_from_index"])
            video_file_base[fkey] = min(video_file_base.get(fkey, gfrom), gfrom)

    # One open video reader per camera key; recreated when the file changes.
    readers: dict[str, SeqVideoReader] = {}
    reader_paths: dict[str, Path] = {}
    # Loaded data parquet files (small: states/actions/timestamps, no images).
    data_cache: dict[Path, pd.DataFrame] = {}

    grand_total = grand_kept = 0
    for ep_idx in range(n_episodes):
        row = ep_meta.loc[ep_idx]
        gfrom = int(row["dataset_from_index"])
        gto = int(row["dataset_to_index"])

        data_path = src / data_tmpl.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        if data_path not in data_cache:
            data_cache[data_path] = pd.read_parquet(data_path)
        full = data_cache[data_path]
        # Slice this episode's rows out of a (possibly multi-episode) file.
        df = (full[full["episode_index"] == ep_idx]
              .sort_values("frame_index")
              .reset_index(drop=True))
        if len(df) != gto - gfrom:
            sys.exit(
                f"ep {ep_idx}: data parquet yields {len(df)} rows but episode "
                f"metadata says {gto - gfrom} -- episode_index mismatch"
            )

        actions = np.stack(
            [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
            axis=0,
        )
        held = held_mask(actions, args.motion_eps)
        keep_idx = np.where(~held)[0]
        keep_set = {int(k) for k in keep_idx}

        # Decode this episode's slice of every (possibly packed) video stream.
        vid_kept: dict[str, dict[int, np.ndarray]] = {}
        for vkey in video_keys:
            vchunk = int(row[f"videos/{vkey}/chunk_index"])
            vfile = int(row[f"videos/{vkey}/file_index"])
            vpath = src / video_tmpl.format(
                video_key=vkey, chunk_index=vchunk, file_index=vfile,
            )
            if reader_paths.get(vkey) != vpath:
                if vkey in readers:
                    readers[vkey].close()
                readers[vkey] = SeqVideoReader(vpath)
                reader_paths[vkey] = vpath
            base = video_file_base[(vkey, vchunk, vfile)]
            kept = readers[vkey].take(gfrom - base, len(df), keep_set)
            if len(kept) != len(keep_idx):
                sys.exit(
                    f"ep {ep_idx}: video {vkey} yielded {len(kept)} kept frames, "
                    f"expected {len(keep_idx)} -- frame/row alignment broken"
                )
            vid_kept[vkey] = kept

        task = row["tasks"][0] if len(row["tasks"]) else ""
        for k in keep_idx:
            frame: dict = {"task": task}
            for vkey in video_keys:
                frame[vkey] = vid_kept[vkey][int(k)]
            for col in scalar_keys:
                ft = features[col]
                val = df[col].iloc[k]
                arr = np.asarray(val, dtype=np.dtype(ft["dtype"]))
                if arr.ndim == 0:  # scalar column -> shape (1,) feature
                    arr = arr.reshape(1)
                frame[col] = arr
            # timestamp deliberately omitted: LeRobot re-derives it as
            # frame_index / fps, giving the cropped episode a contiguous
            # timeline with the stalls removed.
            out.add_frame(frame)

        out.save_episode()
        grand_total += len(df)
        grand_kept += len(keep_idx)
        print(
            f"ep {ep_idx:3d}: {len(df):5d} -> {len(keep_idx):5d} frames "
            f"({int(held.sum())} held dropped)",
            flush=True,
        )

    for reader in readers.values():
        reader.close()

    # Restore the cosmetic file-size fields to the source values. The actual
    # one-file-per-episode split is already baked into the episode metadata;
    # these fields only affect any *future* appends to the dataset.
    out_info_path = dst / "meta" / "info.json"
    out_info = json.loads(out_info_path.read_text())
    out_info["data_files_size_in_mb"] = info.get("data_files_size_in_mb", 100)
    out_info["video_files_size_in_mb"] = info.get("video_files_size_in_mb", 500)
    out_info_path.write_text(json.dumps(out_info, indent=4))

    print(
        f"\nDone. {n_episodes} episodes, {grand_total} -> {grand_kept} frames "
        f"({grand_total - grand_kept} held frames removed, "
        f"{100 * (grand_total - grand_kept) / max(grand_total, 1):.1f}%)."
    )
    print(f"Cropped dataset written to: {dst}")


if __name__ == "__main__":
    main()
