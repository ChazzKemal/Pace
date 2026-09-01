#!/usr/bin/env python3
"""Crop stalls AND convert rotation Euler XYZ -> rotvec (axis-angle) in one pass.

Derivative of ``crop_stalls.py``. Same stall-removal and dataset-rebuild logic,
plus an inline conversion of the rotation channels of ``action`` (cols 3:6) and
``observation.state.cartesian`` (cols 3:6) from Euler XYZ to rotvec.

Why a separate script: the upstream ``crop_stalls.py`` is correct for axis-angle
sources and other callers may depend on it. This variant exists for fine-tuning
an axis-angle-trained ACT model on a freshly recorded Euler-XYZ dataset.

Per-episode continuity: after canonical rotvec conversion via scipy, the script
applies the same antipodal-flip rule as ``ManipulatorEnv._flip_rotation_vector_if_needed``
(first frame: flip if rotvec[0] < 0; subsequent frames: flip if dot product
with the previous rotvec is negative). The continuity sweep operates on the
kept-frame sequence -- the same temporal axis the policy will see.

Usage
-----
    python3 crop_stalls_euler_to_rotvec.py \
        --src data/datasets/real/speedup_20260525_233612_trimmed \
        --dst data/datasets/real/speedup_20260525_233612_trimmed_nostall \
        --keep-features observation.images.camera,observation.images.d405,observation.state.cartesian,action
"""

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

MOTION_EPS = 1e-6

# Columns whose [3:6] slice gets Euler XYZ -> rotvec conversion.
ROTATION_COLUMNS = ("action", "observation.state.cartesian")


def held_mask(actions: np.ndarray, motion_eps: float) -> np.ndarray:
    n = len(actions)
    mask = np.zeros(n, dtype=bool)
    if n >= 2:
        step = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
        mask[1:] = step <= float(motion_eps)
    return mask


def euler_xyz_to_rotvec_with_flip(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert (N, 3) Euler XYZ to (N, 3) rotvec, with antipodal-flip continuity."""
    rv = Rotation.from_euler("xyz", euler_xyz).as_rotvec()
    if len(rv) == 0:
        return rv
    if rv[0, 0] < 0:
        rv[0] = -rv[0]
    for i in range(1, len(rv)):
        if np.dot(rv[i - 1], rv[i]) < 0:
            rv[i] = -rv[i]
    return rv


class SeqVideoReader:
    def __init__(self, path: Path):
        self.path = path
        self._container = av.open(str(path))
        self._gen = self._container.decode(video=0)
        self._pos = 0

    def take(self, start: int, n: int, keep_set: set[int]) -> dict[int, np.ndarray]:
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
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--motion-eps", type=float, default=MOTION_EPS)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--keep-features", default=None,
                    help="comma-separated feature keys to keep (default: keep all "
                         "non-bookkeeping features)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"not a LeRobot dataset: {src}")
    if dst.exists():
        sys.exit(f"destination already exists, refusing to overwrite: {dst}")

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(info["fps"])

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

    if args.keep_features:
        wanted = {k.strip() for k in args.keep_features.split(",") if k.strip()}
        missing = wanted - set(features)
        if missing:
            sys.exit(
                f"--keep-features: unknown feature(s) {sorted(missing)}; "
                f"available: {sorted(features)}"
            )
        features = {k: v for k, v in features.items() if k in wanted}

    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    scalar_keys = [k for k in features if k not in video_keys]

    # Sanity check: rotation conversion only makes sense if the column is in the
    # kept feature set and its shape has at least 6 dims with the rotation in
    # cols 3:6. We expect (6,) for state, (7,) for action.
    convertible = [c for c in ROTATION_COLUMNS if c in scalar_keys]
    if not convertible:
        sys.exit(
            "no rotation columns in kept features -- nothing to convert. "
            f"Expected at least one of {ROTATION_COLUMNS}."
        )
    for c in convertible:
        s = features[c]["shape"]
        if s[0] < 6:
            sys.exit(f"rotation column {c!r} has shape {s}; need at least 6 dims")

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
    out.meta.update_chunk_settings(
        data_files_size_in_mb=1e-3, video_files_size_in_mb=1e-3,
    )

    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]

    video_file_base: dict = {}
    for vkey in video_keys:
        for e in ep_meta.index:
            r = ep_meta.loc[e]
            fkey = (vkey,
                    int(r[f"videos/{vkey}/chunk_index"]),
                    int(r[f"videos/{vkey}/file_index"]))
            gfrom = int(r["dataset_from_index"])
            video_file_base[fkey] = min(video_file_base.get(fkey, gfrom), gfrom)

    readers: dict[str, SeqVideoReader] = {}
    reader_paths: dict[str, Path] = {}
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
        df = (full[full["episode_index"] == ep_idx]
              .sort_values("frame_index")
              .reset_index(drop=True))
        if len(df) != gto - gfrom:
            print(
                f"WARN ep {ep_idx}: parquet has {len(df)} rows but episode meta "
                f"says {gto - gfrom}. Trusting parquet+video (one-file-per-episode "
                f"layout: base==gfrom, so video reads still start at frame 0).",
                flush=True,
            )

        actions = np.stack(
            [np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()],
            axis=0,
        )
        held = held_mask(actions, args.motion_eps)
        keep_idx = np.where(~held)[0]
        keep_set = {int(k) for k in keep_idx}

        # Precompute converted rotvec sequences over kept frames for every
        # rotation column. Per-episode antipodal continuity is enforced on the
        # kept-frame timeline (the one the policy sees).
        converted: dict[str, np.ndarray] = {}
        for col in convertible:
            full_col = np.stack(
                [np.asarray(v, dtype=np.float64) for v in df[col].to_numpy()],
                axis=0,
            )
            kept_col = full_col[keep_idx]  # (K, D)
            kept_col[:, 3:6] = euler_xyz_to_rotvec_with_flip(kept_col[:, 3:6])
            converted[col] = kept_col

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
        for local_i, k in enumerate(keep_idx):
            frame: dict = {"task": task}
            for vkey in video_keys:
                frame[vkey] = vid_kept[vkey][int(k)]
            for col in scalar_keys:
                ft = features[col]
                if col in converted:
                    arr = converted[col][local_i].astype(np.dtype(ft["dtype"]))
                else:
                    val = df[col].iloc[k]
                    arr = np.asarray(val, dtype=np.dtype(ft["dtype"]))
                    if arr.ndim == 0:
                        arr = arr.reshape(1)
                frame[col] = arr
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

    out_info_path = dst / "meta" / "info.json"
    out_info = json.loads(out_info_path.read_text())
    out_info["data_files_size_in_mb"] = info.get("data_files_size_in_mb", 100)
    out_info["video_files_size_in_mb"] = info.get("video_files_size_in_mb", 500)
    out_info_path.write_text(json.dumps(out_info, indent=4))

    print(
        f"\nDone. {n_episodes} episodes, {grand_total} -> {grand_kept} frames "
        f"({grand_total - grand_kept} held frames removed, "
        f"{100 * (grand_total - grand_kept) / max(grand_total, 1):.1f}%). "
        f"Rotation columns converted Euler XYZ -> rotvec: {convertible}."
    )
    print(f"Cropped dataset written to: {dst}")


if __name__ == "__main__":
    main()
