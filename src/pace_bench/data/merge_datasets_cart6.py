#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets into one with a 6-D cartesian proprio.

Variant of ``merge_datasets.py`` (which builds a 7-D state = cartesian+gripper)
specifically for fine-tuning the existing ACT ``cart7_v2_angleaxis_nogrip``
model. That checkpoint expects:

    input  : observation.images.{camera, d405}  +  observation.state (6,)
    output : action (7,)  = [x, y, z, rx, ry, rz, gripper]

i.e. gripper is OUTPUT-only, not part of the proprio state -- "no grip" in the
model name refers to the absence of gripper in ``observation.state``.

For source datasets whose ``observation.state`` is the 13-D
joints+cart+gripper bundle (e.g. ``speedup_20260527_223430``), we substitute
the 6-D ``observation.state.cartesian`` column under the ``observation.state``
name. Every other ``observation.*`` column (joints, gripper, timestamps,
duplicated cartesian) is dropped -- LeRobot would otherwise feed them all to
the policy as STATE inputs.

Rotation is assumed to already be in axis-angle (rotvec); no conversion is
performed. Stalls are assumed to already be trimmed if relevant.

Usage
-----
    python3 merge_datasets_cart6.py \\
        --src data/datasets/real/speedup_20260527_234316 \\
        --src data/datasets/real/speedup_20260527_223430 \\
        --src data/datasets/real/pickplace_cart7_v3 \\
        --src data/datasets/real/speedup_20260525_233612_trimmed_nostall \\
        --dst data/datasets/real/merged_act_finetune_20260528
"""

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CART_SUBSTITUTE_SRC = "observation.state.cartesian"


class SeqVideoReader:
    """Forward-only sequential reader over a single mp4 file."""

    def __init__(self, path: Path):
        self.path = path
        self._container = av.open(str(path))
        self._gen = self._container.decode(video=0)
        self._pos = 0

    def take(self, start: int, n: int) -> list[np.ndarray]:
        if start < self._pos:
            raise RuntimeError(
                f"non-monotonic read on {self.path}: asked for frame {start} "
                f"but the reader is already at {self._pos}"
            )
        while self._pos < start:
            next(self._gen)
            self._pos += 1
        out: list[np.ndarray] = []
        for _ in range(n):
            try:
                fr = next(self._gen)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{self.path}: ran out of frames at absolute index {self._pos}"
                ) from exc
            out.append(fr.to_ndarray(format="rgb24"))
            self._pos += 1
        return out

    def close(self) -> None:
        self._container.close()


def build_target_features(reference_info: dict) -> dict:
    src = reference_info["features"]
    feats: dict = {}
    for key in ("observation.images.camera", "observation.images.d405"):
        ft = src[key]
        feats[key] = {
            "dtype": "video",
            "shape": tuple(ft["shape"]),
            "names": ft.get("names", ["height", "width", "channels"]),
        }
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "rx", "ry", "rz"],
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
    }
    return feats


def iter_source_episodes(src: Path):
    """Yield (ep_idx, row, df, vid_frames_dict) for each episode in src.

    Handles both one-file-per-episode and packed multi-episode video files.
    """
    info = json.loads((src / "meta" / "info.json").read_text())
    ep_meta = pd.concat(
        [pd.read_parquet(p) for p in
         sorted((src / "meta" / "episodes").rglob("file-*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index").sort_index()

    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    data_cache: dict[Path, pd.DataFrame] = {}

    video_keys = ("observation.images.camera", "observation.images.d405")
    video_file_base: dict = {}
    for vkey in video_keys:
        for e in ep_meta.index:
            r = ep_meta.loc[e]
            fkey = (
                vkey,
                int(r[f"videos/{vkey}/chunk_index"]),
                int(r[f"videos/{vkey}/file_index"]),
            )
            gfrom = int(r["dataset_from_index"])
            video_file_base[fkey] = min(video_file_base.get(fkey, gfrom), gfrom)

    readers: dict[str, SeqVideoReader] = {}
    reader_paths: dict[str, Path] = {}

    try:
        for ep_idx in ep_meta.index:
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
            n_rows = len(df)
            if n_rows == 0:
                print(f"  WARN ep {ep_idx}: 0 rows in data parquet, skipping",
                      flush=True)
                continue
            if n_rows != gto - gfrom:
                print(
                    f"  WARN ep {ep_idx}: parquet has {n_rows} rows but meta "
                    f"says {gto - gfrom}. Trusting parquet+video.",
                    flush=True,
                )

            vid_frames: dict[str, list[np.ndarray]] = {}
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
                vid_frames[vkey] = readers[vkey].take(gfrom - base, n_rows)

            yield ep_idx, row, df, vid_frames
    finally:
        for r in readers.values():
            r.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", action="append", required=True,
                    help="path to a source LeRobot v3 dataset (repeat for each)")
    ap.add_argument("--dst", required=True, help="destination dir (must not exist)")
    args = ap.parse_args()

    srcs = [Path(s).resolve() for s in args.src]
    dst = Path(args.dst).resolve()
    if dst.exists():
        sys.exit(f"destination already exists, refusing to overwrite: {dst}")
    for s in srcs:
        if not (s / "meta" / "info.json").exists():
            sys.exit(f"not a LeRobot dataset: {s}")

    ref_info = json.loads((srcs[0] / "meta" / "info.json").read_text())
    fps = int(ref_info["fps"])
    features = build_target_features(ref_info)

    out = LeRobotDataset.create(
        repo_id=dst.name,
        fps=fps,
        features=features,
        root=dst,
        robot_type=ref_info.get("robot_type"),
        use_videos=True,
    )
    out.meta.update_chunk_settings(
        data_files_size_in_mb=1e-3, video_files_size_in_mb=1e-3,
    )

    grand_eps = grand_frames = 0
    for src in srcs:
        info = json.loads((src / "meta" / "info.json").read_text())
        if int(info["fps"]) != fps:
            sys.exit(f"fps mismatch: {src} has fps={info['fps']}, expected {fps}")

        src_state_shape = tuple(info["features"]["observation.state"]["shape"])
        if src_state_shape == (6,):
            proprio_col = "observation.state"
        elif CART_SUBSTITUTE_SRC in info["features"] and \
                tuple(info["features"][CART_SUBSTITUTE_SRC]["shape"]) == (6,):
            proprio_col = CART_SUBSTITUTE_SRC
            print(
                f"  {src.name}: observation.state has shape {src_state_shape}; "
                f"substituting from {CART_SUBSTITUTE_SRC}",
                flush=True,
            )
        else:
            sys.exit(
                f"{src.name}: cannot find 6-D cartesian proprio "
                f"(observation.state is {src_state_shape}, no usable "
                f"observation.state.cartesian)"
            )

        print(f"\n=== {src.name} ===", flush=True)
        src_eps = src_frames = 0
        for ep_idx, row, df, vid_frames in iter_source_episodes(src):
            task = (row["tasks"][0]
                    if "tasks" in row and len(row["tasks"]) else "")
            n_rows = len(df)
            proprio = np.stack(
                [np.asarray(v, dtype=np.float32) for v in df[proprio_col].to_numpy()],
                axis=0,
            )
            actions = np.stack(
                [np.asarray(v, dtype=np.float32) for v in df["action"].to_numpy()],
                axis=0,
            )
            for i in range(n_rows):
                frame = {
                    "task": task,
                    "observation.images.camera": vid_frames["observation.images.camera"][i],
                    "observation.images.d405": vid_frames["observation.images.d405"][i],
                    "observation.state": proprio[i],
                    "action": actions[i],
                }
                out.add_frame(frame)
            out.save_episode()
            src_eps += 1
            src_frames += n_rows
            print(f"  ep {ep_idx:3d}: {n_rows:5d} frames", flush=True)
        print(f"  -> {src_eps} eps, {src_frames} frames", flush=True)
        grand_eps += src_eps
        grand_frames += src_frames

    out_info_path = dst / "meta" / "info.json"
    out_info = json.loads(out_info_path.read_text())
    out_info["data_files_size_in_mb"] = ref_info.get("data_files_size_in_mb", 100)
    out_info["video_files_size_in_mb"] = ref_info.get("video_files_size_in_mb", 500)
    out_info_path.write_text(json.dumps(out_info, indent=4))

    print(
        f"\nDone. {grand_eps} episodes, {grand_frames} frames merged into "
        f"{dst.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
