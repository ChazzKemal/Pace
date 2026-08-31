#!/usr/bin/env python3
"""Merge the two de-stalled gripper datasets into one ACT/Diffusion training set.

Takes the ``*_nostall`` datasets produced by ``crop_stalls.py`` and rebuilds a
single LeRobot v3 dataset purpose-built for policy training:

  input  (observation):  two camera streams + ``observation.state``
  output (action):       ``action``

The recorded ``observation.state`` is 13-dim (6 joint angles + 6 cartesian pose
+ 1 gripper). The policies here should consume only **current pose + gripper**,
so this script rebuilds ``observation.state`` as the 7-dim vector

    [x, y, z, roll, pitch, yaw, gripper]

i.e. ``concat(observation.state.cartesian, observation.state.gripper)``. The
joint angles and the per-stream timestamp features are dropped, and dropping
them is **load-bearing, not tidiness**: LeRobot classifies every key beginning
``observation.`` as a policy input (``utils/feature_utils.py:170``) and builds
the policy's input projections from that set (``policies/factory.py:333-335``).
Nothing is ignored. A column left in here is a column the network reads.

(This docstring used to claim the opposite -- "LeRobot ignores non-standard
``observation.*`` keys during training anyway" -- and that belief is what let
``stackcups_20260829_merged`` be built with three absolute wall-clock columns
still attached, which then trained an ACT baseline for 39k steps on seven inputs
instead of three. See ``docs/PLAN.md``.)

The rule that follows: a merge **allowlists**. It names the features it wants and
never looks at the rest, so a column added to the recorder next year cannot leak
into a training set. ``pace_bench.data.specs`` holds those names, and
``tests/test_dataset_specs.py`` checks each dataset against them.

``action`` is copied through unchanged: it is already [x, y, z, roll, pitch,
yaw, gripper] -- the desired "target pose + target gripper".

The merge is done through the LeRobot API (create / add_frame / save_episode),
so parquet indices, episode metadata and per-feature statistics are consistent
by construction. Episodes are concatenated in source order.

Usage
-----
    python3 merge_datasets.py                       # uses the defaults below
    python3 merge_datasets.py --srcs A B --dst OUT
"""

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_SRCS = [
    "data/datasets/real/speedup_20260515_230303_gripper_nostall",
    "data/datasets/real/speedup_20260516_232116_nostall",
]
DEFAULT_DST = "data/datasets/real/pickplace_merged_nostall"

# The 7-dim training state: current cartesian pose + gripper.
STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
# Slice into a 13-dim `observation.state` (6 joints + 6 cartesian + 1 gripper)
# that yields [x, y, z, roll, pitch, yaw, gripper].
STATE_13_KEEP = slice(6, 13)


def decode_video_frames(path: Path) -> list[np.ndarray]:
    """Decode an mp4 to a list of HWC uint8 RGB frames."""
    container = av.open(str(path))
    frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
    container.close()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--srcs", nargs="+", default=DEFAULT_SRCS,
                    help="source LeRobot v3 datasets to merge, in order")
    ap.add_argument("--dst", default=DEFAULT_DST,
                    help="destination dir (must not exist)")
    args = ap.parse_args()

    srcs = [Path(s).resolve() for s in args.srcs]
    dst = Path(args.dst).resolve()
    for src in srcs:
        if not (src / "meta" / "info.json").exists():
            sys.exit(f"not a LeRobot dataset: {src}")
    if dst.exists():
        sys.exit(f"destination already exists, refusing to overwrite: {dst}")

    # Validate the sources share the schema we depend on, and grab fps. Each
    # source must yield a 7-D [x,y,z,roll,pitch,yaw,gripper] state via ONE of:
    #   - "observation.state.cartesian" + "observation.state.gripper" subkeys
    #     (raw recording format),
    #   - flat 13-D "observation.state" (we slice indices 6:13),
    #   - flat 7-D "observation.state" (we pass it through).
    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in srcs]
    state_formats: list[str] = []  # one of "subkeys", "flat13", "flat7"
    fps = int(infos[0]["fps"])
    for src, info in zip(srcs, infos):
        if int(info["fps"]) != fps:
            sys.exit(f"fps mismatch: {src} has {info['fps']}, expected {fps}")
        if "action" not in info["features"]:
            sys.exit(f"{src} is missing required feature 'action'")
        feats = info["features"]
        if "observation.state.cartesian" in feats and "observation.state.gripper" in feats:
            state_formats.append("subkeys")
        else:
            state = feats.get("observation.state")
            if state is None:
                sys.exit(
                    f"{src} has neither (observation.state.cartesian + .gripper) "
                    f"nor a flat observation.state — cannot build 7-D state."
                )
            shape = tuple(state.get("shape", ()))
            if shape == (13,):
                state_formats.append("flat13")
            elif shape == (7,):
                state_formats.append("flat7")
            else:
                sys.exit(
                    f"{src}: unsupported observation.state shape {shape}; "
                    f"need (13,) (joints+cart+gripper) or (7,) (cart+gripper)."
                )
    for src, fmt in zip(srcs, state_formats):
        print(f"  {src.name}: state_format={fmt}")

    info0 = infos[0]
    video_keys = [k for k, v in info0["features"].items() if v["dtype"] == "video"]

    # Training-only feature set: cameras + rebuilt 7-dim state + action.
    features: dict = {}
    for vkey in video_keys:
        ft = info0["features"][vkey]
        features[vkey] = {
            "dtype": "video",
            "shape": tuple(ft["shape"]),
            "names": ft.get("names", ["height", "width", "channels"]),
        }
    features["observation.state"] = {
        "dtype": "float32", "shape": (7,), "names": STATE_NAMES,
    }
    act = info0["features"]["action"]
    features["action"] = {
        "dtype": "float32", "shape": tuple(act["shape"]), "names": act["names"],
    }

    out = LeRobotDataset.create(
        repo_id=dst.name,
        fps=fps,
        features=features,
        root=dst,
        robot_type=info0.get("robot_type"),
        use_videos=bool(video_keys),
    )

    def extract_state_7d(df_row, fmt: str) -> np.ndarray:
        """Return the 7-D [x,y,z,roll,pitch,yaw,gripper] state for one frame."""
        if fmt == "subkeys":
            cartesian = np.asarray(df_row["observation.state.cartesian"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(df_row["observation.state.gripper"], dtype=np.float32).reshape(-1)
            return np.concatenate([cartesian, gripper])
        if fmt == "flat13":
            arr = np.asarray(df_row["observation.state"], dtype=np.float32).reshape(-1)
            if arr.shape[0] != 13:
                raise ValueError(f"expected 13-D state, got {arr.shape}")
            return np.ascontiguousarray(arr[STATE_13_KEEP])
        if fmt == "flat7":
            arr = np.asarray(df_row["observation.state"], dtype=np.float32).reshape(-1)
            if arr.shape[0] != 7:
                raise ValueError(f"expected 7-D state, got {arr.shape}")
            return arr
        raise ValueError(f"unknown state format {fmt!r}")

    grand_frames = grand_eps = 0
    for src, info, fmt in zip(srcs, infos, state_formats):
        data_tmpl = info["data_path"]
        video_tmpl = info["video_path"]
        ep_meta = pd.concat(
            [pd.read_parquet(p) for p in
             sorted((src / "meta" / "episodes").rglob("file-*.parquet"))],
            ignore_index=True,
        ).set_index("episode_index").sort_index()

        for ep_idx in range(len(ep_meta)):
            row = ep_meta.loc[ep_idx]
            data_path = src / data_tmpl.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            df = pd.read_parquet(data_path)

            vid_frames: dict[str, list[np.ndarray]] = {}
            for vkey in video_keys:
                vpath = src / video_tmpl.format(
                    video_key=vkey,
                    chunk_index=int(row[f"videos/{vkey}/chunk_index"]),
                    file_index=int(row[f"videos/{vkey}/file_index"]),
                )
                decoded = decode_video_frames(vpath)
                if len(decoded) != len(df):
                    sys.exit(
                        f"{src.name} ep {ep_idx}: video {vkey} has "
                        f"{len(decoded)} frames but parquet has {len(df)} rows"
                    )
                vid_frames[vkey] = decoded

            task = row["tasks"][0] if len(row["tasks"]) else ""
            for k in range(len(df)):
                frame: dict = {"task": task}
                vfidx = int(df["frame_index"].iloc[k])
                for vkey in video_keys:
                    frame[vkey] = vid_frames[vkey][vfidx]
                frame["observation.state"] = extract_state_7d(df.iloc[k], fmt)
                frame["action"] = np.asarray(
                    df["action"].iloc[k], dtype=np.float32
                ).reshape(-1)
                out.add_frame(frame)

            out.save_episode()
            grand_frames += len(df)
            grand_eps += 1
            print(f"{src.name} ep {ep_idx:3d}: {len(df):4d} frames", flush=True)

    print(
        f"\nDone. Merged {len(srcs)} datasets -> {grand_eps} episodes, "
        f"{grand_frames} frames."
    )
    print(f"Merged dataset written to: {dst}")


if __name__ == "__main__":
    main()
