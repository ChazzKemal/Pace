#!/usr/bin/env python3
"""Watch a LeRobot training run's `checkpoints/` dir and keep only the latest
checkpoint, deleting earlier ones to save disk.

LeRobot has no checkpoint-retention setting (only `save_checkpoint: bool`), and
each checkpoint (`pretrained_model/` + `training_state/`) is large. This watcher
prunes old ones automatically while training runs.

Race-free by construction: LeRobot writes `checkpoints/<step>/` fully and *then*
re-points the `checkpoints/last` symlink to it (see `save_checkpoint` /
`update_last_checkpoint` in lerobot/utils/train_utils.py). So we only ever delete
numbered checkpoints whose step is STRICTLY LESS than the step `last` points to.
A half-written newest checkpoint has a step >= the current `last` target, so it
is never touched until its save completes and `last` advances.

Usage:
    python prune_checkpoints.py RUN_DIR [--interval SECONDS] [--keep N] [--once]

    RUN_DIR     the --output-dir of the run (the dir that contains checkpoints/)
    --interval  poll period in seconds (default: 30)
    --keep      how many most-recent checkpoints to keep (default: 1)
    --once      prune a single time and exit (no watching)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

LAST_LINK = "last"  # lerobot.utils.constants.LAST_CHECKPOINT_LINK


def _numbered_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] for fully-named numeric checkpoint dirs, sorted."""
    out = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and not p.is_symlink() and p.name.isdigit():
            out.append((int(p.name), p))
    out.sort(key=lambda t: t[0])
    return out


def _last_target_step(ckpt_dir: Path) -> int | None:
    """Step that `checkpoints/last` resolves to, or None if no/dangling link."""
    link = ckpt_dir / LAST_LINK
    if not link.is_symlink():
        return None
    target = link.resolve()
    if not target.is_dir() or not target.name.isdigit():
        return None
    return int(target.name)


def prune_once(ckpt_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` checkpoints, never above the `last` target.

    Returns the list of deleted paths.
    """
    if not ckpt_dir.is_dir():
        return []
    ckpts = _numbered_checkpoints(ckpt_dir)
    if len(ckpts) <= keep:
        return []

    # Never delete a checkpoint at/above what `last` currently points to: that
    # protects an in-progress save whose symlink hasn't been flipped yet.
    last_step = _last_target_step(ckpt_dir)
    ceiling = last_step if last_step is not None else ckpts[-1][0]

    # Candidates to keep: the newest `keep` by step number.
    keep_steps = {step for step, _ in ckpts[-keep:]}

    deleted = []
    for step, path in ckpts:
        if step in keep_steps:
            continue
        if step >= ceiling:
            continue  # at/above the live `last` target — leave it alone
        shutil.rmtree(path)
        deleted.append(path)
    return deleted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Training --output-dir (contains checkpoints/).")
    ap.add_argument("--interval", type=float, default=30.0, help="Poll period seconds (default 30).")
    ap.add_argument("--keep", type=int, default=1, help="Most-recent checkpoints to keep (default 1).")
    ap.add_argument("--once", action="store_true", help="Prune once and exit.")
    args = ap.parse_args()

    ckpt_dir = args.run_dir / "checkpoints"
    print(f"[prune] watching {ckpt_dir}  keep={args.keep}  interval={args.interval}s", flush=True)

    while True:
        try:
            deleted = prune_once(ckpt_dir, args.keep)
            for p in deleted:
                print(f"[prune] deleted {p}", flush=True)
        except FileNotFoundError:
            pass  # dir churn between listdir and rmtree; retry next tick
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
