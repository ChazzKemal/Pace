#!/usr/bin/env python3
"""Build `checkpoints/` -- one symlink per (task, policy, arm) cell of the benchmark.

Checkpoints land wherever their training run wrote them, under names that encode the
run rather than the experiment (`outputs/train/ds_libero10_speedup/checkpoints/last/
pretrained_model`). That is fine for the pipeline scripts, which know what they just
produced, but it means the state of the benchmark -- which cells of {task} x {policy}
x {arm} actually have a trained policy behind them -- cannot be read off the disk.

So this writes the view the experiment has, and the pipelines keep the layout they
have. Each cell becomes a relative symlink to the `pretrained_model` directory every
consumer already takes as `--policy_path`, and the cells with nothing behind them are
reported rather than linked: a dangling symlink is not an honest way to say "not
trained yet", and it breaks anything that walks the tree.

MATRIX below is the declaration of what the benchmark is *supposed* to contain,
because absence cannot be inferred from disk -- a cell that was never scheduled and a
cell whose run crashed look identical there. It mirrors the three pipeline scripts,
and every existing run's own config is checked against the cell it is filed under, so
a drifting MATRIX shows up as a mismatch instead of a plausible wrong answer.

    python link_checkpoints.py            # rebuild the tree, print the matrix
    python link_checkpoints.py --check    # print the matrix, touch nothing
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "outputs" / "train"
DATA = Path(os.environ.get("PACE_DATA_ROOT", REPO.parent / "data"))
OUT = REPO / "checkpoints"


@dataclass(frozen=True)
class Cell:
    """One arm of one policy family on one task."""

    task: str
    policy: str
    arm: str  # the --method.type the run trained under
    run: str | None  # output dir under outputs/train, or None if nothing schedules it
    note: str = ""

    @property
    def method(self) -> str:
        return "none" if self.arm == "baseline" else self.arm


#: The benchmark's intended contents. `run=None` means no pipeline script produces
#: this cell -- a gap in the *plan*, not a failed run.
MATRIX = [
    # -- real, UR10e: run_demospeedup_stackcups.sh -----------------------------
    Cell("stack_cups", "act", "baseline", "cups_act_base"),
    Cell("stack_cups", "act", "demospeedup", "cups_act_speedup"),
    Cell("stack_cups", "diffusion", "baseline", "cups_diffusion_base"),
    Cell("stack_cups", "diffusion", "demospeedup", None, "no stage in run_demospeedup_stackcups.sh"),
    # -- real, UR10e: run_demospeedup_pickplace.sh -----------------------------
    Cell("pickplace", "act", "baseline", "pickplace_act_base"),
    Cell("pickplace", "act", "demospeedup", "pickplace_act_speedup"),
    Cell("pickplace", "diffusion", "baseline", "pickplace_diffusion_base"),
    Cell("pickplace", "diffusion", "demospeedup", "pickplace_diffusion_speedup"),
    # -- sim, LIBERO-10: run_demospeedup_libero10.sh ---------------------------
    Cell("libero_10", "xvla", "baseline", "ds_libero10_base"),
    Cell("libero_10", "xvla", "demospeedup", "ds_libero10_speedup"),
]

#: Weights that are an input to the benchmark rather than one of its cells.
PRETRAINED = {"xvla_libero_patched": DATA / "checkpoints" / "xvla_libero_patched"}

# PACE and B-spline are absent from MATRIX on purpose, and for different reasons.
# PACE acts at *evaluation* time, so it produces no checkpoint of its own -- it is
# run against a baseline arm's weights. B-spline changes the action space and so
# would train its own arm, but `methods/bspline` is a library today with no
# `--method.type` registered, so no run can be scheduled for it yet.


@dataclass
class State:
    """What is actually on disk for one cell."""

    status: str  # trained | partial | crashed | absent | unscheduled
    detail: str
    target: Path | None = None


def inspect(cell: Cell) -> State:
    if cell.run is None:
        return State("unscheduled", cell.note or "not scheduled")

    run_dir = TRAIN / cell.run
    if not run_dir.exists():
        return State("absent", "never started")

    last = run_dir / "checkpoints" / "last"
    if not last.exists():
        # The run dir exists but holds no checkpoint: it got as far as creating its
        # output directory (and usually a wandb run) and then died.
        return State("crashed", "started, no checkpoint")

    model = last / "pretrained_model"
    config_path = model / "train_config.json"
    if not config_path.exists():
        return State("crashed", "checkpoint without train_config.json")

    config = json.loads(config_path.read_text())
    policy, method = config["policy"], config.get("method", {})

    # The cell says what this run is meant to be; its own config says what it is.
    # Disagreement means MATRIX has drifted from the pipeline scripts, and filing it
    # anyway would put a baseline where the eval expects a speedup arm.
    if policy["type"] != cell.policy:
        return State("crashed", f"MISFILED: config says policy={policy['type']}, not {cell.policy}")
    if method.get("type") != cell.method:
        return State("crashed", f"MISFILED: config says method={method.get('type')}, not {cell.method}")

    # `last` points at a numbered checkpoint; that number is how far the run got.
    reached = int(os.path.basename(os.path.realpath(last)))
    planned = int(config.get("steps") or 0)
    if planned and reached < planned:
        return State("partial", f"{reached // 1000}k of {planned // 1000}k steps", model)
    return State("trained", f"{reached // 1000}k steps", model)


def link(destination: Path, target: Path) -> None:
    """Point `destination` at `target`, relatively, replacing whatever was there.

    Relative so the tree survives the checkout being moved or bind-mounted, which is
    the whole reason the pipeline scripts stopped hardcoding absolute paths.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(os.path.relpath(target, destination.parent))


MARK = {
    "trained": "OK  ", "partial": "WIP ", "crashed": "FAIL",
    "absent": "--  ", "unscheduled": "n/a ",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only; write nothing")
    args = parser.parse_args()

    states = {cell: inspect(cell) for cell in MATRIX}

    if not args.check:
        # Rebuilt from scratch: a cell that has gone away since the last run must not
        # leave a stale symlink behind claiming a checkpoint that no longer exists.
        # rmtree unlinks symlinks rather than descending through them, which is the
        # only reason it is safe to point at a tree whose every leaf is a link into
        # outputs/train -- deleting *through* those links would delete the run.
        if OUT.exists():
            shutil.rmtree(OUT)
        for cell, state in states.items():
            if state.target is not None:
                link(OUT / cell.task / cell.policy / cell.arm, state.target)
        for name, path in PRETRAINED.items():
            if path.exists():
                link(OUT / "pretrained" / name, path)

    lines = ["# Trained checkpoints", "",
             "Generated by `link_checkpoints.py`; every entry is a symlink to the",
             "`pretrained_model` directory a run wrote. Regenerate after any training run.", ""]
    width = max(len(f"{c.task}/{c.policy}/{c.arm}") for c in MATRIX)
    for cell, state in states.items():
        label = f"{cell.task}/{cell.policy}/{cell.arm}"
        lines.append(f"    {MARK[state.status]} {label:<{width}}  {state.detail}")
    lines += ["", "OK = linked, WIP = training now, FAIL = needs a rerun,",
              "-- = never started, n/a = no pipeline stage produces it.", "",
              "PACE has no cell: it acts at eval time on a baseline arm's weights.",
              "B-spline has none either: `methods/bspline` is a library with no",
              "`--method.type` registered, so no run can be scheduled for it yet.", ""]
    report = "\n".join(lines)

    if not args.check:
        (OUT / "INDEX.md").write_text(report)
    print(report)

    counts = {}
    for state in states.values():
        counts[state.status] = counts.get(state.status, 0) + 1
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
