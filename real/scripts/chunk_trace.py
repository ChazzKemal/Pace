#!/usr/bin/env python
"""What happened to one chunk, from the policy's prediction to the arm's reply.

``report.py`` answers "how did the run go". This answers "what became of chunk N" --
which of the policy's predicted rows survived each pipeline stage, and how much of
what was finally commanded the arm actually reached.

The stages are reconstructed rather than logged, because nothing in the deploy path
records intermediate row sets. The reconstruction is exact for the parts that are
deterministic given the run config (striding, the gripper exemption, the
``n_action_steps`` truncation, the blend hold-back) and the total is checked against
the ``K`` the loop recorded -- a mismatch is reported rather than hidden, since it
means this script's model of the pipeline has drifted from the pipeline.

    python real/scripts/chunk_trace.py                 # newest run, chunk 0
    python real/scripts/chunk_trace.py RUN_DIR -c 5
    python real/scripts/chunk_trace.py RUN_DIR -c 5 -o chunk5.png

Needs the run to carry trace.npz (``record_trace``), commands.csv and poses.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

CACHE = Path.home() / ".cache/huggingface/lerobot/deploy_runs"

#: A push gap this large means a new chunk was handed to the queue. Commands inside a
#: chunk are pushed in a batch microseconds apart, so any real gap is a chunk boundary.
CHUNK_GAP_S = 0.2


def newest_run() -> Path:
    runs = [p for p in CACHE.iterdir() if p.is_dir()] if CACHE.exists() else []
    if not runs:
        sys.exit(f"no runs under {CACHE}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def chunk_bounds(cmd: dict) -> list[tuple[int, int]]:
    """(start, end) into the command rows for each chunk, from the push-time gaps."""
    t = cmd["t_wall"]
    gaps = np.where(np.diff(t) > CHUNK_GAP_S)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps + 1, len(t)]
    return list(zip(starts.tolist(), ends.tolist()))


def stages(predicted: np.ndarray, rc: dict) -> dict:
    """Row indices surviving each stage, for one predicted chunk.

    Order mirrors the deploy path: crisp_gym strides (loop.stride) before the method
    pipeline, PACE strides and then truncates inside PaceSpeedStep, and the loop holds
    back the blend overlap last.
    """
    import torch

    from pace_bench.methods.config import PaceMethod
    from pace_bench.methods.pace.speed import stride_indices

    n = predicted.shape[0]
    m = rc.get("method", {})
    out = {"predicted": list(range(n))}

    if m.get("type") != "pace":
        out["strided"] = out["predicted"]
        out["exempt_added"] = []
    else:
        cfg = PaceMethod(**{k: v for k, v in m.items() if k != "type"}).to_pace_config()
        t = torch.from_numpy(predicted[None]).float()
        kept = stride_indices(t, cfg)
        plain = list(range(0, n, max(1, cfg.action_stride)))
        out["strided"] = kept
        out["exempt_added"] = sorted(set(kept) - set(plain))

    n_act = rc.get("n_action_steps")
    kept = out["strided"]
    out["after_truncate"] = kept[:n_act] if n_act else kept
    out["truncated_off"] = kept[len(out["after_truncate"]):]

    overlap = int((rc.get("blend") or {}).get("overlap", 0) or 0)
    hold = min(overlap, len(out["after_truncate"]) // 2) if overlap else 0
    out["emitted"] = out["after_truncate"][:len(out["after_truncate"]) - hold] if hold else out["after_truncate"]
    out["blend_held"] = out["after_truncate"][len(out["emitted"]):]
    return out


def reach(cmd: dict, poses: dict, lo: int, hi: int, control_dt: float,
          lag_s: float = 0.0) -> dict:
    """How much of what was commanded in [lo, hi) the arm covered, per axis.

    Amplitude ratio -- achieved span over commanded span. Over a whole run that needs
    no time alignment, which is what makes it report.py's most robust number. Over ONE
    chunk it does: the window is under two seconds and the servo lags a good fraction
    of that, so an unshifted window is measuring the tail of the *previous* chunk's
    command as much as this one's. The window is therefore shifted by the run's median
    lag, and the caller is told the shift so the number can be read for what it is.
    """
    dwell = cmd["cycles"][lo:hi] * control_dt
    t0 = cmd["t_wall"][lo]
    t_cmd = t0 + np.concatenate([[0.0], np.cumsum(dwell)[:-1]])
    t_end = t_cmd[-1] + dwell[-1]
    win = (poses["t_wall"] >= t0 + lag_s) & (poses["t_wall"] <= t_end + lag_s)
    out = {"n_poses": int(win.sum()), "window_s": float(t_end - t0),
           "lag_s": float(lag_s), "axes": {}}
    for ax in ("x", "y", "z"):
        c = cmd[f"cmd_{ax}"][lo:hi]
        a = poses[f"ach_{ax}"][win]
        cs = float(c.max() - c.min())
        as_ = float(a.max() - a.min()) if a.size else 0.0
        out["axes"][ax] = {"cmd_mm": cs * 1000, "ach_mm": as_ * 1000,
                           "ratio": (as_ / cs) if cs > 1e-6 else float("nan")}
    return out


def run_median_lag(poses: dict, cmd: dict, summary: dict, run: Path) -> float:
    """The run's median best-fit servo lag, from report.py so the two agree."""
    try:
        from pace_bench.real.report import analyse
        man_path = run / "manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        return float(analyse(poses, cmd, summary, man).median_lag_ms) / 1000.0
    except (ImportError, KeyError, ValueError, OSError):
        # No lag correction is better than a wrong one; the caller prints the shift.
        return 0.0


def report(run: Path, idx: int, out_png: Path | None) -> int:
    rc = yaml.safe_load((run / "run_config.yaml").read_text())
    cmd = read_csv(run / "commands.csv")
    poses = read_csv(run / "poses.csv")
    if cmd is None:
        sys.exit(f"{run.name} has no commands.csv -- rerun with recording enabled")
    trace_path = run / "trace.npz"
    if not trace_path.exists():
        sys.exit(f"{run.name} has no trace.npz -- rerun with record_trace on")
    C = np.load(trace_path, allow_pickle=True)["chunk"]

    bounds = chunk_bounds(cmd)
    if idx >= len(bounds) or idx >= len(C):
        sys.exit(f"chunk {idx} out of range (trace {len(C)}, commands {len(bounds)})")
    lo, hi = bounds[idx]
    st = stages(C[idx], rc)
    # The sender's own cycle (~2 ms), NOT 1/fps: `cycles` counts controller cycles,
    # and the sender quantises dwell to whole ones. Same source report.py reads.
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    control_dt = float(summary.get("control_dt_ms", 2.0)) / 1000.0

    print(f"run   {run.name}")
    print(f"chunk {idx} of {len(bounds)}   method={rc.get('method', {}).get('type')}")
    print()
    print("HOW THE CHUNK WAS BUILT")
    n = len(st["predicted"])
    stride = rc.get("method", {}).get("action_stride", 1)
    rows = [
        ("policy predicted", n, ""),
        (f"after stride {stride}", len(st["strided"]),
         f"+{len(st['exempt_added'])} kept by the gripper exemption" if st["exempt_added"] else ""),
        (f"after n_action_steps={rc.get('n_action_steps')}", len(st["after_truncate"]),
         f"-{len(st['truncated_off'])} cut from the tail" if st["truncated_off"] else ""),
        (f"after blend hold-back {(rc.get('blend') or {}).get('overlap')}", len(st["emitted"]),
         f"-{len(st['blend_held'])} held for the next seam" if st["blend_held"] else ""),
    ]
    for label, k, note in rows:
        print(f"  {label:<34}{k:>5} rows   {note}")
    logged = hi - lo
    mark = "OK" if logged == len(st["emitted"]) else "MISMATCH"
    print(f"  {'commands actually logged':<34}{logged:>5} rows   {mark}")
    if logged != len(st["emitted"]):
        print("     -> this script's model of the pipeline disagrees with the run;")
        print("        trust the logged number, not the reconstruction above.")
    print()

    if st["exempt_added"] and st["truncated_off"]:
        print(f"  NOTE the exemption kept {len(st['exempt_added'])} extra rows around the grasp and the")
        print(f"       truncation then cut {len(st['truncated_off'])} from the tail. The exemption does not")
        print("       lengthen the chunk; it moves rows from the end into the grasp.")
        print()

    print("HOW MUCH OF IT THE ARM REACHED")
    if poses is None:
        print("  no poses.csv in this run")
    else:
        lag_s = run_median_lag(poses, cmd, summary, run)
        r = reach(cmd, poses, lo, hi, control_dt, lag_s)
        print(f"  window {r['window_s']:.2f} s, {r['n_poses']} pose samples, "
              f"shifted {lag_s * 1000:.0f} ms for servo lag")
        print(f"  {'axis':<6}{'commanded':>12}{'achieved':>11}{'reached':>10}")
        for ax, v in r["axes"].items():
            print(f"  {ax:<6}{v['cmd_mm']:>10.1f}mm{v['ach_mm']:>9.1f}mm{v['ratio']:>9.0%}")
        print()
        print("  Over a whole run this ratio needs no time alignment. Over one chunk it does --")
        print("  the window is ~2 s and the servo lags a large fraction of it -- so the pose")
        print("  window is shifted by the run's median lag. Read a single chunk's number as")
        print("  indicative; report.py's run-level ratio is the robust one.")

    s = cmd["s_eff"][lo:hi]
    print()
    print(f"SPEED IN THIS CHUNK   median {np.median(s):.3f}x   range {s.min():.3f}-{s.max():.3f}x")

    if out_png:
        plot(run, idx, st, cmd, poses, lo, hi, control_dt, out_png)
        print(f"\nwrote {out_png}")
    return 0


def plot(run, idx, st, cmd, poses, lo, hi, control_dt, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                             gridspec_kw={"height_ratios": [1, 2, 1.4]})
    fig.suptitle(f"{run.name} — chunk {idx}", fontsize=12, x=0.02, ha="left")

    # 1. row fate strip
    ax = axes[0]
    n = len(st["predicted"])
    fate = np.zeros(n)
    for i in st["strided"]:
        fate[i] = 1
    for i in st["exempt_added"]:
        fate[i] = 2
    for i in st["truncated_off"]:
        fate[i] = 3
    for i in st["blend_held"]:
        fate[i] = 4
    cmap = {0: "#d8dde2", 1: "#2a78d6", 2: "#eb6834", 3: "#9aa4ad", 4: "#eda100"}
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=cmap[int(fate[i])]))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("predicted row index")
    ax.set_title("what became of each predicted row", fontsize=10, loc="left")
    for c, lab in ((cmap[0], "dropped by stride"), (cmap[1], "executed"),
                   (cmap[2], "kept by gripper exemption"), (cmap[3], "cut by n_action_steps"),
                   (cmap[4], "held for the blend")):
        ax.plot([], [], "s", color=c, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=5, frameon=False, fontsize=8)

    # 2. commanded vs achieved
    ax = axes[1]
    dwell = cmd["cycles"][lo:hi] * control_dt
    t0 = cmd["t_wall"][lo]
    t_cmd = np.concatenate([[0.0], np.cumsum(dwell)[:-1]])
    t_end = t_cmd[-1] + dwell[-1]
    for ax_name, col in (("x", "#2a78d6"), ("y", "#eb6834"), ("z", "#1baf7a")):
        ax.plot(t_cmd, cmd[f"cmd_{ax_name}"][lo:hi] * 1000, color=col, lw=1.8,
                label=f"{ax_name} commanded")
        if poses is not None:
            w = (poses["t_wall"] >= t0) & (poses["t_wall"] <= t0 + t_end)
            if w.any():
                ax.plot(poses["t_wall"][w] - t0, poses[f"ach_{ax_name}"][w] * 1000,
                        color=col, lw=1.2, ls="--", alpha=0.75, label=f"{ax_name} achieved")
    ax.set_ylabel("mm")
    ax.set_xlabel("seconds into the chunk")
    ax.set_title("commanded (solid) against achieved (dashed)", fontsize=10, loc="left")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.15)

    # 3. speed
    ax = axes[2]
    ax.step(t_cmd, cmd["s_eff"][lo:hi], where="post", color="#eb6834", lw=1.8)
    ax.set_ylabel("s_eff")
    ax.set_xlabel("seconds into the chunk")
    ax.set_title("commanded speed", fontsize=10, loc="left")
    ax.grid(alpha=0.15)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", help="deploy_runs/<ts>; default: newest")
    ap.add_argument("-c", "--chunk", type=int, default=0)
    ap.add_argument("-o", "--out", default=None, help="write a PNG as well")
    a = ap.parse_args()
    run = Path(a.run_dir) if a.run_dir else newest_run()
    return report(run, a.chunk, Path(a.out) if a.out else None)


if __name__ == "__main__":
    sys.exit(main())
