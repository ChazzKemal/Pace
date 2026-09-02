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

    python real/scripts/chunk_trace.py                 # browse runs, then chunks
    python real/scripts/chunk_trace.py RUN_DIR -c 5
    python real/scripts/chunk_trace.py RUN_DIR -c 5 -o chunk5.png
    python real/scripts/chunk_trace.py --gui              # 3-D, coloured by s_eff
    python real/scripts/chunk_trace.py --gui -c 1 -o c1.png   # ...to a file, headless

Needs the run to carry trace.npz (``record_trace``), commands.csv and poses.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import curses
import json
import os
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
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in col])
        except ValueError:
            # chunks.csv carries text columns too (anchor_mode); keep them readable
            # rather than dropping the whole file over one non-numeric field.
            out[k] = np.array(col, dtype=object)
    return out


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


class RunData:
    """Everything one run's files yield, loaded once so the browser can page chunks."""

    def __init__(self, run: Path):
        self.run = run
        self.rc = yaml.safe_load((run / "run_config.yaml").read_text())
        self.cmd = read_csv(run / "commands.csv")
        self.poses = read_csv(run / "poses.csv")
        sp = run / "summary.json"
        self.summary = json.loads(sp.read_text()) if sp.exists() else {}
        tp = run / "trace.npz"
        self.chunks = np.load(tp, allow_pickle=True)["chunk"] if tp.exists() else None
        # The sender's own cycle (~2 ms), NOT 1/fps: `cycles` counts controller cycles,
        # and the sender quantises dwell to whole ones. Same source report.py reads.
        self.control_dt = float(self.summary.get("control_dt_ms", 2.0)) / 1000.0
        self.bounds = chunk_bounds(self.cmd) if self.cmd else []
        self._lag = None

    @property
    def missing(self) -> str:
        if self.cmd is None:
            return "no commands.csv -- rerun with recording enabled"
        if self.chunks is None:
            return "no trace.npz -- rerun with record_trace on"
        return ""

    @property
    def n_chunks(self) -> int:
        return min(len(self.bounds), 0 if self.chunks is None else len(self.chunks))

    @property
    def lag_s(self) -> float:
        if self._lag is None:
            self._lag = (run_median_lag(self.poses, self.cmd, self.summary, self.run)
                         if self.poses is not None else 0.0)
        return self._lag


def describe(d: RunData, idx: int) -> list[str]:
    """The per-chunk breakdown, as lines -- shared by the CLI and the browser."""
    lo, hi = d.bounds[idx]
    st = stages(d.chunks[idx], d.rc)
    rc, out = d.rc, []
    out.append(f"chunk {idx} of {d.n_chunks}   method={rc.get('method', {}).get('type')}")
    out.append("")
    out.append("HOW THE CHUNK WAS BUILT")
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
        out.append(f"  {label:<34}{k:>5} rows   {note}")
    logged = hi - lo
    ok = logged == len(st["emitted"])
    out.append(f"  {'commands actually logged':<34}{logged:>5} rows   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        out.append("     -> this script's model of the pipeline disagrees with the run;")
        out.append("        trust the logged number, not the reconstruction above.")
    out.append("")
    if st["exempt_added"] and st["truncated_off"]:
        out.append(f"  NOTE the exemption kept {len(st['exempt_added'])} extra rows around the grasp and")
        out.append(f"       the truncation then cut {len(st['truncated_off'])} from the tail. The exemption does")
        out.append("       not lengthen the chunk; it moves rows from the end into the grasp.")
        out.append("")

    out.append("HOW MUCH OF IT THE ARM REACHED")
    if d.poses is None:
        out.append("  no poses.csv in this run")
    else:
        r = reach(d.cmd, d.poses, lo, hi, d.control_dt, d.lag_s)
        out.append(f"  window {r['window_s']:.2f} s, {r['n_poses']} pose samples, "
                   f"shifted {d.lag_s * 1000:.0f} ms for servo lag")
        out.append(f"  {'axis':<6}{'commanded':>12}{'achieved':>11}{'reached':>10}")
        for ax, v in r["axes"].items():
            out.append(f"  {ax:<6}{v['cmd_mm']:>10.1f}mm{v['ach_mm']:>9.1f}mm{v['ratio']:>9.0%}")
        out.append("")
        out.append("  Over a whole run this ratio needs no time alignment. Over one chunk it does --")
        out.append("  the window is ~2 s and the servo lags a large fraction of it -- so the pose")
        out.append("  window is shifted by the run's median lag. Read one chunk as indicative;")
        out.append("  report.py's run-level ratio is the robust one.")

    sp = d.cmd["s_eff"][lo:hi]
    out.append("")
    out.append(f"SPEED IN THIS CHUNK   median {np.median(sp):.3f}x   "
               f"range {sp.min():.3f}-{sp.max():.3f}x")
    return out


def report(run: Path, idx: int, out_png: Path | None) -> int:
    d = RunData(run)
    if d.missing:
        sys.exit(f"{run.name}: {d.missing}")
    if idx >= d.n_chunks:
        sys.exit(f"chunk {idx} out of range (this run has {d.n_chunks})")
    print(f"run   {run.name}")
    for line in describe(d, idx):
        print(line)
    if out_png:
        lo, hi = d.bounds[idx]
        plot(run, idx, stages(d.chunks[idx], d.rc), d.cmd, d.poses, lo, hi,
             d.control_dt, out_png)
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




# ---------------------------------------------------------------------------
# Picking a run, and paging through its chunks
# ---------------------------------------------------------------------------

class RunRow:
    """One selectable run: what it was, and whether it carries what we need."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        man = path / "manifest.json"
        m = json.loads(man.read_text()) if man.exists() else {}
        self.task = str(m.get("task") or "?")
        self.method = str(m.get("method") or "?")
        have = (path / "commands.csv").exists()
        trace = (path / "trace.npz").exists()
        self.blocked = ("" if have and trace
                        else "no commands.csv" if not have else "no trace.npz")
        self.n_chunks = 0
        if not self.blocked:
            try:
                self.n_chunks = len(np.load(path / "trace.npz", allow_pickle=True)["chunk"])
            except (OSError, ValueError, KeyError):
                self.blocked = "trace.npz unreadable"


def discover_runs(root: Path = CACHE, limit: int = 40) -> list[RunRow]:
    """Newest first. Runs missing their recordings are listed and flagged, not hidden:
    "that run has no trace" is the answer to why it cannot be opened."""
    if not root.exists():
        return []
    dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return [RunRow(p) for p in dirs[:limit]]


def _colors() -> dict:
    if not curses.has_colors():
        return dict.fromkeys(("head", "dim", "warn", "key", "ok"), 0)
    curses.start_color()
    curses.use_default_colors()
    for i, fg in enumerate((curses.COLOR_CYAN, curses.COLOR_WHITE, curses.COLOR_YELLOW,
                            curses.COLOR_MAGENTA, curses.COLOR_GREEN), start=1):
        curses.init_pair(i, fg, -1)
    return {"head": curses.color_pair(1) | curses.A_BOLD, "dim": curses.A_DIM,
            "warn": curses.color_pair(3), "key": curses.color_pair(4),
            "ok": curses.color_pair(5)}


def _put(scr, y, x, text, attr=0):
    h, w = scr.getmaxyx()
    if 0 <= y < h and x < w:
        scr.addnstr(y, x, text, max(0, w - x - 1), attr)


def _pick_run(scr, runs: list[RunRow], col) -> RunRow | None:
    cur = 0
    while True:
        scr.erase()
        _put(scr, 0, 0, "  deploy runs", col["head"])
        _put(scr, 1, 0, f"  {CACHE}", col["dim"])
        wname = max(len(r.name) for r in runs)
        for i, r in enumerate(runs):
            y = 3 + i
            if y >= scr.getmaxyx()[0] - 3:
                break
            row = f" {'>' if i == cur else ' '} {r.name:<{wname}}  {r.method:<12} "
            _put(scr, y, 0, row, curses.A_REVERSE if i == cur else 0)
            tail = (f"! {r.blocked}" if r.blocked
                    else f"{r.n_chunks} chunks   task {r.task}")
            _put(scr, y, len(row), tail, col["warn"] if r.blocked else col["dim"])
        _put(scr, min(3 + len(runs) + 1, scr.getmaxyx()[0] - 2), 0,
             "  up/down move   enter open   q quit", col["key"])
        scr.refresh()
        k = scr.getch()
        if k in (curses.KEY_UP, ord("k")):
            cur = (cur - 1) % len(runs)
        elif k in (curses.KEY_DOWN, ord("j")):
            cur = (cur + 1) % len(runs)
        elif k in (curses.KEY_ENTER, 10, 13):
            if not runs[cur].blocked:
                return runs[cur]
        elif k in (27, ord("q"), ord("Q")):
            return None


def _browse(scr, row: RunRow, col) -> Path | None:
    """Page through the run's chunks. Returns a PNG path if one was written."""
    d = RunData(row.path)
    idx, wrote = 0, None
    while True:
        scr.erase()
        _put(scr, 0, 0, f"  {row.name}", col["head"])
        _put(scr, 1, 0, f"  {row.task} / {row.method}   {d.n_chunks} chunks", col["dim"])
        try:
            lines = describe(d, idx)
        except (IndexError, ValueError) as exc:
            lines = [f"cannot describe chunk {idx}: {exc}"]
        for i, line in enumerate(lines):
            attr = 0
            if line.startswith(("HOW ", "SPEED ")):
                attr = col["head"]
            elif "MISMATCH" in line or line.strip().startswith(("NOTE", "->")):
                attr = col["warn"]
            elif line.strip().startswith("OK") or line.endswith("OK"):
                attr = col["ok"]
            _put(scr, 3 + i, 2, line, attr)
        _put(scr, scr.getmaxyx()[0] - 2, 0,
             "  left/right chunk   p write PNG   b back   q quit", col["key"])
        if wrote:
            _put(scr, scr.getmaxyx()[0] - 3, 2, f"wrote {wrote}", col["ok"])
        scr.refresh()
        k = scr.getch()
        if k in (curses.KEY_RIGHT, ord("l"), ord(" ")):
            idx = (idx + 1) % d.n_chunks
            wrote = None
        elif k in (curses.KEY_LEFT, ord("h")):
            idx = (idx - 1) % d.n_chunks
            wrote = None
        elif k in (ord("p"), ord("P")):
            out = Path.cwd() / f"{row.name}_chunk{idx}.png"
            lo, hi = d.bounds[idx]
            plot(row.path, idx, stages(d.chunks[idx], d.rc), d.cmd, d.poses,
                 lo, hi, d.control_dt, out)
            wrote = out
        elif k in (ord("b"), ord("B")):
            return None
        elif k in (ord("q"), ord("Q")):
            # Deliberately NOT bare ESC: arrow keys arrive as ESC-[-C, and if ncurses
            # has not assembled the sequence yet a lone 27 would read as quit mid-page.
            # `b` goes back, `q` exits; ESC keeps its meaning only on the run list.
            raise KeyboardInterrupt


def _screen(scr, runs):
    curses.curs_set(0)
    # Resolve a lone ESC quickly so it does not swallow the start of an arrow key.
    with contextlib.suppress(AttributeError, curses.error):
        curses.set_escdelay(25)
    col = _colors()
    while True:
        row = _pick_run(scr, runs, col)
        if row is None:
            return
        _browse(scr, row, col)


def browse_tui() -> int:
    """Pick a run, then page its chunks. Falls back to a message if not a terminal."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("not a terminal -- pass a run directory, e.g. "
                 "chunk_trace.py RUN_DIR -c 3")
    runs = discover_runs()
    if not runs:
        sys.exit(f"no runs under {CACHE}")
    try:
        curses.wrapper(_screen, runs)
    except KeyboardInterrupt:
        pass
    except curses.error:
        sys.exit("could not open the browser; pass a run directory explicitly")
    return 0




# ---------------------------------------------------------------------------
# 3-D viewer
# ---------------------------------------------------------------------------

def _backend(save: Path | None) -> str:
    """A matplotlib backend that exists here, rather than one we wish existed.

    Order matters. Writing a file must work headless, so that is Agg regardless of
    what is installed. For an interactive window WebAgg is the nicest over SSH -- a
    browser UI with no X server, which is why crisp_gym's 27_speedup_slider_viewer
    defaults to it -- but it needs tornado, and this environment does not ship it.
    TkAgg is present (matplotlib's own default here) and needs a display. An explicit
    MPLBACKEND always wins; this only fills in the default.
    """
    if save:
        return "Agg"
    with contextlib.suppress(ImportError):
        import tornado  # noqa: F401
        return "WebAgg"
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        with contextlib.suppress(ImportError):
            import tkinter  # noqa: F401
            return "TkAgg"
    sys.exit(
        "no interactive matplotlib backend available.\n"
        "  - write a file instead:  --gui -c N -o chunk.png\n"
        "  - or over SSH, install tornado for the browser UI:\n"
        "      pixi add --manifest-path real/pixi.toml --feature dev tornado\n"
        "  - or forward a display (ssh -X) and matplotlib will use TkAgg."
    )


def pick_run_interactively() -> Path | None:
    """The curses run list, used on its own so --gui need not be given a path."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    runs = discover_runs()
    if not runs:
        sys.exit(f"no runs under {CACHE}")

    chosen = {}

    def _screen_once(scr):
        curses.curs_set(0)
        with contextlib.suppress(AttributeError, curses.error):
            curses.set_escdelay(25)
        row = _pick_run(scr, runs, _colors())
        if row is not None:
            chosen["path"] = row.path

    with contextlib.suppress(curses.error, KeyboardInterrupt):
        curses.wrapper(_screen_once)
    return chosen.get("path")


def gui(run: Path, start: int | None = None, save: Path | None = None) -> int:
    """The whole run in 3-D: what the policy generated, what actually ran, and when.

    Shaped after crisp_gym's examples/27_speedup_slider_viewer.py, which colours a
    *dataset* episode by a schedule being tuned. This shows a *recorded deploy*, so
    three things are drawn together that only exist after a run:

    * every action the policy generated (faint), against the subset that executed --
      the gap between them is striding, truncation and the blend hold-back, made
      visual rather than arithmetic;
    * the achieved path beside the commanded one, which is where amplitude loss shows;
    * when inference for the *next* chunk began, as a fraction of the current one.
      The producer asks for it once the queue falls to `overlap_threshold`, so this is
      the real inference budget rather than the configured one.

    WebAgg by default when tornado is present, else TkAgg with a display. See _backend.
    """
    import os
    os.environ.setdefault("MPLBACKEND", _backend(save))
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, RangeSlider

    d = RunData(run)
    if d.missing:
        sys.exit(f"{run.name}: {d.missing}")
    n = d.n_chunks

    # Per-chunk execution windows, and where the next inference began inside each.
    ch = read_csv(run / "chunks.csv") or {}
    q_inf = ch.get("q_before_inf")
    synth = ch.get("synth_ms")
    spans = []
    for i in range(n):
        lo, hi = d.bounds[i]
        dur = float((d.cmd["cycles"][lo:hi] * d.control_dt).sum())
        t0 = float(d.cmd["t_wall"][lo] - d.cmd["t_wall"][0])
        k = hi - lo
        # Inference is requested when the queue drains to q_before_inf items, so the
        # fraction of this chunk already consumed by then is (K - q) / K.
        frac = (1.0 - float(q_inf[i]) / k) if q_inf is not None and i < len(q_inf) and k else None
        spans.append({"t0": t0, "dur": dur, "k": k, "frac": frac,
                      "synth": float(synth[i]) / 1000.0 if synth is not None and i < len(synth) else 0.0})

    fig = plt.figure(figsize=(15.5, 9))
    fig.canvas.manager.set_window_title(run.name)
    ax3 = fig.add_axes([0.01, 0.26, 0.50, 0.68], projection="3d")
    axi = fig.add_axes([0.58, 0.60, 0.40, 0.33])
    axt = fig.add_axes([0.58, 0.24, 0.40, 0.30]); axt.axis("off")
    ax_rs = fig.add_axes([0.08, 0.13, 0.44, 0.03])
    ax_rb = fig.add_axes([0.60, 0.06, 0.13, 0.13])
    init = (start, start) if start is not None and 0 <= start < n else (0, max(n - 1, 1))
    rs = RangeSlider(ax_rs, "chunks", 0, max(n - 1, 1), valinit=init, valstep=1)
    rb = RadioButtons(ax_rb, ("colour: s_eff", "colour: chunk", "hide generated"))
    state = {"cb": None, "mode": 0}

    def draw(*_):
        lo_c, hi_c = (int(v) for v in rs.val)
        mode = state["mode"]
        ax3.clear()

        # Everything the policy generated, for the chunks in view.
        if mode != 2 and d.chunks is not None:
            g = np.concatenate([d.chunks[i][:, :3] for i in range(lo_c, hi_c + 1)])
            ax3.scatter(g[:, 0], g[:, 1], g[:, 2], s=3, c="0.75", alpha=0.35,
                        label="generated, not executed")

        rows = np.concatenate([np.arange(*d.bounds[i]) for i in range(lo_c, hi_c + 1)])
        cx, cy, cz = (d.cmd[f"cmd_{a}"][rows] for a in "xyz")
        if mode == 1:
            cid = np.concatenate([np.full(d.bounds[i][1] - d.bounds[i][0], i)
                                  for i in range(lo_c, hi_c + 1)])
            sc = ax3.scatter(cx, cy, cz, c=cid, cmap="turbo", s=16)
            label = "chunk index"
        else:
            se = d.cmd["s_eff"][rows]
            sc = ax3.scatter(cx, cy, cz, c=se, cmap="viridis", s=16,
                             vmin=1.0, vmax=max(float(se.max()), 1.01))
            label = "s_eff"

        # Chunk starts, so a point in space can be tied back to a chunk number.
        for i in range(lo_c, hi_c + 1):
            b = d.bounds[i][0]
            ax3.text(d.cmd["cmd_x"][b], d.cmd["cmd_y"][b], d.cmd["cmd_z"][b],
                     str(i), fontsize=7, color="#b5533a")

        if d.poses is not None:
            # Window from the RECONSTRUCTED execution times, not t_wall: every command
            # in a chunk carries the same push timestamp, so a t_wall window over one
            # chunk has zero width and selects no poses at all.
            base = d.cmd["t_wall"][0] + d.lag_s
            t0 = base + spans[lo_c]["t0"]
            t1 = base + spans[hi_c]["t0"] + spans[hi_c]["dur"]
            w = (d.poses["t_wall"] >= t0) & (d.poses["t_wall"] <= t1)
            if w.any():
                ax3.plot(d.poses["ach_x"][w], d.poses["ach_y"][w], d.poses["ach_z"][w],
                         color="#eb6834", lw=1.4, alpha=0.9, label="achieved")
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")
        ax3.set_title(f"chunks {lo_c}-{hi_c} of {n - 1}", fontsize=10)
        ax3.legend(loc="upper left", fontsize=7.5)
        if state["cb"] is None:
            state["cb"] = fig.colorbar(sc, ax=ax3, fraction=0.02, pad=0.10, shrink=0.55)
        else:
            state["cb"].update_normal(sc)
        state["cb"].set_label(label)

        # When the next chunk was inferred, as a fraction of this one.
        axi.clear()
        for i in range(lo_c, hi_c + 1):
            sp = spans[i]
            axi.barh(i, sp["dur"], left=sp["t0"], height=0.6, color="#2a78d6", alpha=.30)
            if sp["frac"] is not None:
                x = sp["t0"] + sp["frac"] * sp["dur"]
                axi.plot([x, x], [i - .34, i + .34], color="#b5533a", lw=1.6)
                axi.barh(i, max(sp["synth"], 0.02), left=x, height=0.6, color="#b5533a")
        axi.set_xlabel("seconds into the run"); axi.set_ylabel("chunk")
        axi.set_title("execution window, and when the next chunk was inferred", fontsize=10)
        axi.grid(alpha=0.15, axis="x")
        axi.set_yticks(range(lo_c, hi_c + 1) if hi_c - lo_c < 12
                       else range(lo_c, hi_c + 1, max(1, (hi_c - lo_c) // 10)))
        axi.set_ylim(hi_c + 0.6, lo_c - 0.6)

        axt.clear(); axt.axis("off")
        gen = sum(len(d.chunks[i]) for i in range(lo_c, hi_c + 1)) if d.chunks is not None else 0
        used = len(rows)
        fr = [sp["frac"] for sp in spans[lo_c:hi_c + 1] if sp["frac"] is not None]
        sy = [sp["synth"] * 1000 for sp in spans[lo_c:hi_c + 1]]
        txt = [
            f"chunks {lo_c}-{hi_c}   method={d.rc.get('method', {}).get('type')}",
            "",
            f"  actions generated      {gen:>6}",
            f"  actions executed       {used:>6}   ({used / max(gen, 1):.0%} of generated)",
            f"  dropped before sending {gen - used:>6}   striding, n_action_steps, blend",
            "",
            "NEXT CHUNK INFERRED",
            f"  after {np.median(fr):.0%} of the current chunk had run" if fr else "  (no chunks.csv)",
            f"  inference took {np.median(sy):.0f} ms median, {max(sy):.0f} ms worst" if sy else "",
            f"  leaving {(1 - np.median(fr)) * np.median([s['dur'] for s in spans]) * 1000:.0f} ms"
            " of queue as margin" if fr else "",
            "",
            *describe(d, lo_c)[2:8],
        ]
        axt.text(0, 1, "\n".join(t for t in txt if t is not None), family="monospace",
                 fontsize=7.6, va="top", transform=axt.transAxes)
        fig.canvas.draw_idle()

    def on_radio(lbl):
        state["mode"] = ("colour: s_eff", "colour: chunk", "hide generated").index(lbl)
        draw()

    rs.on_changed(draw)
    rb.on_clicked(on_radio)
    draw()
    if save:
        fig.savefig(save, dpi=130)
        print(f"wrote {save}  ({n} chunks)")
        return 0
    print(f"{run.name}: {n} chunks — drag the range slider to filter")
    plt.show()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", help="deploy_runs/<ts>; default: newest")
    ap.add_argument("-c", "--chunk", type=int, default=None,
                    help="one chunk; in --gui this focuses the filter on it")
    ap.add_argument("-o", "--out", default=None, help="write a PNG as well")
    ap.add_argument("--gui", action="store_true",
                    help="3-D viewer: commanded path coloured by s_eff, chunk slider")
    ap.add_argument("--no-tui", action="store_true",
                    help="never open the browser; use the newest run")
    a = ap.parse_args()
    # No run named and nothing else asked for: browse. An explicit --chunk or --out
    # means the caller wants one answer on stdout, so honour that without a screen.
    if a.gui:
        run = Path(a.run_dir) if a.run_dir else pick_run_interactively()
        if run is None:
            return 0
        return gui(run, a.chunk, Path(a.out) if a.out else None)
    if not a.run_dir and a.chunk is None and not a.out and not a.no_tui:
        return browse_tui()
    run = Path(a.run_dir) if a.run_dir else newest_run()
    return report(run, a.chunk or 0, Path(a.out) if a.out else None)


if __name__ == "__main__":
    sys.exit(main())
