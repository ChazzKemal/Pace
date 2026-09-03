"""When each chunk was inferred, relative to what the arm was executing at the time.

The deploy loop overlaps inference with execution: it asks for the next chunk once
the queue drains to ``overlap_threshold`` items, then appends the result *after* the
last queued deadline (``loop.py:436-451``). That is the right thing for cadence -- the
sender never runs dry -- but it means every chunk executes later than the policy
predicted it for. The policy saw the arm at ``t_obs`` and emitted rows for
``t_obs + j x dt``; row 0 actually goes out only once the queue in front of it has been
published. Nothing in the run folder states that offset, and it is the number that
decides how stale an executed action is. This module puts it on the clock.

Everything here is reconstructed from files the run already writes, so it works on
every run recorded since ``commands.csv`` existed, not only future ones:

* **When inference ran.** ``chunks.csv`` carries ``synth_ms`` but no timestamp;
  ``trace.npz`` stamps ``wall_ns`` immediately *after* the request returns
  (``loop.py:160``). Start is therefore end minus duration, and the observation was
  taken ``get_obs_ms`` before that. Runs recorded with :class:`record.ChunkClock`
  carry ``inference.csv`` with these measured directly; it is preferred when present.
* **When each target was published.** The producer anchors a chunk's deadlines at the
  previous chunk's last deadline (``overlap``) or at the push instant (``fresh``), and
  ``cycles x control_dt`` is the exact dwell, so deadlines chain from the first push
  by cumulative sum. The C++ sender publishes at the deadline when it is on time
  (``frames.csv`` slack is positive) and at ``deadline - slack`` when late, plus the
  recorded sleep overshoot. That recovers the publish instant to well under a
  millisecond without the absolute stamps crisp_gym's ``_ingest_stats`` discards.
* **Which executed rows are the seam bridge.** The blend rewrites raw rows ``[0:N)``
  of a chunk *before* the method pipeline strides it (``loop.py:282`` runs ahead of
  ``:351``), so the executed bridge is whichever of those raw rows survive striding.
  With ``action_stride: 2`` and ``overlap: 8`` that is 4 executed rows, not 8.

One consequence of that ordering is worth stating because it is invisible in the
data: the Hermite bridge's start tangent ``v_start`` is a difference of two *emitted*
rows (post-stride, ``loop.py:388``) while its end tangent ``v_end`` is a difference of
two *raw* rows, so the two velocities are in units that differ by the stride. The
report flags this whenever a run has both a stride above 1 and a Hermite blend.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: A push gap this large means a new chunk was handed to the queue. Commands inside a
#: chunk are pushed in a batch microseconds apart, so any real gap is a chunk boundary.
#: Used only for runs whose commands.csv predates the recorded ``chunk`` column.
CHUNK_GAP_S = 0.2


def read_csv(path: Path) -> dict[str, np.ndarray] | None:
    """Columns as arrays; numeric where every value parses, text otherwise.

    ``chunks.csv`` carries ``anchor_mode`` as text, and that column is what says where
    the deadline chain restarts, so it cannot be dropped the way ``np.genfromtxt``
    would (it reads as NaN).
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col, dtype=object)
    return out


# ------------------------------------------------------------------ row fates


def stages(predicted: np.ndarray, rc: dict) -> dict:
    """Row indices surviving each pipeline stage, for one predicted chunk.

    Order mirrors the deploy path: crisp_gym strides (``loop.stride``) before the
    method pipeline, PACE strides and then truncates inside ``PaceSpeed``, and the loop
    holds back the blend overlap last. Exact given the run config; ``chunk_trace``
    checks the total against the ``K`` the loop logged and reports any drift.

    Plain striding needs no torch. The adaptive and gripper-exempt paths are PACE's
    own :func:`stride_indices` on the predicted chunk, which does.
    """
    n = int(predicted.shape[0])
    m = rc.get("method", {}) or {}
    out: dict = {"predicted": list(range(n))}

    if m.get("type") != "pace":
        out["strided"] = out["predicted"]
        out["exempt_added"] = []
    else:
        stride = max(1, int(m.get("action_stride", 1) or 1))
        plain = list(range(0, n, stride))
        if m.get("adaptive_stride") or m.get("gripper_stride_exempt"):
            import torch

            from pace_bench.methods.config import PaceMethod
            from pace_bench.methods.pace.speed import stride_indices

            cfg = PaceMethod(**{k: v for k, v in m.items() if k != "type"}).to_pace_config()
            kept = stride_indices(torch.from_numpy(np.asarray(predicted)[None]).float(), cfg)
        else:
            kept = plain
        out["strided"] = kept
        out["exempt_added"] = sorted(set(kept) - set(plain))

    n_act = rc.get("n_action_steps")
    kept = out["strided"]
    out["after_truncate"] = kept[:n_act] if n_act else kept
    out["truncated_off"] = kept[len(out["after_truncate"]):]

    overlap = int((rc.get("blend") or {}).get("overlap", 0) or 0)
    hold = min(overlap, len(out["after_truncate"]) // 2) if overlap else 0
    emitted = out["after_truncate"]
    out["emitted"] = emitted[:len(emitted) - hold] if hold else emitted
    out["blend_held"] = emitted[len(out["emitted"]):]
    return out


def bridge_rows(rc: dict, n_raw: int, emitted: list[int]) -> int:
    """How many of a chunk's executed rows are the seam bridge, not policy output.

    The blend fills raw rows ``[0:N)`` with ``N = min(overlap, K_raw // 2)``
    (``loop.py:276``) and the pipeline then strides them, so the executed count is
    the number of emitted raw indices below N. Zero for the first chunk, which has
    no seam to bridge from -- the caller handles that; this is per-chunk geometry.
    """
    overlap = int((rc.get("blend") or {}).get("overlap", 0) or 0)
    if overlap <= 0 or n_raw < 2:
        return 0
    n = min(overlap, n_raw // 2)
    return sum(1 for i in emitted if i < n)


# ------------------------------------------------------------------ the clock


def chunk_bounds(cmd: dict) -> list[tuple[int, int]]:
    """``(lo, hi)`` row ranges for each chunk, in push order.

    From the recorded ``chunk`` column when the run has one; otherwise from gaps in
    the push time, which is reliable because a chunk's rows are queued in one burst
    and the producer then parks in ``drain_wait`` for most of a second.
    """
    ch = cmd.get("chunk")
    if ch is not None and np.all(np.isfinite(ch)):
        edges = np.where(np.diff(ch) != 0)[0]
    else:
        edges = np.where(np.diff(cmd["t_wall"]) > CHUNK_GAP_S)[0]
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges + 1, len(cmd["t_wall"])]
    return list(zip(starts.tolist(), ends.tolist()))


def deadlines(cmd: dict, control_dt: float, bounds: list[tuple[int, int]],
              anchor_modes=None) -> np.ndarray:
    """Planned publish instant of every commanded row, wall clock.

    The producer's rule (``loop.py:436-451``): a chunk's deadlines are
    ``anchor + cumsum(dt_eff)``, with the anchor being the previous chunk's last
    deadline while it is still in the future and *now* otherwise. Chained from the
    first push, re-anchored on each ``fresh`` chunk at its own push time. Recorded
    ``deadline_wall`` (from :class:`record.ChunkClock`) wins when present -- it is the
    value the sender was actually handed.
    """
    rec = cmd.get("deadline_wall")
    if rec is not None and np.all(np.isfinite(rec)):
        return np.asarray(rec, dtype=float)
    dwell = cmd["cycles"] * control_dt
    t = np.empty(len(dwell))
    anchor = float(cmd["t_wall"][0])
    for ci, (lo, hi) in enumerate(bounds):
        fresh = (anchor_modes is not None and ci < len(anchor_modes)
                 and str(anchor_modes[ci]) == "fresh")
        if ci > 0 and (fresh or anchor < cmd["t_wall"][lo]):
            anchor = float(cmd["t_wall"][lo])
        t[lo:hi] = anchor + np.cumsum(dwell[lo:hi])
        anchor = float(t[hi - 1])
    return t


def published(deadline: np.ndarray, frames: dict | None) -> np.ndarray:
    """When each row actually went out. NaN for rows the sender never reached.

    ``frames.csv`` is written in publish order, one row per published target, so it
    aligns positionally with the command rows; the sender stops at the sentinel and
    the rows still queued at Ctrl-C are the difference in length. A late frame is
    published ``-slack`` after its deadline; an on-time one at the deadline, plus the
    measured overshoot of the sleep.
    """
    pub = np.full(len(deadline), np.nan)
    if frames is None or "slack_ms" not in frames:
        return deadline.copy()
    m = min(len(deadline), len(frames["slack_ms"]))
    late = np.maximum(0.0, -frames["slack_ms"][:m]) / 1000.0
    over = np.nan_to_num(frames.get("sleep_overshoot_ms", np.zeros(m))[:m]) / 1000.0
    pub[:m] = deadline[:m] + late + over
    return pub


@dataclass
class ChunkTiming:
    """One chunk on the wall clock. Times are absolute seconds; None when unknown."""

    idx: int
    lo: int
    hi: int
    t_obs: float | None
    t_req: float | None
    t_ret: float | None
    t_push: float
    t_first: float
    t_last: float
    n_rows: int
    n_published: int
    n_late: int
    n_bridge: int
    #: Queue depth the producer logged when it asked (``q_before_inf``).
    q_logged: int | None = None
    #: Rows of the previous chunk whose publish instant is after ``t_req`` --
    #: the same quantity, but measured from the timeline rather than logged.
    q_measured: int | None = None
    t_first_policy: float | None = None

    @property
    def inference_ms(self) -> float | None:
        if self.t_req is None or self.t_ret is None:
            return None
        return (self.t_ret - self.t_req) * 1000.0

    @property
    def latency_first_ms(self) -> float | None:
        """Observation to the first executed row of this chunk."""
        return None if self.t_obs is None else (self.t_first - self.t_obs) * 1000.0

    @property
    def latency_policy_ms(self) -> float | None:
        """Observation to the first row the policy predicted verbatim (past the bridge)."""
        if self.t_obs is None or self.t_first_policy is None:
            return None
        return (self.t_first_policy - self.t_obs) * 1000.0

    @property
    def exec_s(self) -> float:
        return self.t_last - self.t_first


@dataclass
class Timeline:
    chunks: list[ChunkTiming] = field(default_factory=list)
    #: Publish instant per command row (NaN where never published).
    t_pub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Planned deadline per command row.
    t_deadline: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fps: float = 20.0
    control_dt: float = 0.002
    overlap_threshold: int | None = None
    blend_overlap: int = 0
    blend_mode: str = ""
    action_stride: int = 1
    #: Whether inference times are measured (inference.csv) or reconstructed.
    inference_source: str = "none"
    #: "queue" (crisp_gym's producer loop) or "splice" (splice_loop.py).
    mode: str = "queue"

    def _median(self, attr: str) -> float | None:
        v = [getattr(c, attr) for c in self.chunks if getattr(c, attr) is not None]
        return float(np.median(v)) if v else None

    @property
    def median_latency_first_ms(self) -> float | None:
        # Chunk 0 is inferred during bring-up, before anything executes; its latency
        # is the startup delay, not the loop's behaviour, so it is left out.
        v = [c.latency_first_ms for c in self.chunks[1:] if c.latency_first_ms is not None]
        return float(np.median(v)) if v else None

    @property
    def median_latency_policy_ms(self) -> float | None:
        v = [c.latency_policy_ms for c in self.chunks[1:] if c.latency_policy_ms is not None]
        return float(np.median(v)) if v else None

    @property
    def median_inference_ms(self) -> float | None:
        v = [c.inference_ms for c in self.chunks[1:] if c.inference_ms is not None]
        return float(np.median(v)) if v else None

    @property
    def median_frame_ms(self) -> float:
        d = np.diff(self.t_deadline)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) * 1000.0 if d.size else 1000.0 / self.fps

    @property
    def stride_mismatch(self) -> bool:
        """The Hermite tangents are in units that differ by the stride (module doc)."""
        return (self.mode == "queue" and self.blend_overlap > 0
                and self.blend_mode == "hermite" and self.action_stride > 1)


@dataclass
class RunFiles:
    cmd: dict
    chunks: dict | None = None
    frames: dict | None = None
    inference: dict | None = None
    trace: dict | None = None
    #: ``splices.csv`` from ``loop.mode: splice`` runs -- the recorded seam facts.
    splices: dict | None = None
    summary: dict = field(default_factory=dict)
    rc: dict = field(default_factory=dict)


def load_run(run_dir: Path) -> RunFiles:
    """Everything the timeline reads. Only ``commands.csv`` is required."""
    run_dir = Path(run_dir)
    cmd = read_csv(run_dir / "commands.csv")
    if cmd is None:
        raise FileNotFoundError(f"{run_dir} has no commands.csv")
    trace = None
    tp = run_dir / "trace.npz"
    if tp.exists():
        z = np.load(tp, allow_pickle=True)
        trace = {k: z[k] for k in ("chunk_idx", "wall_ns", "chunk") if k in z}
    sp = run_dir / "summary.json"
    rp = run_dir / "run_config.yaml"
    rc: dict = {}
    if rp.exists():
        import yaml
        rc = yaml.safe_load(rp.read_text()) or {}
    return RunFiles(
        cmd=cmd,
        chunks=read_csv(run_dir / "chunks.csv"),
        frames=read_csv(run_dir / "frames.csv"),
        inference=read_csv(run_dir / "inference.csv"),
        splices=read_csv(run_dir / "splices.csv"),
        trace=trace,
        summary=json.loads(sp.read_text()) if sp.exists() else {},
        rc=rc,
    )


def _inference_times(run: RunFiles, n_chunks: int):
    """``(t_obs, t_req, t_ret)`` per chunk, each an array with NaN for unknown."""
    obs = np.full(n_chunks, np.nan)
    req = np.full(n_chunks, np.nan)
    ret = np.full(n_chunks, np.nan)
    inf = run.inference
    if inf is not None and "t_req_wall" in inf:
        idx = inf.get("chunk", np.arange(len(inf["t_req_wall"]))).astype(int)
        for i, c in enumerate(idx):
            if 0 <= c < n_chunks:
                req[c] = inf["t_req_wall"][i]
                ret[c] = inf["t_ret_wall"][i]
        source = "measured"
    elif run.trace is not None and "wall_ns" in run.trace:
        wall = run.trace["wall_ns"].astype(float) / 1e9
        ids = (run.trace["chunk_idx"].astype(int) if "chunk_idx" in run.trace
               else np.arange(len(wall)))
        synth = run.chunks.get("synth_ms") if run.chunks else None
        for i, c in enumerate(ids):
            if 0 <= c < n_chunks:
                ret[c] = wall[i]
                if synth is not None and c < len(synth):
                    req[c] = wall[i] - synth[c] / 1000.0
        source = "reconstructed"
    else:
        return obs, req, ret, "none"
    get_obs = run.chunks.get("get_obs_ms") if run.chunks else None
    for c in range(n_chunks):
        if np.isfinite(req[c]):
            obs[c] = req[c] - ((get_obs[c] / 1000.0)
                               if get_obs is not None and c < len(get_obs)
                               and np.isfinite(get_obs[c]) else 0.0)
    return obs, req, ret, source


def reconstruct(run: RunFiles) -> Timeline:
    """The run on one clock. Never needs the robot; numpy on the run's files."""
    cmd = run.cmd
    summary = run.summary or {}
    control_dt = float(summary.get("control_dt_ms", 2.0)) / 1000.0
    bounds = chunk_bounds(cmd)
    modes = run.chunks.get("anchor_mode") if run.chunks else None
    t_dl = deadlines(cmd, control_dt, bounds, modes)
    t_pub = published(t_dl, run.frames)
    n = len(bounds)
    obs, req, ret, source = _inference_times(run, n)

    rc = run.rc or {}
    args = summary.get("args") or {}
    tl = Timeline(
        t_pub=t_pub, t_deadline=t_dl,
        fps=float(summary.get("fps_baseline", rc.get("fps", 20.0)) or 20.0),
        control_dt=control_dt,
        overlap_threshold=(int(args["overlap_threshold"]) if "overlap_threshold" in args
                           else (rc.get("loop") or {}).get("overlap_threshold")),
        blend_overlap=int((rc.get("blend") or {}).get("overlap", args.get("blend_overlap", 0)) or 0),
        blend_mode=str((rc.get("blend") or {}).get("mode", args.get("blend_mode", "")) or ""),
        action_stride=int((rc.get("method") or {}).get("action_stride", 1) or 1),
        inference_source=source,
    )

    q_logged = run.chunks.get("q_before_inf") if run.chunks else None
    preds = run.trace.get("chunk") if run.trace else None
    slack = run.frames.get("slack_ms") if run.frames else None

    # Under the splice loop the seam facts are recorded, not reconstructed.
    sp = run.splices
    if sp is not None and "n_bridge" in sp:
        tl.mode = "splice"
        bridge_by_chunk = {int(c): int(n) for c, n in zip(sp["chunk"], sp["n_bridge"])}
    else:
        tl.mode = str((rc.get("loop") or {}).get("mode", "queue") or "queue")
        bridge_by_chunk = None

    for ci, (lo, hi) in enumerate(bounds):
        # Executed rows that are the seam bridge rather than policy output. Needs the
        # predicted chunk for the adaptive/exempt stride paths; plain striding does not.
        n_bridge = 0
        if bridge_by_chunk is not None:
            n_bridge = min(bridge_by_chunk.get(ci, 0), hi - lo)
        elif ci > 0 and tl.blend_overlap > 0:
            if preds is not None and ci < len(preds):
                st = stages(np.asarray(preds[ci]), rc)
                n_bridge = bridge_rows(rc, len(preds[ci]), st["emitted"])
            else:
                n_raw = int(args.get("n_act") or rc.get("n_action_steps") or (hi - lo))
                n_bridge = bridge_rows(rc, max(n_raw, 2 * tl.blend_overlap),
                                       list(range(0, n_raw, tl.action_stride)))
            n_bridge = min(n_bridge, hi - lo)

        pub = t_pub[lo:hi]
        fin = np.isfinite(pub)
        first = float(pub[0]) if fin[0] else float(t_dl[lo])
        last = float(pub[fin][-1]) if fin.any() else float(t_dl[hi - 1])
        n_late = int((slack[lo:hi] < 0).sum()) if slack is not None and len(slack) >= hi else 0
        qm = None
        if ci > 0 and np.isfinite(req[ci]):
            plo, phi = bounds[ci - 1]
            qm = int(np.sum(t_pub[plo:phi] > req[ci]))
        fp_row = lo + n_bridge
        c = ChunkTiming(
            idx=ci, lo=lo, hi=hi,
            t_obs=None if np.isnan(obs[ci]) else float(obs[ci]),
            t_req=None if np.isnan(req[ci]) else float(req[ci]),
            t_ret=None if np.isnan(ret[ci]) else float(ret[ci]),
            t_push=float(cmd["t_wall"][lo]),
            t_first=first, t_last=last,
            n_rows=hi - lo, n_published=int(fin.sum()), n_late=n_late,
            n_bridge=n_bridge,
            q_logged=(int(q_logged[ci]) if q_logged is not None and ci < len(q_logged)
                      and np.isfinite(q_logged[ci]) else None),
            q_measured=qm,
            t_first_policy=(float(t_dl[fp_row]) if fp_row < hi else None),
        )
        tl.chunks.append(c)
    return tl


def describe(tl: Timeline) -> list[tuple[str, str]]:
    """What the timeline says, as (title, detail) findings for the report."""
    out: list[tuple[str, str]] = []
    if len(tl.chunks) < 2 or tl.inference_source == "none":
        out.append(("Chunk timing not available", (
            "The run has no trace.npz or inference.csv, so inference cannot be placed "
            "on the clock. Publish times are still reconstructed below.")))
        return out

    frame = tl.median_frame_ms
    lat = tl.median_latency_first_ms
    latp = tl.median_latency_policy_ms
    inf = tl.median_inference_ms
    qs = [c.q_logged for c in tl.chunks[1:] if c.q_logged is not None]
    q = int(np.median(qs)) if qs else tl.overlap_threshold
    if lat is not None and tl.mode == "splice":
        nb = [c.n_bridge for c in tl.chunks[1:]]
        out.append(("Chunks are spliced at the time-aligned row", (
            f"Median {lat:.0f} ms ({lat / frame:.1f} frames of {frame:.0f} ms) from the "
            f"observation to the first bridge row"
            + (f", {latp:.0f} ms ({latp / frame:.1f} frames) to the first verbatim row, "
               f"which is the one the policy predicted for that moment"
               if latp is not None else "")
            + f". Bridge {int(np.median(nb)) if nb else 0} rows; inference {inf:.0f} ms, "
            "run inside the dwell of the observed row.")))
    elif lat is not None:
        detail = (f"Median {lat:.0f} ms ({lat / frame:.1f} frames of {frame:.0f} ms) "
                  f"from the observation to this chunk's first published row. That is "
                  f"the {q} frames queued when inference was requested"
                  + (f" (overlap_threshold {tl.overlap_threshold})"
                     if tl.overlap_threshold is not None else "")
                  + ", plus the one the sender was already holding, plus one frame "
                  "because the new chunk is anchored one dt_eff after the last queued "
                  f"deadline. Inference itself took {inf:.0f} ms and is hidden entirely "
                  "inside that queue: a lower overlap_threshold is what shortens this "
                  "latency; faster inference does not.")
        nb = [c.n_bridge for c in tl.chunks[1:]]
        if latp is not None and nb and max(nb) > 0:
            detail += (f" The first {int(np.median(nb))} executed rows are the seam "
                       f"bridge, not policy output, so the first row the policy "
                       f"predicted verbatim runs at {latp:.0f} ms "
                       f"({latp / frame:.1f} frames).")
        out.append(("Every chunk executes later than it was predicted for", detail))

    if tl.stride_mismatch:
        out.append(("Hermite bridge tangents are in mismatched units", (
            f"blend.mode is hermite with action_stride {tl.action_stride}. The bridge "
            "is built on the raw chunk before the pipeline strides it (loop.py:282 "
            "runs ahead of :351): its start velocity is a difference of two *emitted* "
            f"rows, {tl.action_stride} raw frames apart, while its end velocity is a "
            "difference of two *raw* rows, one frame apart. The seam therefore leaves "
            f"the old chunk {tl.action_stride}x faster than it arrives at the new "
            f"one, and only every {tl.action_stride}th bridge sample executes "
            f"({tl.blend_overlap} raw rows -> "
            f"{-(-tl.blend_overlap // tl.action_stride)} executed).")))

    late = sum(c.n_late for c in tl.chunks)
    if late:
        out.append(("Late frames", (
            f"{late} published after their deadline. Their publish instants below "
            "are shifted by the recorded slack, so the timeline shows when they "
            "actually went out.")))
    return out
