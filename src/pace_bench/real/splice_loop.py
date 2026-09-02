"""One deadline-driven loop: replan on a fixed period, splice at the time-aligned row.

Replaces crisp_gym's ``run_producer_loop`` for ``loop.mode: splice``. What changes,
and why, in the terms ``timeline.py`` measures:

* **Where the new chunk starts.** The producer loop appends chunk B *behind* whatever
  of chunk A is still queued, so B's first row runs ``overlap_threshold + 2`` rows
  after the observation it was predicted from (8 rows, 270 ms, on the 2026-09-02
  runs), and its first verbatim row -- raw row 8 -- runs 12 rows late. Here the
  sender never holds more than one row ahead of the one executing, so a new chunk
  can be spliced in ``commit_rows + 1`` rows after the observation, and it enters at
  the row it predicted *for that moment*: output row ``i* = k_s + h_b - k_obs``.
  Replayed on the recorded predictions, the gap the seam has to bridge drops from
  49 mm to ~30 mm median and the velocity jump halves.

* **What the seam is anchored on.** The command stream at the splice -- where the
  plan is now -- not A's far tail (row 46, predicted 2.3 s of demo time out) and not
  the achieved pose. The policy's own output lives in command space: its row 0 lands
  2-17 mm from the last command and 40-240 mm from the arm, because teleop data
  taught it that the action leads the state by the servo lag. Splicing onto the arm
  would collapse that lead at every seam and brake the arm.

* **The bridge.** Cubic Hermite with both tangents as differences over one executed
  row. The producer loop's bridge differs a 2-raw-frame step against a 1-raw-frame
  step (``loop.py:292-300`` vs ``:297``), so with ``action_stride 2`` it leaves the
  seam at twice the executed velocity. A quintic (C²) variant is kept for
  comparison; on the replay it did not beat the cubic -- its acceleration
  constraints come from second differences of a noisy stream and it overshoots to
  honour them.

* **No hold-back, no carry.** Every output row past ``i*`` is executed verbatim
  until the next splice retracts it. ``blend.*`` and ``loop.overlap_threshold`` are
  unused in this mode.

Feeding the sender. crisp_gym's ``TargetSenderThread`` pops one item, sleeps to its
deadline, publishes, pops the next. This loop pushes row ``p+1`` the moment row ``p``
is published and nothing before that, so the sender holds exactly one row ahead of
the executing one -- that is ``commit_rows = 1``. Inference blocks the loop; the
next row is due one dwell after the observation, but the sender blocks in ``get()``
rather than skipping, and its own deadline for that row is two dwells out. So a
late frame needs inference plus the splice to exceed two dwells (68 ms at 1.5x,
100 ms at 1x) against a measured ~25 ms. ``frames.csv`` slack says if it happens.

The C++ sender is not used here: it cannot retract queued rows, and the one thing
it buys -- cadence under GIL contention -- is what a one-row queue no longer needs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Pose channels the bridge interpolates: xyz + rotation vector. The gripper (6) is
#: never interpolated; bridge rows take the new chunk's time-aligned value.
POSE = slice(0, 6)
GRIP = 6


@dataclass
class SpliceConfig:
    """The three numbers that define the seam, plus the polynomial."""

    #: H -- executed rows between observations.
    replan_every: int = 12
    #: h_c -- rows after the observation that are never retracted. 1 = the row the
    #: sender is holding. 2 leaves one more queued, which only delays the seam.
    commit_rows: int = 1
    #: h_b -- bridge length in executed rows. 3 is the knee for the median seam; 4
    #: also halves peak acceleration on the worst ones (where the policy changed
    #: its plan) at one extra row of latency.
    bridge_rows: int = 4
    #: "cubic" (C¹, tangents in per-row units) or "quintic" (C²).
    bridge: str = "cubic"


def cubic_bridge(p0, v0, p1, v1, n: int) -> np.ndarray:
    """``n`` interior samples of the cubic Hermite from (p0, v0) to (p1, v1).

    Tangents are per executed row and the bridge spans ``n + 1`` rows, so both are
    scaled by ``T = n + 1`` -- the standard form, and the same convention on both
    ends, which is the fix over ``loop.py``'s bridge.
    """
    T = n + 1
    s = (np.arange(n) + 1) / T
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return (h00[:, None] * p0 + (h10 * T)[:, None] * v0
            + h01[:, None] * p1 + (h11 * T)[:, None] * v1)


def quintic_bridge(p0, v0, a0, p1, v1, a1, n: int) -> np.ndarray:
    """``n`` interior samples of the quintic Hermite (position, velocity, acceleration
    matched at both ends). Kept for comparison; see the module docstring."""
    T = n + 1
    s = (np.arange(n) + 1) / T
    H0 = 1 - 10 * s**3 + 15 * s**4 - 6 * s**5
    H1 = s - 6 * s**3 + 8 * s**4 - 3 * s**5
    H2 = 0.5 * s**2 - 1.5 * s**3 + 1.5 * s**4 - 0.5 * s**5
    H3 = 10 * s**3 - 15 * s**4 + 6 * s**5
    H4 = -4 * s**3 + 7 * s**4 - 3 * s**5
    H5 = 0.5 * s**3 - s**4 + 0.5 * s**5
    return (H0[:, None] * p0 + (H1 * T)[:, None] * v0 + (H2 * T * T)[:, None] * a0
            + H3[:, None] * p1 + (H4 * T)[:, None] * v1 + (H5 * T * T)[:, None] * a1)


@dataclass
class Splice:
    """What one replan did to the plan. Written to ``splices.csv``."""

    chunk: int
    k_obs: int
    k_s: int
    i_star: int
    n_bridge: int
    gap_mm: float
    rows_retracted: int
    rows_added: int
    inference_ms: float = 0.0
    get_obs_ms: float = 0.0
    build_ms: float = 0.0
    late: bool = False


@dataclass
class Plan:
    """The command stream: rows already sent and rows still to send.

    Row ``k`` is the k-th executed row of the run. ``actions`` are absolute
    ``[x, y, z, r0, r1, r2, grip]`` as the pipeline emits them; ``speeds`` the
    per-row multiplier the cycle-snap turns into a dwell.
    """

    actions: list[np.ndarray] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def splice(self, k_s: int, new_actions: np.ndarray, new_speeds: np.ndarray, *,
               k_obs: int, cfg: SpliceConfig, chunk: int = 0) -> Splice:
        """Replace everything from row ``k_s`` on with a bridge into the new chunk.

        Output row ``i`` of the pipeline is the policy's prediction for executed row
        ``k_obs + i`` -- one dwell per row, whatever the stride did to the raw
        indices. So the verbatim part starts at ``i* = k_s + h_b - k_obs``, and the
        bridge fills rows ``k_s .. k_s + h_b - 1`` from the last two sent rows to
        ``new[i*]``. Rows ``new[:i*]`` are discarded: they were predicted for rows
        already executed, committed, or covered by the bridge.

        With no history to bridge from (the first chunk, or a plan shorter than two
        rows) the new chunk is placed time-aligned with no bridge at all.
        """
        new_actions = np.asarray(new_actions, dtype=np.float64)
        new_speeds = np.asarray(new_speeds, dtype=np.float64).reshape(-1)
        h_b = int(cfg.bridge_rows)
        i_s = k_s - k_obs                       # first output row not yet spoken for
        if i_s < 0:
            raise ValueError(f"splice row {k_s} precedes the observation row {k_obs}")
        retracted = max(len(self.actions) - k_s, 0)
        del self.actions[k_s:], self.speeds[k_s:]
        if len(self.actions) != k_s:
            raise ValueError(f"plan has {len(self.actions)} rows; cannot splice at {k_s}")

        can_bridge = h_b > 0 and k_s >= 2 and (i_s + h_b + 1) < len(new_actions)
        if not can_bridge:
            i_star, bridge = i_s, np.zeros((0, new_actions.shape[1]))
            gap = 0.0
        else:
            i_star = i_s + h_b
            p0 = self.actions[k_s - 1][POSE]
            v0 = p0 - self.actions[k_s - 2][POSE]
            p1 = new_actions[i_star][POSE]
            v1 = new_actions[i_star + 1][POSE] - p1
            if cfg.bridge == "quintic" and k_s >= 3 and i_star >= 1:
                a0 = p0 - 2 * self.actions[k_s - 2][POSE] + self.actions[k_s - 3][POSE]
                a1 = new_actions[i_star + 1][POSE] - 2 * p1 + new_actions[i_star - 1][POSE]
                pose = quintic_bridge(p0, v0, a0, p1, v1, a1, h_b)
            else:
                pose = cubic_bridge(p0, v0, p1, v1, h_b)
            bridge = np.empty((h_b, new_actions.shape[1]))
            bridge[:, POSE] = pose
            # Everything past the pose (gripper, and any extra channels) is taken
            # from the time-aligned output rows, never interpolated.
            bridge[:, GRIP:] = new_actions[i_s:i_star, GRIP:]
            gap = float(np.linalg.norm(p1[:3] - p0[:3]) * 1000.0)

        added = list(bridge) + list(new_actions[i_star:])
        self.actions.extend(np.asarray(r, dtype=np.float64) for r in added)
        self.speeds.extend(float(s) for s in new_speeds[i_s:i_s + len(bridge)])
        self.speeds.extend(float(s) for s in new_speeds[i_star:])
        assert len(self.speeds) == len(self.actions)
        return Splice(chunk=chunk, k_obs=k_obs, k_s=k_s, i_star=int(i_star),
                      n_bridge=len(bridge), gap_mm=gap, rows_retracted=retracted,
                      rows_added=len(added))


def rows_to_push(k_next: int, executing: int, commit_rows: int = 1) -> range:
    """Which plan rows to hand the sender now, given the row it is publishing.

    The sender holds one popped row; with ``commit_rows = 1`` the queue is otherwise
    kept empty, so row ``p + 1`` goes out when ``p`` is published and nothing
    further. Each extra commit row leaves one more queued. At the start
    ``executing`` is -1 and rows ``0 .. commit_rows`` go out together so the sender
    can pop 0 and have the rest waiting.
    """
    h_c = max(int(commit_rows), 1)
    end = h_c + 1 if executing < 0 else executing + 1 + h_c
    return range(k_next, end)


def needs_replan(executing: int, k_obs_last: int, k_next: int, n_plan: int,
                 cfg: SpliceConfig) -> bool:
    """Replan on the period -- or earlier if the plan is about to run out."""
    if n_plan == 0:
        return True
    if executing - k_obs_last >= cfg.replan_every:
        return True
    return (n_plan - k_next) < cfg.bridge_rows + 3


def run_splice_loop(
    *, env, chunk_source, q, sender, args, rec, dt_base: float, obs_schema,
    gripper_enabled: bool, gripper_unnormalize_fn, obs_buf, last_obs, steps,
    cfg: SpliceConfig,
) -> list[Splice]:
    """Run until ``--max-chunks``, the source is exhausted, or Ctrl-C.

    Telemetry goes on the caller's ``RunRecord`` with the same keys the producer loop
    writes, so ``summary.json``/``chunks.csv``/``trace.npz`` keep their shape; the
    per-splice facts are returned for ``splices.csv``.
    """
    from crisp_gym.deploy.obs import _get_obs_zerofill
    from crisp_gym.deploy.pipeline import Chunk, run_pipeline
    from crisp_gym.deploy.sender import TargetItem
    from crisp_gym.deploy.sources import DatasetExhausted
    from crisp_gym.deploy.timing import _pre_compute_chunk_arrays, build_speed_queue_arrays

    plan = Plan()
    splices: list[Splice] = []
    k_next = 0
    k_obs_last = -10**9
    deadline: float | None = None
    p_prev = -1
    chunk_count = rec.chunk_count
    stopped_by = rec.stopped_by
    starvation = rec.starvation_event_count
    stages = rec.stage_samples_producer

    def executing() -> int:
        # The sender counts publishes; row p is executing once it has published p+1 rows.
        return int(getattr(sender, "n_published", 0)) - 1

    def push(k: int) -> None:
        nonlocal deadline
        row = plan.actions[k][None, :]
        cycles, dt_eff, s_eff = build_speed_queue_arrays(
            np.array([plan.speeds[k]]), dt_base, 1, retime=True)
        xyz, quat, grip, act32 = _pre_compute_chunk_arrays(
            row, args=args, gripper_enabled=gripper_enabled,
            gripper_unnormalize_fn=gripper_unnormalize_fn,
            rotation_from_action=env.action_to_rotation)
        now = time.monotonic()
        # Chain deadlines; re-anchor only after a real stall (a full dwell behind).
        if deadline is None or deadline < now - float(dt_eff[0]):
            deadline = now
        deadline += float(dt_eff[0])
        q.put(TargetItem(
            pose_xyz=xyz[0], pose_quat=quat[0],
            grip_raw=float(grip[0]) if gripper_enabled else None,
            action=act32[0], deadline_mono=deadline, frame_idx=k,
            s_eff=float(s_eff[0]), cycles=int(cycles[0])))

    logger.info("Phase 4: splice loop -- H=%d rows, h_c=%d, h_b=%d, %s bridge. Ctrl-C to stop.",
                cfg.replan_every, cfg.commit_rows, cfg.bridge_rows, cfg.bridge)
    try:
        while True:
            if args.max_chunks > 0 and chunk_count >= args.max_chunks:
                stopped_by = "normal"
                break
            p = executing()
            for k in rows_to_push(k_next, p, cfg.commit_rows):
                if k < len(plan):
                    push(k)
                    k_next = k + 1

            # Observe on a publish edge, so inference has the whole dwell in front of
            # it; the first chunk has no edge to wait for.
            edge = p != p_prev
            p_prev = p
            if not needs_replan(p, k_obs_last, k_next, len(plan), cfg) or not (edge or len(plan) == 0):
                time.sleep(0.001)
                continue

            k_obs = max(p, 0)
            _t = time.perf_counter()
            obs_buf.append(_get_obs_zerofill(env, obs_schema, last_obs))
            get_obs_ms = (time.perf_counter() - _t) * 1000.0
            stages["get_obs_ms"].append(get_obs_ms)

            t_send = time.monotonic()
            try:
                chunk = chunk_source.request(obs_buf)
            except DatasetExhausted as e:
                logger.info("dataset exhausted (%s); exiting", e)
                stopped_by = "dataset_exhausted"
                break
            except (BrokenPipeError, EOFError):
                logger.error("chunk source pipe closed; exiting")
                stopped_by = "chunk_source_pipe_closed"
                break
            inf_ms = (time.monotonic() - t_send) * 1000.0
            rec.pred_dt_samples.append(inf_ms / 1000.0)
            stages["synth_ms"].append(inf_ms)
            chunk_count += 1

            if args.record_trace and (chunk_count - 1) % max(1, args.record_trace_every) == 0:
                obs_now = obs_buf[-1]
                record = {"chunk_idx": chunk_count - 1, "wall_ns": int(time.time_ns()),
                          "mono_ns": int(time.monotonic_ns()),
                          "chunk": np.asarray(chunk, dtype=np.float32)}
                for key, v in obs_now.items():
                    if key.startswith("observation.state."):
                        record[key] = np.asarray(v, dtype=np.float32).reshape(-1)
                if obs_now.get("task", ""):
                    record["task"] = str(obs_now["task"])
                rec.trace_records.append(record)

            if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[0] == 0:
                logger.warning("chunk %d: unexpected payload %r; skipped", chunk_count,
                               getattr(chunk, "shape", type(chunk).__name__))
                continue

            _t = time.perf_counter()
            out = run_pipeline(Chunk.nominal(chunk.astype(np.float64)), steps)
            # The splice row is the first row not yet handed to the sender. The feed
            # rule makes that k_obs + commit_rows + 1 on every chunk but the first
            # (which has no plan to splice into), and nothing is pushed while
            # inference blocks, so it holds even when inference runs long -- the
            # sender is then waiting in get() for exactly this row.
            k_s = k_next
            info = plan.splice(k_s, out.actions, out.speeds, k_obs=k_obs, cfg=cfg,
                               chunk=chunk_count - 1)
            build_ms = (time.perf_counter() - _t) * 1000.0
            stages["build_ms"].append(build_ms)
            stages["push_ms"].append(0.0)
            stages["drain_wait_ms"].append(0.0)
            info.inference_ms, info.get_obs_ms, info.build_ms = inf_ms, get_obs_ms, build_ms
            # Late: the sender has already published k_obs+1 and is waiting for k_s.
            info.late = executing() >= k_s - 1 and k_s > 0
            if info.late:
                starvation += 1
            splices.append(info)
            k_obs_last = k_obs

            rec.chunk_rows.append({
                "chunk_idx": chunk_count, "q_before_inf": int(q.qsize()),
                "q_before_push": int(q.qsize()), "anchor_mode": "splice",
                "K": info.rows_added, "dt_eff_mean_ms": 0.0,
                "get_obs_ms": get_obs_ms, "synth_ms": inf_ms, "build_ms": build_ms,
                "push_ms": 0.0, "drain_wait_ms": 0.0,
            })
            logger.info(
                "chunk %d: inf=%.1fms obs@row %d splice@row %d -> out[%d:], bridge %d rows "
                "over %.0f mm, retracted %d, plan now %d rows%s",
                chunk_count, inf_ms, k_obs, k_s, info.i_star, info.n_bridge,
                info.gap_mm, info.rows_retracted, len(plan), "  LATE" if info.late else "")
    finally:
        rec.chunk_count = chunk_count
        rec.stopped_by = stopped_by
        rec.starvation_event_count = starvation
    return splices
