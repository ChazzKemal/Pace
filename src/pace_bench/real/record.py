"""What a deploy run actually did, recorded frame by frame.

``crisp_gym``'s own artifacts answer "was inference keeping up?" very well --
``summary.json`` carries five stages of producer timing and per-frame sender slack.
They cannot answer "was the *arm* keeping up?", because nothing in the deploy path
records where the arm actually was. ``trace.npz`` samples the achieved state once per
chunk, which is one sample every K executed steps (13 for ``bspline_2x``, 46 for
``pace_fast`` at the current ``n_action_steps``), and the tracking error that limits
how fast the arm can be pushed lives entirely in between those samples.

This module fills that in at control rate, and it does so without a ``crisp_gym``
patch -- but *not* through ``state_capture_fn``, which is the obvious-looking hook and
the wrong one. Under the Python sender that callback does fire once per published
frame (``sender.py:363``). Under the C++ sender -- which is this rig's default, and
what every recorded run used -- it fires inside ``CppSenderHandle.put``
(``cpp_sender.py:465``), and ``put`` is called by the *producer*, K times back to back
in the push loop immediately after inference (``loop.py:452-463``). Sampling the arm
there yields K readings microseconds apart, once per chunk: chunk-rate data wearing
control-rate clothing, which is worse than none because it looks dense.

So the achieved pose is sampled on its own fixed-rate thread instead, decoupled from
the command path entirely. That is sender-agnostic, gives a uniform time grid to
cross-correlate against, and adds nothing at all to the sender's hot loop. The
commanded side is written separately from the sender's own ``replay_log`` -- which it
has been accumulating for every run and which crisp_gym writes nowhere -- and the two
are joined offline on their timestamps.

Two deliberate non-features, both because they would bake an error into the raw data:

* **No error columns.** Commanded and achieved go to separate files and nothing is
  subtracted. A published target is not yet a response -- the arm moves over
  the following control cycles -- so a naive difference is dominated by pure transport
  latency and reads as "the arm cannot keep up" when it can. Cross-correlate offline to
  recover the lag first; the lag is the more useful number anyway, because it is the
  phase margin that says how much faster this rig can be driven.
* **No rotation-vector unflipping.** ``manipulator_env.py:383`` runs
  ``_flip_rotation_vector_if_needed`` after ``to_array`` because a rotation vector flips
  sign near pi. Reproducing that here would put branchy state in the sender's hot loop
  for something an offline pass does better. Raw is recorded; unflip before differencing
  or the flips read as enormous phantom excursions.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Axis suffixes for a 6-DoF cartesian pose. ``r*`` rather than roll/pitch/yaw
#: because the deploy env is ``..._rotvec_*``: these are rotation-vector components,
#: not Euler angles, and crisp_gym's own ``act_roll/pitch/yaw`` labels in
#: ``sender.py:326`` are misnamed for this representation.
POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def read_cartesian_gains(env, controller_node: str, *, scaler=None,
                        timeout: float = 10.0) -> dict[str, float] | None:
    """Read the controller's cartesian stiffness/damping once, at bring-up.

    This is what makes an *effort* estimate possible on a rig whose joint efforts are
    unreadable: ``crisp_py``'s ``_callback_current_joint`` (``robot/robot.py:634``)
    keeps position and velocity and drops ``msg.effort``, and exposes no
    ``joint_efforts``, so no torque reaches this process at any rate. But the
    controller is an impedance law, so commanded wrench is ``kp * (target - achieved)``
    per axis -- and with ``gains.scale_kp: false`` (this rig's default) kp is constant
    for the whole run, which makes a single reading here sufficient to turn the
    tracking error recorded below into a force in Newtons.

    With ``scale_kp: true`` this is only the *baseline*: ``ReplayScaler.step_to``
    pushes ``kp_base * s_eff**kp_exp`` fire-and-forget, never waiting for the
    controller to acknowledge (``gains.py:496-506``), so the kp in force at any instant
    is not knowable from this process. Treat the estimate as indicative there, and lean
    on the recorded ``s_eff`` to know when gains were in flight.

    ``timeout`` matches ``gains.py``'s own reasoning for being generous: this fires
    just after the controller is activated, and FastDDS discovery over WiFi can take
    seconds to publish its parameter endpoints. It is paid only when the controller
    does not answer, and always before anything moves.

    Never raises: a run must not fail because a diagnostic could not be read.
    """
    # With --scale-kp the scaler has already made this exact round-trip in apply()
    # and cached the result; a second one would be pure latency during bring-up.
    cached = dict(getattr(scaler, "_original_kp", None) or {})
    cached.update(getattr(scaler, "_original_kd", None) or {})
    if cached:
        logger.info("gains: reusing the %d value(s) the scaler already read",
                    len(cached))
        return cached

    try:
        from crisp_gym.deploy.gains import (
            KD_TASK_KEYS,
            KP_TASK_KEYS,
            _get_params_batch,
        )
        from rclpy.node import Node as RclpyNode
    except Exception:
        logger.exception("gains readback: imports unavailable; skipping")
        return None

    helper = None
    try:
        # A dedicated node, not attached to any executor -- the same reason
        # ReplayScaler keeps its own (gains.py:322-325): crisp_py is already
        # spinning env.robot.node on a background thread.
        helper = RclpyNode("pace_gain_reader", namespace="")
        names = list(KP_TASK_KEYS) + list(KD_TASK_KEYS)
        # Timed and logged: on the happy path this returns as soon as the service is
        # reachable, but when the controller is not there it sits out the full
        # timeout during bring-up. An operator watching the log should be able to see
        # that the stall is a diagnostic read, not the arm refusing to home.
        _t = time.monotonic()
        values = _get_params_batch(helper, controller_node, names, timeout)
        logger.info("gains readback from %s took %.1f s", controller_node,
                    time.monotonic() - _t)
        if values is None:
            logger.warning("gains readback: %s did not answer; skipping",
                           controller_node)
            return None
        return {n: (float(v) if v is not None else None)
                for n, v in zip(names, values)}
    except Exception:
        logger.exception("gains readback failed; continuing without it")
        return None
    finally:
        if helper is not None:
            try:
                helper.destroy_node()
            except Exception:
                logger.exception("gains readback: helper node teardown")


#: The controller YAML's rotation-to-translation stiffness ratio: k_rot 100 over
#: k_pos 400 in tum09_bringup/config/crisp_controllers.yaml. A kp selection that
#: names only the translation keeps this shape, so the arm is tuned the way the rig
#: already is, just stiffer.
KP_ROT_RATIO = 0.25
KP_POS_KEYS = ("task.k_pos_x", "task.k_pos_y", "task.k_pos_z")
KP_ROT_KEYS = ("task.k_rot_x", "task.k_rot_y", "task.k_rot_z")


def kp_targets(kp_pos: float | None, kp_rot: float | None = None,
               ) -> list[tuple[str, float]]:
    """The six stiffness parameters a deploy-time kp selection writes.

    Damping is deliberately absent. ``d_*`` is auto on this rig (``-1`` in the YAML,
    read back as ``0.0``), the controller derives ``2*sqrt(k)`` per axis from
    whatever kp is in force, and an explicit kd was tried on 2026-09-02 and
    reverted. Writing kp alone keeps the controller critically damped at the new
    stiffness without this process having to know its inertia model.
    """
    if kp_pos is None:
        return []
    if kp_pos <= 0:
        raise ValueError(f"gains.kp_pos must be positive, got {kp_pos}")
    rot = float(kp_pos) * KP_ROT_RATIO if kp_rot is None else float(kp_rot)
    if rot <= 0:
        raise ValueError(f"gains.kp_rot must be positive, got {rot}")
    return ([(k, float(kp_pos)) for k in KP_POS_KEYS]
            + [(k, rot) for k in KP_ROT_KEYS])


def set_cartesian_kp(controller_node: str, kp_pos: float | None,
                     kp_rot: float | None = None, *, timeout: float = 10.0,
                     ) -> dict[str, float] | None:
    """Write an absolute stiffness to the live controller. Returns what was there.

    The same ``SetParameters`` round-trip ``tum09_custom/reset_crisp_kp.py`` makes,
    with the values coming from the run config instead of the bringup YAML. It runs
    *before* ``phase_scaler``, so ``ReplayScaler`` reads these as its base and
    ``scale_kp`` scales from the requested stiffness rather than from whatever the
    previous run left behind -- which, before the Ctrl-C fix, was the previous run's
    peak: the 13:10 run on 2026-09-02 started from 713 N/m believing it was 400.

    Raises when the write does not land. A run asked to test 500 N/m must not proceed
    at 400 and report otherwise; that is the one bring-up diagnostic worth failing on.
    Returns None without touching anything when ``kp_pos`` is unset.
    """
    targets = kp_targets(kp_pos, kp_rot)
    if not targets:
        return None
    from crisp_gym.deploy.gains import _get_params_batch, _set_params_batch
    from rclpy.node import Node as RclpyNode

    helper = RclpyNode("pace_kp_writer", namespace="")
    try:
        names = [n for n, _ in targets]
        before = _get_params_batch(helper, controller_node, names, timeout)
        if before is None:
            raise RuntimeError(
                f"{controller_node} did not answer get_parameters within {timeout:.0f}s; "
                "cannot set kp for this run")
        failures = _set_params_batch(helper, controller_node, targets, timeout)
        if failures:
            raise RuntimeError(f"{controller_node} refused the kp write: {failures}")
        was = {n: float(v) for n, v in zip(names, before) if v is not None}
        logger.info("kp set for this run: pos %.0f -> %.0f N/m, rot %.0f -> %.0f Nm/rad",
                    was.get(KP_POS_KEYS[0], float("nan")), targets[0][1],
                    was.get(KP_ROT_KEYS[0], float("nan")), targets[3][1])
        return was
    finally:
        try:
            helper.destroy_node()
        except Exception:
            logger.exception("kp writer: helper node teardown")


def restore_cartesian_kp(controller_node: str, before: dict[str, float] | None,
                         *, timeout: float = 10.0) -> None:
    """Put the pre-run stiffness back. Teardown: never raises, but is loud.

    Runs after ``ReplayScaler.restore`` (which returns the controller to *our* base),
    so the next run starts from the YAML values, not from this run's selection.
    """
    if not before:
        return
    try:
        import rclpy
        from crisp_gym.deploy.gains import _set_params_batch
        from rclpy.node import Node as RclpyNode

        if not rclpy.ok():
            logger.error("rclpy context is down; kp CANNOT be restored from here. "
                         "Run `ros2 run tum09_custom reset_crisp_kp.py` before the next run.")
            return
        helper = RclpyNode("pace_kp_restorer", namespace="")
        try:
            failures = _set_params_batch(helper, controller_node, list(before.items()),
                                         timeout)
        finally:
            helper.destroy_node()
        if failures:
            logger.error("kp restore refused: %s -- run `ros2 run tum09_custom "
                         "reset_crisp_kp.py`", failures)
        else:
            logger.info("kp restored to pre-run values: pos %.0f, rot %.0f",
                        before.get(KP_POS_KEYS[0], float("nan")),
                        before.get(KP_ROT_KEYS[0], float("nan")))
    except Exception:
        logger.exception("kp restore failed; run `ros2 run tum09_custom reset_crisp_kp.py`")


class PoseSampler:
    """Where the arm actually is, on a uniform clock of its own.

    A daemon thread reading ``end_effector_pose`` at a fixed rate. Deliberately not
    driven by the sender: see the module docstring for why ``state_capture_fn`` is the
    wrong hook under the C++ sender. Being independent also means the ground truth
    keeps its cadence when the command path stutters -- which is precisely the case
    the recording exists to catch, and precisely when a command-triggered sampler
    would go blind.

    Cheap enough to ignore: one attribute read and a list append per tick, ~50 times a
    second, against a producer loop that spends 600-800 ms per chunk parked in
    ``drain_wait``.
    """

    def __init__(self, env, rate_hz: float = 50.0) -> None:
        self._robot = env.robot
        self._repr = env.config.orientation_representation
        self.rate_hz = max(1.0, float(rate_hz))
        self.rows: list[tuple[float, float, Any]] = []
        self.n_errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="pace-pose-sampler", daemon=True,
        )
        self._thread.start()
        logger.info("pose sampler started at %.1f Hz", self.rate_hz)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        logger.info("pose sampler stopped: %d sample(s), %d error(s)",
                    len(self.rows), self.n_errors)

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            next_t += period
            try:
                self.rows.append((
                    time.monotonic(),
                    time.time(),
                    self._robot.end_effector_pose.to_array(representation=self._repr),
                ))
            except Exception:  # noqa: BLE001 - counted, not logged; see below
                # Silent by design: a robot whose pose topic has gone quiet would
                # otherwise emit a log line 50 times a second. Counted and reported
                # once at teardown, as `obs._ZEROFILL_COUNTS` does for sensors.
                self.n_errors += 1
            slack = next_t - time.monotonic()
            if slack > 0:
                self._stop.wait(slack)
            else:
                # Fell behind (GIL contention, a slow callback). Resync to now rather
                # than spinning to catch up, which would burst samples and distort the
                # very cadence this is meant to measure.
                next_t = time.monotonic()


class ChunkClock:
    """When inference ran and when each target was due -- measured, not reconstructed.

    ``timeline.py`` can put a run on the clock from the files crisp_gym already
    writes, but two of its inputs are inferred: the inference *start* is the trace
    stamp minus ``synth_ms`` (so it needs ``record_trace`` on, and it is missing for a
    chunk whose ``chunks.csv`` row was never written), and a row's deadline is a
    cumulative sum that assumes the anchoring rule in ``loop.py:436-451`` and a chunk
    boundary detected from a gap in push times. This records both directly.

    Two wrappers, neither a crisp_gym patch: ``request`` on the chunk source, timed
    around the call, and ``put`` on the sender handle, which is the one place the
    ``TargetItem`` -- carrying ``deadline_mono`` -- passes through this process. Each
    queued row is tagged with the chunk whose request produced it, so chunk membership
    is a recorded fact rather than a heuristic, including for a chunk the loop skipped
    (``loop.py:234-242``) which pushes nothing and would otherwise be invisible.

    Cost: three clock reads per chunk and two per pushed row, on the producer thread,
    which spends most of every chunk parked in ``drain_wait``.
    """

    def __init__(self, q) -> None:
        self._q = q
        self._chunk = -1
        self.inferences: list[dict] = []
        self.pushes: list[dict] = []

    def wrap_source(self, src):
        """Time ``src.request``. Returns ``src`` for chaining."""
        orig = src.request

        def request(obs_buf):
            t_req_wall, t_req_mono = time.time(), time.monotonic()
            q_at_req = int(self._q.qsize())
            chunk = orig(obs_buf)
            # `else`, not `finally`: a request that raised (DatasetExhausted, a dead
            # pipe) produced no chunk, and the loop does not count it either.
            self._chunk += 1
            self.inferences.append({
                "chunk": self._chunk,
                "t_req_wall": t_req_wall,
                "t_ret_wall": time.time(),
                "inference_ms": (time.monotonic() - t_req_mono) * 1000.0,
                "q_at_req": q_at_req,
            })
            return chunk

        src.request = request
        return src

    def tap_queue(self, q):
        """Record each ``TargetItem``'s deadline as it is handed to the sender."""
        orig = q.put

        def put(item):
            if item is not None:
                # Both clocks read together so the monotonic deadline lands on the
                # same wall axis as everything else the run writes.
                wall, mono = time.time(), time.monotonic()
                self.pushes.append({
                    "frame_index": int(item.frame_idx),
                    "chunk": self._chunk,
                    "deadline_wall": float(item.deadline_mono) + (wall - mono),
                    "t_put_wall": wall,
                })
            return orig(item)

        q.put = put
        return q


def write_inference(clock: ChunkClock | None, out_dir: Path) -> Path | None:
    """``inference.csv``: one row per chunk request -- when it began and returned."""
    if clock is None:
        return None
    rows = clock.inferences
    fields = list(rows[0].keys()) if rows else []
    return _write_csv(Path(out_dir) / "inference.csv", fields, rows, "inference")


def write_splices(splices: list, out_dir: Path) -> Path | None:
    """``splices.csv``: one row per replan under ``loop.mode: splice`` -- the
    observation row, the splice row, the output row the chunk entered at, and the
    bridge. This is the ground truth ``timeline.py`` uses for those runs."""
    if not splices:
        return None
    rows = [dict(vars(s)) for s in splices]
    return _write_csv(Path(out_dir) / "splices.csv", list(rows[0].keys()), rows, "splice")


def command_table(replay_log: list[dict], clock: ChunkClock | None = None,
                  ) -> tuple[list[str], list[dict]]:
    """Flatten the sender's replay rows into CSV columns.

    With a :class:`ChunkClock`, each row also carries ``chunk`` and ``deadline_wall``.
    The two logs are joined by position: the clock's ``put`` wrapper and the sender's
    ``_replay_log.append`` run in the same call on the same thread, so the i-th entry
    of each is the same item. That is checked against ``frame_index`` row by row, and
    on any disagreement the columns are left out rather than written wrong.

    Kept a free function over plain dicts, with no robot or ROS in sight, so the column
    contract is testable on a laptop -- the same bargain ``deploy_flags`` makes. Rows
    are built by the sender (``sender.py:355-360``, ``cpp_sender.py:457-463``) and
    carry ``frame_index``, ``timestamp``, ``replay.s_eff``, ``replay.cycles`` and
    ``replay.action``.

    ``t_wall`` is when the frame was *queued*, not when it was published: under the C++
    sender the row is built in ``put``. Publish-side timing lives in crisp_gym's own
    ``frames.csv``, joinable on ``frame_index`` -- with the caveat that ``frame_idx`` is
    computed as ``(chunk_count - 1) * K + i`` (``loop.py:459``) using the *current*
    chunk's K, so it skips or collides whenever K varies between chunks. It is constant
    only while nothing strides or replicates rows.

    Array columns are named per axis rather than indexed, because an analysis that has
    to remember which of ``cmd_3..cmd_5`` is the rotation is one nobody runs twice.
    """
    if not replay_log:
        return [], []

    def _cmd_name(i: int, n: int) -> str:
        # cart7 for this rig: xyz + rotvec + gripper. Anything else falls back to an
        # index rather than guessing a layout that is not there.
        if n == 7 and i == 6:
            return "cmd_grip"
        if n in (6, 7) and i < len(POSE_AXES):
            return f"cmd_{POSE_AXES[i]}"
        return f"cmd_{i}"

    rows: list[dict] = []
    for r in replay_log:
        out: dict[str, Any] = {
            "frame_index": r.get("frame_index"),
            "t_wall": r.get("timestamp"),
            "s_eff": r.get("replay.s_eff"),
            "cycles": r.get("replay.cycles"),
        }
        act = r.get("replay.action")
        if act is not None:
            n = len(act)
            for i in range(n):
                out[_cmd_name(i, n)] = float(act[i])
        rows.append(out)

    if clock is not None and clock.pushes:
        pushes = clock.pushes
        aligned = len(pushes) == len(rows) and all(
            p["frame_index"] == r["frame_index"] for p, r in zip(pushes, rows))
        if aligned:
            for p, r in zip(pushes, rows):
                r["chunk"] = p["chunk"]
                r["deadline_wall"] = p["deadline_wall"]
        else:
            logger.warning(
                "ChunkClock saw %d push(es) but the replay log has %d row(s), or their "
                "frame indices disagree; chunk/deadline columns not written",
                len(pushes), len(rows))

    # Union of keys in first-seen order: DictWriter needs the full set up front, and a
    # row whose action was absent defines fewer than its neighbours.
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    return fields, rows


def pose_table(samples: list[tuple]) -> tuple[list[str], list[dict]]:
    """Flatten :class:`PoseSampler` rows. Pure, for the same reason as above."""
    rows: list[dict] = []
    for t_mono, t_wall, pose in samples:
        out: dict[str, Any] = {"t_mono": t_mono, "t_wall": t_wall}
        for i in range(len(pose)):
            axis = POSE_AXES[i] if i < len(POSE_AXES) else str(i)
            out[f"ach_{axis}"] = float(pose[i])
        rows.append(out)
    fields = list(rows[0].keys()) if rows else []
    return fields, rows


def _write_csv(path: Path, fields: list[str], rows: list[dict], what: str) -> Path | None:
    """Write one table. Never raises -- a run that reached the arm is not a failure
    because its bookkeeping could not be written."""
    if not rows:
        logger.warning("no %s rows captured; %s not written", what, path.name)
        return None
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info("%s written to %s (N=%d)", path.name, path, len(rows))
        return path
    except Exception:
        logger.exception("failed to write %s", path.name)
        return None


def write_commands(replay_log: list[dict], out_dir: Path,
                   clock: ChunkClock | None = None) -> Path | None:
    """``commands.csv``: one row per queued target -- the commanded side."""
    fields, rows = command_table(replay_log, clock)
    return _write_csv(Path(out_dir) / "commands.csv", fields, rows, "command")


def write_poses(samples: list[tuple], out_dir: Path) -> Path | None:
    """``poses.csv``: the achieved trajectory on a uniform grid -- the ground truth.

    Join to ``commands.csv`` offline on the clock, after recovering the transport lag
    by cross-correlation. Do not index-align the two: they are on different cadences
    on purpose.
    """
    fields, rows = pose_table(samples)
    return _write_csv(Path(out_dir) / "poses.csv", fields, rows, "pose")


def _git(path: Path | str, *args: str) -> str | None:
    """Run one read-only git command in ``path``. None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is nice to have, never required
        return None


def _git_sha(path: Path | str) -> str | None:
    """Short SHA of the repo containing ``path``, or None. Never raises."""
    return _git(path, "rev-parse", "--short", "HEAD")


def _dependency_provenance(module_path: Path, own_root: Path) -> dict:
    """Identify a dependency that may or may not be a checkout of its own.

    ``git -C`` answers for whichever repository *contains* the path, which is not
    always the repository you meant. In the robot environment ``crisp_gym`` imports
    from ``real/.pixi/envs/jazzy-lerobot/.../site-packages/crisp_gym`` -- a copy
    living inside the Pace working tree -- so asking git about it returns *Pace's*
    SHA. The first manifest this wrote recorded pace and crisp_gym at the same
    commit, which is not a plausible fact and would have quietly misattributed every
    run to the wrong dependency version.

    So the toplevel is compared against our own: same repo means the import is a
    vendored copy, and the honest answer is the path plus no SHA.
    """
    top = _git(module_path, "rev-parse", "--show-toplevel")
    own_top = _git(own_root, "rev-parse", "--show-toplevel")
    if top is not None and top == own_top:
        return {"sha": None, "path": str(module_path),
                "note": "installed copy inside this repo, not a separate checkout"}
    return {"sha": _git_sha(module_path), "path": str(module_path)}


def write_manifest(out_dir: Path, *, cfg, method, deployed_path: str,
                   gains: dict | None = None,
                   gains_before: dict | None = None) -> Path | None:
    """Write ``manifest.json``: which checkpoint this run actually executed.

    The gap this closes is the one that makes 661 existing run folders hard to trust.
    ``deploy_args`` never sets ``args.pretrained_path`` -- the path goes straight to
    ``_LeRobotChunkSource`` -- so every ``summary.json`` on this rig records
    ``pretrained_path: null``. And for a B-spline run what executes is not even the
    configured path but the decode-free rewrite ``without_postprocessor_steps``
    produces, which until now appeared in no artifact at all.

    ``deployed_path`` is therefore recorded separately from ``cfg.policy_path``: the
    first is what ran, the second is what was asked for, and on B-spline they differ.

    Never raises.
    """
    try:
        import crisp_gym

        pace_root = Path(__file__).resolve().parents[3]
        crisp_gym_root = Path(crisp_gym.__file__).resolve().parents[1]
        manifest = {
            "task": cfg.task,
            "method": getattr(method, "type", "none"),
            "configured_policy_path": cfg.policy_path,
            "deployed_policy_path": deployed_path,
            "n_action_steps": cfg.n_action_steps,
            "fps": cfg.fps,
            "env_config": cfg.env_config,
            "blend": {"overlap": cfg.blend.overlap, "mode": cfg.blend.mode,
                      "skip": cfg.blend.skip},
            "loop": {"overlap_threshold": cfg.loop.overlap_threshold,
                     "stride": cfg.loop.stride,
                     **{k: getattr(cfg.loop, k) for k in
                        ("mode", "replan_every", "commit_rows", "bridge_rows", "bridge")
                        if hasattr(cfg.loop, k)}},
            "gripper": {"slowdown_frames": cfg.gripper.slowdown_frames,
                        "invert": cfg.gripper.invert,
                        "bspline_low_v": cfg.gripper.bspline_low_v,
                        "hold_s": getattr(cfg.gripper, "hold_s", None)},
            "gains": {"scale_kp": cfg.gains.scale_kp, "kp_exp": cfg.gains.kp_exp,
                      "kd_exp": cfg.gains.kd_exp,
                      "kp_pos": getattr(cfg.gains, "kp_pos", None),
                      "kp_rot": getattr(cfg.gains, "kp_rot", None)},
            # What the controller had before this run wrote its kp selection (None
            # when it wrote nothing), and what it was at once the selection and the
            # scaler had both read/applied. With kp_pos set, the second is the base
            # the scaler scales from.
            "controller_gains_before_run": gains_before,
            "controller_gains_at_startup": gains,
            "git": {
                "pace": _git_sha(pace_root),
                "crisp_gym": _dependency_provenance(crisp_gym_root, pace_root),
            },
        }
        path = Path(out_dir) / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("manifest written to %s", path)
        return path
    except Exception:
        logger.exception("failed to write manifest.json")
        return None
