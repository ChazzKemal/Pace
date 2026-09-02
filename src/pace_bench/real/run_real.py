"""Real-robot evaluation: the sibling of ``eval/run_libero.py``.

Same method registry, same ``--method.type`` CLI, same draccus config files -- a
LIBERO ablation and its robot counterpart differ only by which runner reads the
YAML. That symmetry is the point of the split: PACE decides speeds here, and
``crisp_gym.deploy`` applies them on hardware.

Config precedence is draccus's: dataclass defaults < ``--config_path file.yaml`` <
CLI override. ``run_libero.py`` already dumps a re-parsable ``run_config.yaml`` on
every run; this does the same, which matters more on hardware than in simulation
because a robot run cannot be replayed from a seed.

Status: complete, and unverified on hardware. Config, checkpoint validation, step
construction and the bring-up sequence are all in place; what has not happened is a
run on the arm, because the rig was down when this was written. The phase ordering is
the one property no offline check covers.
"""

# NOTE: no `from __future__ import annotations` here. It turns annotations into
# strings, and draccus reads main()'s annotation at runtime to find the config
# class -- it would receive the string "RealEvalConfig" and fail with
# "must be called with a dataclass type or instance". The `X | None` syntax used
# below needs no future import on Python 3.10+.
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import rclpy
from crisp_gym.deploy import session
from crisp_gym.deploy.cli import build_parser
from crisp_gym.deploy.loop import run_producer_loop
from crisp_gym.deploy.obs import _build_obs_schema, _get_obs_zerofill
from crisp_gym.deploy.sources import _LeRobotChunkSource
from crisp_gym.deploy.trace import RunRecord, write_run_artifacts

from pace_bench.methods.config import (
    BSplineMethod,
    MethodConfig,
    MethodPipelineConfig,
    NoMethod,
)
from pace_bench.real.checkpoint import (
    BSPLINE_DECODE_STEP,
    read_checkpoint,
    validate_method,
    without_postprocessor_steps,
)
from pace_bench.real.configs import materialise, resolve_policy_path
from pace_bench.real.deploy_flags import blend_overlap_for, set_flag
from pace_bench.real.deploy_steps import deploy_steps
from pace_bench.real.picker import maybe_pick_config
from pace_bench.real.record import (
    ChunkClock,
    PoseSampler,
    read_cartesian_gains,
    write_commands,
    write_inference,
    write_manifest,
    write_poses,
    write_splices,
)
from pace_bench.real.splice_loop import SpliceConfig, run_splice_loop

logger = logging.getLogger(__name__)


@dataclass
class SenderConfig:
    """How targets reach the robot.

    Defaults are the values this rig actually deploys with, not
    ``19_deploy_policy.py``'s historical argparse defaults. Those defaults are
    conservative -- Python sender, no startup delay -- and silently produce a
    different run from the one that has 256 sessions behind it.
    """

    #: C++ subprocess sender. Keeps cadence under GIL contention.
    cpp: bool = True
    #: Seconds to wait after starting the sender before the first chunk. With the
    #: C++ sender this is not optional: DDS needs the subscriber match to complete
    #: or the first chunk is published into the void.
    startup_delay: float = 4.0
    #: SCHED_FIFO priority for the C++ sender; 0 disables real-time scheduling.
    rt_priority: int = 0


@dataclass
class BlendConfig:
    """Chunk-seam smoothing.

    A new chunk is predicted from a fresh observation and need not continue the old
    one, so the join is where the arm jerks. ``hermite`` bridges position *and*
    velocity across the seam; ``linear`` averages the two predictions.
    """

    overlap: int = 4
    mode: str = "hermite"
    skip: int = 0


@dataclass
class LoopConfig:
    """When the producer goes back for more.

    ``overlap_threshold`` is the inference budget: the next chunk is requested once
    the queue falls to this many items, so inference must finish within
    ``overlap_threshold x dt_eff`` or the sender drains dry.
    """

    overlap_threshold: int = 2
    stride: int = 1
    #: "splice" runs `splice_loop.run_splice_loop`: one row ahead of the arm, a
    #: replan every `replan_every` rows, the new chunk spliced at the time-aligned
    #: row behind a `bridge_rows` cubic. `overlap_threshold` and `blend.*` are unused
    #: there. "queue" is crisp_gym's producer loop unchanged.
    mode: str = "splice"
    replan_every: int = 0      # 0 = n_action_steps - commit_rows - 1
    commit_rows: int = 1
    bridge_rows: int = 4
    bridge: str = "cubic"


@dataclass
class GainsConfig:
    """Controller stiffness scaling. Off unless asked -- it writes to the robot."""

    scale_kp: bool = False
    kp_exp: float = 2.0
    kd_exp: float = 1.0


@dataclass
class GripperConfig:
    """How this run gives the gripper back the time a speedup took from it.

    The stroke is fixed hardware: ~0.57 s at the driver's 0.150 m/s maximum, ~2.27 s
    at the 0.0375 m/s the deploy path uses. Measured across five recorded datasets the
    commanded channel is binary -- every transition is one frame -- so the window
    cannot be derived from the data alone and is stated here instead.
    """

    #: Frames held at nominal speed after an open->close edge (`pace`, `none`).
    #: Queue mode only; the splice loop uses `hold_s`.
    slowdown_frames: int = 5
    #: Seconds of demo time the arm runs at demo cadence after the *sent* gripper
    #: command closes, and how far past a predicted gripper movement PACE keeps
    #: every frame instead of striding. One number for one physical fact: the jaws
    #: need >= 0.57 s at the driver maximum. The 17:00 run on 2026-09-02 had 250 ms
    #: of hold against that stroke and resumed 2.9x while the cup was still open.
    hold_s: float = 0.7
    #: Gripper channel is inverted in some recordings.
    invert: bool = False
    #: Rows each gripper-motion row is repeated for (`bspline`). B-spline compresses
    #: waypoints rather than time, so it pays the gripper in rows like demospeedup --
    #: but it has no recorded stride to inherit, and the realised rate varies per
    #: chunk. 0 leaves it unstated, which `deploy_steps` refuses unless
    #: `slowdown_frames` is also 0. Start from the decoded `bspline_rate`.
    bspline_low_v: int = 0


@dataclass
class RecordConfig:
    """What this run leaves behind besides the robot having moved.

    Every field here maps onto a crisp_gym deploy flag that already exists and that
    ``19_deploy_policy.py`` has been using all along -- there are 397 ``trace.npz`` and
    315 ``.mp4`` files in ``deploy_runs/`` to prove it. None of them were reachable
    from this runner, because ``deploy_args`` seeds the namespace from the parser's
    defaults and never overrode them, so every pace_bench run so far recorded timing
    and nothing else.
    """

    #: Per-chunk obs->chunk pairing, written as trace.npz. Cheap without images.
    #: NOTE: what it stores is the chunk as the *policy* emitted it -- capture is at
    #: loop.py:156, ahead of --stride (253), the method pipeline (351) and the blend
    #: hold-back (382). For B-spline that means spline parameters, not a trajectory.
    trace: bool = True
    #: Capture every Nth chunk only.
    trace_every: int = 1
    #: Skip the per-chunk camera JPEGs. On by default: the images dominate both the
    #: buffer and the shutdown write, and the numerical arrays answer most questions.
    trace_no_images: bool = True
    #: Spawn one crisp_video_recorder subprocess per camera.
    save_video: bool = False
    #: Which cameras --save-video records; "all" for every camera the env has.
    video_camera: str = "all"
    #: Rate the achieved pose is sampled at, Hz. Independent of `fps`: the point is
    #: to oversample the 20 Hz command stream so the transport lag can be recovered by
    #: cross-correlation rather than assumed. 0 disables the sampler.
    pose_rate_hz: float = 50.0
    #: Suffix appended to the run folder name. Without it the 661 folders under
    #: deploy_runs/ are bare timestamps, and configs that differ only by method are
    #: indistinguishable after the fact. Left empty, the task and method are used.
    run_tag: str = ""


@dataclass
class RealEvalConfig(MethodPipelineConfig):
    """Everything one robot evaluation needs. ``--method.type`` selects the method."""

    policy_path: str = ""
    #: Which task's checkpoint to deploy. Resolved against `real/configs/tasks.yaml`
    #: using this run's `--method.type`, and only when `policy_path` is empty -- an
    #: explicit path always wins. Left blank, nothing is looked up, so every existing
    #: config keeps working unchanged.
    task: str = ""
    #: The registry `task` is resolved against. Relative to the repo, not to whichever
    #: directory the operator happened to launch from.
    tasks_file: str = "real/configs/tasks.yaml"
    out: Path = Path("outputs/eval_real")
    env_config: str = "ur10e_ridgeback_dual_cam_deploy_env_rotvec"
    fps: float = 20.0
    #: Executed steps per query. A property of the policy, not of any method, so it
    #: lives here rather than among the method's own fields.
    n_action_steps: int | None = None
    max_chunks: int = -1
    #: Seconds the homing trajectory is given, overriding the robot config's own value
    #: (crisp_py's default is 5.0; the deploy env YAML does not set one). None leaves it
    #: alone. This is a *motion* parameter -- the arm covers the same joint distance in
    #: less time -- so it is stated per run and recorded in run_config.yaml rather than
    #: baked into the environment, where it would also apply to teleop and recording.
    time_to_home: float | None = None
    gripper: GripperConfig = field(default_factory=GripperConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    sender: SenderConfig = field(default_factory=SenderConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    gains: GainsConfig = field(default_factory=GainsConfig)
    dry_run: bool = False
    #: Deploy a checkpoint whose trained method contradicts --method.type.
    force: bool = False

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.fps, 1e-9)


def _default_run_tag(cfg: RealEvalConfig, method=None) -> str:
    """A folder name that says what the run was, when the operator did not.

    `phase_video_and_delay` names the run folder for the wall clock and appends
    `--run-tag` if given. Nothing set it, so all 661 folders under `deploy_runs/` are
    bare timestamps -- and since a method config differs from its neighbours by little
    more than `--method.type`, finding last night's bspline run means opening
    summary.json files one at a time.
    """
    parts = [p for p in (cfg.task, getattr(method, "type", None)) if p]
    return "-".join(parts)


def deploy_args(cfg: RealEvalConfig, method=None):
    """crisp_gym's deploy vocabulary, seeded from its own CLI defaults.

    The actuation layer speaks the argparse namespace ``19_deploy_policy.py`` builds
    -- some sixty flags whose defaults were tuned against this rig. Rather than
    restate them in draccus and risk drift, the parser is asked for its own defaults
    and only what this config actually exposes is overridden. A flag nobody has
    thought about therefore keeps the value that was proven on hardware, instead of
    silently getting a fresh one.
    """
    args = build_parser().parse_args([])
    args.env_config = cfg.env_config
    args.fps = cfg.fps
    args.dry_run = cfg.dry_run
    args.max_chunks = cfg.max_chunks
    args.gripper_slowdown_frames = cfg.gripper.slowdown_frames
    args.invert_gripper = cfg.gripper.invert
    args.bspline_gripper_low_v = cfg.gripper.bspline_low_v
    args.cpp_sender = cfg.sender.cpp
    args.startup_delay = cfg.sender.startup_delay
    args.rt_priority = cfg.sender.rt_priority
    args.blend_overlap = cfg.blend.overlap
    args.blend_mode = cfg.blend.mode
    args.blend_skip = cfg.blend.skip
    args.overlap_threshold = cfg.loop.overlap_threshold
    args.stride = cfg.loop.stride
    args.scale_kp = cfg.gains.scale_kp
    args.kp_exp = cfg.gains.kp_exp
    args.kd_exp = cfg.gains.kd_exp
    args.yes = True                  # the method and checkpoint were already validated
    if cfg.n_action_steps is not None:
        args.n_act = cfg.n_action_steps

    # Recording flags. Set through `set_flag`, not by assignment: a Namespace accepts
    # any attribute, so a flag renamed under a crisp_gym pin bump would land on a dead
    # name and the run would record nothing while reporting that it had. That failure
    # is invisible until you go looking for the artifacts, which is the one moment it
    # is too late. (The older assignments above predate `set_flag` and are not yet
    # guarded this way.)
    set_flag(args, "record_trace", cfg.record.trace)
    set_flag(args, "record_trace_every", cfg.record.trace_every)
    set_flag(args, "record_trace_no_images", cfg.record.trace_no_images)
    set_flag(args, "save_video", cfg.record.save_video)
    set_flag(args, "video_camera", cfg.record.video_camera)
    set_flag(args, "run_tag", cfg.record.run_tag or _default_run_tag(cfg, method))

    # Seam blending is vetoed for methods whose chunks do not overlap in motion, so
    # this overrides the config's `blend.overlap` rather than reading it: the veto must
    # not be reachable by editing YAML. See `deploy_flags.blend_overlap_for`.
    if method is not None:
        args.blend_overlap = blend_overlap_for(method, cfg.blend.overlap)

    # The method owns the speed decision, so the heuristic schedule stays neutral --
    # for anything but `none` it is not even in the pipeline.
    args.max_speed = args.min_speed = 1.0

    # ...but the gain scaler is NOT part of the pipeline: phase_scaler reads
    # args.max_speed directly to size peak stiffness (kp = kp_base * s_eff**2). Leave
    # it at 1.0 and a method that drives the arm at 2.0 gets gains for 1.0 -- the arm
    # lags and cuts corners at exactly the speeds the method exists to reach. So the
    # method's own peak is propagated when it has one.
    peak = getattr(method, "max_speed", None) if method is not None else None
    if peak is not None:
        args.max_speed = float(peak)
        args.min_speed = float(getattr(method, "min_speed", None) or peak / 2)
    return args


def apply_task(cfg: RealEvalConfig) -> None:
    """Fill in ``policy_path`` from ``--task``, in place.

    Runs before :func:`resolve_method`, which reads the checkpoint to validate the
    method against it -- so the path has to exist by then. A task that resolves to
    nothing raises there rather than here, naming what the task does have.
    """
    if not cfg.task:
        return
    if cfg.policy_path:
        logger.info("--task=%s ignored: --policy_path was given explicitly", cfg.task)
        return
    declared = getattr(cfg.method, "type", "none")
    path = resolve_policy_path(cfg.task, declared, cfg.tasks_file)
    logger.info("task=%s method=%s -> %s", cfg.task, declared, path)
    cfg.policy_path = str(path)


def resolve_method(cfg: RealEvalConfig) -> MethodConfig:
    """Validate --method.type against the checkpoint and fill in what it knows.

    The flag stays the source of truth; the checkpoint is consulted to (a) refuse a
    launch that contradicts it and (b) default ``low_v``, which the operator should
    not have to retype and must not get wrong.
    """
    if not cfg.policy_path:
        return cfg.method
    facts = read_checkpoint(cfg.policy_path)
    declared = getattr(cfg.method, "type", "none")
    validate_method(declared, facts, force=cfg.force)

    method = cfg.method
    if declared == "demospeedup" and facts.low_v is not None:
        # Only when the operator left it at the dataclass default: an explicit
        # --method.low_v must still win.
        if getattr(method, "low_v", None) in (None, 2) and facts.low_v != 2:
            logger.info("low_v=%d taken from the checkpoint's built pipeline", facts.low_v)
            method.low_v = facts.low_v
    if facts.method_type:
        logger.info(
            "checkpoint: method=%s low_v=%s chunk=%s source_chunk=%s halved=%s",
            facts.method_type, facts.low_v, facts.chunk_size,
            facts.source_chunk, facts.halving_applied,
        )
    return method


def build_steps(cfg: RealEvalConfig, method: MethodConfig, dataset_stats=None) -> list:
    """The deploy pipeline this run will hand to crisp_gym's producer loop."""
    if cfg.gripper.hold_s > 0 and hasattr(method, "gripper_stride_exempt_frames"):
        # The stride exemption and the grasp hold describe the same stroke, so they
        # are sized from the same number; otherwise strided rows land inside the
        # hold and "demo cadence" becomes 100 ms rows.
        frames = round(cfg.gripper.hold_s * cfg.fps)
        if method.gripper_stride_exempt_frames != frames:
            logger.info("gripper_stride_exempt_frames %d -> %d (gripper.hold_s=%.2f s at %g fps)",
                        method.gripper_stride_exempt_frames, frames, cfg.gripper.hold_s, cfg.fps)
            method.gripper_stride_exempt_frames = frames
    return deploy_steps(
        method, args=deploy_args(cfg, method),
        n_action_steps=cfg.n_action_steps,
        control_dt=cfg.control_dt,
        dataset_stats=dataset_stats,
    )


def dump_run_config(cfg: RealEvalConfig, out_dir: Path | None = None) -> Path:
    """Write a re-parsable record of exactly this run, as run_libero.py does.

    Called twice. The first goes to ``cfg.out``, before the robot is touched, so a
    launch that dies during bring-up still leaves a record of what was attempted. The
    second goes into the run folder once ``phase_video_and_delay`` has named it --
    which is the copy that matters, because ``cfg.out / run_config.yaml`` is a single
    flat path that the *next* run overwrites. Every one of the 661 folders under
    ``deploy_runs/`` was written without one, and none of them can now be matched back
    to the config that produced it.
    """
    dest = Path(out_dir) if out_dir is not None else cfg.out
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "run_config.yaml"
    with open(path, "w") as f:
        # Encoded against RealEvalConfig so the method's choice key is included,
        # which is what makes the file re-parsable with --config_path.
        draccus.dump(cfg, f)
    return path


def quiesce_target_pose_timer(env) -> None:
    """Stop crisp_py's target-pose timer before the publisher is taken away.

    ``enable_target_pose_publishing`` (crisp_gym.deploy.patches) creates the
    publisher *and* a 20 Hz timer calling ``Robot._callback_publish_target_pose``.
    ``phase_publish_channels`` then hands the topic to the sender by setting
    ``robot._target_pose_publisher = None`` -- and cancels nothing. The callback
    re-reads that attribute after its own ``is None`` guard has passed (crisp_py
    robot.py:466 guards, :471 dereferences), so a handover landing in that window
    raises AttributeError inside the shared executor's spin thread. That thread has
    no try/except (crisp_gym manipulator_env.py:136-138), so it dies, and every
    robot and gripper callback dies with it while the run keeps going on frozen
    observations.

    Cancelling before the handover closes the window; the sleep lets a callback
    already in flight finish, since the executor runs on its own thread.
    """
    robot = env.robot
    cb = robot._callback_publish_target_pose
    cancelled = 0
    for timer in list(robot.node.timers):
        if getattr(timer, "callback", None) == cb:
            timer.cancel()
            cancelled += 1
    if cancelled:
        time.sleep(2.0 / max(robot.config.publish_frequency, 1.0))
    logger.info("target_pose timer(s) cancelled before handover: %d", cancelled)


def _raw_index_fn(method):
    """Which raw frame each pipeline output row came from, for the grasp hold.

    PACE strides, so its output rows map to raw frames through `stride_indices` --
    the same call `PaceSpeed.plan` makes, on the same config, so the two agree row
    for row. Every other method keeps the demo cadence and needs no map.
    """
    if getattr(method, "type", None) != "pace":
        return None
    import numpy as np
    import torch

    from pace_bench.methods.pace.speed import stride_indices

    pc = method.to_pace_config()
    return lambda chunk: stride_indices(
        torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None], pc)


def own_the_interrupt() -> None:
    """Make Ctrl-C reach *this* code while ROS is still alive.

    ``rclpy.init()`` (inside ``build_env``) installs its own SIGINT/SIGTERM
    handlers, which invalidate the ROS context the instant the signal arrives.
    Our ``finally`` then runs against a dead context, and ``ReplayScaler.restore``
    -- which needs one SetParameters round-trip -- can only log that it CANNOT
    restore kp/kd. Every Ctrl-C'd run since gain scaling went on has left the
    controller at the last scaled gains, which is what the next run then tunes
    against.

    So once the env is up, rclpy's handlers are removed and Python's default one
    put back: SIGINT raises KeyboardInterrupt in the main thread (inside the loop,
    where the ``finally`` handles it), and SIGTERM does the same. The context is
    shut down by us, last, after the gains are back.
    """
    try:
        from rclpy.signals import uninstall_signal_handlers
        uninstall_signal_handlers()
    except Exception:
        logger.exception("could not remove rclpy's signal handlers; a Ctrl-C may "
                         "leave scaled gains on the controller")
        return

    def _term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _term)
    logger.info("Ctrl-C now raises KeyboardInterrupt here; rclpy stays up until teardown")


def restore_gains(scaler) -> None:
    """Put kp/kd back, loudly if it cannot be done."""
    if scaler is None:
        return
    try:
        if not rclpy.ok():
            logger.error("rclpy context is down; kp/kd CANNOT be restored from here. "
                         "Run `ros2 run tum09_custom reset_crisp_kp.py` before the next run.")
        scaler.restore()
    except Exception:
        logger.exception("gain restore failed; run `ros2 run tum09_custom reset_crisp_kp.py`")


def run_on_robot(cfg: RealEvalConfig, steps: list, args, method=None) -> None:
    """Bring the robot up, run the method's pipeline, tear down.

    The phases run in the order the hardware requires -- controller switched before
    anything is published, gains scaled before the first chunk, sender started and
    matched before the producer. Teardown is in a ``finally`` because a live sender
    thread and scaled controller gains outlive a crash.
    """
    _t = time.monotonic()
    env = session.build_env(args)
    logger.info("build_env took %.1f s", time.monotonic() - _t)
    own_the_interrupt()

    if cfg.time_to_home is not None:
        was = getattr(env.robot.config, "time_to_home", None)
        env.robot.config.time_to_home = float(cfg.time_to_home)
        logger.info("time_to_home %.1f s -> %.1f s", was, cfg.time_to_home)

    # A B-spline checkpoint ships its own decode step, with the num_actions frozen at
    # training. This pipeline supplies the decode instead, built from the run
    # configuration -- which is the point, since num_actions is a decode-time choice
    # and the whole claim of the method. Left in place the checkpoint's step runs
    # first inside the inference subprocess: our decode then gets actions where it
    # expects parameters, and the configured value is discarded without a word.
    # Inert for every other method, and for a B-spline checkpoint saved without one.
    pretrained = cfg.policy_path
    if isinstance(method, BSplineMethod):
        pretrained = str(without_postprocessor_steps(pretrained, (BSPLINE_DECODE_STEP,)))
        if pretrained != cfg.policy_path:
            logger.info("deploying a decode-free view of the checkpoint: %s", pretrained)

    src = _LeRobotChunkSource(
        pretrained_path=pretrained, env=env,
        # Without this the checkpoint's own n_action_steps wins and cfg is ignored:
        # a config asking for 32 would quietly run the policy's full 100.
        n_action_steps=cfg.n_action_steps,
    )
    n_obs, n_act = src.n_obs, src.n_act

    scaler = sender = rec = None
    sampler = clock = None
    video_recorders: list = []
    splices: list = []
    started_mono = time.monotonic()
    try:
        _t = time.monotonic()
        session.phase_home(env, args)
        logger.info("phase_home took %.1f s", time.monotonic() - _t)
        session.phase_switch_controller(env, args)
        scaler = session.phase_scaler(env, args)
        # Read once, after the controller is live and before anything moves. This is
        # the coefficient that turns the tracking error recorded below into a force:
        # joint efforts never reach this process (crisp_py drops msg.effort), but the
        # controller is an impedance law, so wrench = kp * (target - achieved).
        gains = read_cartesian_gains(env, args.controller_node, scaler=scaler)
        session.phase_pin_gripper_speed(env, args)
        session.phase_gil_hygiene(env, args)
        quiesce_target_pose_timer(env)
        ch = session.phase_publish_channels(env, args)
        # Both senders drive the splice loop: with commit_rows=1 nothing is ever
        # retracted from the sender itself, only from the plan. `sender.cpp` is the
        # A/B switch; frames.csv slack and overshoot say which cadence is better.
        sender, q = session.phase_start_sender(env, args, scaler, ch)
        # When each inference ran and when each target was due, measured at the two
        # points they pass through this process. `timeline.py` can reconstruct both
        # from trace.npz and the anchoring rule, but this makes them a recorded fact
        # -- including for the last chunk, whose chunks.csv row is never written
        # when the run ends by Ctrl-C mid-drain.
        clock = ChunkClock(q)
        clock.wrap_source(src)
        clock.tap_queue(q)
        # Ground truth on its own clock. NOT hung off the sender's `state_capture_fn`:
        # under the C++ sender that hook fires inside `put()`, which the producer calls
        # K times back-to-back per chunk, so it would sample the arm in bursts rather
        # than at a steady rate. See `record`'s module docstring.
        if cfg.record.pose_rate_hz > 0:
            sampler = PoseSampler(env, cfg.record.pose_rate_hz)
            sampler.start()
        started_at, out_dir, video_recorders = session.phase_video_and_delay(
            env, args, n_obs, n_act)
        write_manifest(out_dir, cfg=cfg, method=method, deployed_path=pretrained,
                       gains=gains)
        dump_run_config(cfg, out_dir)

        rec = RunRecord(out_dir=out_dir, run_started_at=started_at, duration_s=0.0,
                        n_obs=n_obs, n_act=n_act, chunk_count=0, stopped_by="init")
        schema = _build_obs_schema(env)
        last = [None]
        buf: deque = deque(maxlen=n_obs)
        while len(buf) < n_obs:
            buf.append(_get_obs_zerofill(env, schema, last))

        started_mono = time.monotonic()
        common = {
            "env": env, "chunk_source": src, "q": q, "args": args, "rec": rec,
            "dt_base": cfg.control_dt, "obs_schema": schema,
            "gripper_enabled": ch.gripper_enabled,
            "gripper_unnormalize_fn": ch.gripper_unnormalize_fn,
            "obs_buf": buf, "last_obs": last, "steps": steps,
        }
        if cfg.loop.mode == "splice":
            run_splice_loop(
                splices=splices,
                sender=sender,
                cfg=SpliceConfig(replan_every=cfg.loop.replan_every,
                                 commit_rows=cfg.loop.commit_rows,
                                 bridge_rows=cfg.loop.bridge_rows,
                                 bridge=cfg.loop.bridge,
                                 hold_s=cfg.gripper.hold_s, fps=cfg.fps,
                                 grip_invert=cfg.gripper.invert),
                n_action_steps=n_act, raw_index_fn=_raw_index_fn(method), **common)
        elif cfg.loop.mode == "queue":
            run_producer_loop(lookbehind_buf=deque(maxlen=8), **common)
        else:
            raise ValueError(f"loop.mode must be 'splice' or 'queue', not {cfg.loop.mode!r}")
    finally:
        # Drain the sender BEFORE reporting, so its counters are final.
        if sender is not None:
            try:
                q.put(None); sender.join(5.0)
            except Exception:
                logger.exception("sender shutdown")
        # Gains go back the moment the arm has stopped -- before the artifact
        # writes, which can take seconds, and long before rclpy.shutdown().
        restore_gains(scaler)
        # Stop the video recorders once the arm has stopped, not before -- the drain
        # above is still real motion. `stop()` sends SIGINT, which is what makes the
        # mp4 valid: rclcpp shutdown -> node destructor -> cv::VideoWriter release ->
        # trailer flushed (video.py:93-99). This runner spawned them and never stopped
        # them; the subprocesses outlived the run and their mp4s had no trailer. It
        # went unnoticed only because `save_video` was unreachable from here, so the
        # list was always empty -- exposing the flag is what makes the leak real.
        for vrec in video_recorders:
            try:
                vrec.stop(timeout=5.0)
            except Exception:
                logger.exception("video recorder shutdown")
        if video_recorders:
            logger.info("stopped %d video recorder(s)", len(video_recorders))
        if sampler is not None:
            sampler.stop()
        # Report from the finally, not the try: a run normally ends by Ctrl-C or by
        # an exception out of the loop, and the record is most wanted exactly then.
        if rec is not None:
            try:
                rec.duration_s = time.monotonic() - started_mono
                rec.sender_stage_samples = getattr(sender, "stage_samples", {}) or {}
                write_run_artifacts(rec, args, sender, None)
            except Exception:
                logger.exception("failed to write run artifacts")
            # The sender has been accumulating one row per queued target into
            # `_replay_log` for the whole run, and crisp_gym writes it nowhere.
            try:
                write_commands(getattr(sender, "_replay_log", []) or [], rec.out_dir,
                               clock)
                write_inference(clock, rec.out_dir)
                write_splices(splices, rec.out_dir)
                if sampler is not None:
                    write_poses(sampler.rows, rec.out_dir)
            except Exception:
                logger.exception("failed to write commands.csv / poses.csv")
        if sampler is not None and sampler.n_errors:
            logger.warning(
                "achieved-pose read failed on %d sample(s) of %d; poses.csv is "
                "correspondingly sparse", sampler.n_errors, len(sampler.rows))
        try:
            src.shutdown()
        except Exception:
            logger.exception("chunk source shutdown")
        try:
            env.close()
        except Exception:
            logger.exception("env close")
        if rclpy.ok():
            rclpy.shutdown()


@draccus.wrap()
def main(cfg: RealEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    apply_task(cfg)
    method = resolve_method(cfg)
    steps = build_steps(cfg, method)
    logger.info("method=%s | steps=%s",
                getattr(method, "type", "none"), [type(s).__name__ for s in steps])
    logger.info("run config -> %s", dump_run_config(cfg))
    run_on_robot(cfg, steps, deploy_args(cfg, method), method)


def _resolve_config_arg(argv: list[str]) -> list[str]:
    """Expand ``_include`` in a --config_path before draccus opens the file.

    draccus reads one YAML by path and knows nothing about inheritance, so the merge
    happens here and it is handed the resolved file. A config without an ``_include``
    is passed through by path, so it keeps its own name in logs.
    """
    out = list(argv)
    for i, a in enumerate(out):
        if a == "--config_path" and i + 1 < len(out):
            out[i + 1] = str(materialise(out[i + 1]))
        elif a.startswith("--config_path="):
            out[i] = "--config_path=" + str(materialise(a.split("=", 1)[1]))
    return out


if __name__ == "__main__":
    # Import the package-qualified main so draccus resolves the same class object
    # (under `python -m`, this file is __main__ and RealEvalConfig would differ).
    from pace_bench.real.run_real import main as packaged_main

    # No --config_path and a terminal to ask in: offer the choice rather than falling
    # through to dataclass defaults, which are not a run anybody meant to start.
    # Returns None only when the operator cancelled, which must launch nothing.
    chosen = maybe_pick_config(sys.argv)
    if chosen is None:
        sys.exit(0)

    sys.argv = _resolve_config_arg(chosen)
    packaged_main()
