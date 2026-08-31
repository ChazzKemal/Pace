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
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from pace_bench.methods.config import MethodConfig, MethodPipelineConfig, NoMethod
from pace_bench.real.checkpoint import read_checkpoint, validate_method
from pace_bench.real.configs import materialise
from pace_bench.real.deploy_steps import deploy_steps

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
    slowdown_frames: int = 5
    #: Gripper channel is inverted in some recordings.
    invert: bool = False


@dataclass
class RealEvalConfig(MethodPipelineConfig):
    """Everything one robot evaluation needs. ``--method.type`` selects the method."""

    policy_path: str = ""
    out: Path = Path("outputs/eval_real")
    env_config: str = "ur10e_ridgeback_dual_cam_deploy_env_rotvec"
    fps: float = 20.0
    #: Executed steps per query. A property of the policy, not of any method, so it
    #: lives here rather than among the method's own fields.
    n_action_steps: int | None = None
    max_chunks: int = -1
    gripper: GripperConfig = field(default_factory=GripperConfig)
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


def deploy_args(cfg: RealEvalConfig, method=None):
    """crisp_gym's deploy vocabulary, seeded from its own CLI defaults.

    The actuation layer speaks the argparse namespace ``19_deploy_policy.py`` builds
    -- some sixty flags whose defaults were tuned against this rig. Rather than
    restate them in draccus and risk drift, the parser is asked for its own defaults
    and only what this config actually exposes is overridden. A flag nobody has
    thought about therefore keeps the value that was proven on hardware, instead of
    silently getting a fresh one.
    """
    from crisp_gym.deploy.cli import build_parser

    args = build_parser().parse_args([])
    args.env_config = cfg.env_config
    args.fps = cfg.fps
    args.dry_run = cfg.dry_run
    args.max_chunks = cfg.max_chunks
    args.gripper_slowdown_frames = cfg.gripper.slowdown_frames
    args.invert_gripper = cfg.gripper.invert
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
    return deploy_steps(
        method, args=deploy_args(cfg, method),
        n_action_steps=cfg.n_action_steps,
        control_dt=cfg.control_dt,
        dataset_stats=dataset_stats,
    )


def dump_run_config(cfg: RealEvalConfig) -> Path:
    """Write a re-parsable record of exactly this run, as run_libero.py does."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    path = cfg.out / "run_config.yaml"
    with open(path, "w") as f:
        # Encoded against RealEvalConfig so the method's choice key is included,
        # which is what makes the file re-parsable with --config_path.
        draccus.dump(cfg, f)
    return path


def run_on_robot(cfg: RealEvalConfig, steps: list, args) -> None:
    """Bring the robot up, run the method's pipeline, tear down.

    The phases run in the order the hardware requires -- controller switched before
    anything is published, gains scaled before the first chunk, sender started and
    matched before the producer. Teardown is in a ``finally`` because a live sender
    thread and scaled controller gains outlive a crash.
    """
    import time
    from collections import deque

    import rclpy
    from crisp_gym.deploy import session
    from crisp_gym.deploy.loop import run_producer_loop
    from crisp_gym.deploy.obs import _build_obs_schema, _get_obs_zerofill
    from crisp_gym.deploy.sources import _LeRobotChunkSource
    from crisp_gym.deploy.trace import RunRecord, write_run_artifacts

    env = session.build_env(args)
    src = _LeRobotChunkSource(
        pretrained_path=cfg.policy_path, env=env,
        # Without this the checkpoint's own n_action_steps wins and cfg is ignored:
        # a config asking for 32 would quietly run the policy's full 100.
        n_action_steps=cfg.n_action_steps,
    )
    n_obs, n_act = src.n_obs, src.n_act

    scaler = sender = rec = None
    started_mono = time.monotonic()
    try:
        session.phase_home(env, args)
        session.phase_switch_controller(env, args)
        scaler = session.phase_scaler(env, args)
        session.phase_pin_gripper_speed(env, args)
        session.phase_gil_hygiene(env, args)
        ch = session.phase_publish_channels(env, args)
        sender, q = session.phase_start_sender(env, args, scaler, ch)
        started_at, out_dir, recorders = session.phase_video_and_delay(
            env, args, n_obs, n_act)

        rec = RunRecord(out_dir=out_dir, run_started_at=started_at, duration_s=0.0,
                        n_obs=n_obs, n_act=n_act, chunk_count=0, stopped_by="init")
        schema = _build_obs_schema(env)
        last = [None]
        buf: deque = deque(maxlen=n_obs)
        while len(buf) < n_obs:
            buf.append(_get_obs_zerofill(env, schema, last))

        started_mono = time.monotonic()
        run_producer_loop(
            env=env, chunk_source=src, q=q, args=args, rec=rec,
            dt_base=cfg.control_dt, obs_schema=schema,
            gripper_enabled=ch.gripper_enabled,
            gripper_unnormalize_fn=ch.gripper_unnormalize_fn,
            obs_buf=buf, last_obs=last, lookbehind_buf=deque(maxlen=8),
            steps=steps,
        )
    finally:
        # Drain the sender BEFORE reporting, so its counters are final.
        if sender is not None:
            try:
                q.put(None); sender.join(5.0)
            except Exception:
                logger.exception("sender shutdown")
        # Report from the finally, not the try: a run normally ends by Ctrl-C or by
        # an exception out of the loop, and the record is most wanted exactly then.
        if rec is not None:
            try:
                rec.duration_s = time.monotonic() - started_mono
                rec.sender_stage_samples = getattr(sender, "stage_samples", {}) or {}
                write_run_artifacts(rec, args, sender, None)
            except Exception:
                logger.exception("failed to write run artifacts")
        try:
            src.shutdown()
        except Exception:
            logger.exception("chunk source shutdown")
        if scaler is not None:
            scaler.restore()
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
    method = resolve_method(cfg)
    steps = build_steps(cfg, method)
    logger.info("method=%s | steps=%s",
                getattr(method, "type", "none"), [type(s).__name__ for s in steps])
    logger.info("run config -> %s", dump_run_config(cfg))
    run_on_robot(cfg, steps, deploy_args(cfg, method))


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
    import sys

    from pace_bench.real.run_real import main as packaged_main

    sys.argv = _resolve_config_arg(sys.argv)
    packaged_main()
