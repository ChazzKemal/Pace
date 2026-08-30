"""Real-robot evaluation: the sibling of ``eval/run_libero.py``.

Same method registry, same ``--method.type`` CLI, same draccus config files -- a
LIBERO ablation and its robot counterpart differ only by which runner reads the
YAML. That symmetry is the point of the split: PACE decides speeds here, and
``crisp_gym.deploy`` applies them on hardware.

Config precedence is draccus's: dataclass defaults < ``--config_path file.yaml`` <
CLI override. ``run_libero.py`` already dumps a re-parsable ``run_config.yaml`` on
every run; this does the same, which matters more on hardware than in simulation
because a robot run cannot be replayed from a seed.

Status: the config, checkpoint validation and step construction below are complete
and tested. Driving the robot needs ``main()``'s remaining ~737 lines of hardware
bring-up (env construction, controller switch, gain scaling, sender startup) lifted
out of ``19_deploy_policy.py`` into a reusable session object -- see
``build_session()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from pace_bench.methods.config import MethodConfig, MethodPipelineConfig, NoMethod
from pace_bench.real.checkpoint import read_checkpoint, validate_method
from pace_bench.real.deploy_steps import deploy_steps

logger = logging.getLogger(__name__)


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
    dry_run: bool = False
    #: Deploy a checkpoint whose trained method contradicts --method.type.
    force: bool = False

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.fps, 1e-9)


class _StepArgs:
    """The handful of deploy-CLI values the crisp_gym steps read.

    crisp_gym's steps take the argparse namespace 19_deploy_policy.py builds. Rather
    than reshape those steps for draccus, this presents the same attribute names --
    the adapter stays in one place instead of spreading a second config vocabulary
    through the actuation layer.
    """

    def __init__(self, cfg: RealEvalConfig):
        self.gripper_slowdown_frames = cfg.gripper.slowdown_frames
        self.invert_gripper = cfg.gripper.invert
        self.fps = cfg.fps
        # HeuristicSpeed reads these; a method that supplies its own speeds ignores
        # them, and method `none` reproduces the pre-method defaults exactly.
        self.max_speed = 1.0
        self.min_speed = 1.0
        self.clamp_deg = 5.0
        self.lookahead = 0
        self.lookbehind = 0
        self.cum_lookahead = 0


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
        method, args=_StepArgs(cfg),
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


def build_session(cfg: RealEvalConfig, steps: list):
    """Bring the robot up and run the loop. NOT YET IMPLEMENTED.

    Needs main()'s hardware bring-up lifted out of 19_deploy_policy.py into a
    reusable session: env construction and readiness, controller switch, gain
    scaling, GIL hygiene, sender startup and the startup delay. Once that exists
    this becomes a handful of lines, because crisp_gym.deploy.loop already accepts
    the `steps` list built above.
    """
    raise NotImplementedError(
        "hardware bring-up still lives in 19_deploy_policy.py's main(); extract it "
        "into crisp_gym.deploy.session before run_real can drive the robot"
    )


@draccus.wrap()
def main(cfg: RealEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    method = resolve_method(cfg)
    steps = build_steps(cfg, method)
    logger.info("method=%s | steps=%s",
                getattr(method, "type", "none"), [type(s).__name__ for s in steps])
    logger.info("run config -> %s", dump_run_config(cfg))
    build_session(cfg, steps)


if __name__ == "__main__":
    # Import the package-qualified main so draccus resolves the same class object
    # (under `python -m`, this file is __main__ and RealEvalConfig would differ).
    from pace_bench.real.run_real import main as packaged_main

    packaged_main()
