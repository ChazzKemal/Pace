"""Evaluate a policy on LIBERO, one task per output directory.

Mirrors upstream's ``eval_main`` -- same env factory, policy loader, processors,
``eval_policy_all`` and ``eval_info.json`` schema -- with two additions: the method's
pipeline steps are attached to the policy, and the policy is introduced to the vector
env so its actuator can reach the simulator.

One task per output directory, matching the layout of the recorded results this is
compared against.

    python -m robot_stack.eval.run_libero --out outputs/baseline           # no method

    python -m robot_stack.eval.run_libero --out outputs/pace \\
        --method.type=pace --method.max_speed=1.5 --method.action_stride=2 \\
        --method.n_lookahead=4 --method.lookahead_agg=cumulative_bending \\
        --method.lookahead_target=angle
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
from lerobot.configs.eval import EvalPipelineConfig  # noqa: F401  (registers env choices)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

from robot_stack.eval.pace_policy import attach_pace
from robot_stack.eval.sim_time import wrap_vector_env
from robot_stack.methods.config import MethodPipelineConfig, NoMethod
from robot_stack.methods.pace.actuator import RobosuiteSpeedActuator
from robot_stack.methods.pace.processor import PaceSpeedStep

logger = logging.getLogger(__name__)


@dataclass
class ActuationConfig:
    """How a chosen speed is realised in robosuite. See :mod:`..methods.pace.actuator`."""

    kpkd_scale_exp: float = 2.0
    disable_kpkd_scaling: bool = False
    disable_gripper_speedup: bool = False
    # "up" reproduces the recorded runs and can exceed max_speed by about 6 percent.
    # "down" makes max_speed a true ceiling, at some cost in throughput.
    speed_rounding: str = "up"
    # Compute speeds and stride the chunk, but leave the simulator nominal. Isolates
    # the action-side speedup from the controller-side gain compensation.
    disabled: bool = False


@dataclass
class LiberoEvalConfig(MethodPipelineConfig):
    """Everything one evaluation needs. ``--method.type`` selects the method."""

    out: Path = Path("outputs/eval")
    policy_path: str = "lerobot/xvla-libero"
    task_suite: str = "libero_10"
    tasks: str = "0-9"  # "0-9", "2", "0-3,7"
    seed: int = 42
    n_episodes: int = 50
    batch_size: int = 10
    # How much of each chunk is executed before the policy is queried again. A
    # property of the policy rather than of any method, so it lives here.
    n_action_steps: int = 32
    device: str | None = None
    actuation: ActuationConfig = field(default_factory=ActuationConfig)


def parse_tasks(spec: str) -> list[int]:
    """``"0-9"`` or ``"2,5,7"`` or ``"0-3,7"`` -> a list of task ids."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def action_stats(postprocessor) -> dict | None:
    """Find the action normalization stats inside a postprocessor pipeline.

    PACE measures geometry -- angles, degrees -- so it needs absolute units. Where a
    policy emits normalized actions, the statistics needed to undo that are already
    carried by the unnormalizer step, so they are borrowed rather than re-derived.

    Returning None is normal, not a failure: xvla-libero normalizes actions with
    NormalizationMode.IDENTITY, so its actions are absolute already and its stats
    dict is empty. PACE's unnormalization is then a no-op, which is correct.
    """
    for step in getattr(postprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and "action" in stats:
            return stats
    return None


def build_speed_step(cfg: LiberoEvalConfig, stats: dict | None) -> PaceSpeedStep:
    """The method's postprocessor step, or an inert one when no method is selected.

    An inert ``PaceSpeedStep`` and no step at all are the same thing -- with default
    config the chunk passes through untouched at speed 1.0 -- so the baseline reuses
    one code path instead of branching.
    """
    steps = cfg.method.postprocessor_steps()
    step = steps[0] if steps else PaceSpeedStep()
    step.n_action_steps = cfg.n_action_steps
    step.dataset_stats = stats
    return step


@draccus.wrap()
def main(cfg: LiberoEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.n_action_steps = cfg.n_action_steps
    if cfg.device:
        policy_cfg.device = cfg.device

    device = get_safe_torch_device(policy_cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy = make_policy(cfg=policy_cfg, env_cfg=LiberoEnv(task=cfg.task_suite))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    stats = action_stats(postprocessor)
    logger.info("action stats: %s", "from unnormalizer" if stats else "none (identity normalization)")

    # No method means no actuation: there is no speed to realise.
    actuate = not (cfg.actuation.disabled or isinstance(cfg.method, NoMethod))
    actuator = (
        RobosuiteSpeedActuator(
            kpkd_scale_exp=cfg.actuation.kpkd_scale_exp,
            disable_kpkd_scaling=cfg.actuation.disable_kpkd_scaling,
            disable_gripper_speedup=cfg.actuation.disable_gripper_speedup,
            speed_rounding=cfg.actuation.speed_rounding,
            action_stride=getattr(cfg.method, "action_stride", 1),
        )
        if actuate
        else None
    )

    paced = attach_pace(policy, build_speed_step(cfg, stats), actuator)
    logger.info("method=%s | %s", cfg.method.type, paced.pace.get_config())

    cfg.out.mkdir(parents=True, exist_ok=True)
    with open(cfg.out / "run_config.json", "w") as f:
        # Encoded against LiberoEvalConfig so the method's choice key is included,
        # which is what makes the file re-parsable with --config_path.
        draccus.dump(cfg, f)

    for task_id in parse_tasks(cfg.tasks):
        task_out = cfg.out / f"task_{task_id}"
        if (task_out / "eval_info.json").exists():
            logger.info("task %d already evaluated, skipping", task_id)
            continue

        set_seed(cfg.seed)
        env_cfg = LiberoEnv(task=cfg.task_suite, task_ids=[task_id], control_mode="absolute")
        envs = make_env(env_cfg, n_envs=cfg.batch_size, use_async_envs=False)

        # One task in, one vector env out -- so binding is unambiguous.
        (vec_env,) = (v for group in envs.values() for v in group.values())
        recorders = wrap_vector_env(vec_env)
        paced.bind_env(vec_env)

        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        autocast = (
            torch.autocast(device_type=device.type)
            if getattr(policy_cfg, "use_amp", False)
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            info = eval_policy_all(
                envs=envs,
                policy=paced,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.n_episodes,
                start_seed=cfg.seed,
            )
        close_envs(envs)

        task_out.mkdir(parents=True, exist_ok=True)
        _record_timings(info, recorders, cfg, task_id, paced)
        (task_out / "eval_info.json").write_text(json.dumps(info, indent=2))
        (task_out / "applied_speeds.json").write_text(json.dumps(paced.pace_speed_log))
        paced.pace_speed_log.clear()
        logger.info(
            "task %d: SR %.1f%%  ATR %s s",
            task_id,
            info["overall"]["pc_success"],
            f"{info['overall']['avg_success_sim_s']:.2f}" if info["overall"]["avg_success_sim_s"] else "n/a",
        )


def _record_timings(info: dict, recorders, cfg: LiberoEvalConfig, task_id: int, paced) -> None:
    """Fold per-episode durations and applied speeds into upstream's schema.

    Only *successful* episodes are reliably captured: a success terminates the env,
    whereas a failure usually runs to the step cap and upstream ends the rollout there
    without the env ever truncating. That is enough -- ATR averages successes, and TPR
    needs failures only as a flag, which `pc_success` carries -- but it means the
    recorded success count is worth checking against upstream's before either is
    trusted.
    """
    episodes = [ep for rec in recorders for ep in rec.episodes]
    succeeded = [e["sim_time"] for e in episodes if e["success"] and e["sim_time"] is not None]

    expected = round(info["overall"]["pc_success"] / 100 * cfg.n_episodes)
    n_recorded = sum(1 for e in episodes if e["success"])
    if n_recorded != expected:
        logger.warning(
            "task %d: recorded %d successful episodes but SR implies %d -- "
            "ATR is computed over the recorded subset",
            task_id,
            n_recorded,
            expected,
        )

    info["episodes"] = episodes
    info["overall"]["avg_success_sim_s"] = sum(succeeded) / len(succeeded) if succeeded else None
    info["overall"]["n_success_timed"] = len(succeeded)

    # Speeds actually delivered to the simulator (env 0), post-quantization. The mean
    # is the single number saying whether the method ran as fast as intended: sim time
    # scales as 1/speed, so a few percent here is a few percent of ATR.
    speeds = paced.pace_speed_log
    if speeds:
        info["overall"]["applied_speed_mean"] = sum(speeds) / len(speeds)
        info["overall"]["applied_speed_min"] = min(speeds)
        info["overall"]["applied_speed_max"] = max(speeds)
        info["overall"]["applied_speed_below_1"] = sum(1 for v in speeds if v < 1.0) / len(speeds)
    # Encoded through draccus rather than asdict: it handles Path and the method's
    # choice key, and matches what run_config.json holds.
    info["config"] = draccus.encode(cfg)


if __name__ == "__main__":
    main()
