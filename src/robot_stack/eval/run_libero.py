"""Evaluate a PACE-paced policy on LIBERO, one task per run.

Mirrors upstream's ``eval_main`` -- same env factory, same policy loader, same
processors, same ``eval_policy_all`` and the same ``eval_info.json`` schema -- with
two additions: PACE is attached to the policy by :func:`attach_pace`, and the policy
is introduced to the vector env so its actuator can reach the simulator.

One task per invocation, writing ``<output_dir>/task_<id>/eval_info.json``, because
that is the layout the recorded results use and comparing against them is the point.

    python -m robot_stack.eval.run_libero \\
        --policy-path lerobot/xvla-libero --seed 42 --tasks 0-9 \\
        --max-speed 1.5 --min-speed 0.75 --action-stride 2 \\
        --n-lookahead 4 --lookahead-agg cumulative_bending --lookahead-target angle \\
        --no-enable-ori-axis --out outputs/pace_look4cb_skip2_1.5
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.device_utils import get_safe_torch_device

from robot_stack.eval.pace_policy import attach_pace
from robot_stack.eval.sim_time import wrap_vector_env
from robot_stack.methods.pace.actuator import RobosuiteSpeedActuator
from robot_stack.methods.pace.processor import PaceSpeedStep
from robot_stack.methods.pace.speed import PaceConfig

logger = logging.getLogger(__name__)


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
    being carried by the unnormalizer step, so they are borrowed rather than
    re-derived.

    Returning None is normal, not a failure: xvla-libero normalizes actions with
    NormalizationMode.IDENTITY, so its actions are absolute already and its stats
    dict is empty. PACE's unnormalization is then a no-op, which is correct.
    """
    for step in getattr(postprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and "action" in stats:
            return stats
    return None


def build_pace(args, n_action_steps: int, stats: dict | None) -> PaceSpeedStep:
    cfg = PaceConfig(
        max_speed=args.max_speed,
        min_speed=args.max_speed / 2 if args.min_speed is None else args.min_speed,
        clamp_deg=args.clamp_deg,
        action_stride=args.action_stride,
        adaptive_stride=args.adaptive_stride,
        n_lookahead=args.n_lookahead,
        lookahead_agg=args.lookahead_agg,
        lookahead_target=args.lookahead_target,
        enable_angle=args.enable_angle,
        enable_ori=args.enable_ori,
        enable_ori_axis=args.enable_ori_axis,
        speed_quantize=args.speed_quantize,
        quantize_angle_thr=args.quantize_angle_thr,
    )
    return PaceSpeedStep(cfg, n_action_steps=n_action_steps, dataset_stats=stats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", default="lerobot/xvla-libero")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--tasks", default="0-9", help='e.g. "0-9", "2", "0-3,7"')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--n-action-steps", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default=None)

    g = p.add_argument_group("PACE speed selection")
    g.add_argument("--max-speed", type=float, default=1.0)
    g.add_argument("--min-speed", type=float, default=None, help="default: max_speed / 2")
    g.add_argument("--clamp-deg", type=float, default=5.0)
    g.add_argument("--action-stride", type=int, default=1)
    g.add_argument("--adaptive-stride", action="store_true")
    g.add_argument("--n-lookahead", type=int, default=0)
    g.add_argument("--lookahead-agg", default="min", choices=["min", "mean", "cumulative_bending"])
    g.add_argument("--lookahead-target", default="all", choices=["all", "angle", "ori", "ori_axis"])
    g.add_argument("--enable-angle", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--enable-ori", action=argparse.BooleanOptionalAction, default=True)
    # Note: the policy default is True, but every recorded ablation ran with the axis
    # channel off. Pass --enable-ori-axis to turn it back on.
    g.add_argument("--enable-ori-axis", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--speed-quantize", action="store_true")
    g.add_argument("--quantize-angle-thr", type=float, default=22.5)

    a = p.add_argument_group("PACE actuation")
    a.add_argument("--kpkd-scale-exp", type=float, default=2.0)
    a.add_argument(
        "--speed-rounding",
        default="up",
        choices=["up", "down"],
        help="'up' reproduces the recorded runs (max_speed can be exceeded by ~6%%); "
        "'down' makes max_speed a real ceiling, costing some throughput",
    )
    a.add_argument("--disable-kpkd-scaling", action="store_true")
    a.add_argument("--disable-gripper-speedup", action="store_true")
    a.add_argument(
        "--no-actuate",
        action="store_true",
        help="compute speeds and stride the chunk, but leave the simulator nominal",
    )

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.envs.configs import LiberoEnv  # noqa: PLC0415

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    if args.device:
        policy_cfg.device = args.device
    policy_cfg.n_action_steps = args.n_action_steps

    device = get_safe_torch_device(policy_cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy = make_policy(cfg=policy_cfg, env_cfg=LiberoEnv(task=args.task_suite))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    stats = action_stats(postprocessor)
    logger.info("PACE action stats: %s", "from unnormalizer" if stats else "none (identity normalization)")

    actuator = (
        None
        if args.no_actuate
        else RobosuiteSpeedActuator(
            kpkd_scale_exp=args.kpkd_scale_exp,
            disable_kpkd_scaling=args.disable_kpkd_scaling,
            disable_gripper_speedup=args.disable_gripper_speedup,
            action_stride=args.action_stride,
            speed_rounding=args.speed_rounding,
        )
    )
    paced = attach_pace(policy, build_pace(args, args.n_action_steps, stats), actuator)
    logger.info("PACE config: %s", paced.pace.get_config())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pace_config.json").write_text(
        json.dumps({"pace": paced.pace.get_config(), "args": vars(args)}, indent=2, default=str)
    )

    for task_id in parse_tasks(args.tasks):
        task_out = args.out / f"task_{task_id}"
        if (task_out / "eval_info.json").exists():
            logger.info("task %d already evaluated, skipping", task_id)
            continue

        set_seed(args.seed)
        env_cfg = LiberoEnv(task=args.task_suite, task_ids=[task_id], control_mode="absolute")
        envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)

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
                n_episodes=args.n_episodes,
                start_seed=args.seed,
            )
        close_envs(envs)

        # Fold in per-episode durations, which upstream's schema has no slot for.
        #
        # Only *successful* episodes are reliably captured: a success terminates the
        # env, whereas a failure usually runs to the step cap, and upstream ends the
        # rollout there without the env ever truncating. That is enough -- ATR
        # averages successes, and TPR needs failures only as a flag, which
        # `pc_success` already carries -- but it means the recorded success count is
        # worth checking against upstream's before either is trusted.
        episodes = [ep for rec in recorders for ep in rec.episodes]
        succeeded = [e["sim_time"] for e in episodes if e["success"] and e["sim_time"] is not None]

        expected = round(info["overall"]["pc_success"] / 100 * args.n_episodes)
        n_recorded = sum(1 for e in episodes if e["success"])
        if n_recorded != expected:
            logger.warning(
                "task %d: recorded %d successful episodes but SR implies %d -- "
                "ATR is computed over the recorded subset",
                task_id,
                n_recorded,
                expected,
            )

        task_out.mkdir(parents=True, exist_ok=True)
        info["episodes"] = episodes
        info["overall"]["avg_success_sim_s"] = sum(succeeded) / len(succeeded) if succeeded else None
        info["overall"]["n_success_timed"] = len(succeeded)

        # Speeds actually delivered to the simulator (env 0), post-quantization. The
        # mean is the single number that says whether PACE ran as fast as intended:
        # sim time scales as 1/speed, so a few percent here is a few percent of ATR.
        speeds = paced.pace_speed_log
        if speeds:
            info["overall"]["applied_speed_mean"] = sum(speeds) / len(speeds)
            info["overall"]["applied_speed_min"] = min(speeds)
            info["overall"]["applied_speed_max"] = max(speeds)
            info["overall"]["applied_speed_below_1"] = sum(1 for v in speeds if v < 1.0) / len(speeds)
            (task_out / "applied_speeds.json").parent.mkdir(parents=True, exist_ok=True)
            (task_out / "applied_speeds.json").write_text(json.dumps(speeds))
        paced.pace_speed_log.clear()

        (task_out / "eval_info.json").write_text(json.dumps(info, indent=2))
        logger.info(
            "task %d: SR %.1f%%  ATR %s s  (%d episodes recorded)",
            task_id,
            info["overall"]["pc_success"],
            f"{info['overall']['avg_success_sim_s']:.2f}" if succeeded else "n/a",
            len(episodes),
        )


if __name__ == "__main__":
    main()
