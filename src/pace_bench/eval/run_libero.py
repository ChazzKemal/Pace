"""Evaluate a policy on LIBERO, one task per output directory.

Mirrors upstream's ``eval_main`` -- same env factory, policy loader, processors,
``eval_policy_all`` and ``eval_info.json`` schema -- with two additions: the method's
pipeline steps are attached to the policy, and the policy is introduced to the vector
env so its actuator can reach the simulator.

One task per output directory, matching the layout of the recorded results this is
compared against.

    python -m pace_bench.eval.run_libero --out outputs/baseline           # no method

    python -m pace_bench.eval.run_libero --out outputs/pace \\
        --method.type=pace --method.max_speed=1.5 --method.action_stride=2 \\
        --method.n_lookahead=4 --method.lookahead_agg=cumulative_bending \\
        --method.lookahead_target=angle
"""

import json
import logging
import os
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

from pace_bench.eval.bspline_policy import attach_bspline
from pace_bench.eval.pace_policy import attach_pace
from pace_bench.eval.sim_time import wrap_vector_env
from pace_bench.methods.bspline.actuator import BSplineTrackingActuator
from pace_bench.methods.config import (
    BSplineMethod,
    DemoSpeedupMethod,
    MethodPipelineConfig,
    NoMethod,
)
from pace_bench.methods.demospeedup.actuator import DemoSpeedupTrackingActuator
from pace_bench.methods.pace.actuator import DEFAULT_CONTROL_DT, RobosuiteSpeedActuator
from pace_bench.methods.pace.processor import PaceSpeedStep

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
    # Set when policy_path is a PEFT adapter directory rather than a full checkpoint.
    # The adapter's config names the base model, so nothing else has to be supplied.
    use_peft: bool = False
    task_suite: str = "libero_10"
    tasks: str = "0-9"  # "0-9", "2", "0-3,7"
    seed: int = 42
    n_episodes: int = 50
    batch_size: int = 10
    # How much of each chunk is executed before the policy is queried again. A
    # property of the policy rather than of any method, so it lives here.
    n_action_steps: int = 32
    device: str | None = None
    # Steps an episode may take before it is cut off. None keeps LIBERO's own cap
    # (520 for these tasks). It has to be settable because the cap is counted in
    # *executed actions*, not in demonstrated motion: a method that executes the same
    # trajectory more finely spends more steps covering it, so leaving the cap fixed
    # would score slow execution as failure whatever the control quality.
    episode_length: int | None = None
    # Episodes per task to render to mp4, written under `<out>/task_<id>/videos/`.
    # 0 (the default) renders nothing, which is what a scoring run wants: rendering
    # forces every frame through the offscreen renderer and roughly doubles the
    # wall clock, and a 200-episode sweep does not need 200 videos. Raise it when a
    # number needs explaining rather than reporting -- a 0% arm is a trajectory
    # question, and the score alone cannot say whether the arm freezes, drifts,
    # overshoots, or thrashes.
    max_episodes_rendered: int = 0
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
    NormalizationMode.IDENTITY, so its actions are absolute already. The check is
    on the norm *map*, not on whether a stats dict exists -- a checkpoint saved by
    training carries the dataset stats even for IDENTITY features, where the
    unnormalizer never applies them; PACE must not apply them either.
    """
    from lerobot.configs.types import FeatureType, NormalizationMode

    for step in getattr(postprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if not (stats and "action" in stats):
            continue
        norm_map = getattr(step, "norm_map", None) or {}
        if norm_map.get(FeatureType.ACTION) == NormalizationMode.IDENTITY:
            return None
        return stats
    return None


def drop_steps(pipeline, name_fragment: str) -> list:
    """Remove every step of ``pipeline`` whose class name contains ``name_fragment``.

    Twice now a checkpoint has carried a step that the eval path also supplies, and
    running both is never merely redundant: the ImageNet step would normalize an
    already-normalized image, and a second B-spline decode would read a decoded action
    as though it were a curve. Keep exactly one application, and say which one went.

    Returns the removed steps, so the caller can report what it dropped.
    """
    dropped = [step for step in pipeline.steps if name_fragment in type(step).__name__]
    if dropped:
        pipeline.steps = [step for step in pipeline.steps if step not in dropped]
    return dropped


def strip_parameter_unnormalizer(postprocessor, stats: dict | None) -> list:
    """Drop the checkpoint's unnormalizer once B-spline decoding has consumed it.

    `attach_bspline` restores the parameters to natural units *before* evaluating the
    curve, because the knot column has to be a time in frames for the decode to mean
    anything. Upstream's rollout then hands whatever `select_action` returns to the
    checkpoint's postprocessor -- whose unnormalizer holds the statistics of the
    *parameter matrix*, a 20-wide knot-first mean/std, and applies them to a 20-wide
    ee6d action. Column 0 of those statistics is the knot (mean 16.95, std 19.3), so
    the arm's x target became ``16.95 + 19.3 * x``: every command landed ~15 m away and
    the arm was driven off the table before the second query.

    Invisible on an IDENTITY checkpoint, where the step is the identity -- which is
    every xVLA arm but `--method.normalize_parameters=true`. Returns what was dropped.
    """
    if not stats:
        return []
    return drop_steps(postprocessor, "Unnormalizer")


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
    # With the control period known, the step also publishes per-step dt (DT_KEY),
    # the TimedActions view of the same decision.
    step.control_dt = DEFAULT_CONTROL_DT
    return step


@draccus.wrap()
def main(cfg: LiberoEvalConfig) -> None:
    # force=True because robosuite installs a root handler at import time, and a
    # plain basicConfig is a no-op once the root logger has one -- which silently
    # swallowed every INFO this module emits, including "restored a trained pos_emb".
    # An eval whose provenance lines are invisible is an eval you cannot check.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    # `from_pretrained` reads `action_mode` straight out of the checkpoint's config,
    # and a B-spline xVLA checkpoint names an action space only this repo defines.
    # `BSplineMethod.__post_init__` has already registered it by the time draccus
    # hands the config over -- see the note there.
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.use_peft = cfg.use_peft
    policy_cfg.n_action_steps = cfg.n_action_steps
    if cfg.device:
        policy_cfg.device = cfg.device

    device = get_safe_torch_device(policy_cfg.device, log=True)
    # cuDNN's autotuner probes convolution algorithms by allocating workspaces, and on
    # a card that is already busy -- an eval running beside a training job -- that
    # probe *segfaults* inside the conv instead of raising, taking the run with it.
    # On by default because it is worth real throughput when the card is ours alone;
    # set PACE_CUDNN_BENCHMARK=0 to share.
    torch.backends.cudnn.benchmark = os.environ.get("PACE_CUDNN_BENCHMARK", "1") != "0"
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
    if actuator is not None and isinstance(cfg.method, DemoSpeedupMethod):
        # DemoSpeedup's actuation is its own (methods/demospeedup/actuator.py):
        # constant tracking stiffening at low_v, time untouched. It shares only the
        # per-step apply() duck-type with PACE's actuator.
        actuator = DemoSpeedupTrackingActuator(
            low_v=cfg.method.low_v,
            kpkd_scale_exp=cfg.actuation.kpkd_scale_exp,
            disable_kpkd_scaling=cfg.actuation.disable_kpkd_scaling,
            disable_gripper_speedup=cfg.actuation.disable_gripper_speedup,
        )

    if isinstance(cfg.method, BSplineMethod):
        # A checkpoint trained with `--method.unfreeze_pos_emb_rows` carries its learned
        # positional embedding in a file beside the adapter, because a PEFT adapter
        # cannot hold a bare nn.Parameter. Without this the evaluation would silently
        # run the *pretrained* table and read as a null result for the one thing that
        # arm changed.
        from pace_bench.methods.bspline.pos_emb import restore as restore_pos_emb

        if restore_pos_emb(policy, cfg.policy_path):
            logger.info("restored a trained pos_emb from %s", cfg.policy_path)

        # B-spline predicts curve parameters, not actions, so the policy has to decode
        # before anything can be executed. `num_actions` is the speed lever and is
        # chosen here rather than baked into the checkpoint.
        (decode,) = cfg.method.postprocessor_steps()
        # A B-spline checkpoint saves its own `bspline_decode` into the postprocessor,
        # and `attach_bspline` below decodes inside `select_action` -- leaving both in
        # place decodes twice, the second time handing a single decoded action to a
        # step that expects a parameter matrix. The attached one wins: `num_actions` is
        # the speed lever and belongs to this run, not to the checkpoint, which baked
        # in whatever it trained at.
        for step in drop_steps(postprocessor, "BSplineDecodeStep"):
            logger.info(
                "dropped checkpoint-side bspline_decode (num_actions=%s); this run "
                "decodes at num_actions=%s",
                step.num_actions, decode.num_actions,
            )

        # Its actuation is upstream's: a constant arm-kp multiple, kd and gripper left
        # nominal. Not PACE's per-step law and not DemoSpeedup's low_v scaling.
        paced = attach_bspline(
            policy,
            decode,
            BSplineTrackingActuator(
                kp_scale=cfg.method.stiffness_kp_scale,
                disable_kp_scaling=cfg.actuation.disable_kpkd_scaling,
            )
            if actuate
            else None,
            # Decoding happens inside `select_action`, upstream of the postprocessor
            # that would undo normalization -- so the statistics have to come with it.
            # None here for every IDENTITY checkpoint, which is what `action_stats`
            # returns for them, and the unnormalizer is then a no-op.
            action_stats=None if stats is None else stats.get("action"),
        )
        for step in strip_parameter_unnormalizer(postprocessor, stats):
            logger.info(
                "dropped checkpoint-side %s: the parameters are unnormalized before "
                "decode, so the decoded actions are already in natural units",
                type(step).__name__,
            )
        logger.info(
            "method=%s | %s | kp x%.2f | parameters %s",
            cfg.method.type, decode.get_config(), cfg.method.stiffness_kp_scale,
            "unnormalized before decode" if stats else "already absolute",
        )
    else:
        paced = attach_pace(policy, build_speed_step(cfg, stats), actuator)
        logger.info("method=%s | %s", cfg.method.type, paced.pace.get_config())

    cfg.out.mkdir(parents=True, exist_ok=True)
    # draccus.dump emits YAML -- name the file accordingly. (Its .json-named
    # predecessor cost a debugging session: a json.load probe declared it empty.)
    with open(cfg.out / "run_config.yaml", "w") as f:
        # Encoded against LiberoEvalConfig so the method's choice key is included,
        # which is what makes the file re-parsable with --config_path.
        draccus.dump(cfg, f)

    for task_id in parse_tasks(cfg.tasks):
        task_out = cfg.out / f"task_{task_id}"
        if (task_out / "eval_info.json").exists():
            logger.info("task %d already evaluated, skipping", task_id)
            continue

        set_seed(cfg.seed)
        env_cfg = LiberoEnv(
            task=cfg.task_suite,
            task_ids=[task_id],
            control_mode="absolute",
            **({"episode_length": cfg.episode_length} if cfg.episode_length else {}),
        )
        envs = make_env(env_cfg, n_envs=cfg.batch_size, use_async_envs=False)

        # One task in, one vector env out -- so binding is unambiguous.
        (vec_env,) = (v for group in envs.values() for v in group.values())
        recorders = wrap_vector_env(vec_env)
        paced.bind_env(vec_env)

        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        # The env pipeline for xVLA+LIBERO ImageNet-normalizes images itself. The hub
        # checkpoint's own preprocessor does NOT (which is why training from it needs
        # the ImageNet-patched base) -- but a checkpoint SAVED from that patched
        # lineage carries the step, and running both would normalize twice; the
        # step's own guard rejects that. Keep exactly one application: the env's.
        if any("ImageNet" in type(step).__name__ for step in env_pre.steps):
            if drop_steps(preprocessor, "ImageNet"):
                logger.info("dropped checkpoint-side ImageNet step (env pipeline provides it)")
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
                max_episodes_rendered=cfg.max_episodes_rendered,
                videos_dir=task_out / "videos",
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
    # Sorted by upstream's episode numbering, and truncated the way upstream truncates
    # its own metrics: a batch runs a full n_envs episodes even when the last few are
    # past `n_episodes`, and those extras are not part of the reported run.
    episodes = [ep for rec in recorders for ep in rec.episodes if ep["episode_index"] < cfg.n_episodes]
    episodes.sort(key=lambda e: e["episode_index"])
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
