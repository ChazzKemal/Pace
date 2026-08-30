"""Reading what a checkpoint says about itself, without importing LeRobot.

``--method.type`` is the source of truth for what a deploy run does. This module
exists to *check* that claim against the checkpoint and to supply the one value the
operator should not have to retype (``low_v``).

Everything is read as raw JSON on purpose. ``TrainPipelineConfig.from_pretrained``
would reject the file outright -- ``method`` is not a field it knows -- and parsing it
as ``SpeedupTrainConfig`` would drag in validation of training-only paths that do not
exist on the robot (dataset roots on the training host, wandb settings). crisp_gym
learned the same lesson: see the comment at async_lerobot_policy.py:261.

Three signals, in increasing order of authority:

1. ``train_config.json`` -> ``method``      -- what was *asked for*
2. policy ``config.json`` chunk geometry    -- whether adjust_policy actually halved
3. the serialized preprocessor config       -- what was *built*, and carries low_v

(3) is authoritative because it records the pipeline that actually ran. A wandb run
config showing ``chunk_size: 100`` with ``source_chunk: null`` is the *pre*-adjustment
view -- ``source_chunk`` is set by ``adjust_policy`` -- so it is not evidence that
halving did not happen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAIN_CONFIG = "train_config.json"
POLICY_CONFIG = "config.json"
PREPROCESSOR = "policy_preprocessor.json"

#: Registered name of demospeedup's training-time step, as serialized by LeRobot.
DEMOSPEEDUP_STEP = "demospeedup_retime"
PACE_STEP = "pace_speed"


@dataclass
class CheckpointFacts:
    """What a checkpoint directory reveals about how it was trained."""

    path: Path
    method_type: str | None = None      # from train_config.json
    low_v: int | None = None            # preferred from the preprocessor config
    high_v: int | None = None
    halve_chunk: bool | None = None
    source_chunk: int | None = None     # chunk size *before* halving, if recorded
    chunk_size: int | None = None       # the policy's actual (possibly halved) chunk
    n_action_steps: int | None = None
    policy_type: str | None = None
    built_steps: tuple[str, ...] = ()   # registered step names found in the pipeline

    @property
    def halving_applied(self) -> bool | None:
        """True when the policy's chunk is half what training started from."""
        if self.chunk_size is None or self.source_chunk is None:
            return None
        return self.chunk_size * 2 == self.source_chunk


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


#: LeRobot serializes each pipeline step as {"registry_name": ..., "config": {...}}.
#: Verified against a real checkpoint rather than assumed -- the plausible-looking
#: "registered_name" finds nothing and would silently report an unmodified baseline.
STEP_NAME_KEY = "registry_name"


def _registered_names(obj: Any, out: list[str]) -> None:
    """Collect every step's registry name from a serialized processor pipeline."""
    if isinstance(obj, dict):
        n = obj.get(STEP_NAME_KEY)
        if isinstance(n, str):
            out.append(n)
        for v in obj.values():
            _registered_names(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _registered_names(v, out)


def read_checkpoint(path: str | Path) -> CheckpointFacts:
    """Gather what the checkpoint says, tolerating any file being absent."""
    p = Path(path)
    f = CheckpointFacts(path=p)

    train = _load(p / TRAIN_CONFIG)
    method = train.get("method") or {}
    if isinstance(method, dict):
        f.method_type = method.get("type")
        f.high_v = method.get("high_v")
        f.halve_chunk = method.get("halve_chunk")
        f.source_chunk = method.get("source_chunk")
        if method.get("low_v") is not None:
            f.low_v = int(method["low_v"])

    policy = _load(p / POLICY_CONFIG)
    f.policy_type = policy.get("type")
    f.chunk_size = policy.get("chunk_size", policy.get("horizon"))
    f.n_action_steps = policy.get("n_action_steps")

    names: list[str] = []
    _registered_names(_load(p / PREPROCESSOR), names)
    f.built_steps = tuple(names)

    # The preprocessor config records what was *built*; prefer its low_v.
    if DEMOSPEEDUP_STEP in names:
        pre = _load(p / PREPROCESSOR)
        found: list[dict] = []
        def walk(o):
            if isinstance(o, dict):
                if o.get(STEP_NAME_KEY) == DEMOSPEEDUP_STEP:
                    found.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(pre)
        for blk in found:
            cfg = blk.get("config", blk)
            if isinstance(cfg, dict) and cfg.get("low_v") is not None:
                f.low_v = int(cfg["low_v"])
                break
    return f


class MethodMismatch(RuntimeError):
    """The requested method contradicts what the checkpoint was trained as."""


def validate_method(declared: str, facts: CheckpointFacts, *, force: bool = False) -> None:
    """Refuse a launch whose method contradicts the checkpoint.

    Both conflicts present as *the robot working badly* rather than as a
    misconfiguration, which is why they are errors rather than warnings:

    * ``none`` on a demospeedup checkpoint runs a policy whose waypoints sit 2-4x
      further apart with none of the gripper compensation -- the gripper is still
      half-open when the arm lifts, and the object drops.
    * ``pace`` on one multiplies PACE's speed-up onto a demonstration already
      compressed in the weights, well past anything that was evaluated.
    """
    trained = facts.method_type
    if trained == declared:
        return

    # Asking for a *training-time* method on a checkpoint that was not trained with
    # it. demospeedup's compensation only makes sense against weights trained on
    # retimed targets: applied to a baseline it slows the gripper for no reason and
    # yields a benchmark number that means nothing. `pace` is exempt -- it is an
    # eval-time choice and applies to any checkpoint.
    if declared == "demospeedup" and trained != "demospeedup":
        msg = (
            f"--method.type=demospeedup was requested, but the checkpoint at "
            f"{facts.path} was trained as {trained or 'a plain baseline (no method '
            'recorded)'}. Its waypoints are not retimed, so the gripper "
            "compensation would slow the grasp for no reason and the run would not "
            "be comparable to a real demospeedup arm."
        )
        if not force:
            raise MethodMismatch(msg + " Pass --force to override.")
        return

    if trained is None:
        return
    if trained == "demospeedup" and declared in ("none", "pace"):
        msg = (
            f"checkpoint at {facts.path} was trained with method 'demospeedup' "
            f"(low_v={facts.low_v}), but --method.type={declared} was requested. "
        )
        msg += (
            "Running it as 'none' drops the gripper compensation its retimed "
            "waypoints require."
            if declared == "none" else
            "PACE would multiply its speed-up onto a demonstration already "
            "compressed in the weights."
        )
        if not force:
            raise MethodMismatch(msg + " Pass --force to override.")
    return
