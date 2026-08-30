"""Translating a method into the steps the robot's deploy loop runs.

The split this file sits on: PACE decides *speeds*, crisp_gym *applies* them. The
decision layer is portable and lives here with the rest of the method; the actuation
layer is not portable -- it knows about controller cycles, DDS and a Robotiq gripper --
and lives in ``crisp_gym.deploy``. ``PaceSpeedStep``'s own docstring draws the same
line ("the step decides speeds; it does not apply them"), and exposes ``plan()``
separately from ``__call__`` precisely so a caller that is not driving a transition
pipeline -- "tests, the eval runner, the deploy loop" -- can use it.

So this module is deliberately thin. It is the only place where ``pace_bench`` and
``crisp_gym`` meet.
"""

from __future__ import annotations

import numpy as np
import torch

from crisp_gym.deploy.pipeline import Chunk, GripperHold, GripperReplicate, HeuristicSpeed
from pace_bench.methods.config import (
    BSplineMethod,
    DemoSpeedupMethod,
    MethodConfig,
    NoMethod,
    PaceMethod,
)
from pace_bench.methods.bspline.processor import BSplineDecodeStep
from pace_bench.methods.pace.processor import PaceSpeedStep


class PaceSpeed:
    """PACE's speed decision, as a deploy step.

    ``plan()`` may also *drop* rows -- PACE strides a chunk where the path is straight
    -- so the kept actions and their speeds are returned together and stay aligned by
    construction.
    """

    def __init__(self, config, *, n_action_steps: int | None = None,
                 control_dt: float | None = None, dataset_stats=None):
        self.step = PaceSpeedStep(
            config, n_action_steps=n_action_steps,
            dataset_stats=dataset_stats, control_dt=control_dt,
        )

    def __call__(self, chunk: Chunk) -> Chunk:
        # PaceSpeedStep is defined on (B, T, D); the deploy loop carries one chunk.
        t = torch.from_numpy(np.ascontiguousarray(chunk.actions)).unsqueeze(0)
        kept, speeds = self.step.plan(t)
        return Chunk(
            actions=kept[0].cpu().numpy(),
            speeds=speeds[0].cpu().numpy().astype(np.float64),
        )


class BSplineDecode:
    """Evaluate the predicted spline into executable actions -- the `decode` slot.

    B-spline is the method that proves the pipeline needed a decode stage at all: the
    policy emits a knot column beside control points, which no robot can execute, so
    something must turn parameters into a trajectory before anything else looks at it.
    Every other method leaves this slot as identity.

    ``num_actions`` is the speed lever and is a *decode-time* choice needing no
    retraining -- the curve covers a fixed stretch of demonstrated motion, so asking
    for fewer samples covers that stretch in fewer executed steps.
    """

    def __init__(self, step: BSplineDecodeStep):
        self.step = step

    def __call__(self, chunk: Chunk) -> Chunk:
        t = torch.from_numpy(np.ascontiguousarray(chunk.actions)).unsqueeze(0)
        actions, _rates = self.step.decode_batch(t)
        a = actions[0].cpu().numpy().astype(np.float64)
        # Decoding changes the row count, so speeds are rebuilt at nominal rather
        # than carried over: the speed-up lives in how far each decoded action
        # travels, exactly as it does for demospeedup.
        return Chunk(actions=a, speeds=np.ones(a.shape[0], dtype=np.float64))


class GripperCompensationUndecided(NotImplementedError):
    """Raised rather than silently running a method with no gripper compensation."""


def deploy_steps(method: MethodConfig, *, args, n_action_steps: int | None = None,
                 control_dt: float | None = None, dataset_stats=None) -> list:
    """The ordered steps a method contributes to the deploy loop.

    Order follows what 19_deploy_policy.py already did: shape rows, then decide
    speeds, then apply the gripper modifier. ``args`` carries the deploy CLI values
    the heuristic and the gripper window need.

    Raises on an unknown method rather than silently running a baseline -- deploying
    a demospeedup checkpoint as if it were unmodified is not a degraded run, it is a
    dropped object.
    """
    n_grip = int(getattr(args, "gripper_slowdown_frames", 0))
    invert = bool(getattr(args, "invert_gripper", False))

    if isinstance(method, NoMethod):
        # Bit-identical to the pre-method deploy path.
        return [HeuristicSpeed(args), GripperHold(n_grip, invert=invert)]

    if isinstance(method, PaceMethod):
        return [
            PaceSpeed(method.to_pace_config(), n_action_steps=n_action_steps,
                      control_dt=control_dt, dataset_stats=dataset_stats),
            GripperHold(n_grip, invert=invert),
        ]

    if isinstance(method, DemoSpeedupMethod):
        # No speed step: the speedup is in the weights, and retiming time as well
        # would apply it twice. The gripper is paid for in rows instead.
        return [GripperReplicate(int(method.low_v))]

    if isinstance(method, BSplineMethod):
        decode = BSplineDecode(BSplineDecodeStep(
            num_actions=int(getattr(args, "bspline_num_actions", 0)) or 16,
            degree=method.degree,
            relative_knots=method.relative_knots,
        ))
        # B-spline compresses *waypoints*, like demospeedup, so a speed pin buys the
        # gripper nothing -- speeds are already 1.0. It must be paid in rows, and the
        # factor is the realised bspline_rate (source frames advanced per executed
        # action), which varies per chunk rather than being a config constant. Until
        # that is decided, refuse rather than return a step list whose gripper entry
        # is a silent no-op: the arm would lift with the gripper still half closed.
        low_v = getattr(args, "bspline_gripper_low_v", None)
        if not low_v:
            raise GripperCompensationUndecided(
                "B-spline decodes fewer actions across the same motion, so the "
                "gripper gets proportionally less wall-clock to close -- but unlike "
                "demospeedup there is no low_v recorded to size the compensation, "
                "and the realised rate varies per chunk. Set "
                "--gripper.bspline_low_v explicitly (start from the decoded "
                "bspline_rate), or pass --gripper.slowdown_frames=0 to state that "
                "no compensation is wanted."
            )
        return [decode, GripperReplicate(int(low_v))]

    raise ValueError(
        f"no deploy steps defined for method {type(method).__name__!r} "
        f"(type={getattr(method, 'type', '?')}). Add them here rather than falling "
        "back to a baseline: a method whose steps are missing runs the policy with "
        "none of the compensation it was trained to require."
    )
