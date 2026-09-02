"""Attach B-spline decoding to a stock policy, without modifying it or its class.

A B-spline checkpoint does not predict actions. It predicts the parameters of a
curve, and something has to evaluate that curve before a robot can be driven by it --
so a policy that is otherwise perfectly ordinary cannot be handed to an evaluator
unchanged.

The join mirrors :func:`pace_bench.eval.pace_policy.attach_pace`, and for the same
reason: upstream's evaluator asserts ``isinstance(policy, PreTrainedPolicy)``, so a
wrapper object is refused. Rebinding ``select_action`` on one policy *object* leaves
it exactly what upstream demands.

What changes is only where actions come from. ``predict_action_chunk`` returns a
parameter matrix; this decodes it once per query into ``num_actions`` executable
actions and hands them out one at a time. The number of actions is the speed lever
and is chosen here, at decode time -- the same checkpoint runs at any speed.
"""

from __future__ import annotations

import types
from collections import deque

import torch

from pace_bench.methods.bspline.actuator import BSplineTrackingActuator
from pace_bench.methods.bspline.processor import BSplineDecodeStep


def _unnormalizer(stats: dict | None):
    """``(B, rows, channels)`` of normalized parameters -> natural units.

    None stats mean the policy's actions are absolute already -- which is every xVLA
    arm that keeps `NormalizationMode.IDENTITY` -- and then this is the identity.
    """
    if not stats or "mean" not in stats or "std" not in stats:
        return lambda parameters: parameters

    mean = torch.as_tensor(stats["mean"])
    std = torch.as_tensor(stats["std"])

    def unnormalize(parameters: torch.Tensor) -> torch.Tensor:
        m = mean.to(parameters.device, parameters.dtype)
        s = std.to(parameters.device, parameters.dtype)
        return parameters * s + m

    return unnormalize


def attach_bspline(
    policy,
    decode: BSplineDecodeStep,
    actuator: BSplineTrackingActuator | None = None,
    action_stats: dict | None = None,
):
    """Make ``policy`` decode its own predictions. Returns it, modified in place.

    Args:
        policy: A policy exposing ``predict_action_chunk`` and ``reset``, trained to
            emit B-spline parameters.
        decode: Configured decode step. Its ``num_actions`` decides both how many
            actions each query yields and how fast the curve is traversed.
        actuator: Stiffens the simulator so it can track waypoints that are further
            apart. ``None`` still decodes and executes but leaves the plant nominal,
            which is the action-side-only ablation.
        action_stats: The unnormalizer's ``action`` statistics, or None when the
            policy's actions are already absolute. **Required whenever the checkpoint
            normalizes its action**, because this bypasses the postprocessor -- see
            the note on `select_action`.
    """
    policy.bspline = decode
    policy.bspline_actuator = actuator
    # Undoing normalization is normally the postprocessor's job, and this decodes
    # *before* the postprocessor runs -- so if the checkpoint normalizes, the
    # parameters have to be restored here or the curve is built out of z-scores.
    policy.bspline_unnormalize = _unnormalizer(action_stats)
    policy.bspline_env = None
    policy.bspline_queue = deque()
    #: Realised rate per query -- source frames advanced per executed action. Varies
    #: with the span the policy predicted, so it is recorded rather than assumed.
    policy.bspline_rate_log = []
    # The eval runner writes `applied_speeds.json` and summarises it from
    # `pace_speed_log`. For B-spline the realised rate *is* the applied speed-up, so
    # the same list serves under both names rather than branching the artifact code.
    policy.pace_speed_log = policy.bspline_rate_log

    def bind_env(self, vec_env) -> None:
        """Point the actuator at the vector env this policy is about to drive.

        Needed for the same reason as PACE's: upstream's ``rollout`` gives the policy
        no access to the environment, so the caller owning both introduces them.
        """
        self.bspline_env = vec_env

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **_) -> torch.Tensor:
        if not self.bspline_queue:
            parameters = self.predict_action_chunk(batch)
            # `predict_action_chunk` returns the policy's raw output, which for a
            # checkpoint with MEAN_STD action normalization is in z-scores. The decode
            # step needs natural units -- its knot column is a time in source frames --
            # and it never sees the postprocessor that would have restored them,
            # because upstream's rollout applies that only to what `select_action`
            # RETURNS, by which point the curve has already been evaluated.
            #
            # Silent when wrong, and the way it is wrong is not obvious: normalized
            # knots average about zero, so every chunk spans ~0 frames, every decode
            # yields one action, and the arm crawls a step at a time while the run
            # reports a perfectly ordinary success rate of zero.
            parameters = self.bspline_unnormalize(parameters)
            # Sequential: row i of this batch is env i, one chunk after another on
            # its own curve -- so each row aligns to its own anchor. Without this the
            # step falls back to its batch-of-one rule and a vector env decodes every
            # chunk from the curve's beginning, jumping at each seam.
            actions, rates = self.bspline.decode_batch(parameters, sequential=True)
            # (B, T, ...) -> T entries of (B, ...), the queue upstream's rollout expects.
            self.bspline_queue.extend(actions.transpose(0, 1))
            self.bspline_rate_log.append(float(rates[0]))

        # Re-applied every step, not once at binding: a robosuite reset rebuilds the
        # robot and its controller, so a one-shot bump vanishes at the first episode
        # boundary.
        if self.bspline_actuator is not None and self.bspline_env is not None:
            for member in self.bspline_env.envs:
                self.bspline_actuator.apply(member)

        return self.bspline_queue.popleft()

    original_reset = policy.reset

    def reset(self) -> None:
        self.bspline_queue.clear()
        # The decode step remembers where the last chunk left the arm, to resume the
        # next curve there. Across an episode boundary that anchor points at the
        # previous episode's final pose, so it has to go with the queue.
        self.bspline.reset()
        original_reset()

    policy.bind_env = types.MethodType(bind_env, policy)
    policy.select_action = types.MethodType(select_action, policy)
    policy.reset = types.MethodType(reset, policy)
    return policy
