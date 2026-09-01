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


def attach_bspline(policy, decode: BSplineDecodeStep, actuator: BSplineTrackingActuator | None = None):
    """Make ``policy`` decode its own predictions. Returns it, modified in place.

    Args:
        policy: A policy exposing ``predict_action_chunk`` and ``reset``, trained to
            emit B-spline parameters.
        decode: Configured decode step. Its ``num_actions`` decides both how many
            actions each query yields and how fast the curve is traversed.
        actuator: Stiffens the simulator so it can track waypoints that are further
            apart. ``None`` still decodes and executes but leaves the plant nominal,
            which is the action-side-only ablation.
    """
    policy.bspline = decode
    policy.bspline_actuator = actuator
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
            actions, rates = self.bspline.decode_batch(parameters)
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
        original_reset()

    policy.bind_env = types.MethodType(bind_env, policy)
    policy.select_action = types.MethodType(select_action, policy)
    policy.reset = types.MethodType(reset, policy)
    return policy
