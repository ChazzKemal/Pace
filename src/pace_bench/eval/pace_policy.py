"""Attach PACE to a stock policy, without modifying it or its class.

In the fork, PACE lived inside ``XVLAPolicy.select_action``, and the eval script
reached into the simulator after every step to actuate it. Both are avoided here:
the policy is upstream LeRobot's, unmodified, and upstream's ``rollout`` /
``eval_policy_all`` are reused untouched.

The join is :func:`attach_pace`, which adds the behaviour to *one policy object*
rather than to its class. A wrapper object would have been tidier, but upstream's
evaluator asserts ``isinstance(policy, PreTrainedPolicy)``, and something that merely
quacks like a policy is refused -- reasonably, since half of LeRobot expects a real
``nn.Module``. Attaching keeps the object exactly what upstream demands while
overriding the two methods PACE needs, and touches no other policy instance.

The consequence worth stating: adding PACE to a policy no longer requires editing
that policy. Any policy exposing ``predict_action_chunk`` can be paced.
"""

from __future__ import annotations

import types
from collections import deque

import torch

from pace_bench.methods.pace.actuator import SpeedActuator
from pace_bench.methods.pace.processor import PaceSpeedStep


def attach_pace(policy, pace: PaceSpeedStep, actuator: SpeedActuator | None = None):
    """Make ``policy`` pace itself. Returns the same object, modified in place.

    ``select_action`` becomes: pull a whole chunk, decide speeds across it, then hand
    out one action per call, actuating the simulator just before each is returned --
    the same instant the fork actuated at, since nothing between here and
    ``env.step`` touches the robot.

    Args:
        policy: A policy exposing ``predict_action_chunk`` and ``reset``.
        pace: Configured speed step. Its ``n_action_steps`` decides how much of each
            chunk is consumed before the policy is queried again.
        actuator: Backend that realises a speed. ``None`` still computes speeds and
            strides the chunk but leaves the simulator nominal, which is exactly the
            "action-side only" ablation.
    """
    policy.pace = pace
    policy.pace_actuator = actuator
    policy.pace_env = None
    policy.pace_queue = deque()
    policy.pace_speed_log = []

    def bind_env(self, vec_env) -> None:
        """Point the actuator at the vector env this policy is about to drive.

        Needed because upstream's ``rollout`` gives the policy no access to the
        environment, so the caller that owns both has to introduce them. Evaluating
        one task per env keeps this unambiguous.
        """
        self.pace_env = vec_env

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **_) -> torch.Tensor:
        if not self.pace_queue:
            chunk = self.predict_action_chunk(batch)
            actions, speeds = self.pace.plan(chunk)
            # (B, T, ...) -> T entries of (B, ...), matching the queue the fork kept.
            self.pace_queue.extend(zip(actions.transpose(0, 1), speeds.transpose(0, 1), strict=True))

        action, speed = self.pace_queue.popleft()

        if self.pace_actuator is not None and self.pace_env is not None:
            for i, member in enumerate(self.pace_env.envs):
                applied = self.pace_actuator.apply(member, float(speed[i]))
                if i == 0:
                    self.pace_speed_log.append(applied)

        return action

    original_reset = policy.reset

    def reset(self) -> None:
        self.pace_queue.clear()
        original_reset()

    policy.bind_env = types.MethodType(bind_env, policy)
    policy.select_action = types.MethodType(select_action, policy)
    policy.reset = types.MethodType(reset, policy)
    return policy
