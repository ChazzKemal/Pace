"""PACE as a LeRobot pipeline step.

Placing PACE here rather than inside a policy is the whole point of the port. In the
fork it lived in ``XVLAPolicy.select_action``, which meant it existed once per policy
class and was copied by hand into the real-robot deploy script. As a registered
``ProcessorStep`` it is instead:

  * policy-agnostic  -- it sees an action chunk, not a network;
  * serialized       -- ``get_config`` lands in the checkpoint, so inference
                        reconstructs the exact configuration that was evaluated,
                        rather than depending on flags passed at deploy time;
  * shared           -- simulator and robot run the same object.

The step decides speeds; it does not apply them. Turning a speed into controller
gains or a timed action is the actuator's job, which is deliberately separate
because that part is *not* portable between robosuite and a real UR10e.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from robot_stack.methods.pace.speed import PaceConfig, compute_speeds, stride_indices, unnormalize_actions

# Key under which per-step speed multipliers are published in the transition's
# complementary data. Downstream actuators read this; nothing else writes it.
SPEED_KEY = "pace_speed"


@ProcessorStepRegistry.register("pace_speed")
class PaceSpeedStep(ProcessorStep):
    """Select execution speeds for an action chunk, and drop skipped steps.

    Input: ``transition[ACTION]`` shaped ``(B, T, action_dim)`` -- a chunk, not a
    single action, because every PACE channel is defined by differences between
    consecutive steps and lookahead needs the future ones.

    Output: the same transition with the chunk restricted to the kept steps and
    truncated to ``n_action_steps``, plus ``complementary_data[SPEED_KEY]`` holding
    one speed multiplier per delivered step.
    """

    def __init__(
        self,
        config: PaceConfig | dict | None = None,
        n_action_steps: int | None = None,
        dataset_stats: dict | None = None,
    ):
        """
        Args:
            config: PACE knobs. A plain dict is accepted so eval kwargs can be passed
                straight through.
            n_action_steps: How many steps of the chunk are actually executed before
                the policy is queried again. ``None`` delivers the whole chunk.
            dataset_stats: Only needed when this step runs *upstream* of the
                normalization step and therefore sees normalized actions. In the
                normal ordering (PACE last) leave it None.
        """
        if isinstance(config, dict):
            config = PaceConfig.from_dict(config)
        self.config = config or PaceConfig()
        self.n_action_steps = n_action_steps
        self.dataset_stats = dataset_stats

    def __call__(self, transition):
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        actions = new_transition.get(TransitionKey.ACTION)
        actions, speeds = self.plan(actions)

        new_transition[TransitionKey.ACTION] = actions
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary[SPEED_KEY] = speeds
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def plan(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk in, (kept chunk, per-step speeds) out. The step's whole behaviour.

        Exposed separately from ``__call__`` so callers that are not driving a
        transition pipeline -- tests, the eval runner, the deploy loop -- can use
        PACE without constructing one. The shape check lives here rather than in
        ``__call__`` for that reason: it guards every route in.
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"PaceSpeedStep expects a tensor action chunk, got {type(actions)}")
        if actions.ndim != 3:
            raise ValueError(
                f"PaceSpeedStep expects a chunk shaped (batch, steps, action_dim), got {tuple(actions.shape)}. "
                "PACE is defined on consecutive steps, so it cannot run on a single action."
            )

        abs_actions = unnormalize_actions(actions, self.dataset_stats)

        keep = stride_indices(abs_actions, self.config)
        if len(keep) != actions.shape[1]:
            actions = actions[:, keep, :]
            abs_actions = abs_actions[:, keep, :]

        speeds = compute_speeds(abs_actions, self.config)

        end = actions.shape[1] if self.n_action_steps is None else min(self.n_action_steps, actions.shape[1])
        return actions[:, :end], speeds[:, :end]

    def get_config(self) -> dict[str, Any]:
        """Serialized into the checkpoint, so deploy inherits the evaluated config."""
        return {**self.config.to_dict(), "n_action_steps": self.n_action_steps}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Unchanged: striding removes steps from a chunk, not dimensions from an action."""
        return features
