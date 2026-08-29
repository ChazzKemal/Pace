"""PACE: eval-time speed modulation for action-chunking policies.

Speed up where the trajectory is smooth, brake where it bends. No retraining.
"""

from pace_bench.methods.pace.processor import SPEED_KEY, PaceSpeedStep
from pace_bench.methods.pace.speed import (
    PaceConfig,
    apply_lookahead,
    compute_speeds,
    per_step_angle,
    per_step_orientation_angle,
    speed_from_angle,
    speed_from_orientation,
    speed_from_orientation_angle,
    stride_indices,
    unnormalize_actions,
)

__all__ = [
    "SPEED_KEY",
    "PaceConfig",
    "PaceSpeedStep",
    "apply_lookahead",
    "compute_speeds",
    "per_step_angle",
    "per_step_orientation_angle",
    "speed_from_angle",
    "speed_from_orientation",
    "speed_from_orientation_angle",
    "stride_indices",
    "unnormalize_actions",
]
