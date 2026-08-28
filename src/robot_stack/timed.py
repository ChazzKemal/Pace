"""TimedActions: an action stream that says when each action happens.

Every consumer of an action stream in this stack -- the robosuite actuator, the
UR10e's interpolating controller, replay tooling, throughput metrics -- has until
now assumed one action per ``1/fps`` tick. Each method in scope breaks that
assumption in its own way:

* **PACE** allots each action ``control_dt / speed_t`` seconds, per step.
* **B-spline** samples a continuous curve, so spacing is whatever the sampler chose.
* **DemoSpeedup** is the instructive non-example: its executed stream *is* uniform
  ``1/fps`` -- the acceleration lives in the weights -- so it needs no declaration.
  That only works because uniformity is stated somewhere, which is this contract's
  job.

``dt[i]`` is the time allotted to action ``i``: the interval between issuing action
``i`` and action ``i+1``, in seconds. A scalar ``dt`` means uniform spacing. The
baseline -- ``TimedActions.uniform(actions, fps)`` -- is by construction a no-op
description of what every consumer already does.

Inside a processor pipeline the same information travels as a tensor under
``DT_KEY`` in complementary data (pipelines carry tensors, not dataclasses);
``PaceSpeedStep`` publishes it when it knows the control period. This class is the
contract at code boundaries: runner to actuator, planner to deploy loop.
"""

from dataclasses import dataclass

import torch

# Complementary-data key for per-step dt, seconds, shaped like the leading dims of
# the action stream it accompanies. Published by method steps; read by actuators.
DT_KEY = "action_dt"


@dataclass(frozen=True)
class TimedActions:
    """An action sequence plus the time each action is allotted.

    Args:
        actions: ``(T, action_dim)`` or ``(B, T, action_dim)``.
        dt: Seconds per action. A float for uniform spacing, or a tensor shaped
            like ``actions`` minus its last dimension for per-step spacing.
    """

    actions: torch.Tensor
    dt: torch.Tensor | float

    def __post_init__(self):
        if self.actions.ndim not in (2, 3):
            raise ValueError(f"actions must be (T, D) or (B, T, D), got {tuple(self.actions.shape)}")
        if isinstance(self.dt, torch.Tensor):
            if self.dt.shape != self.actions.shape[:-1]:
                raise ValueError(
                    f"per-step dt must be shaped {tuple(self.actions.shape[:-1])} "
                    f"(actions minus the action dim), got {tuple(self.dt.shape)}"
                )
            if not torch.all(self.dt > 0):
                raise ValueError("every dt must be positive")
        elif self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

    # -- constructors ---------------------------------------------------------

    @classmethod
    def uniform(cls, actions: torch.Tensor, fps: float) -> "TimedActions":
        """The baseline: one action per ``1/fps``. Describes today's behaviour exactly."""
        return cls(actions, 1.0 / fps)

    @classmethod
    def from_speeds(cls, actions: torch.Tensor, speeds: torch.Tensor, control_dt: float) -> "TimedActions":
        """PACE's view: a speed multiplier per action, against a nominal period.

        ``speed=1`` gives the uniform baseline; ``speed=2`` halves the allotted time.
        """
        return cls(actions, control_dt / speeds)

    # -- derived views --------------------------------------------------------

    @property
    def is_uniform(self) -> bool:
        """True when every step is allotted the same time (scalar dt included)."""
        if not isinstance(self.dt, torch.Tensor):
            return True
        flat = self.dt.reshape(-1)
        return bool(torch.all(flat == flat[0]))

    def per_step_dt(self) -> torch.Tensor:
        """dt as a tensor shaped like the actions' leading dims, scalar expanded."""
        if isinstance(self.dt, torch.Tensor):
            return self.dt
        return torch.full(self.actions.shape[:-1], self.dt, dtype=torch.float64)

    def speeds(self, control_dt: float) -> torch.Tensor:
        """The inverse of :meth:`from_speeds`, for backends whose API is a multiplier."""
        return control_dt / self.per_step_dt()

    def timestamps(self) -> torch.Tensor:
        """Issue time of each action, from 0: ``t[i] = sum(dt[:i])``.

        This is what a time-parameterised trajectory controller consumes: the UR10e
        follows ``(pose, t)`` pairs, not a rate.
        """
        dt = self.per_step_dt()
        t = torch.cumsum(dt, dim=-1)
        return torch.cat([torch.zeros_like(t[..., :1]), t[..., :-1]], dim=-1)

    def duration(self) -> torch.Tensor:
        """Total time the stream occupies: what throughput metrics divide by."""
        return self.per_step_dt().sum(dim=-1)
