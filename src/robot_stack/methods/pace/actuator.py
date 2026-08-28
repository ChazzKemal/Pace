"""PACE actuation: turning a chosen speed into something the hardware does.

The other half of PACE. :mod:`speed` decides *how fast* each step should run; this
decides *what that means* for a particular machine, and the two are separated
because only the first half is portable.

Executing an action faster is not a matter of scaling the action. The action is a
target pose; reaching it sooner means the controller has less time to close the gap,
so the position error at each instant is larger and the tracking gains have to rise
to compensate. In simulation that is three coupled adjustments -- fewer substeps,
higher kp, higher kd -- and on a real arm it is a different mechanism entirely
(controller parameters over ROS, at a cadence the network can sustain).

Hence :class:`SpeedActuator` as an interface with one simulator implementation here.
The real-robot implementation lands with the robot side of the port.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

# Robosuite drives the MuJoCo model at 500 Hz and accepts a new action at 20 Hz, so
# one action is normally exhausted over 25 physics substeps.
DEFAULT_CONTROL_DT = 0.05
DEFAULT_MODEL_DT = 0.002


class SpeedActuator(ABC):
    """Applies one PACE speed to one robot for the coming control step."""

    @abstractmethod
    def apply(self, handle, speed: float) -> float:
        """Realise ``speed`` on ``handle``.

        Returns the speed that was *actually* applied, which may differ from the one
        requested -- see :meth:`RobosuiteSpeedActuator.quantize`.
        """
        ...


class RobosuiteSpeedActuator(SpeedActuator):
    """Speed via substep count and OSC gain compensation, for robosuite/LIBERO.

    Three things move together:

    * **substeps** -- ``set_action_exhaust_speed`` makes the environment exhaust the
      action over fewer, larger physics substeps. This is what actually saves time.
    * **gains** -- kp rises as ``speed ** exp`` and kd as ``speed ** (exp/2)``. The
      exponents are not free: for a second-order system, critical damping needs
      ``kd = 2*sqrt(kp)``, so halving the kp exponent for kd is what keeps the arm
      critically damped instead of ringing as it is driven harder. ``exp = 2``
      corresponds to holding tracking error constant as speed rises.
    * **gripper** -- opens and closes faster, so the finger stroke still completes
      inside the shortened step.

    Requires the patched robosuite pinned in ``pyproject.toml``; stock 1.4.0 exposes
    none of these hooks.
    """

    def __init__(
        self,
        control_dt: float = DEFAULT_CONTROL_DT,
        model_dt: float = DEFAULT_MODEL_DT,
        kpkd_scale_exp: float = 2.0,
        disable_kpkd_scaling: bool = False,
        disable_gripper_speedup: bool = False,
        action_stride: int = 1,
        speed_rounding: str = "up",
    ):
        """
        Args:
            kpkd_scale_exp: Exponent on speed for kp (kd gets half of it). 2.0 is the
                error-preserving value; lowering it trades tracking for gentler
                torques, which is the axis the no-gain-bump ablation sweeps.
            disable_kpkd_scaling: Run the faster action stream on nominal gains --
                isolates the action-side speedup from the gain bump.
            disable_gripper_speedup: Leave the gripper at its nominal stroke rate.
            action_stride: PACE's stride, which also multiplies the gripper rate: a
                strided chunk means each delivered action must cover several
                original ones, the gripper's included.
            speed_rounding: Which way to break the substep grid (see
                :meth:`quantize`). ``"up"`` reproduces the recorded experiments and
                lets the delivered speed exceed ``max_speed``; ``"down"`` makes
                ``max_speed`` a true ceiling at the cost of some throughput.
        """
        if speed_rounding not in ("up", "down"):
            raise ValueError(f"speed_rounding must be 'up' or 'down', got {speed_rounding!r}")
        self.control_dt = control_dt
        self.steps_ideal = control_dt / model_dt
        self.kpkd_scale_exp = kpkd_scale_exp
        self.disable_kpkd_scaling = disable_kpkd_scaling
        self.disable_gripper_speedup = disable_gripper_speedup
        self.action_stride = action_stride
        self.speed_rounding = speed_rounding

    def apply_dt(self, handle, dt: float) -> float:
        """Realise a per-action time budget instead of a speed multiplier.

        The TimedActions-facing entry: ``dt`` seconds per action against the nominal
        ``control_dt``. Delegates to :meth:`apply`; returns the *realised* dt, which
        differs from the request by the substep quantization exactly as realised
        speed differs from requested speed.
        """
        return self.control_dt / self.apply(handle, self.control_dt / dt)

    def quantize(self, speed: float) -> float:
        """Round a requested speed to one the simulator can actually deliver.

        Substeps are whole physics ticks, so the achievable speeds are ``25/n`` for
        integer ``n``: ..., 1.0, 1.09, 1.19, 1.32, 1.47, 1.5625, 1.67, ... A request
        almost never lands on a rung, and which way it breaks matters:

        ``"up"`` (default) truncates the substep count, so the speed rounds **up**:
        1.5 executes at 1.5625 (25/16), 3.0 at 3.125, 4.0 at 4.167. This is what the
        recorded experiments did, so it is the default -- but note the consequence:
        ``max_speed`` is not actually a ceiling, and is exceeded by up to ~6%.

        ``"down"`` rounds the speed down instead, making ``max_speed`` a real upper
        bound. The cost is throughput, and it is worst where the grid is coarse: a
        1.5 request drops to 1.47, and anything under 1.04 collapses to exactly 1.0,
        i.e. no speedup at all.

        Either way the grid coarsens with speed -- past 3x, neighbouring requests
        collapse onto the same rung -- and only the fast end is bounded, at one
        substep per action = 25x. Speeds below real time are representable
        (0.75 -> 25/33 = 0.7576), so a ``min_speed`` floor is unaffected.
        """
        exact = self.steps_ideal / speed
        steps_actual = int(exact) if self.speed_rounding == "up" else math.ceil(exact - 1e-12)
        return self.steps_ideal / max(1, steps_actual)

    def apply(self, handle, speed: float) -> float:
        """Set substeps, gripper rate and OSC gains on one robosuite env.

        ``handle`` is a single vector-env member; the robosuite env and robot are
        reached through the gym wrappers.
        """
        effective = self.quantize(speed)

        # Reach the robosuite env through `.unwrapped`, not through the handle
        # directly: gym wrappers do not forward underscore-prefixed attributes, so
        # any wrapper added around the env (sim-time recording, video) would
        # otherwise hide `_env`.
        inner = getattr(handle.unwrapped, "_env", None)
        sim_env = getattr(inner, "env", None)
        if sim_env is None:
            raise AttributeError(
                f"{type(handle).__name__} does not expose a robosuite env at "
                "`.unwrapped._env.env`; PACE actuation needs it to set substeps."
            )

        sim_env.set_action_exhaust_speed(effective)

        robot = sim_env.robots[0] if getattr(sim_env, "robots", None) else None
        if robot is None:
            return effective

        if not self.disable_gripper_speedup and hasattr(robot.gripper, "speed"):
            # Assign, never multiply. The rate for this step is a fixed multiple of
            # the gripper's *nominal* stroke rate, so it cannot accumulate across
            # steps -- a compounding rate would saturate the clip inside the
            # gripper's own `format_action` within a few steps and leave it snapping
            # fully open/closed, which loses grasps.
            #
            # The fork reached the same place differently: it multiplied here and
            # restored the nominal rate after `env.step`. Assigning is equivalent and
            # has no torn state if a step raises.
            #
            # The stride is part of the multiple because a strided chunk means each
            # delivered action must cover several original ones, the gripper's
            # included. This channel is load-bearing: switching it off costs 24-30pp
            # success at no throughput gain, because a sped-up arm otherwise reaches
            # for the object before the fingers have closed.
            gripper = robot.gripper
            if not hasattr(gripper, "_pace_nominal_speed"):
                # Captured per gripper object, so an episode reset that rebuilds the
                # robot re-reads the true nominal rather than inheriting ours.
                gripper._pace_nominal_speed = gripper.speed
            gripper.speed = gripper._pace_nominal_speed * effective * self.action_stride

        # Gains are likewise assigned outright each step, so they too are a pure
        # function of this step's speed rather than of the sequence before it.
        if not self.disable_kpkd_scaling:
            ctrl = getattr(robot, "controller", None)
            if hasattr(ctrl, "update_kp_scale"):
                ctrl.update_kp_scale(effective**self.kpkd_scale_exp)
            if hasattr(ctrl, "update_kd_scale"):
                ctrl.update_kd_scale(effective ** (self.kpkd_scale_exp / 2.0))

        return effective
