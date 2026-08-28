"""DemoSpeedup's eval-time actuation: stiffen tracking, never accelerate time.

At inference a DemoSpeedup policy emits waypoints ``low_v`` (or more) raw frames
apart while the control rate stays nominal -- the speedup lives in the action
stream. Two things must scale so the plant can follow that stream, and one thing
must not:

* **gripper stroke rate** x ``low_v``: the arm covers ground faster, so the
  fingers must close faster or the arm reaches the object before the grasp forms.
  This is exactly what upstream DemoSpeedup does at eval -- its high-gain XMLs
  double the gripper kp (aloha: finger kp 200 -> 400 = x low_v) and nothing else.
* **OSC gains**: ``kp`` as ``low_v ** exp``, ``kd`` as ``low_v ** (exp/2)`` --
  critical damping is ``kd = 2*sqrt(kp)``, so kd takes half the exponent. This is
  the arm-side extension of the same tracking argument, borrowed from PACE's
  actuation law but at a constant factor.
* **time** -- untouched. Raising the simulator's substep exhaust would apply the
  speedup a second time on top of the retimed actions (the fork's stage-4 notes
  warn about exactly this double application).

``low_v`` is the floor of the retiming: every executed step covers at least
``low_v`` raw frames, so it is the constant the plant must track everywhere;
faster (``high_v``) segments are transient.

The application is per-step, not once at binding, because a robosuite reset
rebuilds the robot and its controller -- a one-shot bump would silently vanish at
the first episode boundary. ``apply`` therefore has the same duck-type as PACE's
actuator so the eval loop can call it each step; the per-step speed argument is
ignored (there is no per-step decision to make) and the returned *time* speed is
1.0, because time runs nominal.
"""


class DemoSpeedupTrackingActuator:
    """Constant gains+gripper scaling at ``low_v`` on a robosuite env."""

    def __init__(
        self,
        low_v: int,
        kpkd_scale_exp: float = 2.0,
        disable_kpkd_scaling: bool = False,
        disable_gripper_speedup: bool = False,
    ):
        """
        Args:
            low_v: The retiming's precision stride -- the constant tracking factor.
            kpkd_scale_exp: Exponent on ``low_v`` for kp (kd gets half of it).
            disable_kpkd_scaling: Leave OSC gains nominal (gripper still scales) --
                isolates the gripper channel, or reproduces upstream's exact recipe,
                which bumps only the gripper.
            disable_gripper_speedup: Leave the gripper at its nominal stroke rate.
        """
        self.low_v = float(low_v)
        self.kpkd_scale_exp = kpkd_scale_exp
        self.disable_kpkd_scaling = disable_kpkd_scaling
        self.disable_gripper_speedup = disable_gripper_speedup

    def apply(self, handle, speed: float = 1.0) -> float:
        """Apply the constant bump to one vector-env member. Returns time speed (1.0).

        ``speed`` is accepted and ignored so this quacks like a per-step speed
        actuator for the eval loop; DemoSpeedup has no per-step speed decision.
        """
        # Reach the robosuite env through `.unwrapped`: gym wrappers do not forward
        # underscore-prefixed attributes (same reach-through as PACE's actuator).
        inner = getattr(handle.unwrapped, "_env", None)
        sim_env = getattr(inner, "env", None)
        if sim_env is None:
            raise AttributeError(
                f"{type(handle).__name__} does not expose a robosuite env at "
                "`.unwrapped._env.env`; DemoSpeedup tracking actuation needs it."
            )

        robot = sim_env.robots[0] if getattr(sim_env, "robots", None) else None
        if robot is None:
            return 1.0

        if not self.disable_gripper_speedup and hasattr(robot.gripper, "speed"):
            # Assign from the captured nominal, never multiply in place: the rate
            # must be a fixed multiple of nominal, not compound across steps.
            gripper = robot.gripper
            if not hasattr(gripper, "_demospeedup_nominal_speed"):
                gripper._demospeedup_nominal_speed = gripper.speed
            gripper.speed = gripper._demospeedup_nominal_speed * self.low_v

        if not self.disable_kpkd_scaling:
            ctrl = getattr(robot, "controller", None)
            if hasattr(ctrl, "update_kp_scale"):
                ctrl.update_kp_scale(self.low_v**self.kpkd_scale_exp)
            if hasattr(ctrl, "update_kd_scale"):
                ctrl.update_kd_scale(self.low_v ** (self.kpkd_scale_exp / 2.0))

        return 1.0
