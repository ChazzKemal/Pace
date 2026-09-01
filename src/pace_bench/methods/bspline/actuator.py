"""B-spline's eval-time actuation: stiffen position tracking, nothing else.

A B-spline policy executes the same demonstrated path in fewer steps, so the
commanded waypoints are further apart while the control rate stays nominal. The
plant has to keep up with that, and upstream's answer is one number.

Reproduced from upstream's deployment
(`real_env/yam_teleop/yam_server.py:_apply_stiffness_scale`, reached through
`--stiffness-kp-scale` in `scripts/rollout_x5_bspline.py`, whose default is 2.0):

* **kp x 2.0**, and only on the arm joints (upstream scales ``kp[:6]``).
* **kd untouched** -- ``update_kp_kd(kp, self._base_kd.copy())`` passes the base kd
  back unchanged. This is where B-spline differs from both of the other methods
  here: PACE scales kd as ``s**(exp/2)`` and DemoSpeedup as ``low_v**(exp/2)``,
  both chasing critical damping. Upstream B-spline simply does not, so neither does
  this.
* **gripper untouched** -- upstream's comment is explicit that the gripper kp is
  left alone. DemoSpeedup scales gripper stroke rate; B-spline does not.
* **constant, not speed-dependent** -- a fixed multiple of the default gains,
  applied relative to them so repeated application cannot compound.
* **time untouched** -- the speed-up is already in the action stream. Raising the
  simulator's substep exhaust would apply it a second time.

Upstream also slows execution while the gripper is moving, ramping the speed-up
back from 1.0 over 7 steps. That is a *time-advance* mechanism with no equivalent
here -- our decode commits to ``num_actions`` samples per query rather than
advancing a clock -- and it is off by default upstream
(``gripper_slowdown_enabled=False``), so it is not reproduced. See the plan's
real-inference gripper postprocessor item.

Applied per step rather than once at binding, because a robosuite reset rebuilds
the robot and its controller: a one-shot bump would vanish at the first episode
boundary. ``apply`` therefore has the same duck-type as PACE's actuator.
"""

#: Upstream's default (`rollout_x5_bspline.py --stiffness-kp-scale`). 1.0 is nominal.
DEFAULT_KP_SCALE = 2.0


class BSplineTrackingActuator:
    """Constant arm-kp stiffening on a robosuite env, upstream's recipe."""

    def __init__(self, kp_scale: float = DEFAULT_KP_SCALE, disable_kp_scaling: bool = False):
        """
        Args:
            kp_scale: Multiplier on the arm position gain, relative to nominal.
                Upstream's default is 2.0; >1 holds commanded poses more stiffly.
                Upstream warns that too large a value causes buzzing or oscillation
                on contact, so this is not a knob to raise casually.
            disable_kp_scaling: Leave gains nominal -- the action-side-only ablation,
                which isolates how much of the result is the representation rather
                than the actuation.
        """
        if kp_scale <= 0:
            raise ValueError(f"kp_scale must be > 0, got {kp_scale}")
        self.kp_scale = float(kp_scale)
        self.disable_kp_scaling = disable_kp_scaling

    def apply(self, handle, speed: float = 1.0) -> float:
        """Apply the constant bump to one vector-env member. Returns time speed (1.0).

        ``speed`` is accepted and ignored so this quacks like a per-step speed
        actuator for the eval loop; B-spline makes no per-step gain decision.
        """
        # Reach the robosuite env through `.unwrapped`: gym wrappers do not forward
        # underscore-prefixed attributes (same reach-through as the other actuators).
        inner = getattr(handle.unwrapped, "_env", None)
        sim_env = getattr(inner, "env", None)
        if sim_env is None:
            raise AttributeError(
                f"{type(handle).__name__} does not expose a robosuite env at "
                "`.unwrapped._env.env`; B-spline tracking actuation needs it."
            )

        robot = sim_env.robots[0] if getattr(sim_env, "robots", None) else None
        if robot is None:
            return 1.0

        if not self.disable_kp_scaling:
            controller = getattr(robot, "controller", None)
            if hasattr(controller, "update_kp_scale"):
                # A scale relative to nominal, so re-applying every step is idempotent
                # rather than compounding. kd is deliberately not touched.
                controller.update_kp_scale(self.kp_scale)

        return 1.0
