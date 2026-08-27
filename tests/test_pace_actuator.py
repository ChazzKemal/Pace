"""The actuation law, checked without a simulator.

Speed quantization and gain compensation are arithmetic; only the three hook calls
need robosuite. A fake robot stands in for it so these run anywhere, and the fake
also records call order, which is the part a numerical check would miss.
"""

from __future__ import annotations

import math

import pytest

from robot_stack.methods.pace.actuator import RobosuiteSpeedActuator


class FakeController:
    def __init__(self):
        self.kp = self.kd = None

    def update_kp_scale(self, v):
        self.kp = v

    def update_kd_scale(self, v):
        self.kd = v


class FakeGripper:
    speed = 0.01


class FakeRobot:
    def __init__(self):
        self.controller = FakeController()
        self.gripper = FakeGripper()


class FakeEnv:
    """Mimics the wrapper nesting the actuator reaches through."""

    def __init__(self):
        self.robot = FakeRobot()
        self.exhaust = None
        inner = type("Inner", (), {"robots": [self.robot], "set_action_exhaust_speed": self._set})()
        self._env = type("Mid", (), {"env": inner})()
        self.unwrapped = self

    def _set(self, v):
        self.exhaust = v


class HidingWrapper:
    """A gym-style wrapper: forwards `unwrapped`, hides underscore attributes.

    Regression guard -- adding the sim-time recorder around each env broke the
    actuator exactly this way.
    """

    def __init__(self, inner):
        self.unwrapped = inner


def test_reaches_through_a_wrapper():
    env = FakeEnv()
    RobosuiteSpeedActuator().apply(HidingWrapper(env), 2.0)
    assert env.exhaust == pytest.approx(25 / 12)


def test_missing_sim_env_says_so():
    class NotAnEnv:
        unwrapped = object()

    with pytest.raises(AttributeError, match="robosuite env"):
        RobosuiteSpeedActuator().apply(NotAnEnv(), 1.5)


def test_quantize_matches_substep_arithmetic():
    """Achievable speeds are 25/n. Truncating n rounds the *speed* up, never down."""
    act = RobosuiteSpeedActuator()
    assert act.quantize(1.0) == 1.0
    assert act.quantize(1.5) == pytest.approx(25 / 16)  # 1.5625 -- the headline 1.5x run
    assert act.quantize(2.0) == pytest.approx(25 / 12)
    assert act.quantize(3.0) == pytest.approx(25 / 8)
    for req in (1.0, 1.2, 1.47, 1.5, 2.0, 2.6, 3.0, 4.0, 5.0):
        assert act.quantize(req) >= req - 1e-12, "quantization must never fall below the request"


def test_rounding_down_makes_max_speed_a_real_ceiling():
    """The point of the option: `up` overshoots the requested ceiling, `down` cannot."""
    up, down = RobosuiteSpeedActuator(), RobosuiteSpeedActuator(speed_rounding="down")
    for req in (1.1, 1.5, 2.0, 3.0, 4.0):
        assert up.quantize(req) >= req - 1e-12
        assert down.quantize(req) <= req + 1e-12
    assert up.quantize(1.5) == pytest.approx(25 / 16)  # 1.5625 -- over the ceiling
    assert down.quantize(1.5) == pytest.approx(25 / 17)  # 1.4706 -- under it


def test_rounding_down_costs_throughput_near_real_time():
    """Where the grid is coarsest below 1.04x, rounding down means no speedup at all."""
    down = RobosuiteSpeedActuator(speed_rounding="down")
    assert down.quantize(1.02) == 1.0
    assert down.quantize(1.0) == 1.0


def test_rounding_is_validated():
    with pytest.raises(ValueError, match="speed_rounding"):
        RobosuiteSpeedActuator(speed_rounding="nearest")


def test_exact_rungs_are_untouched_either_way():
    """A request already on the grid must not be nudged by the rounding rule."""
    for mode in ("up", "down"):
        act = RobosuiteSpeedActuator(speed_rounding=mode)
        for n in (1, 8, 12, 16, 25):
            assert act.quantize(25 / n) == pytest.approx(25 / n), f"{mode} moved an exact rung"


def test_quantize_grid_coarsens_with_speed():
    """Why high-speed cells blur together: neighbouring requests share a rung."""
    act = RobosuiteSpeedActuator()
    assert act.quantize(3.0) == act.quantize(3.1)  # both 25/8
    assert act.quantize(1.1) != act.quantize(1.2)  # still resolvable down here


def test_quantize_floors_only_at_the_fast_end():
    """One substep per action is the ceiling; sub-real-time speeds pass through."""
    act = RobosuiteSpeedActuator()
    assert act.quantize(1000.0) == 25.0
    assert act.quantize(0.75) == pytest.approx(25 / 33)  # PACE's min_speed can sit below 1


def test_gains_stay_critically_damped():
    """kd must remain 2*sqrt(kp) as the gains rise, or the arm rings."""
    env = FakeEnv()
    RobosuiteSpeedActuator(kpkd_scale_exp=2.0).apply(env, 3.0)
    kp, kd = env.robot.controller.kp, env.robot.controller.kd
    assert kd == pytest.approx(math.sqrt(kp))
    assert kp == pytest.approx(env.exhaust**2)


def test_exponent_is_tunable():
    env = FakeEnv()
    RobosuiteSpeedActuator(kpkd_scale_exp=1.0).apply(env, 3.0)
    assert env.robot.controller.kp == pytest.approx(env.exhaust)
    assert env.robot.controller.kd == pytest.approx(math.sqrt(env.exhaust))


def test_no_gain_bump_still_changes_substeps():
    """The ablation that isolates the action-side speedup from the gain bump."""
    env = FakeEnv()
    eff = RobosuiteSpeedActuator(disable_kpkd_scaling=True).apply(env, 2.0)
    assert env.exhaust == eff > 1.0
    assert env.robot.controller.kp is None


def test_gripper_rate_does_not_compound():
    """The rate must depend on this step's speed only, never on the sequence.

    Regression guard: multiplying instead of assigning grows the rate geometrically,
    saturates the clip in the gripper's `format_action` within a few steps, and
    leaves it snapping fully open/closed -- which quietly loses grasps.
    """
    env = FakeEnv()
    act = RobosuiteSpeedActuator(action_stride=2)
    eff = act.apply(env, 1.5)
    assert env.robot.gripper.speed == pytest.approx(0.01 * eff * 2)
    for _ in range(10):
        act.apply(env, 1.5)
    assert env.robot.gripper.speed == pytest.approx(0.01 * eff * 2), "rate accumulated across steps"


def test_gripper_rate_tracks_a_changing_speed():
    """Slowing down must restore a slower gripper, not keep the fast one."""
    env = FakeEnv()
    act = RobosuiteSpeedActuator()
    act.apply(env, 3.0)
    fast = env.robot.gripper.speed
    eff_slow = act.apply(env, 1.0)
    assert env.robot.gripper.speed == pytest.approx(0.01 * eff_slow)
    assert env.robot.gripper.speed < fast


def test_gripper_speedup_can_be_disabled():
    env = FakeEnv()
    RobosuiteSpeedActuator(disable_gripper_speedup=True).apply(env, 2.0)
    assert env.robot.gripper.speed == 0.01
