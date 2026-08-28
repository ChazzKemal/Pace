"""DemoSpeedup's eval-time tracking bump: reference tracking scales, time does not.

Upstream DemoSpeedup's high-gain XMLs double the gripper kp at eval while the
control rate stays nominal; accelerating time here would apply the speedup a
second time on top of the retimed action stream. The class under test lives in the
demospeedup package because the constant-low_v decision is the method's own -- it
shares only the per-step apply() duck-type with PACE's actuator.
"""

import pytest

from robot_stack.methods.demospeedup.actuator import DemoSpeedupTrackingActuator


class FakeController:
    kp = kd = None

    def update_kp_scale(self, v):
        self.kp = v

    def update_kd_scale(self, v):
        self.kd = v


class FakeEnv:
    def __init__(self):
        gripper = type("Grip", (), {"speed": 0.01})()
        self.robot = type("Robot", (), {"controller": FakeController(), "gripper": gripper})()
        self.exhaust = None
        inner = type("Inner", (), {"robots": [self.robot],
                                   "set_action_exhaust_speed": lambda _s, v: setattr(self, "exhaust", v)})()
        self._env = type("Mid", (), {"env": inner})()
        self.unwrapped = self


def test_scales_gains_and_gripper_but_never_time():
    actuator = DemoSpeedupTrackingActuator(low_v=2, kpkd_scale_exp=2.0)
    env = FakeEnv()
    assert actuator.apply(env) == 1.0            # reported time speed: nominal
    assert env.exhaust is None                   # time untouched
    assert env.robot.controller.kp == pytest.approx(4.0)   # 2^2
    assert env.robot.controller.kd == pytest.approx(2.0)   # 2^1
    assert env.robot.gripper.speed == pytest.approx(0.02)


def test_reapplication_does_not_compound():
    """Per-step re-application (reset-proofing) must be idempotent on the gripper."""
    actuator = DemoSpeedupTrackingActuator(low_v=2)
    env = FakeEnv()
    for _ in range(5):
        actuator.apply(env)
    assert env.robot.gripper.speed == pytest.approx(0.02)


def test_upstreams_exact_recipe_is_expressible():
    """disable_kpkd_scaling=True leaves only the gripper bump -- upstream's eval."""
    actuator = DemoSpeedupTrackingActuator(low_v=2, disable_kpkd_scaling=True)
    env = FakeEnv()
    actuator.apply(env)
    assert env.robot.controller.kp is None
    assert env.robot.gripper.speed == pytest.approx(0.02)


def test_ignores_per_step_speed_argument():
    """The eval loop passes PACE-style per-step speeds; there is no per-step decision."""
    actuator = DemoSpeedupTrackingActuator(low_v=2)
    a, b = FakeEnv(), FakeEnv()
    actuator.apply(a, 1.0)
    actuator.apply(b, 3.7)
    assert a.robot.controller.kp == b.robot.controller.kp == pytest.approx(4.0)
