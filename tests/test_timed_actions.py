"""The TimedActions contract, and the batch gate: the baseline is a no-op.

Every consumer today assumes one action per 1/fps tick. This contract makes that
assumption explicit, so the load-bearing property is that *stating* it changes
nothing: uniform dt through every adapter must reproduce byte-for-byte what the
code did before the contract existed.
"""

import pytest
import torch
from lerobot.lerobot_types import TransitionKey

from pace_bench.methods.pace.actuator import DEFAULT_CONTROL_DT, RobosuiteSpeedActuator
from pace_bench.methods.pace.processor import SPEED_KEY, PaceSpeedStep
from pace_bench.timed import DT_KEY, TimedActions

CONTROL_DT = 0.05


# -- the contract itself ------------------------------------------------------


def test_uniform_is_the_declared_baseline():
    ta = TimedActions.uniform(torch.randn(4, 7), fps=20)
    assert ta.dt == pytest.approx(CONTROL_DT)
    assert ta.is_uniform
    assert ta.duration() == pytest.approx(4 * CONTROL_DT)


def test_speed_one_is_uniform():
    """PACE at speed 1 and the baseline are the same statement."""
    actions = torch.randn(4, 7)
    from_speeds = TimedActions.from_speeds(actions, torch.ones(4), CONTROL_DT)
    assert from_speeds.is_uniform
    torch.testing.assert_close(
        from_speeds.per_step_dt(), torch.full((4,), CONTROL_DT), rtol=0, atol=0
    )


def test_speeds_roundtrip():
    speeds = torch.tensor([1.0, 2.0, 1.5625, 4.0])
    ta = TimedActions.from_speeds(torch.randn(4, 3), speeds, CONTROL_DT)
    # Roundtrip through two float32 divisions: exact to float32 ulps, not bit-exact.
    torch.testing.assert_close(ta.speeds(CONTROL_DT), speeds)
    assert not ta.is_uniform


def test_timestamps_are_exclusive_cumsum():
    """t[i] is when action i is issued: 0 for the first, sum of prior dts after."""
    ta = TimedActions(torch.randn(3, 2), torch.tensor([0.1, 0.2, 0.4]))
    torch.testing.assert_close(ta.timestamps(), torch.tensor([0.0, 0.1, 0.3]))
    assert ta.duration() == pytest.approx(0.7)


def test_batched_shapes():
    ta = TimedActions(torch.randn(2, 4, 7), torch.full((2, 4), 0.05))
    assert ta.timestamps().shape == (2, 4)
    torch.testing.assert_close(ta.duration(), torch.full((2,), 0.2))


def test_invalid_shapes_and_values_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        TimedActions(torch.randn(4, 7), 0.0)
    with pytest.raises(ValueError, match="positive"):
        TimedActions(torch.randn(4, 7), torch.tensor([0.05, -0.05, 0.05, 0.05]))
    with pytest.raises(ValueError, match="shaped"):
        TimedActions(torch.randn(4, 7), torch.full((5,), 0.05))
    with pytest.raises(ValueError, match="actions must be"):
        TimedActions(torch.randn(7), 0.05)


# -- the gate: baseline through the adapters is a no-op -----------------------


def test_gate_default_pace_step_publishes_uniform_dt_and_touches_nothing():
    """A default PaceSpeedStep with control_dt set: chunk unchanged, dt all 1/fps.

    This is the batch gate. Declaring time must not alter behaviour: the published
    dt merely *states* the uniform spacing that execution already had.
    """
    step = PaceSpeedStep(control_dt=CONTROL_DT)
    chunk = torch.randn(2, 6, 7)
    out = step({TransitionKey.ACTION: chunk.clone(), TransitionKey.COMPLEMENTARY_DATA: {}})

    torch.testing.assert_close(out[TransitionKey.ACTION], chunk, rtol=0, atol=0)
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    torch.testing.assert_close(comp[SPEED_KEY], torch.ones(2, 6), rtol=0, atol=0)
    torch.testing.assert_close(comp[DT_KEY], torch.full((2, 6), CONTROL_DT), rtol=0, atol=0)


def test_step_without_control_dt_publishes_no_dt():
    """The key appears only when the period is actually known -- no guessed units."""
    out = PaceSpeedStep()({TransitionKey.ACTION: torch.randn(1, 4, 7), TransitionKey.COMPLEMENTARY_DATA: {}})
    assert DT_KEY not in out[TransitionKey.COMPLEMENTARY_DATA]


def test_dt_matches_speeds_exactly():
    """DT_KEY and SPEED_KEY are two views of one decision, never two decisions."""
    step = PaceSpeedStep({"max_speed": 2.0, "min_speed": 1.0}, control_dt=CONTROL_DT)
    out = step({TransitionKey.ACTION: torch.randn(1, 8, 7), TransitionKey.COMPLEMENTARY_DATA: {}})
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    torch.testing.assert_close(comp[DT_KEY], CONTROL_DT / comp[SPEED_KEY], rtol=0, atol=0)


class _FakeEnv:
    """Minimal stand-in for the wrapper nesting apply() reaches through.

    tests/test_pace_actuator.py carries the full-fidelity fake; tests are not an
    importable package, so this file keeps its own copy of the required minimum.
    """

    def __init__(self):
        controller = type("Ctrl", (), {"update_kp_scale": lambda s, v: setattr(s, "kp", v),
                                       "update_kd_scale": lambda s, v: setattr(s, "kd", v)})()
        gripper = type("Grip", (), {"speed": 0.01})()
        self.robot = type("Robot", (), {"controller": controller, "gripper": gripper})()
        self.exhaust = None
        inner = type("Inner", (), {"robots": [self.robot],
                                   "set_action_exhaust_speed": lambda _s, v: setattr(self, "exhaust", v)})()
        self._env = type("Mid", (), {"env": inner})()
        self.unwrapped = self


def test_actuator_apply_dt_at_nominal_is_the_speed_one_path():
    """dt = control_dt through the actuator is exactly apply(speed=1)."""
    actuator = RobosuiteSpeedActuator()
    env_dt, env_speed = _FakeEnv(), _FakeEnv()
    realized_dt = actuator.apply_dt(env_dt, DEFAULT_CONTROL_DT)
    realized_speed = actuator.apply(env_speed, 1.0)

    assert realized_dt == pytest.approx(DEFAULT_CONTROL_DT)
    assert realized_speed == pytest.approx(1.0)
    assert env_dt.exhaust == env_speed.exhaust
    assert env_dt.robot.controller.kp == env_speed.robot.controller.kp


def test_actuator_apply_dt_reports_realized_time():
    """Quantization surfaces in the returned dt, same as it does in speed.

    A 1.5x request (dt = 0.0333s) lands on the 25/16 substep rung = 1.5625x, so the
    realised dt is 0.032s -- the contract reports what will actually happen.
    """
    actuator = RobosuiteSpeedActuator()
    realized = actuator.apply_dt(_FakeEnv(), DEFAULT_CONTROL_DT / 1.5)
    assert realized == pytest.approx(DEFAULT_CONTROL_DT / 1.5625)
