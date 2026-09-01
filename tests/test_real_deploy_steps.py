"""PACE with striding must repay the gripper in rows, not only in time.

`GripperHold` pins the speed to 1.0 near a close edge, which is the right currency
while PACE only modulates speed. `action_stride` drops rows, so each surviving row
covers `action_stride` source frames -- and a pinned speed buys one row of nominal
wall-clock where the demonstration had `action_stride` of them. That half of the debt
is only payable in rows.

Needs crisp_gym (deploy_steps imports its pipeline), so it skips on a machine with no
robot stack rather than failing there.
"""

import numpy as np
import pytest

pytest.importorskip("crisp_gym", reason="deploy_steps needs the robot stack")

from argparse import Namespace

from crisp_gym.deploy.pipeline import Chunk

from pace_bench.methods.config import NoMethod, PaceMethod
from pace_bench.real.deploy_steps import deploy_steps

GRIP_COL = 6


def args(slowdown=5, invert=False):
    return Namespace(gripper_slowdown_frames=slowdown, invert_gripper=invert)


def grasp_chunk(k=12, edge=4, ramp=2):
    """A chunk whose gripper closes over `ramp` rows starting at `edge`."""
    a = np.zeros((k, 7), dtype=np.float64)
    a[:, 0] = np.linspace(0.0, 0.3, k)          # the arm keeps moving through it
    a[edge:edge + ramp, GRIP_COL] = np.linspace(0.3, 1.0, ramp)
    a[edge + ramp:, GRIP_COL] = 1.0
    return Chunk(actions=a, speeds=np.full(k, 2.0))


def steps_for(stride):
    return deploy_steps(PaceMethod(max_speed=2.0, action_stride=stride),
                        args=args(), n_action_steps=32, control_dt=0.05)


def gripper_stage(stride):
    """The two gripper steps, skipping PaceSpeed (which needs a real policy chunk)."""
    return steps_for(stride)[1:]


def run(stage, chunk):
    for s in stage:
        chunk = s(chunk)
    return chunk


class TestStrideIsRepaidInRows:
    def test_the_default_stride_changes_nothing(self):
        # Every shipped config is action_stride=1, so this path must stay identical.
        before = grasp_chunk()
        after = run(gripper_stage(1), grasp_chunk())
        assert after.actions.shape == before.actions.shape

    @pytest.mark.parametrize("stride", [2, 3])
    def test_the_gripper_run_grows_by_the_stride(self, stride):
        plain = run(gripper_stage(1), grasp_chunk())
        strided = run(gripper_stage(stride), grasp_chunk())
        # Only the moving rows are repeated, so the chunk grows by less than stride x.
        assert strided.actions.shape[0] > plain.actions.shape[0]

    def test_only_the_moving_rows_are_repeated(self):
        # Repeating settled rows would cover the whole transport and destroy the
        # speedup; GripperMotionRun is bounded to |dgrip| > eps for that reason.
        k = 12
        out = run(gripper_stage(2), grasp_chunk(k=k, edge=4, ramp=2))
        assert out.actions.shape[0] == k + 2      # two moving rows, doubled

    def test_the_replicas_keep_the_held_speed(self):
        # GripperHold writes speeds and GripperReplicate repeats them by index, so a
        # replica must carry 1.0 too. If the order were reversed they would carry 2.0
        # and the extra rows would be executed at speed -- paying nothing.
        out = run(gripper_stage(2), grasp_chunk())
        assert out.speeds.min() == pytest.approx(1.0)
        assert (out.speeds == 1.0).sum() >= 2

    def test_a_chunk_with_no_gripper_motion_is_untouched(self):
        # Held OPEN, not zeros: a zero gripper reads as closed, and with no previous
        # chunk to compare against that is an open->close edge on row 0 -- so
        # GripperHold fires, correctly. Transport with the gripper open is the case
        # where neither step should do anything.
        flat = Chunk(actions=np.concatenate(
            [np.zeros((8, 6)), np.ones((8, 1))], axis=1), speeds=np.full(8, 2.0))
        out = run(gripper_stage(3), flat)
        assert out.actions.shape == (8, 7)
        assert np.array_equal(out.speeds, np.full(8, 2.0))

    def test_a_first_chunk_that_starts_closed_is_treated_as_a_grab(self):
        # Documents the above: the detector cannot know the prior level on the first
        # chunk, so starting closed is read as an edge. Conservative in the safe
        # direction -- it protects a grasp that may already have happened.
        closed = Chunk(actions=np.zeros((8, 7)), speeds=np.full(8, 2.0))
        out = run(gripper_stage(1), closed)
        assert out.speeds[0] == pytest.approx(1.0)

    def test_the_arm_stays_on_the_same_path(self):
        # Each row of the run is repeated, not one row held: the arm is still moving
        # during a grasp, so a hold would freeze it and then jump.
        out = run(gripper_stage(2), grasp_chunk())
        x = out.actions[:, 0]
        assert np.all(np.diff(x) >= -1e-12), "the path must stay monotone, not jump back"


class TestOtherMethodsAreUnaffected:
    def test_none_keeps_its_two_steps(self):
        names = [type(s).__name__ for s in deploy_steps(
            NoMethod(), args=args(), n_action_steps=32, control_dt=0.05)]
        assert "GripperReplicate" not in names
