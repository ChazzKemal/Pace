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


class TestGripperStrideExemption:
    """Not dropping the grasp beats dropping it and paying it back.

    The rows kept are the poses the policy predicted. A repaid row is either a hold
    (duplicate) or an estimate of what was deleted (interpolate), and the repayment
    arithmetic has to be exactly right on runs 2-4 rows long. Exempting sidesteps all
    of it.
    """

    def chunk(self, k=12, edge=5):
        """A straight approach with the gripper closing at `edge` -- the case
        adaptive_stride is blind to, because nothing about the path bends."""
        import torch
        a = torch.zeros(1, k, 7)
        a[0, :, 0] = torch.linspace(0.0, 0.3, k)
        a[0, edge:, GRIP_COL] = 1.0
        return a

    def idx(self, **kw):
        # frames=0 throughout this class: these tests are about *detecting* the
        # transition. The window that follows it is TestExemptionWindow's subject.
        from pace_bench.methods.pace.speed import stride_indices
        kw.setdefault("gripper_stride_exempt_frames", 0)
        return stride_indices(self.chunk(), PaceMethod(action_stride=2, **kw).to_pace_config())

    def test_plain_striding_drops_the_transition(self):
        # step 5 is where the gripper moves, and 5 % 2 != 0.
        assert 5 not in self.idx()

    def test_adaptive_stride_alone_does_not_save_it(self):
        # It restores steps where the PATH bends; a grasp on a straight approach has
        # no bend to trigger on. This is the gap the new flag closes.
        assert self.idx(adaptive_stride=True) == self.idx()

    def test_the_exemption_keeps_both_ends_of_the_transition(self):
        got = self.idx(gripper_stride_exempt=True)
        assert 4 in got and 5 in got

    def test_it_keeps_the_stride_everywhere_else(self):
        # Only the grasp is exempt; the speedup elsewhere must survive.
        got = self.idx(gripper_stride_exempt=True)
        assert got == [0, 2, 4, 5, 6, 8, 10]

    def test_it_is_inert_without_a_stride(self):
        from pace_bench.methods.pace.speed import stride_indices
        cfg = PaceMethod(action_stride=1, gripper_stride_exempt=True).to_pace_config()
        assert stride_indices(self.chunk(), cfg) == list(range(12))

    def test_exempting_switches_off_the_row_repayment(self):
        # Nothing was taken from the grasp, so repaying would run it action_stride
        # times SLOWER than demonstrated -- the opposite failure.
        steps = deploy_steps(
            PaceMethod(max_speed=2.0, action_stride=2, gripper_stride_exempt=True),
            args=args(), n_action_steps=32, control_dt=0.05)
        (replicate,) = [s for s in steps if type(s).__name__ == "GripperReplicate"]
        assert replicate.low_v == 1

    def test_not_exempting_still_repays(self):
        steps = deploy_steps(PaceMethod(max_speed=2.0, action_stride=2),
                             args=args(), n_action_steps=32, control_dt=0.05)
        (replicate,) = [s for s in steps if type(s).__name__ == "GripperReplicate"]
        assert replicate.low_v == 2


class TestExemptionWindow:
    """The command edge is not the grasp. The jaws keep travelling after it settles.

    A command transition is 1-3 rows; the stroke is ~2.27 s at the deploy rate, some
    45 control steps at 20 Hz. Exempting only the edge lets the arm resume striding
    mid-close -- the arm lifting with the gripper half shut is the failure the whole
    compensation exists to prevent.
    """

    def chunk(self, k=20, edge=5):
        import torch
        a = torch.zeros(1, k, 7)
        a[0, :, 0] = torch.linspace(0.0, 0.5, k)
        a[0, edge, GRIP_COL] = 0.5          # a two-row ramp, as a policy emits
        a[0, edge + 1:, GRIP_COL] = 1.0
        return a

    def idx(self, frames):
        from pace_bench.methods.pace.speed import stride_indices
        cfg = PaceMethod(action_stride=2, gripper_stride_exempt=True,
                         gripper_stride_exempt_frames=frames).to_pace_config()
        return stride_indices(self.chunk(), cfg)

    def test_zero_frames_covers_only_the_transition(self):
        # The old behaviour, kept reachable: striding resumes as soon as the command
        # settles, which is too early on hardware.
        assert 7 not in self.idx(0)

    def test_the_window_keeps_striding_off_while_the_jaws_travel(self):
        got = self.idx(5)
        assert all(i in got for i in (7, 9, 11)), got

    def test_the_window_ends(self):
        # Only the grasp is protected; the speedup must return afterwards.
        got = self.idx(5)
        assert 13 not in got and 15 not in got

    def test_a_longer_window_protects_further(self):
        assert set(self.idx(5)) < set(self.idx(10))

    def test_the_default_matches_the_deploy_slowdown_window(self):
        # gripper_slowdown_frames is 5 in deploy_defaults.yaml and pins speed over the
        # same window. Two knobs, one physical question -- they must not drift.
        from pace_bench.real.configs import resolve_config
        shipped = resolve_config("real/configs/deploy_defaults.yaml")["gripper"]["slowdown_frames"]
        assert PaceMethod().gripper_stride_exempt_frames == shipped
