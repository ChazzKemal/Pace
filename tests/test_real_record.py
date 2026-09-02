"""`command_table` / `pose_table` shape a run's rows into columns an analysis can read.

The sampler needs a robot; the column contract does not -- and the contract is the part
that rots silently, because a renamed key in crisp_gym's replay row yields a CSV that is
well-formed and empty rather than an error. These run on a laptop, the same bargain
`deploy_flags` makes.
"""

import numpy as np
import pytest

from pace_bench.real.record import command_table, pose_table


def replay_row(frame_index=0, action=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0), s_eff=1.5):
    """A row shaped like the one both senders build (sender.py:355-360)."""
    return {
        "frame_index": frame_index,
        "timestamp": 1000.0 + frame_index,
        "replay.s_eff": s_eff,
        "replay.cycles": 10 + frame_index,
        "replay.action": np.asarray(action, dtype=np.float32),
    }


class TestCommandTable:
    def test_it_returns_nothing_for_an_empty_log(self):
        assert command_table([]) == ([], [])

    def test_it_names_cart7_columns_by_axis(self):
        fields, rows = command_table([replay_row()])
        for name in ("cmd_x", "cmd_y", "cmd_z", "cmd_rx", "cmd_ry", "cmd_rz",
                     "cmd_grip"):
            assert name in fields
        assert rows[0]["cmd_x"] == pytest.approx(0.1)
        assert rows[0]["cmd_grip"] == pytest.approx(1.0)

    def test_it_carries_the_speed_and_the_queue_clock(self):
        _, rows = command_table([replay_row(s_eff=2.0)])
        assert rows[0]["s_eff"] == pytest.approx(2.0)
        assert rows[0]["t_wall"] == pytest.approx(1000.0)

    def test_it_holds_no_achieved_pose(self):
        # Ground truth comes from PoseSampler on its own clock, never from the
        # sender's capture hook -- under the C++ sender that hook fires in `put()`,
        # K times per chunk. See the module docstring.
        fields, _ = command_table([replay_row()])
        assert not [f for f in fields if f.startswith("ach_")]

    def test_it_subtracts_nothing(self):
        fields, _ = command_table([replay_row()])
        assert not [f for f in fields if f.startswith("err_")]

    def test_it_falls_back_to_indices_for_an_unfamiliar_action_width(self):
        fields, _ = command_table([replay_row(action=(1.0,) * 10)])
        assert "cmd_0" in fields and "cmd_9" in fields

    def test_a_row_missing_its_action_still_yields_a_row(self):
        r = replay_row(1)
        del r["replay.action"]
        fields, rows = command_table([replay_row(0), r])
        assert "cmd_x" in fields
        assert len(rows) == 2
        assert "cmd_x" not in rows[1]


class TestPoseTable:
    def test_it_returns_nothing_for_no_samples(self):
        assert pose_table([]) == ([], [])

    def test_it_names_the_pose_by_axis_and_keeps_both_clocks(self):
        samples = [(500.0, 1000.0, np.arange(6, dtype=np.float32))]
        fields, rows = pose_table(samples)
        assert fields[:2] == ["t_mono", "t_wall"]
        for name in ("ach_x", "ach_y", "ach_z", "ach_rx", "ach_ry", "ach_rz"):
            assert name in fields
        assert rows[0]["ach_x"] == pytest.approx(0.0)
        assert rows[0]["t_mono"] == pytest.approx(500.0)

    def test_the_achieved_pose_has_no_gripper_column(self):
        fields, _ = pose_table([(0.0, 0.0, np.zeros(6, dtype=np.float32))])
        assert "ach_grip" not in fields
