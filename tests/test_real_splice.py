"""The splice loop, on plans whose right answer is known.

`Plan.splice` is checked for the three things the design turns on: the new chunk
enters at the time-aligned row, the bridge is continuous in position and per-row
velocity at both ends, and nothing from the old plan past the splice survives. The
loop itself runs against a fake sender that behaves like crisp_gym's -- pop, sleep to
the deadline, publish -- so the feed rule's `commit_rows = 1` claim is checked as a
timing fact rather than an intention.
"""

import itertools
import queue
import sys
import threading
import time
import types
from dataclasses import dataclass

import numpy as np
import pytest

from pace_bench.real.splice_loop import (
    Plan,
    SpliceConfig,
    cubic_bridge,
    needs_replan,
    quintic_bridge,
    rows_to_push,
)


def line(start, step, n, grip=0.0):
    """n rows of a straight line: xyz advancing by `step`, rotvec fixed, gripper const."""
    rows = np.zeros((n, 7))
    rows[:, :3] = np.asarray(start) + np.arange(n)[:, None] * np.asarray(step)
    rows[:, 3:6] = [2.1, -2.1, 0.15]
    rows[:, 6] = grip
    return rows


class TestBridges:
    def test_cubic_matches_position_and_per_row_velocity_at_both_ends(self):
        p0, v0 = np.array([0.0, 0, 0, 0, 0, 0]), np.array([1.0, 0, 0, 0, 0, 0])
        p1, v1 = np.array([10.0, 0, 0, 0, 0, 0]), np.array([1.0, 0, 0, 0, 0, 0])
        # Same velocity both ends and p1 = p0 + 10 v over T = 10 rows: the cubic must be
        # the straight line, one unit per row -- any curvature would be a bug.
        b = cubic_bridge(p0, v0, p1, v1, 9)
        assert np.allclose(b[:, 0], np.arange(1, 10))

    def test_quintic_reduces_to_the_line_too(self):
        p0, v0, a0 = np.zeros(6), np.eye(6)[0], np.zeros(6)
        p1, v1, a1 = 10 * np.eye(6)[0], np.eye(6)[0], np.zeros(6)
        b = quintic_bridge(p0, v0, a0, p1, v1, a1, 9)
        assert np.allclose(b[:, 0], np.arange(1, 10))


class TestPlanSplice:
    cfg = SpliceConfig(commit_rows=1, bridge_rows=4, bridge="cubic")

    def test_first_chunk_is_placed_verbatim_with_no_bridge(self):
        plan = Plan()
        new = line([0, 0, 0], [1, 0, 0], 32)
        info = plan.splice(0, new, np.ones(32), k_obs=0, cfg=self.cfg)
        assert info.n_bridge == 0 and info.i_star == 0 and info.rows_added == 32
        assert np.array_equal(np.array(plan.actions), new)

    def test_new_chunk_enters_at_the_time_aligned_row(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 32), np.ones(32), k_obs=0, cfg=self.cfg)
        # Observation at row 10, splice at 12: the bridge covers rows 12..15 and the
        # verbatim part starts at output row i* = 12 + 4 - 10 = 6.
        new = line([100, 0, 0], [1, 0, 0], 32)
        info = plan.splice(12, new, np.ones(32), k_obs=10, cfg=self.cfg)
        assert (info.k_s, info.i_star, info.n_bridge) == (12, 6, 4)
        assert info.rows_retracted == 32 - 12
        assert np.array_equal(plan.actions[16], new[6])
        assert np.array_equal(plan.actions[17], new[7])
        assert len(plan) == 12 + 4 + (32 - 6)

    def test_bridge_is_continuous_at_both_ends(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 32), np.ones(32), k_obs=0, cfg=self.cfg)
        # A chunk that continues the same line but offset 3 units up in z: the seam has
        # to bend from the old line to the new one.
        new = line([10, 0, 3], [1, 0, 0], 32)
        plan.splice(12, new, np.ones(32), k_obs=10, cfg=self.cfg)
        P = np.array(plan.actions)[:, :3]
        d = np.diff(P, axis=0)
        # First bridge step continues the old velocity (Hermite: p'(0) = v0), the last
        # arrives with the new one; interior steps stay bounded -- no overshoot beyond
        # the 3 mm the seam has to climb.
        assert np.allclose(d[10], [1, 0, 0])                       # before the seam
        assert d[11][0] == pytest.approx(1.0, abs=0.15)             # into the bridge
        assert d[15][0] == pytest.approx(1.0, abs=0.15)             # out of the bridge
        assert np.allclose(d[16], [1, 0, 0])
        assert P[12:16, 2].min() >= -1e-9 and P[12:16, 2].max() <= 3 + 1e-9
        assert np.all(np.diff(P[11:17, 2]) >= -1e-9)                # monotone climb

    def test_bridge_gripper_is_the_new_chunks_time_aligned_value(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 32, grip=0.0), np.ones(32), k_obs=0, cfg=self.cfg)
        new = line([0, 0, 0], [1, 0, 0], 32, grip=1.0)
        new[:4, 6] = 0.5    # rows the bridge overlaps in time carry 0.5
        plan.splice(12, new, np.ones(32), k_obs=10, cfg=self.cfg)
        # bridge rows 12..15 <-> output rows 2..5: two of them are 0.5, two are 1.0
        assert [a[6] for a in plan.actions[12:16]] == [0.5, 0.5, 1.0, 1.0]

    def test_speeds_stay_aligned_with_rows(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 32), np.ones(32), k_obs=0, cfg=self.cfg)
        new = line([0, 0, 0], [1, 0, 0], 32)
        sp = np.linspace(1.0, 2.0, 32)
        plan.splice(12, new, sp, k_obs=10, cfg=self.cfg)
        assert len(plan.speeds) == len(plan.actions)
        assert plan.speeds[12:16] == pytest.approx(sp[2:6].tolist())   # bridge: aligned rows
        assert plan.speeds[16] == pytest.approx(sp[6])                  # i*

    def test_a_short_chunk_falls_back_to_no_bridge(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 8), np.ones(8), k_obs=0, cfg=self.cfg)
        info = plan.splice(4, line([0, 0, 0], [1, 0, 0], 6), np.ones(6), k_obs=2, cfg=self.cfg)
        assert info.n_bridge == 0 and info.i_star == 2

    def test_splice_before_the_observation_is_refused(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 8), np.ones(8), k_obs=0, cfg=self.cfg)
        with pytest.raises(ValueError):
            plan.splice(3, line([0, 0, 0], [1, 0, 0], 8), np.ones(8), k_obs=5, cfg=self.cfg)


class TestFeedRules:
    def test_two_rows_go_out_before_anything_is_published(self):
        assert list(rows_to_push(0, -1)) == [0, 1]

    def test_then_exactly_one_row_per_publish(self):
        assert list(rows_to_push(2, 0)) == []          # row 0 executing, 1 held: nothing
        assert list(rows_to_push(2, 1)) == [2]         # row 1 published: hand over 2
        assert list(rows_to_push(3, 1)) == []

    def test_replan_on_the_period_or_when_the_plan_runs_short(self):
        cfg = SpliceConfig(replan_every=12, bridge_rows=4)
        assert needs_replan(0, -10**9, 0, 0, cfg)             # nothing yet
        assert not needs_replan(5, 0, 7, 32, cfg)
        assert needs_replan(12, 0, 14, 32, cfg)                # H rows since the last obs
        assert needs_replan(5, 0, 26, 32, cfg)                 # only 6 rows left < h_b + 3


# --------------------------------------------------------------------- the loop

@dataclass
class _Item:
    pose_xyz: np.ndarray
    pose_quat: np.ndarray
    grip_raw: float | None
    action: np.ndarray
    deadline_mono: float
    frame_idx: int
    s_eff: float
    cycles: int


class FakeSender(threading.Thread):
    """crisp_gym's sender in miniature: pop, sleep to the deadline, publish."""

    def __init__(self, q):
        super().__init__(daemon=True)
        self.q, self.n_published, self.published = q, 0, []

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            time.sleep(max(0.0, item.deadline_mono - time.monotonic()))
            self.published.append((item.frame_idx, time.monotonic(), item.action.copy(),
                                   self.q.qsize()))
            self.n_published += 1


def _stub_crisp_gym(monkeypatch, dt):
    """Just enough of crisp_gym.deploy for the loop to import, with no robot."""
    def pipeline_stub():
        m = types.ModuleType("crisp_gym.deploy.pipeline")

        class Chunk:
            def __init__(self, actions, speeds):
                self.actions, self.speeds = actions, speeds

            @classmethod
            def nominal(cls, a):
                return cls(np.asarray(a), np.ones(len(a)))
        m.Chunk = Chunk
        m.run_pipeline = lambda chunk, steps: chunk
        return m

    timing = types.ModuleType("crisp_gym.deploy.timing")
    timing.build_speed_queue_arrays = lambda s, dt_base, n, retime: (
        np.full(n, 25), np.full(n, dt_base), np.ones(n))
    timing._pre_compute_chunk_arrays = lambda row, **kw: (
        row[:, :3], np.tile([0, 0, 0, 1.0], (len(row), 1)), row[:, 6], row.astype(np.float32))
    obs = types.ModuleType("crisp_gym.deploy.obs")
    obs._get_obs_zerofill = lambda env, schema, last: {"observation.state.cartesian": np.zeros(6)}
    sender = types.ModuleType("crisp_gym.deploy.sender")
    sender.TargetItem = _Item
    sources = types.ModuleType("crisp_gym.deploy.sources")
    sources.DatasetExhausted = type("DatasetExhausted", (Exception,), {})
    for name, mod in [("crisp_gym", types.ModuleType("crisp_gym")),
                      ("crisp_gym.deploy", types.ModuleType("crisp_gym.deploy")),
                      ("crisp_gym.deploy.pipeline", pipeline_stub()),
                      ("crisp_gym.deploy.timing", timing), ("crisp_gym.deploy.obs", obs),
                      ("crisp_gym.deploy.sender", sender), ("crisp_gym.deploy.sources", sources)]:
        monkeypatch.setitem(sys.modules, name, mod)


class TestLoop:
    def test_one_row_ahead_and_time_aligned_splices(self, monkeypatch):
        from pace_bench.real.splice_loop import run_splice_loop

        dt = 0.02
        _stub_crisp_gym(monkeypatch, dt)
        q = queue.Queue()
        sender = FakeSender(q)
        sender.start()
        requests = []

        class Source:
            def request(self, obs_buf):
                # The policy sees the row that is executing and predicts a straight line
                # from there, so consecutive chunks agree and the seam should be smooth.
                k = max(sender.n_published - 1, 0)
                requests.append(k)
                time.sleep(0.006)                           # "inference"
                return line([float(k), 0, 0], [1, 0, 0], 32)

        rec = types.SimpleNamespace(
            chunk_count=0, stopped_by="init", starvation_event_count=0,
            stage_samples_producer={k: [] for k in
                                    ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms")},
            pred_dt_samples=[], trace_records=[], chunk_rows=[])
        args = types.SimpleNamespace(max_chunks=5, record_trace=True, record_trace_every=1)
        env = types.SimpleNamespace(action_to_rotation=None)
        cfg = SpliceConfig(replan_every=12, commit_rows=1, bridge_rows=4)
        splices = run_splice_loop(
            env=env, chunk_source=Source(), q=q, sender=sender, args=args, rec=rec,
            dt_base=dt, obs_schema=None, gripper_enabled=False, gripper_unnormalize_fn=None,
            obs_buf=[], last_obs=[None], steps=[], cfg=cfg)
        # let the sender drain what was queued, then stop it
        time.sleep(0.2)
        q.put(None)
        sender.join(1.0)

        assert rec.chunk_count == 5 and rec.stopped_by == "normal"
        assert len(splices) == 5 and splices[0].n_bridge == 0
        for s in splices[1:]:
            assert s.k_s == s.k_obs + 2, s                    # commit_rows = 1: obs row + held row
            assert s.i_star == s.k_s + 4 - s.k_obs            # time-aligned entry
            assert s.n_bridge == 4 and not s.late
        # Replans came on the period, measured in published rows.
        assert all(b - a >= 12 for a, b in itertools.pairwise(requests))
        # The sender published frame indices in order with no gaps, and never had more
        # than one row queued behind the one it was holding.
        idx = [p[0] for p in sender.published]
        assert idx == list(range(len(idx)))
        assert max(p[3] for p in sender.published) <= 1
        # Cadence held: publishes one dwell apart, no late frames.
        t = np.array([p[1] for p in sender.published])
        assert np.abs(np.diff(t) - dt).max() < 0.008
        # And the executed stream is the straight line the policy kept predicting: the
        # splices were time-aligned, so the seams introduced no jump.
        x = np.array([p[2][0] for p in sender.published])
        assert np.abs(np.diff(x) - 1.0).max() < 0.05
