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
    GraspHold,
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
    cfg = SpliceConfig(commit_rows=1, bridge_rows=4, bridge="cubic", bridge_step_mm=0)   # rows are unit-less here

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


class TestAdaptiveBridge:
    def test_bridge_grows_so_no_row_steps_more_than_the_cap(self):
        cfg = SpliceConfig(bridge_rows=4, bridge_rows_max=12, bridge_step_mm=15.0)
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [0.001, 0, 0], 32), np.ones(32), k_obs=0, cfg=cfg)
        # New chunk 200 mm away in y: 4 rows would be 50 mm per row.
        new = line([0.012, 0.2, 0], [0.001, 0, 0], 32)
        info = plan.splice(12, new, np.ones(32), k_obs=10, cfg=cfg)
        assert info.n_bridge == 12                      # ceil(200 / 15) = 14, capped at 12
        assert info.i_star == 2 + 12
        steps = np.linalg.norm(np.diff(np.array(plan.actions)[11:11 + 14, :3], axis=0), axis=1) * 1000
        assert steps.max() < 30                         # vs ~50 with 4 rows

    def test_small_gaps_keep_the_nominal_bridge(self):
        cfg = SpliceConfig(bridge_rows=4, bridge_rows_max=12, bridge_step_mm=15.0)
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [0.001, 0, 0], 32), np.ones(32), k_obs=0, cfg=cfg)
        info = plan.splice(12, line([0.012, 0.01, 0], [0.001, 0, 0], 32), np.ones(32), k_obs=10, cfg=cfg)
        assert info.n_bridge == 4

    def test_zero_step_disables_the_adaptation(self):
        cfg = SpliceConfig(bridge_rows=4, bridge_step_mm=0.0)
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [0.001, 0, 0], 32), np.ones(32), k_obs=0, cfg=cfg)
        info = plan.splice(12, line([0, 0.3, 0], [0.001, 0, 0], 32), np.ones(32), k_obs=10, cfg=cfg)
        assert info.n_bridge == 4


class FakeScaler:
    """ReplayScaler's cached state after apply(), with kd on auto: kp 400 on the
    translation axes, 100 on rotation, as on the rig."""

    def __init__(self, kp=400.0, k_rot=100.0):
        self._original_kp = {**{f"task.k_pos_{a}": kp for a in "xyz"},
                             **{f"task.k_rot_{a}": k_rot for a in "xyz"}}
        self._original_kd = {**{f"task.d_pos_{a}": 0.0 for a in "xyz"},
                             **{f"task.d_rot_{a}": 0.0 for a in "xyz"}}
        self._kd_is_auto = {k: True for k in self._original_kd}
        self.kd_exp = 1.0
        self.restored = None

    def restore(self):
        self.restored = (dict(self._original_kd), dict(self._kd_is_auto))


class TestRaiseDamping:
    def test_pushes_ratio_times_auto_and_tracks_sqrt_kp(self):
        from pace_bench.real.gains import raise_damping
        sc = FakeScaler(kp=400.0)
        out = raise_damping(sc, kd_ratio=1.5, kp_exp=1.5)
        # 1.5 x 2 sqrt(400) = 60 on every axis, now non-auto, scaling as s^(1.5/2)
        assert out["kd_at_s1"]["task.d_pos_x"] == pytest.approx(60.0)
        assert out["kd_at_s1"]["task.d_rot_x"] == pytest.approx(30.0)   # 1.5 x 2 sqrt(100)
        assert not any(sc._kd_is_auto.values())
        assert sc.kd_exp == pytest.approx(0.75)

    def test_restore_puts_auto_back(self):
        from pace_bench.real.gains import raise_damping
        sc = FakeScaler()
        raise_damping(sc, kd_ratio=1.5, kp_exp=1.5)
        sc.restore()
        kd, auto = sc.restored
        assert all(v == 0.0 for v in kd.values()) and all(auto.values())

    def test_kd_base_is_pushed_verbatim_and_scales_with_kd_exp(self):
        from pace_bench.real.gains import raise_damping
        sc = FakeScaler(kp=400.0)
        out = raise_damping(sc, kp_exp=1.0, kd_exp=1.5, kd_base=100.0, kd_ratio=1.5)
        assert out["kd_base"] == 100.0 and out["kd_ratio"] is None
        assert sc.kd_exp == pytest.approx(1.5)          # kd_base wins over kd_ratio
        # translation gets 100; rotation is NOT touched -- it stays on auto
        for a in "xyz":
            assert sc._original_kd[f"task.d_pos_{a}"] == pytest.approx(100.0)
            assert sc._kd_is_auto[f"task.d_pos_{a}"] is False
            assert sc._original_kd[f"task.d_rot_{a}"] == 0.0
            assert sc._kd_is_auto[f"task.d_rot_{a}"] is True
        assert "task.d_rot_x" not in out["kd_at_s1"]

    def test_rotation_gets_its_own_base_only_when_asked(self):
        from pace_bench.real.gains import raise_damping
        sc = FakeScaler()
        out = raise_damping(sc, kp_exp=1.0, kd_exp=1.5, kd_base=100.0, kd_base_rot=30.0)
        assert out["kd_at_s1"]["task.d_rot_x"] == pytest.approx(30.0)
        assert sc._kd_is_auto["task.d_rot_x"] is False

    def test_ratio_one_or_no_scaler_changes_nothing(self):
        from pace_bench.real.gains import raise_damping
        sc = FakeScaler()
        assert raise_damping(sc, kd_ratio=1.0, kp_exp=1.5) is None
        assert all(sc._kd_is_auto.values()) and sc.kd_exp == 1.0
        assert raise_damping(None, kd_ratio=1.5, kp_exp=1.5) is None


class TestSpans:
    cfg = SpliceConfig(commit_rows=1, bridge_rows=4, bridge="cubic", bridge_step_mm=0)   # rows are unit-less here

    def test_rows_carry_the_raw_frames_they_advance(self):
        plan = Plan()
        raw0 = list(range(0, 64, 2))
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 32), np.ones(32), k_obs=0, cfg=self.cfg, raw_index=raw0)
        assert plan.spans[:3] == [2.0, 2.0, 2.0]
        # A chunk whose exemption kept raw 4..11 every frame: kept = 0,2,4,5,...,11,12,14,...
        raw1 = [0, 2] + list(range(4, 12)) + list(range(12, 12 + 2 * 22, 2))
        info = plan.splice(12, line([0, 0, 0], [1, 0, 0], 32), np.ones(32), k_obs=10, cfg=self.cfg, raw_index=raw1)
        # bridge rows 12..15 span raw[i*=6]=8 minus raw[i_s=2]=4 over 4 rows: 1 frame each
        assert plan.spans[12:16] == [1.0] * 4
        # then verbatim from out[6] = raw 8: 8->9->10->11 one frame, 11->12 one, 12->14 two
        assert plan.spans[16:21] == [1.0, 1.0, 1.0, 1.0, 2.0]
        assert info.i_star == 6

    def test_without_a_raw_index_every_row_is_one_frame(self):
        plan = Plan()
        plan.splice(0, line([0, 0, 0], [1, 0, 0], 8), np.ones(8), k_obs=0, cfg=self.cfg)
        assert plan.spans == [1.0] * 8


class TestGraspHold:
    def test_arms_on_the_close_edge_and_consumes_frames_by_span(self):
        h = GraspHold(4)
        assert h.step(1.0, 1) is None                 # open
        assert h.step(0.9, 1) is None                 # still open
        assert h.step(0.2, 1) == pytest.approx(1.0)   # edge: hold row 1 of 4 frames
        assert h.step(0.0, 2) == pytest.approx(0.5)   # strided row: 100 ms cap, uses 2 frames
        assert h.step(0.0, 1) == pytest.approx(1.0)   # 4th frame
        assert h.step(0.0, 1) is None                 # hold exhausted
        assert h.n_holds == 1

    def test_staying_closed_does_not_rearm_and_reopening_does_not_fire(self):
        h = GraspHold(2)
        h.step(1.0, 1); h.step(0.0, 1); h.step(0.0, 1)
        assert h.step(0.0, 1) is None and h.n_holds == 1
        assert h.step(1.0, 1) is None                 # opening: no hold
        assert h.step(0.0, 1) is not None and h.n_holds == 2   # a second grasp

    def test_inverted_channel(self):
        h = GraspHold(2, invert=True)
        assert h.step(0.0, 1) is None                 # 0 = open when inverted
        assert h.step(1.0, 1) == pytest.approx(1.0)   # 1 = closed

    def test_a_first_row_that_is_already_closed_is_not_an_edge(self):
        h = GraspHold(2)
        assert h.step(0.0, 1) is None and h.n_holds == 0


class TestFeedRules:
    def test_two_rows_go_out_before_anything_is_published(self):
        assert list(rows_to_push(0, -1)) == [0, 1]

    def test_then_exactly_one_row_per_publish(self):
        assert list(rows_to_push(2, 0)) == []          # row 0 executing, 1 held: nothing
        assert list(rows_to_push(2, 1)) == [2]         # row 1 published: hand over 2
        assert list(rows_to_push(3, 1)) == []

    def test_replan_on_the_period_or_when_the_plan_runs_out(self):
        assert needs_replan(0, -10**9, 0, 0, 12)               # nothing yet
        assert not needs_replan(5, 0, 7, 32, 12)
        assert needs_replan(12, 0, 14, 32, 12)                 # H rows since the last obs
        assert not needs_replan(5, 0, 31, 32, 12)              # one row still to push
        assert needs_replan(5, 0, 32, 32, 12)                  # nothing left to push

    def test_auto_period_uses_every_kept_row(self):
        # 32 kept: 1 committed + 4 bridge + 26 verbatim = 31 rows after the obs, so the
        # next observation is at row 30 and its splice (30 + 2) lands where the plan ends.
        assert SpliceConfig(replan_every=0, commit_rows=1).resolve_replan_every(32) == 30
        assert SpliceConfig(replan_every=0, commit_rows=2).resolve_replan_every(32) == 29
        assert SpliceConfig(replan_every=12).resolve_replan_every(32) == 12


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
    """crisp_gym's Python sender in miniature: pop, sleep to the deadline, publish."""

    def __init__(self, q):
        super().__init__(daemon=True)
        self.q, self.n_published, self.published = q, 0, []

    def _record(self, item):
        self.published.append((item.frame_idx, time.monotonic(), item.action.copy(),
                               self.q.qsize()))

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            time.sleep(max(0.0, item.deadline_mono - time.monotonic()))
            self._record(item)
            self.n_published += 1


class FakeCppSender(FakeSender):
    """The C++ handle's observable surface: `n_published` is stale until join(), and
    the live publish count is the stats ring head at offset 0 of `_stats_mm`."""

    def __init__(self, q):
        super().__init__(q)
        self._stats_mm = bytearray(64)
        self._head = 0

    def run(self):
        import struct
        while True:
            item = self.q.get()
            if item is None:
                return
            time.sleep(max(0.0, item.deadline_mono - time.monotonic()))
            self._record(item)
            self._head += 1
            struct.pack_into("<Q", self._stats_mm, 0, self._head)   # live, like crisp_sender.cpp:599
            # n_published deliberately NOT advanced: the real handle fills it at join().


def _stub_crisp_gym(monkeypatch, dt, keep_fn=None, speed=1.0):
    """Just enough of crisp_gym.deploy for the loop to import, with no robot.

    `keep_fn(chunk) -> indices` stands in for the method pipeline's striding and
    `speed` for its per-row multiplier; the timing stub honours the multiplier so a
    hold's dwell is observable in the fake sender's publish times.
    """
    def pipeline_stub():
        m = types.ModuleType("crisp_gym.deploy.pipeline")

        class Chunk:
            def __init__(self, actions, speeds):
                self.actions, self.speeds = actions, speeds

            @classmethod
            def nominal(cls, a):
                return cls(np.asarray(a), np.ones(len(a)))

        def run_pipeline(chunk, steps):
            a = chunk.actions
            if keep_fn is not None:
                a = a[keep_fn(a)]
            return Chunk(a, np.full(len(a), float(speed)))
        m.Chunk = Chunk
        m.run_pipeline = run_pipeline
        return m

    timing = types.ModuleType("crisp_gym.deploy.timing")
    timing.build_speed_queue_arrays = lambda s, dt_base, n, retime: (
        np.rint(dt_base / np.asarray(s, dtype=float) / 0.002).astype(int),
        dt_base / np.asarray(s, dtype=float), np.asarray(s, dtype=float))
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
    @pytest.mark.parametrize("sender_cls", [FakeSender, FakeCppSender])
    @pytest.mark.parametrize("period", [12, 0])     # explicit, and auto (= 32 - 1 - 1 = 30)
    def test_one_row_ahead_and_time_aligned_splices(self, monkeypatch, sender_cls, period):
        from pace_bench.real.splice_loop import published_count, run_splice_loop

        dt = 0.02
        _stub_crisp_gym(monkeypatch, dt)
        q = queue.Queue()
        sender = sender_cls(q)
        sender.start()
        requests = []

        class Source:
            def request(self, obs_buf):
                # The policy sees the row that is executing and predicts a straight line
                # from there, so consecutive chunks agree and the seam should be smooth.
                k = max(published_count(sender) - 1, 0)
                requests.append(k)
                time.sleep(0.006)                           # "inference"
                return line([float(k), 0, 0], [1, 0, 0], 32)

        rec = types.SimpleNamespace(
            chunk_count=0, stopped_by="init", starvation_event_count=0,
            stage_samples_producer={k: [] for k in
                                    ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms")},
            pred_dt_samples=[], trace_records=[], chunk_rows=[])
        args = types.SimpleNamespace(max_chunks=5 if period else 3, record_trace=True, record_trace_every=1)
        env = types.SimpleNamespace(action_to_rotation=None)
        cfg = SpliceConfig(replan_every=period, commit_rows=1, bridge_rows=4, bridge_step_mm=0)
        H = cfg.resolve_replan_every(32)
        splices = run_splice_loop(
            env=env, chunk_source=Source(), q=q, sender=sender, args=args, rec=rec,
            dt_base=dt, obs_schema=None, gripper_enabled=False, gripper_unnormalize_fn=None,
            obs_buf=[], last_obs=[None], steps=[], cfg=cfg, n_action_steps=32)
        # let the sender drain what was queued, then stop it
        time.sleep(0.2)
        q.put(None)
        sender.join(1.0)

        assert rec.chunk_count == args.max_chunks and rec.stopped_by == "normal"
        assert len(splices) == args.max_chunks and splices[0].n_bridge == 0
        for s in splices[1:]:
            assert s.k_s == s.k_obs + 2, s                    # commit_rows = 1: obs row + held row
            assert s.i_star == s.k_s + 4 - s.k_obs            # time-aligned entry
            assert s.n_bridge == 4 and not s.over_dwell
        # Replans came on the period, measured in published rows.
        assert all(b - a >= H for a, b in itertools.pairwise(requests))
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

    def test_grasp_in_the_discarded_head_still_gets_the_full_hold(self, monkeypatch):
        """The failure of the 17:00 run: the policy predicts the close at raw rows 4-6,
        the splice discards out[0..5] and bridges over them. The hold has to key on
        the command actually sent and run at demo cadence for the whole window."""
        from pace_bench.real.splice_loop import run_splice_loop

        dt = 0.02                      # demo cadence in this test
        EPS, WIN = 0.1, 10             # exemption window = hold window = 10 frames

        def keep(a):                   # stride 2, every frame within WIN of a gripper move
            g = a[:, 6]; moving = np.abs(np.diff(g)) > EPS
            keep = set(range(0, len(a), 2))
            for j in np.where(moving)[0]:
                keep.update(range(j, min(len(a), j + 2 + WIN)))
            return sorted(keep)[:32]

        _stub_crisp_gym(monkeypatch, dt, keep_fn=keep, speed=2.0)   # 2x: 10 ms rows outside the hold
        q = queue.Queue(); sender = FakeSender(q); sender.start()
        n_req = [0]

        class Source:
            def request(self, obs_buf):
                n_req[0] += 1
                c = line([0, 0, 0], [1, 0, 0], 100, grip=1.0)
                if n_req[0] == 2:      # second chunk: close predicted at raw rows 4..6
                    c[4:, 6] = 0.0
                return c

        rec = types.SimpleNamespace(
            chunk_count=0, stopped_by="init", starvation_event_count=0,
            stage_samples_producer={k: [] for k in
                                    ("get_obs_ms", "synth_ms", "build_ms", "push_ms", "drain_wait_ms")},
            pred_dt_samples=[], trace_records=[], chunk_rows=[])
        args = types.SimpleNamespace(max_chunks=3, record_trace=False, record_trace_every=1)
        cfg = SpliceConfig(replan_every=12, bridge_rows=4, hold_s=WIN * dt, fps=1 / dt, bridge_step_mm=0)
        run_splice_loop(env=types.SimpleNamespace(action_to_rotation=None), chunk_source=Source(),
                        q=q, sender=sender, args=args, rec=rec, dt_base=dt, obs_schema=None,
                        gripper_enabled=False, gripper_unnormalize_fn=None, obs_buf=[],
                        last_obs=[None], steps=[], cfg=cfg, n_action_steps=32, raw_index_fn=keep)
        time.sleep(0.3); q.put(None); sender.join(1.0)

        t = np.array([p[1] for p in sender.published])
        g = np.array([p[2][6] for p in sender.published])
        first_closed = int(np.argmax(g < 0.5))
        assert first_closed > 0, "the close command never went out"
        # The sender publishes AT each deadline, and deadline[i+1] = deadline[i] +
        # dt_eff[i+1], so the interval t[i+1] - t[i] is row i+1's dwell.
        dwell_of = np.diff(t)                      # dwell_of[i] = dwell of row i + 1
        row_dwell = np.r_[np.nan, dwell_of]        # index by row
        # Outside the hold rows go at 2x (10 ms); from the close command on, WIN rows
        # dwell at demo cadence (20 ms) -- the bridge rows included -- then 2x resumes.
        assert np.nanmedian(row_dwell[1:first_closed]) == pytest.approx(dt / 2, abs=0.004)
        held = row_dwell[first_closed:first_closed + WIN]
        assert np.all(np.abs(held - dt) < 0.006), held
        after = row_dwell[first_closed + WIN:first_closed + WIN + 5]
        assert np.median(after) == pytest.approx(dt / 2, abs=0.004), after

