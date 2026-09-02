"""The chunk timeline, on a run whose clock is known in advance.

The reconstruction rests on three rules of the deploy loop -- deadlines chain from
the previous chunk's last one, a ``fresh`` chunk re-anchors at its push, and the blend
rewrites raw rows before the pipeline strides them. Each is checked here on a
synthetic run where the right answer follows from the rule, and the ChunkClock is
checked to record the same quantities the reconstruction infers.
"""

import types

import numpy as np
import pytest

from pace_bench.real.record import ChunkClock, command_table
from pace_bench.real.timeline import (
    RunFiles,
    bridge_rows,
    chunk_bounds,
    deadlines,
    describe,
    published,
    reconstruct,
    stages,
)

DT = 0.002          # control cycle
CYC = 25            # 50 ms frames
T0 = 1000.0


def synthetic_run(n_chunks=3, k=8, q=2, inf_s=0.02, stride=1, overlap=0,
                  with_trace=True, with_inference=False, fresh_at=None):
    """A run whose loop behaved exactly as loop.py says it does.

    Chunk c+1 is requested when q rows of chunk c remain, inference takes inf_s, and
    its rows are anchored after chunk c's last deadline. Push timestamps are the
    request return time plus a few ms, as the sender's replay log records.
    """
    frame = CYC * DT
    t_wall, cycles, chunk_col = [], [], []
    req, ret, obs, q_log, modes = [], [], [], [], []
    last_deadline = None
    for c in range(n_chunks):
        if c == 0:
            t_req = T0 - 0.3
            anchor = T0
            modes.append("fresh")
        else:
            # The sender pops the next item right after publishing one and sleeps to
            # its deadline, so the queue reads q once row k-2-q has gone out; the
            # producer polls every 5 ms. q+1 rows are then still to be published.
            t_req = last_deadline - (q + 1) * frame + 0.005
            if fresh_at == c:
                anchor = t_req + inf_s + 0.5      # pushed late, after the queue emptied
                modes.append("fresh")
            else:
                anchor = last_deadline
                modes.append("overlap")
        t_ret = t_req + inf_s
        push = anchor if c == 0 else max(t_ret + 0.003, anchor if modes[-1] == "fresh" else 0)
        for _ in range(k):
            t_wall.append(push)
            cycles.append(CYC)
            chunk_col.append(c)
        last_deadline = anchor + k * frame
        req.append(t_req); ret.append(t_ret); obs.append(t_req - 0.002)
        q_log.append(0 if c == 0 else q)

    cmd = {"t_wall": np.array(t_wall), "cycles": np.array(cycles, dtype=float),
           "frame_index": np.arange(len(t_wall), dtype=float)}
    chunks = {"anchor_mode": np.array(modes, dtype=object),
              "synth_ms": np.array([inf_s * 1000] * n_chunks),
              "get_obs_ms": np.array([2.0] * n_chunks),
              "q_before_inf": np.array(q_log, dtype=float)}
    rc = {"method": {"type": "pace", "action_stride": stride} if stride > 1 else {"type": "none"},
          "n_action_steps": k + overlap, "blend": {"overlap": overlap, "mode": "hermite"}}
    trace = None
    if with_trace:
        n_raw = 2 * (k + overlap) * stride
        trace = {"chunk_idx": np.arange(n_chunks), "wall_ns": (np.array(ret) * 1e9).astype(np.int64),
                 "chunk": np.zeros((n_chunks, n_raw, 7), dtype=np.float32)}
    inference = None
    if with_inference:
        inference = {"chunk": np.arange(n_chunks, dtype=float), "t_req_wall": np.array(req),
                     "t_ret_wall": np.array(ret), "inference_ms": np.array([inf_s * 1000] * n_chunks),
                     "q_at_req": np.array(q_log, dtype=float)}
    summary = {"control_dt_ms": 2.0, "fps_baseline": 20.0,
               "args": {"overlap_threshold": q, "n_act": k + overlap}}
    files = RunFiles(cmd=cmd, chunks=chunks, trace=trace, inference=inference,
                     summary=summary, rc=rc)
    return files, {"frame": frame, "req": req, "ret": ret, "obs": obs}


class TestChunkBounds:
    def test_from_push_gaps(self):
        files, _ = synthetic_run(n_chunks=3, k=8)
        assert chunk_bounds(files.cmd) == [(0, 8), (8, 16), (16, 24)]

    def test_recorded_chunk_column_wins_over_gaps(self):
        # Same push time on every row -- no gap to detect -- but the column says 2+2.
        cmd = {"t_wall": np.full(4, T0), "cycles": np.full(4, CYC),
               "chunk": np.array([0, 0, 1, 1], dtype=float)}
        assert chunk_bounds(cmd) == [(0, 2), (2, 4)]


class TestDeadlines:
    def test_overlap_chunks_chain_from_the_previous_last_deadline(self):
        files, w = synthetic_run(n_chunks=3, k=4)
        t = deadlines(files.cmd, DT, chunk_bounds(files.cmd), files.chunks["anchor_mode"])
        # One unbroken 50 ms grid from the first anchor, regardless of when pushes happened
        assert np.allclose(np.diff(t), w["frame"])
        assert t[0] == pytest.approx(T0 + w["frame"])

    def test_a_fresh_chunk_reanchors_at_its_push(self):
        files, w = synthetic_run(n_chunks=3, k=8, fresh_at=2)
        b = chunk_bounds(files.cmd)
        t = deadlines(files.cmd, DT, b, files.chunks["anchor_mode"])
        lo = b[2][0]
        assert t[lo] == pytest.approx(files.cmd["t_wall"][lo] + w["frame"])
        assert t[lo] - t[lo - 1] > w["frame"] * 5   # the gap the queue ran dry for

    def test_recorded_deadlines_are_used_verbatim(self):
        files, _ = synthetic_run(n_chunks=2, k=3)
        files.cmd["deadline_wall"] = np.arange(6, dtype=float) + 5.0
        t = deadlines(files.cmd, DT, chunk_bounds(files.cmd), None)
        assert np.array_equal(t, files.cmd["deadline_wall"])


class TestPublished:
    def test_on_time_frames_publish_at_the_deadline_plus_overshoot(self):
        dl = np.array([1.0, 1.05, 1.10])
        fr = {"slack_ms": np.array([30.0, 30.0, 30.0]),
              "sleep_overshoot_ms": np.array([0.1, 0.1, 0.1])}
        assert np.allclose(published(dl, fr), dl + 0.0001)

    def test_late_frames_are_shifted_by_their_slack(self):
        dl = np.array([1.0, 1.05])
        fr = {"slack_ms": np.array([-20.0, 5.0]), "sleep_overshoot_ms": np.zeros(2)}
        p = published(dl, fr)
        assert p[0] == pytest.approx(1.02)
        assert p[1] == pytest.approx(1.05)

    def test_rows_the_sender_never_reached_are_nan(self):
        dl = np.array([1.0, 1.05, 1.10])
        fr = {"slack_ms": np.array([30.0]), "sleep_overshoot_ms": np.zeros(1)}
        p = published(dl, fr)
        assert np.isfinite(p[0]) and np.isnan(p[1]) and np.isnan(p[2])


class TestReconstruct:
    def test_latency_is_queue_plus_in_flight_plus_anchor_step(self):
        # q rows queued at the request, the sender holding one more, and the new
        # chunk anchored one frame after the last deadline: (q + 1 + 1) frames, less
        # the 5 ms poll and plus the 2 ms get_obs the synthetic run bakes in.
        files, w = synthetic_run(n_chunks=4, k=8, q=2)
        tl = reconstruct(files)
        expect = (2 + 1 + 1) * w["frame"] * 1000 - 5.0 + 2.0
        for c in tl.chunks[1:]:
            assert c.latency_first_ms == pytest.approx(expect, abs=0.5)
            assert c.inference_ms == pytest.approx(20.0)
            assert c.q_logged == 2
            assert c.q_measured == 3     # q queued + the one already popped
        assert tl.inference_source == "reconstructed"
        assert tl.median_latency_first_ms == pytest.approx(expect, abs=0.5)

    def test_measured_inference_is_preferred_and_covers_the_last_chunk(self):
        files, w = synthetic_run(n_chunks=3, k=8, with_inference=True)
        # chunks.csv lost its last row, as it does on Ctrl-C mid-drain
        for key in ("synth_ms", "get_obs_ms", "q_before_inf", "anchor_mode"):
            files.chunks[key] = files.chunks[key][:-1]
        tl = reconstruct(files)
        assert tl.inference_source == "measured"
        assert tl.chunks[-1].t_req == pytest.approx(w["req"][-1])
        assert tl.chunks[-1].inference_ms == pytest.approx(20.0)

    def test_without_trace_or_inference_the_clock_is_still_built(self):
        files, _ = synthetic_run(n_chunks=3, k=8, with_trace=False)
        tl = reconstruct(files)
        assert tl.inference_source == "none"
        assert len(tl.chunks) == 3
        assert all(c.t_req is None for c in tl.chunks)
        assert tl.chunks[1].t_first > tl.chunks[0].t_last

    def test_bridge_rows_follow_the_stride(self):
        # overlap 8 rewrites raw rows 0..7; with stride 2 only 0,2,4,6 execute.
        files, w = synthetic_run(n_chunks=3, k=8, stride=2, overlap=8)
        tl = reconstruct(files)
        assert tl.chunks[0].n_bridge == 0
        assert tl.chunks[1].n_bridge == 4
        c = tl.chunks[1]
        assert c.latency_policy_ms == pytest.approx(c.latency_first_ms + 4 * w["frame"] * 1000)
        assert tl.stride_mismatch

    def test_bridge_rows_at_stride_one(self):
        files, _ = synthetic_run(n_chunks=2, k=8, stride=1, overlap=8)
        tl = reconstruct(files)
        assert tl.chunks[1].n_bridge == 8
        assert not tl.stride_mismatch


class TestStages:
    def test_bridge_count_is_emitted_rows_below_n(self):
        rc = {"blend": {"overlap": 8}}
        assert bridge_rows(rc, 100, list(range(0, 100, 2))) == 4
        assert bridge_rows(rc, 100, list(range(100))) == 8
        assert bridge_rows({"blend": {"overlap": 0}}, 100, list(range(100))) == 0

    def test_plain_stride_needs_no_torch(self):
        pred = np.zeros((100, 7))
        st = stages(pred, {"method": {"type": "pace", "action_stride": 2},
                           "n_action_steps": 32, "blend": {"overlap": 8}})
        assert st["strided"] == list(range(0, 100, 2))
        assert len(st["after_truncate"]) == 32
        assert len(st["emitted"]) == 24 and len(st["blend_held"]) == 8
        assert st["exempt_added"] == []


class TestDescribe:
    def test_names_the_latency_and_the_stride_mismatch(self):
        files, _ = synthetic_run(n_chunks=3, k=8, stride=2, overlap=8)
        titles = [t for t, _ in describe(reconstruct(files))]
        assert any("later than it was predicted" in t for t in titles)
        assert any("mismatched units" in t for t in titles)

    def test_says_so_when_nothing_places_inference(self):
        files, _ = synthetic_run(n_chunks=3, k=8, with_trace=False)
        titles = [t for t, _ in describe(reconstruct(files))]
        assert titles == ["Chunk timing not available"]


class TestChunkClock:
    def _item(self, idx, deadline_mono):
        return types.SimpleNamespace(frame_idx=idx, deadline_mono=deadline_mono,
                                     s_eff=1.0, cycles=25, action=np.zeros(7))

    def test_tags_each_push_with_the_chunk_that_produced_it(self):
        pushed = []
        q = types.SimpleNamespace(qsize=lambda: 3, put=pushed.append)
        src = types.SimpleNamespace(request=lambda obs: "chunk")
        clock = ChunkClock(q)
        clock.wrap_source(src)
        clock.tap_queue(q)

        assert src.request(None) == "chunk"
        q.put(self._item(0, 10.0)); q.put(self._item(1, 10.05))
        src.request(None)
        q.put(self._item(2, 10.10))
        q.put(None)      # the shutdown sentinel passes through untouched

        assert pushed[-1] is None and len(pushed) == 4
        assert [p["chunk"] for p in clock.pushes] == [0, 0, 1]
        assert [i["chunk"] for i in clock.inferences] == [0, 1]
        assert clock.inferences[0]["q_at_req"] == 3
        assert clock.inferences[0]["t_ret_wall"] >= clock.inferences[0]["t_req_wall"]
        # The deadline is carried onto the wall axis with the offset read at the push,
        # so two pushes 50 ms apart in monotonic deadline are 50 ms apart in wall.
        gap = clock.pushes[1]["deadline_wall"] - clock.pushes[0]["deadline_wall"]
        assert gap == pytest.approx(0.05, abs=0.005)

    def test_a_failed_request_is_not_counted_as_a_chunk(self):
        def boom(obs):
            raise RuntimeError("pipe closed")
        q = types.SimpleNamespace(qsize=lambda: 0, put=lambda i: None)
        src = types.SimpleNamespace(request=boom)
        clock = ChunkClock(q)
        clock.wrap_source(src)
        with pytest.raises(RuntimeError):
            src.request(None)
        assert clock.inferences == []

    def test_command_table_joins_by_position_and_checks_frame_index(self):
        q = types.SimpleNamespace(qsize=lambda: 0, put=lambda i: None)
        clock = ChunkClock(q)
        clock.tap_queue(q)
        clock._chunk = 0
        q.put(self._item(0, 5.0)); q.put(self._item(1, 5.05))
        log = [{"frame_index": 0, "timestamp": 1.0, "replay.s_eff": 1.0,
                "replay.cycles": 25, "replay.action": np.zeros(7)},
               {"frame_index": 1, "timestamp": 1.0, "replay.s_eff": 1.0,
                "replay.cycles": 25, "replay.action": np.zeros(7)}]
        fields, rows = command_table(log, clock)
        assert "chunk" in fields and "deadline_wall" in fields
        assert rows[1]["deadline_wall"] - rows[0]["deadline_wall"] == pytest.approx(0.05, abs=1e-3)

        # A mismatch leaves the columns out rather than writing them wrong.
        log[1]["frame_index"] = 7
        fields, rows = command_table(log, clock)
        assert "chunk" not in fields
