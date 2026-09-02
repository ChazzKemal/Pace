"""The report's arithmetic, on signals whose answer is known in advance.

`fit_axis` is the load-bearing piece: everything the report concludes rests on it
recovering a delay correctly and, just as importantly, on it *not* reporting a good
fit when the arm failed to reach where it was told. Both are checked against
synthetic trajectories, because on real data neither has a ground truth.
"""

import numpy as np
import pytest

from pace_bench.real.report import (
    Analysis,
    AxisFit,
    command_timeline,
    fit_axis,
    verdict,
)


def signals(lag_s=0.0, gain=1.0, fs=50.0, dur=20.0, f=0.15):
    """A commanded sine and an achieved copy, delayed and/or scaled by known amounts."""
    t = np.arange(0, dur, 1 / fs)
    cmd = np.sin(2 * np.pi * f * t)
    ach = gain * np.sin(2 * np.pi * f * (t - lag_s))
    return t, cmd, ach


class TestCommandTimeline:
    def test_it_uses_integer_controller_cycles(self):
        cmd = {"cycles": np.array([10.0, 20.0, 10.0]), "t_wall": np.array([100.0] * 3)}
        t = command_timeline(cmd, 0.002)
        # first row starts at the anchor; each subsequent start adds the previous dwell
        assert t[0] == pytest.approx(100.0)
        assert t[1] == pytest.approx(100.02)   # 10 cycles x 2 ms
        assert t[2] == pytest.approx(100.06)   # + 20 cycles x 2 ms


class TestFitAxis:
    def test_it_recovers_a_known_delay(self):
        t, cmd, ach = signals(lag_s=0.30)
        f = fit_axis("x", t[50:-50], t, cmd, t, ach)
        assert f.lag_ms == pytest.approx(300, abs=20)
        assert f.rms_aligned_mm < 0.05 * f.rms_naive_mm

    def test_zero_delay_is_reported_as_zero(self):
        t, cmd, ach = signals(lag_s=0.0)
        f = fit_axis("x", t[50:-50], t, cmd, t, ach)
        assert f.lag_ms == pytest.approx(0, abs=20)

    def test_amplitude_loss_is_detected_and_not_explained_away_by_lag(self):
        # The arm reaches only 60% of the commanded swing, with no delay. Shifting in
        # time cannot fix a scale error, so alignment must buy almost nothing -- that
        # gap is the signature the report keys on.
        t, cmd, ach = signals(lag_s=0.0, gain=0.6)
        f = fit_axis("x", t[50:-50], t, cmd, t, ach)
        assert f.amplitude_ratio == pytest.approx(0.6, abs=0.03)
        assert f.lag_improvement < 0.25

    def test_pure_delay_leaves_amplitude_intact(self):
        t, cmd, ach = signals(lag_s=0.25, gain=1.0)
        f = fit_axis("x", t[50:-50], t, cmd, t, ach)
        assert f.amplitude_ratio == pytest.approx(1.0, abs=0.03)
        assert f.lag_improvement > 0.8


class TestVerdict:
    def _analysis(self, ratio, lag):
        a = Analysis()
        a.fits = [AxisFit(ax, lag, 10.0, 20.0, ratio, 0.9) for ax in "xyz"]
        a.kp = [400.0, 400.0, 400.0]
        return a

    def test_healthy_tracking_says_so(self):
        v = verdict(self._analysis(0.98, 40), {})
        assert any("healthy" in t.lower() for t, _ in v)

    def test_amplitude_loss_refuses_to_name_a_cause_from_one_run(self):
        # The earlier version blamed speed; a 1.0x baseline showed the same ratio.
        v = verdict(self._analysis(0.62, 40), {})
        body = " ".join(d for _, d in v)
        assert "1.0x" in body and "compare" in body

    def test_it_flags_scale_kp_being_off(self):
        v = verdict(self._analysis(0.62, 40), {"gains": {"scale_kp": False}})
        assert "scale_kp" in " ".join(d for _, d in v)

    def test_large_lag_is_related_to_the_gains(self):
        v = verdict(self._analysis(0.98, 700), {})
        body = " ".join(d for _, d in v)
        assert "400" in body and "phase margin" in body
