"""The LIBERO run comparator, and the episode indices that make pairing possible.

Two runs of `run_libero` attempt the same scenes in the same order -- LIBERO's init
states are fixed and upstream numbers episodes batch-major -- so the comparison that
matters is per-episode, not per-rate. These check that the pairing is actually
carried through the files, and that a run whose scene set differs is refused rather
than silently paired.
"""

from __future__ import annotations

import json

import gymnasium as gym
import pytest

from pace_bench.eval.compare_libero import (
    compare,
    format_report,
    format_summary,
    holm_adjust,
    load_run,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilson_interval,
)
from pace_bench.eval.sim_time import SimTimeRecorder

# --- fixtures ---------------------------------------------------------------


def write_run(root, *, tasks, successes, sim_times=None, config=None, indexed=True):
    """Write an eval output directory in `run_libero`'s own layout.

    `successes[task_id]` is the per-episode success list; `sim_times[task_id]` maps an
    episode index to its duration. With `indexed=False` the durations are written the
    way runs recorded before episode indices existed wrote them: a bare list.
    """
    base = {"task_suite": "libero_10", "n_episodes": 4, "batch_size": 2, "seed": 42}
    base.update(config or {})
    for task_id in tasks:
        ok = successes[task_id]
        times = (sim_times or {}).get(task_id, {})
        episodes = [
            ({"episode_index": i, "sim_time": t, "success": True, "reward": 1.0} if indexed
             else {"sim_time": t, "success": True, "reward": 1.0})
            for i, t in sorted(times.items())
        ]
        info = {
            "per_task": [{"task_group": "libero_10", "task_id": task_id, "metrics": {"successes": ok}}],
            "overall": {"pc_success": 100 * sum(ok) / len(ok), "n_episodes": len(ok)},
            "episodes": episodes,
            "config": base,
        }
        d = root / f"task_{task_id}"
        d.mkdir(parents=True)
        (d / "eval_info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def two_runs(tmp_path):
    """A reference and a faster-but-worse run over the same two tasks, 4 episodes each."""
    ref = write_run(
        tmp_path / "base",
        tasks=[0, 1],
        successes={0: [True, True, True, False], 1: [True, True, False, False]},
        sim_times={0: {0: 10.0, 1: 12.0, 2: 14.0}, 1: {0: 10.0, 1: 10.0}},
    )
    other = write_run(
        tmp_path / "fast",
        tasks=[0, 1],
        # Episode 0/2 lost, episode 0/3 won: one disagreement each way on task 0.
        successes={0: [True, True, False, True], 1: [True, False, False, False]},
        sim_times={0: {0: 5.0, 1: 8.0, 3: 6.0}, 1: {0: 5.0}},
    )
    return load_run(ref, "base"), load_run(other, "fast")


# --- statistics -------------------------------------------------------------


def test_mcnemar_matches_scipy():
    """Our exact test is scipy's two-sided binomial test on the discordant pairs."""
    binomtest = pytest.importorskip("scipy.stats").binomtest
    for b, c in [(21, 8), (1, 0), (5, 5), (0, 7), (13, 2)]:
        assert mcnemar_exact(b, c) == pytest.approx(binomtest(min(b, c), b + c, 0.5).pvalue)


def test_mcnemar_no_disagreement_is_not_evidence():
    """Two runs that agree on every episode say nothing about a difference."""
    assert mcnemar_exact(0, 0) == 1.0


def test_wilson_interval_stays_inside_the_unit_range():
    """The reason for Wilson rather than the normal approximation: 100% at n=20."""
    lo, hi = wilson_interval(20, 20)
    assert 0 <= lo < 100 and hi == pytest.approx(100.0)
    assert wilson_interval(0, 0) != wilson_interval(0, 1)  # empty is nan, not zero


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(184, 200)
    assert lo < 92.0 < hi


# --- loading ----------------------------------------------------------------


def test_load_run_reads_every_task(two_runs):
    ref, _ = two_runs
    assert sorted(ref.tasks) == [0, 1]
    assert ref.success_rate == pytest.approx(100 * 5 / 8)
    assert ref.avg_success_time == pytest.approx((10 + 12 + 14 + 10 + 10) / 5)


def test_load_run_skips_a_task_still_running(tmp_path):
    """A task directory with no eval_info.json is in flight, not empty."""
    root = write_run(tmp_path / "partial", tasks=[0], successes={0: [True] * 4})
    (root / "task_1").mkdir()
    run = load_run(root)
    assert sorted(run.tasks) == [0]


def test_load_run_rejects_a_directory_with_no_results(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "empty")


def test_legacy_timings_still_average(tmp_path):
    """A run recorded before episode indices contributes an ATR but no paired time."""
    root = write_run(
        tmp_path / "legacy",
        tasks=[0],
        successes={0: [True, True, False, False]},
        sim_times={0: {0: 10.0, 1: 20.0}},
        indexed=False,
    )
    run = load_run(root)
    assert run.avg_success_time == pytest.approx(15.0)
    assert run.tasks[0].sim_times == {}


# --- comparison -------------------------------------------------------------


def test_compare_counts_disagreements_not_rates(two_runs):
    ref, other = two_runs
    c = compare(ref, other)
    assert c.paired
    assert c.n_paired == 8
    assert (c.both, c.only_reference, c.only_other) == (3, 2, 1)
    assert c.delta_sr == pytest.approx(100 * (4 - 5) / 8)
    assert c.p_value == pytest.approx(mcnemar_exact(2, 1))


def test_compare_times_only_the_episodes_both_solved(two_runs):
    """Episode 0/2 is timed by the reference alone, and must not enter the average."""
    ref, other = two_runs
    c = compare(ref, other)
    assert c.n_time_paired == 3  # (0,0), (0,1), (1,0)
    assert c.ref_time == pytest.approx((10 + 12 + 10) / 3)
    assert c.other_time == pytest.approx((5 + 8 + 5) / 3)
    assert c.speedup == pytest.approx(c.ref_time / c.other_time)


def test_compare_refuses_to_pair_a_differently_configured_run(tmp_path, two_runs):
    """A different seed means different scenes, so the paired test does not apply."""
    ref, _ = two_runs
    other = load_run(
        write_run(
            tmp_path / "other_seed",
            tasks=[0, 1],
            successes={0: [True] * 4, 1: [True] * 4},
            config={"seed": 7},
        ),
        "seed7",
    )
    c = compare(ref, other)
    assert not c.paired
    assert c.n_paired == 0
    # Success rates remain comparable even when the episodes do not line up.
    assert c.delta_sr == pytest.approx(100.0 - ref.success_rate)


def test_compare_refuses_to_pair_a_different_task_set(tmp_path, two_runs):
    ref, _ = two_runs
    other = load_run(
        write_run(tmp_path / "one_task", tasks=[0], successes={0: [True] * 4}), "subset"
    )
    assert not compare(ref, other).paired


def test_report_names_every_run_and_the_test(two_runs):
    ref, other = two_runs
    report = format_report([ref, other], ref)
    assert "base" in report and "fast" in report
    assert "McNemar" in report
    assert "t0" in report and "t1" in report


def test_report_flags_an_unpaired_run(tmp_path, two_runs):
    ref, _ = two_runs
    other = load_run(
        write_run(
            tmp_path / "other_seed",
            tasks=[0, 1],
            successes={0: [True] * 4, 1: [True] * 4},
            config={"seed": 7},
        ),
        "seed7",
    )
    assert "NOT paired" in format_report([ref, other], ref)


# --- the indices themselves -------------------------------------------------


class _FakeEnv(gym.Env):
    """Terminates after `steps_to_done` steps, or never if that is None."""

    def __init__(self, steps_to_done):
        self.steps_to_done = steps_to_done
        self.n = 0

    def step(self, action):
        self.n += 1
        done = self.steps_to_done is not None and self.n >= self.steps_to_done
        if done:
            self.n = 0
        return None, 1.0 if done else 0.0, done, False, {}

    def reset(self, **kwargs):
        return None, {}


def test_recorder_numbers_episodes_batch_major():
    """Episode index = batch * n_envs + env index -- upstream's own numbering.

    Position in `episodes` cannot stand in for it: a failing episode usually runs to
    the step cap without terminating, so env 1 below records nothing for batch 0 and
    its first entry belongs to batch 1.
    """
    fast, slow = SimTimeRecorder(_FakeEnv(1), env_index=0, stride=2), SimTimeRecorder(_FakeEnv(3), 1, 2)
    for batch in range(2):
        for rec in (fast, slow):
            rec.arm()
        for _ in range(2 if batch == 0 else 3):  # batch 0 too short for the slow env
            for rec in (fast, slow):
                rec.step(None)

    assert [e["episode_index"] for e in fast.episodes] == [0, 2]
    assert [e["episode_index"] for e in slow.episodes] == [3]
    assert all(e["success"] for e in fast.episodes)


def test_recorder_takes_only_the_first_termination_of_a_batch():
    """An env that finishes early is auto-reset and keeps stepping within the batch."""
    rec = SimTimeRecorder(_FakeEnv(1), env_index=0, stride=1)
    rec.arm()
    for _ in range(5):
        rec.step(None)
    assert [e["episode_index"] for e in rec.episodes] == [0]


# --- the closing summary ----------------------------------------------------


def test_holm_leaves_a_single_comparison_alone():
    """One arm against one baseline is not a multiple-comparison problem."""
    assert holm_adjust([0.03]) == [0.03]


def test_holm_penalises_the_weakest_evidence_most():
    adjusted = holm_adjust([0.01, 0.04, 0.5])
    assert adjusted == pytest.approx([0.03, 0.08, 0.5])
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.5], strict=True))


def test_holm_is_monotone_in_the_original_order():
    """Adjustment must not reorder the evidence -- a smaller p stays smaller."""
    raw = [0.2, 0.001, 0.049, 0.6]
    adjusted = holm_adjust(raw)
    assert sorted(range(4), key=lambda i: raw[i]) == sorted(range(4), key=lambda i: adjusted[i])


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    pairs = [(10.0, 5.0)] * 20 + [(10.0, 9.0)] * 20
    stat = lambda s: sum(a for a, _ in s) / sum(b for _, b in s)  # noqa: E731
    first = paired_bootstrap_ci(pairs, stat, n_resamples=500)
    assert first == paired_bootstrap_ci(pairs, stat, n_resamples=500)
    assert first[0] < stat(pairs) < first[1]


def test_bootstrap_ci_collapses_when_every_pair_is_identical():
    """No spread in the data, no spread in the interval."""
    lo, hi = paired_bootstrap_ci([(10.0, 5.0)] * 30, lambda s: s[0][0] / s[0][1], n_resamples=200)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_summary_reports_the_paired_speedup_and_its_sign_test(two_runs):
    ref, other = two_runs
    text = format_summary([ref, other], ref)
    assert "faster on 3/3 shared successes" in text
    assert "on the episodes both solved" in text
    # 3 of 8 episodes disagree, which at n=8 is not significant.
    assert "no detectable success difference" in text


def test_summary_states_how_much_of_the_benchmark_was_covered(two_runs):
    ref, other = two_runs
    assert "4 of the 50 initial states" in format_summary([ref, other], ref)


def test_summary_refuses_a_verdict_on_an_unpaired_run(tmp_path, two_runs):
    ref, _ = two_runs
    other = load_run(
        write_run(
            tmp_path / "other_seed",
            tasks=[0, 1],
            successes={0: [True] * 4, 1: [True] * 4},
            config={"seed": 7},
        ),
        "seed7",
    )
    text = format_summary([ref, other], ref)
    assert "not comparable episode by episode" in text
    assert "NOT shared by seed7" in text


def test_summary_applies_holm_only_with_several_arms(two_runs, tmp_path):
    ref, other = two_runs
    assert "Holm" not in format_summary([ref, other], ref)
    third = load_run(
        write_run(
            tmp_path / "third",
            tasks=[0, 1],
            successes={0: [False] * 4, 1: [False] * 4},
            sim_times={},
        ),
        "third",
    )
    assert "Holm" in format_summary([ref, other, third], ref)


def test_report_ends_with_the_summary(two_runs):
    ref, other = two_runs
    assert format_report([ref, other], ref).endswith(format_summary([ref, other], ref))


def test_batch_size_does_not_break_pairing(tmp_path, two_runs):
    """Episode k is scene k at any batch size, so a smaller batch still pairs.

    Upstream seeds episode k with `start_seed + k` and LIBERO hands it init state k
    regardless of how many envs run at once -- which is what lets an arm sharing a GPU
    run at a smaller batch than one that has the card to itself.
    """
    ref, _ = two_runs
    other = load_run(
        write_run(
            tmp_path / "small_batch",
            tasks=[0, 1],
            successes={0: [True, True, True, False], 1: [True, True, False, False]},
            config={"batch_size": 2},
        ),
        "batch2",
    )
    c = compare(ref, other)
    assert c.paired
    assert c.n_paired == 8
