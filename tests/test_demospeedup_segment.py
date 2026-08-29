"""Segmentation: parity with upstream, and the meaning of the two cluster rules."""

from pathlib import Path

import numpy as np
import pytest

from robot_stack.methods.demospeedup.segment import (
    NON_PRECISION,
    PRECISION,
    remove_outliers,
    segment,
)

GOLDEN = Path(__file__).parent / "assets" / "segment_golden.npz"


def golden_cases():
    data = np.load(GOLDEN)
    for name in sorted({key.split("__")[0] for key in data.files}):
        yield pytest.param(data[f"{name}__trace"], data[f"{name}__labels"], id=name)


@pytest.mark.parametrize(("trace", "expected"), list(golden_cases()))
def test_matches_upstream(trace, expected):
    """Bit-exact against lingxiao-guo/DemoSpeedup's own `hdbscan_with_custom_merge`.

    The vectors were produced by executing upstream's function out of a clone, with
    its own `hdbscan` backend -- see `tests/assets/gen_segment_golden.py`. Only the
    IsolationForest seed is pinned, which upstream leaves to chance.
    """
    assert np.array_equal(segment(trace, rule="upstream", seed=0), expected)


def test_golden_is_not_vacuous():
    """Guard against a golden file of all-zeros silently passing everything."""
    data = np.load(GOLDEN)
    labels = [data[k] for k in data.files if k.endswith("__labels")]
    assert len(labels) >= 6
    assert any(NON_PRECISION in lab for lab in labels), "no case exercises non-precision"
    assert any(PRECISION in lab for lab in labels), "no case exercises precision"


# --- the two cluster rules are genuinely different algorithms ----------------


def structured_trace():
    """Mostly precision, three stretches of confident transit. A real demo's shape."""
    rng = np.random.default_rng(0)
    trace = rng.normal(1.0, 0.25, 300)
    for start in (30, 120, 210):
        trace[start : start + 45] = rng.normal(3.2, 0.4, 45)
    return trace


def test_upstream_rule_almost_never_marks_non_precision():
    """Upstream calls a cluster precision unless *every* frame in it sits above +1 sigma.

    Clusters are at most `max_cluster_size` frames of a z-scored trace, so one quiet
    frame is enough to spare the whole cluster. On a well-separated trace that means
    essentially nothing is marked non-precision, and DemoSpeedup's retiming has no
    fast stretches to take. This is what the reference implementation does; the test
    exists so that fact stays visible rather than being rediscovered.
    """
    labels = segment(structured_trace(), rule="upstream", seed=0)
    assert labels.mean() < 0.05


def test_mean_rule_recovers_the_planted_transit_stretches():
    """The `mean` reading finds the three high-entropy stretches the trace was built from."""
    labels = segment(structured_trace(), rule="mean", seed=0)
    assert 0.35 < labels.mean() < 0.55

    padded = np.r_[0, labels, 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    assert len(starts) == 3, f"expected 3 non-precision runs, got {len(starts)}"
    # Long runs, not per-frame flicker: this is why time is a clustering feature.
    assert (ends - starts).min() > 20


def test_the_two_rules_disagree_on_a_real_trace():
    trace = structured_trace()
    assert not np.array_equal(segment(trace, rule="upstream", seed=0), segment(trace, rule="mean", seed=0))


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="rule must be"):
        segment(np.zeros(10), rule="whatever")


# --- degenerate inputs ------------------------------------------------------


@pytest.mark.parametrize("trace", [np.ones(50), np.zeros(50), np.full(3, 2.5)])
def test_flat_trace_is_all_precision(trace):
    """No spread means no evidence about where precision is needed: retime slowly.

    Upstream divides by a zero standard deviation here and produces NaNs.
    """
    labels = segment(trace)
    assert np.array_equal(labels, np.zeros(len(trace), dtype=np.int64))


def test_single_frame_episode():
    assert segment(np.array([1.0])).tolist() == [PRECISION]


def test_labels_are_binary_and_aligned():
    trace = structured_trace()
    for rule in ("upstream", "mean"):
        labels = segment(trace, rule=rule, seed=0)
        assert labels.shape == trace.shape
        assert labels.dtype == np.int64
        assert set(np.unique(labels)) <= {PRECISION, NON_PRECISION}


def test_is_reproducible():
    trace = structured_trace()
    assert np.array_equal(segment(trace, seed=7), segment(trace, seed=7))


def test_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1-D entropy trace"):
        segment(np.zeros((10, 2)))


# --- outlier removal --------------------------------------------------------


def test_outliers_are_replaced_not_dropped():
    """Length is load-bearing: labels are per-frame and must stay aligned to frames."""
    rng = np.random.default_rng(0)
    trace = rng.normal(0.0, 1.0, 100)
    trace[50] = 500.0
    cleaned = remove_outliers(trace, contamination=0.1, seed=0)
    assert len(cleaned) == len(trace)
    assert cleaned[50] < 10.0


def test_outlier_removal_leaves_the_input_untouched():
    trace = np.r_[np.zeros(50), 100.0, np.zeros(49)]
    before = trace.copy()
    remove_outliers(trace, seed=0)
    assert np.array_equal(trace, before)
