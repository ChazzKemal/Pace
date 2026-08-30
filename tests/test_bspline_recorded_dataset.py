"""Reproduce the recorded B-spline dataset from the demonstrations it was built from.

Parity against the upstream copy shows the maths is right. This shows that the whole
chain -- cart7 to 6D rotation, adaptive fit, chunking, per-frame knot shift --
reproduces `merged_bspline_20260528` from `merged_act_finetune_20260528` exactly,
which is what lets that dataset stop being an input we cannot regenerate. The
converter that originally produced it lived in the crisp_gym fork and is gone, so
this test *is* the surviving specification of the conversion.

`meta/bspline.json` records the parameters it was built with, and the test reads
them rather than hardcoding, so a dataset built with different settings is checked
against its own settings or not at all.

Skips when the datasets are not on this machine.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pace_bench.methods.bspline import (
    assign_chunks_to_frames,
    chunk_parameters,
    encode_relative_knots,
    fit_episode,
    to_spline_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PACE_DATA_ROOT", REPO_ROOT.parent / "data"))
SOURCE = DATA_ROOT / "datasets" / "real" / "merged_act_finetune_20260528"
BSPLINE = DATA_ROOT / "datasets" / "real" / "merged_bspline_20260528"

pytestmark = pytest.mark.skipif(
    not (SOURCE.is_dir() and BSPLINE.is_dir()),
    reason="the recorded UR10e datasets are not on this machine",
)

# float32 on disk; the fit itself agrees far more tightly than this.
STORAGE_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def provenance():
    return json.loads((BSPLINE / "meta" / "bspline.json").read_text())


@pytest.fixture(scope="module")
def reconstruction(provenance):
    """Fit every episode once and collect what both checks below need.

    One pass, because the adaptive fit is the expensive part of the whole method
    (`generate_knots` fits a spline per candidate knot vector) and running it twice
    doubles the slowest test in the suite for nothing.
    """
    source = sorted(glob.glob(str(SOURCE / "data" / "chunk-000" / "file-*.parquet")))
    target = sorted(glob.glob(str(BSPLINE / "data" / "chunk-000" / "file-*.parquet")))
    assert source and len(source) == len(target)

    shape = (provenance["n_action_steps"], provenance["n_action_channels"])
    degree = provenance["degree"]
    deviations, knots_per_frame = [], []

    for episode, (source_path, target_path) in enumerate(zip(source, target, strict=True)):
        raw = np.stack(pd.read_parquet(source_path)["action"].values).astype(np.float64)
        recorded = np.stack(pd.read_parquet(target_path)["action"].values).astype(np.float64)
        assert recorded.shape == (len(raw), provenance["flat_action_dim"])

        # Fit once and reuse: the adaptive fit is ~1.4 s per episode and dominates
        # this test entirely.
        spline, converged = fit_episode(
            to_spline_actions(raw), max_error=provenance["max_error"], degree=degree
        )
        assert converged, f"episode {episode} did not reach max_error={provenance['max_error']}"
        chunks = chunk_parameters(spline, provenance["chunk_size"], degree=degree, stride=1)
        rebuilt = assign_chunks_to_frames(chunks, len(raw), degree=degree)
        deviations.append(float(np.abs(rebuilt - recorded.reshape(len(raw), *shape)).max()))

        # Unique knots -- upstream's `extract_unique_knots`, which drops the repeated
        # boundary knots at each end. That is the count the recorded metadata used.
        knots_per_frame.append(len(spline.tck[0][degree:-degree]) / len(raw))

    return deviations, knots_per_frame


def test_provenance_matches_this_port_s_conventions(provenance):
    """The recorded settings are the ones this module implements, including the
    rotation layout -- reading the 6D block as columns would transpose every frame."""
    assert provenance["representation"] == "bspline_policy_action_chunk"
    assert provenance["degree"] == 3
    assert provenance["stride"] == 1
    assert provenance["relative_knots"] is False
    assert provenance["n_action_steps"] == provenance["chunk_size"] + 2 * provenance["degree"]
    assert provenance["n_action_channels"] == len(provenance["control_point_names"]) + 1
    assert provenance["flat_action_dim"] == (
        provenance["n_action_steps"] * provenance["n_action_channels"]
    )
    assert provenance["policy_action_layout"] == "xyz(3) + rot6d(6) + gripper(1)"
    assert provenance["knot_units"] == "source frames, relative to the current frame"


def test_every_episode_reconstructs(reconstruction):
    deviations, _ = reconstruction
    worst = max(deviations)
    assert worst < STORAGE_TOLERANCE, (
        f"worst episode differs by {worst:.3e}; "
        f"{sum(d >= STORAGE_TOLERANCE for d in deviations)} of {len(deviations)} episodes fail"
    )


def test_the_fit_statistics_match_what_was_recorded(provenance, reconstruction):
    """`meta/bspline.json` records how tight the fit came out. Reproducing the
    parameter matrices but not these numbers would mean matching by coincidence."""
    _, knots_per_frame = reconstruction
    assert np.mean(knots_per_frame) == pytest.approx(
        provenance["fit"]["knots_per_frame_mean"], rel=1e-9
    )
    assert max(knots_per_frame) == pytest.approx(
        provenance["fit"]["knots_per_frame_max"], rel=1e-9
    )


def test_relative_knots_flatten_the_row_ramp():
    """Why a B-spline arm should train on `relative_knots=True`.

    Absolute knots are mostly decided by which row they sit in -- the column averages
    -7.7 at row 0 and +50.9 at row 25 -- so the single per-column statistic LeRobot's
    normaliser computes leaves that ramp in the regression target. Encoding the column
    as consecutive differences makes it stationary across rows, which is what lets the
    stock normaliser be used instead of the step owning normalisation itself.

    This lives with the recorded data rather than beside the synthetic parity tests
    because it is a claim about *demonstrations*: a smooth synthetic path fits with
    near-uniform knots, whose differences have almost no spread, and the ratio below
    is then meaningless.
    """
    parameters = np.concatenate([
        np.stack(pd.read_parquet(f)["action"].values)
        for f in sorted(glob.glob(str(BSPLINE / "data" / "chunk-000" / "file-*.parquet")))
    ]).astype(np.float64)
    provenance = json.loads((BSPLINE / "meta" / "bspline.json").read_text())
    parameters = parameters.reshape(
        len(parameters), provenance["n_action_steps"], provenance["n_action_channels"]
    )

    def ramp(column):
        """Row-to-row spread over within-row spread. >1 means row index dominates."""
        return column.mean(axis=0).std() / column.std(axis=0).mean()

    absolute = parameters[:, :, 0]
    relative = encode_relative_knots(parameters, degree=provenance["degree"])[:, :, 0]

    assert ramp(absolute) > 1.5, ramp(absolute)
    assert ramp(relative) < 0.5, ramp(relative)
    # and for scale: a control-point column has essentially no row dependence
    assert ramp(parameters[:, :, 1]) < 0.1
