"""The B-spline port, against the verbatim upstream copy and against its own claims.

Parity here is what licenses calling this "B-spline Policy" rather than "a spline
fit we wrote": :mod:`tests.upstream_reference_bspline` holds the original functions
and every parity test runs them live. The property tests cover the two things
upstream does not state and our stack depends on -- that a chunk spans more source
frames than the samples drawn from it (the speed lever) and that the 6D rotation
survives a round trip through the manifold projection.
"""

import numpy as np
import pytest
from upstream_reference_bspline import (
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    decode_bspline_action,
)

from pace_bench.methods.bspline import (
    DEGREE,
    chunk_parameters,
    decode_chunk,
    episode_parameter_chunks,
    fit_episode,
    from_spline_actions,
    to_spline_actions,
)

CHUNK = 20  # the recorded UR10e dataset's geometry
MAX_ERROR = 0.01


def trajectory(length: int = 200, seed: int = 0, dim: int = 10) -> np.ndarray:
    """A smooth, non-uniform-speed path: fast transits and slow dwells, as recorded."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    # a few random low-frequency components -- smooth enough for a spline to fit,
    # varied enough that the adaptive knot search does not settle on a uniform grid
    out = np.zeros((length, dim))
    for d in range(dim):
        for freq, amp in zip(rng.integers(1, 5, 3), rng.uniform(0.2, 1.0, 3), strict=True):
            out[:, d] += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return out


@pytest.fixture(scope="module")
def path():
    return trajectory()


class TestUpstreamParity:
    def test_fit_finds_the_same_knots_and_coefficients(self, path):
        reference = ScipyBSplineCompression(degree=DEGREE)
        reference.compress(path, max_error=MAX_ERROR)
        spline, converged = fit_episode(path, max_error=MAX_ERROR)
        assert converged
        np.testing.assert_array_equal(spline.tck[0], reference.spline.tck[0])
        np.testing.assert_allclose(spline.tck[1], reference.spline.tck[1], rtol=0, atol=0)

    def test_unconverged_fit_returns_upstreams_fallback_spline(self):
        """Upstream refits on the last knot vector when it never met the tolerance;
        we return the spline already built from those knots. Same spline, and the
        caller is told, which upstream only prints."""
        path = trajectory(length=120, seed=3)
        reference = ScipyBSplineCompression(degree=DEGREE)
        reference.compress(path, max_error=0.0)  # no error is < 0, so never met
        spline, converged = fit_episode(path, max_error=0.0)
        assert not converged
        np.testing.assert_array_equal(spline.tck[0], reference.spline.tck[0])
        np.testing.assert_allclose(spline.tck[1], reference.spline.tck[1])

    def test_chunking_matches_window_for_window(self, path):
        reference = ScipyBSplineCompression(degree=DEGREE)
        reference.compress(path, max_error=MAX_ERROR)
        expected = chunk_bspline_trajectory(reference, chunk_size=CHUNK, stride=1)

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        got = chunk_parameters(spline, CHUNK, stride=1)

        assert len(got) == len(expected)
        for ours, theirs in zip(got, expected, strict=True):
            np.testing.assert_array_equal(ours[:, 0], theirs["t"])
            np.testing.assert_array_equal(ours[:, 1:], theirs["c"])

    @pytest.mark.parametrize("num_actions", [1, 2, 8, 20, 59])
    def test_decode_matches_upstream(self, path, num_actions):
        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        parameters = chunk_parameters(spline, CHUNK, stride=1)[0]
        np.testing.assert_allclose(
            decode_chunk(parameters, num_actions),
            decode_bspline_action(parameters, degree=DEGREE, num_actions=num_actions),
            rtol=0,
            atol=1e-6,  # upstream casts its result to float32 on the way out
        )


class TestRotationRepresentation:
    def test_round_trip_is_the_identity_on_cart7(self):
        rng = np.random.default_rng(1)
        raw = np.concatenate(
            [
                rng.normal(size=(50, 3)),
                rng.normal(size=(50, 3)) * 0.9,  # rotvecs comfortably inside the ball
                rng.uniform(0, 1, size=(50, 1)),
            ],
            axis=1,
        )
        np.testing.assert_allclose(from_spline_actions(to_spline_actions(raw)), raw, atol=1e-9)

    def test_six_d_is_the_first_two_rows_of_the_rotation_matrix(self):
        """Pinned by the recorded dataset, not by upstream: the convention there is
        R[:2, :] row-major, and reading it as columns transposes every rotation."""
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(2)
        raw = np.zeros((4, 7))
        raw[:, 3:6] = rng.normal(size=(4, 3)) * 0.5
        matrices = Rotation.from_rotvec(raw[:, 3:6]).as_matrix()
        np.testing.assert_allclose(
            to_spline_actions(raw)[:, 3:9], matrices[:, :2, :].reshape(4, 6), atol=1e-12
        )

    def test_a_perturbed_frame_is_projected_back_onto_so3(self):
        """What a sampled spline actually hands back: two rows that are nearly, but
        not exactly, orthonormal."""
        rng = np.random.default_rng(3)
        raw = np.zeros((16, 7))
        raw[:, 3:6] = rng.normal(size=(16, 3)) * 0.5
        perturbed = to_spline_actions(raw)
        perturbed[:, 3:9] += rng.normal(size=(16, 6)) * 1e-3
        recovered = to_spline_actions(from_spline_actions(perturbed))[:, 3:9]
        # the projection lands within the perturbation, and is exactly orthonormal
        assert np.abs(recovered - perturbed[:, 3:9]).max() < 5e-3
        rows = recovered.reshape(16, 2, 3)
        np.testing.assert_allclose(np.linalg.norm(rows, axis=2), 1.0, atol=1e-12)
        np.testing.assert_allclose((rows[:, 0] * rows[:, 1]).sum(axis=1), 0.0, atol=1e-12)


class TestEpisodeChunks:
    def test_one_matrix_per_frame_with_the_documented_shape(self, path):
        chunks, converged = episode_parameter_chunks(path, CHUNK, max_error=MAX_ERROR)
        assert converged
        assert chunks.shape == (len(path), CHUNK + 2 * DEGREE, 1 + path.shape[1])

    def test_knots_are_offsets_from_the_frame_that_owns_them(self, path):
        """Consecutive frames sharing a chunk differ by exactly one frame of shift --
        that is what lets the policy predict the same curve from anywhere on it."""
        chunks, _ = episode_parameter_chunks(path, CHUNK, max_error=MAX_ERROR)
        shifts = chunks[:-1, :, 0] - chunks[1:, :, 0]
        shared = (chunks[:-1, :, 1:] == chunks[1:, :, 1:]).all(axis=(1, 2))
        assert shared.any()
        np.testing.assert_array_equal(shifts[shared], 1.0)

    def test_a_chunk_reaches_further_than_the_samples_drawn_from_it(self, path):
        """The whole point of the representation: 20 knot spans covering far more
        than 20 source frames, so N samples walk that path in N executed steps."""
        chunks, _ = episode_parameter_chunks(path, CHUNK, max_error=MAX_ERROR)
        spans = chunks[:, -(DEGREE + 1), 0] - chunks[:, DEGREE, 0]
        assert spans.min() > 0
        assert np.median(spans) > CHUNK

    def test_assignment_is_clamped_to_the_episode(self):
        """Our one divergence from upstream, whose loop has no episode bound and
        writes past the end when the last chunk's knot outruns the episode."""
        short = trajectory(length=40, seed=5)
        chunks, _ = episode_parameter_chunks(short, CHUNK, max_error=MAX_ERROR)
        assert len(chunks) == len(short)
        assert np.isfinite(chunks).all()
        assert (chunks[:, :, 1:] != 0).any(axis=(1, 2)).all()  # no frame left unfilled


class TestRelativeKnots:
    """The knot column as differences rather than absolute positions.

    Why it exists: absolute knots are mostly determined by *which row* they are --
    on the UR10e the column averages -7.7 at row 0 and +50.9 at row 25 -- so the one
    per-column statistic LeRobot's normaliser computes leaves a deterministic ramp in
    the target. Differences are stationary across rows, which is what makes per-column
    normalisation appropriate and saves the step from owning normalisation itself.
    """

    def test_matches_upstream_both_ways(self, path):
        from upstream_reference_bspline import (
            decode_relative_knots as upstream_decode,
        )
        from upstream_reference_bspline import (
            encode_relative_knots as upstream_encode,
        )

        from pace_bench.methods.bspline import decode_relative_knots, encode_relative_knots

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        parameters = chunk_parameters(spline, CHUNK, stride=1)[3]

        np.testing.assert_allclose(
            encode_relative_knots(parameters), upstream_encode(parameters.copy(), degree=DEGREE)
        )
        encoded = encode_relative_knots(parameters)
        np.testing.assert_allclose(
            decode_relative_knots(encoded), upstream_decode(encoded.copy(), degree=DEGREE)
        )

    def test_round_trip_recovers_the_absolute_knots(self, path):
        from pace_bench.methods.bspline import decode_relative_knots, encode_relative_knots

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        for parameters in chunk_parameters(spline, CHUNK, stride=1)[:5]:
            recovered = decode_relative_knots(encode_relative_knots(parameters))
            np.testing.assert_allclose(recovered, parameters, atol=1e-12)

    def test_decode_chunk_accepts_the_encoded_form(self, path):
        """The curve is the same curve; only the storage of the knots changed."""
        from pace_bench.methods.bspline import encode_relative_knots

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        parameters = chunk_parameters(spline, CHUNK, stride=1)[2]
        np.testing.assert_allclose(
            decode_chunk(encode_relative_knots(parameters), 20, relative_knots=True),
            decode_chunk(parameters, 20),
            atol=1e-9,
        )


class TestPaddedChunkEndpoint:
    """A chunk at the episode tail repeats its final knot, and the endpoint is a limit.

    Deliberate divergence from upstream, whose `decode_bspline_action` evaluates the
    padded end exactly and gets the all-zero vector -- every basis function is zero on
    a zero-length knot span. As an action in an absolute space that is a command to
    the world origin, and it makes the 6D rotation un-normalizable.
    """

    def test_the_endpoint_of_a_padded_chunk_is_its_last_control_point(self, path):
        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        chunks = chunk_parameters(spline, CHUNK, stride=1)
        padded = [c for c in chunks if c[-1, 0] == c[-(DEGREE + 1), 0]]
        assert padded, "expected the tail chunks to be padded"
        for parameters in padded[:5]:
            decoded = decode_chunk(parameters, 8)
            assert np.isfinite(decoded).all()
            np.testing.assert_allclose(decoded[-1], parameters[-(DEGREE + 1) - 1, 1:], atol=1e-6)

    def test_no_chunk_decodes_to_an_all_zero_action(self, path):
        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        for parameters in chunk_parameters(spline, CHUNK, stride=1):
            decoded = decode_chunk(parameters, 12)
            assert np.isfinite(decoded).all()
            assert np.abs(decoded).max(axis=1).min() > 0, "a slot decoded to all zeros"

    def test_an_interior_chunk_is_unaffected_to_float_precision(self, path):
        """The clamp is one ulp, so it must not move a chunk that is not padded."""
        from scipy.interpolate import BSpline

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        parameters = chunk_parameters(spline, CHUNK, stride=1)[4]
        knots = parameters[:, 0]
        raw = BSpline(knots, parameters[: -(DEGREE + 1), 1:], DEGREE, extrapolate=False)(
            np.linspace(knots[DEGREE], knots[-(DEGREE + 1)], 9)
        )
        np.testing.assert_allclose(decode_chunk(parameters, 9), raw, atol=1e-9)


class TestMonotonicKnots:
    """A predicted knot vector need not be ordered; a B-spline requires it to be.

    Upstream applies this at deployment (`safer_knots`). Nothing in any of the losses
    constrains the order -- the network emits N numbers and their ordering is
    something it has to learn -- so decoding has to tolerate a violation rather than
    raise or return nonsense.
    """

    def test_an_ordered_vector_is_untouched(self, path):
        from pace_bench.methods.bspline import monotonic_knots

        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        for parameters in chunk_parameters(spline, CHUNK, stride=1)[:5]:
            np.testing.assert_array_equal(monotonic_knots(parameters[:, 0]), parameters[:, 0])

    def test_violations_are_nudged_not_sorted(self):
        """A sort would move control points relative to their knots, changing which
        part of the trajectory each one governs. A nudge keeps them in place."""
        from pace_bench.methods.bspline import monotonic_knots

        fixed = monotonic_knots(np.array([0.0, 5.0, 3.0, 9.0]))
        assert (np.diff(fixed) >= 0).all()
        np.testing.assert_allclose(fixed[:2], [0.0, 5.0])
        assert fixed[2] == pytest.approx(5.0, abs=1e-5)  # nudged up to its predecessor
        np.testing.assert_allclose(fixed[3], 9.0)

    def test_a_scrambled_prediction_still_decodes(self, path):
        """The failure this prevents: scipy refusing, or silently returning nonsense,
        on a chunk the policy predicted slightly out of order."""
        spline, _ = fit_episode(path, max_error=MAX_ERROR)
        parameters = chunk_parameters(spline, CHUNK, stride=1)[2].copy()
        parameters[6, 0], parameters[7, 0] = parameters[7, 0], parameters[6, 0]  # swap
        decoded = decode_chunk(parameters, 12)
        assert np.isfinite(decoded).all()
        assert decoded.shape == (12, path.shape[1])
