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
