"""The B-spline method's plumbing: layout, the fitted-spline store, the step, the config.

The maths is covered in `test_bspline.py` and the reconstruction in
`test_bspline_recorded_dataset.py`. What is checked here is everything between that
maths and a training run: which columns of a given dataset get splined, that the step
emits the parameter matrix a policy is then built for, and that the geometry it
imposes is one each policy family can actually accept.
"""

import numpy as np
import pytest
import torch
from lerobot.lerobot_types import TransitionKey
from lerobot.utils.constants import ACTION

from pace_bench.methods.bspline.layout import LAYOUTS, resolve_layout
from pace_bench.methods.bspline.processor import ACTION_IS_PAD, BSplineChunkStep, EpisodeSplines
from pace_bench.methods.bspline.spline import DEGREE, decode_relative_knots
from pace_bench.methods.config import BSplineMethod, NoMethod

CHUNK = 10
WIDTH = CHUNK + 2 * DEGREE


def path(length=180, dim=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    out = np.zeros((length, dim))
    for d in range(dim):
        for freq, amp in zip(rng.integers(1, 5, 3), rng.uniform(0.2, 1.0, 3), strict=True):
            out[:, d] += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return out


@pytest.fixture(scope="module")
def splines():
    episodes = {0: path(seed=0), 1: path(length=140, seed=1)}
    return EpisodeSplines(episodes, LAYOUTS["identity"].__class__(
        "identity", 10, lambda a: a, lambda a: a, 10), CHUNK)


class TestLayout:
    def test_cart7_widens_the_rotation(self):
        layout = resolve_layout("cart7", 7)
        assert layout.spline_dim == 10
        assert layout.to_spline(np.zeros((5, 7))).shape == (5, 10)

    def test_ee6d20_drops_the_zero_padding_and_restores_it(self):
        layout = resolve_layout("ee6d20", 20)
        raw = np.zeros((4, 20))
        raw[:, :10] = np.arange(40).reshape(4, 10)
        spline_actions = layout.to_spline(raw)
        assert spline_actions.shape == (4, 10)
        restored = layout.from_spline(spline_actions)
        assert restored.shape == (4, 20)
        np.testing.assert_array_equal(restored, raw)

    def test_identity_adopts_the_dataset_width(self):
        assert resolve_layout("identity", 14).spline_dim == 14

    def test_a_mismatched_width_is_refused_not_silently_fitted(self):
        """The one guard against the silent failure: naming cart7 for a 20-wide
        LIBERO action would fit three of its columns as a rotation vector."""
        with pytest.raises(ValueError, match="expects a 7-dim action"):
            resolve_layout("cart7", 20)

    def test_an_unknown_layout_names_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown action layout"):
            resolve_layout("quaternion", 8)


class TestEpisodeSplines:
    def test_every_frame_maps_to_a_chunk_that_exists(self, splines):
        for episode, mapping in splines.frame_to_chunk.items():
            assert mapping.min() >= 0
            assert mapping.max() < len(splines.chunks[episode])

    def test_parameters_are_shifted_to_the_requesting_frame(self, splines):
        """Two frames sharing a chunk differ by exactly the shift, and by nothing
        else -- the control points are the same curve."""
        shared = np.flatnonzero(splines.frame_to_chunk[0][:-1] == splines.frame_to_chunk[0][1:])
        assert len(shared) > 0
        frame = int(shared[0])
        a = splines.parameters(0, frame)
        b = splines.parameters(0, frame + 1)
        np.testing.assert_allclose(a[:, 0] - b[:, 0], 1.0)
        np.testing.assert_array_equal(a[:, 1:], b[:, 1:])

    def test_it_holds_splines_not_a_matrix_per_frame(self, splines):
        """The point of the design: what is kept scales with knots, not with frames."""
        frames = sum(len(f) for f in splines.frame_to_chunk.values())
        per_frame = frames * WIDTH * (1 + 10) * 4
        assert splines.nbytes() < per_frame


class TestBSplineChunkStep:
    def transition(self, batch, episodes, frames):
        return {
            TransitionKey.ACTION: torch.zeros(batch, 7, 10),  # the loader window, ignored
            TransitionKey.COMPLEMENTARY_DATA: {
                "episode_index": torch.tensor(episodes),
                "frame_index": torch.tensor(frames),
            },
        }

    def test_it_replaces_the_action_with_the_parameter_matrix(self, splines):
        step = BSplineChunkStep(splines=splines, relative_knots=False)
        out = step(self.transition(2, [0, 1], [5, 9]))
        actions = out[TransitionKey.ACTION]
        assert actions.shape == (2, WIDTH, 11)
        np.testing.assert_allclose(actions[0].numpy(), splines.parameters(0, 5), rtol=1e-6)

    def test_relative_knots_are_invertible_back_to_what_was_fitted(self, splines):
        step = BSplineChunkStep(splines=splines, relative_knots=True)
        emitted = step(self.transition(1, [0], [7]))[TransitionKey.ACTION][0].numpy()
        np.testing.assert_allclose(
            decode_relative_knots(emitted.astype(np.float64), DEGREE),
            splines.parameters(0, 7),
            atol=1e-4,
        )

    def test_nothing_is_masked_because_nothing_is_padded(self, splines):
        """A B-spline chunk is fixed-size by construction; a short tail repeats the
        last knot and control point rather than padding, so no slot is fake."""
        step = BSplineChunkStep(splines=splines)
        out = step(self.transition(3, [0, 1, 0], [0, 3, 170]))
        assert not out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD].any()

    def test_a_frame_index_can_be_reconstructed_from_the_global_index(self, splines):
        direct = BSplineChunkStep(splines=splines, relative_knots=False)
        via = BSplineChunkStep(
            splines=splines, episode_starts={0: 0, 1: 180}, relative_knots=False
        )
        want = direct(self.transition(2, [0, 1], [5, 9]))[TransitionKey.ACTION]
        got = via({
            TransitionKey.ACTION: torch.zeros(2, 7, 10),
            TransitionKey.COMPLEMENTARY_DATA: {
                "episode_index": torch.tensor([0, 1]),
                "index": torch.tensor([5, 189]),
            },
        })[TransitionKey.ACTION]
        torch.testing.assert_close(got, want)

    def test_without_a_fit_it_passes_through(self):
        """Selectable before a dataset exists, the way the DemoSpeedup step is."""
        step = BSplineChunkStep()
        actions = torch.randn(2, 7, 10)
        out = step({TransitionKey.ACTION: actions, TransitionKey.COMPLEMENTARY_DATA: {}})
        torch.testing.assert_close(out[TransitionKey.ACTION], actions)


class TestBSplineMethodConfig:
    class FakePolicy:
        type = "act"
        chunk_size = 100
        n_action_steps = 100

    class FakeDiffusion:
        type = "diffusion"
        horizon = 64
        n_action_steps = 8
        down_dims = (512, 1024, 2048)

    def test_the_policy_chunk_becomes_the_matrix_width(self):
        policy = self.FakePolicy()
        BSplineMethod(chunk_size=10).adjust_policy(policy)
        assert policy.chunk_size == 16 and policy.n_action_steps == 16

    def test_knots_default_to_absolute_offsets_from_the_current_frame(self):
        """Upstream's shipped configs and the recorded dataset both use offsets, not
        differences. Only the *time* column is at issue either way -- the control
        points are absolute poses regardless, and the knots are relative to the
        sample's own frame regardless."""
        assert BSplineMethod().relative_knots is False

    def test_the_default_width_suits_every_policy_family(self):
        """16 rows: upstream's own real-robot horizon, and a multiple of 8 as
        Diffusion's temporal U-Net requires."""
        method = BSplineMethod()
        assert method.width == 16
        method.adjust_policy(self.FakeDiffusion())

    def test_a_width_the_unet_cannot_halve_is_refused_with_the_usable_values(self):
        """LeRobot checks this in DiffusionConfig.__post_init__, which has already run
        by the time a method mutates the config -- so without this guard the run dies
        mid-forward with a tensor-size error naming neither horizon nor method."""
        with pytest.raises(ValueError, match=r"multiple of 8"):
            BSplineMethod(chunk_size=20).adjust_policy(self.FakeDiffusion())

    def test_an_unknown_policy_family_is_a_loud_error(self):
        class Unknown:
            type = "octo"

        with pytest.raises(ValueError, match="does not know the chunk fields"):
            BSplineMethod().adjust_policy(Unknown())

    def test_degenerate_geometry_is_refused(self):
        with pytest.raises(ValueError, match="chunk_size must be"):
            BSplineMethod(chunk_size=1)
        with pytest.raises(ValueError, match="degree must be"):
            BSplineMethod(degree=0)

    def test_adjust_dataset_is_a_no_op_for_methods_that_keep_the_action_space(self):
        """The hook is called for every method, so the baseline must be unaffected."""
        sentinel = object()
        NoMethod().adjust_dataset(sentinel)

    def test_it_is_selectable_as_a_method_type(self):
        assert BSplineMethod().type == "bspline"
        assert ACTION  # the constant the metadata rewrite keys on still exists
