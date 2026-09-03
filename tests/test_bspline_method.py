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
from pace_bench.methods.bspline.spline import DEGREE, decode_relative_knots, ramp_gripper
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


class TestBSplineDecodeStep:
    """The inverse of the chunk step, and what makes a checkpoint executable at all."""

    def matrix(self, splines, episode=0, frame=5):
        return torch.from_numpy(splines.parameters(episode, frame).astype(np.float32))

    def test_it_decodes_parameters_into_actions(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(num_actions=12)
        out = step({
            TransitionKey.ACTION: torch.stack([self.matrix(splines), self.matrix(splines, 0, 9)]),
            TransitionKey.COMPLEMENTARY_DATA: {},
        })
        assert out[TransitionKey.ACTION].shape == (2, 12, 10)

    def test_an_unbatched_chunk_stays_unbatched(self, splines):
        """A control loop hands over one chunk, training code hands a batch."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        out = BSplineDecodeStep(num_actions=6)({
            TransitionKey.ACTION: self.matrix(splines), TransitionKey.COMPLEMENTARY_DATA: {}
        })
        assert out[TransitionKey.ACTION].shape == (6, 10)

    def test_fewer_actions_cover_the_same_span_faster(self, splines):
        """The speed lever: `a_exec(t) = a(nt)`. The curve is a fixed stretch of
        demonstrated motion, so halving the samples doubles the rate."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        rates = {}
        for num_actions in (5, 9, 17):
            out = BSplineDecodeStep(num_actions=num_actions)({
                TransitionKey.ACTION: self.matrix(splines), TransitionKey.COMPLEMENTARY_DATA: {}
            })
            rates[num_actions] = float(out[TransitionKey.COMPLEMENTARY_DATA]["bspline_rate"][0])
        assert rates[5] > rates[9] > rates[17]
        assert rates[5] == pytest.approx(2 * rates[9], rel=1e-6)

    def test_the_layout_maps_actions_back_to_the_dataset_space(self, splines):
        """A robot consumes cart7, not the 10-dim space the spline lives in."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        out = BSplineDecodeStep(num_actions=4, layout=resolve_layout("cart7", 7))({
            TransitionKey.ACTION: self.matrix(splines), TransitionKey.COMPLEMENTARY_DATA: {}
        })
        assert out[TransitionKey.ACTION].shape == (4, 7)

    def test_decoding_at_the_demonstrated_rate_reproduces_the_curve(self, splines):
        """Sampled at one point per source frame, the decode is the fit itself --
        so any error here is the fit's, not the round trip's."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep
        from pace_bench.methods.bspline.spline import decode_chunk

        matrix = splines.parameters(0, 5)
        span = int(matrix[-(DEGREE + 1), 0] - matrix[DEGREE, 0])
        out = BSplineDecodeStep(num_actions=span + 1)({
            TransitionKey.ACTION: torch.from_numpy(matrix.astype(np.float32)),
            TransitionKey.COMPLEMENTARY_DATA: {},
        })
        np.testing.assert_allclose(
            out[TransitionKey.ACTION].numpy(),
            decode_chunk(matrix, span + 1).astype(np.float32),
            atol=1e-4,
        )
        assert float(out[TransitionKey.COMPLEMENTARY_DATA]["bspline_rate"][0]) == pytest.approx(1.0)

    def test_relative_knots_are_decoded_the_way_they_were_encoded(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep
        from pace_bench.methods.bspline.spline import encode_relative_knots

        matrix = splines.parameters(0, 5)
        absolute = BSplineDecodeStep(num_actions=8)({
            TransitionKey.ACTION: torch.from_numpy(matrix.astype(np.float32)),
            TransitionKey.COMPLEMENTARY_DATA: {},
        })[TransitionKey.ACTION]
        relative = BSplineDecodeStep(num_actions=8, relative_knots=True)({
            TransitionKey.ACTION: torch.from_numpy(
                encode_relative_knots(matrix, DEGREE).astype(np.float32)
            ),
            TransitionKey.COMPLEMENTARY_DATA: {},
        })[TransitionKey.ACTION]
        torch.testing.assert_close(relative, absolute, atol=1e-3, rtol=0)

    def test_a_degenerate_sample_count_is_refused(self):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        with pytest.raises(ValueError, match="num_actions must be"):
            BSplineDecodeStep(num_actions=0)

    def test_the_method_contributes_it_with_the_configured_geometry(self):
        method = BSplineMethod(chunk_size=10, num_actions=7, relative_knots=True)
        (step,) = method.postprocessor_steps()
        assert step.get_config()["num_actions"] == 7
        assert step.get_config()["relative_knots"] is True
        # defaulting to the matrix width is roughly demonstration speed
        assert BSplineMethod().postprocessor_steps()[0].get_config()["num_actions"] == 16


class TestMatrixArrangement:
    """Where the knot column sits in the tensor the policy sees.

    ACT and Diffusion read their action as an undifferentiated vector, so upstream's
    knot-first layout is fine. xVLA slices by hardcoded index and does not normalize
    at all, so it needs both a different column order and knots in seconds.
    """

    def test_knot_first_is_the_identity(self):
        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = resolve_arrangement("knot_first")
        matrix = np.arange(16 * 11, dtype=np.float64).reshape(16, 11)
        np.testing.assert_array_equal(arrangement.emit(matrix), matrix)
        np.testing.assert_array_equal(arrangement.recover(matrix, 10), matrix)

    def test_xvla_puts_the_control_point_where_the_structured_loss_expects_it(self):
        """xVLA's POS_IDX_1 is (0,1,2) and ROT_IDX_1 is (3..8): a knot at column 0
        is trained as an x-coordinate, which showed as position_loss 122840."""
        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = resolve_arrangement("xvla_ee6d20")
        matrix = np.zeros((16, 11))
        matrix[:, 0] = 7.0                       # knots
        matrix[:, 1:] = np.arange(10) + 1.0      # control point
        emitted = arrangement.emit(matrix)
        assert emitted.shape == (16, 20)
        np.testing.assert_array_equal(emitted[:, :10], matrix[:, 1:])
        np.testing.assert_array_equal(emitted[:, 10], matrix[:, 0])
        assert not emitted[:, 11:].any(), "slots 11..19 must stay zero for a single arm"

    def test_the_round_trip_restores_the_matrix(self):
        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = resolve_arrangement("xvla_ee6d20")
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(16, 11))
        np.testing.assert_allclose(arrangement.recover(arrangement.emit(matrix), 10), matrix)

    def test_the_knot_scale_survives_the_round_trip(self):
        """xVLA normalizes nothing (its normalization_mapping is IDENTITY throughout),
        so knots have to reach its loss already comparable to metres. 1/fps makes the
        column seconds; the curve must be unchanged by that."""
        from dataclasses import replace

        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = replace(resolve_arrangement("xvla_ee6d20"), knot_scale=1 / 20)
        matrix = np.zeros((16, 11))
        matrix[:, 0] = np.arange(16) * 3.0
        emitted = arrangement.emit(matrix)
        np.testing.assert_allclose(emitted[:, 10], matrix[:, 0] / 20)
        np.testing.assert_allclose(arrangement.recover(emitted, 10), matrix)

    def test_knot_first20_pads_without_rearranging(self):
        """The knot stays where upstream puts it -- slot 0 -- and the pose follows it,
        because a run that scores the matrix uniformly has no index-sliced loss to
        satisfy. Only the width is xVLA's."""
        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = resolve_arrangement("knot_first20")
        matrix = np.zeros((16, 11))
        matrix[:, 0] = 7.0
        matrix[:, 1:] = np.arange(10) + 1.0
        emitted = arrangement.emit(matrix)
        assert emitted.shape == (16, 20)
        np.testing.assert_array_equal(emitted[:, 0], matrix[:, 0])
        np.testing.assert_array_equal(emitted[:, 1:11], matrix[:, 1:])
        assert not emitted[:, 11:].any(), "slots 11..19 are pad"
        np.testing.assert_allclose(arrangement.recover(emitted, 10), matrix)

    def test_both_20_wide_arrangements_describe_the_same_curve(self):
        """The claim that makes the uniform arrangement safe: knot_first20 changes where
        the numbers sit in the vector and nothing about the geometry. Whichever way round
        a chunk is emitted, `recover` must hand the decoder back the same matrix."""
        from pace_bench.methods.bspline.layout import resolve_arrangement

        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(16, 11))
        for name in ("xvla_ee6d20", "knot_first20"):
            arrangement = resolve_arrangement(name)
            np.testing.assert_allclose(
                arrangement.recover(arrangement.emit(matrix), 10), matrix,
                err_msg=f"{name} did not round-trip",
            )

    def test_only_the_unnormalised_arrangement_rescales_knots(self):
        """xvla_ee6d20 carries knots in seconds because xVLA normalizes nothing.
        knot_first20 is paired with real normalization, which handles the scale -- a
        second rescaling there would only hide it."""
        from pace_bench.methods.bspline.layout import resolve_arrangement

        assert resolve_arrangement("xvla_ee6d20").scale_knots_by_fps is True
        assert resolve_arrangement("knot_first20").scale_knots_by_fps is False
        assert resolve_arrangement("knot_first").scale_knots_by_fps is False
        method = BSplineMethod(arrangement="knot_first20", fps=20.0)
        assert method._arrange().knot_scale == 1.0
        assert BSplineMethod(arrangement="xvla_ee6d20", fps=20.0)._arrange().knot_scale == 1 / 20

    def test_an_unknown_arrangement_names_the_known_ones(self):
        from pace_bench.methods.bspline.layout import resolve_arrangement

        with pytest.raises(ValueError, match="unknown matrix arrangement"):
            resolve_arrangement("knot_last")


class TestXVLABSplineActionSpace:
    """xVLA's structured loss applied to slots that hold spline coefficients.

    Every slot of the arranged matrix is a B-spline parameter, so every term has to
    be a regression. Stock ee6d classifies slot 9, which is right for a 0/1 gripper
    command and wrong for the gripper's control point.
    """

    #: The control-point extremes a binary 0/1 channel actually produces. Measured
    #: over libero_10_ee6d at max_error=0.01; the value is the cubic B-spline's
    #: structural overshoot for a unit step, not a property of that dataset.
    OVERSHOOT = (-0.366, 1.366)

    @staticmethod
    def space():
        from lerobot.policies.xvla.action_hub import build_action_space

        import pace_bench.methods.bspline.xvla_action  # noqa: F401

        return build_action_space("ee6d_bspline")

    def test_it_registers_and_scores_the_knot_column(self):
        from lerobot.policies.xvla.action_hub import build_action_space

        import pace_bench.methods.bspline.xvla_action  # noqa: F401

        space = build_action_space("ee6d_bspline")
        assert space.dim_action == 20
        assert space.KNOT_IDX == (10,)
        assert space.gripper_idx == (9,)  # single arm; upstream ee6d is (9, 19)

        pred = torch.zeros(2, 16, 20)
        target = torch.zeros(2, 16, 20)
        target[:, :, 10] = 1.0  # a knot error and nothing else
        losses = space.compute_loss(pred, target)
        assert losses["knot_loss"] > 0
        assert losses["position_loss"] == 0 and losses["rotate6D_loss"] == 0

    def test_the_structured_groups_do_not_overlap_the_knot(self):
        """The whole point: no index is scored as two different quantities."""
        from lerobot.policies.xvla.action_hub import build_action_space

        import pace_bench.methods.bspline.xvla_action  # noqa: F401

        space = build_action_space("ee6d_bspline")
        groups = [set(space.POS_IDX), set(space.ROT_IDX), set(space.gripper_idx), set(space.KNOT_IDX)]
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                assert not (a & b)

    def test_the_gripper_term_is_a_regression(self):
        """A control point outside [0, 1] must cost more the further off it is.

        Under BCE this fails in the worst possible way: the loss is unbounded below
        for a target outside [0, 1], so saturating the logit scores *better* than
        predicting the coefficient. Here `far` must simply cost more than `near`.
        """
        space = self.space()
        target = torch.zeros(2, 4, 20)
        target[:, :, 9] = self.OVERSHOOT[1]

        near = target.clone()
        near[:, :, 9] = self.OVERSHOOT[1] - 0.1
        far = target.clone()
        far[:, :, 9] = 30.0  # a saturating "logit"

        assert space.compute_loss(target.clone(), target)["gripper_loss"].item() == 0.0
        assert (
            space.compute_loss(far, target)["gripper_loss"].item()
            > space.compute_loss(near, target)["gripper_loss"].item()
            > 0.0
        )

    def test_no_loss_term_can_go_negative(self):
        """The BCE pathology, pinned: every term is a scaled MSE, so none is
        unbounded below no matter what the coefficients are."""
        space = self.space()
        rng = torch.Generator().manual_seed(0)
        target = torch.rand(2, 4, 20, generator=rng) * 3 - 1  # spans well outside [0, 1]
        pred = torch.randn(2, 4, 20, generator=rng) * 20
        assert all(v.item() >= 0.0 for v in space.compute_loss(pred, target).values())

    def test_postprocess_leaves_the_control_point_alone(self):
        """Stock ee6d sigmoids slot 9. That would clamp the coefficient into (0, 1)
        and make the overshoot -- which is what encodes the gripper edge --
        unrepresentable, so this space must return the prediction as regressed."""
        space = self.space()
        action = torch.zeros(1, 3, 20)
        action[:, :, 9] = torch.tensor(list(self.OVERSHOOT) + [0.5])
        np.testing.assert_allclose(
            space.postprocess(action.clone())[:, :, 9].numpy(),
            action[:, :, 9].numpy(),
        )

    def test_every_slot_of_the_matrix_is_scored(self):
        """Nothing may be silently unsupervised: slots 0..10 are all real parameters
        (11..19 are the zero pad a single arm leaves behind)."""
        space = self.space()
        scored = set(space.POS_IDX) | set(space.ROT_IDX) | set(space.gripper_idx) | set(space.KNOT_IDX)
        assert scored == set(range(11))


class TestUniformBSplineActionSpace:
    """Upstream B-spline's loss on xVLA: one uniform MSE over the parameter matrix.

    The claim to protect is that there is nothing to tune here. The four reported terms
    are channel-count contributions, so their sum has to be *exactly* the unweighted MSE
    over the 11 real channels -- if that identity ever breaks, a weight has crept in.
    """

    @staticmethod
    def space():
        from lerobot.policies.xvla.action_hub import build_action_space

        import pace_bench.methods.bspline.xvla_action  # noqa: F401

        return build_action_space("bspline_uniform")

    def test_the_terms_sum_to_an_unweighted_mse(self):
        space = self.space()
        rng = torch.Generator().manual_seed(1)
        target = torch.zeros(3, 16, 20)
        target[:, :, :11] = torch.randn(3, 16, 11, generator=rng)
        pred = torch.randn(3, 16, 20, generator=rng)
        total = sum(space.compute_loss(pred, target).values())
        uniform = torch.nn.functional.mse_loss(pred[:, :, :11], target[:, :, :11])
        torch.testing.assert_close(total, uniform)

    def test_the_pad_channels_are_scored_by_nothing(self):
        """XVLAPolicy pads 11 real channels out to dim_action=20. Slots 11..19 are that
        pad, and must not enter the loss however they are filled."""
        space = self.space()
        pred, target = torch.zeros(2, 16, 20), torch.zeros(2, 16, 20)
        target[:, :, :11] = 0.5
        before = sum(space.compute_loss(pred, target).values())
        target[:, :, 11:] = 99.0
        pred[:, :, 11:] = -7.0
        torch.testing.assert_close(sum(space.compute_loss(pred, target).values()), before)

    def test_no_parameter_is_masked_and_nothing_is_squashed(self):
        """Unlike the stock ee6d space: no channel here is a 0/1 command, so there is no
        gripper to mask out of the flow input and no logit to sigmoid on the way out.
        The only thing `preprocess` touches is the nine pad slots past the parameters,
        which the pretrained head writes into and the generation loop would otherwise
        feed back (see tests/test_xvla_pad_mask.py)."""
        space = self.space()
        assert space.gripper_idx == ()
        action = torch.randn(2, 16, 20)
        torch.testing.assert_close(space.postprocess(action.clone()), action)
        proprio = torch.randn(2, 20)
        p_out, a_out = space.preprocess(proprio.clone(), action.clone())
        torch.testing.assert_close(p_out, proprio)
        torch.testing.assert_close(a_out[..., :11], action[..., :11])
        assert torch.equal(a_out[..., 11:], torch.zeros(2, 16, 9))

    def test_a_matrix_narrower_than_the_parameter_count_is_refused(self):
        space = self.space()
        with pytest.raises(ValueError, match="knot_first20 vector"):
            space.compute_loss(torch.zeros(1, 16, 8), torch.zeros(1, 16, 8))


class TestParameterNormalisation:
    """The pairing and the guard around `--method.normalize_parameters`.

    A uniform loss only balances if the channels are comparable. ACT and Diffusion get
    that from their own normalizer; xVLA normalizes nothing, so selecting the uniform
    space there without switching normalization on is a silent trap -- the knot column
    is in source frames and would swamp everything.
    """

    @staticmethod
    def policy_cfg(kind, mode=None, action_norm="IDENTITY"):
        from lerobot.configs.types import NormalizationMode

        from pace_bench.methods.config import POLICY_CHUNK_FIELDS

        fields = POLICY_CHUNK_FIELDS[kind]
        cfg = type("FakePolicyCfg", (), {})()
        cfg.type, cfg.action_mode = kind, mode
        cfg.normalization_mapping = {"ACTION": NormalizationMode[action_norm]}
        setattr(cfg, fields.chunk, 30)
        setattr(cfg, fields.executed, 30)
        return cfg

    def test_the_default_leaves_every_other_arm_alone(self):
        """Off by default, and a no-op for a policy that already normalizes."""
        cfg = self.policy_cfg("act", action_norm="MEAN_STD")
        BSplineMethod().adjust_policy(cfg)
        assert cfg.normalization_mapping["ACTION"].value == "MEAN_STD"

    def test_the_uniform_space_is_refused_without_normalisation(self):
        cfg = self.policy_cfg("xvla", "bspline_uniform")
        with pytest.raises(ValueError, match="normalize_parameters"):
            BSplineMethod(arrangement="knot_first20",
                          xvla_action_space="bspline_uniform").adjust_policy(cfg)

    def test_the_flag_switches_the_action_feature_to_mean_std(self):
        cfg = self.policy_cfg("xvla", "bspline_uniform")
        BSplineMethod(arrangement="knot_first20", xvla_action_space="bspline_uniform",
                      normalize_parameters=True).adjust_policy(cfg)
        assert cfg.normalization_mapping["ACTION"].value == "MEAN_STD"

    def test_the_uniform_space_refuses_the_wrong_arrangement(self):
        """The space reads the knot from slot 0; xvla_ee6d20 puts it in slot 10. Both
        are 20 wide, so nothing downstream would notice the disagreement."""
        cfg = self.policy_cfg("xvla", "bspline_uniform")
        with pytest.raises(ValueError, match="knot_first20"):
            BSplineMethod(arrangement="xvla_ee6d20", xvla_action_space="bspline_uniform",
                          normalize_parameters=True).adjust_policy(cfg)

    def test_naming_one_half_of_the_pairing_is_refused(self):
        """The method registers the space, the policy selects it. One without the other
        trains a loss the run did not ask for, and nothing downstream would notice."""
        cfg = self.policy_cfg("xvla", "ee6d_bspline")
        with pytest.raises(ValueError, match="xvla_action_space"):
            BSplineMethod(arrangement="knot_first20", xvla_action_space="bspline_uniform",
                          normalize_parameters=True).adjust_policy(cfg)


class TestEvalPath:
    """What it takes to score a B-spline checkpoint, which has no dataset to consult.

    The bug this covers: `run_libero` builds the method's postprocessor step with no
    dataset, and the step used to come back with `layout=None, arrangement=None`. For
    an xVLA checkpoint that reads column 0 as the knot when the knot is in column 10 --
    decoding a position as a time, silently.
    """

    def test_the_decode_step_resolves_itself_without_a_dataset(self):
        method = BSplineMethod(layout="ee6d20", arrangement="xvla_ee6d20")
        (step,) = method.postprocessor_steps()
        assert step.layout is not None and step.layout.raw_dim == 20
        assert step.arrangement is not None and step.arrangement.channels == 20

    def test_the_knot_scale_is_reconstructed_from_config_not_the_dataset(self):
        """It must match what the checkpoint trained under, and evaluation cannot ask
        the dataset -- so fps is a config field."""
        (step,) = BSplineMethod(arrangement="xvla_ee6d20", fps=20.0).postprocessor_steps()
        assert step.arrangement.knot_scale == pytest.approx(0.05)
        (other,) = BSplineMethod(arrangement="xvla_ee6d20", fps=50.0).postprocessor_steps()
        assert other.arrangement.knot_scale == pytest.approx(0.02)

    def test_knot_first_needs_no_arrangement_scaling(self):
        (step,) = BSplineMethod(layout="cart7").postprocessor_steps()
        assert step.arrangement.channels is None
        assert step.arrangement.knot_scale == 1.0

    def test_decode_batch_returns_actions_and_the_realised_rate(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        matrices = torch.stack([
            torch.from_numpy(splines.parameters(0, 5).astype(np.float32)),
            torch.from_numpy(splines.parameters(0, 9).astype(np.float32)),
        ])
        actions, rates = BSplineDecodeStep(num_actions=9).decode_batch(matrices)
        assert actions.shape == (2, 9, 10)
        assert rates.shape == (2,)
        assert (rates > 0).all()


class TestAttachBSpline:
    """Decoding attached to one policy object, the way `attach_pace` attaches PACE."""

    class FakePolicy:
        """Enough of a policy for the rebinding: a chunk source and a reset."""

        def __init__(self, matrix):
            self.matrix = matrix
            self.queries = 0
            self.resets = 0

        def predict_action_chunk(self, batch):
            self.queries += 1
            return self.matrix

        def reset(self):
            self.resets += 1

    def policy(self, splines, batch=2):
        matrix = torch.from_numpy(splines.parameters(0, 5).astype(np.float32))
        return self.FakePolicy(matrix.expand(batch, *matrix.shape).clone())

    def test_one_query_serves_num_actions_steps(self, splines):
        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        policy = attach_bspline(self.policy(splines), BSplineDecodeStep(num_actions=5))
        actions = [policy.select_action({}) for _ in range(5)]
        assert policy.queries == 1, "the chunk should be decoded once, not per step"
        assert all(a.shape == (2, 10) for a in actions)
        policy.select_action({})
        assert policy.queries == 2, "a sixth step must trigger a fresh query"

    def test_fewer_actions_means_fewer_queries_per_unit_of_motion(self, splines):
        """The speed lever, seen from the eval loop: the same curve, traversed in
        fewer executed steps."""
        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        rates = {}
        for num_actions in (4, 8):
            policy = attach_bspline(
                self.policy(splines), BSplineDecodeStep(num_actions=num_actions)
            )
            policy.select_action({})
            rates[num_actions] = policy.bspline_rate_log[0]
        assert rates[4] > rates[8]

    def test_reset_drops_the_queue_and_still_resets_the_policy(self, splines):
        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        policy = attach_bspline(self.policy(splines), BSplineDecodeStep(num_actions=6))
        policy.select_action({})
        assert len(policy.bspline_queue) == 5
        policy.reset()
        assert not policy.bspline_queue
        assert policy.resets == 1
        policy.select_action({})
        assert policy.queries == 2, "after a reset the next step must re-query"


class TestBSplineActuation:
    """Upstream's recipe, reproduced: arm kp only, kd and gripper left nominal.

    This is where B-spline differs from both other methods. PACE scales kd as
    `s**(exp/2)` and DemoSpeedup as `low_v**(exp/2)`, both chasing critical damping;
    upstream B-spline scales `kp[:6]` and passes the base kd straight back
    (`update_kp_kd(kp, self._base_kd.copy())`).
    """

    class FakeController:
        def __init__(self):
            self.kp_scale = None
            self.kd_scale = None

        def update_kp_scale(self, scale):
            self.kp_scale = scale

        def update_kd_scale(self, scale):
            self.kd_scale = scale

    class FakeGripper:
        speed = 0.1

    def handle(self):
        import types as _types

        controller, gripper = self.FakeController(), self.FakeGripper()
        robot = _types.SimpleNamespace(controller=controller, gripper=gripper)
        sim = _types.SimpleNamespace(robots=[robot])
        return _types.SimpleNamespace(unwrapped=_types.SimpleNamespace(_env=_types.SimpleNamespace(env=sim)))

    def test_it_scales_kp_by_upstreams_default(self):
        from pace_bench.methods.bspline.actuator import DEFAULT_KP_SCALE, BSplineTrackingActuator

        assert DEFAULT_KP_SCALE == 2.0  # rollout_x5_bspline.py --stiffness-kp-scale
        handle = self.handle()
        BSplineTrackingActuator().apply(handle)
        assert handle.unwrapped._env.env.robots[0].controller.kp_scale == 2.0

    def test_kd_is_left_nominal(self):
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator

        handle = self.handle()
        BSplineTrackingActuator().apply(handle)
        assert handle.unwrapped._env.env.robots[0].controller.kd_scale is None

    def test_the_gripper_is_left_nominal(self):
        """DemoSpeedup scales gripper stroke rate; upstream B-spline explicitly does
        not ("leave the gripper kp (last element) and all kd untouched")."""
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator

        handle = self.handle()
        before = handle.unwrapped._env.env.robots[0].gripper.speed
        BSplineTrackingActuator().apply(handle)
        assert handle.unwrapped._env.env.robots[0].gripper.speed == before

    def test_time_runs_nominal(self):
        """The speed-up is already in the action stream; scaling time too would apply
        it twice."""
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator

        assert BSplineTrackingActuator().apply(self.handle()) == 1.0

    def test_reapplying_does_not_compound(self):
        """It runs every step, because a robosuite reset rebuilds the controller."""
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator

        handle, actuator = self.handle(), BSplineTrackingActuator(kp_scale=3.0)
        for _ in range(4):
            actuator.apply(handle)
        assert handle.unwrapped._env.env.robots[0].controller.kp_scale == 3.0

    def test_the_ablation_leaves_gains_alone(self):
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator

        handle = self.handle()
        BSplineTrackingActuator(disable_kp_scaling=True).apply(handle)
        assert handle.unwrapped._env.env.robots[0].controller.kp_scale is None

    def test_the_method_carries_upstreams_default(self):
        assert BSplineMethod().stiffness_kp_scale == 2.0

    def test_the_policy_actuates_every_step_once_bound(self, splines):
        import types as _types

        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.actuator import BSplineTrackingActuator
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        matrix = torch.from_numpy(splines.parameters(0, 5).astype(np.float32))

        class P:
            def predict_action_chunk(self, batch):
                return matrix[None]

            def reset(self):
                pass

        handle = self.handle()
        policy = attach_bspline(P(), BSplineDecodeStep(num_actions=4), BSplineTrackingActuator())
        policy.bind_env(_types.SimpleNamespace(envs=[handle]))
        for _ in range(4):
            policy.select_action({})
        assert handle.unwrapped._env.env.robots[0].controller.kp_scale == 2.0


def test_eval_config_registers_the_xvla_action_space(tmp_path):
    """Parsing a B-spline eval config must register `ee6d_bspline` on its own.

    xVLA resolves its action space while the model is being built, and an eval loads
    the policy straight from a checkpoint whose config names `ee6d_bspline` -- so if
    nothing registered it first, `make_policy` dies with a bare KeyError before a
    single episode runs. It did, once. Run in a subprocess because registration is a
    process-wide side effect: in this suite's process another test has already
    imported the module, and the check would pass for the wrong reason.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import draccus
        from lerobot.policies.xvla.action_hub import build_action_space
        from pace_bench.eval.run_libero import LiberoEvalConfig

        try:
            build_action_space("ee6d_bspline")
            raise SystemExit("registered before the config was built")
        except KeyError:
            pass

        draccus.parse(
            config_class=LiberoEvalConfig,
            args=["--method.type=bspline", "--method.layout=ee6d20",
                  "--method.arrangement=xvla_ee6d20"],
        )
        print(type(build_action_space("ee6d_bspline")).__name__)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == "EE6DBSplineActionSpace"


def test_decode_step_accepts_the_names_it_serialises():
    """`get_config` writes layout and arrangement by name, so `__init__` must read them.

    A saved pipeline carries the names, and LeRobot rebuilds the step from exactly
    that dict -- so if the constructor only accepted objects, a checkpoint's own
    decode step came back holding strings and died on the first chunk. It did.
    """
    from pace_bench.methods.bspline.processor import BSplineDecodeStep

    step = BSplineDecodeStep(num_actions=8, degree=3, layout="ee6d20", arrangement="xvla_ee6d20")
    assert step.layout.name == "ee6d20"
    assert step.arrangement.name == "xvla_ee6d20"
    # The round trip has to close: rebuilding from the config gives the same step.
    rebuilt = BSplineDecodeStep(**step.get_config())
    assert rebuilt.get_config() == step.get_config()
    assert rebuilt.layout is step.layout


def test_decode_step_keeps_an_already_resolved_layout():
    """Objects still pass through untouched -- the training path passes those."""
    from pace_bench.methods.bspline.layout import LAYOUTS
    from pace_bench.methods.bspline.processor import BSplineDecodeStep

    layout = LAYOUTS["ee6d20"]
    assert BSplineDecodeStep(layout=layout).layout is layout


def test_eval_drops_a_checkpoint_side_decode_step():
    """Two decode steps is not redundancy, it is a decoded action read as a curve."""
    from pace_bench.eval.run_libero import drop_steps
    from pace_bench.methods.bspline.processor import BSplineDecodeStep

    class Pipeline:
        def __init__(self, steps):
            self.steps = steps

    class Unnormalizer:
        pass

    saved = BSplineDecodeStep(num_actions=16, layout="ee6d20", arrangement="xvla_ee6d20")
    pipeline = Pipeline([Unnormalizer(), saved])
    assert drop_steps(pipeline, "BSplineDecodeStep") == [saved]
    assert [type(s).__name__ for s in pipeline.steps] == ["Unnormalizer"]
    # Nothing left to drop, and a pipeline without one is untouched.
    assert drop_steps(pipeline, "BSplineDecodeStep") == []


class TestDecodeBatchSequential:
    """A vector env is `n_envs` sequential streams, not one batch of strangers."""

    @staticmethod
    def _step(**kwargs):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        return BSplineDecodeStep(
            num_actions=8, degree=3, layout="ee6d20", arrangement="xvla_ee6d20", **kwargs
        )

    @staticmethod
    def _matrices(rows: int):
        """`rows` plausible parameter matrices: a rising knot column, smooth points."""
        import numpy as np
        import torch

        from pace_bench.methods.bspline.layout import LAYOUTS

        layout = LAYOUTS["ee6d20"]
        width = 10 + 2 * 3
        out = np.zeros((rows, width, 1 + layout.spline_dim))
        for r in range(rows):
            out[r, :, 0] = np.arange(width) * 1.5 + r  # knots, strictly increasing
            out[r, :, 1:] = np.linspace(0, 1, width)[:, None] * (r + 1)
        arrangement = LAYOUTS and None
        return torch.tensor(out)

    def test_a_wide_batch_aligns_when_told_it_is_sequential(self):
        """The bug: `align=True` was silently inert for every batched eval."""
        step = self._step(align=True)
        matrices = self._matrices(4)
        step.decode_batch(matrices, sequential=True)
        assert len(step._anchors) == 4
        assert all(a is not None for a in step._anchors), "every row must carry its own anchor"

    def test_a_wide_batch_does_not_align_by_default(self):
        """A training batch's rows are unrelated samples; aligning them is nonsense."""
        step = self._step(align=True)
        step.decode_batch(self._matrices(4))
        assert all(a is None for a in step._anchors)

    def test_rows_get_independent_anchors(self):
        """Row 0's place on its curve must not leak into row 1's."""
        import numpy as np

        step = self._step(align=True)
        step.decode_batch(self._matrices(3), sequential=True)
        first, second = step._anchors[0], step._anchors[1]
        assert not np.allclose(first, second)

    def test_a_batch_of_one_still_aligns_without_the_flag(self):
        """The single-robot control loop has nowhere else to be."""
        step = self._step(align=True)
        step.decode_batch(self._matrices(1))
        assert step._anchors[0] is not None

    def test_reset_clears_every_row(self):
        step = self._step(align=True)
        step.decode_batch(self._matrices(2), sequential=True)
        step.reset()
        assert step._anchors == []

    def test_a_changed_batch_width_reallocates_the_anchors(self):
        step = self._step(align=True)
        step.decode_batch(self._matrices(4), sequential=True)
        step.decode_batch(self._matrices(2), sequential=True)
        assert len(step._anchors) == 2


class TestPredictBeforeEnd:
    """The tail of a predicted curve is held back rather than executed."""

    @staticmethod
    def _step(**kwargs):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        return BSplineDecodeStep(
            num_actions=8, degree=3, layout="ee6d20", arrangement="xvla_ee6d20", **kwargs
        )

    def test_zero_margin_executes_the_whole_curve(self):
        step = self._step(predict_before_end=0.0)
        actions, _ = step.decode_batch(TestDecodeBatchSequential._matrices(1))
        assert actions.shape[1] == 8

    def test_a_margin_holds_back_the_tail(self):
        """Upstream never drives the arm into the end of a prediction."""
        step = self._step(predict_before_end=0.5, fps=20.0)
        actions, rates = step.decode_batch(TestDecodeBatchSequential._matrices(1))
        per_action_s = float(rates[0]) / 20.0
        held = int(-(-0.5 // per_action_s))  # ceil
        assert actions.shape[1] == max(1, 8 - held)
        assert actions.shape[1] < 8

    def test_at_least_one_action_survives_any_margin(self):
        """A margin longer than the curve must still leave something to execute."""
        step = self._step(predict_before_end=1e6, fps=20.0)
        actions, _ = step.decode_batch(TestDecodeBatchSequential._matrices(1))
        assert actions.shape[1] == 1

    def test_the_anchor_is_the_last_executed_action_not_the_last_decoded_one(self):
        """Anchoring to an action never commanded would align to a place the arm
        never reached, which is the whole failure the margin exists to avoid."""
        import numpy as np

        held = self._step(align=True, predict_before_end=0.5, fps=20.0)
        full = self._step(align=True, predict_before_end=0.0)
        matrices = TestDecodeBatchSequential._matrices(1)
        held.decode_batch(matrices, sequential=True)
        full.decode_batch(matrices, sequential=True)
        assert not np.allclose(held._anchors[0], full._anchors[0])


class TestFixedRateDecodeStep:
    """The step-level half of `rate`: what a control loop actually receives."""

    def matrix(self, splines, episode=0, frame=5):
        return torch.from_numpy(splines.parameters(episode, frame).astype(np.float32))

    def span(self, splines, episode=0, frame=5):
        m = splines.parameters(episode, frame)
        return float(m[-(DEGREE + 1), 0] - m[DEGREE, 0])

    def test_the_row_count_follows_the_span(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(rate=2.0)
        for frame in (5, 40, 90):
            out = step({TransitionKey.ACTION: self.matrix(splines, 0, frame), TransitionKey.COMPLEMENTARY_DATA: {}})
            expected = int(np.floor(self.span(splines, 0, frame) / 2.0 + 1e-6)) + 1
            assert out[TransitionKey.ACTION].shape == (expected, 10)

    def test_the_realised_rate_is_the_configured_one(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        out = BSplineDecodeStep(rate=1.5)({
            TransitionKey.ACTION: self.matrix(splines), TransitionKey.COMPLEMENTARY_DATA: {}
        })
        assert float(out[TransitionKey.COMPLEMENTARY_DATA]["bspline_rate"][0]) == pytest.approx(1.5)

    def test_the_hold_back_is_measured_in_seconds_through_the_rate(self, splines):
        """0.06 s at 20 fps is 1.2 source frames: two rows at rate 1, one row at rate 2."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        def rows(rate, hold):
            out = BSplineDecodeStep(rate=rate, fps=20.0, predict_before_end=hold)({
                TransitionKey.ACTION: self.matrix(splines), TransitionKey.COMPLEMENTARY_DATA: {}
            })
            return out[TransitionKey.ACTION].shape[0]

        assert rows(1.0, 0.0) - rows(1.0, 0.06) == 2
        assert rows(2.0, 0.0) - rows(2.0, 0.06) == 1

    def test_the_config_carries_the_rate_and_rebuilds_from_it(self, splines):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(rate=1.9)
        config = step.get_config()
        assert config["rate"] == 1.9
        rebuilt = BSplineDecodeStep(**config)
        m = self.matrix(splines)
        torch.testing.assert_close(
            rebuilt({TransitionKey.ACTION: m, TransitionKey.COMPLEMENTARY_DATA: {}})[TransitionKey.ACTION],
            step({TransitionKey.ACTION: m, TransitionKey.COMPLEMENTARY_DATA: {}})[TransitionKey.ACTION],
        )

    def test_the_method_passes_it_through(self):
        (step,) = BSplineMethod(rate=1.0).postprocessor_steps()
        assert step.rate == 1.0
        assert BSplineMethod().postprocessor_steps()[0].rate is None

    def test_two_speed_knobs_at_once_are_refused(self):
        with pytest.raises(ValueError, match="two speed knobs"):
            BSplineMethod(rate=1.0, num_actions=8)

    def test_a_non_positive_rate_is_refused(self):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        with pytest.raises(ValueError, match="rate must be"):
            BSplineMethod(rate=0.0)
        with pytest.raises(ValueError, match="rate must be"):
            BSplineDecodeStep(rate=-1.0)


class TestGripperRamp:
    """A 0/1 command becomes a ramp the fit can follow; what the env executes is unchanged."""

    @staticmethod
    def command(length=200, edges=(60, 130)):
        g = np.zeros(length)
        state = 0.0
        for f in range(length):
            if f in edges:
                state = 1.0 - state
            g[f] = state
        return g

    def test_the_crossings_stay_on_the_recorded_edge_frames(self):
        g = self.command()
        for width in (5, 9, 13):
            r = ramp_gripper(g, width)
            assert r.min() >= 0.0 and r.max() <= 1.0
            # Same first frame above 0.5 and same first frame at or below it, every edge.
            np.testing.assert_array_equal(r > 0.5, g > 0.5)

    def test_zero_is_off_and_even_widths_are_refused(self):
        g = self.command()
        np.testing.assert_array_equal(ramp_gripper(g, 0), g)
        with pytest.raises(ValueError):
            ramp_gripper(g, 8)

    def test_the_fit_stops_overshooting_and_spends_fewer_knots(self):
        """The two things the ramp is for, measured on the fit itself."""
        length = 200
        actions = path(length=length, seed=3)
        actions[:, -1] = self.command(length)
        identity = LAYOUTS["identity"].__class__("identity", 10, lambda a: a, lambda a: a, 10)
        step = EpisodeSplines({0: actions}, identity, CHUNK)
        ramped = EpisodeSplines({0: actions}, identity, CHUNK, gripper_ramp=9)

        def knots_and_extremes(splines):
            chunks = splines.chunks[0]
            knots = np.unique(np.concatenate([chunks[0, :, 0], chunks[1:, -1, 0]]))
            gripper = chunks[:, : -(DEGREE + 1), -1]
            return len(knots), gripper.min(), gripper.max()

        n_step, lo_step, hi_step = knots_and_extremes(step)
        n_ramp, lo_ramp, hi_ramp = knots_and_extremes(ramped)
        assert hi_step > 1.02 or lo_step < -0.02, "the step fit should overshoot -- the premise"
        assert -0.05 <= lo_ramp and hi_ramp <= 1.05, (lo_ramp, hi_ramp)  # step fit: [-0.36, 1.36]
        assert n_ramp < n_step
