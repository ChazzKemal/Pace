"""The decode step must be fed parameters in natural units.

`attach_bspline` rebinds `select_action`, so decoding happens *inside* the policy --
upstream of the postprocessor that would undo action normalization. A checkpoint with
MEAN_STD actions therefore hands the decoder z-scores unless the statistics travel
with it, and the failure is silent in the worst way: normalized knots average about
zero, so every chunk spans ~0 frames, every decode yields a single action, and the arm
crawls one step at a time while the run reports an ordinary-looking 0% success rate.

Measured on `ds_libero10_bspline_uniform_posemb` before the fix: 517 decoder queries
for a 520-step episode (1.01 actions each), knot column mean -1.06 against saved
statistics of mean 16.95 / std 19.30.
"""

from __future__ import annotations

import numpy as np
import torch

from pace_bench.eval.bspline_policy import _unnormalizer


class TestUnnormalizer:
    def test_identity_when_the_policy_is_already_absolute(self):
        """Every xVLA arm that keeps NormalizationMode.IDENTITY, which `action_stats`
        reports as None. Must not touch the tensor."""
        x = torch.randn(2, 16, 20)
        for stats in (None, {}, {"mean": torch.zeros(20)}):  # std missing -> not usable
            torch.testing.assert_close(_unnormalizer(stats)(x), x)

    def test_it_restores_the_training_units(self):
        mean = torch.arange(20, dtype=torch.float32)
        std = torch.full((20,), 2.0)
        normalized = torch.randn(3, 16, 20)
        restored = _unnormalizer({"mean": mean, "std": std})(normalized)
        torch.testing.assert_close(restored, normalized * std + mean)

    def test_the_knot_column_comes_back_at_its_real_scale(self):
        """The column that actually breaks. A knot in z-space is ~0, and a chunk whose
        knots are all ~0 spans no time -- which `decode_chunk` either refuses outright
        or turns into a single-action crawl."""
        # The real saved statistics for this arm's knot column.
        mean = torch.zeros(20); mean[0] = 16.954
        std = torch.ones(20); std[0] = 19.300
        # A plausible normalized knot column: increasing, roughly unit scale.
        normalized = torch.zeros(1, 16, 20)
        normalized[0, :, 0] = torch.linspace(-0.9, 1.7, 16)
        out = _unnormalizer({"mean": mean, "std": std})(normalized)[0, :, 0]
        span = float(out[-4] - out[3])
        assert span > 20, f"a real chunk spans tens of source frames, got {span:.1f}"
        assert float(normalized[0, -4, 0] - normalized[0, 3, 0]) < 3, (
            "and the un-restored column spans almost nothing, which is the bug"
        )


class TestAttachPassesTheStats:
    def test_a_normalizing_checkpoint_gets_its_units_back(self):
        """The wiring, not the arithmetic: `attach_bspline` must actually install the
        unnormalizer where `select_action` will reach it."""
        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        class Stub:
            def reset(self):
                pass

        policy = Stub()
        stats = {"mean": torch.full((20,), 5.0), "std": torch.full((20,), 3.0)}
        attach_bspline(policy, BSplineDecodeStep(num_actions=4), None, action_stats=stats)
        got = policy.bspline_unnormalize(torch.zeros(1, 16, 20))
        torch.testing.assert_close(got, torch.full((1, 16, 20), 5.0))

    def test_an_identity_checkpoint_is_left_alone(self):
        from pace_bench.eval.bspline_policy import attach_bspline
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        class Stub:
            def reset(self):
                pass

        policy = Stub()
        attach_bspline(policy, BSplineDecodeStep(num_actions=4), None, action_stats=None)
        x = torch.randn(1, 16, 20)
        torch.testing.assert_close(policy.bspline_unnormalize(x), x)


class TestGripperIsNotMasked:
    """`ee6d_bspline` scores slot 9 with MSE against a least-squares coefficient, so it
    must not also hide that slot from the model. Masking x_t while supervising x_0
    leaves the conditional mean as the optimal prediction, and the arm never grasps."""

    def test_the_action_reaches_the_model_intact(self):
        from pace_bench.methods.bspline.xvla_action import EE6DBSplineActionSpace

        space = EE6DBSplineActionSpace()
        proprio = torch.ones(2, 16, 20)
        action = torch.ones(2, 16, 20)
        proprio_m, action_m = space.preprocess(proprio, action)
        assert action_m[..., 9].eq(1).all(), (
            "slot 9 is a spline coefficient regressed with MSE; zeroing it in x_t makes "
            "it unlearnable and the gripper collapses to the mean of a binary channel"
        )
        assert proprio_m[..., 9].eq(0).all(), "the proprio gripper *state* stays masked"


class TestKnotScaleSurvivesSerialisation:
    """`knot_scale` is 1/fps -- a property of the run, not of the arrangement -- so a
    step rebuilt from the arrangement's name alone would divide by 1 where `emit`
    multiplied by 0.05, and every decoded curve would span twenty times too long."""

    def test_the_config_round_trips(self):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep
        from pace_bench.methods.config import BSplineMethod

        method = BSplineMethod(layout="ee6d20", chunk_size=10, degree=3, fps=20.0,
                               arrangement="xvla_ee6d20", num_actions=8)
        (step,) = method.postprocessor_steps()
        assert step.arrangement.knot_scale == 1 / 20

        config = step.get_config()
        assert config["knot_scale"] == 1 / 20
        assert config["fps"] == 20.0
        assert config["predict_before_end"] == method.predict_before_end

        rebuilt = BSplineDecodeStep(**config)
        assert rebuilt.arrangement.knot_scale == step.arrangement.knot_scale
        assert rebuilt.fps == step.fps
        assert rebuilt.predict_before_end == step.predict_before_end

    def test_a_name_without_the_scale_still_builds(self):
        """knot_first20 does not scale, so the name alone is complete for it."""
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(num_actions=8, arrangement="knot_first20")
        assert step.arrangement.knot_scale == 1.0


class TestBothStepsRoundTripTheirConfig:
    """A `get_config` key with no matching `__init__` parameter makes every checkpoint
    the run writes unloadable, and the run itself never notices -- training builds its
    steps from the config object, not from the checkpoint. Found the hard way: adding
    `knot_scale` to `BSplineDecodeStep.get_config` by string replacement also added it
    to `BSplineChunkStep.get_config`, whose `__init__` did not take it, so
    `ds_libero10_bspline_v2` trained for 5000 steps writing checkpoints that raised
    `unexpected keyword argument 'knot_scale'` on load.
    """

    def test_every_key_a_step_serialises_can_be_passed_back(self):
        from pace_bench.methods.bspline.processor import BSplineChunkStep, BSplineDecodeStep

        for step in (
            BSplineChunkStep(arrangement="xvla_ee6d20", knot_scale=0.05, degree=3),
            BSplineDecodeStep(num_actions=16, arrangement="xvla_ee6d20", knot_scale=0.05),
            BSplineDecodeStep(num_actions=8, arrangement="knot_first20", layout="ee6d20"),
        ):
            config = step.get_config()
            rebuilt = type(step)(**config)  # the exact call lerobot's registry makes
            assert rebuilt.get_config() == config, (
                f"{type(step).__name__} does not survive its own config"
            )
