"""The unscored pad slots of the xVLA B-spline action spaces are zeroed in the noised input.

Why this exists: the pretrained ee6d head keeps writing its second-arm gripper logit
(about -23) into slot 19, which nothing in the B-spline loss scores. Training never
notices, because the noised *target* has zeros there. Generation does: the model's own
output is noised for the next denoising step, so the trunk reads -20 where it only ever
saw noise. Measured on `ds_libero10_bspline_v3` at 10k steps: one-step prediction from
pure noise 2.18 cm per coordinate, ten-step generation 5.04 cm; with the pad zeroed in
the feedback, 2.18 cm. The mask applies in training too, so the distributions match.
"""

from __future__ import annotations

import torch

import pace_bench.methods.bspline.xvla_action  # noqa: F401  (registers the spaces)
from lerobot.policies.xvla.action_hub import build_action_space


class TestEE6DBSplinePad:
    def test_the_pad_is_zeroed_and_every_parameter_slot_is_kept(self):
        space = build_action_space("ee6d_bspline")
        action = torch.randn(2, 16, 20)
        action[..., 19] = -23.0  # what the pretrained head emits there
        proprio = torch.randn(2, 20)
        proprio_m, action_m = space.preprocess(proprio, action)
        assert torch.equal(action_m[..., 11:], torch.zeros(2, 16, 9))
        torch.testing.assert_close(action_m[..., :11], action[..., :11])
        # slot 9 is a regressed coefficient and must stay visible to the flow
        torch.testing.assert_close(action_m[..., 9], action[..., 9])
        # the proprio gripper keeps upstream's mask
        assert torch.equal(proprio_m[..., 9], torch.zeros(2))
        torch.testing.assert_close(proprio_m[..., :9], proprio[..., :9])

    def test_the_input_is_not_modified_in_place(self):
        space = build_action_space("ee6d_bspline")
        action = torch.full((1, 16, 20), -23.0)
        space.preprocess(torch.zeros(1, 20), action)
        assert torch.equal(action, torch.full((1, 16, 20), -23.0))

    def test_the_loss_and_the_mask_partition_the_vector(self):
        """Every slot is either scored or zeroed; none is both, none is neither."""
        space = build_action_space("ee6d_bspline")
        scored = set(space.POS_IDX) | set(space.ROT_IDX) | set(space.gripper_idx) | set(space.KNOT_IDX)
        assert scored.isdisjoint(space.PAD_IDX)
        assert scored | set(space.PAD_IDX) == set(range(space.dim_action))


class TestUniformBSplinePad:
    def test_the_pad_is_zeroed_and_the_eleven_parameters_are_kept(self):
        space = build_action_space("bspline_uniform")
        action = torch.randn(3, 16, 20)
        proprio = torch.randn(3, 20)
        proprio_m, action_m = space.preprocess(proprio, action)
        assert torch.equal(action_m[..., 11:], torch.zeros(3, 16, 9))
        torch.testing.assert_close(action_m[..., :11], action[..., :11])
        torch.testing.assert_close(proprio_m, proprio)

    def test_the_loss_and_the_mask_partition_the_vector(self):
        space = build_action_space("bspline_uniform")
        scored = set(space.KNOT_IDX) | set(space.POS_IDX) | set(space.ROT_IDX) | set(space.GRIP_IDX)
        assert scored.isdisjoint(space.PAD_IDX)
        assert scored | set(space.PAD_IDX) == set(range(space.dim_action))
