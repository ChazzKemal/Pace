"""An xVLA action space that understands a B-spline knot column.

xVLA does not treat its action vector as an undifferentiated set of numbers. Its
loss slices by hardcoded index -- `POS_IDX_1 = (0, 1, 2)`, `ROT_IDX_1 = (3..8)`,
gripper at 9 and 19 -- with per-group scales (`XYZ_SCALE = 500`, `ROT_SCALE = 10`)
that encode how much each part of the pose matters. Handing it a matrix whose first
column is a knot therefore trains a *time* as an x-coordinate; on LIBERO that showed
as a position loss of 122840 beside a rotation loss of 6.3.

So the parameter matrix is arranged for xVLA (see `layout.ARRANGEMENTS`) -- control
point in slots 0..9 where the structured loss expects it, knot in slot 10 -- and this
action space adds the knot as a fourth loss term. The pretrained decoder's width of
20 is unchanged; slots 11..19 stay zero, as they already are for a single arm.

The gripper is the second divergence, and for the same reason. Stock ee6d scores slot
9 with `BCEWithLogitsLoss` and inverts it with a sigmoid, which is right when the slot
holds a 0/1 command. Under this arrangement it holds the gripper's *B-spline control
point* -- a least-squares coefficient of the episode's fit. Fitting a binary channel
overshoots at every edge (control points span [-0.366, +1.366] on libero_10_ee6d; 8.8%
land outside [-0.05, 1.05]), and BCE is unbounded below on a target outside [0, 1], so
that term is minimised by saturating the logit rather than by matching the coefficient.
Slot 9 is therefore regressed with MSE like every other control point, and `postprocess`
returns the prediction untouched.

Registered under ``ee6d_bspline``, selected with ``--policy.action_mode=ee6d_bspline``.
Nothing in upstream xVLA or upstream B-spline is edited: the registry is xVLA's own
extension point, and B-spline on xVLA is this project's construction rather than a
port -- the paper uses Diffusion Policy and ACT.
"""

import torch
from lerobot.policies.xvla.action_hub import BaseActionSpace, register_action
from torch import nn


@register_action("ee6d_bspline")
class EE6DBSplineActionSpace(BaseActionSpace):
    """Single-arm ee6d control point in slots 0..9, B-spline knot in slot 10."""

    dim_action = 20
    gripper_idx = (9,)

    POS_IDX = (0, 1, 2)
    ROT_IDX = (3, 4, 5, 6, 7, 8)
    KNOT_IDX = (10,)

    #: An MSE weight here, not the BCE one it is upstream: slot 9 carries the
    #: gripper's *control point*, not a 0/1 command (see `postprocess`). Untuned, the
    #: same status as KNOT_SCALE -- the stock value is inherited from a different loss.
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    #: The one weight with no reference to inherit from -- upstream xVLA has no knot
    #: and upstream B-spline has no xVLA. Set to the rotation scale: the knot decides
    #: *when* the arm is somewhere, which matters more than orientation for a method
    #: whose whole purpose is speed, but less than being in the right place at all.
    #: Untuned; the first thing to sweep if a B-spline xVLA arm tracks poorly in time.
    KNOT_SCALE = 10.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx])
            * self.GRIPPER_SCALE
        )
        pos_loss = self.mse(pred[:, :, self.POS_IDX], target[:, :, self.POS_IDX]) * self.XYZ_SCALE
        rot_loss = self.mse(pred[:, :, self.ROT_IDX], target[:, :, self.ROT_IDX]) * self.ROT_SCALE
        knot_loss = (
            self.mse(pred[:, :, self.KNOT_IDX], target[:, :, self.KNOT_IDX]) * self.KNOT_SCALE
        )
        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
            "knot_loss": knot_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Mask the gripper channel exactly as the stock ee6d space does."""
        proprio_m, action_m = proprio.clone(), action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Returned as regressed -- no sigmoid, unlike the stock ee6d space.

        There, slot 9 is the gripper command itself, so BCE makes it a logit and a
        sigmoid inverts that. Here it is the gripper's B-spline control point, and a
        sigmoid could not express one: fitting a binary 0/1 channel overshoots to
        [-0.366, +1.366] at every edge (measured across libero_10_ee6d, 8.8% of
        coefficients beyond [-0.05, 1.05]), and that overshoot is precisely what
        encodes the edge. Clipping the gripper to [0, 1] is a *decode*-side decision,
        made in `BSplineDecodeStep` on executable actions, not on coefficients.
        """
        return action
