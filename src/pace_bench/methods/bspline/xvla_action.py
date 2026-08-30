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
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        gripper_loss = (
            sum(self.bce(pred[:, :, i], target[:, :, i]) for i in self.gripper_idx)
            / len(self.gripper_idx)
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
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action
