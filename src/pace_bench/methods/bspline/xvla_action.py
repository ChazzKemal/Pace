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
    #: Slots that hold nothing. Zeroed in the noised action (see `preprocess`), because
    #: the pretrained head still writes there and the generation loop would feed that
    #: back into the trunk.
    PAD_IDX = tuple(range(11, 20))

    #: 10.0, which is upstream's own value for an MSE gripper -- `agibot_ee6d`
    #: ("using MSE for all components") sets exactly this, while the 1.0 that stock
    #: `ee6d` uses belongs to its `BCEWithLogitsLoss`. This space regresses slot 9
    #: (see `postprocess`), so it inherited the wrong constant along with the wrong
    #: loss, and the two were fixed a commit apart.
    #:
    #: What 1.0 cost, measured on `ds_libero10_bspline` at 20k steps -- each term
    #: against its own mean-predictor floor, so the scales cancel:
    #:
    #:     term       scale   chance    final     R^2
    #:     position     500    5.746    0.521    0.91
    #:     knot          10    8.760    1.017    0.88
    #:     rotate6D      10    0.255    0.081    0.68
    #:     gripper        1    0.289    0.197    0.32
    #:
    #: The least-weighted channel was the least-learned one, by a wide margin, and it
    #: is the channel that decides whether anything is grasped: the arm scored 0% on
    #: every LIBERO-10 task while its total loss looked converged.
    GRIPPER_SCALE = 10.0
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
        """Slot 9 stays unmasked, unlike the stock ee6d space; the pad slots are zeroed.

        Upstream zeroes slot 9 of the *noisy* action (`modeling_xvla.py:223` in
        training, `:262` at every denoising step) while scoring the unmasked target.
        That is coherent there: slot 9 is a binary command read off a BCE head, so the
        channel is classified from the observation and never integrated along the flow.

        Here slot 9 is a least-squares coefficient regressed with MSE. Masking it hands
        the model x_t with that channel pinned to zero and then asks it to recover x_0,
        which no amount of training can do: with no information about where the channel
        started, the conditional mean is the optimal prediction and the model duly
        learns it. Measured on `ds_libero10_bspline`, the predicted gripper control
        points sat at mean +0.459 with 86.8% inside (0.1, 0.9) -- the mean of a binary
        {0, 1} channel -- against ground-truth fits that put ~3% there. The gripper
        never committed to open or closed, so nothing was ever grasped and the arm
        scored 0% on every LIBERO-10 task while its regression losses looked converged.

        Slots 11-19 are the opposite case, and the reason this method exists. They hold
        nothing -- the arranged matrix ends at the knot in slot 10 -- so `compute_loss`
        scores none of them, and the pretrained head keeps writing what it always did
        there: slot 19 was the second arm's gripper logit, and every checkpoint on this
        path emits about -23 in it. In training that is invisible, since the noised
        target has zeros there. At generation the model's own output is what gets
        noised for the next step, so from the second denoising step on the trunk reads
        a -20 in a slot where it only ever saw noise, and the prediction drifts.
        Measured on `ds_libero10_bspline_v3` at 10k steps, on training frames: one-step
        prediction from pure noise 2.18 cm per coordinate, ten-step generation 5.04 cm;
        with slot 19 zeroed in the feedback, generation 2.18 cm, and the knot from 0.94 s
        back to 0.25 s. Zeroing the pad in the noised input, in training and at every
        denoising step, makes the two distributions the same and takes the head's pad
        outputs out of the loop. Checkpoints trained before this mask existed are still
        evaluated with it, since the same measurement shows it helps them too (v2:
        3.59 -> 3.24 cm) -- though that checkpoint's gripper coefficient moved the other
        way (0.45 -> 0.77), which is one more reason the arm trained under the mask is
        the one to read.

        `proprio` keeps its mask: that channel really is the gripper's *state*, a 0/1
        reading, and hiding it is upstream's choice about observation leakage rather
        than anything to do with the action space.
        """
        proprio_m = proprio.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m = action.clone()
        action_m[..., self.PAD_IDX] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Returned as regressed -- no sigmoid, unlike the stock ee6d space.

        There, slot 9 is the gripper command itself, so BCE makes it a logit and a
        sigmoid inverts that. Here it is the gripper's B-spline control point, and a
        sigmoid could not express one: fitting a binary 0/1 channel overshoots to
        [-0.366, +1.366] at every edge (measured across libero_10_ee6d, 8.8% of
        coefficients beyond [-0.05, 1.05]), and that overshoot is precisely what
        encodes the edge.

        Nothing clips the decoded gripper either, and it would not help if it did.
        `LiberoProcessorStep`'s sibling `xvla_rotation_6d_to_axis_angle` already ends
        with ``np.where(g > 0.5, 1.0, -1.0)``, so the env takes a hard decision at 0.5
        regardless -- clipping to [0, 1] moves nothing across that line. What matters is
        that the *regression* put the curve confidently on one side of it: measured on
        the GRIPPER_SCALE=1.0 arm, the decoded gripper hovered at mean +0.426 and the
        threshold turned that hover into a flip every 6.1 executed steps, against one
        every 76.1 frames in the demonstrations. That is the whole reason this channel's
        weight is not a free parameter -- it is the only one the env passes through a
        hard nonlinearity, so a regression error that would be negligible anywhere else
        becomes categorical here.
        """
        return action


@register_action("bspline_uniform")
class UniformBSplineActionSpace(BaseActionSpace):
    """Upstream B-spline's own loss, on xVLA's trunk: one uniform MSE, no tuning.

    Pairs with ``--method.arrangement=knot_first20``, and only with that: this space reads
    the knot from slot 0 and the control point from slots 1..10, which is exactly what that
    arrangement emits. `ee6d_bspline` above exists to make xVLA's structured loss line up
    with a parameter matrix rearranged to look like an ee6d action. This space asks the opposite question:
    the arrangement was only ever needed because the *loss* slices by hardcoded index,
    and the loss is ours to write -- so drop the rearrangement and score the matrix
    upstream's way.

    What that removes. Upstream B-spline (`diffusion_unet_image_policy.py:192`) normalizes
    the parameter matrix per channel and then applies a single `F.mse_loss` over the whole
    thing, weighting nothing: knot column and control points count alike. Here that means
    no `xvla_ee6d20` arrangement, no `knot_scale`, and none of `KNOT_SCALE`, `GRIPPER_SCALE`,
    `XYZ_SCALE`, `ROT_SCALE` -- four constants that were set by argument rather than by
    measurement, two of them with no reference to inherit from at all.

    What it needs in exchange. A uniform loss only balances if the channels are comparable,
    which upstream gets from normalization and xVLA does not have: its `normalization_mapping`
    is IDENTITY throughout. Under `knot_first` the knot column is in *frames* (the knot_scale
    that carried seconds belonged to the arrangement), running to ~50 against positions of
    ~1.3, so an unnormalized uniform loss would be swamped by time. `BSplineMethod` therefore
    refuses this action space unless `--method.normalize_parameters=true`, which switches the
    action feature to MEAN_STD against statistics the method already computes.

    Column map, for the ``knot_first20`` vector at spline dimension 10 -- which is both
    shipped layouts, ``cart7`` and ``ee6d20``::

        0        the knot, in source frames relative to this sample
        1..3     xyz control point
        4..9     rot6d control point
        10       gripper control point
        11..19   XVLAPolicy's zero pad, scored by nothing

    ``dim_action`` stays 20 so `lerobot/xvla-libero`'s domain tables still load at their
    saved shape, and `knot_first20` emits at that width too -- so the action is one width
    end to end and the statistics, the normalizer and the unnormalizer all agree. (Emitting
    the natural 11 instead would make the policy pad to 20 *after* normalization, leaving
    the unnormalizer to meet a 20-wide tensor with 11-wide statistics.) Nine dead output
    channels is the price of not rebuilding those tables; an 11-wide action space is the
    cleaner end state and needs a load hook to drop the two mismatched tensors.

    `postprocess` is not overridden: nothing in this matrix is a gripper *command*, so
    there is no logit to squash. `preprocess` zeroes the pad slots in the noised action
    for the reason given on `EE6DBSplineActionSpace.preprocess` -- the pretrained head
    writes into them, the loss never reads them, and the generation loop would feed
    them back.
    """

    dim_action = 20
    #: No channel here is a 0/1 command, so the base class's "no gripper" default holds.
    gripper_idx = ()

    KNOT_IDX = (0,)
    POS_IDX = (1, 2, 3)
    ROT_IDX = (4, 5, 6, 7, 8, 9)
    GRIP_IDX = (10,)
    #: Real parameter channels. The rest of `dim_action` is pad.
    N_PARAMS = 11
    PAD_IDX = tuple(range(N_PARAMS, 20))

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def preprocess(self, proprio, action, mode="train"):
        action_m = action.clone()
        action_m[..., self.PAD_IDX] = 0.0
        return proprio, action_m

    def compute_loss(self, pred, target):
        """One uniform MSE over channels 0..10, reported in four parts.

        The four terms are *contributions*, not independent losses: each is its group's
        MSE times that group's share of the 11 channels, so they sum to exactly the
        uniform MSE over the parameter matrix. The weights are channel counts, not
        tuning knobs -- there is nothing here to sweep, which is the point. Split at
        all only because the trainer logs whatever this returns, and watching position
        against knot against gripper is how the arm gets diagnosed.
        """
        assert pred.shape == target.shape, "pred/target shapes must match"
        width = pred.shape[-1]
        if width < self.N_PARAMS:
            raise ValueError(
                f"bspline_uniform scores {self.N_PARAMS} parameter channels but the action "
                f"is {width} wide. This space expects the knot_first20 vector at spline "
                "dimension 10 (--method.arrangement=knot_first20, layout cart7 or ee6d20)."
            )

        def contribution(idx):
            return self.mse(pred[:, :, idx], target[:, :, idx]) * (len(idx) / self.N_PARAMS)

        return {
            "knot_loss": contribution(self.KNOT_IDX),
            "position_loss": contribution(self.POS_IDX),
            "rotate6D_loss": contribution(self.ROT_IDX),
            "gripper_loss": contribution(self.GRIP_IDX),
        }
