"""How a dataset's action vector maps to something a spline can be fitted through.

A B-spline is fitted through a *curve*, so every column it sees has to be a quantity
that interpolates. Which columns those are is a property of the dataset, not of the
method, and it differs across the three shapes in this project:

* ``cart7``      -- UR10e: xyz + angle-axis + gripper. The rotation must be converted:
  interpolating an angle-axis vector is meaningless across the pi wrap and merely
  wrong elsewhere, because the space it lives in is not the manifold.
* ``ee6d``       -- LIBERO: xyz + 6D rotation + gripper *already*, followed by zero
  padding to a fixed width (20 for ``libero_10_ee6d``). The pad columns must be
  dropped before fitting -- a constant column costs the fit nothing but widens every
  control point -- and restored on the way back.
* ``identity``   -- joint-space or anything already interpolable end to end.

Getting this wrong is silent. A transposed rotation still fits, still decodes, and
still produces a trajectory; it is simply the wrong trajectory. So the layout is
named explicitly per run rather than guessed from the action width.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pace_bench.methods.bspline.spline import from_spline_actions, to_spline_actions


@dataclass(frozen=True)
class ActionLayout:
    """A named conversion between a dataset's action space and the spline's."""

    name: str
    spline_dim: int
    to_spline: Callable[[np.ndarray], np.ndarray]
    from_spline: Callable[[np.ndarray], np.ndarray]
    #: Width of the dataset's own action vector, or None when the layout accepts any.
    raw_dim: int | None = None


def _ee6d_to_spline(raw: np.ndarray) -> np.ndarray:
    """Keep the first 10 columns; the rest are zero padding to a fixed width."""
    return np.asarray(raw, dtype=np.float64)[:, :10]


def _make_ee6d_from_spline(raw_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    def from_spline(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        out = np.zeros((len(actions), raw_dim), dtype=np.float64)
        out[:, :10] = actions
        return out

    return from_spline


def _identity(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions, dtype=np.float64)


LAYOUTS: dict[str, ActionLayout] = {
    # UR10e recordings: xyz(3) + angle-axis(3) + gripper(1) -> xyz + rot6d + gripper.
    "cart7": ActionLayout("cart7", 10, to_spline_actions, from_spline_actions, raw_dim=7),
    # LIBERO ee6d: xyz(3) + rot6d(6) + gripper(1), zero-padded to 20.
    "ee6d20": ActionLayout("ee6d20", 10, _ee6d_to_spline, _make_ee6d_from_spline(20), raw_dim=20),
    # Already interpolable at full width -- joint space, or an ee6d dataset with no pad.
    "identity": ActionLayout("identity", 0, _identity, _identity),
}


def resolve_layout(name: str, raw_dim: int) -> ActionLayout:
    """Look a layout up by name and check it against the dataset's actual width.

    The width check is the only guard against the silent failure mode: a run that
    names ``cart7`` for a 20-wide LIBERO action would otherwise fit the first seven
    columns as though three of them were a rotation vector.
    """
    if name not in LAYOUTS:
        raise ValueError(f"unknown action layout {name!r}; known: {sorted(LAYOUTS)}")
    layout = LAYOUTS[name]
    if layout.raw_dim is not None and layout.raw_dim != raw_dim:
        raise ValueError(
            f"action layout {name!r} expects a {layout.raw_dim}-dim action, but this "
            f"dataset's action is {raw_dim}-dim. Name the layout that matches it "
            f"(known: {sorted(LAYOUTS)})."
        )
    if layout.spline_dim == 0:  # "identity" adopts whatever the dataset is
        return ActionLayout(layout.name, raw_dim, layout.to_spline, layout.from_spline, raw_dim)
    return layout


@dataclass(frozen=True)
class MatrixArrangement:
    """How a parameter matrix's columns are laid out in the tensor a policy sees.

    Upstream puts the knot column first and the control points after it, which is
    right for any policy that treats its action vector as an undifferentiated set of
    numbers -- ACT and Diffusion both do. xVLA does not: its loss slices the action by
    hardcoded index (``POS_IDX_1 = (0, 1, 2)``, ``ROT_IDX_1 = (3..8)``, gripper at 9
    and 19), so a knot column at index 0 is trained as an x-coordinate. On LIBERO that
    showed up as a position loss of 122840 against a rotation loss of 6.3.
    """

    name: str
    #: Channels emitted, or None to mean "one knot column plus the control points".
    channels: int | None
    #: Multiplier on the knot column, for policies that do not normalize their
    #: actions. xVLA's normalization_mapping is IDENTITY throughout, so raw magnitudes
    #: reach its loss: knots in frames run to ~50 while positions in metres run to
    #: ~1.3, and the knot term then swamps everything by scale alone (a knot loss of
    #: 7365 beside a position loss of 2.1). Set to 1/fps, the knot column is seconds
    #: and the two are comparable. Inverted on `recover`, so the curve is unchanged.
    knot_scale: float = 1.0
    #: Where the knot column sits in the emitted vector. Knot-last is what xVLA's
    #: structured loss needs -- it wants slots 0..9 to be the pose it slices by index.
    #: Knot-first is upstream's own order and the one to keep when the loss does not
    #: slice, because it leaves the matrix reading as the `(knot, control points)` pair
    #: it actually is.
    knot_last: bool = True
    #: Whether `BSplineMethod` should rescale the knot column into seconds. Only a
    #: policy that does not normalize needs it: xVLA's IDENTITY mapping lets raw
    #: magnitudes reach the loss, and knots in frames (~50) would swamp positions
    #: (~1.3). An arrangement paired with real normalization must leave this off --
    #: the normalizer handles the scale, and a second one only obscures it.
    scale_knots_by_fps: bool = False

    def emit(self, matrix: np.ndarray) -> np.ndarray:
        """`(rows, 1 + spline_dim)` -> what the policy regresses."""
        if self.channels is None:
            return matrix
        out = np.zeros((matrix.shape[0], self.channels), dtype=matrix.dtype)
        points = matrix.shape[1] - 1
        if self.knot_last:
            out[:, :points] = matrix[:, 1:]
            out[:, points] = matrix[:, 0] * self.knot_scale
        else:
            out[:, 0] = matrix[:, 0] * self.knot_scale
            out[:, 1 : 1 + points] = matrix[:, 1:]
        return out

    def recover(self, emitted: np.ndarray, spline_dim: int) -> np.ndarray:
        """The inverse, for decoding what a policy predicted.

        Also the place a padded arrangement sheds its pad: `emitted` is as wide as the
        policy's action, `out` is as wide as the curve actually needs, and everything
        past `spline_dim` was never a parameter.
        """
        if self.channels is None:
            return emitted
        out = np.empty((emitted.shape[0], 1 + spline_dim), dtype=emitted.dtype)
        if self.knot_last:
            out[:, 0] = emitted[:, spline_dim] / self.knot_scale
            out[:, 1:] = emitted[:, :spline_dim]
        else:
            out[:, 0] = emitted[:, 0] / self.knot_scale
            out[:, 1:] = emitted[:, 1 : 1 + spline_dim]
        return out


ARRANGEMENTS: dict[str, MatrixArrangement] = {
    # Upstream's own: knot column first, then control points.
    "knot_first": MatrixArrangement("knot_first", None),
    # xVLA's ee6d vector: control point in slots 0..9 exactly where its structured
    # loss expects xyz / rot6d / gripper, the knot in slot 10, the rest zero. Keeps
    # the pretrained action decoder's width of 20 untouched.
    "xvla_ee6d20": MatrixArrangement("xvla_ee6d20", 20, knot_last=True, scale_knots_by_fps=True),
    # The same 20-wide vector, but in upstream's own order and with nothing rescaled:
    # knot in slot 0, control points in 1..10, zero pad in 11..19. For an xVLA run that
    # scores the matrix with one unweighted MSE (`--policy.action_mode=bspline_uniform`)
    # instead of xVLA's index-sliced loss -- nothing then needs the pose to sit where
    # POS_IDX/ROT_IDX expect it, so the rearrangement has no reason to exist. The width
    # stays 20 so the pretrained domain tables load at their saved shape, and so the
    # action is one width end to end: the statistics, the normalizer and the
    # unnormalizer all see 20, and `recover` sheds the pad on the way to the curve.
    "knot_first20": MatrixArrangement("knot_first20", 20, knot_last=False),
}


def resolve_arrangement(name: str) -> MatrixArrangement:
    if name not in ARRANGEMENTS:
        raise ValueError(f"unknown matrix arrangement {name!r}; known: {sorted(ARRANGEMENTS)}")
    return ARRANGEMENTS[name]


def coerce_layout(value: "ActionLayout | str | None") -> "ActionLayout | None":
    """Accept a layout or its name, and return the layout.

    ``get_config`` serialises these by name, so a step reconstructed from a
    checkpoint's ``policy_postprocessor.json`` is handed the *string*. Without this
    the name is stored verbatim and the failure surfaces deep in decoding as
    ``'str' object has no attribute 'recover'`` -- inside the inference subprocess,
    after the robot is up.

    ``identity`` cannot be rebuilt from a name: it adopts the dataset's own width, so
    the width is not recoverable from the config alone. That is refused explicitly
    rather than returning a layout whose ``spline_dim`` is 0.
    """
    if value is None or not isinstance(value, str):
        return value
    if value not in LAYOUTS:
        raise ValueError(f"unknown action layout {value!r}; known: {sorted(LAYOUTS)}")
    layout = LAYOUTS[value]
    if layout.raw_dim is None:
        raise ValueError(
            f"action layout {value!r} adopts the dataset's action width, so it cannot "
            "be rebuilt from a name alone. Pass the resolved ActionLayout (see "
            "`resolve_layout`) rather than its name."
        )
    return layout


def coerce_arrangement(value: "MatrixArrangement | str | None") -> "MatrixArrangement | None":
    """Accept an arrangement or its name, and return the arrangement. See `coerce_layout`."""
    if value is None or not isinstance(value, str):
        return value
    return resolve_arrangement(value)
