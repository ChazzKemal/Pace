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
