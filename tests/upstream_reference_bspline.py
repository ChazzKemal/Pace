"""Upstream B-spline Policy, copied verbatim, so this project's port has a reference.

Every definition below is a byte-exact copy from B-spline-policy/bspline-policy
at commit ``61ed5f4``:

  * ``ScipyBSplineCompression``    bspline_policy/common/bspline_action.py
  * ``chunk_bspline_trajectory``   bspline_policy/common/bspline_action.py
  * ``decode_bspline_action``      bspline_policy/common/bspline_action.py
  * ``encode_relative_knots``      bspline_policy/common/knots.py
  * ``decode_relative_knots``      bspline_policy/common/knots.py

Copied rather than depended on, for the same reason as
:mod:`tests.upstream_reference`: the repo is a research distribution built on a
vendored fork of ``diffusion_policy``, and ``bspline_action.py`` alone imports
``diffusion_policy.common.replay_buffer`` and ``filelock``. The four functions and
one class here are the entire surface this project checks itself against.

**Do not tidy this code.** Its value is being unchanged, so that "matches upstream"
keeps meaning something. The port lives in ``pace_bench.methods.bspline``; the one
deliberate divergence -- clamping the per-frame chunk assignment to the episode --
is documented and tested there.

These were extracted with ``ast.get_source_segment``, not retyped.
"""

# `Optional[int]` and the import order are upstream's own. Rewriting them to
# `int | None` would be an edit to the reference, which is the one thing this file
# must not contain -- so the lint is silenced rather than satisfied.
# ruff: noqa: I001, UP045

import hashlib  # noqa: F401  -- part of the copied module's namespace
from typing import Optional

import numpy as np
import torch
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

def encode_relative_knots(action_data, degree: int = 3):
    """Encode knot values as first valid knot plus adjacent differences."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    knots = result[..., 0]
    original_knots = knots.clone() if torch.is_tensor(knots) else knots.copy()

    knots[..., 0] = original_knots[..., degree]
    knots[..., 1:] = original_knots[..., 1:] - original_knots[..., :-1]
    return result

def decode_relative_knots(action_data, degree: int = 3):
    """Decode the representation produced by encode_relative_knots."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    encoded = result[..., 0].clone() if torch.is_tensor(result) else result[..., 0].copy()
    knots = result[..., 0]
    n_knots = knots.shape[-1]

    knots[..., degree] = encoded[..., 0]
    for knot_idx in range(degree - 1, -1, -1):
        knots[..., knot_idx] = knots[..., knot_idx + 1] - encoded[..., knot_idx + 1]
    for knot_idx in range(degree + 1, n_knots):
        knots[..., knot_idx] = knots[..., knot_idx - 1] + encoded[..., knot_idx]

    return result

class ScipyBSplineCompression:
    """Fit a multi-dimensional trajectory with a reduced-knot B-spline."""

    def __init__(self, degree: int = 3):
        self.degree = int(degree)
        self.spline = None
        self.knots = None

    def compress(
        self,
        data: np.ndarray,
        max_error: float = 0.01,
        verbose: bool = False,
        s: float = 1e-12,
    ) -> np.ndarray:
        t = np.arange(len(data))
        last_knots = None
        last_error = None
        for knots in generate_knots(t, data, s=s):
            spl = make_lsq_spline(t, data, knots)
            pred_data = spl(t)
            error = np.abs(pred_data - data).max()
            last_knots = knots
            last_error = error
            if error < max_error:
                self.knots = knots
                self.spline = spl
                break

        if self.knots is None:
            print(
                "Failing to compress trajectory with max error "
                f"{max_error}, use min error we can find. Error is {last_error}. "
                "You can try to increase the s value."
            )
            self.knots = last_knots
            self.spline = make_lsq_spline(t, data, self.knots)

        if verbose:
            print(f"compression ratio: {len(self.knots) / len(t)}")

        return self.knots

def extract_unique_knots(t_full: np.ndarray, degree: int) -> np.ndarray:
    """Extract the unique knot span from FITPACK's repeated-boundary format."""
    return t_full[degree:-degree]


def chunk_bspline_trajectory(
    compressor: ScipyBSplineCompression,
    chunk_size: int = 8,
    stride: Optional[int] = None,
    episode_length: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
    """Split a fitted B-spline into fixed-size parameter chunks."""
    del episode_length
    if compressor.spline is None:
        raise ValueError("Please call compress() before chunking")

    if stride is None:
        stride = chunk_size - 1

    degree = compressor.degree
    t_full, c_full, _ = compressor.spline.tck
    unique_t = extract_unique_knots(t_full, degree)
    n_unique = len(unique_t)
    chunks = []

    if verbose:
        print(
            f"B-spline chunking: len(t)={len(t_full)}, len(c)={len(c_full)}, "
            f"degree={degree}, unique_knots={n_unique}, chunk_size={chunk_size}, "
            f"stride={stride}"
        )

    for start_idx in range(0, n_unique - 1, stride):
        first_pos = start_idx + degree
        last_pos = start_idx + chunk_size + degree

        t_start = max(0, first_pos - degree)
        t_end = min(len(t_full), last_pos + degree)

        chunk_t = t_full[t_start:t_end]
        chunk_c = c_full[t_start:t_end]
        expected_len = chunk_size + 2 * degree

        if len(chunk_t) < expected_len:
            chunk_t = np.concatenate(
                [chunk_t, np.full(expected_len - len(chunk_t), chunk_t[-1])]
            )
        if len(chunk_c) < expected_len:
            pad = np.repeat(chunk_c[-1:], expected_len - len(chunk_c), axis=0)
            chunk_c = np.concatenate([chunk_c, pad], axis=0)

        if len(chunk_t) != expected_len:
            raise AssertionError("chunk_t length should equal chunk_size + 2 * degree")
        if len(chunk_c) != expected_len:
            raise AssertionError("chunk_c length should equal chunk_size + 2 * degree")

        chunks.append({"t": chunk_t, "c": chunk_c, "k": degree})

    return chunks

def decode_bspline_action(
    action_params,
    degree: int = 3,
    num_actions: int = 8,
    relative_knots: bool = False,
) -> np.ndarray:
    """Decode one B-spline parameter matrix into regular action vectors."""
    if torch.is_tensor(action_params):
        action_params = action_params.detach().cpu().numpy()
    action_params = np.asarray(action_params, dtype=np.float64)
    if relative_knots:
        action_params = decode_relative_knots(action_params, degree=degree)

    knots = action_params[:, 0].copy()
    control_points = action_params[: -(degree + 1), 1:].copy()
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")

    if num_actions <= 1:
        t_eval = np.asarray([t_min], dtype=np.float64)
    else:
        t_eval = np.linspace(t_min, t_max, int(num_actions), dtype=np.float64)
    return BSpline(knots, control_points, degree, extrapolate=False)(t_eval).astype(
        np.float32
    )
