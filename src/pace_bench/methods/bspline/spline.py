"""B-spline action chunks: fitting, chunking, and decoding.

The method (`B-spline Policy <https://github.com/B-spline-policy/bspline-policy>`_,
Han et al., arXiv:2607.09648) changes *what a policy regresses*. Instead of a dense
sequence of actions it predicts the parameters of a B-spline -- a knot vector plus
control points -- and the executable actions are that curve, sampled. This is a
different lever from the other methods here: PACE picks a speed per chunk and
DemoSpeedup drops frames by an uncertainty label, but both keep the action space.
B-spline changes the action space, and the speed follows from it. One chunk's knots
span far more source frames than the chunk has rows (on the UR10e, ~59 frames for a
26-row matrix), so evaluating the curve at N points walks the whole demonstrated
path in N steps regardless of how many frames it was recorded over.

The parameter matrix is ``(chunk_size + 2 * degree, 1 + action_dim)``: column 0 is
the knot vector, the rest are control points. Knots are in *source frames, relative
to the current frame*, so a policy conditioned on one observation predicts both the
shape of the path and the time it should take.

Fitting is per episode and adaptive. :func:`fit_episode` grows the knot count until
the least-squares spline is within ``max_error`` of the demonstration, so a fast
straight transit costs few knots and a slow precise insertion costs many -- the same
place DemoSpeedup gets its signal from, reached without a policy to ask.

Rotations are carried as 6D, never as the axis-angle the UR10e records.
Interpolating axis-angle is meaningless near the pi wrap and merely wrong elsewhere,
because the vector space it lives in is not the manifold. The 6D form (Zhou et al.,
the first two rows of the rotation matrix) is continuous, so a spline through it
stays close to SO(3) and :func:`from_spline_actions` projects back exactly.
"""

import numpy as np
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline
from scipy.spatial.transform import Rotation

#: Cubic. Upstream's every config and the recorded UR10e dataset agree on it.
DEGREE = 3
#: Max fit error, in action units, before the knot search stops adding knots.
MAX_ERROR = 0.01
#: `generate_knots` smoothing floor. Upstream's value; effectively "interpolate".
SMOOTHING = 1e-12

#: cart7, as the UR10e records it: metres, angle-axis rotation vector, gripper.
RAW_DIM = 7
#: What the spline is fitted through: metres, 6D rotation, gripper.
SPLINE_DIM = 10


def to_spline_actions(raw: np.ndarray) -> np.ndarray:
    """cart7 ``(T, 7)`` -> ``(T, 10)``, replacing angle-axis with a 6D rotation."""
    raw = np.asarray(raw, dtype=np.float64)
    matrices = Rotation.from_rotvec(raw[:, 3:6]).as_matrix()
    return np.concatenate(
        [raw[:, :3], matrices[:, :2, :].reshape(len(raw), 6), raw[:, 6:7]], axis=1
    )


def from_spline_actions(actions: np.ndarray) -> np.ndarray:
    """``(T, 10)`` -> cart7 ``(T, 7)``, projecting the 6D rotation back onto SO(3).

    A sampled spline lands near the rotation manifold but not on it, so the two
    rows are re-orthonormalised (Gram-Schmidt, then a cross product for the third)
    before the matrix is read as a rotation. Without this the conversion is
    ill-defined exactly where the curve is most useful -- between control points.
    """
    actions = np.asarray(actions, dtype=np.float64)
    first = actions[:, 3:6]
    second = actions[:, 6:9]
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second - (second * first).sum(axis=1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    matrices = np.stack([first, second, np.cross(first, second)], axis=1)
    rotvec = Rotation.from_matrix(matrices).as_rotvec()
    return np.concatenate([actions[:, :3], rotvec, actions[:, 9:10]], axis=1)


def fit_episode(
    actions: np.ndarray,
    max_error: float = MAX_ERROR,
    degree: int = DEGREE,
    smoothing: float = SMOOTHING,
) -> tuple[BSpline, bool]:
    """Least-squares spline through a whole episode, with as few knots as it allows.

    ``generate_knots`` yields knot vectors of increasing size; the first one whose
    spline is within ``max_error`` everywhere wins. Returns that spline and whether
    the tolerance was actually met -- upstream prints and continues when it is not,
    and the caller here is told instead, because "the fit silently missed" is the
    one failure that would corrupt a dataset without any other symptom.

    Args:
        actions: ``(T, dim)``, already in the interpolable representation.
    """
    t = np.arange(len(actions))
    spline = None
    for knots in generate_knots(t, actions, s=smoothing):
        spline = make_lsq_spline(t, actions, knots)
        if np.abs(spline(t) - actions).max() < max_error:
            return spline, True
    if spline is None:  # pragma: no cover -- generate_knots always yields once
        raise ValueError("generate_knots produced no candidate knot vector")
    return spline, False


def chunk_parameters(
    spline: BSpline, chunk_size: int, degree: int = DEGREE, stride: int = 1
) -> list[np.ndarray]:
    """Cut a fitted spline into ``(chunk_size + 2 * degree, 1 + dim)`` matrices.

    Each chunk is a *window* of the full knot vector and its control points, not a
    clamped spline of its own: only the first chunk carries the leading repeated
    boundary knots. That is what makes the windows composable -- a chunk starting
    mid-episode describes the curve there using the same knots the whole-episode fit
    chose. Short windows at the tail are padded by repeating the last knot and
    control point, which parks the curve rather than extrapolating it.
    """
    knots, coefficients, _ = spline.tck
    width = chunk_size + 2 * degree
    chunks = []
    for start in range(0, len(knots[degree:-degree]) - 1, stride):
        lo = max(0, start)
        hi = min(len(knots), start + chunk_size + 2 * degree)
        window_t, window_c = knots[lo:hi], coefficients[lo:hi]
        # Padded independently, because they run out at different points: scipy's
        # coefficient array is `degree + 1` shorter than its knot vector, so a window
        # at the tail can be full-length in `t` and still short in `c`.
        if len(window_t) < width:
            window_t = np.concatenate([window_t, np.full(width - len(window_t), window_t[-1])])
        if len(window_c) < width:
            window_c = np.concatenate(
                [window_c, np.repeat(window_c[-1:], width - len(window_c), axis=0)], axis=0
            )
        chunk = np.empty((width, 1 + coefficients.shape[1]), dtype=np.float64)
        chunk[:, 0] = window_t
        chunk[:, 1:] = window_c
        chunks.append(chunk)
    return chunks


def assign_chunks_to_frames(
    chunks: list[np.ndarray], length: int, degree: int = DEGREE
) -> np.ndarray:
    """Give every frame its chunk, knots shifted to offsets from that frame.

    Successive frames sharing a chunk differ only by the shift, which is what lets
    a policy predict the same curve from anywhere along it.

    Deviates from upstream in one place: upstream's loop runs while the frame index
    is ``<= chunk_t[degree]`` with no episode bound, so a final chunk whose knot
    reaches past the episode writes into the next episode's slots -- and off the end
    of the array for the last episode. Here the walk is clamped to the episode,
    which is what upstream means everywhere the two agree.
    """
    out = np.zeros((length, *chunks[0].shape), dtype=np.float64)
    frame = 0
    for chunk in chunks:
        while frame < length and frame <= chunk[degree, 0]:
            out[frame] = chunk
            out[frame, :, 0] -= frame
            frame += 1
    while frame < length:  # tail: the last chunk, shifted, until the episode ends
        out[frame] = chunks[-1]
        out[frame, :, 0] -= frame
        frame += 1
    return out


def episode_parameter_chunks(
    actions: np.ndarray,
    chunk_size: int,
    degree: int = DEGREE,
    max_error: float = MAX_ERROR,
    stride: int = 1,
) -> tuple[np.ndarray, bool]:
    """One parameter matrix per frame: ``(T, chunk_size + 2 * degree, 1 + dim)``.

    Fit, cut, assign. Callers that need the fitted spline as well should compose
    :func:`fit_episode`, :func:`chunk_parameters` and :func:`assign_chunks_to_frames`
    directly rather than fitting twice -- the adaptive fit is the expensive step.
    """
    spline, converged = fit_episode(actions, max_error=max_error, degree=degree)
    chunks = chunk_parameters(spline, chunk_size, degree=degree, stride=stride)
    return assign_chunks_to_frames(chunks, len(actions), degree=degree), converged


def encode_relative_knots(parameters: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """Rewrite the knot column as first-valid-knot plus consecutive differences.

    Absolute knots are a poor regression target because their value is mostly
    decided by *which row* they are: on the UR10e the knot column averages -7.7 at
    row 0 and +50.9 at row 25, so a single per-column normalisation statistic --
    which is all LeRobot's normaliser computes -- leaves a deterministic ramp of
    about 2.9 normalised units in the target, with the real per-sample signal only
    0.44 of a unit on top of it. Differences are stationary across rows (row means
    1.16 / 2.27 / 1.96), which puts the knot column on the same footing as the
    control points and makes per-column normalisation the right thing.

    Slot 0 holds the first *valid* knot (index ``degree``) so the absolute position
    of the span survives; every later slot holds a step. Inverse of
    :func:`decode_relative_knots`.
    """
    result = np.array(parameters, dtype=np.float64, copy=True)
    original = result[..., 0].copy()
    result[..., 0, 0] = original[..., degree]
    result[..., 1:, 0] = original[..., 1:] - original[..., :-1]
    return result


def decode_relative_knots(parameters: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """Invert :func:`encode_relative_knots`, rebuilding absolute knots."""
    result = np.array(parameters, dtype=np.float64, copy=True)
    encoded = result[..., 0].copy()
    knots = result[..., 0]
    knots[..., degree] = encoded[..., 0]
    for index in range(degree - 1, -1, -1):
        knots[..., index] = knots[..., index + 1] - encoded[..., index + 1]
    for index in range(degree + 1, knots.shape[-1]):
        knots[..., index] = knots[..., index - 1] + encoded[..., index]
    return result


def monotonic_knots(knots: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Force a knot vector to be non-decreasing, nudging each violation upwards.

    Upstream's `safer_knots`, applied at deployment in both of its policy scripts.
    It is needed because nothing in the loss constrains a predicted knot column to be
    ordered -- the network emits 16 numbers and their order is a property it has to
    learn -- while a B-spline is undefined on an unordered knot vector.

    Deliberately a nudge rather than a sort: a sort would silently reorder the curve's
    control points relative to its knots, changing which part of the trajectory each
    control point governs. Nudging keeps every control point where it was and only
    collapses the offending span to zero length.
    """
    knots = np.asarray(knots, dtype=np.float64).copy()
    for index in range(1, len(knots)):
        if knots[index] < knots[index - 1]:
            knots[index] = knots[index - 1] + epsilon
    return knots


def decode_chunk(
    parameters: np.ndarray,
    num_actions: int,
    degree: int = DEGREE,
    relative_knots: bool = False,
) -> np.ndarray:
    """Evaluate one parameter matrix into ``(num_actions, dim)`` actions.

    The curve is sampled uniformly across the chunk's valid span, which runs from
    ``knots[degree]`` to ``knots[-(degree + 1)]``. ``num_actions`` is the whole
    speed knob and is free at inference: the span is a fixed stretch of demonstrated
    motion, so asking for fewer samples covers it in fewer executed steps.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if relative_knots:
        parameters = decode_relative_knots(parameters, degree=degree)
    # A knot vector must be non-decreasing. Nothing constrains a *predicted* one to
    # be, and scipy will either refuse it or return nonsense -- so violations are
    # nudged, which is what upstream does at deployment (`safer_knots`, in both its
    # policy scripts). A no-op on any knot vector that came from a real fit.
    knots = monotonic_knots(parameters[:, 0])
    parameters = parameters.copy()
    parameters[:, 0] = knots
    control_points = parameters[: -(degree + 1), 1:]
    start, end = knots[degree], knots[-(degree + 1)]
    if not end > start:
        raise ValueError(f"chunk spans no time: knots run [{start}, {end}]")
    if num_actions <= 1:
        samples = np.asarray([start], dtype=np.float64)
    else:
        samples = np.linspace(start, end, int(num_actions), dtype=np.float64)
    # Take the last sample as a limit from the left. A chunk at the episode tail is
    # padded by repeating its final knot, so `end` sits at the bottom of a run of
    # identical knots -- a zero-length span, where every basis function is zero and
    # the curve evaluates to the all-zero vector. Upstream returns that zero as an
    # action; in an absolute action space it is a command to the world origin, and
    # after a 6D rotation is normalized it is a division by zero. One ulp inside, the
    # value is exactly the last control point, which is what the endpoint means.
    samples = np.minimum(samples, np.nextafter(end, start))
    return BSpline(knots, control_points, degree, extrapolate=False)(samples)
