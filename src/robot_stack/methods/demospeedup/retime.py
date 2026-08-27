"""DemoSpeedup retiming: one stride walk, two ways of applying it.

A demonstration is not uniformly informative. Most of it is transit -- moving to
roughly the right place -- and a small fraction is precision: the approach, the
grasp, the insertion. DemoSpeedup measures that with the entropy of a proxy
policy's action distribution, labels each frame *precision* or *not*, and then
plays the demonstration back at a variable rate: one waypoint every ``low_v``
frames where precision matters, one every ``high_v`` frames where it does not.
Executed at the original control rate, that chunk drives the arm 2-4x faster
exactly where speed is free.

The walk itself is :func:`keep_indices`, and it exists once here. Two ways of
applying it were previously implemented separately:

* **chunk level** (:func:`retime_chunk`) -- subsample the action chunk that a given
  observation is trained against, leaving the dataset alone. Every frame remains a
  training sample. This is upstream's formulation and the fork's.
* **episode level** (:func:`episode_keep_indices`) -- keep only the walked frames of
  the episode, producing a shorter dataset. The (observation, chunk) pairs match,
  but dropped frames no longer start chunks, so there are 2-4x fewer samples.

Only chunk-level retiming is used here; episode-level is kept because the recorded
real-robot dataset was built with it, and reproducing that selection is what proves
the two agree.

The acceleration lives in the action deltas, not in any metadata: replay at the
source fps, or the speedup is applied twice.
"""

import numpy as np
import torch

# Upstream's defaults: half rate through precision, quarter rate elsewhere.
LOW_V = 2
HIGH_V = 4


def keep_indices(labels, low_v: int = LOW_V, high_v: int = HIGH_V, start: int = 0) -> list[int]:
    """The stride walk. Returns the indices it lands on, excluding the start anchor.

    At a precision frame take a ``low_v`` step. At a non-precision frame take a
    ``high_v`` step only if the whole span stays non-precision -- otherwise jump to
    the next precision frame instead, so a fast stride can never skip over the
    beginning of a precision segment. That guard is the whole safety argument: speed
    is only taken where the label says the entire jump is uninformative.

    ``start=-1`` reproduces upstream's literal loop, whose first consulted label is
    ``labels[-1]`` -- the *last* frame, almost certainly a slip. Both implementations
    in this project use ``start=0``; the option exists so the difference is testable
    rather than folklore.
    """
    labels = np.asarray(labels)
    horizon = len(labels)
    indices: list[int] = []
    i = start
    while i < horizon:
        if labels[i] == 0 and i + low_v < horizon:
            i += low_v
            indices.append(i)
        elif labels[i] == 1:
            if i + high_v < horizon and np.all(labels[i : i + high_v] == 1):
                i += high_v
                indices.append(i)
            else:
                next_precision = np.flatnonzero(labels[i + 1 :] == 0)
                if len(next_precision) == 0:
                    break  # non-precision all the way to the end; nothing left to land on
                i = i + 1 + int(next_precision[0])
                indices.append(i)
        else:
            i += 1
    return indices


def episode_keep_indices(
    labels, low_v: int = LOW_V, high_v: int = HIGH_V, keep_last: bool = True
) -> np.ndarray:
    """Frames of a whole episode to keep. Frame 0 is always the anchor.

    With ``keep_last``, the final frame is appended and the gap to it filled at
    stride ``high_v``: the walk stops short of the end, and dropping a demonstration's
    last pose would truncate the task itself.

    Retained for checking against the recorded real-robot dataset, which was built
    this way; training here retimes chunks instead.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    keep = [0, *keep_indices(labels, low_v, high_v, start=0)]
    if keep_last and keep[-1] != n - 1:
        keep.extend(range(keep[-1] + high_v, n - 1, high_v))
        keep.append(n - 1)
    out = np.array(sorted({int(k) for k in keep}), dtype=np.int64)
    return out[(out >= 0) & (out < n)]


def retime_chunk(
    actions: torch.Tensor,
    labels,
    is_pad: torch.Tensor | None = None,
    low_v: int = LOW_V,
    high_v: int = HIGH_V,
    pad_mode: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Subsample one action chunk in place of its uniform-rate original.

    The chunk keeps its original length; the walked waypoints are packed to the
    front and the tail is filled per ``pad_mode``:

    * ``"zero"`` leaves zeros, correct when the loss is masked by ``action_is_pad``
      (ACT, Diffusion).
    * ``"hold"`` repeats the last kept waypoint, required when the policy regresses
      the whole chunk unmasked (xVLA's flow matching). In an *absolute* action space
      a zero tail is not neutral -- it is a command to drive to the world origin.

    Args:
        actions: ``(horizon, action_dim)``.
        labels: ``(horizon,)`` in {0, 1}; 0 = precision.
        is_pad: optional ``(horizon,)`` mask, subsampled alongside.

    Returns:
        ``(retimed_actions, retimed_is_pad)``, both at the original horizon.
    """
    if pad_mode not in ("zero", "hold"):
        raise ValueError(f"pad_mode must be 'zero' or 'hold', got {pad_mode!r}")
    if actions.ndim != 2:
        raise ValueError(f"retime_chunk expects (horizon, action_dim), got {tuple(actions.shape)}")

    horizon = actions.shape[0]
    indices = [0, *keep_indices(labels, low_v, high_v, start=0)]

    new_actions = torch.zeros_like(actions)
    # Default True so anything past the valid region is masked out of the loss.
    new_is_pad = torch.ones_like(is_pad) if is_pad is not None else None

    n = len(indices)
    new_actions[:n] = actions[indices]
    if pad_mode == "hold" and n < horizon:
        new_actions[n:] = new_actions[n - 1]
    if new_is_pad is not None and is_pad is not None:
        new_is_pad[:n] = is_pad[indices]

    return new_actions, new_is_pad


def speedup_factor(labels, low_v: int = LOW_V, high_v: int = HIGH_V) -> float:
    """How much shorter the episode becomes. Reported by the labelling tools."""
    n = len(np.asarray(labels))
    kept = len(episode_keep_indices(labels, low_v, high_v))
    return n / max(kept, 1)
