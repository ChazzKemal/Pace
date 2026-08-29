"""How tightly determined the next action is, one demonstration frame at a time.

DemoSpeedup's premise is that a demonstration is only worth replaying slowly where
the trajectory actually has to be followed, and that a policy trained on the
demonstrations already knows where that is: ask it for several action chunks from
the same observation and see how much they disagree.

Read the direction carefully, because it is the opposite of the obvious one.
Disagreement does *not* mean "hard". It means **many different actions are
acceptable here** -- crossing open space, where any of a dozen paths reaches the
same place -- so the recorded trajectory carries little information and can be
skipped through. Agreement means the action is **pinned**: on the approach, the
grasp, the insertion there is one way to move, every sample finds it, and the
demonstration must be followed closely.

So low entropy is a *precision* frame (short stride) and high entropy is
*non-precision* (long stride). Measured on the real stack_cups proxy, precision
frames average 0.0096 and non-precision frames 0.0263.

This module is only the measurement. Turning an entropy trace into precision labels
is :mod:`pace_bench.methods.demospeedup.segment`; drawing the samples is
:mod:`pace_bench.methods.demospeedup.sampler`. Keeping them apart is what lets a
labelling run be re-segmented without re-running a policy over the dataset.

Reference implementation: ``lingxiao-guo/DemoSpeedup`` @ ``34bd43a``,
``robobase/robobase/utils.py`` (``gaussian_kernel``, ``KDE.kde_entropy``). Upstream
wraps this in a ``KDE`` class carrying a bandwidth-estimation path that its own
``kde_entropy`` overwrites with ``bandwidth = 1`` on the next line, and returns the
highest-density sample alongside the entropy -- that sample is upstream's *teacher
action*, which retiming here never uses because it subsamples the recorded
demonstration rather than a policy rollout. Both are dropped; the arithmetic that
remains is upstream's.
"""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_kernel(samples: Tensor, bandwidth: float = 1.0) -> Tensor:
    """Pairwise Gaussian kernel over a set of action samples.

    Args:
        samples: ``(batch, num_samples, dim)``.
        bandwidth: Kernel width. Upstream fixes this at 1.0 and the actions it is
            applied to are normalized, so the default is the reference value.

    Returns:
        ``(batch, num_samples, num_samples)`` kernel matrix.
    """
    if samples.ndim != 3:
        raise ValueError(f"expected (batch, num_samples, dim), got {tuple(samples.shape)}")
    difference = samples.unsqueeze(2) - samples.unsqueeze(1)
    squared_distance = torch.sum(difference**2, dim=-1)
    return torch.exp(-squared_distance / (2 * bandwidth**2))


def kde_entropy(samples: Tensor, bandwidth: float = 1.0) -> Tensor:
    """Entropy of the sampled action distribution, per batch element.

    The density at each sample is the mean kernel value to every sample (itself
    included), and the entropy is the negative mean log density. High entropy means
    the samples are spread out -- many actions are acceptable here, so the exact
    trajectory does not matter and DemoSpeedup is free to skip through it. Low
    entropy means they agree on one action, which is what *precision* means.

    Args:
        samples: ``(batch, num_samples, dim)`` actions drawn for one observation.
        bandwidth: Passed to :func:`gaussian_kernel`.

    Returns:
        ``(batch,)`` entropies.
    """
    if samples.shape[1] < 2:
        raise ValueError(
            f"entropy needs at least 2 samples to have any spread, got {samples.shape[1]}"
        )
    kernel = gaussian_kernel(samples, bandwidth)
    density = kernel.sum(dim=2) / samples.shape[1]
    # The 1e-8 is upstream's, and it matters: an isolated sample in a wide-open
    # region can drive the density to zero, and log(0) would poison the whole trace.
    return -torch.log(density + 1e-8).mean(dim=1)
