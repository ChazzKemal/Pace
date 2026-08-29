"""Turn an entropy trace into per-frame precision labels.

The trace out of :mod:`pace_bench.methods.demospeedup.entropy` is a noisy real
signal; retiming needs a binary one. DemoSpeedup gets there by clustering, not
thresholding, and the reason is temporal: a threshold decides each frame alone and
produces one-frame flickers that the retiming walk then has to step around, whereas
clustering over ``(time, entropy)`` finds *segments* -- runs of frames that are
uniformly loose, or uniformly pinned. That is why time is a clustering feature
rather than just an index, and it is why the recorded labels come in long runs
(13.9 frames on stack_cups) instead of the 1.23 a per-frame rule would give.

Label ``0`` (precision, short stride) is the **low**-entropy end of the trace: the
sampled actions agreed, so the trajectory is pinned and must be followed. Label
``1`` is the high-entropy end, where many actions were acceptable and the frames can
be skipped through.

The pipeline, from ``lingxiao-guo/DemoSpeedup`` @ ``34bd43a``
(``robobase/robobase/utils.py::hdbscan_with_custom_merge``):

1. z-score the trace, replace IsolationForest outliers by interpolation, z-score again;
2. HDBSCAN over the 2-D ``(normalized time, normalized entropy)`` points;
3. split clusters longer than ``max_cluster_size`` so one cluster cannot span the
   whole episode;
4. call each cluster precision or not, and emit ``0`` / ``1``.

Step 4 is the one place this file has to make a decision rather than a copy -- see
:data:`Rule`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

#: Label values. Retiming reads these: precision frames get the short stride.
PRECISION = 0
NON_PRECISION = 1

#: How a cluster's entropy decides its label. The two rules are not a tuning knob;
#: they are a genuine ambiguity in the reference implementation, named so that
#: choosing between them stays a decision instead of drifting.
#:
#: ``"upstream"``
#:     Literally what ``lingxiao-guo/DemoSpeedup`` @ ``34bd43a`` runs::
#:
#:         if np.mean(cluster_points[:, 1] < 1):
#:
#:     The comparison is *inside* the mean, so this is the **fraction of the
#:     cluster's frames below +1σ**, used as a truth value: the cluster is precision
#:     unless every last frame in it sits at or above +1σ. HDBSCAN noise points keep
#:     upstream's ``-1`` and come out non-precision. Read as written, the
#:     parenthesis looks misplaced -- but this is the code that produced the CoRL
#:     2025 results, so it is the default here.
#:
#: ``"mean"``
#:     ``np.mean(cluster_points[:, 1]) < 1`` -- the cluster is precision when its
#:     *mean* entropy is below +1σ, plus the natural companion fix of letting
#:     low-entropy HDBSCAN noise be precision too. This is the reading the lerobot
#:     fork adopted, and it is what produced this project's recorded stack_cups
#:     labels (18.4% non-precision). It marks far more of an episode non-precision
#:     than ``"upstream"`` does, so the two are not interchangeable.
Rule = Literal["upstream", "mean"]


def remove_outliers(values: np.ndarray, contamination: float = 0.1, seed: int | None = 0) -> np.ndarray:
    """Replace IsolationForest outliers with an interpolation of their neighbours.

    Episode boundaries and one-off states produce entropy spikes that would
    otherwise dominate the z-scoring and pull HDBSCAN's clusters apart. Upstream
    replaces rather than drops them, which keeps the trace aligned with the frames.

    Args:
        values: 1-D entropy trace, already z-scored.
        contamination: Fraction of frames IsolationForest should treat as outliers.
        seed: ``random_state`` for the forest. Upstream leaves this unset, so its
            labels differ slightly run to run; pinning it is a deliberate deviation
            in favour of reproducible labelling.

    Returns:
        A cleaned copy of ``values``, same length.
    """
    # Imported here, not at module scope: scikit-learn and hdbscan together cost a
    # couple of seconds to import, and the label constants above are read by the
    # training path, which needs neither.
    from sklearn.ensemble import IsolationForest

    predictions = IsolationForest(contamination=contamination, random_state=seed).fit_predict(
        values.reshape(-1, 1)
    )
    values = values.copy()
    is_outlier = predictions == -1
    if not is_outlier.any():
        return values

    good = np.flatnonzero(~is_outlier)
    if len(good) == 0:
        # Everything is an outlier; there is nothing to interpolate from.
        return values

    for i in np.flatnonzero(is_outlier):
        before = good[good < i]
        after = good[good > i]
        if len(before) and len(after):
            values[i] = (values[before[-1]] + values[after[0]]) / 2
        elif len(before):
            values[i] = values[before[-1]]
        else:
            values[i] = values[after[0]]
    return values


def _split_large_clusters(labels: np.ndarray, max_size: int) -> np.ndarray:
    """Chop any cluster longer than ``max_size`` into consecutive pieces.

    Without this, HDBSCAN happily calls a whole uneventful episode one cluster, and
    a single verdict over 300 frames is no segmentation at all.
    """
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for label in np.unique(labels):
        if label == -1:
            continue
        members = np.flatnonzero(labels == label)
        if len(members) <= max_size:
            continue
        for start in range(0, len(members), max_size):
            labels[members[start : start + max_size]] = next_label
            next_label += 1
    return labels


def segment(
    entropy: np.ndarray,
    *,
    rule: Rule = "upstream",
    min_cluster_size: int = 5,
    max_cluster_size: int = 25,
    contamination: float = 0.1,
    seed: int | None = 0,
) -> np.ndarray:
    """Cluster one episode's entropy trace into ``PRECISION`` / ``NON_PRECISION``.

    Args:
        entropy: 1-D trace, one value per frame, in raw entropy units.
        rule: Which cluster verdict to apply -- see :data:`Rule`.
        min_cluster_size: HDBSCAN's minimum cluster size (upstream: 5).
        max_cluster_size: Longest run one cluster may cover (upstream: 25).
        contamination: Outlier fraction for :func:`remove_outliers` (upstream: 0.1).
        seed: Reproducibility seed for the outlier forest.

    Returns:
        ``int64`` array of the same length, values in ``{0, 1}``.
    """
    if rule not in ("upstream", "mean"):
        raise ValueError(f"rule must be 'upstream' or 'mean', got {rule!r}")
    entropy = np.asarray(entropy, dtype=np.float64)
    if entropy.ndim != 1:
        raise ValueError(f"expected a 1-D entropy trace, got shape {entropy.shape}")

    normalized = _zscore(entropy)
    if normalized is None:
        # A flat trace carries no information about where precision is needed.
        # Upstream would divide by zero here; calling the whole episode precision
        # is the conservative reading -- it retimes at the short stride throughout.
        return np.zeros(len(entropy), dtype=np.int64)

    normalized = remove_outliers(normalized, contamination=contamination, seed=seed)
    normalized = _zscore(normalized)
    if normalized is None:
        return np.zeros(len(entropy), dtype=np.int64)

    # Time is a clustering feature, normalized the same way, so that a cluster is a
    # contiguous stretch of similar entropy rather than a scatter across the episode.
    time = _zscore(np.arange(len(normalized), dtype=np.float64))
    if time is None:  # a 1-frame episode
        return np.zeros(len(entropy), dtype=np.int64)
    points = np.stack((time, normalized), axis=-1)

    clusters = _hdbscan_labels(points, min_cluster_size=min_cluster_size)
    clusters = _split_large_clusters(clusters, max_size=max_cluster_size)

    labels = np.full(len(normalized), NON_PRECISION, dtype=np.int64)
    for label in np.unique(clusters[clusters >= 0]):
        member = clusters == label
        below = points[member, 1] < 1
        precise = bool(np.mean(below)) if rule == "upstream" else bool(points[member, 1].mean() < 1)
        if precise:
            labels[member] = PRECISION

    if rule == "mean":
        # Noise points are single frames HDBSCAN could not place. Upstream condemns
        # them all to non-precision; under this rule a quiet one is judged on its
        # own entropy, like any other frame.
        noise = clusters == -1
        labels[noise & (points[:, 1] < 1)] = PRECISION

    return labels


def _zscore(values: np.ndarray) -> np.ndarray | None:
    """Centre and scale, or ``None`` when there is no spread to scale by."""
    std = float(np.std(values))
    if std < 1e-8:
        return None
    return (values - np.mean(values)) / std


def _hdbscan_labels(points: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN cluster ids, ``-1`` for noise.

    Upstream's own ``hdbscan`` package, not scikit-learn's port of it. The two agree
    on ~99% of same-cluster/different-cluster pairs for entropy-shaped traces, but
    they diverge sharply where a trace has no cluster structure to find -- a
    monotonic ramp, or pure noise -- and there the disagreement reaches the labels
    (52% agreement on a ramp). Since those labels decide how a demonstration is
    retimed, matching the reference exactly is worth the extra dependency.
    """
    import hdbscan  # deferred for import cost -- see remove_outliers

    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(points).labels_
