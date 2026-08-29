"""Loading per-frame precision labels, from either place they are stored.

Labelling (DemoSpeedup stage 2) is a one-off offline pass: a proxy policy is sampled
repeatedly at each frame, the entropy of its action distribution is measured, and
frames are split into *precision* (0) and *not* (1). Retiming then reads those labels
every training step, so they have to be loadable cheaply and identically on both
sides of the stack.

Two formats exist because the two halves grew separately:

* **parquet sidecar** -- ``meta/demospeedup/labels.parquet`` beside a LeRobot dataset,
  with ``meta/demospeedup.json`` recording how it was produced. Used on the real
  robot. Self-describing and travels with the dataset, so this is the preferred form.
* **directory of .npy** -- ``episode_<i>.npy`` per episode. Used in sim. Carries no
  provenance, so a labels directory cannot tell you which proxy produced it.

:func:`load_labels` accepts either and returns the same thing, so nothing downstream
has to care.
"""

import json
from pathlib import Path

import numpy as np

SIDECAR_LABELS = Path("meta") / "demospeedup" / "labels.parquet"
SIDECAR_CONFIG = Path("meta") / "demospeedup.json"


def load_npy_dir(path: Path) -> dict[int, np.ndarray]:
    """``episode_<i>.npy`` files -> ``{episode_index: labels}``."""
    labels = {}
    for f in sorted(path.glob("episode_*.npy")):
        labels[int(f.stem.split("_")[1])] = np.load(f).astype(np.int64)
    if not labels:
        raise FileNotFoundError(f"no episode_*.npy files in {path}")
    return labels


def load_sidecar(dataset_root: Path) -> tuple[dict[int, np.ndarray], dict]:
    """Parquet sidecar -> ``({episode_index: labels}, provenance)``."""
    import pandas as pd  # noqa: PLC0415  (heavy; only needed for this format)

    labels_path = dataset_root / SIDECAR_LABELS
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found -- label this dataset first")
    frame = pd.read_parquet(labels_path)

    by_episode = {}
    for episode_index, group in frame.groupby("episode_index", sort=True):
        ordered = group.sort_values("frame_index")
        by_episode[int(episode_index)] = ordered["label"].to_numpy().astype(np.int64)

    config_path = dataset_root / SIDECAR_CONFIG
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return by_episode, config


def load_labels(path: str | Path) -> tuple[dict[int, np.ndarray], dict]:
    """Load labels from a dataset root (sidecar) or a directory of .npy files.

    Which one is decided by what is actually there, not by a flag: a caller that has
    to say *how* its labels are stored is a caller that can get it wrong.
    """
    path = Path(path)
    if (path / SIDECAR_LABELS).exists():
        return load_sidecar(path)
    if path.is_dir() and any(path.glob("episode_*.npy")):
        return load_npy_dir(path), {}
    raise FileNotFoundError(
        f"{path} holds neither a DemoSpeedup sidecar ({SIDECAR_LABELS}) nor episode_*.npy files"
    )


def describe(labels: dict[int, np.ndarray], config: dict | None = None) -> str:
    """One line for the training log, so a run records what it was retimed against."""
    total = sum(len(v) for v in labels.values())
    fast = sum(int((v == 1).sum()) for v in labels.values())
    provenance = ""
    if config:
        proxy = config.get("policy_path") or config.get("proxy_path")
        if proxy:
            provenance = f" | proxy={Path(str(proxy)).name}"
        if "segmenter" in config:
            provenance += f" segmenter={config['segmenter']}"
    pct = 100 * fast / total if total else 0.0
    return f"{len(labels)} episodes, {total} frames, {pct:.1f}% non-precision{provenance}"
