"""Reproduce the frame selection of the real retimed dataset.

Upstream parity shows the walk is right. This shows the walk, driven by the labels
that were actually computed for the UR10e demonstrations, selects exactly the frames
the recorded accelerated dataset contains -- which is what justifies retiring the
disk converter that produced it.

Skips when the datasets are not on this machine.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from robot_stack.methods.demospeedup import episode_keep_indices, load_labels

SOURCE = Path("/home/batur/Coding/data/merged_act_finetune_20260528")
RETIMED = Path("/home/batur/Coding/data/merged_speedup_20260528")

pytestmark = pytest.mark.skipif(
    not (SOURCE.is_dir() and RETIMED.is_dir()),
    reason="the recorded UR10e datasets are not on this machine",
)


def episode_lengths(root: Path) -> dict[int, int]:
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    return dict(zip(frames["episode_index"].astype(int), frames["length"].astype(int), strict=True))


@pytest.fixture(scope="module")
def recorded():
    provenance = json.loads((RETIMED / "meta" / "demospeedup_source.json").read_text())
    labels, _ = load_labels(SOURCE)
    return provenance, labels, episode_lengths(RETIMED)


def test_every_episode_keeps_the_same_frames(recorded):
    provenance, labels, lengths = recorded
    low_v, high_v = int(provenance["low_v"]), int(provenance["high_v"])
    keep_last = bool(provenance.get("keep_last", True))

    for episode, recorded_length in sorted(lengths.items()):
        predicted = episode_keep_indices(labels[episode], low_v, high_v, keep_last=keep_last)
        assert len(predicted) == recorded_length, (
            f"episode {episode}: this implementation keeps {len(predicted)} frames, "
            f"the recorded dataset has {recorded_length}"
        )


def test_totals_match_the_recorded_provenance(recorded):
    """The converter wrote its own totals; they are an independent second opinion."""
    provenance, labels, lengths = recorded
    assert sum(lengths.values()) == provenance["kept_frames"]
    assert sum(len(v) for v in labels.values()) == provenance["source_frames"]
    assert len(lengths) == provenance["n_episodes"]
