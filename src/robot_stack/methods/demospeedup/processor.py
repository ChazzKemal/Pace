"""DemoSpeedup retiming as a LeRobot pipeline step.

Unlike PACE, this acts at *training* time and on the *targets*: the observation a
sample carries is untouched, while the action chunk it is regressed against is
subsampled per the frame's label. The policy therefore learns to cover more ground
per step, and at inference it simply runs at the ordinary control rate.

Consolidating on this -- rather than pre-writing a retimed dataset to disk -- keeps
every frame as a chunk start (a disk conversion drops 2-4x of them along with the
frames themselves) and means the labels, not a derived artifact, are the thing that
has to be reproduced.

Finding the label for a sample needs its index *within* its episode. LeRobot's batch
carries ``frame_index``, but the batch-to-transition converter forwards only a fixed
key set that does not include it -- so the step reconstructs it from the global
``index`` and a table of episode start offsets. See :func:`episode_starts_from_metadata`.
"""

from typing import Any

import numpy as np
import torch
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION

from robot_stack.methods.demospeedup.retime import HIGH_V, LOW_V, retime_chunk

ACTION_IS_PAD = f"{ACTION}_is_pad"


def episode_starts_from_metadata(meta) -> dict[int, int]:
    """``{episode_index: global index of its first frame}``.

    Derived from cumulative episode lengths, which is the only place LeRobot records
    the mapping a training sample needs to locate itself within its own episode.
    """
    starts, running = {}, 0
    for episode_index in range(meta.total_episodes):
        starts[episode_index] = running
        running += meta.episodes[episode_index]["length"]
    return starts


@ProcessorStepRegistry.register("demospeedup_retime")
class DemoSpeedupRetimeStep(ProcessorStep):
    """Subsample each sample's action chunk according to its precision labels.

    Input/output: ``transition[ACTION]`` shaped ``(B, horizon, action_dim)``, with
    ``episode_index`` and ``index`` in complementary data. Samples whose episode has
    no labels pass through untouched, so a partially-labelled dataset degrades to
    ordinary training on the unlabelled part instead of failing.
    """

    def __init__(
        self,
        labels: dict[int, np.ndarray] | None = None,
        episode_starts: dict[int, int] | None = None,
        low_v: int = LOW_V,
        high_v: int = HIGH_V,
        pad_mode: str = "zero",
    ):
        """
        Args:
            labels: ``{episode_index: (T,) labels}``, from :mod:`..demospeedup.labels`.
            episode_starts: ``{episode_index: global start index}``. Required to turn
                a sample's global index into its position inside its episode.
            low_v: Stride through precision frames.
            high_v: Stride through non-precision frames.
            pad_mode: ``"zero"`` for loss masked by ``action_is_pad`` (ACT, Diffusion);
                ``"hold"`` for policies regressing the whole chunk (xVLA).
        """
        self.labels = labels or {}
        self.episode_starts = episode_starts or {}
        self.low_v = low_v
        self.high_v = high_v
        self.pad_mode = pad_mode

    def __call__(self, transition):
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        actions = new_transition.get(TransitionKey.ACTION)
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3 or not self.labels:
            return new_transition

        complementary = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        episode_index = complementary.get("episode_index")
        if episode_index is None:
            raise KeyError(
                "DemoSpeedupRetimeStep needs `episode_index` in complementary data to "
                "find each sample's labels; none was present."
            )
        frame_index = self._frame_indices(complementary, episode_index)

        is_pad = complementary.get(ACTION_IS_PAD)
        actions = actions.clone()
        is_pad = is_pad.clone() if is_pad is not None else None
        horizon = actions.shape[1]

        for i in range(actions.shape[0]):
            ep = int(episode_index[i])
            if ep not in self.labels:
                continue
            window = self._label_window(ep, int(frame_index[i]), horizon)
            actions[i], retimed_pad = retime_chunk(
                actions[i],
                window,
                is_pad[i] if is_pad is not None else None,
                self.low_v,
                self.high_v,
                self.pad_mode,
            )
            if is_pad is not None and retimed_pad is not None:
                is_pad[i] = retimed_pad

        new_transition[TransitionKey.ACTION] = actions
        if is_pad is not None:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = {**complementary, ACTION_IS_PAD: is_pad}
        return new_transition

    def _frame_indices(self, complementary: dict, episode_index) -> torch.Tensor:
        """Position of each sample inside its own episode.

        Uses ``frame_index`` when a caller has supplied it, otherwise reconstructs it
        from the global ``index``, which is what actually survives the converter.
        """
        if (frame_index := complementary.get("frame_index")) is not None:
            return frame_index
        index = complementary.get("index")
        if index is None:
            raise KeyError(
                "DemoSpeedupRetimeStep needs `frame_index` or `index` in complementary "
                "data to locate a sample within its episode; neither was present."
            )
        if not self.episode_starts:
            raise ValueError(
                "`index` is a dataset-global frame number, so episode_starts is required "
                "to convert it to a within-episode position. Build it with "
                "episode_starts_from_metadata(dataset.meta)."
            )
        starts = torch.tensor([self.episode_starts[int(e)] for e in episode_index], device=index.device)
        return index - starts

    def _label_window(self, episode: int, frame: int, horizon: int) -> np.ndarray:
        """The `horizon` labels this chunk spans, extended past the episode end.

        A chunk that runs off the end of an episode is padded with the final label
        rather than with zeros: zero means *precision*, so padding with it would brake
        the tail of every episode-ending chunk for no reason.
        """
        labels = self.labels[episode]
        window = labels[frame : frame + horizon]
        if len(window) < horizon:
            fill = window[-1] if len(window) else 0
            window = np.pad(window, (0, horizon - len(window)), constant_values=fill)
        return window

    def get_config(self) -> dict[str, Any]:
        """Labels are data, not configuration -- only the retiming knobs serialize."""
        return {"low_v": self.low_v, "high_v": self.high_v, "pad_mode": self.pad_mode}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Unchanged: the chunk keeps its length and the action its dimension."""
        return features
