"""DemoSpeedup retiming as a LeRobot pipeline step.

Unlike PACE, this acts at *training* time and on the *targets*: the observation a
sample carries is untouched, while the action chunk it is regressed against is
replaced by a retimed one. The policy therefore learns to cover more ground per
step, and at inference it simply runs at the ordinary control rate.

The step treats the episode as the object it is. LeRobot's loader serves a fixed
action window per sample, but the stride walk needs the episode *tail* -- upstream
walks the whole remainder and truncates to the chunk, which is what keeps every
chunk slot a real waypoint mid-episode. So the step preloads the full action table
(~8 MB for LIBERO-10; the same order of size as the labels it already holds) and at
batch time substitutes each sample's action chunk with a walk over its episode's
tail. The loader's own action window is ignored.

Because the substituted actions come from the raw table, this step must run
*before* the normalization step, not after it (see ``run_train``); index selection
commutes with per-dim affine normalization, so the semantics match upstream's
normalize-then-retime either way.

Finding the tail for a sample needs its index *within* its episode. LeRobot's batch
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

from pace_bench.methods.demospeedup.retime import HIGH_V, LOW_V, retime_tail

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
    """Replace each sample's action chunk with a walk over its episode tail.

    Input/output: ``transition[ACTION]`` shaped ``(B, chunk_len, action_dim)`` in the
    raw action space, with ``episode_index`` and ``index`` in complementary data.
    Samples whose episode has no labels pass through untouched, so a
    partially-labelled dataset degrades to ordinary training on the unlabelled part
    instead of failing.
    """

    def __init__(
        self,
        labels: dict[int, np.ndarray] | None = None,
        episode_actions: dict[int, np.ndarray] | None = None,
        episode_starts: dict[int, int] | None = None,
        low_v: int = LOW_V,
        high_v: int = HIGH_V,
        pad_mode: str = "zero",
        out_len: int | None = None,
    ):
        """
        Args:
            labels: ``{episode_index: (T,) labels}``, from :mod:`..demospeedup.labels`.
            episode_actions: ``{episode_index: (T, action_dim) raw actions}`` -- the
                episode's full action trajectory, aligned frame-for-frame with the
                labels. Required for every labelled episode.
            episode_starts: ``{episode_index: global start index}``. Required to turn
                a sample's global index into its position inside its episode.
            low_v: Stride through precision frames.
            high_v: Stride through non-precision frames.
            pad_mode: ``"zero"`` only when the policy's loss is masked by
                ``action_is_pad`` (ACT); ``"hold"`` for unmasked chunk losses (xVLA,
                Diffusion under its defaults). Reached only at episode ends.
            out_len: The chunk length the POLICY trains -- after halving, smaller
                than the loader's window. The step must emit exactly this length:
                xVLA truncates over-length action inputs, but ACT's VAE encoder
                consumes the sequence at exactly chunk_size and a mismatch is a
                shape error. ``None`` keeps the incoming window length.
        """
        self.labels = labels or {}
        self.episode_actions = episode_actions or {}
        self.episode_starts = episode_starts or {}
        self.low_v = low_v
        self.high_v = high_v
        self.pad_mode = pad_mode
        self.out_len = out_len

        # A labelled episode without its actions -- or with actions of a different
        # length -- cannot be retimed and must not silently train as a baseline.
        # Misalignment would retime against the wrong frames and still train, so it
        # is a construction-time error, not a per-batch fallback.
        for episode, episode_labels in self.labels.items():
            actions = self.episode_actions.get(episode)
            if actions is None:
                raise ValueError(
                    f"episode {episode} has labels but no actions; pass episode_actions "
                    "covering every labelled episode."
                )
            if len(actions) != len(episode_labels):
                raise ValueError(
                    f"episode {episode}: {len(episode_labels)} labels vs {len(actions)} actions -- "
                    "the label files do not match this dataset."
                )

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
        chunk_len = min(self.out_len or actions.shape[1], actions.shape[1])
        # Truncating to the trained chunk is correct for pass-through rows too: a
        # chunk-`out_len` policy fed from a chunk-`out_len` window would have seen
        # exactly these first `out_len` actions.
        actions = actions[:, :chunk_len].clone()
        # Rows we substitute get a freshly constructed mask; rows we pass through
        # keep whatever the dataset said (all-False when it said nothing).
        is_pad = (
            is_pad[:, :chunk_len].clone()
            if is_pad is not None
            else torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
        )

        for i in range(actions.shape[0]):
            ep = int(episode_index[i])
            if ep not in self.labels:
                continue
            frame = int(frame_index[i])
            tail_actions = torch.from_numpy(self.episode_actions[ep][frame:]).to(
                device=actions.device, dtype=actions.dtype
            )
            chunk, chunk_pad = retime_tail(
                tail_actions,
                self.labels[ep][frame:],
                chunk_len,
                self.low_v,
                self.high_v,
                self.pad_mode,
            )
            actions[i] = chunk
            is_pad[i] = chunk_pad

        new_transition[TransitionKey.ACTION] = actions
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

    def get_config(self) -> dict[str, Any]:
        """Labels and actions are data, not configuration -- only the knobs serialize."""
        return {"low_v": self.low_v, "high_v": self.high_v, "pad_mode": self.pad_mode, "out_len": self.out_len}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Unchanged: the chunk keeps its length and the action its dimension."""
        return features
