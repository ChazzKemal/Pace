"""B-spline retargeting as a LeRobot pipeline step.

Like DemoSpeedup this acts at training time on the *targets*, but it goes further:
it changes the action space. The observation is untouched; the action chunk a sample
is regressed against is replaced by a B-spline parameter matrix -- a knot column
beside control points -- of shape ``(chunk_size + 2 * degree, 1 + spline_dim)``.

**What the step holds is the fitted spline, not a label.** One episode's fit is a
knot vector and a coefficient array (about 1 MB for all 45 pickplace episodes
together); the parameter matrix a given sample needs is a *slice* of that, built on
demand in microseconds. Materialising a matrix per frame instead would be 36 MB for
the same dataset and would recompute nothing faster.

The fits themselves happen once, when the step is built, and cannot sensibly be made
lazy. Training samples are drawn at random, so a batch of 32 touches ~32 different
episodes; refitting on access at ~1.4 s per episode would cost ~45 s per batch. One
startup pass is 3 s for LIBERO-10 and 50 s for pickplace, against training runs of
hours.
"""

import logging
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION

from pace_bench.methods.bspline.layout import ActionLayout
from pace_bench.methods.bspline.spline import (
    DEGREE,
    MAX_ERROR,
    chunk_parameters,
    encode_relative_knots,
    fit_episode,
)

ACTION_IS_PAD = f"{ACTION}_is_pad"
logger = logging.getLogger(__name__)


class EpisodeSplines:
    """Every episode's fit, plus the frame -> chunk map that locates a sample in it.

    Built once. Holds ``chunks`` per episode as a stacked array rather than the raw
    spline, because a chunk is a *window* of the knot vector -- slicing it per sample
    would repeat the same padding work on every access -- while the chunks themselves
    are what a sample actually needs. ``frame_to_chunk`` is upstream's
    ``timestep_to_chunk``, clamped to the episode.
    """

    def __init__(
        self,
        episode_actions: dict[int, np.ndarray],
        layout: ActionLayout,
        chunk_size: int,
        degree: int = DEGREE,
        max_error: float = MAX_ERROR,
    ):
        self.chunk_size = chunk_size
        self.degree = degree
        self.width = chunk_size + 2 * degree
        self.channels = 1 + layout.spline_dim
        self.chunks: dict[int, np.ndarray] = {}
        self.frame_to_chunk: dict[int, np.ndarray] = {}

        missed = []
        for episode, actions in sorted(episode_actions.items()):
            spline, converged = fit_episode(
                layout.to_spline(actions), max_error=max_error, degree=degree
            )
            if not converged:
                missed.append(episode)
            chunks = chunk_parameters(spline, chunk_size, degree=degree, stride=1)
            self.chunks[episode] = np.stack(chunks).astype(np.float32)
            self.frame_to_chunk[episode] = self._assign(chunks, len(actions), degree)
        if missed:
            # Loud, because a missed tolerance is the one failure with no other
            # symptom: the fit still returns a spline, just a worse one.
            logger.warning(
                "B-spline: %d of %d episodes did not reach max_error=%s (%s%s)",
                len(missed), len(episode_actions), max_error,
                missed[:8], " ..." if len(missed) > 8 else "",
            )

    @staticmethod
    def _assign(chunks: list[np.ndarray], length: int, degree: int) -> np.ndarray:
        """Which chunk covers each frame. See `spline.assign_chunks_to_frames`."""
        out = np.zeros(length, dtype=np.int64)
        frame = 0
        for index, chunk in enumerate(chunks):
            while frame < length and frame <= chunk[degree, 0]:
                out[frame] = index
                frame += 1
        out[frame:] = len(chunks) - 1
        return out

    def parameters(self, episode: int, frame: int) -> np.ndarray:
        """The parameter matrix for one sample: its chunk, knots shifted to it."""
        matrix = self.chunks[episode][self.frame_to_chunk[episode][frame]].copy()
        matrix[:, 0] -= frame
        return matrix

    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.chunks.values()) + sum(
            f.nbytes for f in self.frame_to_chunk.values()
        )


@ProcessorStepRegistry.register("bspline_chunk")
class BSplineChunkStep(ProcessorStep):
    """Replace each sample's action chunk with its B-spline parameter matrix.

    Input: ``transition[ACTION]`` shaped ``(B, any, raw_dim)`` -- the loader's own
    action window, which is ignored. Output: ``(B, chunk + 2*degree, 1 + spline_dim)``.

    Runs *before* the normalizer, like the DemoSpeedup step: the fit is defined on raw
    actions, and the parameters it produces are what the normalizer should then scale.
    """

    def __init__(
        self,
        splines: EpisodeSplines | None = None,
        episode_starts: dict[int, int] | None = None,
        relative_knots: bool = False,
        degree: int = DEGREE,
    ):
        self.splines = splines
        self.episode_starts = episode_starts or {}
        self.relative_knots = relative_knots
        self.degree = degree

    def __call__(self, transition):
        new_transition = transition.copy()
        actions = new_transition.get(TransitionKey.ACTION)
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3 or self.splines is None:
            return new_transition

        complementary = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        episode_index = complementary.get("episode_index")
        if episode_index is None:
            raise KeyError(
                "BSplineChunkStep needs `episode_index` in complementary data to find "
                "each sample's episode fit; none was present."
            )
        frame_index = self._frame_indices(complementary, episode_index)

        batch = actions.shape[0]
        out = np.empty((batch, self.splines.width, self.splines.channels), dtype=np.float32)
        for i in range(batch):
            matrix = self.splines.parameters(int(episode_index[i]), int(frame_index[i]))
            out[i] = encode_relative_knots(matrix, self.degree) if self.relative_knots else matrix

        new_transition[TransitionKey.ACTION] = torch.from_numpy(out).to(
            device=actions.device, dtype=actions.dtype
        )
        # Every slot of a B-spline chunk is a real parameter: the representation is
        # fixed-size by construction, and a short tail is filled by repeating the last
        # knot and control point rather than by padding. So nothing is masked out.
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = {
            **complementary,
            ACTION_IS_PAD: torch.zeros(
                (batch, self.splines.width), dtype=torch.bool, device=actions.device
            ),
        }
        return new_transition

    def _frame_indices(self, complementary: dict, episode_index) -> torch.Tensor:
        """Position of each sample inside its own episode. See the DemoSpeedup step."""
        if (frame_index := complementary.get("frame_index")) is not None:
            return frame_index
        index = complementary.get("index")
        if index is None:
            raise KeyError(
                "BSplineChunkStep needs `frame_index` or `index` in complementary data "
                "to locate a sample within its episode; neither was present."
            )
        starts = torch.tensor(
            [self.episode_starts[int(e)] for e in episode_index], device=index.device
        )
        return index - starts

    def get_config(self) -> dict[str, Any]:
        """The fits are data, not configuration -- only the knobs serialize."""
        return {"relative_knots": self.relative_knots, "degree": self.degree}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The action feature becomes the parameter matrix's channel count."""
        if self.splines is None:
            return features
        action_features = features.get(PipelineFeatureType.ACTION, {})
        for key, feature in list(action_features.items()):
            action_features[key] = PolicyFeature(
                type=feature.type, shape=(self.splines.channels,)
            )
        return features
