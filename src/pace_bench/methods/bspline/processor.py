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

from pace_bench.methods.bspline.layout import (
    ActionLayout,
    MatrixArrangement,
    coerce_arrangement,
    coerce_layout,
)
from pace_bench.methods.bspline.spline import (
    DEGREE,
    MAX_ERROR,
    chunk_parameters,
    decode_chunk,
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
        arrangement: MatrixArrangement | None = None,
    ):
        self.splines = splines
        self.episode_starts = episode_starts or {}
        self.relative_knots = relative_knots
        self.degree = degree
        self.arrangement = coerce_arrangement(arrangement)

    @property
    def channels(self) -> int:
        """Emitted width: the arrangement's, or one knot column plus control points."""
        if self.arrangement is not None and self.arrangement.channels is not None:
            return self.arrangement.channels
        return 0 if self.splines is None else self.splines.channels

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
        out = np.empty((batch, self.splines.width, self.channels), dtype=np.float32)
        for i in range(batch):
            matrix = self.splines.parameters(int(episode_index[i]), int(frame_index[i]))
            if self.relative_knots:
                matrix = encode_relative_knots(matrix, self.degree)
            out[i] = matrix if self.arrangement is None else self.arrangement.emit(matrix)

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
        return {
            "relative_knots": self.relative_knots,
            "degree": self.degree,
            "arrangement": None if self.arrangement is None else self.arrangement.name,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The action feature becomes the parameter matrix's channel count."""
        if self.splines is None:
            return features
        action_features = features.get(PipelineFeatureType.ACTION, {})
        for key, feature in list(action_features.items()):
            action_features[key] = PolicyFeature(type=feature.type, shape=(self.channels,))
        return features


@ProcessorStepRegistry.register("bspline_decode")
class BSplineDecodeStep(ProcessorStep):
    """Turn the policy's predicted spline parameters back into executable actions.

    The inverse of :class:`BSplineChunkStep`, and the only reason a B-spline
    checkpoint can be run at all: what the policy emits is a knot column beside
    control points, which no robot can execute. This evaluates that curve and maps it
    back to the dataset's own action space.

    ``num_actions`` is the speed lever, and it is a *decode-time* choice needing no
    retraining -- the paper's ``a_exec(t) = a(nt)``. The curve covers a fixed stretch
    of demonstrated motion; asking for fewer samples covers that stretch in fewer
    executed steps. The realised factor is published per sample as ``bspline_rate``
    (source frames advanced per executed action), because it is a property of the
    predicted span and so varies from chunk to chunk rather than being the constant
    the config asks for.

    Runs *after* the unnormalizer: it needs parameters in their own units, since the
    knot column is a time in source frames and the control points are poses.
    """

    def __init__(
        self,
        num_actions: int = 16,
        degree: int = DEGREE,
        relative_knots: bool = False,
        layout: ActionLayout | None = None,
        arrangement: MatrixArrangement | None = None,
        align: bool = False,
        fps: float = 20.0,
        predict_before_end: float = 0.0,
    ):
        if num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {num_actions}")
        self.num_actions = num_actions
        self.degree = degree
        self.relative_knots = relative_knots
        # Names, not objects, arrive when the step is rebuilt from a checkpoint's
        # policy_postprocessor.json -- get_config serialises them by name.
        self.layout = coerce_layout(layout)
        self.arrangement = coerce_arrangement(arrangement)
        #: Resume each chunk where the previous one left the arm, rather than at the
        #: curve's own beginning. Sequential control only -- see `decode_batch`.
        self.align = bool(align)
        #: Seconds of each curve left unexecuted before the next chunk is requested.
        #: Upstream re-plans while the current curve still has `predict_before_end`
        #: (0.06 s) to run rather than after it is spent, so the arm is never driven
        #: into the tail of a prediction -- the least constrained part of the fit, and
        #: where a chunk padded at the episode edge degenerates outright. 0 executes
        #: the whole curve, which is what a training pipeline wants.
        self.predict_before_end = float(predict_before_end)
        self.fps = float(fps)
        self.reset()

    def reset(self) -> None:
        """Forget the previous chunk. Must be called at every episode boundary.

        An anchor carried across a reset would align the first chunk of a new episode
        to wherever the last one ended, which is a different place entirely.
        """
        #: One anchor per batch row. A vector env is not one stream but `n_envs` of
        #: them, each with its own place on its own curve.
        self._anchors: list[np.ndarray | None] = []

    @property
    def compare_dim(self) -> int | None:
        """Columns the alignment distance uses: everything but the gripper.

        The gripper is the last spline dimension in every layout here. It is
        near-binary, so it barely constrains *where along the path* the arm is while
        its single step would dominate the distance -- upstream excludes it too
        (`consider_gripper_during_align`, default off). None means "all columns", for
        a layout with no gripper to drop.
        """
        return None if self.layout is None else max(self.layout.spline_dim - 1, 1)

    @torch.no_grad()
    def decode_batch(
        self, matrices: torch.Tensor, sequential: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, width, channels)`` of parameters -> ``(B, executed, dim)`` actions.

        ``sequential`` says the rows are *streams*: row i is the same robot, one chunk
        after another, so it carries its own anchor. That is what a vector env is, and
        it is the whole of the difference from a training batch, whose rows are
        unrelated samples that must not be aligned to anything.

        The tensor-level entry point, so the eval loop can decode a predicted chunk
        without building a transition around it. Also returns the realised rate per
        sample -- source frames advanced per executed action, which varies with the
        span the policy predicted rather than being the constant the config asks for.
        """
        # A batch of one is sequential whether the caller says so or not: that is the
        # single-robot control loop, which has nowhere else to be. Anything wider has
        # to declare itself, because a training batch's rows are unrelated samples and
        # aligning those would be meaningless rather than merely slow.
        aligning = self.align and (sequential or matrices.shape[0] == 1)
        if len(self._anchors) != matrices.shape[0]:
            self._anchors = [None] * matrices.shape[0]
        samples_per_row, rates = [], []
        for row, emitted in enumerate(matrices.detach().cpu().numpy().astype(np.float64)):
            matrix = (
                emitted
                if self.arrangement is None or self.layout is None
                else self.arrangement.recover(emitted, self.layout.spline_dim)
            )
            samples = decode_chunk(
                matrix, self.num_actions, degree=self.degree,
                relative_knots=self.relative_knots,
                align_to=self._anchors[row] if aligning else None,
                compare_dim=self.compare_dim,
            )
            samples_per_row.append(samples)
            span = matrix[-(self.degree + 1), 0] - matrix[self.degree, 0]
            rates.append(span / max(self.num_actions - 1, 1))

        # How many of each row's actions actually get executed. Alignment shortens a
        # row by however far along its curve the arm already was, so rows can differ in
        # length; they are cut to the shortest, since the caller executes them in
        # lockstep. `predict_before_end` then holds back the tail, in seconds of
        # demonstrated motion converted through the realised rate.
        keep = min(len(row) for row in samples_per_row)
        if self.predict_before_end > 0 and self.fps > 0:
            per_action_s = max(rates) / self.fps
            if per_action_s > 0:
                keep = max(1, keep - int(np.ceil(self.predict_before_end / per_action_s)))

        decoded = []
        for row, samples in enumerate(samples_per_row):
            samples = samples[:keep]
            if aligning:
                # In the spline's own space, before `from_spline`: a distance taken
                # after the rotation is back in axis-angle is meaningless across the
                # pi wrap, which is the same reason the fit uses 6D at all. The last
                # *executed* sample, not the last decoded one -- the held-back tail is
                # never commanded, so the arm never reaches it.
                self._anchors[row] = samples[-1].copy()
            if self.layout is not None:
                samples = self.layout.from_spline(samples)
            decoded.append(samples)
        actions = torch.from_numpy(np.stack(decoded)).to(
            device=matrices.device, dtype=matrices.dtype
        )
        return actions, torch.tensor(rates, dtype=torch.float32, device=matrices.device)

    def __call__(self, transition):
        new_transition = transition.copy()
        actions = new_transition.get(TransitionKey.ACTION)
        if not isinstance(actions, torch.Tensor):
            return new_transition

        # A postprocessor sees a batch when driven from training code and a single
        # chunk from a control loop; both shapes are the same object.
        unbatched = actions.ndim == 2
        decoded, rates = self.decode_batch(actions[None] if unbatched else actions)

        new_transition[TransitionKey.ACTION] = decoded[0] if unbatched else decoded
        complementary = new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = {
            **complementary,
            "bspline_rate": rates.cpu(),
        }
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "degree": self.degree,
            "relative_knots": self.relative_knots,
            "align": self.align,
            "layout": None if self.layout is None else self.layout.name,
            "arrangement": None if self.arrangement is None else self.arrangement.name,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Back to the dataset's action width, which is what a robot consumes."""
        if self.layout is None or self.layout.raw_dim is None:
            return features
        action_features = features.get(PipelineFeatureType.ACTION, {})
        for key, feature in list(action_features.items()):
            action_features[key] = PolicyFeature(type=feature.type, shape=(self.layout.raw_dim,))
        return features
