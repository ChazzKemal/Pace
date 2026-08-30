"""One config choice selects the method. ``--method.type=pace`` and nothing else.

The problem this replaces: in the fork, choosing a method meant setting a handful of
loose flags that had to agree with each other -- ``use_speedup`` plus a labels path
plus two stride numbers, or a dozen speed knobs threaded through
``env.extra_gym_kwargs``. Nothing stopped you from passing PACE's knobs with
DemoSpeedup selected, or from setting ``speedup_low_v`` with speedup off. The flags
were flat, so the config could express states the code could not honour.

Here each method is a class registered under a name. Selecting ``pace`` makes PACE's
fields available and no others; selecting ``none`` accepts no knobs at all. States
that were previously expressible-but-meaningless are now unrepresentable.

A method answers one question -- *which pipeline steps do you contribute?* -- so
training and inference read the same object, and adding a method means adding a
class rather than threading flags through the call graph.

    --method.type=none                      baseline, contributes nothing
    --method.type=pace --method.max_speed=1.5 --method.action_stride=2
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field, fields, replace

import draccus
import numpy as np
import torch
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION

from pace_bench.methods.demospeedup.labels import describe, load_labels
from pace_bench.methods.demospeedup.processor import DemoSpeedupRetimeStep, episode_starts_from_metadata
from pace_bench.methods.demospeedup.retime import HIGH_V, LOW_V
from pace_bench.methods.pace.processor import PaceSpeedStep
from pace_bench.methods.pace.speed import PaceConfig


@dataclass
class MethodConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore[misc]
    """A speedup method, as a config choice.

    Subclasses declare their own knobs and say which pipeline steps they contribute.
    The two step hooks are separate because the methods in scope act at different
    points: DemoSpeedup retimes *training* targets (a preprocessor concern), while
    PACE and B-spline transform actions on the way *out* (a postprocessor one).
    """

    @property
    def type(self) -> str:
        """The registered name, e.g. ``"pace"``. What ``--method.type`` selects."""
        return self.get_choice_name(self.__class__)

    def preprocessor_steps(self, dataset=None) -> list[ProcessorStep]:
        """Steps to insert before the policy. Empty for methods that act on output.

        Args:
            dataset: The training ``LeRobotDataset``, when the caller has one. Methods
                that need the episodes themselves (DemoSpeedup preloads each episode's
                action trajectory and start offset) read it; the rest ignore it, so a
                runner with no dataset -- an eval loop -- can still build its steps.
        """
        return []

    def postprocessor_steps(self) -> list[ProcessorStep]:
        """Steps to insert after the policy."""
        return []

    def adjust_policy(self, policy_cfg) -> None:
        """Mutate the policy config before the datasets and the policy are built.

        Exists for methods that change the geometry the policy trains (DemoSpeedup's
        chunk halving). Running before the datasets is deliberate: the loader derives
        its action window from this config, so a method that changes the chunk here
        changes what the loader fetches.

        Called for every method, so it must stay a no-op by default -- PACE and the
        baseline contribute nothing here and must be unaffected by it.
        """

    def adjust_dataset(self, dataset) -> None:
        """Mutate the dataset's *metadata* after it is built, before the policy is.

        Exists because `make_policy` derives `output_features` from `ds_meta.features`
        and its normalization buffers from `ds_meta.stats`, overwriting whatever
        `adjust_policy` put on the policy config. A method that changes the action
        space -- B-spline replaces actions with spline parameters -- can therefore
        only be seen by the policy through the metadata.

        No-op by default: PACE, DemoSpeedup and the baseline all keep the dataset's
        own action space.
        """

    def adjust_processors(self, preprocessor, postprocessor) -> None:
        """Correct the built pipelines, after the factory has made them.

        Exists for the case `adjust_dataset` cannot reach: a policy loaded from a
        checkpoint (`--policy.path`) gets its normalization buffers from that
        checkpoint's saved processor, not from the dataset metadata. A method that
        changed the action space has to overwrite them or the new columns are scaled
        by statistics belonging to a different quantity.

        No-op by default.
        """


@MethodConfig.register_subclass("none")
@dataclass
class NoMethod(MethodConfig):
    """The baseline: the stock policy, untouched.

    Deliberately has no fields. It exists so that "no method" is a choice like any
    other rather than a special case in the calling code -- every runner builds its
    pipeline the same way, and the baseline is the one that contributes no steps.
    """


@MethodConfig.register_subclass("pace")
@dataclass
class PaceMethod(MethodConfig):
    """Eval-time speed modulation. See :mod:`pace_bench.methods.pace.speed`.

    Fields mirror :class:`PaceConfig` one-for-one so the CLI surface and the
    algorithm cannot drift apart; :meth:`to_pace_config` is checked against that at
    import time by the test suite.
    """

    max_speed: float = 1.0
    # None means "half of max_speed", the convention every recorded experiment used.
    # Spelling it as None rather than baking in 0.5 keeps the coupling visible when a
    # caller overrides max_speed alone.
    min_speed: float | None = None
    clamp_deg: float = 5.0

    action_stride: int = 1
    adaptive_stride: bool = False

    n_lookahead: int = 0
    lookahead_agg: str = "min"
    lookahead_target: str = "all"

    enable_angle: bool = True
    enable_ori: bool = True
    # The policy's own default is True, but every recorded ablation ran with the axis
    # channel off. Defaulting to False here matches the experiments; pass
    # --method.enable_ori_axis=true to restore it.
    enable_ori_axis: bool = False

    speed_quantize: bool = False
    quantize_angle_thr: float = 22.5

    def to_pace_config(self) -> PaceConfig:
        resolved = {f.name: getattr(self, f.name) for f in fields(self)}
        resolved["min_speed"] = self.max_speed / 2 if self.min_speed is None else self.min_speed
        return PaceConfig(**resolved)

    def postprocessor_steps(self) -> list[ProcessorStep]:
        # `n_action_steps` is not a PACE knob -- it is how much of a chunk any policy
        # executes before re-querying -- so the runner sets it on the built step
        # rather than it appearing among the method's own fields.
        return [PaceSpeedStep(self.to_pace_config())]


@dataclass
class MethodPipelineConfig:
    """Mixin holding the method choice, for runners to embed.

    Defaults to :class:`NoMethod`, so a config that never mentions a method behaves
    exactly as it did before methods existed.
    """

    method: MethodConfig = field(default_factory=NoMethod)


@dataclass(frozen=True)
class ChunkFields:
    """Which config fields hold a policy family's chunk geometry.

    An explicit registry, keyed on the policy config's registered ``type`` --
    never inferred by probing attribute names, which treats an interface as a
    coincidence of naming. A policy family outside the registry is a loud error,
    not a silent no-op.
    """

    chunk: str  # length of the action sequence the policy trains/predicts
    executed: str  # steps executed per query before re-planning


POLICY_CHUNK_FIELDS: dict[str, ChunkFields] = {
    "act": ChunkFields(chunk="chunk_size", executed="n_action_steps"),
    "diffusion": ChunkFields(chunk="horizon", executed="n_action_steps"),
    "xvla": ChunkFields(chunk="chunk_size", executed="n_action_steps"),
}


@MethodConfig.register_subclass("demospeedup")
@dataclass
class DemoSpeedupMethod(MethodConfig):
    """Entropy-guided demonstration retiming. Acts at training time, on the targets.

    Unlike PACE this contributes a *pre*processor step: the observation is unchanged
    and the action chunk it is regressed against is subsampled, so the policy learns
    to cover more ground per step.
    """

    # A dataset root holding meta/demospeedup/labels.parquet, or a directory of
    # episode_<i>.npy. Which one is decided by what is there, not by a flag.
    labels_path: str | None = None
    low_v: int = LOW_V
    high_v: int = HIGH_V
    # "zero" only for policies whose loss is masked by action_is_pad. That is ACT
    # unconditionally (modeling_act masks its L1), and Diffusion ONLY with
    # do_mask_loss_for_padding=true -- its default is false, and an unmasked zero
    # tail is trained as a real target. "hold" (repeat the last kept waypoint) for
    # any unmasked loss: xVLA always, Diffusion under its defaults. In an absolute
    # action space a trained zero is a command to the world origin.
    pad_mode: str = "zero"
    # Halve the policy's chunk: 15 retimed waypoints span the motion of the original
    # 30. Mirrors upstream's bigym halving and the fork's `speedup_halve_chunk`.
    halve_chunk: bool = True
    # The chunk geometry as it stood BEFORE halving. Recorded automatically on the
    # first apply and serialized with the rest of the config, so a resumed run --
    # which parses its own checkpoint's already-halved policy config -- halves from
    # the original rather than from the halved value. Not meant to be set by hand.
    source_chunk: int | None = None
    source_executed: int | None = None

    def __post_init__(self):
        if self.pad_mode not in ("zero", "hold"):
            raise ValueError(f"pad_mode must be 'zero' or 'hold', got {self.pad_mode!r}")
        if self.low_v < 1 or self.high_v < 1:
            raise ValueError(f"strides must be >= 1, got low_v={self.low_v}, high_v={self.high_v}")

    def preprocessor_steps(self, dataset=None) -> list[ProcessorStep]:
        """Build the retiming step, loading labels if a path was given.

        Without a path the step is constructed empty and passes every sample through.
        That keeps `--method.type=demospeedup` selectable while a labelling run is
        still pending, rather than making an unlabelled dataset a hard error.

        With labels, the ``dataset`` is required: the step preloads each episode's
        full raw action trajectory from it (the walk consumes episode tails, exactly
        as upstream does) plus the start offsets that locate a sample within its
        episode. The step validates label/action lengths per episode, so a label set
        from a different dataset fails loudly at construction.
        """
        labels, config = ({}, {})
        if self.labels_path:
            labels, config = load_labels(self.labels_path)
            logging.info("DemoSpeedup labels: %s", describe(labels, config))
        if not labels:
            return [DemoSpeedupRetimeStep(low_v=self.low_v, high_v=self.high_v, pad_mode=self.pad_mode)]
        if dataset is None:
            raise ValueError(
                "DemoSpeedup needs the training dataset to preload episode action "
                "trajectories, but preprocessor_steps() was called without it."
            )
        episode_starts = episode_starts_from_metadata(dataset.meta)
        action_table = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
        episode_actions = {
            episode: action_table[start : start + dataset.meta.episodes[episode]["length"]]
            for episode, start in episode_starts.items()
        }
        return [
            DemoSpeedupRetimeStep(
                labels=labels,
                episode_actions=episode_actions,
                episode_starts=episode_starts,
                low_v=self.low_v,
                high_v=self.high_v,
                pad_mode=self.pad_mode,
                out_len=getattr(self, "_trained_chunk", None),
            )
        ]
    def adjust_policy(self, policy_cfg) -> None:
        """Halve the chunk the policy trains; also guard the pad_mode/loss pairing.

        Field names come from :data:`POLICY_CHUNK_FIELDS`, keyed on the policy's
        registered ``type``. Running before the datasets are built means the loader's
        action window is the halved chunk, so it stops fetching the half that the
        retiming step would discard. The retimed output is the same either way --
        the step substitutes chunks from its own preloaded episode table, and the
        first half of a full window is exactly a half-length window.
        """
        if self.pad_mode == "zero" and getattr(policy_cfg, "do_mask_loss_for_padding", True) is False:
            raise ValueError(
                "pad_mode='zero' requires the policy to mask padded actions out of its "
                "loss, but this policy sets do_mask_loss_for_padding=False (Diffusion's "
                "default): the zero tail would be trained as a real target. Use "
                "--method.pad_mode=hold or --policy.do_mask_loss_for_padding=true."
            )
        if not self.halve_chunk:
            return
        policy_type = policy_cfg.type
        fields = POLICY_CHUNK_FIELDS.get(policy_type)
        if fields is None:
            raise ValueError(
                f"DemoSpeedup does not know the chunk fields of policy type {policy_type!r}; "
                f"add it to POLICY_CHUNK_FIELDS (known: {sorted(POLICY_CHUNK_FIELDS)})."
            )
        # Halve from the geometry recorded on the first apply, never from whatever
        # the config happens to hold now. A resumed run is parsed from its own
        # checkpoint's train_config.json, whose policy chunk is ALREADY halved and
        # whose method is still demospeedup -- halving that again would quarter it
        # (30 -> 15 -> 7). Computing the target from a recorded invariant makes
        # re-applying a no-op by construction rather than by a guard that has to
        # guess whether it has run before.
        chunk = self.source_chunk if self.source_chunk is not None else getattr(policy_cfg, fields.chunk)
        executed = (
            self.source_executed
            if self.source_executed is not None
            else getattr(policy_cfg, fields.executed)
        )
        self.source_chunk, self.source_executed = chunk, executed
        setattr(policy_cfg, fields.chunk, chunk // 2)
        setattr(policy_cfg, fields.executed, executed // 2)
        # The retime step must emit chunks of exactly the trained length: ACT
        # consumes its action input at chunk_size, no truncation.
        self._trained_chunk = chunk // 2
        logging.info(
            "DemoSpeedup: halved %s to %d, %s to %d (policy type %r)",
            fields.chunk, chunk // 2, fields.executed, executed // 2, policy_type,
        )


@MethodConfig.register_subclass("bspline")
@dataclass
class BSplineMethod(MethodConfig):
    """B-spline action representation (Han et al., arXiv:2607.09648).

    The only method here that changes the action *space*. Instead of regressing a
    dense sequence of actions the policy regresses one B-spline's parameters -- a
    knot column beside control points -- and the executable actions are that curve,
    sampled. Speed then comes from sampling the same span at fewer points rather than
    from dropping or reweighting demonstration frames.

    Needs no labelling stage: the parameters are a geometric fit of the demonstration,
    computed in the preprocessor when training starts (3 s for LIBERO-10, 50 s for
    pickplace).
    """

    #: Knot spans per chunk. The emitted matrix is `chunk_size + 2*degree` rows, and
    #: that width becomes the policy's chunk -- so it is the width, not this, that has
    #: to suit the policy. 10 gives width 16, which is upstream's own real-robot
    #: configuration (`clean_bspline_policy_*.yaml`, horizon 16) and is a multiple of
    #: 8, as Diffusion's temporal U-Net requires. The recorded UR10e dataset used 20
    #: (width 26); that is fine for ACT and xVLA but Diffusion rejects it.
    chunk_size: int = 10
    degree: int = 3
    #: Fit tolerance, in the dataset's own action units, applied per element -- so a
    #: gripper edge constrains the knot search as hard as a position does.
    max_error: float = 0.01
    #: Which columns of this dataset's action can be splined. See `bspline.layout`.
    layout: str = "cart7"
    #: How the parameter matrix's columns sit in the tensor the policy sees.
    #: "knot_first" is upstream's and suits ACT and Diffusion, which treat the action
    #: as an undifferentiated vector. xVLA slices it by hardcoded index, so a knot at
    #: column 0 trains as an x-coordinate -- use "xvla_ee6d20" there, together with
    #: --policy.action_mode=ee6d_bspline.
    arrangement: str = "knot_first"
    #: Store the knot column as consecutive differences instead of as offsets from
    #: the current frame. Off by default, matching every shipped upstream config and
    #: the recorded UR10e dataset. Note this concerns *time* only: the control points
    #: are absolute poses either way, and the knots are relative to the sample's own
    #: frame either way -- the choice is offsets (-7, 0, 4, 7, ... 51) against the
    #: differences of those (4, 3, 2, ...). Offsets normalize to roughly [-1.6, +1.4],
    #: which is fine; the cost is that a single per-column statistic spans the
    #: row-to-row ramp, attenuating the per-sample knot signal about 2.3x relative to
    #: the control points. Turning this on removes the ramp if that ever matters.
    relative_knots: bool = False
    #: Actions decoded from one predicted spline, at inference. The speed lever, and a
    #: decode-time choice needing no retraining: the curve covers a fixed stretch of
    #: demonstrated motion, so fewer samples cover it in fewer executed steps. The
    #: realised factor varies per chunk with the predicted span and is published as
    #: `bspline_rate`. Defaults to the matrix width, i.e. roughly demonstration speed.
    num_actions: int | None = None
    #: Frame rate of the demonstrations, used to express knots in seconds for
    #: arrangements that need it. A config field rather than something read off the
    #: dataset, because evaluation has no dataset and must reconstruct the exact knot
    #: scaling the checkpoint trained under. Checked against the dataset when there is
    #: one, so a mismatch fails loudly instead of silently rescaling time.
    fps: float = 20.0
    #: Arm position-gain multiplier at eval, so the plant can track waypoints that
    #: are further apart. Upstream's own default (`--stiffness-kp-scale`, 2.0); kd
    #: and the gripper are deliberately left nominal, which is what upstream does.
    stiffness_kp_scale: float = 2.0

    def __post_init__(self):
        if self.chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {self.chunk_size}")
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")

    def _arrange(self):
        """The arrangement with its knot scale resolved, from config alone.

        Independent of the dataset, so evaluation -- which has none -- rebuilds
        exactly the scaling the checkpoint trained under.
        """
        from pace_bench.methods.bspline.layout import resolve_arrangement

        arrangement = resolve_arrangement(self.arrangement)
        if arrangement.channels is None:
            return arrangement
        # Knots in seconds, not frames: an arrangement exists because the policy reads
        # its action vector structurally, and such a policy (xVLA) does not normalize
        # it either, so raw magnitudes decide how much each channel counts.
        return replace(arrangement, knot_scale=1.0 / self.fps)

    def _resolved_layout(self):
        """The action layout, from config alone wherever the name fixes the width."""
        from pace_bench.methods.bspline.layout import LAYOUTS, resolve_layout

        declared = LAYOUTS.get(self.layout)
        if declared is None or declared.raw_dim is None:
            # "identity" adopts the dataset's own width, so only a run that has a
            # dataset can resolve it.
            return getattr(self, "_layout", None)
        return resolve_layout(self.layout, declared.raw_dim)

    @property
    def width(self) -> int:
        """Rows of the parameter matrix: what the policy's chunk field becomes."""
        return self.chunk_size + 2 * self.degree

    def _build(self, dataset):
        """Fit every episode. Cached on the instance so the two hooks share one pass."""
        if getattr(self, "_splines", None) is not None:
            return self._splines, self._episode_starts
        from pace_bench.methods.bspline.layout import resolve_arrangement, resolve_layout
        from pace_bench.methods.bspline.processor import EpisodeSplines

        raw_dim = int(dataset.meta.features[ACTION]["shape"][0])
        layout = resolve_layout(self.layout, raw_dim)
        arrangement = resolve_arrangement(self.arrangement)
        if float(dataset.meta.fps) != self.fps:
            raise ValueError(
                f"--method.fps is {self.fps} but this dataset records "
                f"{dataset.meta.fps}. The knot column is scaled by 1/fps, so a mismatch "
                f"silently rescales time; set --method.fps={dataset.meta.fps}."
            )
        arrangement = self._arrange()
        if arrangement.name == "xvla_ee6d20":
            # Importing registers `ee6d_bspline` in xVLA's own action registry, which
            # `--policy.action_mode` then resolves. Done here rather than at package
            # import so that selecting any other method never pulls in xVLA.
            import pace_bench.methods.bspline.xvla_action  # noqa: F401
        starts = episode_starts_from_metadata(dataset.meta)
        action_table = np.asarray(dataset.hf_dataset[ACTION], dtype=np.float64)
        episode_actions = {
            episode: action_table[start : start + dataset.meta.episodes[episode]["length"]]
            for episode, start in starts.items()
        }
        started = time.perf_counter()
        splines = EpisodeSplines(
            episode_actions, layout, self.chunk_size, self.degree, self.max_error
        )
        logging.info(
            "B-spline: fitted %d episodes in %.1f s, holding %.2f MB of splines "
            "(layout %r, %d-dim action -> %d spline dims, matrix %dx%d)",
            len(episode_actions), time.perf_counter() - started, splines.nbytes() / 1e6,
            layout.name, raw_dim, layout.spline_dim, self.width, splines.channels,
        )
        self._splines, self._episode_starts = splines, starts
        self._layout, self._arrangement = layout, arrangement
        return splines, starts

    def adjust_policy(self, policy_cfg) -> None:
        """The policy predicts one parameter matrix, so its chunk is that matrix.

        Its `n_action_steps` matches: a B-spline chunk is not a sequence executed
        step by step but a single object decoded into however many actions the
        deployment asks for, so re-planning happens once per predicted spline.
        """
        fields = POLICY_CHUNK_FIELDS.get(policy_cfg.type)
        if fields is None:
            raise ValueError(
                f"B-spline does not know the chunk fields of policy type "
                f"{policy_cfg.type!r}; add it to POLICY_CHUNK_FIELDS "
                f"(known: {sorted(POLICY_CHUNK_FIELDS)})."
            )
        # Diffusion's temporal U-Net halves the horizon once per `down_dims` stage,
        # so the width must be a multiple of 2**len(down_dims). LeRobot checks this in
        # `DiffusionConfig.__post_init__`, which has already run by the time a method
        # mutates the config -- so without this the run dies mid-forward with
        # "Sizes of tensors must match except in dimension 1", tens of seconds after
        # the fit, naming neither the horizon nor the method.
        if (down_dims := getattr(policy_cfg, "down_dims", None)) is not None:
            factor = 2 ** len(down_dims)
            if self.width % factor:
                usable = [c for c in range(2, 64) if (c + 2 * self.degree) % factor == 0]
                raise ValueError(
                    f"B-spline emits a {self.width}-row matrix (chunk_size="
                    f"{self.chunk_size} + 2*degree={self.degree}), and policy type "
                    f"{policy_cfg.type!r} needs that to be a multiple of {factor} "
                    f"(2**len(down_dims), down_dims={tuple(down_dims)}). Use "
                    f"--method.chunk_size from {usable[:6]}..."
                )
        setattr(policy_cfg, fields.chunk, self.width)
        setattr(policy_cfg, fields.executed, self.width)
        logging.info(
            "B-spline: set %s and %s to %d (policy type %r)",
            fields.chunk, fields.executed, self.width, policy_cfg.type,
        )

    def adjust_dataset(self, dataset) -> None:
        """Point the metadata at the parameter matrix, so the policy is built for it.

        `make_policy` reads `ds_meta.features` for the action's width and
        `ds_meta.stats` for its normalization buffers, so both have to describe the
        parameters rather than the raw actions. The statistics are computed from the
        fits themselves -- the dataset's own action stats describe a different
        quantity in different units and would mis-scale every column.
        """
        from pace_bench.methods.bspline.spline import encode_relative_knots

        splines, _ = self._build(dataset)

        # Statistics must describe exactly what the step emits, which is the chunk
        # *shifted* so its knots read as offsets from the sample's own frame. The
        # unshifted chunks carry absolute episode positions running to the episode
        # length -- on pickplace that is a knot column with std 341 instead of 2.5,
        # and a normalizer built from it would divide the knot signal away. Only
        # column 0 is affected (a shift cancels in the differences and never touches
        # a control point), but it is computed by replaying the step rather than by
        # reasoning about which columns move.
        arrangement = self._arrangement
        channels = arrangement.channels or splines.channels
        total = np.zeros(channels, dtype=np.float64)
        total_sq = np.zeros(channels, dtype=np.float64)
        low = np.full(channels, np.inf)
        high = np.full(channels, -np.inf)
        count = 0
        for episode, length in ((e, len(f)) for e, f in splines.frame_to_chunk.items()):
            for frame in range(length):
                matrix = splines.parameters(episode, frame)
                if self.relative_knots:
                    matrix = encode_relative_knots(matrix, self.degree)
                matrix = arrangement.emit(matrix)
                total += matrix.sum(axis=0)
                total_sq += (matrix.astype(np.float64) ** 2).sum(axis=0)
                low = np.minimum(low, matrix.min(axis=0))
                high = np.maximum(high, matrix.max(axis=0))
                count += matrix.shape[0]
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))

        dataset.meta.features[ACTION] = {**dataset.meta.features[ACTION], "shape": (channels,)}
        self._action_stats = {
            "mean": mean.astype(np.float32),
            "std": np.maximum(std, 1e-8).astype(np.float32),
            "min": low.astype(np.float32),
            "max": high.astype(np.float32),
            "count": np.array([count]),
        }
        dataset.meta.stats[ACTION] = dict(self._action_stats)
        logging.info(
            "B-spline: action metadata now %d-dim (%s, knot x%.4g); knot mean %.2f std %.2f",
            channels, arrangement.name, arrangement.knot_scale,
            mean[splines.channels - 1] if arrangement.channels else mean[0],
            std[splines.channels - 1] if arrangement.channels else std[0],
        )

    def adjust_processors(self, preprocessor, postprocessor) -> None:
        """Install the parameter statistics into whatever normalizers were built.

        A from-scratch policy gets them through the dataset metadata, but a pretrained
        one (`--policy.path`, which is how every xVLA arm runs) carries its own saved
        normalizer, whose action statistics describe the *dataset's* actions. On
        LIBERO the knot column then lands in a slot the checkpoint recorded as
        constant zero, so it is normalized by mean 0 / std 1 and reaches the loss at
        its raw magnitude -- a knot loss of 7365 beside a position loss of 2.1.
        """
        stats = getattr(self, "_action_stats", None)
        if stats is None:
            return
        installed = 0
        for pipeline in (preprocessor, postprocessor):
            for step in getattr(pipeline, "steps", []):
                step_stats = getattr(step, "stats", None)
                if isinstance(step_stats, dict) and ACTION in step_stats:
                    step_stats[ACTION] = {
                        key: torch.as_tensor(value) for key, value in stats.items()
                    }
                    installed += 1
        logging.info("B-spline: installed parameter statistics into %d normalizer(s)", installed)

    def postprocessor_steps(self) -> list[ProcessorStep]:
        """Decode predicted parameters into executable actions.

        Appended, so it runs *after* the unnormalizer: the knot column is a time in
        source frames and the control points are poses, and both have to be in their
        own units before the curve can be evaluated.
        """
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        return [
            BSplineDecodeStep(
                num_actions=self.num_actions or self.width,
                degree=self.degree,
                relative_knots=self.relative_knots,
                layout=self._resolved_layout(),
                arrangement=self._arrange(),
            )
        ]

    def preprocessor_steps(self, dataset=None) -> list[ProcessorStep]:
        from pace_bench.methods.bspline.processor import BSplineChunkStep

        if dataset is None:
            return [BSplineChunkStep(relative_knots=self.relative_knots, degree=self.degree)]
        splines, starts = self._build(dataset)
        return [
            BSplineChunkStep(
                splines=splines,
                episode_starts=starts,
                relative_knots=self.relative_knots,
                degree=self.degree,
                arrangement=self._arrangement,
            )
        ]
