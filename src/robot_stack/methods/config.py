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

import numpy as np
from dataclasses import dataclass, field, fields

import draccus
from lerobot.processor.pipeline import ProcessorStep

from robot_stack.methods.demospeedup.labels import describe, load_labels
from robot_stack.methods.demospeedup.processor import DemoSpeedupRetimeStep, episode_starts_from_metadata
from robot_stack.methods.demospeedup.retime import HIGH_V, LOW_V
from robot_stack.methods.pace.processor import PaceSpeedStep
from robot_stack.methods.pace.speed import PaceConfig


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

    def adjust_policy_after_datasets(self, policy_cfg) -> None:
        """Mutate the policy config after the datasets are built, before the policy is.

        Exists for methods that change the chunk the policy trains (DemoSpeedup's
        halving). No-op by default.
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
    """Eval-time speed modulation. See :mod:`robot_stack.methods.pace.speed`.

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
    def adjust_policy_after_datasets(self, policy_cfg) -> None:
        """Halve the chunk the policy trains; also guard the pad_mode/loss pairing.

        Field names come from :data:`POLICY_CHUNK_FIELDS`, keyed on the policy's
        registered ``type``. The dataset's own action window no longer matters --
        the retiming step substitutes chunks from its preloaded episode table -- so
        ordering relative to dataset creation is not load-bearing; this hook is
        simply where the trainer calls us.
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
        chunk = getattr(policy_cfg, fields.chunk)
        setattr(policy_cfg, fields.chunk, chunk // 2)
        # The retime step must emit chunks of exactly the trained length: ACT
        # consumes its action input at chunk_size, no truncation.
        self._trained_chunk = chunk // 2
        executed = getattr(policy_cfg, fields.executed)
        setattr(policy_cfg, fields.executed, executed // 2)
        logging.info(
            "DemoSpeedup: halved %s to %d, %s to %d (policy type %r)",
            fields.chunk, chunk // 2, fields.executed, executed // 2, policy_type,
        )
