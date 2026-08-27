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
from dataclasses import dataclass, field, fields

import draccus
from lerobot.processor.pipeline import ProcessorStep

from robot_stack.methods.demospeedup.labels import describe, load_labels
from robot_stack.methods.demospeedup.processor import DemoSpeedupRetimeStep
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

    def preprocessor_steps(self) -> list[ProcessorStep]:
        """Steps to insert before the policy. Empty for methods that act on output."""
        return []

    def postprocessor_steps(self) -> list[ProcessorStep]:
        """Steps to insert after the policy."""
        return []


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
    # "zero" for policies whose loss is masked by action_is_pad (ACT, Diffusion);
    # "hold" for policies regressing the whole chunk unmasked (xVLA), where a zero
    # tail in an absolute action space commands the world origin.
    pad_mode: str = "zero"

    def __post_init__(self):
        if self.pad_mode not in ("zero", "hold"):
            raise ValueError(f"pad_mode must be 'zero' or 'hold', got {self.pad_mode!r}")
        if self.low_v < 1 or self.high_v < 1:
            raise ValueError(f"strides must be >= 1, got low_v={self.low_v}, high_v={self.high_v}")

    def preprocessor_steps(self) -> list[ProcessorStep]:
        """Build the retiming step, loading labels if a path was given.

        Without a path the step is constructed empty and passes every sample through.
        That keeps `--method.type=demospeedup` selectable while a labelling run is
        still pending, rather than making an unlabelled dataset a hard error.
        """
        labels, config = ({}, {})
        if self.labels_path:
            labels, config = load_labels(self.labels_path)
            logging.info("DemoSpeedup labels: %s", describe(labels, config))
        return [
            DemoSpeedupRetimeStep(labels=labels, low_v=self.low_v, high_v=self.high_v, pad_mode=self.pad_mode)
        ]
