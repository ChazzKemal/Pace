"""The method choice is the config's single source of truth.

The failure this guards against is not a crash but a lie: a config that *looks* like
it selected a method while the code does something else. So these tests care about
what is representable, not only about what parses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import draccus
import pytest

from robot_stack.methods.config import MethodConfig, MethodPipelineConfig, NoMethod, PaceMethod
from robot_stack.methods.pace.processor import PaceSpeedStep
from robot_stack.methods.pace.speed import PaceConfig


@dataclass
class Runner(MethodPipelineConfig):
    """Stands in for a real runner config embedding the method choice."""

    tasks: str = "0-9"


def parse(args):
    return draccus.parse(Runner, args=args)


def test_default_is_the_baseline():
    """A config that never mentions a method must behave as it did before methods."""
    cfg = parse([])
    assert isinstance(cfg.method, NoMethod)
    assert cfg.method.preprocessor_steps() == []
    assert cfg.method.postprocessor_steps() == []


def test_selecting_pace_builds_the_step():
    cfg = parse(["--method.type=pace", "--method.max_speed=1.5", "--method.action_stride=2"])
    (step,) = cfg.method.postprocessor_steps()
    assert isinstance(step, PaceSpeedStep)
    assert step.config.max_speed == 1.5
    assert step.config.action_stride == 2


def test_a_methods_knobs_are_unavailable_to_other_methods():
    """The point of the registry: no more flags that are silently ignored.

    Under the fork's flat flags, `speedup_low_v` could be set with speedup off, and
    PACE's knobs could be passed while DemoSpeedup was selected. Both are now parse
    errors rather than quietly dead configuration.
    """
    with pytest.raises(Exception, match="(?i)max_speed|unexpected|decod"):
        parse(["--method.type=none", "--method.max_speed=2.0"])


def test_min_speed_defaults_to_half_of_max():
    """The convention every recorded experiment used, applied in exactly one place."""
    assert PaceMethod(max_speed=3.0).to_pace_config().min_speed == 1.5
    assert PaceMethod(max_speed=3.0, min_speed=2.0).to_pace_config().min_speed == 2.0


def test_cli_surface_matches_the_algorithm():
    """PaceMethod and PaceConfig must not drift apart.

    A knob present in one and missing from the other is invisible until someone sets
    it and nothing happens, so it is checked structurally instead.
    """
    cli = {f.name for f in fields(PaceMethod)}
    algo = {f.name for f in fields(PaceConfig)}
    assert cli == algo, f"only in CLI: {cli - algo}; only in PaceConfig: {algo - cli}"


def test_axis_channel_defaults_to_the_experiments_not_the_policy():
    """The policy defaults enable_ori_axis True; every recorded ablation ran it False."""
    assert PaceMethod().enable_ori_axis is False
    assert PaceConfig().enable_ori_axis is True


def test_type_reports_the_registered_name():
    assert NoMethod().type == "none"
    assert PaceMethod().type == "pace"
    assert set(MethodConfig.get_known_choices()) >= {"none", "pace"}


def test_round_trips_through_draccus():
    """Encoding must carry the choice, or a saved config cannot be reloaded."""
    original = PaceMethod(max_speed=1.5, action_stride=2, lookahead_agg="cumulative_bending")
    restored = draccus.decode(MethodConfig, draccus.encode(original, MethodConfig))
    assert restored == original
