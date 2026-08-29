"""The method choice is the config's single source of truth.

The failure this guards against is not a crash but a lie: a config that *looks* like
it selected a method while the code does something else. So these tests care about
what is representable, not only about what parses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import draccus
import pytest

from pace_bench.methods.config import MethodConfig, MethodPipelineConfig, NoMethod, PaceMethod
from pace_bench.methods.pace.processor import PaceSpeedStep
from pace_bench.methods.pace.speed import PaceConfig


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


def test_halving_uses_the_typed_registry_not_attribute_probing():
    """Chunk fields come from POLICY_CHUNK_FIELDS keyed on the policy type.

    An xVLA/ACT-shaped config halves chunk_size; a diffusion-shaped one halves
    horizon; a policy type outside the registry is a loud error, never a silent
    no-op or a lucky attribute match.
    """

    @dataclass
    class ActShaped:
        type: str = "act"
        chunk_size: int = 30
        n_action_steps: int = 30

    @dataclass
    class DiffusionShaped:
        type: str = "diffusion"
        horizon: int = 16
        n_action_steps: int = 8
        do_mask_loss_for_padding: bool = True

    @dataclass
    class UnknownPolicy:
        type: str = "groot"
        chunk_size: int = 30
        n_action_steps: int = 30

    method = parse(["--method.type=demospeedup"]).method
    knobs = ActShaped()
    method.adjust_policy(knobs)
    assert (knobs.chunk_size, knobs.n_action_steps) == (15, 15)
    assert method._trained_chunk == 15

    knobs = DiffusionShaped()
    parse(["--method.type=demospeedup", "--method.pad_mode=hold"]).method.adjust_policy(
        knobs
    )
    assert (knobs.horizon, knobs.n_action_steps) == (8, 4)

    with pytest.raises(ValueError, match="POLICY_CHUNK_FIELDS"):
        method.adjust_policy(UnknownPolicy())

    knobs = ActShaped()
    parse(["--method.type=demospeedup", "--method.halve_chunk=false"]).method.adjust_policy(
        knobs
    )
    assert (knobs.chunk_size, knobs.n_action_steps) == (30, 30)

    knobs = ActShaped()
    parse([]).method.adjust_policy(knobs)  # NoMethod: untouched
    assert (knobs.chunk_size, knobs.n_action_steps) == (30, 30)


def test_halving_is_idempotent_so_a_resumed_run_does_not_re_halve():
    """Re-applying must land on the same chunk, not halve again.

    `save_checkpoint` writes the config AFTER this hook has mutated it, so a
    checkpoint's train_config.json records the already-halved chunk together with
    `method=demospeedup`. Resuming from it re-runs this hook, and halving the
    halved value would quarter the chunk (30 -> 15 -> 7) -- silently training a
    policy a quarter the intended length.
    """

    @dataclass
    class ActShaped:
        type: str = "act"
        chunk_size: int = 30
        n_action_steps: int = 30

    method = parse(["--method.type=demospeedup"]).method
    knobs = ActShaped()
    method.adjust_policy(knobs)
    assert (knobs.chunk_size, knobs.n_action_steps) == (15, 15)

    method.adjust_policy(knobs)
    assert (knobs.chunk_size, knobs.n_action_steps) == (15, 15)
    assert method._trained_chunk == 15


def test_the_pre_halve_geometry_survives_a_config_round_trip():
    """The guard only works if `source_chunk` reaches the resumed process.

    A resume parses a fresh config object out of the checkpoint's JSON, so an
    in-memory marker would be lost; this pins that the recorded geometry is a real
    serialized field and that a config carrying it halves from the original.
    """

    @dataclass
    class ActShaped:
        type: str = "act"
        chunk_size: int = 30
        n_action_steps: int = 30

    method = parse(["--method.type=demospeedup"]).method
    method.adjust_policy(ActShaped())
    encoded = draccus.encode(method)
    assert encoded["source_chunk"] == 30
    assert encoded["source_executed"] == 30

    # What the resumed process sees: this config, and an already-halved policy.
    resumed = parse(
        [
            "--method.type=demospeedup",
            f"--method.source_chunk={encoded['source_chunk']}",
            f"--method.source_executed={encoded['source_executed']}",
        ]
    ).method
    knobs = ActShaped(chunk_size=15, n_action_steps=15)
    resumed.adjust_policy(knobs)
    assert (knobs.chunk_size, knobs.n_action_steps) == (15, 15)


def test_tail_walk_fills_every_slot_mid_episode():
    """Upstream's property, now by construction: mid-episode chunks have no pads.

    The walk advances at most high_v raw frames per kept waypoint, so any tail of at
    least chunk*high_v frames fills the chunk with real waypoints -- for ANY label
    content. This is the property the fork's fixed 2x window violated (~7% of its
    executed steps were trained dwells).
    """
    import numpy as np
    import torch

    from pace_bench.methods.demospeedup.retime import retime_tail

    chunk, high_v = 15, 4
    rng = np.random.default_rng(0)
    for _ in range(300):
        tail_len = rng.integers(chunk * high_v, 300)
        labels = (rng.random(tail_len) < rng.random()).astype(np.int64)
        actions = torch.arange(tail_len, dtype=torch.float32).unsqueeze(1)
        out, is_pad = retime_tail(actions, labels, chunk, 2, high_v)
        assert not is_pad.any(), f"walk ran dry mid-episode: {labels.tolist()}"
        vals = out.squeeze(1).tolist()
        assert vals == sorted(set(vals)), "waypoints must be distinct and in order"
        assert vals[0] == 0.0
