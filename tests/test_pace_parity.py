"""PACE must reproduce the fork bit-for-bit.

The published PACE numbers were produced by the speed logic embedded in
``XVLAPolicy.select_action`` in the lerobot_uncertainty fork. Re-running those
evaluations to validate the port would cost GPU-days and still only compare noisy
success rates. Instead the fork's own code was driven directly -- with the network
stubbed out and everything downstream of it intact -- over a grid of configurations,
and its outputs frozen into ``assets/pace_golden.npz`` (regenerate with
``assets/gen_pace_golden.py`` under the fork's environment).

If a refactor here changes a single float, these tests fail. That is the point: the
port is only allowed to change *where* the logic lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_stack.methods.pace import PaceConfig, PaceSpeedStep

ASSETS = Path(__file__).parent / "assets"
GOLDEN = np.load(ASSETS / "pace_golden.npz")
META = json.loads((ASSETS / "pace_golden.json").read_text())

# Same two normalization regimes the golden generator used. _unnormalize_actions
# branches on which keys are present, and PACE's geometry is computed after that
# branch, so both paths need covering.
DIM = META["dim"]
STATS = {
    "meanstd": {"action": {"mean": torch.linspace(-0.5, 0.5, DIM), "std": torch.linspace(0.5, 2.0, DIM)}},
    "minmax": {"action": {"min": -torch.ones(DIM) * 1.5, "max": torch.ones(DIM) * 2.5}},
    "none": {},
}

CASES = [
    (seed, stats_name, cfg_name)
    for seed in META["seeds"]
    for stats_name in META["stats"]
    for cfg_name in META["configs"]
]


def _eval_str(v):
    """The generator wrote its config dicts through ``json.dumps(default=str)``."""
    return {"True": True, "False": False}.get(v, v) if isinstance(v, str) else v


@pytest.mark.parametrize(("seed", "stats_name", "cfg_name"), CASES, ids=lambda x: str(x))
def test_matches_fork(seed, stats_name, cfg_name):
    raw = {k: _eval_str(v) for k, v in META["configs"][cfg_name].items()}
    step = PaceSpeedStep(
        raw,
        n_action_steps=META["n_action_steps"],
        dataset_stats=STATS[stats_name],
    )
    chunk = torch.from_numpy(GOLDEN[f"chunk__{seed}"])
    actions, speeds = step.plan(chunk)

    key = f"{seed}__{stats_name}__{cfg_name}"
    # The fork stacked its per-step queue pops, giving (T, B, ...); ours is (B, T, ...).
    want_actions = torch.from_numpy(GOLDEN[f"act__{key}"]).transpose(0, 1)
    want_speeds = torch.from_numpy(GOLDEN[f"spd__{key}"]).transpose(0, 1)

    torch.testing.assert_close(actions, want_actions, rtol=0, atol=0)
    torch.testing.assert_close(speeds, want_speeds, rtol=0, atol=0)


def test_golden_is_not_vacuous():
    """Guard the guard: a grid of constants would pass parity while proving nothing."""
    varying = 0
    for cfg_name in META["configs"]:
        s = GOLDEN[f"spd__0__meanstd__{cfg_name}"]
        if len(np.unique(np.round(s, 6))) > 1:
            varying += 1
    assert varying >= len(META["configs"]) // 2, "most golden cases have a constant speed profile"


def test_defaults_are_inert():
    """An unconfigured PACE step must not change behaviour.

    This is what makes ``--method.type=none`` and PACE-with-no-flags the same thing,
    and it is the property that lets the step sit in a pipeline unconditionally.
    """
    step = PaceSpeedStep()
    chunk = torch.randn(1, 32, 7)
    actions, speeds = step.plan(chunk)
    torch.testing.assert_close(actions, chunk, rtol=0, atol=0)
    assert torch.all(speeds == 1.0)


def test_chunk_is_required():
    """A single action carries no direction change, so PACE is undefined on one."""
    with pytest.raises(ValueError, match="consecutive steps"):
        PaceSpeedStep().plan(torch.randn(1, 7))


def test_config_ignores_foreign_keys():
    """Eval kwargs mix PACE knobs with env and actuation knobs in one dict."""
    cfg = PaceConfig.from_dict({"max_speed": 2.0, "speed_up": True, "task_ids": [3], "relaxed_limits": True})
    assert cfg.max_speed == 2.0
    assert cfg == PaceConfig(max_speed=2.0)
