"""Tests for config inheritance.

The merge is the part worth testing: if it replaced nested blocks wholesale instead
of merging them, an arm setting one gripper field would silently discard the others
and take dataclass defaults — which is precisely the drift the shared file exists to
prevent, reappearing one level down.
"""

import pytest
import yaml

from pace_bench.real.configs import deep_merge, materialise, resolve_config

REAL_CONFIGS = None  # set below if the repo's own configs are present


def w(tmp, name, data):
    p = tmp / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


# --------------------------------------------------------------------------
# deep_merge
# --------------------------------------------------------------------------

def test_override_wins_at_the_top_level():
    assert deep_merge({"fps": 20.0}, {"fps": 30.0}) == {"fps": 30.0}


def test_nested_blocks_merge_rather_than_replace():
    """The property that keeps an arm from silently dropping inherited fields."""
    base = {"gripper": {"slowdown_frames": 5, "invert": False}}
    got = deep_merge(base, {"gripper": {"slowdown_frames": 0}})
    assert got == {"gripper": {"slowdown_frames": 0, "invert": False}}, \
        "invert must survive an arm that only mentions slowdown_frames"


def test_merge_is_recursive():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    assert deep_merge(base, {"a": {"b": {"c": 9}}}) == {"a": {"b": {"c": 9, "d": 2}}}


def test_lists_are_replaced_not_concatenated():
    assert deep_merge({"xs": [1, 2]}, {"xs": [3]}) == {"xs": [3]}


def test_base_is_not_mutated():
    base = {"g": {"a": 1}}
    deep_merge(base, {"g": {"a": 2}})
    assert base == {"g": {"a": 1}}, "merging must not edit the shared defaults in place"


# --------------------------------------------------------------------------
# resolve_config
# --------------------------------------------------------------------------

def test_include_is_resolved(tmp_path):
    w(tmp_path, "base.yaml", {"fps": 20.0, "sender": {"cpp": True, "rt_priority": 0}})
    arm = w(tmp_path, "arm.yaml", {"_include": "base.yaml", "method": {"type": "pace"}})
    got = resolve_config(arm)
    assert got["fps"] == 20.0
    assert got["sender"] == {"cpp": True, "rt_priority": 0}
    assert got["method"] == {"type": "pace"}
    assert "_include" not in got, "the include key must not reach the config class"


def test_arm_overrides_the_shared_value(tmp_path):
    w(tmp_path, "base.yaml", {"fps": 20.0, "gripper": {"slowdown_frames": 5, "invert": False}})
    arm = w(tmp_path, "arm.yaml", {"_include": "base.yaml", "gripper": {"slowdown_frames": 0}})
    got = resolve_config(arm)
    assert got["gripper"] == {"slowdown_frames": 0, "invert": False}


def test_includes_chain(tmp_path):
    w(tmp_path, "a.yaml", {"x": 1, "y": 1})
    w(tmp_path, "b.yaml", {"_include": "a.yaml", "y": 2})
    c = w(tmp_path, "c.yaml", {"_include": "b.yaml", "z": 3})
    assert resolve_config(c) == {"x": 1, "y": 2, "z": 3}


def test_a_cycle_raises_instead_of_recursing(tmp_path):
    w(tmp_path, "a.yaml", {"_include": "b.yaml"})
    w(tmp_path, "b.yaml", {"_include": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        resolve_config(tmp_path / "a.yaml")


def test_missing_include_names_the_file(tmp_path):
    arm = w(tmp_path, "arm.yaml", {"_include": "nope.yaml"})
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        resolve_config(arm)


def test_include_is_relative_to_the_including_file(tmp_path):
    d = tmp_path / "configs"; d.mkdir()
    w(d, "base.yaml", {"fps": 20.0})
    arm = w(d, "arm.yaml", {"_include": "base.yaml"})
    assert resolve_config(arm)["fps"] == 20.0, "resolution must not depend on cwd"


# --------------------------------------------------------------------------
# materialise
# --------------------------------------------------------------------------

def test_a_plain_config_keeps_its_own_path(tmp_path):
    p = w(tmp_path, "plain.yaml", {"fps": 20.0})
    assert materialise(p) == p, "no include, no temp file — keep the name in logs"


def test_materialise_writes_the_merged_result(tmp_path):
    w(tmp_path, "base.yaml", {"fps": 20.0, "sender": {"cpp": True}})
    arm = w(tmp_path, "arm.yaml", {"_include": "base.yaml", "method": {"type": "pace"}})
    out = materialise(arm)
    assert out != arm
    d = yaml.safe_load(out.read_text())
    assert d["fps"] == 20.0 and d["method"]["type"] == "pace" and "_include" not in d
