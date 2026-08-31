"""Config includes, so transport settings live in exactly one file.

Every arm of a benchmark shares its plumbing -- sender, blending, loop timing,
gripper -- and differs only in its ``method``. Copying that plumbing into each arm's
YAML reintroduces the drift this whole layer exists to prevent: if ``pace_fast.yaml``
and ``baseline.yaml`` disagree on ``blend.overlap``, the comparison measures the
plumbing rather than the method, and nothing in the output says so.

So an arm names what it inherits::

    _include: deploy_defaults.yaml
    method:
      type: pace
      max_speed: 2.0

and its file then shows only what makes it that arm.

draccus reads a single YAML by path, so includes are resolved *before* it sees the
file: the merged result is written to a temp file and that path is handed on. The
merge itself is a pure dict operation, which is the part worth testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

#: Key naming a file to inherit from. Relative to the including file's directory.
INCLUDE_KEY = "_include"


def deep_merge(base: dict, override: dict) -> dict:
    """``override`` wins, but nested dicts merge rather than replace.

    Replacing wholesale would mean an arm setting one gripper field silently
    discarded the others -- ``gripper: {slowdown_frames: 0}`` would drop ``invert``
    and take the dataclass default instead of the shared one. Merging keeps the parts
    the arm did not mention.

    Lists are replaced, not concatenated: a list in config is a value, and appending
    to an inherited one is almost never what the author meant.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML, resolving ``_include`` chains depth-first.

    An include may itself include, so a site could keep a base, a rig-specific layer
    and an arm. Cycles raise rather than recursing forever.
    """
    p = Path(path).resolve()
    if p in _seen:
        chain = " -> ".join(x.name for x in _seen + (p,))
        raise ValueError(f"circular config include: {chain}")
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")

    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping at the top level: {p}")

    parent = data.pop(INCLUDE_KEY, None)
    if parent is None:
        return data
    # Relative to the including file, so a config directory can be moved wholesale.
    base = resolve_config(p.parent / parent, _seen + (p,))
    return deep_merge(base, data)


def materialise(path: str | Path) -> Path:
    """Resolve includes and write the result where draccus can read it.

    Returns the original path untouched when there is nothing to resolve, so a plain
    config keeps its own name in logs and error messages.
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) if p.exists() else None
    if not isinstance(raw, dict) or INCLUDE_KEY not in raw:
        return p

    merged = resolve_config(p)
    tmp = Path(tempfile.mkdtemp(prefix="pace_cfg_")) / p.name
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False))
    return tmp
