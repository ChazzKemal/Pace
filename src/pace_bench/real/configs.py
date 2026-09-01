"""Config includes, so transport settings live in exactly one file.

Every run configuration in a benchmark shares its plumbing -- sender, blending, loop timing,
gripper -- and differs only in its ``method``. Copying that plumbing into each config's
YAML reintroduces the drift this whole layer exists to prevent: if ``pace_fast.yaml``
and ``baseline.yaml`` disagree on ``blend.overlap``, the comparison measures the
plumbing rather than the method, and nothing in the output says so.

So a config names what it inherits::

    _include: deploy_defaults.yaml
    method:
      type: pace
      max_speed: 2.0

and its file then shows only what distinguishes it.

draccus reads a single YAML by path, so includes are resolved *before* it sees the
file: the merged result is written to a temp file and that path is handed on. The
merge itself is a pure dict operation, which is the part worth testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

#: The repo, located from this file rather than from the working directory. The robot
#: environment's `pixi run` starts in `real/`, so anything resolved against the cwd
#: becomes `real/real/...` there -- which is how `--task` failed while the identical
#: command from the repo root worked.
REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(path: str | Path) -> Path:
    """A repo-relative path, made absolute. An absolute path is returned unchanged.

    Used for paths that name a file *inside the repo* -- the task registry -- as
    opposed to a path the operator typed, where relative-to-cwd is what they meant.
    Keeping the stored value relative is deliberate: it is what makes a dumped
    ``run_config.yaml`` re-parsable on another machine.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


#: Key naming a file to inherit from. Relative to the including file's directory.
INCLUDE_KEY = "_include"


def deep_merge(base: dict, override: dict) -> dict:
    """``override`` wins, but nested dicts merge rather than replace.

    Replacing wholesale would mean a config setting one gripper field silently
    discarded the others -- ``gripper: {slowdown_frames: 0}`` would drop ``invert``
    and take the dataclass default instead of the shared one. Merging keeps the parts
    it did not mention.

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
    and a run configuration. Cycles raise rather than recursing forever.
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


# ---------------------------------------------------------------------------
# Task -> checkpoint registry
#
# A method config says *how* to deploy; the task says *which* checkpoint. Keeping
# them apart is what lets one bspline_1x.yaml serve every task that has a B-spline
# arm, instead of a file per (task, method) pair each restating the method block.
#
# The registry is YAML rather than a dict in code because the paths are per-machine:
# this rig's checkpoints live in a sibling checkout, and the very checkpoint we
# deploy records /home/batur/... for the dataset it trained on. A constant in the
# source would be wrong for everyone but one operator.
# ---------------------------------------------------------------------------

#: Registry filename, looked up beside the config that names the task.
TASKS_FILE = "tasks.yaml"

#: Key holding the directory that a task's checkpoint paths are relative to. May be
#: given once at the top level and overridden per task.
ROOT_KEY = "root"


def load_tasks(path: str | Path) -> dict[str, Any]:
    """Read the registry, with the top-level ``root`` pushed into each task."""
    p = repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"task registry not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"task registry must be a mapping at the top level: {p}")
    shared = data.pop(ROOT_KEY, None)
    tasks = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"task {name!r} in {p} must be a mapping of method -> checkpoint, "
                f"got {type(entry).__name__}"
            )
        tasks[name] = {ROOT_KEY: shared, **entry}
    return tasks


def resolve_policy_path(task: str, method_type: str, registry: str | Path) -> Path:
    """The checkpoint trained for this ``(task, method)``, as an absolute path.

    Raises rather than falling back to any checkpoint that happens to exist. A deploy
    that silently ran the wrong arm would produce a number nobody could tell was
    wrong -- the arm moves, the task sometimes succeeds, and the comparison is void.
    """
    tasks = load_tasks(registry)
    if task not in tasks:
        raise ValueError(
            f"unknown task {task!r} in {Path(registry).name}; known: {sorted(tasks)}"
        )
    entry = dict(tasks[task])
    root = entry.pop(ROOT_KEY, None)
    trained = {m: v for m, v in entry.items() if v}
    if method_type not in trained:
        untrained = sorted(m for m in entry if not entry[m])
        hint = f" (listed but not trained: {untrained})" if untrained else ""
        raise ValueError(
            f"task {task!r} has no {method_type!r} checkpoint{hint}; "
            f"trained: {sorted(trained)}. Train it, name another task, or set "
            f"--policy_path explicitly."
        )
    path = Path(trained[method_type])
    if not path.is_absolute():
        if not root:
            raise ValueError(
                f"task {task!r} gives {method_type!r} a relative path "
                f"({path}) but no {ROOT_KEY!r} to resolve it against"
            )
        path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(
            f"task {task!r} method {method_type!r} points at {path}, which does not "
            "exist. The registry is a map of what was trained, not a promise it is "
            "still on this machine."
        )
    return path
