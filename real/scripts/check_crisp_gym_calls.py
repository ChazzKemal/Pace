#!/usr/bin/env python
"""Check every crisp_gym call in run_real.py against the real signature.

The two repos are pinned separately, so crisp_gym can move under Pace's feet. A
renamed parameter or a reordered argument is then a TypeError raised *after* the
robot is powered up, the controller switched and the policy loaded on the GPU --
the most expensive possible moment to discover it.

This catches that statically. It needs crisp_gym importable but nothing else: no
lerobot, no pace_bench, no robot. Run it after bumping the crisp_gym pin.

    PYTHONPATH=/path/to/crisp_gym python real/scripts/check_crisp_gym_calls.py

It also catches the specific bug that prompted it: `n_action_steps` was accepted by
_LeRobotChunkSource and simply not passed, so a config asking for 32 executed steps
silently ran the policy's full 100.
"""

import ast
import importlib
import inspect
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "src/pace_bench/real/run_real.py"

MODULES = {
    "_LeRobotChunkSource": "crisp_gym.deploy.sources",
    "run_producer_loop": "crisp_gym.deploy.loop",
    "write_run_artifacts": "crisp_gym.deploy.trace",
    "RunRecord": "crisp_gym.deploy.trace",
    "build_parser": "crisp_gym.deploy.cli",
    "_build_obs_schema": "crisp_gym.deploy.obs",
    "_get_obs_zerofill": "crisp_gym.deploy.obs",
}


def main() -> int:
    session = importlib.import_module("crisp_gym.deploy.session")
    targets = dict(MODULES)
    for name in dir(session):
        if name.startswith(("phase_", "build_env")):
            targets[name] = "crisp_gym.deploy.session"

    tree = ast.parse(RUNNER.read_text())
    checked = bad = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name not in targets:
            continue
        obj = getattr(importlib.import_module(targets[name]), name)
        sig = inspect.signature(obj)
        kwargs = {k.arg for k in node.keywords if k.arg}
        extra = kwargs - set(sig.parameters)
        slots = [p for p in sig.parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        too_many = len(node.args) > len(slots)
        checked += 1
        if extra or too_many:
            bad += 1
            print(f"  MISMATCH {name} (line {node.lineno}): "
                  f"unexpected={sorted(extra)} positional={len(node.args)}/{len(slots)}")
        else:
            print(f"  OK       {name} (line {node.lineno})")

    print(f"\n{checked} call sites checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
