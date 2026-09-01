"""Setting flags on crisp_gym's deploy namespace without inventing new ones.

``deploy_args`` seeds an ``argparse.Namespace`` from crisp_gym's own deploy parser and
overrides the handful of flags this config exposes. The hazard is that a
``Namespace`` accepts *any* attribute: ``args.blend_overlap = x`` succeeds whether or
not crisp_gym still calls it that. If the flag is renamed under a pin bump, the
assignment lands on a dead name, the real flag keeps the parser's default, and the run
proceeds -- misconfigured, silently, on live hardware. The only symptom is the robot
behaving as though the setting had never been given.

crisp_gym is pinned separately from Pace and moves on its own branch, so this is a
question of when rather than whether. ``set_flag`` turns it into a startup error,
which is the same bargain ``real/scripts/check_crisp_gym_calls.py`` already makes for
call signatures -- fail on the laptop, not after the arm is powered up.

Deliberately free of crisp_gym, torch and lerobot: it is pure attribute plumbing, and
staying importable on a machine with no robot stack is what lets ``tests/`` cover it.
"""

from __future__ import annotations

from argparse import Namespace

#: Methods whose chunks must not be seam-blended. Both compress a chunk's waypoints,
#: so consecutive chunks do not describe overlapping motion and averaging across the
#: seam invents a path neither predicted.
NO_SEAM_BLEND = ("bspline", "demospeedup")


class DeployFlagMissing(RuntimeError):
    """A flag this runner sets no longer exists on crisp_gym's deploy parser."""


def set_flag(args: Namespace, name: str, value: object, *, ours: bool = False) -> None:
    """Set one flag, refusing to create an attribute the parser does not define.

    Args:
        args: the namespace ``crisp_gym.deploy.cli.build_parser`` produced.
        name: the flag's attribute name, as the parser spells it.
        value: what to set it to.
        ours: this attribute is *not* one of crisp_gym's flags -- it is a private
            channel this runner adds to the namespace and reads back itself, so there
            is nothing to check it against. Passing it is how such a name is
            distinguished from a crisp_gym flag that has gone missing, rather than
            both failing the same way.

    Raises:
        DeployFlagMissing: ``name`` is absent from the namespace and ``ours`` is
            False, meaning crisp_gym renamed or removed it under the current pin.
    """
    if not ours and not hasattr(args, name):
        raise DeployFlagMissing(
            f"crisp_gym's deploy parser has no --{name.replace('_', '-')}, so setting "
            f"`{name}` here would leave the real flag at its default and the run would "
            f"continue misconfigured. It was renamed or removed under the current "
            f"crisp_gym pin: check `crisp_gym.deploy.cli.build_parser` and update "
            f"`deploy_args` in pace_bench.real.run_real."
        )
    setattr(args, name, value)


def blend_overlap_for(method, requested: int) -> int:
    """Seam-blend width this method tolerates, given what the config asked for.

    Returns 0 for a method in :data:`NO_SEAM_BLEND`, the request otherwise. Kept here
    rather than in the config files because a value that is unsafe for a method must
    not be reachable by editing YAML -- ``deploy_defaults.yaml`` sets ``overlap: 4``
    and every method config inherits it.
    """
    if getattr(method, "type", "none") in NO_SEAM_BLEND:
        return 0
    return int(requested)


def validate_action_steps(method, n_action_steps) -> None:
    """Refuse an ``n_action_steps`` that would cut a B-spline parameter matrix short.

    For an ordinary policy the chunk is a sequence: you execute a prefix and replan
    before it runs out, so any ``n_act`` up to the horizon is meaningful. A B-spline
    chunk is not a sequence -- its rows are one curve's knots and control points, and
    a prefix of them decodes a different, shorter curve. The only correct value is the
    full matrix width, which is also what the checkpoint carries, so leaving it unset
    and setting it to ``width`` are the same run; anything else is not.

    Raising here rather than letting crisp_gym raise later is the same bargain the rest
    of this module makes: fail on the laptop, not after the arm is powered up.

    Historical note: crisp_gym additionally rejected ``n_act >= chunk_size``, which
    made *both* correct choices fail -- the width directly, and "unset" too, because
    ``AsyncLerobotPolicy`` falls back to the checkpoint's own value and validates that.
    That check was strictly stricter than LeRobot (``ACTConfig`` errors only on
    ``n_action_steps > chunk_size``) and redundant beside
    ``n_action_steps <= horizon - n_obs_steps + 1``, so it was removed in crisp_gym
    49aea0e. Deploying against an older pin will still fail there, whatever is set here.
    """
    if getattr(method, "type", "none") != "bspline" or n_action_steps is None:
        return
    width = getattr(method, "width", None)
    if width is not None and int(n_action_steps) != int(width):
        raise ValueError(
            f"n_action_steps={n_action_steps} but this B-spline emits a {width}-row "
            "parameter matrix. Those rows are one curve's knots and control points, not "
            f"a sequence to execute a prefix of: taking {n_action_steps} of them decodes "
            "a different, shorter curve. Leave n_action_steps unset (null) so the "
            "checkpoint's own value is used -- crisp_gym additionally requires "
            "n_act < chunk_size, which the only correct value here cannot satisfy."
        )
