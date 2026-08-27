"""Record how much *simulated* time each episode took.

Success rate alone cannot express what PACE does: a policy that succeeds equally
often but finishes sooner is strictly better, and that difference is the entire
result. The throughput metrics (ATR, TPR, speedup-over-demo) are all built on
per-episode duration measured in simulated seconds -- not wall clock, which would
report GPU speed rather than robot speed.

Upstream LeRobot's evaluator does not surface this, so it is collected here at the
one place that always knows: the environment itself, reading MuJoCo's own clock.
Wrapping the vector env's members keeps upstream's rollout untouched.
"""

from __future__ import annotations

import gymnasium as gym


def _mujoco_time(env) -> float | None:
    """Simulated seconds elapsed, from MuJoCo's own clock.

    Reached the same way the actuator reaches the simulator -- through
    ``.unwrapped``, because gym wrappers do not forward underscore attributes and
    the robosuite env hangs off a private one.
    """
    sim_env = getattr(getattr(env.unwrapped, "_env", None), "env", None)
    data = getattr(getattr(sim_env, "sim", None), "data", None)
    if data is None or not hasattr(data, "time"):
        return None
    try:
        return float(data.time)
    except (TypeError, ValueError):
        return None


class SimTimeRecorder(gym.Wrapper):
    """Log ``(sim_time, success)`` for every episode this env finishes.

    The clock is read on the terminating step, before the auto-reset that would zero
    it. Episodes land in :attr:`episodes` in completion order.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episodes: list[dict] = []
        self._recorded = False

    def arm(self) -> None:
        """Allow one more episode to be recorded. Called at each batch start.

        Deliberately NOT hooked to this env's own ``reset``: within a batch, an env
        that finishes early is auto-reset and keeps stepping, and re-arming there is
        exactly the overcount this guards against.
        """
        self._recorded = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and not self._recorded:
            # Only the first termination of each batch counts. A batch runs until its
            # slowest env finishes, and envs that finish early are auto-reset and keep
            # going -- so without this guard a fast env contributes several episodes
            # while a slow one contributes one: an overcount, and biased towards short
            # episodes. Upstream masks those out by cumulative `done`; this is the
            # same rule enforced where the clock is read.
            self._recorded = True
            self.episodes.append(
                {
                    "sim_time": _mujoco_time(self),
                    # LIBERO terminates on success and truncates on timeout, so
                    # `terminated` is the success flag; reward is kept as a check.
                    "success": bool(terminated),
                    "reward": float(reward),
                }
            )
        return obs, reward, terminated, truncated, info


def wrap_vector_env(vec_env) -> list[SimTimeRecorder]:
    """Install a recorder on each member of a (synchronous) vector env.

    Also intercepts the *vector* env's ``reset`` to re-arm the recorders. That is the
    only signal available for "a new batch of episodes is starting": upstream's
    rollout resets the vector env once per batch, whereas an env auto-resetting
    mid-batch does so through its own ``reset``, which is left alone.

    Returns the recorders in env order, so the caller can read them afterwards.
    """
    if not hasattr(vec_env, "envs"):
        raise TypeError(
            f"{type(vec_env).__name__} exposes no `.envs`; sim-time recording needs a "
            "synchronous vector env whose members can be wrapped in place."
        )
    recorders = [SimTimeRecorder(member) for member in vec_env.envs]
    vec_env.envs = recorders

    original_reset = vec_env.reset

    def reset(*args, **kwargs):
        for rec in recorders:
            rec.arm()
        return original_reset(*args, **kwargs)

    vec_env.reset = reset
    return recorders
