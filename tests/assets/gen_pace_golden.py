"""Generate PACE golden vectors by driving the FORK's real XVLAPolicy.select_action.

Run under the conda `lerobot` env (the fork). No GPU, no checkpoint: the only
thing stubbed out is the network forward (`_get_action_chunk`), which is replaced
by a fixed pseudo-random chunk. Everything downstream of it -- stride selection,
unnormalisation, the three speed channels, lookahead, the min-combine, the
bang-bang override, and the queue slicing -- is the fork's own code.
"""

import json
import sys
from collections import deque

import numpy as np
import torch

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import ACTION

# PACE never landed upstream -- porting it is the point of this repo. So an
# XVLAPolicy without the speed methods means the wrong interpreter: upstream
# LeRobot (robot_stack's .venv) rather than the fork. Say so here, instead of
# letting it surface later as an AttributeError on `add_speed_configs`.
if not hasattr(XVLAPolicy, "add_speed_configs"):
    raise RuntimeError(
        f"This XVLAPolicy ({XVLAPolicy.__module__}) has no PACE speed methods -- "
        "you are running upstream LeRobot. Golden vectors must come from the fork:\n"
        "  ~/miniconda3/envs/lerobot/bin/python tests/assets/gen_pace_golden.py <out.npz>"
    )

SPEED = "speed"
CHUNK, DIM, N_ACTION_STEPS = 32, 7, 16
OUT = sys.argv[1] if len(sys.argv) > 1 else "pace_golden.npz"


class _Cfg:
    n_action_steps = N_ACTION_STEPS
    device = "cpu"


class Harness(XVLAPolicy):
    """XVLAPolicy with the network removed. Speed logic untouched."""

    def __init__(self, chunk, stats):
        torch.nn.Module.__init__(self)
        self.config = _Cfg()
        self._chunk = chunk
        self.dataset_stats = stats
        self._queues = {ACTION: deque(maxlen=N_ACTION_STEPS), SPEED: deque(maxlen=N_ACTION_STEPS)}

    def eval(self):
        return self

    def _get_action_chunk(self, batch):
        return self._chunk


def make_chunk(seed):
    """A chunk with realistic structure: a smooth arc, a sharp corner, a pause."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, CHUNK)
    pos = torch.stack([torch.sin(3 * t), torch.cos(3 * t), t], dim=-1)
    pos[CHUNK // 2 :] *= -1.0  # hard direction reversal mid-chunk
    pos[CHUNK - 6 :] = pos[CHUNK - 6]  # dwell at the end
    ori = 0.3 * torch.stack([t, torch.sin(5 * t), torch.zeros_like(t)], dim=-1)
    rest = 0.1 * torch.rand(CHUNK, DIM - 6, generator=g)
    traj = torch.cat([pos, ori, rest], dim=-1)
    traj = traj + 0.02 * torch.randn(CHUNK, DIM, generator=g)
    return traj.unsqueeze(0)  # (1, CHUNK, DIM)


# Speed configs: the ablation grid's real corners, plus the degenerate branches.
# Kept one-per-line (fmt: off) so the grid reads as a table of what varies.
CB, LA = "cumulative_bending", "lookahead_agg"
# fmt: off
CONFIGS = {
    "const_1.5":        dict(max_speed=1.5, min_speed=1.5, enable_angle=False, enable_ori=False, enable_ori_axis=False),
    "angle_only":       dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, enable_angle=True, enable_ori=False, enable_ori_axis=False),
    "angle_ori":        dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "all_three":        dict(max_speed=2.0, min_speed=1.0, clamp_deg=5.0, enable_angle=True, enable_ori=True, enable_ori_axis=True),
    "look4_min":        dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, n_lookahead=4, **{LA: "min"}, lookahead_target="angle", enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "look2_mean_all":   dict(max_speed=2.0, min_speed=1.0, clamp_deg=5.0, n_lookahead=2, **{LA: "mean"}, lookahead_target="all", enable_angle=True, enable_ori=True, enable_ori_axis=True),
    "look4_cb":         dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, n_lookahead=4, **{LA: CB}, lookahead_target="angle", enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "look8_cb_oriaxis": dict(max_speed=3.0, min_speed=1.5, clamp_deg=5.0, n_lookahead=8, **{LA: CB}, lookahead_target="ori_axis", enable_angle=True, enable_ori=True, enable_ori_axis=True),
    "headline_look4cb_skip2": dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, action_stride=2, n_lookahead=4, **{LA: CB}, lookahead_target="angle", enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "skip3_only":       dict(max_speed=1.5, min_speed=1.5, action_stride=3, enable_angle=False, enable_ori=False, enable_ori_axis=False),
    "adaptive_stride2": dict(max_speed=1.5, min_speed=0.75, clamp_deg=5.0, action_stride=2, adaptive_stride=True, quantize_angle_thr=22.5, enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "quantize":         dict(max_speed=2.0, min_speed=1.0, clamp_deg=5.0, speed_quantize=True, quantize_angle_thr=22.5, enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "clamp_deg_zero":   dict(max_speed=2.0, min_speed=1.0, clamp_deg=0.0, enable_angle=True, enable_ori=True, enable_ori_axis=False),
    "all_disabled":     dict(max_speed=2.5, min_speed=1.0, enable_angle=False, enable_ori=False, enable_ori_axis=False),
    "defaults_only":    dict(max_speed=1.5),   # exercises every .get() default in the fork
}
# fmt: on

# Two normalisation regimes, since _unnormalize_actions branches on which is present.
STATS = {
    "meanstd": {"action": {"mean": torch.linspace(-0.5, 0.5, DIM), "std": torch.linspace(0.5, 2.0, DIM)}},
    "minmax": {"action": {"min": -torch.ones(DIM) * 1.5, "max": torch.ones(DIM) * 2.5}},
    "none": {},
}

out = {}
for seed in (0, 7):
    chunk = make_chunk(seed)
    out[f"chunk__{seed}"] = chunk.numpy()
    for stats_name, stats in STATS.items():
        for cfg_name, cfg in CONFIGS.items():
            h = Harness(chunk, stats)
            h.add_speed_configs(cfg)
            acts, speeds = [], []
            while True:
                a = h.select_action({})
                acts.append(a)
                speeds.append(h.select_speed())
                if not h._queues[ACTION]:
                    break
            key = f"{seed}__{stats_name}__{cfg_name}"
            out[f"act__{key}"] = torch.stack(acts).numpy()
            out[f"spd__{key}"] = torch.stack(speeds).numpy()

np.savez(OUT, **out)
meta = {
    "chunk": CHUNK,
    "dim": DIM,
    "n_action_steps": N_ACTION_STEPS,
    "seeds": [0, 7],
    "stats": list(STATS),
    "configs": CONFIGS,
    "source": "fork lerobot_uncertainty XVLAPolicy.select_action",
    "torch": torch.__version__,
}
with open(OUT.replace(".npz", ".json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"wrote {len(out)} arrays over {len(STATS) * len(CONFIGS) * 2} cases")
