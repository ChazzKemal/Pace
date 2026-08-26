# robot_stack

Speedup methods for LeRobot policies, shared by simulation and the real UR10e.

| method | sim — LIBERO / xVLA | real — UR10e / ACT, Diffusion |
| --- | --- | --- |
| PACE | eval-time speed modulation | same, via the CRISP controller |
| DemoSpeedup | entropy-guided retiming | same |
| B-spline | `bspline_ee6d` action space | ACT, `chunk_size=1` |

Nothing is implemented yet. This is the skeleton and the pinned dependency.

## The one rule

**LeRobot is a dependency, not a fork.** It is pinned by SHA in `pyproject.toml`
and never edited. Every method here attaches through LeRobot's public registries:

| what we add | how it attaches |
| --- | --- |
| `--method.type=...` | `draccus.ChoiceRegistry` |
| processor steps | `@ProcessorStepRegistry.register()` — serialised into the checkpoint |
| custom policies | `@PreTrainedConfig.register_subclass()` |
| custom optimizers | `@OptimizerConfig.register_subclass()` |

Because processor pipelines are saved beside the weights, a method declared at
training time is rebuilt automatically at inference. Nothing has to be re-specified
on the robot.

## Install

Requires Python >= 3.12 (upstream LeRobot's floor).

```bash
uv sync
```

## The pin

```
huggingface/lerobot @ bf31dd794ffb4f87380aba3912f64421e8352d3c   (2026-08-25)
```

Chosen because it is verified to: ship `xvla`/`rtc`/`wall_x`, load the
`lerobot/xvla-libero` checkpoint unmodified (upstream remaps the older vendored
Florence-2 key layout on load), resolve the policy class for third-party plugins,
and run under numpy 1.26 — which the ROS side needs, since `ros-jazzy` is compiled
against the numpy 1.x ABI.

The real robot will pin **the same SHA** from `crisp_gym`'s pixi manifest, with
`[pypi-options] dependency-overrides = { numpy = ">=1.26,<2" }`. Two environments,
one LeRobot.

## Status

Batch 0 — skeleton and pinned dependency. No methods yet.
