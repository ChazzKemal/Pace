# Idea: fitting B-splines to raw actions at deploy time

**Status: not implemented, and not planned. Recorded because the measurement was
done and the answer is not the one you would guess.** Measured 2026-09-01 on the
`jazzy-lerobot` pixi environment.

## The idea

B-spline is currently a *training-time representation*. The dataset's episodes are fit
offline (`fit_episode`), cut into windows (`chunk_parameters`), and the policy is
trained to predict a parameter matrix; at deploy `BSplineDecodeStep` evaluates that
matrix into actions, and `num_actions` is the speed knob.

The idea is to move the fit to deploy: take the raw action chunk that *any* normally
trained policy emits — ACT, Diffusion, a plain baseline — fit a spline to it online,
and resample fewer points. The speedup would then need no retraining and no B-spline
checkpoint. It would apply to checkpoints that already exist.

## Is it fast enough? Yes, by a wide margin

cart7 → 10-dim spline representation, chunk sizes matching real `n_action_steps`.

| what | p50 | worst seen |
|---|---|---|
| adaptive fit (`fit_episode`), T=32, clean | 0.78 ms | 0.84 ms |
| adaptive fit, T=32, heavy noise | 1.55 ms | 1.62 ms |
| adaptive fit, T=100 | 1.42 ms | 5.13 ms |
| **fixed knot vector, one `make_lsq_spline` solve** | **0.026 ms** | 0.035 ms |
| control period @ 20 Hz | 50 ms | |
| inference budget (`overlap_threshold` × dt) | 100 ms | |

The number that settles it is not the budget but the comparison: `decode_chunk` with
`align_start` costs **0.51 ms** and already runs on the deploy path for every chunk. An
adaptive fit is 1.5x something already shipping in real time, and under 1% of the
inference budget — against a policy forward pass that dominates both.

The cost is also bounded, which matters more than the median for a control loop. The
knot count saturates at scipy's `nest` cap (36 at T=32), so pathological input cannot
make the search diverge: noise at `jerk=0.2`, far past anything real, came in *lower*
than moderate noise, at 1.4 ms.

The expected failure — a one-frame binary gripper edge inside the chunk — did not
materialise. The adaptive search spends knots on the step and meets tolerance:

```
   T  edge  knots  met_tol  grip max err  grip min  grip max
  16     8     16     True        0.0093   -0.0036    1.0093
  32    16     20     True        0.0027   -0.0026    1.0010
```

Overshoot stays under 1% of the commanded range and a 0.5 threshold recovers the edge
exactly.

Caveat on all of it: synthetic smooth motion plus noise, not recorded UR10e data. The
noise sweep makes the shape credible, but a real episode should be measured before
anyone acts on these numbers.

## Why it is not being built

Compute was never the obstacle, so the reasons are about what it would mean.

* **It is a different method wearing the same name.** Today the network is *trained* to
  emit spline parameters, so it has learned a representation that survives resampling.
  Fitting at deploy smooths whatever a normally trained policy happened to emit. That
  is a legitimate idea, but it is a different claim and must not be reported as the
  same one.
* **The speedup would come entirely from resampling fewer points**, which is what PACE
  already does by striding and DemoSpeedup by retiming. The honest framing is a third
  interpolator for a trick the repo has twice, not a new source of speedup.
* **Chunk-local fits give up the global knot vector.** `fit_episode` fits a whole
  episode and `chunk_parameters` windows it, which is exactly what makes the windows
  composable — a mid-episode chunk describes the curve with knots the global fit chose.
  Fitting per chunk online gives every chunk an independent knot vector, agreeing only
  at the seam. `align_start` and the hermite blend would be carrying more weight than
  they were built for.
* **It fits a prediction, not a demonstration.** The offline fit runs on recorded
  ground truth. A deploy-time fit runs on policy output that already contains the
  policy's error, and cannot tell policy noise from intended fast motion.

## If it is ever revisited

Use a **fixed knot vector**, not the adaptive search. On a 16–32 row chunk the search
lands on 16–20 knots — near enough to interpolation that it buys little — and costs
30x. At 0.026 ms the fit could run on every chunk and never appear in a timing budget.

The first experiment worth running is not a benchmark. It is an A/B against PACE on the
same checkpoint, to answer whether this is a different speedup or the same one arrived
at differently. That question is cheap to settle and decides whether any of the rest
matters.
