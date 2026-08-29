# DemoSpeedup on `stack_cups` — overnight run, 2026-08-28

Five-stage pipeline running in tmux session **`dscups`** (`tmux attach -t dscups`),
launched ~22:48. Everything lands under `robot_stack/outputs/`; per-stage logs in
`robot_stack/logs/cups_*.log`. WandB project:
[demospeedup-stackcups](https://wandb.ai/colors-chazz/demospeedup-stackcups).

## The dataset

`/home/batur/Coding/data/stack_cups_20260828` — 12 episodes, 8 875 frames @ 20 fps,
two cameras (`camera`, `d405`), action `[x y z rx ry rz gripper]` (angle-axis).
Verified **absolute** cartesian poses (actions track state within ~4 cm, ~1.7 mm/step)
— the hard requirement for DemoSpeedup, whose retiming deletes frames so surviving
waypoints absorb the motion. Small set: expect a modest policy, not a broken pipeline.

## Stages

| # | what | env | output | ETA |
|---|---|---|---|---|
| 1 | ACT baseline (doubles as stage-2 proxy) — chunk 100, batch 32, 30k steps | pace_bench | `outputs/train/cups_act_base` | ~2h10 |
| 2 | Fork-compatible checkpoint copy (strips `use_peft`, `pretrained_revision` — the fork's strict draccus rejects them) | — | `.../cups_act_base/forkcompat` | s |
| 3 | Entropy labelling — fork's `lerobot-label`, ACT CVAE oracle (`sample_action_chunks`, 10 samples/step, temporal aggregation, KDE + HDBSCAN) | fork (conda `lerobot`) | `outputs/label/stack_cups/speedup_labels/episode_{0..11}.npy` + plots | ~30 min |
| 4 | DemoSpeedup ACT — chunk 100→50, `pad_mode=zero` (ACT's loss is masked by `action_is_pad`) | pace_bench | `outputs/train/cups_act_speedup` | ~2h |
| 5 | Diffusion baseline — batch 64 | pace_bench | `outputs/train/cups_diffusion_base` | ~2h |

Each stage gates on the previous one (stage 3 additionally checks 12/12 label files);
a failure prints `STAGE<N> FAILED` and stops the chain. `PIPELINE_EXIT=0` in the tmux
pane means everything finished.

Steps are 30k, not the cart7 recipe's 100k: that recipe was ~103 epochs on a 3.5×
larger dataset; 30k matches the **epoch budget** (~100) on 8 875 frames. All other
hyperparameters mirror `examples/28` (ACT) and `examples/25` (diffusion).

## What the labels are

`episode_<i>.npy`: one int array per episode, length = that episode's frame count,
values `0` = precision (retiming strides 2) / `1` = non-precision (strides 4).
`entropy_<i>.npy` and `plots/` are diagnostics only. Training loads them with
`--method.type=demospeedup --method.labels_path=outputs/label/stack_cups/speedup_labels`
and validates label length against every episode at startup — wrong-dataset labels
fail loudly, they never silently retime the wrong frames.

## Context: the same method on LIBERO-10 (sim, xVLA), evaluated yesterday

20 episodes/task, seed 42; baseline vs DemoSpeedup finetune of the same pretrained xVLA:

| | SR (mean) | time-to-success | speedup |
|---|---|---|---|
| baseline | 92.0 % | 13.28 s | — |
| DemoSpeedup | 86.5 % | 6.99 s | **1.90×** |

7/10 tasks within noise on SR; the one real casualty is task 2 (100→70), a known
speed-intolerant task. Full results: `outputs/eval/ds_libero10_*/task_*/eval_info.json`.

## Next steps (not automated)

- Real-robot eval of the stack_cups policies needs the deploy-day items: the
  fake-mode dataset diff on the lab machine, and the **gripper postprocessors** —
  the real gripper's stroke rate is fixed hardware, so during gripper motion PACE
  must brake to speed 1 and DemoSpeedup must repeat actions ×`low_v` (negating the
  unrealizable gripper speedup); arm-gain scaling (`kp ∝ v²`, `kd ∝ v`) works IRL
  and stays active.
- **Pick-and-place is queued behind this chain** (tmux `dspick`, starts when
  `dscups` exits): the old 100k-step ACT baseline serves directly as the labelling
  oracle (verified: the fork loads it and CVAE-samples), so its proxy stage is
  skipped — labelling (45 eps, with an automatic label-signal check in
  `logs/pickplace_label_signal.log`) then DemoSpeedup ACT from scratch at **100k**
  steps, matching the baseline's budget for a fair A/B. Done ~midday. The old
  checkpoint doubles unchanged as the A-arm for any later eval; the old
  disk-converted `demospeedup_act` is a bonus third arm comparing the retired
  episode-level formulation against on-the-fly retiming.
