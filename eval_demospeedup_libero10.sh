#!/bin/bash
# Evaluate the two finetunes from run_demospeedup_libero10.sh on LIBERO-10.
#
# PACE is off for both (--method.type=none): the question here is what the policy
# learned, so nothing modulates speed at inference. The DemoSpeedup model executes
# 15 steps per query against the baseline's 30 because its chunk is half as long
# and its waypoints are spaced 2-4x further apart -- the same motion, fewer steps.
set -euo pipefail
cd /home/batur/Coding/robot_stack
export MUJOCO_GL=egl PYTHONUNBUFFERED=1

PY=.venv/bin/python
COMMON=(--use_peft=true --task_suite=libero_10 --tasks=0-9
        --seed=42 --n_episodes=20 --batch_size=10 --method.type=none)

"$PY" -m robot_stack.eval.run_libero \
    --policy_path=outputs/train/ds_libero10_base/checkpoints/last/pretrained_model \
    --n_action_steps=30 --out=outputs/eval/ds_libero10_base "${COMMON[@]}" \
    2>&1 | tee logs/eval_ds_libero10_base.log

"$PY" -m robot_stack.eval.run_libero \
    --policy_path=outputs/train/ds_libero10_speedup/checkpoints/last/pretrained_model \
    --n_action_steps=15 --out=outputs/eval/ds_libero10_speedup "${COMMON[@]}" \
    2>&1 | tee logs/eval_ds_libero10_speedup.log
