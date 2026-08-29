#!/bin/bash
# Evaluate the two finetunes from run_demospeedup_libero10.sh on LIBERO-10.
#
# Nothing modulates speed at inference; the speedup arm executes 15 steps per
# query against the baseline's 30 because its waypoints sit 2-4x further apart.
# Arm B declares --method.type=demospeedup so the runner applies the constant
# tracking bump (gripper stroke rate and OSC kp/kd scaled by low_v=2, time
# untouched) -- the analogue of upstream DemoSpeedup's eval-time high-gain XMLs.
set -euo pipefail
cd /home/batur/Coding/pace_bench
export MUJOCO_GL=egl PYTHONUNBUFFERED=1

PY=.venv/bin/python
COMMON=(--use_peft=true --task_suite=libero_10 --tasks=0-9
        --seed=42 --n_episodes=20 --batch_size=10)

"$PY" -m pace_bench.eval.run_libero \
    --policy_path=outputs/train/ds_libero10_base/checkpoints/last/pretrained_model \
    --n_action_steps=30 --out=outputs/eval/ds_libero10_base \
    --method.type=none "${COMMON[@]}" \
    2>&1 | tee logs/eval_ds_libero10_base.log

"$PY" -m pace_bench.eval.run_libero \
    --policy_path=outputs/train/ds_libero10_speedup/checkpoints/last/pretrained_model \
    --n_action_steps=15 --out=outputs/eval/ds_libero10_speedup \
    --method.type=demospeedup --method.low_v=2 --method.high_v=4 "${COMMON[@]}" \
    2>&1 | tee logs/eval_ds_libero10_speedup.log
