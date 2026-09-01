#!/bin/bash
# Evaluate the three finetunes from run_demospeedup_libero10.sh on LIBERO-10.
#
# Nothing modulates speed at inference; the speedup arm executes 15 steps per
# query against the baseline's 30 because its waypoints sit 2-4x further apart.
# Arm B declares --method.type=demospeedup so the runner applies the constant
# tracking bump (gripper stroke rate and OSC kp/kd scaled by low_v=2, time
# untouched) -- the analogue of upstream DemoSpeedup's eval-time high-gain XMLs.
set -euo pipefail
# The repo is this script's own directory; checkpoints and logs are read and
# written relative to it, exactly where run_demospeedup_libero10.sh left them.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
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

# Arm C decodes before it can execute anything: the checkpoint predicts curve
# parameters, not actions. The geometry below is not a preference -- it must be the
# one the checkpoint trained under, or the decode reconstructs a different curve
# from the same numbers. layout/arrangement/fps/chunk_size/degree therefore repeat
# run_demospeedup_libero10.sh exactly.
#
# --method.num_actions=16 is the speed lever, set here rather than inherited: it is a
# decode-time choice needing no retraining (the paper's `a_exec(t) = a(nt)`), and 16 --
# the parameter matrix's own width -- is demonstration pace. It is passed explicitly
# even though it is also the default, because a silent default is a poor way to record
# the one number this arm's whole claim rests on. The *realised* rate varies per chunk
# with the span the policy predicted and is logged as `bspline_rate`; run this arm at
# 8 for a nominal 2x without retraining anything.
#
# n_action_steps=16 matches the matrix width so predict_action_chunk returns a whole
# parameter matrix. The runner drops the checkpoint's own baked-in decode step in
# favour of this one, and logs that it did.
"$PY" -m pace_bench.eval.run_libero \
    --policy_path=outputs/train/ds_libero10_bspline/checkpoints/last/pretrained_model \
    --n_action_steps=16 --out=outputs/eval/ds_libero10_bspline \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01 \
    --method.num_actions=16 "${COMMON[@]}" \
    2>&1 | tee logs/eval_ds_libero10_bspline.log
