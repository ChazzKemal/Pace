#!/usr/bin/env bash
# Batch 2 GO/NO-GO: reproduce the recorded headline PACE result.
#   config: look4cb + skip2 @ 1.5x   seeds: 42, 1000, 2024   tasks: 0-9   50 episodes
# Reference (fork): SR 92.4 / 92.8 / 92.2 %, ATR ~6.05-6.13 sim s.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=egl
for seed in 42 1000 2024; do
  echo "=== seed $seed ==="
  .venv/bin/python -m robot_stack.eval.run_libero \
    --policy-path lerobot/xvla-libero --seed "$seed" --tasks 0-9 \
    --n-episodes 50 --batch-size 10 --n-action-steps 32 \
    --max-speed 1.5 --min-speed 0.75 --action-stride 2 \
    --n-lookahead 4 --lookahead-agg cumulative_bending --lookahead-target angle \
    --out "outputs/gate_look4cb_skip2_1.5_seed${seed}" || echo "SEED $seed FAILED"
done
echo "GATE_DONE"
