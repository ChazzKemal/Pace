#!/usr/bin/env bash
# Batch 2 GO/NO-GO: reproduce the recorded headline PACE result.
#   config: look4cb + skip2 @ 1.5x   seeds: 42, 1000, 2024   tasks: 0-9   50 episodes
# Reference (fork): SR 92.4 / 92.8 / 92.2 %, ATR ~6.05-6.13 sim s.
# NOTE: upstream xVLA scores ~10pp below the fork on task 1 with PACE disabled, so
# absolute SR is not expected to match there -- compare ours-base vs ours-PACE too.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=egl
for seed in 42 1000 2024; do
  echo "=== seed $seed ==="
  .venv/bin/python -m robot_stack.eval.run_libero \
    --out="outputs/gate_look4cb_skip2_1.5_seed${seed}" \
    --seed="$seed" --tasks=0-9 --n_episodes=50 --batch_size=10 --n_action_steps=32 \
    --method.type=pace --method.max_speed=1.5 --method.action_stride=2 \
    --method.n_lookahead=4 --method.lookahead_agg=cumulative_bending \
    --method.lookahead_target=angle || echo "SEED $seed FAILED"
done
echo "GATE_DONE"
