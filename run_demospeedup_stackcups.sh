#!/bin/bash
# =============================================================================
# DemoSpeedup pipeline on stack_cups_20260828 (UR10e, absolute cart7 actions)
# =============================================================================
# 1. ACT baseline (doubles as the stage-2 proxy)   robot_stack env, chunk 100
# 2. entropy labelling                             robot_stack env, ACT CVAE oracle
# 3. DemoSpeedup ACT (chunk 100 -> 50)             robot_stack env, pad_mode=zero
# 4. Diffusion baseline                            robot_stack env
#
# Every stage runs in this repo's env. Labelling used to shell out to the
# lerobot_uncertainty fork under conda, against a checkpoint copy with config keys
# stripped to satisfy the fork's older draccus; robot_stack.methods.demospeedup.run_label removed
# both the fork and the copy.
#
# Steps: 30k (~100 epochs on 8875 frames) rather than the cart7 recipe's literal
# 100k, which was 103 epochs on its 3.5x larger dataset -- matched epoch budget,
# not matched step count. Other hyperparameters mirror examples/28 (ACT chunk
# 100, batch 32) and examples/25 (diffusion batch 64).
set -uo pipefail
cd /home/batur/Coding/robot_stack
export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA=(--dataset.repo_id=local/stack_cups
      --dataset.root=/home/batur/Coding/data/stack_cups_20260828
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project=demospeedup-stackcups)
mkdir -p logs outputs/label

stage () { echo; echo "═══════════ $1 ═══════════"; }
done_already () { [ -d "outputs/train/$1/checkpoints/last" ] && { echo "$1 already trained, skipping"; return 0; }
    # A leftover dir from a crashed attempt has no checkpoint; upstream's validate
    # refuses to reuse it, so clear it and train fresh.
    rm -rf "outputs/train/$1"; return 1; }

stage "1: ACT baseline / proxy"
done_already cups_act_base || "$PY" -m robot_stack.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=cups_act_base --output_dir=outputs/train/cups_act_base \
    2>&1 | tee logs/cups_act_base.log
[ -d outputs/train/cups_act_base/checkpoints/last ] || { echo "STAGE1 FAILED"; exit 1; }

stage "2: entropy labelling (ACT CVAE oracle)"
if [ "$(ls outputs/label/stack_cups/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq 12 ]; then
    echo "labels already present, skipping"
else
"$PY" -m robot_stack.methods.demospeedup.run_label \
    --policy_path="$PWD/outputs/train/cups_act_base/checkpoints/last/pretrained_model" \
    --dataset_repo_id=local/stack_cups \
    --dataset_root=/home/batur/Coding/data/stack_cups_20260828 \
    --num_action_samples=10 --temporal_aggregation=true --kde_bandwidth=1.0 \
    --min_cluster_size=5 --max_cluster_size=25 --rule=mean \
    --out=outputs/label/stack_cups \
    2>&1 | tee logs/cups_label.log
fi
N=$(ls outputs/label/stack_cups/speedup_labels/episode_*.npy 2>/dev/null | wc -l)
[ "$N" -eq 12 ] || { echo "STAGE2 FAILED: $N/12 label files"; exit 1; }

stage "3: DemoSpeedup ACT (chunk 100 -> 50, masked zero-pad)"
done_already cups_act_speedup || "$PY" -m robot_stack.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --method.type=demospeedup \
    --method.labels_path="$PWD/outputs/label/stack_cups/speedup_labels" \
    --method.pad_mode=zero \
    --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=cups_act_speedup --output_dir=outputs/train/cups_act_speedup \
    2>&1 | tee logs/cups_act_speedup.log
[ -d outputs/train/cups_act_speedup/checkpoints/last ] || { echo "STAGE4 FAILED"; exit 1; }

stage "4: Diffusion baseline"
# batch 32 x 60k, not the recipe's 64 x 30k: same sample budget, but batch-64
# activations extrapolate past this 24GB card (smoke: 8.4GB at batch 8, ~4GB of
# it activations) -- an unattended 3AM OOM is not a hyperparameter.
done_already cups_diffusion_base || "$PY" -m robot_stack.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --batch_size=32 --steps=60000 --save_freq=20000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=cups_diffusion_base --output_dir=outputs/train/cups_diffusion_base \
    2>&1 | tee logs/cups_diffusion_base.log
[ -d outputs/train/cups_diffusion_base/checkpoints/last ] || { echo "STAGE5 FAILED"; exit 1; }

echo "═══════════ PIPELINE DONE ═══════════"
