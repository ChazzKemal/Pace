#!/bin/bash
# =============================================================================
# DemoSpeedup end-to-end on LIBERO-10, xVLA / absolute EE6D
# =============================================================================
# Two LoRA finetunes of the same pretrained xVLA on the same 400 demos, run back
# to back so the only difference between them is the retiming step:
#
#   A  method=none         chunk 30                 demos at their recorded pace
#   B  method=demospeedup  window 60 -> chunk 15    targets retimed 2x/4x by label
#
# Both PASS chunk 30. For B the trainer widens the dataset's raw action window to
# 60 = 15 * high_v BEFORE the dataset is built and sets the trained chunk to 15
# after -- upstream's walk-the-tail-then-truncate semantics on a fixed-window
# loader: every executed slot is a real waypoint except at episode ends. (The
# fork's 2x window under-supplied the walk; ~7% of its executed steps were
# trained dwells.) Both arms are evaluated with PACE off, so any difference in
# time-to-success comes from what the policy learned.
#
# Labels come from the fork's stage-2 run (entropy of the pretrained xVLA's own
# action samples); stage 2 is not ported yet, so they are consumed as given.
# =============================================================================
set -euo pipefail
cd /home/batur/Coding/pace_bench
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 VIDEO_BACKEND=pyav

PY=.venv/bin/python
DATASET=(--dataset.repo_id=local/libero_10_ee6d
         --dataset.root=/home/batur/libero_ee6d/libero_10_ee6d
         --dataset.video_backend=pyav)
# LoRA on all linear layers; the action heads are DomainAwareLinear (nn.Embedding),
# which LoRA cannot target, so they are fully trained instead.
PEFT=(--peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 --peft.target_modules=all-linear
      --peft.full_training_modules='["transformer.soft_prompt_hub","transformer.action_encoder","transformer.action_decoder"]')
# alpha/rank = 1.0 and lr 1e-5: flow matching compounds adapter error over 10
# denoising steps, so both are held low.
COMMON=(--policy.path=/home/batur/xvla_libero_patched
        --policy.device=cuda --policy.push_to_hub=false --policy.optimizer_lr=1e-5
        --batch_size=8 --steps=20000 --save_freq=5000 --log_freq=100
        --num_workers=4 --seed=42
        --wandb.enable=true --wandb.project="${WANDB_PROJECT:-pace_benchmark_libero10}")

run () {  # run <dir> <wandb run name> <extra args...>
    local name=$1 job=$2; shift 2
    if [[ -d "outputs/train/${name}/checkpoints/last" ]]; then
        echo "=== ${name} already trained, skipping ==="
        return 0
    fi
    echo "=== training ${name} ==="
    # --job_name names the wandb run: <policy>_<addon>, per the project convention
    # (project pace_benchmark_<task>). It is kept separate from the output dir so
    # renaming runs never orphans an existing checkpoint.
    "$PY" -m pace_bench.train.run_train \
        "${DATASET[@]}" "${PEFT[@]}" "${COMMON[@]}" "$@" \
        --output_dir="outputs/train/${name}" --job_name="${job}" \
        2>&1 | tee "logs/${name}.log"
}

run ds_libero10_base xvla_baseline \
    --policy.chunk_size=30 --policy.n_action_steps=30 \
    --method.type=none

run ds_libero10_speedup xvla_demospeedup \
    --policy.chunk_size=30 --policy.n_action_steps=30 \
    --method.type=demospeedup \
    --method.labels_path=/home/batur/lerobot_uncertainty/outputs/label/xvla_libero10_ee6d/speedup_labels \
    --method.pad_mode=hold

echo "=== both trainings done ==="
