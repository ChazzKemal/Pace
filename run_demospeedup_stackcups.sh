#!/bin/bash
# =============================================================================
# DemoSpeedup pipeline on stack_cups_20260828 (UR10e, absolute cart7 actions)
# =============================================================================
# 1. ACT baseline (doubles as the stage-2 proxy)   pace_bench env, chunk 100
# 2. entropy labelling                             pace_bench env, ACT CVAE oracle
# 3. DemoSpeedup ACT (chunk 100 -> 50)             pace_bench env, pad_mode=zero
# 4. Diffusion baseline (also its own DP oracle)   pace_bench env, n_obs_steps=1
#
# Every stage runs in this repo's env. Labelling used to shell out to the
# lerobot_uncertainty fork under conda, against a checkpoint copy with config keys
# stripped to satisfy the fork's older draccus; pace_bench.methods.demospeedup.run_label removed
# both the fork and the copy.
#
# Steps: 30k (~100 epochs on 8875 frames) rather than the cart7 recipe's literal
# 100k, which was 103 epochs on its 3.5x larger dataset -- matched epoch budget,
# not matched step count. Other hyperparameters mirror examples/28 (ACT chunk
# 100, batch 32) and examples/25 (diffusion batch 64).
#
# wandb: project pace_benchmark_<task>, run named <policy>_<addon> --
# baselines are <policy>_baseline. Run names are deliberately independent of
# the output dirs below, so renaming a run never orphans a checkpoint.
set -uo pipefail
cd /home/batur/Coding/pace_bench
export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA=(--dataset.repo_id=local/stack_cups
      --dataset.root=/home/batur/Coding/data/stack_cups_20260828
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project=pace_benchmark_stack_cups)
mkdir -p logs outputs/label

stage () { echo; echo "═══════════ $1 ═══════════"; }
done_already () { [ -d "outputs/train/$1/checkpoints/last" ] && { echo "$1 already trained, skipping"; return 0; }
    # A leftover dir from a crashed attempt has no checkpoint; upstream's validate
    # refuses to reuse it, so clear it and train fresh.
    rm -rf "outputs/train/$1"; return 1; }

stage "1: ACT baseline / proxy"
done_already cups_act_base || "$PY" -m pace_bench.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=act_baseline --output_dir=outputs/train/cups_act_base \
    2>&1 | tee logs/cups_act_base.log
[ -d outputs/train/cups_act_base/checkpoints/last ] || { echo "STAGE1 FAILED"; exit 1; }

stage "2: entropy labelling (ACT CVAE oracle)"
if [ "$(ls outputs/label/stack_cups/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq 12 ]; then
    echo "labels already present, skipping"
else
"$PY" -m pace_bench.methods.demospeedup.run_label \
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
done_already cups_act_speedup || "$PY" -m pace_bench.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --method.type=demospeedup \
    --method.labels_path="$PWD/outputs/label/stack_cups/speedup_labels" \
    --method.pad_mode=zero \
    --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=act_demospeedup --output_dir=outputs/train/cups_act_speedup \
    2>&1 | tee logs/cups_act_speedup.log
[ -d outputs/train/cups_act_speedup/checkpoints/last ] || { echo "STAGE4 FAILED"; exit 1; }

stage "4: Diffusion baseline (n_obs_steps=1, so it is also its own oracle)"
# batch 32 x 60k, not the recipe's 64 x 30k: same sample budget, but batch-64
# activations extrapolate past this 24GB card (smoke: 8.4GB at batch 8, ~4GB of
# it activations) -- an unattended 3AM OOM is not a hyperparameter.
#
# n_obs_steps=1 overrides LeRobot's default of 2, for two reasons that agree.
#
# Upstream DemoSpeedup gives DP a single observation frame in both instantiations.
# robobase (the one that has priority) sets `frame_stack: 1` in the DP launch
# config its README names, and has no per-policy obs-step notion at all: history
# would be a FrameStack env wrapper whose time axis is folded into channels
# before the encoder (cfgs/launch/dp_pixel_bigym.yaml, method/utils.py:86). aloha
# agrees explicitly -- `n_obs_steps: 1  # Now only support for 1 obs_step`, with
# the time-axis slice commented out of the policy itself
# (act/image_aloha_diffusion_policy_cnn.yaml:27,
# diffusion_unet_hybrid_image_policy.py:177).
#
# And labelling here asks about one frame at a time with no history to give, so
# DiffusionChunkSampler rejects an n_obs_steps=2 checkpoint outright -- at the
# default this baseline could not double as the DP oracle the way the ACT
# baseline does. (Upstream never hits that: robobase labels by replaying each
# demo through a wrapped env, so the wrapper supplies whatever history the config
# asked for.)
#
# Note the trade: the wider DP literature conditions on 2 frames, so this is a
# slightly weaker policy than a stock DP -- it is upstream's own choice, not ours.
done_already cups_diffusion_base || "$PY" -m pace_bench.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=diffusion --policy.n_obs_steps=1 --policy.device=cuda --policy.push_to_hub=false \
    --batch_size=32 --steps=60000 --save_freq=20000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=diffusion_baseline --output_dir=outputs/train/cups_diffusion_base \
    2>&1 | tee logs/cups_diffusion_base.log
[ -d outputs/train/cups_diffusion_base/checkpoints/last ] || { echo "STAGE5 FAILED"; exit 1; }

echo "═══════════ PIPELINE DONE ═══════════"
