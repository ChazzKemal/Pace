#!/bin/bash
# =============================================================================
# DemoSpeedup on pickplace_cart7_v2_angleaxis_nogrip (UR10e, 45 eps / 31k frames)
# =============================================================================
# Everything trains fresh in THIS stack (user decision 2026-08-29): the old
# Yunfei checkpoint is not used -- as oracle it would label with a policy from a
# different training stack, and as A-arm it would compare across stacks.
#
#   1. ACT baseline (doubles as the labelling proxy), 100k steps (~103 epochs)
#   2. entropy labelling (this baseline as CVAE oracle) + label-signal check
#   3. DemoSpeedup ACT from scratch, chunk 100 -> 50, pad_mode=zero, 100k steps
#
# All in-repo: labelling used to run in the lerobot_uncertainty fork under conda,
# against a checkpoint copy with config keys stripped for the fork's older draccus.
# robot_stack.label.run_label replaced both.
set -uo pipefail
cd /home/batur/Coding/robot_stack
export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA=(--dataset.repo_id=local/pickplace
      --dataset.root=/home/batur/Coding/data/pickplace_cart7_v2_angleaxis_nogrip
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project=demospeedup-pickplace)
mkdir -p logs outputs/label
stage () { echo; echo "═══════════ $1 ═══════════"; }
done_already () { [ -d "outputs/train/$1/checkpoints/last" ] && { echo "$1 already trained, skipping"; return 0; }
    # A leftover dir from a crashed attempt has no checkpoint; upstream's validate
    # refuses to reuse it, so clear it and train fresh.
    rm -rf "outputs/train/$1"; return 1; }

stage "1: ACT baseline / proxy (fresh, this stack)"
done_already pickplace_act_base || "$PY" -m robot_stack.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --batch_size=32 --steps=100000 --save_freq=20000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=pickplace_act_base --output_dir=outputs/train/pickplace_act_base \
    2>&1 | tee logs/pickplace_act_base.log
[ -d outputs/train/pickplace_act_base/checkpoints/last ] || { echo "STAGE1 FAILED"; exit 1; }

stage "2: entropy labelling (oracle = the stage-1 baseline)"
if [ "$(ls outputs/label/pickplace/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq 45 ]; then
    echo "labels already present, skipping"
else
    "$PY" -m robot_stack.label.run_label \
        --policy_path="$PWD/outputs/train/pickplace_act_base/checkpoints/last/pretrained_model" \
        --dataset_repo_id=local/pickplace \
        --dataset_root=/home/batur/Coding/data/pickplace_cart7_v2_angleaxis_nogrip \
        --num_action_samples=10 --temporal_aggregation=true --kde_bandwidth=1.0 \
        --min_cluster_size=5 --max_cluster_size=25 --rule=mean \
        --out=outputs/label/pickplace \
        2>&1 | tee logs/pickplace_label.log
fi
N=$(ls outputs/label/pickplace/speedup_labels/episode_*.npy 2>/dev/null | wc -l)
[ "$N" -eq 45 ] || { echo "LABEL STAGE FAILED: $N/45 files"; exit 1; }

echo "═══════════ label signal check ═══════════"
"$PY" - <<'PYEOF' 2>&1 | tee logs/pickplace_label_signal.log
import glob
import numpy as np
labs = [np.load(f) for f in sorted(glob.glob("outputs/label/pickplace/speedup_labels/episode_*.npy"))]
allc = np.concatenate(labs)
frac = allc.mean()
runs = []
for lab in labs:
    n = 0
    for v in lab:
        if v == 1:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
mean_run = float(np.mean(runs)) if runs else 0.0
expected_random = 1.0 / (1.0 - frac) if frac < 1 else float("inf")
print(f"episodes {len(labs)}  frames {len(allc)}  non-precision {frac:.1%}")
print(f"mean fast-run length {mean_run:.2f} vs {expected_random:.2f} expected if random")
print("SIGNAL:", "OK" if mean_run > 3 * expected_random else "SUSPICIOUS - review plots before trusting")
PYEOF

stage "3: DemoSpeedup ACT (chunk 100 -> 50, 100k steps)"
done_already pickplace_act_speedup || "$PY" -m robot_stack.train.run_train "${DATA[@]}" "${WANDB[@]}" \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda --policy.push_to_hub=false \
    --method.type=demospeedup \
    --method.labels_path="$PWD/outputs/label/pickplace/speedup_labels" \
    --method.pad_mode=zero \
    --batch_size=32 --steps=100000 --save_freq=20000 --log_freq=100 --num_workers=4 --seed=42 \
    --job_name=pickplace_act_speedup --output_dir=outputs/train/pickplace_act_speedup \
    2>&1 | tee logs/pickplace_act_speedup.log
[ -d outputs/train/pickplace_act_speedup/checkpoints/last ] || { echo "STAGE4 FAILED"; exit 1; }
echo "═══════════ PICKPLACE PIPELINE DONE ═══════════"
