#!/bin/bash
# =============================================================================
# DemoSpeedup on pickplace_cart7_v2_angleaxis_nogrip (UR10e, 45 eps / 31k frames)
# =============================================================================
# Everything trains fresh in THIS stack (user decision 2026-08-29): the old
# Yunfei checkpoint is not used -- as oracle it would label with a policy from a
# different training stack, and as A-arm it would compare across stacks.
#
#   1. ACT baseline (doubles as the labelling proxy), 100k steps (~103 epochs)
#   2. fork-compatible copy (strip keys the fork's strict draccus rejects)
#   3. entropy labelling (fork lerobot-label, this baseline as CVAE oracle)
#      + label-signal check
#   4. DemoSpeedup ACT from scratch, chunk 100 -> 50, pad_mode=zero, 100k steps
set -uo pipefail
cd /home/batur/Coding/robot_stack
export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python
FORK_PY=/home/batur/miniconda3/envs/lerobot/bin/python
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

stage "2: fork-compatible proxy copy"
rm -rf outputs/train/pickplace_act_base/forkcompat
cp -r outputs/train/pickplace_act_base/checkpoints/last/pretrained_model outputs/train/pickplace_act_base/forkcompat
python3 - <<'PYEOF'
import json
p = "outputs/train/pickplace_act_base/forkcompat/config.json"
c = json.load(open(p))
for k in ("use_peft", "pretrained_revision"):
    c.pop(k, None)
json.dump(c, open(p, "w"), indent=2)
print("stripped fork-incompatible keys")
PYEOF

stage "3: entropy labelling (oracle = the stage-1 baseline)"
if [ "$(ls outputs/label/pickplace/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq 45 ]; then
    echo "labels already present, skipping"
else
    VIDEO_BACKEND=pyav "$FORK_PY" -m lerobot.scripts.lerobot_label \
        --policy.path="$PWD/outputs/train/pickplace_act_base/forkcompat" \
        "${DATA[@]}" \
        --num_action_samples=10 --temporal_aggregation=true \
        --kde_bandwidth=1.0 --hdbscan_min_cluster_size=5 --hdbscan_max_cluster_size=25 \
        --save_plots=true --output_dir=outputs/label/pickplace \
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

stage "4: DemoSpeedup ACT (chunk 100 -> 50, 100k steps)"
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
