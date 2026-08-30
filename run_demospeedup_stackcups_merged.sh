#!/bin/bash
# =============================================================================
# DemoSpeedup ACT arms on stackcups_20260829_merged (UR10e, absolute cart7)
# =============================================================================
#   1. ACT baseline      chunk 100, 100k steps  (doubles as the labelling oracle)
#   2. entropy labelling oracle = arm 1  -> outputs/label/stack_cups_merged
#   3. ACT DemoSpeedup   chunk 100 -> 50, pad_mode=zero
#
# This supersedes run_demospeedup_stackcups.sh, which trained on
# stack_cups_20260828 -- 12 episodes / 8875 frames. The merged set is 175
# episodes / 61631 frames, 6.9x larger, with identical feature shapes (action
# cart7, state 6, two cameras), so nothing but the budget and the episode count
# changes. The old queue's ACT DemoSpeedup arm never produced a checkpoint: it
# died on a CUDA OOM while a second training held 14.2GB of the card, which is
# why this one is meant to run with the GPU to itself.
#
# Output dirs and the label dir are deliberately NEW names. Reusing cups_act_base
# would hand done_already a checkpoint trained on the 12-episode set and it would
# skip straight past stage 1, labelling the merged data with an oracle that never
# saw it.
#
# BUDGET: 100k steps at batch 32 = 3.2M samples = 51.9 epochs on 61631 frames.
# The repo's other queues are epoch-matched at ~100 epochs (pickplace 102, the old
# cups run 108); on a 6.9x dataset that would be ~193k steps and ~24h for the two
# arms. 100k is the deliberate trade (user decision 2026-08-30): half the wall
# clock, and it matches the pickplace ACT arms step-for-step so the two tasks'
# ACT arms are comparable in optimizer steps. What parity actually requires is
# that the two arms HERE get the same budget, and they do.
#
# bf16 for both arms, as in the pickplace queue: a numerics change, so it is set
# once here and never varied per-arm. The 0.211 s/step this schedule is costed
# against was measured on the pickplace ACT baseline, which ran bf16.
set -uo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"
DATA_ROOT=${PACE_DATA_ROOT:-$(dirname "$REPO_ROOT")/data}
DATASET_ROOT=${STACK_CUPS_MERGED_ROOT:-$DATA_ROOT/datasets/real/stackcups_20260829_merged}
[ -d "$DATASET_ROOT" ] || { echo "no dataset at $DATASET_ROOT -- set STACK_CUPS_MERGED_ROOT"; exit 1; }
export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python

# The card is 24GB and the disk was at 97% when this queue was written, so both
# get checked up front rather than 6 hours in.
PRUNER=${PACE_PRUNER:-$(dirname "$REPO_ROOT")/prune_checkpoints.py}
KEEP=2
[ -f "$PRUNER" ] || echo "WARNING: no pruner at $PRUNER (set PACE_PRUNER); keeping every checkpoint"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dcs '0-9' '\n')
[ "${FREE_GB:-0}" -ge 8 ] || { echo "only ${FREE_GB}GB free on $(pwd) -- an ACT checkpoint is 0.6GB; free space first"; exit 1; }

DATA=(--dataset.repo_id=local/stack_cups_merged
      "--dataset.root=$DATASET_ROOT"
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project=pace_benchmark_stack_cups)
STEPS=100000
BUDGET=(--batch_size=32 --steps=$STEPS --save_freq=10000 --log_freq=100
        --num_workers=4 --seed=42 --policy.device=cuda --policy.push_to_hub=false
        --accelerator.mixed_precision=bf16)
ACT=(--policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100)
N_EPISODES=175

mkdir -p logs outputs/label
stage () { echo; echo "═══════════ $1 ═══════════"; }

# Existence of checkpoints/last is NOT proof the arm finished. save_freq is 10k,
# so a run that dies at step 40k leaves a `last` behind; the old check would then
# report "already trained" and hand the comparison a 40k arm sitting next to a
# 100k one. Resolve the symlink and insist on the full budget. 10# because the
# dirs are zero-padded and bash reads a leading 0 as octal (010000 -> 4096).
done_already () {
    local last="outputs/train/$1/checkpoints/last" step
    if [ -d "$last" ]; then
        step=$(basename "$(readlink -f "$last")")
        if [ "$((10#$step))" -ge "$STEPS" ]; then
            echo "$1 already trained ($((10#$step)) steps), skipping"; return 0
        fi
        echo "$1 stopped at $((10#$step))/$STEPS steps -- crashed attempt, retraining fresh"
    fi
    rm -rf "outputs/train/$1"; return 1
}

train () {  # train <output name> <wandb job name> <policy args...>
    local name=$1 job=$2; shift 2
    done_already "$name" && return 0
    local pruner_pid=
    if [ -f "$PRUNER" ]; then
        "$PY" "$PRUNER" "outputs/train/$name" --keep "$KEEP" --interval 300 \
            >"logs/${name}.prune.log" 2>&1 &
        pruner_pid=$!
    fi
    "$PY" -m pace_bench.train.run_train "${DATA[@]}" "${WANDB[@]}" "${BUDGET[@]}" "$@" \
        --job_name="$job" --output_dir="outputs/train/$name" \
        2>&1 | tee "logs/${name}.log"
    [ -n "$pruner_pid" ] && { kill "$pruner_pid" 2>/dev/null; "$PY" "$PRUNER" "outputs/train/$name" --keep "$KEEP" --once >>"logs/${name}.prune.log" 2>&1; }
    done_already "$name" >/dev/null || { echo "FAILED: $name did not reach $STEPS steps"; exit 1; }
}

stage "1/3: ACT baseline (also the labelling oracle)"
train cups_merged_act_base act_baseline_merged "${ACT[@]}" --method.type=none

stage "2/3: entropy labelling (ACT CVAE oracle = arm 1)"
if [ "$(ls outputs/label/stack_cups_merged/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq "$N_EPISODES" ]; then
    echo "labels already present, skipping"
else
    "$PY" -m pace_bench.methods.demospeedup.run_label \
        --policy_path="$REPO_ROOT/outputs/train/cups_merged_act_base/checkpoints/last/pretrained_model" \
        --dataset_repo_id=local/stack_cups_merged \
        --dataset_root="$DATASET_ROOT" \
        --num_action_samples=10 --temporal_aggregation=true --kde_bandwidth=1.0 \
        --min_cluster_size=5 --max_cluster_size=25 --rule=mean \
        --out=outputs/label/stack_cups_merged \
        2>&1 | tee logs/cups_merged_label.log
fi
N=$(ls outputs/label/stack_cups_merged/speedup_labels/episode_*.npy 2>/dev/null | wc -l)
[ "$N" -eq "$N_EPISODES" ] || { echo "LABEL STAGE FAILED: $N/$N_EPISODES label files"; exit 1; }

# Is the label field structured, or noise with the right marginal? Precision
# frames should come in runs; if the mean run is no longer than a coin flip with
# the same rate would give, the retiming is not tracking the demonstration.
# The heredoc binds to $PY, not to the pipeline's last command -- written after
# `tee` it would feed the source to tee and leave python reading /dev/null.
"$PY" - outputs/label/stack_cups_merged/speedup_labels <<'PYEOF' 2>&1 | tee logs/cups_merged_label_signal.log
import glob
import sys

import numpy as np

labs = [np.load(f) for f in sorted(glob.glob(f"{sys.argv[1]}/episode_*.npy"))]
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

stage "3/3: ACT DemoSpeedup (chunk 100 -> 50, masked zero-pad)"
train cups_merged_act_speedup act_demospeedup_merged "${ACT[@]}" \
    --method.type=demospeedup \
    --method.labels_path="$REPO_ROOT/outputs/label/stack_cups_merged/speedup_labels" \
    --method.pad_mode=zero

echo
echo "═══════════ STACK CUPS MERGED QUEUE DONE ═══════════"
for d in cups_merged_act_base cups_merged_act_speedup; do
    printf '  %-32s %s\n' "$d" "$([ -d "outputs/train/$d/checkpoints/last" ] && echo "trained ($(basename "$(readlink -f "outputs/train/$d/checkpoints/last")"))" || echo MISSING)"
done
