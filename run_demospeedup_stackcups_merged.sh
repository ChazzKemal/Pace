#!/bin/bash
# =============================================================================
# {ACT, Diffusion} arms on stackcups_20260829_merged (UR10e, absolute cart7)
# =============================================================================
#   1. ACT baseline        chunk 100, 100k steps  (doubles as the ACT oracle)
#   2. ACT labels          oracle = arm 1  -> outputs/label/stack_cups_merged
#   3. ACT DemoSpeedup     chunk 100 -> 50, pad_mode=zero
#   4. ACT B-spline        16x11 parameter matrix, no labelling stage
#   5. Diffusion baseline  n_obs_steps=1, 100k steps  (its own oracle)
#   6. Diffusion labels    oracle = arm 5  -> outputs/label/stack_cups_merged_dp
#   7. Diffusion DemoSpeedup  horizon 64 -> 32, n_action_steps 32 -> 16, pad_mode=hold
#   8. Diffusion B-spline  the same 16x11 matrix, no labelling stage
#
# This is the same 2x3 the pickplace queue runs -- {ACT, Diffusion} x {baseline,
# DemoSpeedup, B-spline} -- on the second real dataset, so the two tasks compare
# arm for arm. PACE needs no arm of its own: it runs at eval time on arms 1 and 5.
#
# Every design point below is the pickplace queue's, and deliberately so: the two
# datasets have identical feature shapes (action cart7, state 6, two cameras) at
# the same 20 fps, so anything that differed between the queues would be a
# difference between the *tasks* that is really a difference between the scripts.
# The per-family choices carried over verbatim are n_obs_steps=1 for Diffusion,
# pad_mode zero for ACT and hold for Diffusion, and each family labelling its own
# DemoSpeedup arm. Their justifications live in run_demospeedup_pickplace.sh and
# are not restated here; only what is specific to THIS dataset is.
#
# The ACT arms run first and the Diffusion arms after, rather than interleaved,
# because arms 1-4 were already running when 5-8 were added and the queue is
# resumable: a rerun walks past the finished ACT arms in seconds.
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
# would hand arm_state a checkpoint trained on the 12-episode set and it would skip
# straight past stage 1, labelling the merged data with an oracle that never saw
# it.
#
# BUDGET: 100k steps at batch 32 = 3.2M samples = 51.9 epochs on 61631 frames.
# The repo's other queues are epoch-matched at ~100 epochs (pickplace 102, the old
# cups run 108); on a 6.9x dataset that would be ~193k steps and ~24h per arm.
# 100k is the deliberate trade (user decision 2026-08-30): half the wall clock,
# and it matches the pickplace ACT arms step-for-step so the two tasks' ACT arms
# are comparable in optimizer steps. What parity actually requires is that the
# three arms HERE get the same budget, and they do.
#
# bf16 for every arm, as in the pickplace queue: a numerics change, so it is set
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
PRUNER=${PACE_PRUNER:-$REPO_ROOT/src/pace_bench/data/prune_checkpoints.py}
KEEP=2
[ -f "$PRUNER" ] || echo "WARNING: no pruner at $PRUNER (set PACE_PRUNER); keeping every checkpoint"
# 12GB, up from the 8GB this queue asked for while it was ACT-only: a Diffusion
# checkpoint is 1.1GB against ACT's 0.6GB, and KEEP=2 holds two of them per arm
# while it trains.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dcs '0-9' '\n')
[ "${FREE_GB:-0}" -ge 12 ] || { echo "only ${FREE_GB}GB free on $(pwd) -- a Diffusion checkpoint is 1.1GB and KEEP=$KEEP; free space first"; exit 1; }

DATA=(--dataset.repo_id=local/stack_cups_merged
      "--dataset.root=$DATASET_ROOT"
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project=pace_benchmark_stack_cups)
STEPS=100000
BUDGET=(--batch_size=32 --steps=$STEPS --save_freq=10000 --log_freq=100
        --num_workers=4 --seed=42 --policy.device=cuda --policy.push_to_hub=false
        --accelerator.mixed_precision=bf16)
ACT=(--policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100)
# n_obs_steps=1 overrides LeRobot's default of 2. It is upstream DemoSpeedup's own
# setting for this family, and it is what lets arm 5 double as arm 6's oracle:
# labelling asks about one frame at a time with no history to give, so
# DiffusionChunkSampler rejects an n_obs_steps=2 checkpoint outright. See the long
# note in run_demospeedup_pickplace.sh; the trade (a slightly weaker policy than a
# stock 2-frame DP) is upstream's choice, not this queue's.
DIFFUSION=(--policy.type=diffusion --policy.n_obs_steps=1)
# The B-spline geometry, shared by arms 4 and 8 so the two families fit identical
# curves and differ only in the policy regressing them. A B-spline chunk indexes
# control points rather than timesteps, so ACT[]/DIFFUSION[]'s chunk settings are
# deliberately not spread alongside it: chunk_size 10 + 2*degree 3 gives a 16x11
# matrix, and that width becomes the policy's chunk and n_action_steps.
#
# 16 is upstream's own real-robot horizon, and it is the width Diffusion can take:
# its temporal U-Net halves the horizon once per `down_dims` stage, so the width
# must be a multiple of 2**len(down_dims) = 8. The recorded UR10e dataset's 20
# (width 26) trains on ACT and would be rejected here.
#
# The dataset is cart7 at 20 fps, identical to pickplace's, so layout and fps carry
# over unchanged. num_actions stays unset: it is the speed lever and a decode-time
# choice, so it belongs to evaluation rather than to training.
BSPLINE=(--method.type=bspline --method.layout=cart7 --method.fps=20
         --method.chunk_size=10 --method.degree=3 --method.max_error=0.01)
N_EPISODES=175

mkdir -p logs outputs/label
stage () { echo; echo "═══════════ $1 ═══════════"; }

# Existence of checkpoints/last is NOT proof the arm finished. save_freq is 10k,
# so a run that dies at step 40k leaves a `last` behind; a bare directory check
# would report "already trained" and hand the comparison a 40k arm sitting next to
# a 100k one. Resolve the symlink and insist on the full budget. 10# because the
# dirs are zero-padded and bash reads a leading 0 as octal (010000 -> 4096).
#
# A partial arm is RESUMED rather than thrown away, which is sound HERE and would
# not be everywhere. Upstream's resume restores step, RNG, optimizer moments, and
# the EpisodeAwareSampler offset recomputed from the saved (step, batch_size,
# dp_world_size), so the continuation is sample-exact: the arm that comes out is
# the 100k arm this queue asked for, not a differently-trained one. An arm with a
# checkpoint but no training_state left to load cannot be continued and starts over.
#
# What makes that enough is that no part of the optimisation was sized against a
# shorter budget -- the interrupted run was already configured for these same 100k
# steps. The two families reach that conclusion differently and both have to now.
# ACT carries no LR schedule at all (`get_scheduler_preset()` returns None, so a
# constant 1e-5 AdamW), leaving nothing to restore. Diffusion does have one: cosine
# decay over the full budget after 500 warmup steps. It is restored too --
# `save_training_state` writes `scheduler_state.json` beside the optimizer state and
# `load_training_state` reads it back (lerobot/common/train_utils.py:329 and :492)
# -- so a resumed Diffusion arm continues down the same curve rather than
# re-entering warmup at an already-decayed learning rate.
arm_state () {  # -> done | resume | fresh, on stdout
    local last="outputs/train/$1/checkpoints/last" step
    if [ -d "$last" ]; then
        step=$(basename "$(readlink -f "$last")")
        if [ "$((10#$step))" -ge "$STEPS" ]; then echo done; return; fi
        if [ -f "$last/training_state/training_step.json" ]; then echo resume; return; fi
    fi
    echo fresh
}

at_step () { basename "$(readlink -f "outputs/train/$1/checkpoints/last")"; }

# Numeric step, 0 when the arm has no checkpoint yet -- the retry loop's progress test.
step_of () {
    local last="outputs/train/$1/checkpoints/last"
    if [ -d "$last" ]; then echo "$((10#$(basename "$(readlink -f "$last")")))"; else echo 0; fi
}

# One launch. Everything that is per-attempt lives here; `train` owns the retrying.
launch () {  # launch <name> <job> <state> <policy args...>
    local name=$1 job=$2 state=$3; shift 3
    local pruner_pid=
    if [ -f "$PRUNER" ]; then
        "$PY" "$PRUNER" "outputs/train/$name" --keep "$KEEP" --interval 300 \
            >>"logs/${name}.prune.log" 2>&1 &
        pruner_pid=$!
    fi
    echo "───── $name attempt at $(date '+%F %T'), state=$state ─────" >>"logs/${name}.log"
    if [ "$state" = resume ]; then
        # The checkpoint's own train_config.json is the entire configuration on this
        # path -- steps, batch, bf16, seed, wandb run id and --method.* included --
        # and upstream applies CLI flags *over* it. Re-passing the arg arrays would
        # therefore let a later edit of this file silently change an arm mid-flight,
        # so nothing is passed but the checkpoint and where to keep writing.
        "$PY" -m pace_bench.train.run_train \
            --config_path="$REPO_ROOT/outputs/train/$name/checkpoints/last/pretrained_model/train_config.json" \
            --resume=true --output_dir="outputs/train/$name" \
            2>&1 | tee -a "logs/${name}.log"
    else
        "$PY" -m pace_bench.train.run_train "${DATA[@]}" "${WANDB[@]}" "${BUDGET[@]}" "$@" \
            --job_name="$job" --output_dir="outputs/train/$name" \
            2>&1 | tee -a "logs/${name}.log"
    fi
    [ -n "$pruner_pid" ] && { kill "$pruner_pid" 2>/dev/null; "$PY" "$PRUNER" "outputs/train/$name" --keep "$KEEP" --once >>"logs/${name}.prune.log" 2>&1; }
    return 0
}

# An arm is launched until it reaches the budget, not once.
#
# The 2026-09-01 attempt died at step 66,409 with a general protection fault inside
# libtorch_python.so (dmesg 13:55:36) -- no OOM, no CUDA Xid, no Python traceback,
# 41GB of RAM free, and the pickplace queue's four 100k arms never hit it. One
# native fault in the main process is not something this script can diagnose. What
# it can do is stop letting it cost the whole queue: that crash aborted stages 2-4
# as well, so ~6h of GPU time bought nothing but a 60k checkpoint. Relaunching from
# the last checkpoint caps the damage at the steps since the last save (<=10k,
# ~35min), and resume is sample-exact here, so the arm that finally reaches 100k is
# still the 100k arm the queue asked for.
#
# The guard against spinning: an attempt that ends no further along than it started
# is not a crash worth retrying, it is a run that cannot start. One such attempt is
# tolerated (a fresh arm has no checkpoint until step 10k, so its first crash can
# legitimately show no progress); a second in a row stops the queue.
MAX_ATTEMPTS=${PACE_MAX_ATTEMPTS:-6}

train () {  # train <output name> <wandb job name> <policy args...>
    local name=$1 job=$2; shift 2
    local state attempt=0 before after
    while :; do
        state=$(arm_state "$name")
        if [ "$state" = done ]; then
            [ "$attempt" -eq 0 ] && echo "$name already trained ($(at_step "$name") steps), skipping" \
                                 || echo "$name reached $(at_step "$name") steps in $attempt attempt(s)"
            return 0
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "FAILED: $name stuck at $(step_of "$name")/$STEPS after $MAX_ATTEMPTS attempts"; exit 1
        fi
        before=$(step_of "$name")
        case "$state" in
            resume) echo "[attempt $attempt/$MAX_ATTEMPTS] $name at $before/$STEPS -- resuming from its checkpoint" ;;
            fresh)  echo "[attempt $attempt/$MAX_ATTEMPTS] $name -- training from scratch"; rm -rf "outputs/train/$name" ;;
        esac
        launch "$name" "$job" "$state" "$@"
        after=$(step_of "$name")
        if [ "$after" -le "$before" ] && [ "$attempt" -gt 1 ]; then
            echo "FAILED: $name made no progress on attempt $attempt (still at $after) -- not a crash to retry"; exit 1
        fi
    done
}

# Labelling, once per policy family: each labels its own DemoSpeedup arm with its
# own baseline as the oracle, which is upstream DemoSpeedup's arrangement and the
# reason there are two label dirs here rather than one shared set.
label () {  # label <label dir> <oracle run name> <log tag>
    local out=$1 oracle=$2 tag=$3
    if [ "$(ls "outputs/label/$out"/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq "$N_EPISODES" ]; then
        echo "labels in outputs/label/$out already present, skipping"
    else
        "$PY" -m pace_bench.methods.demospeedup.run_label \
            --policy_path="$REPO_ROOT/outputs/train/$oracle/checkpoints/last/pretrained_model" \
            --dataset_repo_id=local/stack_cups_merged \
            --dataset_root="$DATASET_ROOT" \
            --num_action_samples=10 --temporal_aggregation=true --kde_bandwidth=1.0 \
            --min_cluster_size=5 --max_cluster_size=25 --rule=mean \
            --out="outputs/label/$out" \
            2>&1 | tee "logs/${tag}.log"
    fi
    local n
    n=$(ls "outputs/label/$out"/speedup_labels/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -eq "$N_EPISODES" ] || { echo "LABEL STAGE FAILED ($out): $n/$N_EPISODES label files"; exit 1; }

    # Is the label field structured, or noise with the right marginal? Precision
    # frames should come in runs; if the mean run is no longer than a coin flip with
    # the same rate would give, the retiming is not tracking the demonstration.
    # The heredoc binds to $PY, not to the pipeline's last command -- written after
    # `tee` it would feed the source to tee and leave python reading /dev/null.
    "$PY" - "outputs/label/$out/speedup_labels" <<'PYEOF' 2>&1 | tee "logs/${tag}_signal.log"
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
}

stage "1/8: ACT baseline (also the ACT arm's labelling oracle)"
train cups_merged_act_base act_baseline_merged "${ACT[@]}" --method.type=none

stage "2/8: ACT entropy labelling (CVAE oracle = arm 1)"
label stack_cups_merged cups_merged_act_base cups_merged_label

stage "3/8: ACT DemoSpeedup (chunk 100 -> 50, masked zero-pad)"
train cups_merged_act_speedup act_demospeedup_merged "${ACT[@]}" \
    --method.type=demospeedup \
    --method.labels_path="$REPO_ROOT/outputs/label/stack_cups_merged/speedup_labels" \
    --method.pad_mode=zero

# ACT[] is deliberately not spread here: the geometry comes from BSPLINE[], which
# sets the chunk to the matrix width. See its definition above for why 16.
stage "4/8: ACT B-spline (16x11 parameter matrix, no labelling stage)"
train cups_merged_act_bspline act_bspline_merged \
    --policy.type=act "${BSPLINE[@]}"

# ---------------------------------------------------------------------------
# Diffusion. The same three methods on the other family, so the 2x3 here matches
# the pickplace queue's and the two tasks compare arm for arm.
#
# The Diffusion arms cost more than the ACT ones on this dataset, and stage 6 is
# where it shows: labelling queries the policy once per frame and a Diffusion
# query is 100 sequential DDPM steps, which no batch width makes cheaper. On
# pickplace's 31178 frames that stage took 1.4h at --batch_frames=32; this dataset
# is 61631 frames, so budget roughly 2.8h for it and hours, not minutes, for the
# ACT pass's equivalent. The three training arms are ~6h each at this budget.
# ---------------------------------------------------------------------------
stage "5/8: Diffusion baseline (n_obs_steps=1, so it is also its own oracle)"
train cups_merged_dp_base diffusion_baseline_merged "${DIFFUSION[@]}" --method.type=none

stage "6/8: Diffusion entropy labelling (oracle = arm 5; 100 DDPM steps/chunk, slow)"
label stack_cups_merged_dp cups_merged_dp_base cups_merged_dp_label

# pad_mode=hold, against ACT's zero. Diffusion's loss carries no padding mask
# under its own defaults (do_mask_loss_for_padding=False), and these actions are
# absolute cart7, so a trained zero is a command to the world origin rather than a
# harmless mid-range value. adjust_policy() raises rather than let the pairing go
# wrong silently. The full argument is in run_demospeedup_pickplace.sh.
stage "7/8: Diffusion DemoSpeedup (horizon 64 -> 32, n_action_steps 32 -> 16, hold-pad)"
train cups_merged_dp_speedup diffusion_demospeedup_merged "${DIFFUSION[@]}" \
    --method.type=demospeedup \
    --method.labels_path="$REPO_ROOT/outputs/label/stack_cups_merged_dp/speedup_labels" \
    --method.pad_mode=hold

# DIFFUSION[] keeps n_obs_steps=1 here even though the oracle argument for it does
# not apply (B-spline has no labelling stage to be an oracle for). The reason that
# remains is the one the comparison needs: the three Diffusion arms must differ in
# their method and in nothing else.
stage "8/8: Diffusion B-spline (the same 16x11 matrix, no labelling stage)"
train cups_merged_dp_bspline diffusion_bspline_merged "${DIFFUSION[@]}" "${BSPLINE[@]}"

echo
echo "═══════════ STACK CUPS MERGED QUEUE DONE ═══════════"
for d in cups_merged_act_base cups_merged_act_speedup cups_merged_act_bspline \
         cups_merged_dp_base cups_merged_dp_speedup cups_merged_dp_bspline; do
    printf '  %-32s %s\n' "$d" "$([ -d "outputs/train/$d/checkpoints/last" ] && echo "trained ($(basename "$(readlink -f "outputs/train/$d/checkpoints/last")"))" || echo MISSING)"
done
