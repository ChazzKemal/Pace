#!/bin/bash
# =============================================================================
# DemoSpeedup on pickplace_cart7_v2_angleaxis_nogrip (UR10e, 45 eps / 31k frames)
# =============================================================================
# The full 2x2: {ACT, Diffusion} x {baseline, DemoSpeedup}, plus one entropy
# labelling run per policy family.
#
#   1. ACT baseline                      chunk 100, 100k steps
#   2. ACT labels        oracle = arm 1  -> outputs/label/pickplace_act
#   3. ACT DemoSpeedup                   chunk 100 -> 50,  pad_mode=zero
#   4. Diffusion baseline                n_obs_steps=1, 100k steps
#   5. Diffusion labels  oracle = arm 4  -> outputs/label/pickplace_dp
#   6. Diffusion DemoSpeedup             horizon 64 -> 32, n_action_steps 32 -> 16,
#                                        pad_mode=hold
#
# Everything trains fresh in THIS stack (user decision 2026-08-29): the old
# Yunfei checkpoint is not used -- as oracle it would label with a policy from a
# different training stack, and as A-arm it would compare across stacks.
#
# Two design points follow upstream DemoSpeedup @ 34bd43a rather than being
# invented here:
#
#   * EACH FAMILY LABELS ITS OWN SPEEDUP ARM. Upstream's README runs all three
#     stages under one `launch=` config, and robobase/label.py loads
#     `work_dir/snapshots/best_snapshot.pt` -- the run dir of that same method.
#     So DP entropy retimes the DP arm and ACT entropy the ACT arm. The aloha
#     instantiation agrees, threading --policy_class through train / --label /
#     --speedup alike.
#
#   * EVERY ARM GETS THE SAME BUDGET. act_pixel_bigym.yaml and dp_pixel_bigym.yaml
#     both set num_pretrain_steps: 100000 and batch_size: 256; `speedup: true|false`
#     is the only thing that differs between the four arms. Parity is the
#     invariant, not the literal number -- here it is 100k at batch 32 throughout
#     (~103 epochs on 31k frames). Batch 32 rather than 64 for the same reason the
#     stack_cups queue gives: batch-64 diffusion activations do not fit this 24GB
#     card, and the two datasets have identical feature shapes.
#
# pad_mode differs between the families, and that is upstream-faithful too.
# Upstream zero-pads both (uniform_replay_buffer.py:855), but ACT rebuilds the
# mask from the zeros (act.py:441, `is_pad = actions.sum(axis=-1) == 0`) while
# DP's loss (diffusion.py:241-246) carries no mask at all -- which is harmless
# there only because robobase min-max normalizes actions into [-1,1], so a zero
# is mid-range. These actions are absolute cart7 and the retime step pads before
# the normalizer, so a trained zero is a command to the world origin. `hold` is
# that benign pad's equivalent here; adjust_policy() raises rather than let the
# pairing go wrong silently.
#
# Stage 5 is the expensive one: DP samples 100 DDPM steps per chunk (upstream's
# num_diffusion_iters: 100, and LeRobot's default is the same), against ~27
# frames/s for the ACT pass. Budget it in hours, not minutes.
#
# All in-repo: labelling used to run in the lerobot_uncertainty fork under conda,
# against a checkpoint copy with config keys stripped for the fork's older draccus.
# pace_bench.methods.demospeedup.run_label replaced both.
#
# wandb: project pace_benchmark_<task>, run named <policy>_<addon> --
# baselines are <policy>_baseline. Run names are deliberately independent of
# the output dirs below, so renaming a run never orphans a checkpoint.
# =============================================================================
set -uo pipefail

# Resolved from this script's own location, so a renamed checkout needs no edit:
# the repo is its directory and the datasets sit in `data/` beside it. The two
# inputs this run does not produce are overridable, since neither is in git.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"
DATA_ROOT=${PACE_DATA_ROOT:-$(dirname "$REPO_ROOT")/data}
DATASET_ROOT=${PICKPLACE_ROOT:-$DATA_ROOT/datasets/real/pickplace_cart7_v2_angleaxis_nogrip}
[ -e "$DATASET_ROOT" ] || { echo "missing $DATASET_ROOT -- set PICKPLACE_ROOT"; exit 1; }

# Checkpoint retention. LeRobot has no such setting, and this card's disk sits at
# 95%: four runs x 10 checkpoints would be ~36GB. Keeping 2 makes it ~8GB, and the
# pruner only ever deletes numbered checkpoints strictly older than `last`, so the
# checkpoint each labelling stage reads is never the one being removed. A missing
# pruner is a warning, not a stop -- the queue still fits, with less headroom.
#
# Run it with the venv python, NOT the system python3: this box ships 3.8, and
# the pruner annotates with PEP 585 generics (list[tuple[...]]) that need 3.9+.
# Invoked as `python3` it dies on import, and because it runs in the background
# nothing surfaces that -- the first symptom is a full disk hours later.
PRUNER=${PACE_PRUNER:-$(dirname "$REPO_ROOT")/prune_checkpoints.py}
KEEP=2
[ -f "$PRUNER" ] || echo "WARNING: no pruner at $PRUNER (set PACE_PRUNER); keeping every checkpoint"

export VIDEO_BACKEND=pyav PYTHONUNBUFFERED=1
PY=.venv/bin/python
DATA=(--dataset.repo_id=local/pickplace
      "--dataset.root=$DATASET_ROOT"
      --dataset.video_backend=pyav)
WANDB=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-pace_benchmark_pickplace}")
# Same budget for every arm; save_freq 10000 rather than 20000 because a killed
# run at step 11.9k/100k lost everything once already.
#
# bf16 rather than LeRobot's `mixed_precision: 'no'` default. Weights, gradients
# and the optimizer state stay fp32; only the forward/backward matmuls and
# convolutions run in bfloat16, and the loss accumulates back in fp32. bf16 keeps
# fp32's exponent range (~1e-38..3e38) and spends the difference on mantissa
# precision -- ~3 decimal digits against fp32's ~7 -- so it cannot overflow the
# way fp16 does and needs no gradient scaler. Measured 1.36x on this policy while
# the card was contended, so that is a floor.
#
# It is a numerics change, which is why it is set HERE, once, for every arm: the
# four arms are only comparable if they train in the same precision. Do not vary
# it per-arm.
BUDGET=(--batch_size=32 --steps=100000 --save_freq=10000 --log_freq=100
        --num_workers=4 --seed=42 --policy.device=cuda --policy.push_to_hub=false
        --accelerator.mixed_precision=bf16)
N_EPISODES=45

mkdir -p logs outputs/label
stage () { echo; echo "═══════════ $1 ═══════════"; }

done_already () { [ -d "outputs/train/$1/checkpoints/last" ] && { echo "$1 already trained, skipping"; return 0; }
    # A leftover dir from a crashed attempt has no checkpoint; upstream's validate
    # refuses to reuse it, so clear it and train fresh.
    rm -rf "outputs/train/$1"; return 1; }

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
    [ -d "outputs/train/$name/checkpoints/last" ] || { echo "FAILED: $name produced no checkpoint"; exit 1; }
}

label () {  # label <label dir> <oracle run name> <log tag>
    local out=$1 oracle=$2 tag=$3
    if [ "$(ls "outputs/label/$out"/speedup_labels/episode_*.npy 2>/dev/null | wc -l)" -eq "$N_EPISODES" ]; then
        echo "labels in outputs/label/$out already present, skipping"
    else
        "$PY" -m pace_bench.methods.demospeedup.run_label \
            --policy_path="$REPO_ROOT/outputs/train/$oracle/checkpoints/last/pretrained_model" \
            --dataset_repo_id=local/pickplace \
            --dataset_root="$DATASET_ROOT" \
            --num_action_samples=10 --temporal_aggregation=true --kde_bandwidth=1.0 \
            --min_cluster_size=5 --max_cluster_size=25 --rule=mean \
            --out="outputs/label/$out" \
            2>&1 | tee "logs/${tag}.log"
    fi
    local n
    n=$(ls "outputs/label/$out"/speedup_labels/episode_*.npy 2>/dev/null | wc -l)
    [ "$n" -eq "$N_EPISODES" ] || { echo "LABEL STAGE FAILED ($out): $n/$N_EPISODES files"; exit 1; }
    # Is the label field structured, or is it noise that happens to have the right
    # marginal? Precision frames should come in runs; if the mean run length is no
    # longer than a coin flip with the same rate would give, the retiming is not
    # tracking anything about the demonstration.
    # The heredoc must bind to $PY, not to the pipeline's last command: written
    # after `tee` it feeds the python source to tee, which dutifully echoes it,
    # while python reads the script's own stdin (/dev/null under nohup), sees EOF
    # and runs nothing. Silent, because this check gates no exit.
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

stage "1/6: ACT baseline (also the ACT arm's labelling oracle)"
train pickplace_act_base act_baseline \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --method.type=none

stage "2/6: ACT entropy labelling (oracle = arm 1)"
label pickplace_act pickplace_act_base pickplace_act_label

stage "3/6: ACT DemoSpeedup (chunk 100 -> 50, masked zero-pad)"
train pickplace_act_speedup act_demospeedup \
    --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=100 \
    --method.type=demospeedup \
    --method.labels_path="$REPO_ROOT/outputs/label/pickplace_act/speedup_labels" \
    --method.pad_mode=zero

# n_obs_steps=1 overrides LeRobot's default of 2, for two reasons that agree.
#
# Upstream gives DP a single observation frame in both instantiations. robobase
# (the one that has priority) sets `frame_stack: 1` in dp_pixel_bigym.yaml and has
# no per-policy obs-step notion at all: history would be a FrameStack env wrapper
# whose time axis is folded into channels before the encoder. aloha agrees
# explicitly -- `n_obs_steps: 1  # Now only support for 1 obs_step`.
#
# And labelling asks about one frame at a time with no history to give, so
# DiffusionChunkSampler rejects an n_obs_steps=2 checkpoint outright: at the
# default this baseline could not double as the DP oracle the way the ACT baseline
# does. (Upstream never hits that -- robobase labels by replaying each demo through
# a wrapped env, so the wrapper supplies whatever history the config asked for.)
#
# Note the trade: the wider DP literature conditions on 2 frames, so this is a
# slightly weaker policy than a stock DP -- upstream's choice, not ours.
stage "4/6: Diffusion baseline (n_obs_steps=1, so it is also its own oracle)"
train pickplace_diffusion_base diffusion_baseline \
    --policy.type=diffusion --policy.n_obs_steps=1 \
    --method.type=none

stage "5/6: Diffusion entropy labelling (oracle = arm 4; 100 DDPM steps/chunk, slow)"
label pickplace_dp pickplace_diffusion_base pickplace_dp_label

stage "6/6: Diffusion DemoSpeedup (horizon 64 -> 32, n_action_steps 32 -> 16, hold-pad)"
train pickplace_diffusion_speedup diffusion_demospeedup \
    --policy.type=diffusion --policy.n_obs_steps=1 \
    --method.type=demospeedup \
    --method.labels_path="$REPO_ROOT/outputs/label/pickplace_dp/speedup_labels" \
    --method.pad_mode=hold

echo
echo "═══════════ PICKPLACE 2x2 QUEUE DONE ═══════════"
for d in pickplace_act_base pickplace_act_speedup pickplace_diffusion_base pickplace_diffusion_speedup; do
    printf '  %-32s %s\n' "$d" "$([ -d "outputs/train/$d/checkpoints/last" ] && echo trained || echo MISSING)"
done
