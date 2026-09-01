#!/bin/bash
# =============================================================================
# DemoSpeedup end-to-end on LIBERO-10, xVLA / absolute EE6D
# =============================================================================
# Three LoRA finetunes of the same pretrained xVLA on the same 400 demos, run back
# to back so the only difference between them is the method:
#
#   A  method=none         chunk 30                 demos at their recorded pace
#   B  method=demospeedup  window 60 -> chunk 15    targets retimed 2x/4x by label
#   C  method=bspline      chunk 30 -> a 16x11      spline parameters, no labels
#                          parameter matrix
#
# PACE needs no arm: it runs at eval time on arm A's weights.
#
# A and B both PASS chunk 30. For B the trainer widens the dataset's raw action
# window to 60 = 15 * high_v BEFORE the dataset is built and sets the trained chunk
# to 15 after -- upstream's walk-the-tail-then-truncate semantics on a fixed-window
# loader: every executed slot is a real waypoint except at episode ends. (The
# fork's 2x window under-supplied the walk; ~7% of its executed steps were
# trained dwells.) C passes no chunk at all: a B-spline chunk indexes control
# points rather than timesteps, so the method sets it to the matrix width.
# Every arm is evaluated with PACE off, so any difference in time-to-success comes
# from what the policy learned.
#
# Labels come from the fork's stage-2 run (entropy of the pretrained xVLA's own
# action samples); stage 2 is not ported yet, so they are consumed as given.
# =============================================================================
set -euo pipefail
# Everything is resolved from this script's own location: the repo is its
# directory, and the three inputs this run does not itself produce -- the dataset,
# the pretrained xVLA, the stage-2 labels -- live in `data/` beside the checkout,
# under datasets/sim, checkpoints/ and labels/. All are overridable; none are in git.
# The repo is this script's PARENT directory: the queue scripts live in
# training_scripts/ and every path below is relative to the checkout root.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
DATA_ROOT=${PACE_DATA_ROOT:-$(dirname "$REPO_ROOT")/data}
DATASET_ROOT=${LIBERO10_ROOT:-$DATA_ROOT/datasets/sim/libero_10_ee6d}
# The stock `lerobot/xvla-libero` checkpoint, by hub id rather than a local copy:
# LeRobot at the pinned SHA loads it unmodified (it remaps the older vendored
# Florence-2 key layout on load), and the hub cache is shared, so a 3.3GB second
# copy under data/ bought nothing. Override with a path to pin a local one.
POLICY_PATH=${XVLA_POLICY_PATH:-lerobot/xvla-libero}
LABELS_PATH=${LIBERO10_LABELS_PATH:-$DATA_ROOT/labels/xvla_libero10_ee6d/speedup_labels}
require () {  # require <path> <env var to override it>
    [ -e "$1" ] || { echo "missing $1 -- set $2 to its location"; exit 1; }
}
require "$DATASET_ROOT" LIBERO10_ROOT
# Only a local path can be checked for existence; a hub id is resolved by
# huggingface_hub at load time, so requiring it here would always fail.
case "$POLICY_PATH" in /*|./*|../*) require "$POLICY_PATH" XVLA_POLICY_PATH ;; esac
require "$LABELS_PATH" LIBERO10_LABELS_PATH
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 VIDEO_BACKEND=pyav

PY=.venv/bin/python
DATASET=(--dataset.repo_id=local/libero_10_ee6d
         "--dataset.root=$DATASET_ROOT"
         --dataset.video_backend=pyav)
# LoRA on all linear layers; the action heads are DomainAwareLinear (nn.Embedding),
# which LoRA cannot target, so they are fully trained instead.
PEFT=(--peft.method_type=LORA --peft.r=8 --peft.lora_alpha=8 --peft.target_modules=all-linear
      --peft.full_training_modules='["transformer.soft_prompt_hub","transformer.action_encoder","transformer.action_decoder"]')
# alpha/rank = 1.0 and lr 1e-5: flow matching compounds adapter error over 10
# denoising steps, so both are held low.
COMMON=("--policy.path=$POLICY_PATH"
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
    --method.labels_path="$LABELS_PATH" \
    --method.pad_mode=hold

# ---------------------------------------------------------------------------
# READ THIS BEFORE TRUSTING ARM C. Two things about it are unsettled, and neither
# is settled by running it; both are recorded in docs/PLAN.md.
#
# 1. COMPARABILITY IS AN OPEN DECISION (raised by the user 2026-08-30, still
#    pending). Every LIBERO arm starts from a checkpoint pretrained on *dense
#    action chunks*, and the three methods ask different amounts of that
#    checkpoint. PACE does not touch the targets; DemoSpeedup keeps the action
#    space exactly, so slot k still means "ee6d pose at step k"; B-spline
#    reinterprets the tokens -- the chunk axis stops indexing timesteps and starts
#    indexing control points, and one channel becomes a time. The domain-0 finding
#    weakens this a lot (every arm trains its action head from random init, so no
#    arm inherits an aligned decoder and the asymmetry is confined to the LoRA'd
#    trunk) but does not remove it. The plan's ranked options are: (1) make the
#    real-robot comparison the headline and report LIBERO B-spline separately if at
#    all; (2) reinitialise action_encoder/action_decoder for every LIBERO arm to
#    equalise the handicap, costing the baseline absolute SR; (3) drop B-spline
#    from the LIBERO axis. This arm exists so option 1 has something to report --
#    it is NOT an answer to the question, and choosing (3) means deleting it.
#
# 2. KNOT_SCALE = 10.0 IS UNTUNED. xVLA does not normalize (its normalization
#    mapping is IDENTITY throughout), so raw magnitudes reach the loss and the
#    per-group weights decide what the policy attends to. XYZ_SCALE=500 and
#    ROT_SCALE=10 come from upstream xVLA; the knot term has no reference to
#    inherit from, since upstream xVLA has no knot and upstream B-spline has no
#    xVLA. It is set to the rotation scale by argument, not by measurement, and it
#    is the first thing to sweep if this arm tracks poorly in time. A 300-step
#    probe left knot_loss/KNOT_SCALE at ~1.34 MSE against a target variance of
#    ~0.94 -- no better than predicting the mean knot -- but that was 0.6% of an
#    epoch and still inside LR warmup, so it measures nothing yet.
#
# The two flags that are NOT optional here. xVLA slices its action vector by
# hardcoded index (POS_IDX=(0,1,2), ROT_IDX=(3..8), gripper at 9), so upstream's
# knot-first matrix trains a *time* as an x-coordinate -- which showed up as a
# position loss of 122840 beside a rotation loss of 6.3. --method.arrangement puts
# the control point in slots 0-9 and the knot in slot 10, and
# --policy.action_mode=ee6d_bspline scores that knot as a fourth loss term.
# Selecting one without the other is wrong in a way nothing checks.
#
# `ee6d_bspline` reaches xVLA's action registry as a side effect: the method
# imports the module that registers it while fitting the episodes, which happens
# in adjust_dataset -- inside make_train_eval_datasets, and so before make_policy
# resolves action_mode (upstream train(): datasets, then policy). Verified, but it
# is an ordering this arm alone depends on, so a future reordering of those two
# calls breaks this arm and nothing else.
#
# chunk_size is not passed: adjust_policy sets it to the matrix width (16) and
# n_action_steps with it. layout=ee6d20 drops LIBERO's 10 zero-pad action columns
# for the fit and restores them on decode. num_actions stays unset -- the decode
# rate is an evaluation choice.
run ds_libero10_bspline xvla_bspline \
    --policy.action_mode=ee6d_bspline \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01

echo "=== all three trainings done ==="
