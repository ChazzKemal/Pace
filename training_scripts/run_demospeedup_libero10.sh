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
#   D  method=bspline      as C                     + pos_emb rows 0-15 trained and the
#                                                   visual rows put back where the
#                                                   pretrained table had them
#   E  method=bspline      as C                     lr 1e-4, LoRA r 16, 40k steps: the
#                                                   head's capacity and budget, not a
#                                                   comparison member
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
# Rank, alpha, learning rate and budget are per arm, defaulting to what arms A-D ran
# with: alpha/rank = 1.0 and lr 1e-5, held low because flow matching compounds adapter
# error over 10 denoising steps. An arm that needs more sets ARM_PEFT_R, ARM_PEFT_ALPHA,
# ARM_LR or ARM_STEPS on its `run` line; `launch` and the skip guard both read them, so
# a 40k arm is judged done at 40k and a 20k arm at 20k.
PEFT_TARGETS=(--peft.method_type=LORA --peft.target_modules=all-linear
      --peft.full_training_modules='["transformer.soft_prompt_hub","transformer.action_encoder","transformer.action_decoder"]')
STEPS=20000
COMMON=("--policy.path=$POLICY_PATH"
        --policy.device=cuda --policy.push_to_hub=false
        --batch_size=8 --save_freq=5000 --log_freq=100
        --num_workers=4 --seed=42
        --wandb.enable=true --wandb.project="${WANDB_PROJECT:-pace_benchmark_libero10}")

# Created here rather than assumed: on a fresh cluster checkout `logs/` does not
# exist yet, and the first `tee` into it would take the arm down before it started.
mkdir -p logs

# Existence of checkpoints/last is NOT proof the arm finished. save_freq is 5k, so a
# run cut at step 8k leaves a `last` behind, and a bare directory check would report
# "already trained" and hand the comparison an 8k arm sitting beside a 20k one.
# Resolve the symlink and insist on the full budget. 10# because the dirs are
# zero-padded and bash reads a leading 0 as octal (05000 -> 2560).
#
# This is what makes the queue safe to interrupt, which under SLURM is not an edge
# case but the design: slurm_libero10.sbatch requeues itself at the wall clock, so
# the script is expected to be cut mid-arm and re-entered from the top.
#
# A partial arm is RESUMED, not discarded. Upstream's resume restores step, RNG,
# optimizer moments and the sampler offset, so the continuation is sample-exact and
# the arm that finishes is the 20k arm this queue asked for. An arm with a
# checkpoint but no training_state left to load cannot be continued and starts over.
arm_state () {  # -> done | resume | fresh, on stdout
    local last="outputs/train/$1/checkpoints/last" step
    if [ -d "$last" ]; then
        step=$(basename "$(readlink -f "$last")")
        if [ "$((10#$step))" -ge "${ARM_STEPS:-$STEPS}" ]; then echo done; return; fi
        if [ -f "$last/training_state/training_step.json" ]; then echo resume; return; fi
    fi
    echo fresh
}

at_step () { basename "$(readlink -f "outputs/train/$1/checkpoints/last")"; }

step_of () {  # numeric step, 0 when the arm has no checkpoint yet
    local last="outputs/train/$1/checkpoints/last"
    if [ -d "$last" ]; then echo "$((10#$(basename "$(readlink -f "$last")")))"; else echo 0; fi
}

# `|| true` on both pipelines: this script runs under `set -e`, which would
# otherwise turn a crashed attempt into an aborted queue -- the retry loop below
# exists precisely to decide what a failure means, so the failure has to reach it.
launch () {  # launch <name> <job> <state> <extra args...>
    local name=$1 job=$2 state=$3; shift 3
    echo "───── $name attempt at $(date '+%F %T'), state=$state ─────" >>"logs/${name}.log"
    if [ "$state" = resume ]; then
        # The checkpoint's train_config.json is the whole configuration on this path
        # -- PEFT block, steps, seed, wandb run id and --method.* included -- and
        # upstream applies CLI flags over it. Re-passing the arrays would let a later
        # edit of this file silently change an arm mid-flight, so nothing is passed
        # but the checkpoint and where to keep writing.
        "$PY" -m pace_bench.train.run_train \
            --config_path="$REPO_ROOT/outputs/train/${name}/checkpoints/last/pretrained_model/train_config.json" \
            --resume=true --output_dir="outputs/train/${name}" \
            2>&1 | tee -a "logs/${name}.log" || true
    else
        # --job_name names the wandb run: <policy>_<addon>, per the project convention
        # (project pace_benchmark_<task>). It is kept separate from the output dir so
        # renaming runs never orphans an existing checkpoint.
        "$PY" -m pace_bench.train.run_train \
            "${DATASET[@]}" "${PEFT_TARGETS[@]}" "${COMMON[@]}" \
            --peft.r="${ARM_PEFT_R:-8}" --peft.lora_alpha="${ARM_PEFT_ALPHA:-8}" \
            --policy.optimizer_lr="${ARM_LR:-1e-5}" --steps="${ARM_STEPS:-$STEPS}" "$@" \
            --output_dir="outputs/train/${name}" --job_name="${job}" \
            2>&1 | tee -a "logs/${name}.log" || true
    fi
}

# An arm is launched until it reaches the budget, not once. The guard against
# spinning: an attempt ending no further along than it started is not a crash worth
# retrying but a run that cannot start. One is tolerated (a fresh arm has no
# checkpoint until step 5k); a second in a row stops the queue.
MAX_ATTEMPTS=${PACE_MAX_ATTEMPTS:-6}

run () {  # run <dir> <wandb run name> <extra args...>
    local name=$1 job=$2; shift 2
    local state attempt=0 before after
    while :; do
        state=$(arm_state "$name")
        if [ "$state" = done ]; then
            [ "$attempt" -eq 0 ] && echo "=== ${name} already trained ($(at_step "$name") steps), skipping ===" \
                                 || echo "=== ${name} reached $(at_step "$name") steps in $attempt attempt(s) ==="
            return 0
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "FAILED: ${name} stuck at $(step_of "$name")/${ARM_STEPS:-$STEPS} after $MAX_ATTEMPTS attempts"; exit 1
        fi
        before=$(step_of "$name")
        case "$state" in
            resume) echo "=== [attempt $attempt/$MAX_ATTEMPTS] ${name} at $before/${ARM_STEPS:-$STEPS} -- resuming ===" ;;
            fresh)  echo "=== [attempt $attempt/$MAX_ATTEMPTS] training ${name} from scratch ==="; rm -rf "outputs/train/${name}" ;;
        esac
        launch "$name" "$job" "$state" "$@"
        after=$(step_of "$name")
        if [ "$after" -le "$before" ] && [ "$attempt" -gt 1 ]; then
            echo "FAILED: ${name} made no progress on attempt $attempt (still at $after) -- not a crash to retry"; exit 1
        fi
    done
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
# NEW OUTPUT DIR, deliberately. `ds_libero10_bspline` is the arm trained before
# 2026-09-02, when GRIPPER_SCALE was still 1.0 -- a BCE weight left on an MSE channel,
# which left the gripper control point at R^2 0.32 while every other channel reached
# 0.68-0.91, and the arm scored 0% on every LIBERO-10 task. That checkpoint is kept
# rather than overwritten: it is the evidence, and reusing the name would have the
# queue's skip guard walk straight past it.
run ds_libero10_bspline_v2 xvla_bspline_v2 \
    --policy.action_mode=ee6d_bspline \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01

# ---------------------------------------------------------------------------
# D: arm C with the positional embedding allowed to learn what its rows now index.
# Same base, same budget, same loss and arrangement as C (knot in slot 10, so the
# pretrained head's slots 0-9 keep their meaning); two things differ, both about
# `pos_emb`, and both are what this arm measures:
#
#   --method.unfreeze_pos_emb_rows=16  rows 0-15, the action segment, train. They were
#                                      pretrained to mean "timestep k of a dense chunk"
#                                      and now mean "control point k".
#   --method.realign_pos_emb=true      the 250 visual/text tokens are put back on the
#                                      rows they were pretrained with. The chunk went
#                                      from 30 to 16, which slides every non-action
#                                      token 14 rows down a table where a row and the
#                                      row 14 above it are nearly orthogonal; without
#                                      this the frozen "visual rows" are the wrong ones.
#
# Supersedes `ds_libero10_bspline_uniform_posemb`, which unfroze the same rows but
# also switched to the knot-first arrangement and a uniform normalized loss -- three
# changes at once, and the knot-first one misaligns every slot of the pretrained head
# it starts from. Its evaluation was void anyway (see run_libero's unnormalizer note).
run ds_libero10_bspline_v2_posemb xvla_bspline_v2_posemb \
    --policy.action_mode=ee6d_bspline \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01 \
    --method.unfreeze_pos_emb_rows=16 --method.realign_pos_emb=true

# ---------------------------------------------------------------------------
# E: arm C given the capacity and the budget its head turned out to need.
# Arms C and D both finished at a position term of ~0.65 (500 x MSE), which is 3.6 cm
# per coordinate on the control points against the dense baseline's 0.84 cm on the
# same training frames; the gripper coefficient and the knot were off by similar
# factors, and the decoded curves missed the demonstrations by 4.7 cm on average.
# D's trainable positional rows changed none of it -- its loss tracked C's to within
# a percent at every checkpoint -- so the lever left is optimisation, not indexing.
# What the checkpoint's own head had, and this arm gives back:
#
#   ARM_LR=1e-4          the checkpoint's own optimizer_lr (1e-5 was a tenth of it,
#                        and the head moved 2.1 units where it had to relearn two
#                        channel meanings, against the baseline's 0.43 to fine-tune)
#   ARM_PEFT_R=16        double the adapter rank; alpha follows, so alpha/rank stays 1
#   ARM_STEPS=40000      double the budget, and the cosine stretched to span it --
#                        the checkpoint's 30k decay would otherwise flatline at
#                        2.5e-6 for the last quarter of the run
#
# Everything else is C: knot in slot 10, ee6d_bspline loss, frozen pretrained pos_emb.
# NOT a member of the equal-budget comparison; it asks whether the representation can
# be learned by this trunk at all before anything is concluded from arm C's 0%.
ARM_LR=1e-4 ARM_PEFT_R=16 ARM_PEFT_ALPHA=16 ARM_STEPS=40000 \
run ds_libero10_bspline_v3 xvla_bspline_v3 \
    --policy.action_mode=ee6d_bspline --policy.scheduler_decay_steps=40000 \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01

# ---------------------------------------------------------------------------
# F: arm C with the gripper command ramped before the fit (--method.gripper_ramp=9).
# The one variable that separates this arm from C. LIBERO's gripper is a 0/1 command;
# fitting that step pins knots one frame apart on both sides of every edge and the
# least-squares curve overshoots to ~1.1, so the gripper control points the policy
# regresses carry a step and an overshoot no other channel has. A zero-phase Hann
# ramp over 9 frames (0.45 s) keeps every 0.5 crossing on its recorded frame -- the
# env thresholds at 0.5, so what is executed is identical -- and on episode 0 halves
# the near-edge knots (26 -> 14 of 65) and removes the overshoot. Upstream's own data
# has a continuous teleop setpoint here, never a step; this is the closest LIBERO
# gets to that setting.
run ds_libero10_bspline_v2_ramp xvla_bspline_v2_ramp \
    --policy.action_mode=ee6d_bspline \
    --method.type=bspline --method.layout=ee6d20 --method.arrangement=xvla_ee6d20 \
    --method.fps=20 --method.chunk_size=10 --method.degree=3 --method.max_error=0.01 \
    --method.gripper_ramp=9

echo "=== all six trainings done ==="
