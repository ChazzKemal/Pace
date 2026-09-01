#!/bin/bash
#SBATCH --job-name=pace-libero10
#SBATCH --output=logs/slurm/pace-libero10-%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu              # CHANGE ME: your GPU partition
##SBATCH --account=CHANGEME          # uncomment and set if your site bills accounts
#SBATCH --gres=gpu:1                 # one card; the queues train one arm at a time
#SBATCH --cpus-per-task=8            # --num_workers=4 loaders + the main process
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300          # warn the batch shell 5 min before the limit
#SBATCH --requeue                    # allow the trap below to put the job back

# =============================================================================
# SLURM: the LIBERO-10 xVLA queue -- three LoRA finetunes of the same pretrained
# checkpoint, differing only in method (none / demospeedup / bspline).
# =============================================================================
#   mkdir -p logs/slurm && sbatch training_scripts/slurm_libero10.sbatch
#
# RUNTIME. Three arms at 20k steps, batch 8, LoRA on a frozen xVLA trunk -- much
# shorter than the real-robot queues, and it may well finish inside one 24h job.
# The requeue path is there anyway; it costs nothing when it is not used.
#
# INPUTS. Two of the three are not in git and not produced here:
#   data/datasets/sim/libero_10_ee6d        LIBERO10_ROOT
#   data/labels/xvla_libero10_ee6d/...      LIBERO10_LABELS_PATH  (the fork's stage-2 output)
# The third, the pretrained xVLA, is the hub id `lerobot/xvla-libero` by default and
# is fetched by huggingface_hub. ON A COMPUTE NODE WITHOUT NETWORK that download will
# fail: pre-populate the hub cache from the login node, or set XVLA_POLICY_PATH to a
# local copy. The queue's `require` guards check the two local paths before training,
# but they cannot check a hub id.
#
# MUJOCO_GL=egl is exported by the queue script. Training needs no rendering, but the
# LIBERO import path pulls in MuJoCo regardless, so a node with no display still needs
# EGL present.
#
# Overrides: PACE_DATA_ROOT, LIBERO10_ROOT, LIBERO10_LABELS_PATH, XVLA_POLICY_PATH,
# WANDB_PROJECT, PACE_MAX_ATTEMPTS.
# =============================================================================
QUEUE_SCRIPT=run_demospeedup_libero10.sh

REPO_ROOT=${PACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT -- submit from the checkout root or set PACE_REPO"; exit 1; }
# module load cuda/12.1
source training_scripts/_slurm_common.sh
