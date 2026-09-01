#!/bin/bash
#SBATCH --job-name=pace-stackcups
#SBATCH --output=logs/slurm/pace-stackcups-%j.out
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
# SLURM: the merged stack-cups 2x3 queue -- the same six arms as pickplace on
# stackcups_20260829_merged (175 episodes, 61631 frames), plus two labelling stages.
# =============================================================================
#   mkdir -p logs/slurm && sbatch training_scripts/slurm_stackcups.sbatch
#
# This runs run_demospeedup_stackcups_merged.sh. The older, unmerged
# run_demospeedup_stackcups.sh trains on a 12-episode subset and is superseded by
# this queue; it deliberately has no job of its own (see training_scripts/README.md).
#
# RUNTIME. The queue script budgets ~6h per training arm and ~2.8h for Diffusion
# entropy labelling on this dataset's 61631 frames -- roughly 40h all told, so
# expect one or two self-requeues against the 24h default.
#
# DISK. This queue refuses to start with less than 12GB free: a Diffusion
# checkpoint is 1.1GB and the pruner keeps 2 per arm while it trains. On a cluster
# that check runs against whatever filesystem the checkout sits on, so submit from
# scratch/project space rather than a small home quota.
#
# Overrides: PACE_DATA_ROOT, STACK_CUPS_MERGED_ROOT, PACE_PRUNER, PACE_MAX_ATTEMPTS.
# =============================================================================
QUEUE_SCRIPT=run_demospeedup_stackcups_merged.sh

REPO_ROOT=${PACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT -- submit from the checkout root or set PACE_REPO"; exit 1; }
# module load cuda/12.1
source training_scripts/_slurm_common.sh
