#!/bin/bash
#SBATCH --job-name=pace-eval-lib10
#SBATCH --output=logs/slurm/pace-eval-lib10-%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu              # CHANGE ME: your GPU partition
##SBATCH --account=CHANGEME          # uncomment and set if your site bills accounts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# =============================================================================
# SLURM: evaluate the LIBERO-10 finetunes -- baseline and DemoSpeedup, 10 tasks x
# 20 episodes each, in simulation.
# =============================================================================
#   mkdir -p logs/slurm && sbatch training_scripts/slurm_eval_libero10.sbatch
#
# Run this only after slurm_libero10.sbatch has finished: it reads
# outputs/train/ds_libero10_{base,speedup}/checkpoints/last and fails immediately
# if they are not there. To chain the two in one submission:
#
#   TRAIN=$(sbatch --parsable training_scripts/slurm_libero10.sbatch)
#   sbatch --dependency=afterok:$TRAIN training_scripts/slurm_eval_libero10.sbatch
#
# NO REQUEUE, deliberately, and this is the one job here that differs. The training
# queues checkpoint every few thousand steps and resume from where they were cut, so
# being interrupted costs them minutes. Evaluation has no such state: a rollout that
# is killed halfway leaves nothing to resume from and nothing to skip, so requeueing
# would silently restart the whole sweep and overwrite outputs/eval/. If 8h is not
# enough for your node, raise --time rather than adding --requeue.
#
# MUJOCO_GL=egl is exported by the eval script. Unlike training, this job really does
# render -- every episode is a simulated rollout -- so the node needs working EGL and
# a GPU that can drive it. A missing libEGL surfaces as an import-time MuJoCo error
# in the first seconds, not hours in.
# =============================================================================
set -uo pipefail

# SLURM spools a copy of this script to the node, so $BASH_SOURCE says nothing about
# where the repo is. $SLURM_SUBMIT_DIR -- where `sbatch` ran -- is the answer.
REPO_ROOT=${PACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT -- submit from the checkout root or set PACE_REPO"; exit 1; }
# module load cuda/12.1
[ -x .venv/bin/python ] || { echo "no .venv/bin/python in $REPO_ROOT -- create the environment first"; exit 1; }
[ -x training_scripts/eval_demospeedup_libero10.sh ] || { echo "no eval script in $REPO_ROOT/training_scripts"; exit 1; }

for arm in ds_libero10_base ds_libero10_speedup; do
    [ -d "outputs/train/$arm/checkpoints/last" ] || {
        echo "missing outputs/train/$arm/checkpoints/last -- run slurm_libero10.sbatch first"; exit 1; }
done

echo "═══════════════════════════════════════════════════════════════"
echo "  job          ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-no slurm})"
echo "  node         $(hostname)"
echo "  repo         $REPO_ROOT"
echo "  started      $(date '+%F %T')"
command -v nvidia-smi >/dev/null && nvidia-smi -L | sed 's/^/  gpu          /'
echo "═══════════════════════════════════════════════════════════════"

training_scripts/eval_demospeedup_libero10.sh "$@"
STATUS=$?
echo
echo "[$(date '+%F %T')] eval script exited with status $STATUS"
exit "$STATUS"
