#!/bin/bash
#SBATCH --job-name=pace-pickplace
#SBATCH --output=logs/slurm/pace-pickplace-%j.out
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
# SLURM: the pickplace 2x3 queue -- {ACT, Diffusion} x {baseline, DemoSpeedup,
# B-spline}, plus the two entropy labelling stages. Eight stages, one job.
# =============================================================================
#   sbatch training_scripts/slurm_pickplace.sbatch
#
# Submit FROM THE CHECKOUT ROOT (the job reads $SLURM_SUBMIT_DIR to find the repo;
# PACE_REPO overrides it) and make sure logs/slurm/ exists first -- SLURM will not
# create the directory its --output points at, and a job whose output file cannot
# be opened fails before it prints anything:
#
#   mkdir -p logs/slurm && sbatch training_scripts/slurm_pickplace.sbatch
#
# RUNTIME. By the queue script's own accounting the six training arms are ~6h each
# at 100k steps, and stage 5 -- Diffusion entropy labelling, 100 sequential DDPM
# steps per frame -- took 1.4h on this dataset's 31178 frames. Call it 40h, so this
# will requeue itself once or twice against the 24h default below. That is the
# designed path, not a failure: see _slurm_common.sh.
#
# The inputs it does not produce: data/datasets/real/pickplace_cart7_v2_angleaxis_nogrip
# (PICKPLACE_ROOT overrides), and a writable outputs/ and logs/. The queue checks
# the dataset up front and exits rather than discovering it missing at stage 1.
#
# Per-arm overrides that still work here, since this job only wraps the queue:
#   PACE_DATA_ROOT   the data/ dir, if it is not beside the checkout
#   PICKPLACE_ROOT   the dataset itself
#   WANDB_PROJECT    default pace_benchmark_pickplace
#   PACE_MAX_ATTEMPTS relaunches per arm before the queue gives up (default 6)
# =============================================================================
QUEUE_SCRIPT=run_demospeedup_pickplace.sh

REPO_ROOT=${PACE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT -- submit from the checkout root or set PACE_REPO"; exit 1; }
# module load cuda/12.1                     # if your site needs it; the venv is not activated,
                                            # the queue scripts call .venv/bin/python directly
source training_scripts/_slurm_common.sh
