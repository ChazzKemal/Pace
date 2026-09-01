# =============================================================================
# Shared body for every slurm_*.sbatch in this directory. Sourced, not run.
# =============================================================================
# What a job here does is submit one of the queue scripts beside it and stay out
# of its way. The skip guards, the argument arrays and the retry logic all live in
# the queue script, so a cluster run and a workstation run execute the same code
# and cannot drift apart. This file holds only what is specific to running under
# SLURM: locating the checkout, checking the environment, and handling the wall
# clock.
#
# THE WALL CLOCK. A task is six to eight arms back to back and will not fit in one
# job on most partitions. `#SBATCH --signal=B:USR1@300` has SLURM signal the batch
# shell five minutes before the limit; the trap below requeues the job and lets the
# running arm die where it stands. Its last checkpoint is at most `save_freq` steps
# back, the queue script resumes from it, and arms that already reached their full
# budget are skipped -- so the queue walks forward one job at a time until it is
# done. This is the whole reason the queue scripts insist on the step count rather
# than on the existence of `checkpoints/last`: under requeue an interrupted arm is
# the normal case, not an accident.
#
# The job must NOT `exec` the queue script: exec replaces this shell and takes the
# trap with it, and the requeue would never fire. It runs as a child instead, with
# `wait` retried because a trapped signal makes `wait` return >128 before the child
# has actually exited.
# LOCATING THE CHECKOUT. SLURM copies the submitted script to a node-local spool
# directory, so a job script's own path says nothing about where the repo is --
# $BASH_SOURCE points into the spool, which is why none of these jobs self-locate
# the way the queue scripts do. $SLURM_SUBMIT_DIR, the directory `sbatch` was run
# from, is the reliable answer; PACE_REPO overrides it. Each job resolves that and
# cd's there BEFORE sourcing this file, since sourcing it by relative path already
# depends on being in the right place.
set -uo pipefail

REPO_ROOT=$PWD
QUEUE="training_scripts/$QUEUE_SCRIPT"
[ -x "$QUEUE" ] || { echo "no executable $REPO_ROOT/$QUEUE -- submit from the checkout root or set PACE_REPO"; exit 1; }
# The queue scripts invoke .venv/bin/python directly, so there is nothing to
# activate -- but there is something to check, because the failure without it comes
# hours later as a confusing traceback rather than immediately.
[ -x .venv/bin/python ] || { echo "no .venv/bin/python in $REPO_ROOT -- create the environment first"; exit 1; }

echo "═══════════════════════════════════════════════════════════════"
echo "  job          ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-no slurm})"
echo "  node         $(hostname)"
echo "  repo         $REPO_ROOT"
echo "  queue        $QUEUE"
echo "  started      $(date '+%F %T')"
echo "  restarts     ${SLURM_RESTART_COUNT:-0}"
command -v nvidia-smi >/dev/null && nvidia-smi -L | sed 's/^/  gpu          /'
echo "═══════════════════════════════════════════════════════════════"

_requeue () {
    echo
    echo "[$(date '+%F %T')] wall clock is near -- requeueing job ${SLURM_JOB_ID}."
    echo "  The running arm dies with the job; the next run resumes it from its last"
    echo "  checkpoint and skips every arm already at its full step budget."
    scontrol requeue "$SLURM_JOB_ID"
}
trap _requeue USR1

"$QUEUE" "$@" &
CHILD=$!
# A trapped signal makes `wait` return >128 without the child having exited, so the
# real exit status is only the one that comes back below that threshold.
while :; do
    wait "$CHILD"; STATUS=$?
    [ "$STATUS" -gt 128 ] || break
done

echo
echo "[$(date '+%F %T')] queue script exited with status $STATUS"
exit "$STATUS"
