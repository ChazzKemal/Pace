#!/bin/bash
# =============================================================================
# stack cups: {ACT, Diffusion} x {baseline, DemoSpeedup} -- four cells, no B-spline
# =============================================================================
# The user's scope on 2026-09-04: "cup stacking act and diffusion for the normal
# model and the demospeedup one". That is the left two columns of the cups 2x3,
# and it deliberately leaves the two B-spline arms for later.
#
#   ./training_scripts/run_cups_base_speedup.sh
#   PACE_REENTRIES=1 ./training_scripts/run_cups_base_speedup.sh
#
# Nothing here re-implements the queue. run_demospeedup_stackcups_merged.sh already
# takes a list of stages to run, keyed on the artifact each one produces, so this
# script is that queue plus the drain-and-re-enter loop from run_real_queues.sh --
# which cannot be reused directly because its QUEUES array is fixed and forwards no
# stage selection, so it would pull the two B-spline arms back in.
#
# The five stages named below, in the queue's own order:
#
#   cups_merged_act_base     stage 1  done at 100k since 2026-09-01, skipped in seconds
#   stack_cups_merged        stage 2  the ACT label set, 175/175 present, skipped
#   cups_merged_act_speedup  stage 3  RESUMES from its 30k checkpoint -> 100k
#   cups_merged_dp_base      stage 5  fresh, 100k
#   stack_cups_merged_dp     stage 6  DP labels, oracle = the arm above
#   cups_merged_dp_speedup   stage 7  fresh, 100k, hold-pad
#
# Stages 4 and 8 (the B-spline arms) are simply absent from the list, so the queue
# prints "not selected" and walks past them. The ordering is the queue's and is
# load-bearing at one point only: stage 6 labels with stage 5's checkpoint as its
# oracle, so the DP baseline must finish before the DP DemoSpeedup arm can start.
#
# The 2026-09-02 21:00 queue was walking these same stages when it was cut off at
# 02:17 on 2026-09-03 -- the box rebooted (uptime dates the boot to ~10:25 that
# morning), leaving the ACT DemoSpeedup arm at 30k. No fault of the training code,
# and resume here is sample-exact, so this picks the arm up where it stopped.
#
# Rough budget from the measured rates: ACT 30k->100k ~4h, DP baseline ~6h, DP
# labelling ~2.8h (one policy query per frame is 100 sequential DDPM steps, and
# this dataset has 61631 frames), DP DemoSpeedup ~6h. Call it ~19h.
set -uo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
mkdir -p logs
LOG="logs/run_cups_base_speedup_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] run_cups_base_speedup: log at $LOG"

REENTRIES=${PACE_REENTRIES:-3}
QUEUE=training_scripts/run_demospeedup_stackcups_merged.sh
STAGES=(cups_merged_act_base
        stack_cups_merged
        cups_merged_act_speedup
        cups_merged_dp_base
        stack_cups_merged_dp
        cups_merged_dp_speedup)

# Bounded wait for the card to drain. A process exiting and its memory being
# released are not the same instant, and an earlier cups arm died on a CUDA OOM
# for exactly that reason.
drain () {
    local used
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-99999}" -lt 2000 ] && return 0
        echo "[$(date '+%F %T')] GPU holds ${used}MiB, waiting ..."
        sleep 30
    done
    echo "[$(date '+%F %T')] GPU still holds ${used}MiB after 30 min -- starting anyway"
}

status=0
for n in $(seq "$REENTRIES"); do
    drain
    echo
    echo "═══════ $(date '+%F %T')  $QUEUE  (entry $n/$REENTRIES) ═══════"
    if "$QUEUE" "${STAGES[@]}"; then
        echo "[$(date '+%F %T')] queue finished clean"
        break
    fi
    rc=$?
    echo "[$(date '+%F %T')] queue exited $rc"
    if [ "$n" -eq "$REENTRIES" ]; then
        echo "GIVING UP after $REENTRIES entries -- see the FAILED line above"
        status=1
    else
        sleep 60
    fi
done

echo
echo "═══════ $(date '+%F %T')  CUPS BASELINE+DEMOSPEEDUP DONE (status $status) ═══════"
.venv/bin/python checkpoint_status.py
exit "$status"
