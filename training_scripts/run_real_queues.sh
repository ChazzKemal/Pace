#!/bin/bash
# Both UR10e queues back to back on a workstation: every real-robot arm that still
# needs weights, in the order the grid fills fastest.
#
#   ./training_scripts/run_real_queues.sh            # walk both queues
#   PACE_REENTRIES=1 ./training_scripts/run_real_queues.sh   # each queue gets one entry
#
#   1. run_demospeedup_pickplace.sh          only stage 8 is left: Diffusion B-spline
#   2. run_demospeedup_stackcups_merged.sh   ACT DemoSpeedup 20k -> 100k, ACT B-spline,
#                                            then the three Diffusion arms and their labels
#
# Each queue's own skip guards walk past what is done, so the order of the queues is
# all this script adds. pickplace goes first because one arm completes that task's
# 2x3; stack cups second because five training stages and a labelling pass remain.
#
# A queue exits 1 when its train() gives up on an arm -- two consecutive attempts
# with no new checkpoint -- and that also abandons the queue's later stages. On
# 2026-09-02 the guard fired twice within ten minutes: four segfaults across two arms
# and both launch paths (fresh and resume), at steps 0, ~900, ~1100 and 0, none with
# a Python traceback; the machine was rebooted at 13:58 and nothing has faulted
# since. So a queue that exits non-zero is re-entered here, up to PACE_REENTRIES
# times, once the card has drained. The per-arm loop inside a queue caps a crash at
# the steps since the last save; the re-entry caps a tripped guard at one arm's
# startup. A queue that fails PACE_REENTRIES times in a row is left alone and the
# next queue still runs, so one stuck arm cannot cost the other task its night.
#
# Everything is appended to logs/run_real_queues_<date>.log as well as the terminal;
# the per-arm logs (logs/<arm>.log) are written by the queues as before.
set -uo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
mkdir -p logs
LOG="logs/run_real_queues_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] run_real_queues: log at $LOG"

REENTRIES=${PACE_REENTRIES:-3}
QUEUES=(training_scripts/run_demospeedup_pickplace.sh
        training_scripts/run_demospeedup_stackcups_merged.sh)

# Bounded wait for the card to drop below 2GB -- desktop/Xorg alone sits ~0.5GB.
# "The previous process is gone" and "its memory is released" are not the same
# instant, and the old stack_cups DemoSpeedup arm died on a CUDA OOM for exactly
# that reason (see wait_then_run_cups_merged.sh).
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
for q in "${QUEUES[@]}"; do
    for n in $(seq "$REENTRIES"); do
        drain
        echo
        echo "═══════ $(date '+%F %T')  $q  (entry $n/$REENTRIES) ═══════"
        if "$q"; then
            echo "[$(date '+%F %T')] $q finished clean"
            break
        fi
        rc=$?
        echo "[$(date '+%F %T')] $q exited $rc"
        if [ "$n" -eq "$REENTRIES" ]; then
            echo "GIVING UP on $q after $REENTRIES entries -- see the FAILED line above"
            status=1
        else
            sleep 60
        fi
    done
done

echo
echo "═══════ $(date '+%F %T')  ALL QUEUES WALKED (status $status) ═══════"
.venv/bin/python checkpoint_status.py
exit "$status"
