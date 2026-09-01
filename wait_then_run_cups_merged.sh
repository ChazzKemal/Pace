#!/bin/bash
# Chain the merged stack cups ACT queue behind an already-running pipeline.
#   usage: ./wait_then_run_cups_merged.sh [PID_TO_WAIT_FOR]
# Waits for PID to exit, then waits for the card to actually drain before
# starting. The old stack_cups DemoSpeedup arm died on a CUDA OOM with a second
# training holding 14.2GB of this 24GB card -- "the other job's PID is gone" and
# "the memory is released" are not the same instant, so both are checked.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
    echo "[$(date '+%F %T')] waiting for pid $WAIT_PID to exit ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] pid $WAIT_PID exited"
fi

# Bounded wait for the GPU to drain below 2GB (desktop/Xorg alone sits ~0.4GB).
for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && break
    echo "[$(date '+%F %T')] GPU still holds ${used}MiB, waiting ..."
    sleep 30
done
echo "[$(date '+%F %T')] GPU at ${used}MiB, starting merged stack cups queue"
exec ./run_demospeedup_stackcups_merged.sh
