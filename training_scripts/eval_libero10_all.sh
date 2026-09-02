#!/bin/bash
# Evaluate the LIBERO-10 arms on the same 200 scenes, into one comparable tree.
#
#   ./training_scripts/eval_libero10_all.sh                       # all four, in order
#   ./training_scripts/eval_libero10_all.sh baseline pace         # only these
#   ./training_scripts/eval_libero10_all.sh --wait 12345 bspline  # after pid 12345 exits
#
# Every arm runs `--seed=42 --n_episodes=20 --tasks=0-9`, so episode k of every arm is
# LIBERO init state k of that task and `pace_bench.eval.compare_libero` pairs them
# episode by episode -- including on time, which the older `outputs/eval/ds_libero10_*`
# runs cannot do because they predate per-episode indices. That is why the two
# already-evaluated arms are rerun here rather than reused.
#
# What differs between arms is only the method and how much of a chunk is executed:
#   baseline     30 steps per query, nothing modulates speed
#   demospeedup  15 steps, because its waypoints sit 2-4x further apart; the runner
#                applies the constant tracking bump (kp/kd and gripper x low_v)
#   pace         the baseline's own weights -- PACE acts at eval time, so it needs no
#                arm of its own -- at the recorded look4cb+skip2 setting, 1.5x cap
#   bspline      decodes each predicted curve at 16 points, measured at 1.82x so the
#                success rates are read near DemoSpeedup's 1.91x
#
# That 16 is measured, not reasoned. It was 8 until 2026-09-02, on the argument that
# 8 points instead of the matrix's 16 rows is ~2x -- which is wrong, because the row
# count is not the time span. The span is the knot range, ~33 source frames on
# libero_10_ee6d, so decoding ground-truth parameters at 8 replays an episode at
# 4.12x. Realised speed against num_actions, on ground-truth fits (no policy):
#
#     num_actions    4     8    12    16    22    30    40
#     speed       9.44x 4.12x 2.53x 1.82x 1.30x 0.93x 0.70x
#     pos RMSE     0.36  0.77  0.51  0.48  0.87  1.27  1.21   cm
#     rot RMSE     0.25  0.60  0.43  0.39  0.80  1.30  1.18   deg
#
# Sub-centimetre everywhere, so the representation costs nothing across this range and
# num_actions really is a free speed dial. Use 30 for a 1x reading.
#
# Arms are independent processes, so an arm whose weights are ready can run while
# another is still training -- `--wait` exists for the one that is not ready, and the
# rest need no gate. `checkpoints/last` is resolved when the arm starts, so a gated
# arm picks up the final checkpoint rather than whichever existed at submit time.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# cuDNN's convolution autotuner segfaults inside the vision tower when the card is
# already busy -- it probes algorithms by allocating workspaces it cannot get -- and
# an arm here is expected to run beside a training job. Off for every arm rather than
# only the shared ones, so all four also select the same convolution algorithms.
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PACE_CUDNN_BENCHMARK=0

PY=.venv/bin/python
OUT=outputs/eval/libero10_4arm
COMMON=(--use_peft=true --task_suite=libero_10 --tasks=0-9
        --seed=42 --n_episodes=20 --batch_size=${BATCH_SIZE:-10})
ARMS=(bspline baseline demospeedup pace)
mkdir -p "$OUT" logs

WAIT_PID=""
if [ "${1:-}" = "--wait" ]; then WAIT_PID=$2; shift 2; fi
[ $# -gt 0 ] && ARMS=("$@")

if [ -n "$WAIT_PID" ]; then
    echo "[$(date '+%F %T')] waiting for pid $WAIT_PID to exit ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] pid $WAIT_PID exited"
    # A PID disappearing and its VRAM being released are not the same instant.
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-99999}" -lt 4000 ] && break
        sleep 15
    done
fi

BASE=outputs/train/ds_libero10_base/checkpoints/last/pretrained_model
SPEEDUP=outputs/train/ds_libero10_speedup/checkpoints/last/pretrained_model
BSPLINE=outputs/train/ds_libero10_bspline/checkpoints/last/pretrained_model
# The GRIPPER_SCALE=10 retrain. Same arm, same budget, same flags -- the only
# difference is the loss weight on slot 9, so the two are directly comparable.
BSPLINE_V2=outputs/train/ds_libero10_bspline_v2/checkpoints/last/pretrained_model
POSEMB=outputs/train/ds_libero10_bspline_uniform_posemb/checkpoints/last/pretrained_model

run_arm() {
    local arm=$1
    # An arm whose weights are missing is a setup error, not something to skip past:
    # a four-arm comparison with three arms in it is worse than no comparison.
    local -a args
    case "$arm" in
      baseline)    args=(--policy_path="$BASE" --n_action_steps=30 --method.type=none) ;;
      demospeedup) args=(--policy_path="$SPEEDUP" --n_action_steps=15
                         --method.type=demospeedup --method.low_v=2 --method.high_v=4) ;;
      pace)        args=(--policy_path="$BASE" --n_action_steps=30
                         --method.type=pace --method.max_speed=1.5 --method.min_speed=0.75
                         --method.clamp_deg=5.0 --method.action_stride=2 --method.n_lookahead=4
                         --method.lookahead_agg=cumulative_bending --method.lookahead_target=angle) ;;
      # The fit parameters are reconstruction, not choices: they must be what the
      # checkpoint trained under. Only num_actions is free, and it is the speed lever.
      bspline)     args=(--policy_path="$BSPLINE" --n_action_steps=16
                         --method.type=bspline --method.layout=ee6d20
                         --method.arrangement=xvla_ee6d20 --method.chunk_size=10
                         --method.degree=3 --method.max_error=0.01 --method.fps=20
                         --method.num_actions=16) ;;
      bspline_v2)  args=(--policy_path="$BSPLINE_V2" --n_action_steps=16
                         --method.type=bspline --method.layout=ee6d20
                         --method.arrangement=xvla_ee6d20 --method.chunk_size=10
                         --method.degree=3 --method.max_error=0.01 --method.fps=20
                         --method.num_actions=16) ;;
      # The second B-spline arm, and the only difference that matters is what it was
      # allowed to learn: 16 rows of xVLA's positional embedding, which the frozen arm
      # above had to reuse as pretrained. That changes the action space too -- the
      # parameters are scored uniformly (`bspline_uniform`) on a knot-first matrix
      # rather than through the ee6d head -- so the fit flags below are the
      # checkpoint's, not the arm above's, and must not be copied between them.
      # `restore` in run_libero loads pos_emb.safetensors; without it this arm would
      # silently run the pretrained table and read as a null result.
      bspline_posemb) args=(--policy_path="$POSEMB" --n_action_steps=16
                         --method.type=bspline --method.layout=ee6d20
                         --method.arrangement=knot_first20 --method.chunk_size=10
                         --method.degree=3 --method.max_error=0.01 --method.fps=20
                         --method.xvla_action_space=bspline_uniform
                         --method.normalize_parameters=true
                         --method.num_actions=16) ;;
      *) echo "unknown arm: $arm" >&2; return 1 ;;
    esac
    local ckpt=${args[0]#--policy_path=}
    [ -d "$ckpt" ] || { echo "missing checkpoint for $arm: $ckpt" >&2; return 1; }
    echo "[$(date '+%F %T')] $arm  <- $(readlink -f "$ckpt" | sed 's|.*/checkpoints/||')"
    "$PY" -m pace_bench.eval.run_libero "${args[@]}" --out="$OUT/$arm" "${COMMON[@]}" \
        2>&1 | tee -a "logs/eval_libero10_4arm_${arm}.log"
}

# An arm that fails must not take the queue with it: the arms are independent, and
# losing three good ones to a fourth's misconfiguration is how an evening gets spent.
# Failures are collected and re-reported at the end, and the exit status carries them.
failed=()
for arm in "${ARMS[@]}"; do
    run_arm "$arm" || { failed+=("$arm"); echo "[$(date '+%F %T')] $arm FAILED, continuing" >&2; }
done

# Compare only once every arm is in: whichever queue finishes last writes the report,
# and a queue that finishes while another is still running quietly does nothing.
complete=1
for arm in bspline baseline demospeedup pace; do
    [ "$(ls "$OUT/$arm"/task_*/eval_info.json 2>/dev/null | wc -l)" -eq 10 ] || complete=0
done
if [ "$complete" = 1 ]; then
    echo "[$(date '+%F %T')] all four arms in; comparing"
    "$PY" -m pace_bench.eval.compare_libero \
        "$OUT/baseline" "$OUT/demospeedup" "$OUT/pace" "$OUT/bspline" \
        --labels=baseline,demospeedup,pace,bspline --json="$OUT/comparison.json" \
        2>&1 | tee "$OUT/comparison.txt"
else
    echo "[$(date '+%F %T')] $(basename "$0"): arms still outstanding, no report yet"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "[$(date '+%F %T')] failed arms: ${failed[*]}" >&2
    exit 1
fi
