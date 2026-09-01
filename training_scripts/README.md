# training_scripts

Every end-to-end run in this project, and the SLURM jobs that submit them.

| task | queue script | SLURM job | arms |
|---|---|---|---|
| pickplace (UR10e, 45 eps) | `run_demospeedup_pickplace.sh` | `slurm_pickplace.sbatch` | 6 training + 2 labelling |
| stack cups, merged (UR10e, 175 eps) | `run_demospeedup_stackcups_merged.sh` | `slurm_stackcups.sbatch` | 6 training + 2 labelling |
| LIBERO-10 (xVLA, 400 demos) | `run_demospeedup_libero10.sh` | `slurm_libero10.sbatch` | 3 training |
| LIBERO-10 evaluation | `eval_demospeedup_libero10.sh` | `slurm_eval_libero10.sbatch` | 2 rollout sweeps |
| stack cups, unmerged | `run_demospeedup_stackcups.sh` | — (superseded, see below) | 3 training |
| chain behind a running job | `wait_then_run_cups_merged.sh` | — (workstation only) | — |

The jobs are thin: a `#SBATCH` header, then they run the queue script beside them.
Every skip guard, every training argument and all the reasoning behind them stays in
the queue script, so a cluster run and a workstation run execute the same code and
cannot drift apart. `_slurm_common.sh` holds the part that is genuinely about SLURM —
locating the checkout, checking the environment, and the wall clock.

## Before the first submit

Each `.sbatch` has a resource block with two lines marked `CHANGE ME`:

```bash
#SBATCH --partition=gpu              # CHANGE ME: your GPU partition
##SBATCH --account=CHANGEME          # uncomment and set if your site bills accounts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # --num_workers=4 loaders + the main process
#SBATCH --mem=64G
#SBATCH --time=24:00:00
```

Edit them, or override on the command line — `sbatch` flags beat script directives:

```bash
sbatch --partition=a100 --gres=gpu:a100:1 --time=48:00:00 \
       training_scripts/slurm_pickplace.sbatch
```

A card with **24GB or more** is assumed throughout: the queues were written against
one, and the Diffusion arms at batch 32 were sized to fit it. If your site needs
`module load cuda/...`, uncomment the line in each job — the venv itself is never
activated, because the queue scripts invoke `.venv/bin/python` directly.

## Submitting

Submit **from the checkout root**. The job reads `$SLURM_SUBMIT_DIR` to find the
repo, because SLURM spools a copy of the script to the compute node and its own path
therefore says nothing about where the repo is. `PACE_REPO` overrides it.

```bash
mkdir -p logs/slurm                       # SLURM will not create it, and a job whose
                                          # --output cannot be opened fails silently
sbatch training_scripts/slurm_pickplace.sbatch
```

Chaining training and evaluation in one go:

```bash
TRAIN=$(sbatch --parsable training_scripts/slurm_libero10.sbatch)
sbatch --dependency=afterok:$TRAIN training_scripts/slurm_eval_libero10.sbatch
```

Progress goes to `logs/slurm/<job>-<id>.out` (the job's own narration) and to
`logs/<arm>.log` (the trainer's, appended across attempts).

## The wall clock

A task is six to eight stages back to back — roughly **40 hours** for either UR10e
queue by the scripts' own accounting — so it will not fit in one job on most
partitions. The jobs handle that themselves:

```
#SBATCH --signal=B:USR1@300     SLURM warns the batch shell 5 min before the limit
#SBATCH --requeue               the job is allowed back into the queue
trap _requeue USR1              the trap calls `scontrol requeue $SLURM_JOB_ID`
```

The running arm dies with the job. Its last checkpoint is at most `save_freq` steps
back, the queue script resumes it, and arms already at their full budget are skipped
— so the queue walks forward one job at a time until it finishes. **Being interrupted
mid-arm is the designed path here, not an accident.**

That is also why the skip guards check the *step count* rather than the existence of
`checkpoints/last`. A bare directory check would call an arm cut at step 40k
"already trained" and hand the comparison a 40k arm sitting beside a 100k one.
`arm_state` resolves the checkpoint symlink and returns `done` / `resume` / `fresh`.

Two consequences worth knowing:

* **Resume is sample-exact.** Step, RNG, optimizer moments and the `EpisodeAwareSampler`
  offset are all restored, and Diffusion's cosine LR schedule comes back from
  `scheduler_state.json`, so a resumed arm continues down the same curve rather than
  re-entering warmup. The arm that finishes is the arm the queue asked for.
* **The eval job does not requeue**, deliberately. A rollout sweep has no checkpoint
  to resume from and nothing to skip, so requeueing would restart it and overwrite
  `outputs/eval/`. Raise `--time` instead.

## Overrides

All of these are read by the queue scripts, so they work identically under SLURM and
on a workstation. Pass them with `sbatch --export=ALL,VAR=value` or export them
before submitting.

| variable | what it moves | default |
|---|---|---|
| `PACE_REPO` | the checkout | `$SLURM_SUBMIT_DIR` |
| `PACE_DATA_ROOT` | the `data/` tree | `data/` beside the checkout |
| `PICKPLACE_ROOT` | the pickplace dataset | under `PACE_DATA_ROOT` |
| `STACK_CUPS_MERGED_ROOT` | the merged stack-cups dataset | under `PACE_DATA_ROOT` |
| `LIBERO10_ROOT` | the LIBERO dataset | under `PACE_DATA_ROOT` |
| `LIBERO10_LABELS_PATH` | the stage-2 labels | under `PACE_DATA_ROOT` |
| `XVLA_POLICY_PATH` | the pretrained xVLA | hub id `lerobot/xvla-libero` |
| `WANDB_PROJECT` | the wandb project | `pace_benchmark_<task>` |
| `PACE_MAX_ATTEMPTS` | relaunches per arm before giving up | 6 |
| `PACE_PRUNER` | the checkpoint pruner | `src/pace_bench/data/prune_checkpoints.py` |

**Offline compute nodes.** `slurm_libero10.sbatch` defaults to the hub id
`lerobot/xvla-libero` and will try to download it. If the node has no network,
pre-populate the hub cache from the login node or point `XVLA_POLICY_PATH` at a local
copy — the queue's `require` guards check local paths before training starts, but they
cannot check a hub id.

## Running without SLURM

The queue scripts are unchanged and still stand alone:

```bash
./training_scripts/run_demospeedup_pickplace.sh
```

They resolve the repo from their own location — one level up from here — so they can
be run from anywhere.

## Why `run_demospeedup_stackcups.sh` has no job

It trains on the 12-episode `stack_cups_20260828` recording and is superseded by
`run_demospeedup_stackcups_merged.sh`, which covers the same task with 175 episodes
and carries the full 2x3. It is kept for provenance rather than for running, so it
still has the older bare-directory skip guard: that guard is safe when a run is only
ever interrupted between arms, and **not** safe under requeue, where it would freeze
an interrupted arm at whatever step the wall clock caught it. Porting `arm_state`
into it is a small change if that queue is ever wanted on the cluster.
