# pace_bench

A benchmark harness for **PACE**, an eval-time speed-modulation method for
imitation-learning policies, measured against other speedup methods on the same
datasets, policies and metrics.

One config-driven stack covers both settings the methods are evaluated in:

|          | simulation                | real robot                     |
| -------- | ------------------------- | ------------------------------ |
| platform | LIBERO-10, robosuite      | UR10e + Robotiq, via CRISP     |
| policy   | xVLA                      | ACT, Diffusion                 |
| env      | `uv` (`pyproject.toml`)   | `pixi` (`real/pixi.toml`)      |

A method is selected with a single flag, `--method.type`, and contributes pipeline
steps to training, inference, or both. Everything else — dataset, policy, training
loop — stays exactly as upstream LeRobot provides it.

| method          | `--method.type` | what it does                                              | state       |
| --------------- | --------------- | --------------------------------------------------------- | ----------- |
| baseline        | `none`          | stock policy, no steps                                     | ✅          |
| **PACE**        | `pace`          | per-chunk speed + stride at inference, from action geometry | ✅          |
| DemoSpeedup     | `demospeedup`   | entropy-labels demos, retimes training targets             | ✅          |
| B-spline        | `bspline`       | spline action space, decoded at eval — the speed lever      | ✅ trains; no arm run to completion |

## Install

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync
```

The cmake variable is required: `lerobot[libero]` pulls `egl-probe` 1.0.2, whose
`CMakeLists.txt` declares a pre-3.5 `cmake_minimum_required` that CMake ≥ 4 rejects.
Python is pinned to 3.12 in `.python-version` to match the robot side's ROS Jazzy.

## Usage

Three stages, each a module you run directly, all taking upstream LeRobot's flags
plus `--method.*`. Where a method acts differs: **DemoSpeedup changes what a policy
is trained on**, so it is selected at training time; **PACE changes how a chunk is
executed**, so it is selected at evaluation time.

**Train** — upstream `lerobot-train`, with a training-time method attached:

```bash
python -m pace_bench.train.run_train \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --policy.type=act --policy.chunk_size=100 \
    --method.type=demospeedup --method.labels_path=outputs/label/run/speedup_labels \
    --output_dir=outputs/train/run
```

With `--method.type=none` this is a plain baseline run, byte-for-byte unchanged by
the added plumbing.

**Label** — DemoSpeedup stage 2, and a prerequisite for the run above. Measures a
trained policy's action uncertainty at every frame and segments it into precision /
non-precision labels:

```bash
python -m pace_bench.methods.demospeedup.run_label \
    --policy_path=outputs/train/baseline/checkpoints/last/pretrained_model \
    --dataset_repo_id=local/stack_cups --dataset_root=/path/to/dataset \
    --rule=mean --out=outputs/label/run
```

Any of ACT, xVLA or Diffusion can serve as the proxy policy. Both the labels and the
raw entropy trace are written, so a run can be re-segmented without re-querying the
policy.

**Evaluate** on LIBERO, one task per output directory, recording success rate and
per-episode sim time:

```bash
python -m pace_bench.eval.run_libero \
    --policy_path=outputs/train/run/checkpoints/last/pretrained_model \
    --method.type=pace --method.max_speed=1.5 --method.action_stride=2 \
    --n_action_steps=30 --out=outputs/eval/run
```

**Compare** two or more eval runs. LIBERO ships 50 fixed initial states per task and
the evaluator walks them in order, so two runs sharing `(task_suite, tasks,
n_episodes, batch_size, seed)` attempted the *same scenes in the same order*. The
comparison is therefore paired — McNemar on the episodes the two runs disagree about,
and sim time over the episodes both solved — rather than two independent rates:

```bash
python -m pace_bench.eval.compare_libero \
    outputs/eval/ds_libero10_base outputs/eval/ds_libero10_speedup \
    --labels=baseline,demospeedup --json=outputs/eval/comparison.json
```

It closes with a summary block: the difference in success rate with a bootstrap CI,
Holm-adjusted McNemar p-values when several arms share one baseline, the paired
speedup with a sign test over the episodes both runs solved, and one sentence per arm
saying what the numbers do and do not support. Runs whose configuration would put
them on different scenes are reported side by side but explicitly *not* paired, and
get no verdict.

Those 50 initial states are the benchmark's held-out axis — they are **not** the
states the demonstrations start from. Checked against the raw LIBERO demos: 0 of 500
demo start states coincides with an eval init state, and on the columns the scene
randomizer varies, a demo is exactly as far from its nearest init state as init
states are from each other. So a LIBERO run tests unseen object placements, but the
n_episodes it uses is a *prefix* of the 50 — `--n_episodes=50` is the protocol the
literature reports.

End-to-end pipelines for the recorded experiments live in `training_scripts/`
(`run_demospeedup_*.sh`, `eval_demospeedup_libero10.sh`, and the `slurm_*.sbatch`
jobs that submit them); each stage skip-guards, so a killed run resumes. They hold
no absolute paths: each resolves the repo from its own location and reads everything
it does not produce from `data/` beside the checkout —

```
data/                      (not in git; PACE_DATA_ROOT points elsewhere)
  datasets/real/           UR10e recordings
  datasets/sim/            libero_10_ee6d
  checkpoints/             (empty; xVLA loads from the hub as lerobot/xvla-libero)
  labels/                  DemoSpeedup stage-2 output, per labelling run
```

Each input is also individually overridable; a script names the variable to set when
a directory is missing, and stops before training rather than part way through.

## Runtimes

All on one 24 GB card, from the run logs. Training:

| run                                  | steps × batch | wall clock |
| ------------------------------------ | ------------- | ---------- |
| xVLA LoRA, LIBERO-10 (102k frames)   | 20k × 8       | 1.9 h      |
| ACT, stack_cups (8.9k frames)        | 30k × 32      | 2.2 h      |
| ACT, pickplace (31k frames), bf16    | 100k × 32     | 5.9 h *    |

Labelling is the stage whose cost swings by orders of magnitude, because it queries
the proxy policy once per frame:

| proxy run                    | frames  | wall clock                                   |
| ---------------------------- | ------- | -------------------------------------------- |
| ACT, stack_cups              | 8 875   | 5.5 min (0.038 s/frame)                      |
| Diffusion, pickplace         | 31 178  | 4.8 h at `--batch_frames=1` → **1.4 h** at 32 |
| xVLA, libero_10_ee6d         | 102 033 | 17.1 h (0.60 s/frame, one frame per call)    |

`*` projected from the observed rate, not run to completion.

`--batch_frames` (default 32) samples several frames per policy call, for any family
whose sampler implements `sample_frames`. Diffusion is where it pays: one chunk costs
100 sequential denoising steps however wide the batch is, so the card idles at width
10. Past ~32 it plateaus — the denoiser is latency-bound, not throughput-bound. ACT
drives the model directly, is fast enough not to need it, and takes the original
one-frame-at-a-time path. xVLA inherits the generic `sample_frames`, so the flag
applies to it as well, but its gain has not been measured — the number above predates
the batching. Only diffusion has the specialised path that encodes each frame once
rather than once per sample, worth ~6× in memory, which is what makes a wide frame
batch fit at all.

The LIBERO labels in `data/labels/xvla_libero10_ee6d` came from the fork's stage-2
run over those same 400 episodes and are consumed as given; at ~150 s per episode,
regenerating them through `run_label` is an overnight job, not an interactive one.

## Layout

```
src/pace_bench/
  methods/
    config.py          --method.type registry; chunk geometry per policy family
    pace/              speed decision math, processor step, robosuite actuator
    demospeedup/       entropy, segmentation, chunk samplers, retiming, labelling
    bspline/           spline fit, action layouts, decode, eval-time actuation
  data/                recordings → training sets: crop stalls, merge, reshape columns
    specs.py           what a policy may consume, per robot — checked in tests
  train/run_train.py   lerobot-train + --method.*
  eval/run_libero.py   LIBERO eval runner
  eval/compare_libero.py  paired comparison across LIBERO eval runs
  timed.py             TimedActions — per-action dt, the real robot's (pose, t) view
real/                  pixi manifest + lock for the UR10e deploy environment
docs/PLAN.md           batch plan, current state, and decisions worth not relearning
```

### What a policy is allowed to see

LeRobot *lists* every `observation.*` key in `input_features`
(`utils/feature_utils.py:170`), but a policy reads less than that, and the gap is
worth knowing: the robot state is matched by **exact name** (`configs/policies.py:137`,
`ft_name == OBS_STATE`), so extra scalar columns are inert — while `image_features`
returns **every** VISUAL feature, so an extra camera silently adds a backbone, and a
substituted `observation.state` of the wrong width is read as if it were the right one.

`data/specs.py` names the inputs per robot, `TRAINING_SETS` maps each dataset an arm
trains on to its spec, and `tests/test_dataset_specs.py` checks every one present on
the machine. A merge **allowlists**: it names the features it wants and never looks
at the rest, so a column added to the recorder later cannot leak into a training set.

## Results so far

LIBERO-10, xVLA, 20 episodes × 10 tasks:

| run                | success rate | episode time | speedup |
| ------------------ | ------------ | ------------ | ------- |
| baseline           | 92.0 %       | 13.27 s      | 1.00×   |
| DemoSpeedup        | 85.5 %       | 6.94 s       | 1.91×   |

Straight out of `pace_bench.eval.compare_libero`, which pools all 200 episodes; the same
runs read 92.0 % / 85.5 % and 13.28 s / 6.99 s if the ten per-task figures are
averaged unweighted instead. Paired over the 200 shared init states, DemoSpeedup
loses 21 episodes the baseline solved and wins 8 (McNemar p = 0.024) — the drop is
real, not sampling noise.

PACE's LIBERO A/B has not been rerun on this stack. Real-robot runs are pending
deployment. See `docs/PLAN.md` for what is trained, what is not, and why.

## Tests

```bash
pytest
```

386 tests, no network and no external checkouts; the one skip is a checkpoint
probe that runs only where a local LeRobot checkpoint exists. They include parity
checks of the DemoSpeedup ports against that paper's own code, which is copied
verbatim into `tests/upstream_reference.py` with its provenance.

## LeRobot is a dependency, not a fork

LeRobot is pinned by SHA in `pyproject.toml` and never edited; `real/pixi.toml`
pins the same SHA, so simulation and robot run one LeRobot. Methods attach through
public registries only — `draccus.ChoiceRegistry` for the config choice,
`ProcessorStepRegistry` for pipeline steps. Because processor pipelines are saved
beside the weights, a method declared at training time is rebuilt automatically at
inference, with nothing to re-specify on the robot.

The lab's diverged CRISP state is handled the same way: pinned by SHA on the
`robot-stack-pin` branches of `ChazzKemal/crisp_gym` and `ChazzKemal/crisp_py`,
referenced and never vendored.
