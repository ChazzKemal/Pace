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
| B-spline        | —               | spline action space                                        | not started |

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

End-to-end pipelines for the recorded experiments are in `run_demospeedup_*.sh` and
`eval_demospeedup_libero10.sh`; each stage skip-guards, so a killed run resumes.

## Layout

```
src/pace_bench/
  methods/
    config.py          --method.type registry; chunk geometry per policy family
    pace/              speed decision math, processor step, robosuite actuator
    demospeedup/       entropy, segmentation, chunk samplers, retiming, labelling
  train/run_train.py   lerobot-train + --method.*
  eval/run_libero.py   LIBERO eval runner
  timed.py             TimedActions — per-action dt, the real robot's (pose, t) view
real/                  pixi manifest + lock for the UR10e deploy environment
docs/PLAN.md           batch plan, current state, and decisions worth not relearning
```

## Results so far

LIBERO-10, xVLA, 20 episodes × 10 tasks:

| run                | success rate | episode time | speedup |
| ------------------ | ------------ | ------------ | ------- |
| baseline           | 92.0 %       | 13.28 s      | 1.00×   |
| DemoSpeedup        | 86.5 %       | 6.99 s       | 1.90×   |

PACE's LIBERO A/B has not been rerun on this stack. Real-robot runs are pending
deployment. See `docs/PLAN.md` for what is trained, what is not, and why.

## Tests

```bash
pytest
```

239 tests, no network, no external checkouts, nothing skipped. They include parity
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
