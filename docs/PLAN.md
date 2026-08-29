# robot_stack — batch plan and state

One config-driven stack for LIBERO (sim) and the UR10e (real). Scope, user-fixed:
sim = LIBERO-10 + xVLA; real = UR10e + ACT/Diffusion; methods = PACE, DemoSpeedup,
B-spline. RA-BC is not an axis. LeRobot/robosuite/crisp are pinned dependencies,
never vendored or edited.

Ordering principle: everything verifiable without a GPU first; every batch has a
gate that fails loudly. One batch = one reviewable commit set; the user commits.

## Batches

| # | batch | gate | state |
|---|---|---|---|
| 0 | skeleton + pinned LeRobot (`bf31dd79`) | uv sync; xVLA loads; ACT trains | ✅ committed |
| 1 | PACE speed step (`methods/pace/speed.py`, `processor.py`) | bit-exact parity vs the fork's `select_speed`, 94 golden cases | ✅ committed |
| 2 | SpeedActuator + eval runner (`run_libero.py`) | 3-seed look4cb+skip@1.5× reproduces recorded SR/TPR | ✅ committed; **gate parked by user** (`git show 2789b0e` era `run_gate_b2.sh` reconstructs it) |
| 3 | `--method.type` config (`methods/config.py`) | baseline unchanged through the new plumbing | ✅ committed |
| 4 | DemoSpeedup training step (tail-walk retiming) | upstream-parity tests + on-the-fly == labels | ✅ committed (`0b31a80` + follow-ups) |
| 5 | `TimedActions` contract (`timed.py`) | baseline dt=1/fps is a byte-level no-op | ✅ committed (`432b712`) |
| 6 | real env: crisp forks referenced, same LeRobot SHA (`real/pixi.toml`) | full pixi solve; NEP 50 in installed env | ✅ committed (`b303153`); fake-mode diff **deferred to deploy day** |
| 7 | B-spline, real (ACT) | matches `merged_bspline_20260528` reconstruction | ⏳ not started; source = github.com/B-spline-policy/bspline-policy (user decision), Yunfei impl archived at crisp_gym fork `45dbb06` |
| 8 | B-spline on xVLA (`bspline_ee6d`) | trains and reconstructs | ⏳ not started |

## Implementation state (`src/robot_stack/`, 153 tests passing)

| module | what it is |
|---|---|
| `methods/config.py` | `--method.type` via draccus ChoiceRegistry: `NoMethod` / `PaceMethod` / `DemoSpeedupMethod`; `POLICY_CHUNK_FIELDS` — typed registry mapping policy `type` → chunk fields (act/diffusion/xvla; unknown type = error); `adjust_policy_after_datasets` (chunk halving) |
| `methods/pace/speed.py` | pure PACE decision math, bit-exact vs the fork (94 golden cases) |
| `methods/pace/processor.py` | `PaceSpeedStep` (`pace_speed` registry name): per-chunk speeds + stride; publishes `SPEED_KEY` and, given `control_dt`, per-step `DT_KEY` |
| `methods/pace/actuator.py` | `RobosuiteSpeedActuator`: substep exhaust (quantized 25/n grid), gripper stroke ×speed×stride, kp∝s^exp / kd∝s^(exp/2); `apply_dt` (TimedActions view) |
| `methods/demospeedup/retime.py` | the stride walk (`keep_indices`, upstream-parity-tested), `retime_tail` (episode-tail walk → exactly one chunk, truncation not padding), `retime_chunk` (chunk-level reference) |
| `methods/demospeedup/processor.py` | `DemoSpeedupRetimeStep` (`demospeedup_retime`): preloads the full episode action table, substitutes each sample's chunk with its tail walk; runs pre-normalizer; construction-time label/action validation; `out_len` = trained chunk (ACT does not truncate) |
| `methods/demospeedup/labels.py` | label loading: `episode_<i>.npy` dir or parquet sidecar, auto-detected |
| `methods/demospeedup/actuator.py` | `DemoSpeedupTrackingActuator`: constant gains+gripper ×low_v, time untouched (upstream's high-gain-XML recipe + arm gains); per-step re-application (reset-proof) |
| `timed.py` | `TimedActions` contract: `dt` per action; `uniform`/`from_speeds`, `timestamps()` (exclusive cumsum, the UR10e's `(pose,t)` view), `duration()`; `DT_KEY` for pipelines |
| `train/run_train.py` | upstream `lerobot-train` + `--method.*`: wraps `make_train_eval_datasets` (capture dataset, halve chunk) and `make_pre_post_processors` (insert method steps pre-normalizer); calls `train.__wrapped__` (subclass fails upstream's identity check) |
| `eval/run_libero.py` | draccus eval runner, one task per output dir: method steps attached, actuator per method (PACE per-step / DemoSpeedup constant / none), env↔checkpoint ImageNet dedupe, IDENTITY-stats guard, sim-time recording, `run_config.yaml` |
| `eval/pace_policy.py` | `attach_pace`: instance-level select_action/reset rebinding (upstream rejects wrapper objects), speed queue, applied-speed log |
| `eval/sim_time.py` | `SimTimeRecorder` + vector-env re-arm (autoreset-safe per-episode sim durations) |
| `real/` | deploy env: pixi manifest + lock pinning lerobot @ the shared SHA, crisp fork SHAs, numpy<2 override; site network scripts incl. the cv2 libjpeg preload |

Outside this repo, load-bearing and NOT under robot_stack's git:
- fork `lerobot_uncertainty` (uncommitted working tree): `lerobot-label` stage 2,
  plus this week's fixes to `ACTPolicy.sample_action_chunks` (non-tensor batch
  entries passed through; missing batch dims normalized) — labelling is proven
  end-to-end with both xVLA and ACT oracles. These fixes are uncommitted there.
- crisp forks `ChazzKemal/crisp_{gym,py}` branch `robot-stack-pin`
  (`45dbb06` / `47b3d23`): byte-verified lab snapshots, pinned by `real/pixi.toml`.

## Known gaps / deferred (with owners of the decision)

- **DemoSpeedup stage 2 (entropy labelling) is not ported** — it runs in the fork
  (`lerobot_uncertainty`, conda `lerobot` env) against checkpoints made
  fork-compatible by stripping `use_peft`/`pretrained_revision` from config.json.
  Works (stack_cups labelled); porting it into robot_stack is unscheduled.
- **Fake-mode dataset diff** — deploy-day gate on the lab machine, where the
  baseline env exists (user decision 2026-08-28).
- **Real-inference gripper postprocessors** — PACE: speed=1 during gripper motion;
  DemoSpeedup: repeat gripper-moving actions ×low_v, truncate. Config-gated,
  never serialized into checkpoints. See memory note; not implemented.
- **Batch 2's 3-seed PACE gate** — parked; upstream xVLA has a known ~10pp task-1
  deficit vs the fork that would confound absolute-SR comparison.

## Experiments state (2026-08-29 ~17:00, all runs STOPPED by user)

- **LIBERO A/B (xVLA)**: complete. Baseline 92.0% SR / 13.28 s vs DemoSpeedup
  86.5% / 6.99 s = 1.90× at −5.5 pp; task 2 (−30 pp) is the known speed-intolerant
  task. `outputs/eval/ds_libero10_*`.
- **stack_cups**: ACT baseline ✅ (`outputs/train/cups_act_base`, 30k), labels ✅
  (12/12, 18.4% non-precision, run-length 13.9 vs 1.23 random — real signal),
  DemoSpeedup ACT ❌ not trained (crashed pre-fix, stale dir),
  Diffusion baseline ❌ not trained. Resume: `./run_demospeedup_stackcups.sh`
  (skip-guards resume at first unfinished stage).
- **pickplace**: nothing trained; `./run_demospeedup_pickplace.sh` runs the full
  fresh-baseline chain (baseline 100k → label → speedup 100k). The Yunfei oracle
  is deliberately NOT used (user decision 2026-08-29).

## Conventions that bit us (do not relearn)

- Verify pipeline claims by running the pipeline's entry point, not its
  components (a 1-episode `lerobot-label` smoke would have saved a night).
- Check log timestamps before believing a tail; stale logs have caused two
  false diagnoses.
- `pkill -f` matches your own wrapper shell; filter by `/proc/<pid>/exe`.
- Policy chunk-field names come from `POLICY_CHUNK_FIELDS` (typed registry),
  never from attribute probing.
- xVLA truncates over-length action inputs; ACT does not (`out_len` exists for
  this reason).
