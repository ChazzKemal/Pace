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
| 1 | PACE speed step (`methods/pace/speed.py`, `processor.py`) | bit-exact parity vs the fork's `select_speed`, 90 golden cases | ✅ committed |
| 2 | SpeedActuator + eval runner (`run_libero.py`) | 3-seed look4cb+skip@1.5× reproduces recorded SR/TPR | ✅ committed; **gate parked by user** (`git show 2789b0e` era `run_gate_b2.sh` reconstructs it) |
| 3 | `--method.type` config (`methods/config.py`) | baseline unchanged through the new plumbing | ✅ committed |
| 4 | DemoSpeedup training step (tail-walk retiming) | upstream-parity tests (**only with `DEMOSPEEDUP_UPSTREAM` set**, see below) + on-the-fly == labels | ✅ committed (`0b31a80` + follow-ups) |
| 5 | `TimedActions` contract (`timed.py`) | baseline dt=1/fps is a byte-level no-op | ✅ committed (`432b712`) |
| 6 | real env: crisp forks referenced, same LeRobot SHA (`real/pixi.toml`) | full pixi solve; NEP 50 in installed env | ✅ committed (`b303153`); fake-mode diff **deferred to deploy day** |
| 7 | B-spline, real (ACT) | matches `merged_bspline_20260528` reconstruction | ⏳ not started; source = github.com/B-spline-policy/bspline-policy (user decision), Yunfei impl archived at crisp_gym fork `45dbb06` |
| 8 | B-spline on xVLA (`bspline_ee6d`) | trains and reconstructs | ⏳ not started |
| 9 | DemoSpeedup stage 2 in-repo (`methods/demospeedup/run_label.py`) | bit-exact vs upstream's `hdbscan_with_custom_merge`, 6 golden traces; 1-episode run on the real cups checkpoint; ACT + xVLA + Diffusion oracles | ✅ committed |

## Implementation state (`src/robot_stack/`, 236 passed + 8 skipped of 244)

**The 8 skips are the entire DemoSpeedup upstream-parity suite** (`tests/test_demospeedup_upstream_parity.py`), which runs only when `DEMOSPEEDUP_UPSTREAM` points at a DemoSpeedup clone. A green bare `pytest` is therefore *not* evidence that batch 4's parity gate still holds — re-run it with the env var before trusting the walk against upstream.

| module | what it is |
|---|---|
| `methods/config.py` | `--method.type` via draccus ChoiceRegistry: `NoMethod` / `PaceMethod` / `DemoSpeedupMethod`; `POLICY_CHUNK_FIELDS` — typed registry mapping policy `type` → chunk fields (act/diffusion/xvla; unknown type = error); `adjust_policy` (chunk halving, idempotent so a resume does not re-halve, + `pad_mode`/loss-masking guard; runs before the datasets so the loader window is the halved chunk); `preprocessor_steps` preloads each episode's raw action table from the dataset and hands it to the retime step |
| `methods/pace/speed.py` | pure PACE decision math, bit-exact vs the fork (90 golden cases; `test_pace_parity.py` totals 94 tests — the other 4 guard the golden set itself) |
| `methods/pace/processor.py` | `PaceSpeedStep` (`pace_speed` registry name): per-chunk speeds + stride; publishes `SPEED_KEY` and, given `control_dt`, per-step `DT_KEY` |
| `methods/pace/actuator.py` | `RobosuiteSpeedActuator`: substep exhaust (quantized 25/n grid), gripper stroke ×speed×stride, kp∝s^exp / kd∝s^(exp/2); `apply_dt` (TimedActions view) |
| `methods/demospeedup/retime.py` | the stride walk (`keep_indices`, upstream-parity-tested **only under `DEMOSPEEDUP_UPSTREAM`** — skipped by default), `episode_keep_indices` (episode-level variant, kept to check the recorded real dataset), `retime_tail` (episode-tail walk → exactly one chunk, truncation not padding), `retime_chunk` (chunk-level reference) |
| `methods/demospeedup/processor.py` | `DemoSpeedupRetimeStep` (`demospeedup_retime`): receives the preloaded episode action table (built by `config.py`, first row) and substitutes each sample's chunk with its tail walk; runs pre-normalizer; construction-time label/action validation; `out_len` = trained chunk (ACT does not truncate) |
| `methods/demospeedup/labels.py` | label loading: `episode_<i>.npy` dir or parquet sidecar, auto-detected |
| `methods/demospeedup/entropy.py` | KDE entropy of sampled action chunks — the uncertainty DemoSpeedup labels on. Upstream's arithmetic minus its dead bandwidth-estimation branch and unused teacher-action return |
| `methods/demospeedup/segment.py` | entropy trace → binary labels: z-score, IsolationForest outlier interpolation, HDBSCAN over `(time, entropy)`, oversize-cluster splitting. `rule=` picks the cluster verdict (see below) |
| `methods/demospeedup/sampler.py` | one sampler per policy family. **ACT** is the only one needing machinery: its inference path pins the CVAE latent to `z=0`, so `LatentSamplingACT(ACT)` puts a forward-pre-hook on `encoder_latent_input_proj` to swap in a prior draw — upstream's `forward` runs unmodified, weights load `strict=True`. **xVLA** and **Diffusion** keep their randomness (flow-matching `x1=randn`, denoising prior), so `XVLAChunkSampler` / `DiffusionChunkSampler` just `broadcast()` the observation to N rows and call the policy's public `predict_action_chunk`, resetting first so each frame is judged alone |
| `methods/demospeedup/run_label.py` | draccus labelling runner: checkpoint's own preprocessor, per-frame chunk sampling, temporal aggregation over every chunk covering a frame, `speedup_labels/episode_<i>.npy` + the raw `entropy_<i>.npy` trace, `run_config.yaml` |
| `methods/demospeedup/actuator.py` | `DemoSpeedupTrackingActuator`: constant gains+gripper ×low_v, time untouched (upstream's high-gain-XML recipe + arm gains); per-step re-application (reset-proof) |
| `timed.py` | `TimedActions` contract: `dt` per action; `uniform`/`from_speeds`, `timestamps()` (exclusive cumsum, the UR10e's `(pose,t)` view), `duration()`; `DT_KEY` for pipelines |
| `train/run_train.py` | upstream `lerobot-train` + `--method.*`: wraps `make_train_eval_datasets` (capture dataset, halve chunk) and `make_pre_post_processors` (insert method steps pre-normalizer); calls `train.__wrapped__` (subclass fails upstream's identity check) |
| `eval/run_libero.py` | draccus eval runner, one task per output dir: method steps attached, actuator per method (PACE per-step / DemoSpeedup constant / none), env↔checkpoint ImageNet dedupe, IDENTITY-stats guard, sim-time recording, `run_config.yaml` |
| `eval/pace_policy.py` | `attach_pace`: instance-level select_action/reset rebinding (upstream rejects wrapper objects), speed queue, applied-speed log |
| `eval/sim_time.py` | `SimTimeRecorder` + vector-env re-arm (autoreset-safe per-episode sim durations) |
| `real/` (repo root, **not** under `src/`) | deploy env: pixi manifest + lock pinning lerobot @ the shared SHA, crisp fork SHAs, numpy<2 override; site network scripts incl. the cv2 libjpeg preload |

Outside this repo, load-bearing and NOT under robot_stack's git:
- crisp forks `ChazzKemal/crisp_{gym,py}` branch `robot-stack-pin`
  (`45dbb06` / `47b3d23`): byte-verified lab snapshots, pinned by `real/pixi.toml`.
  (Verified 2026-08-29: both branch heads still match those SHAs exactly.)

**The `lerobot_uncertainty` fork is no longer needed for the ACT pipeline.** Its
labelling stage — `utils/entropy.py`, `configs/label.py`, `scripts/lerobot_label.py`,
and the `sample_action_chunks` / `forward_with_latent` methods it added to LeRobot's
policy classes — is reimplemented here from the *original* DemoSpeedup instead
(`lingxiao-guo/DemoSpeedup` @ `34bd43a`, user decision 2026-08-29; a clean clone is
at `/home/batur/DemoSpeedup_orig`). Both pipeline scripts now run every stage in this
env, and the fork-compat checkpoint copy (stripping `use_peft` /
`pretrained_revision`) is gone with it. All three oracles — ACT, xVLA and Diffusion —
are in-repo, so nothing in the labelling path depends on the fork any more.

## Known gaps / deferred (with owners of the decision)

- **Upstream's cluster rule labels almost nothing non-precision.** Upstream reads
  `if np.mean(cluster_points[:, 1] < 1):` — the comparison is *inside* the mean, so
  it is the fraction of a cluster's frames below +1σ used as a truth value: one
  quiet frame spares the whole cluster. On a realistic trace that yields <5%
  non-precision and so essentially no speedup; what non-precision it does produce
  comes from HDBSCAN noise points. `rule="upstream"` is the default because it is
  the reference, and `rule="mean"` (mean entropy < 1σ, plus low-entropy noise
  treated as precision) is what the fork used and what both pipeline scripts pass.
  Verified on the real cups checkpoint: `rule=mean` gives 20.4% non-precision with
  runs of ~12.7 frames, against the fork's recorded 18.4% / 13.9.
- **Diffusion labelling needs an `n_obs_steps=1` proxy** — `DiffusionChunkSampler`
  refuses anything else at construction: per-frame labelling has no observation
  history to condition on, and the runner reads one frame per query. Note
  `DiffusionConfig`'s default is **2**, and the cups pipeline's diffusion stage does
  not override it — so that baseline cannot double as an oracle the way the ACT
  baseline does. Either train the proxy with `--policy.n_obs_steps=1` or extend the
  runner to read a window of frames per query.
- **No controlled cross-oracle comparison exists.** ACT is trained on stack_cups and
  xVLA on LIBERO, so their precision fractions (79.6% / 84.1%, episode 0, `rule=mean`)
  differ by dataset as much as by policy. Comparing oracles properly needs two
  families trained on one dataset; nothing schedules that.
- **xVLA labelling is ~28x slower per frame than ACT, and that now blocks LIBERO.**
  Measured 2026-08-29 on `ds_libero10_base`: 0.88 s/frame vs ACT's 0.032 s/frame, so
  a full `HuggingFaceVLA/libero` pass (1693 episodes, 273465 frames) is **~67 h**.
  The cause is the tiling: Florence2 encodes all N rows instead of encoding once and
  broadcasting its output. The fix is a ~5-line `forward_vlm` override on an
  `XVLAModel` subclass (encode row 0, expand to N), which leaves upstream's flow-
  matching loop untouched. Not done; profiling now justifies it.
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
- A preprocessed dataset frame carries 0-dim tensors (frame index, timestamp,
  domain id) alongside the batched ones. Anything walking a batch must skip them —
  `value.shape[0]` on a scalar is an `IndexError`, and it only shows up on a real
  dataset, not on a hand-built test batch.
- What the sampler returns is not what the policy config says: ACT hands back
  `chunk_size` steps, a diffusion policy hands back `n_action_steps`. Size chunk
  windows from the returned tensor.
- xVLA truncates over-length action inputs; ACT does not (`out_len` exists for
  this reason).
- A draccus entry point must NOT use `from __future__ import annotations`: the
  stringified annotation stops draccus resolving the config dataclass, and it fails
  in the argument parser with a bare `TypeError: must be called with a dataclass`.
- `logging.basicConfig` is a no-op once lerobot's import has installed a root
  handler — pass `force=True` or the run prints no progress at all.
- Batch shape cannot be inferred by counting dimensions: a batched camera image and
  an unbatched action chunk are both 3-D. Reshape against the policy config's
  declared feature shapes, the same way chunk fields come from
  `POLICY_CHUNK_FIELDS`.
