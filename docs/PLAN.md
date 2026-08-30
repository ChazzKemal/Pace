# pace_bench — batch plan and state

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
| 4 | DemoSpeedup training step (tail-walk retiming) | walk parity vs the copied reference (500 sequences, runs by default) + on-the-fly == labels; recorded-dataset reconstruction | ✅ committed (`0b31a80` + follow-ups) |
| 5 | `TimedActions` contract (`timed.py`) | baseline dt=1/fps is a byte-level no-op | ✅ committed (`432b712`) |
| 6 | real env: crisp forks referenced, same LeRobot SHA (`real/pixi.toml`) | full pixi solve; NEP 50 in installed env | ✅ committed (`b303153`); fake-mode diff **deferred to deploy day** |
| 7 | B-spline, real (ACT) | matches `merged_bspline_20260528` reconstruction | ✅ gate met (all 70 episodes reconstruct exactly) **and integrated**: `--method.type=bspline` trains on 2 datasets × 2 policy families. Not yet trained to completion |
| 8 | B-spline on xVLA (`bspline_ee6d`) | trains and reconstructs | ✅ trains (`--method.arrangement=xvla_ee6d20 --policy.action_mode=ee6d_bspline`); `KNOT_SCALE` untuned. Not trained to completion |
| 9 | DemoSpeedup stage 2 in-repo (`methods/demospeedup/run_label.py`) | bit-exact vs upstream's `hdbscan_with_custom_merge`, 6 golden traces; 1-episode run on the real cups checkpoint; ACT + xVLA + Diffusion oracles | ✅ committed |

## Implementation state (`src/pace_bench/`, 317 passed, 0 skipped)

The suite no longer skips anything and needs no network or external checkout. The
DemoSpeedup repo is not a dependency in any form (user decision 2026-08-29): the
three functions worth checking against are copied verbatim into
`tests/upstream_reference.py` with their provenance, and both parity tests execute
them live — segmentation in `test_demospeedup_segment.py`, the stride walk (500
random label sequences) in `test_demospeedup_processor.py`. Neither skips.

| module | what it is |
|---|---|
| `methods/config.py` | `--method.type` via draccus ChoiceRegistry: `NoMethod` / `PaceMethod` / `DemoSpeedupMethod`; `POLICY_CHUNK_FIELDS` — typed registry mapping policy `type` → chunk fields (act/diffusion/xvla; unknown type = error); `adjust_policy` (chunk halving, idempotent so a resume does not re-halve, + `pad_mode`/loss-masking guard; runs before the datasets so the loader window is the halved chunk); `preprocessor_steps` preloads each episode's raw action table from the dataset and hands it to the retime step |
| `methods/pace/speed.py` | pure PACE decision math, bit-exact vs the fork (90 golden cases; `test_pace_parity.py` totals 94 tests — the other 4 guard the golden set itself) |
| `methods/pace/processor.py` | `PaceSpeedStep` (`pace_speed` registry name): per-chunk speeds + stride; publishes `SPEED_KEY` and, given `control_dt`, per-step `DT_KEY` |
| `methods/pace/actuator.py` | `RobosuiteSpeedActuator`: substep exhaust (quantized 25/n grid), gripper stroke ×speed×stride, kp∝s^exp / kd∝s^(exp/2); `apply_dt` (TimedActions view) |
| `methods/demospeedup/retime.py` | the stride walk (`keep_indices`, parity-tested against the copied reference at its `start=-1` convention), `episode_keep_indices` (episode-level variant, kept to check the recorded real dataset), `retime_tail` (episode-tail walk → exactly one chunk, truncation not padding), `retime_chunk` (chunk-level reference) |
| `methods/demospeedup/processor.py` | `DemoSpeedupRetimeStep` (`demospeedup_retime`): receives the preloaded episode action table (built by `config.py`, first row) and substitutes each sample's chunk with its tail walk; runs pre-normalizer; construction-time label/action validation; `out_len` = trained chunk (ACT does not truncate) |
| `methods/demospeedup/labels.py` | label loading: `episode_<i>.npy` dir or parquet sidecar, auto-detected |
| `methods/demospeedup/entropy.py` | KDE entropy of sampled action chunks — the uncertainty DemoSpeedup labels on. Upstream's arithmetic minus its dead bandwidth-estimation branch and unused teacher-action return |
| `methods/demospeedup/segment.py` | entropy trace → binary labels: z-score, IsolationForest outlier interpolation, HDBSCAN over `(time, entropy)`, oversize-cluster splitting. `rule=` picks the cluster verdict (see below) |
| `methods/demospeedup/sampler.py` | one sampler per policy family. **ACT** is the only one needing machinery: its inference path pins the CVAE latent to `z=0`, so `LatentSamplingACT(ACT)` puts a forward-pre-hook on `encoder_latent_input_proj` to swap in a prior draw — upstream's `forward` runs unmodified, weights load `strict=True`. **xVLA** and **Diffusion** keep their randomness (flow-matching `x1=randn`, denoising prior), so `XVLAChunkSampler` / `DiffusionChunkSampler` just `broadcast()` the observation to N rows and call the policy's public `predict_action_chunk`, resetting first so each frame is judged alone. `sample_frames` on `_BroadcastChunkSampler` answers several frames in one policy call (same distribution, different point in the RNG stream); `DiffusionChunkSampler` overrides it to run the vision encoder once per *frame* rather than once per sample, which is what lets a wide frame batch fit in memory |
| `methods/demospeedup/run_label.py` | draccus labelling runner: checkpoint's own preprocessor, per-frame chunk sampling, temporal aggregation over every chunk covering a frame, `speedup_labels/episode_<i>.npy` + the raw `entropy_<i>.npy` trace, `run_config.yaml`; `--batch_frames` (default 32) routes through `sample_frames` for families that implement it, and ACT keeps the one-frame-at-a-time path |
| `methods/demospeedup/actuator.py` | `DemoSpeedupTrackingActuator`: constant gains+gripper ×low_v, time untouched (upstream's high-gain-XML recipe + arm gains); per-step re-application (reset-proof) |
| `methods/bspline/spline.py` | the B-spline action representation (`B-spline-policy/bspline-policy` @ `61ed5f4`, arXiv:2607.09648). `fit_episode` (adaptive least-squares fit: `generate_knots` grows the knot count until the spline is within `max_error`), `chunk_parameters` (cut the fit into `(chunk_size + 2*degree, 1 + dim)` windows of the *episode's* knot vector — only the first carries the clamped boundary), `assign_chunks_to_frames` (one matrix per frame, knots shifted to offsets from that frame), `decode_chunk` (evaluate the curve at `num_actions` points across its span — this is the speed knob), `to/from_spline_actions` (cart7 ↔ xyz+rot6d+gripper, with Gram-Schmidt back onto SO(3)), `encode/decode_relative_knots` (the knot column as consecutive differences). Parity against `tests/upstream_reference_bspline.py` |
| `methods/bspline/layout.py` | which columns of a dataset's action can be splined, named per run: `cart7` (UR10e, angle-axis → rot6d), `ee6d20` (LIBERO, already rot6d, 10 zero-pad columns dropped and restored), `identity` (joint space). Checked against the dataset's real action width, because naming the wrong one is otherwise silent — a transposed rotation still fits and still decodes |
| `methods/bspline/xvla_action.py` | `ee6d_bspline`, registered in xVLA's own action registry: single-arm ee6d control point in slots 0-9 where its structured loss expects xyz/rot6d/gripper, B-spline knot in slot 10 with a fourth loss term. Needed because xVLA slices its action by hardcoded index, so upstream's knot-first matrix trains a *time* as an x-coordinate |
| `methods/bspline/processor.py` | `BSplineChunkStep` (`bspline_chunk`) + `BSplineDecodeStep` (`bspline_decode`) + `EpisodeSplines`. Holds every episode's **fit**, not a label per frame (7.7 MB vs 36 MB on pickplace); a sample's matrix is its chunk with the knot column shifted to that frame. Fits once at construction — random sampling means a batch of 32 touches ~32 episodes, so lazy refitting at 1.4 s/fit would cost ~45 s per batch. `transform_features` rewrites the action feature to the matrix's channel count. The decode step is the inverse and is what makes a checkpoint executable: it evaluates the predicted curve at `num_actions` points -- the speed lever, `a_exec(t) = a(nt)`, a decode-time choice needing no retraining -- maps back through the layout, and publishes the realised `bspline_rate` per sample, which varies with the predicted span |
| `timed.py` | `TimedActions` contract: `dt` per action; `uniform`/`from_speeds`, `timestamps()` (exclusive cumsum, the UR10e's `(pose,t)` view), `duration()`; `DT_KEY` for pipelines |
| `train/run_train.py` | upstream `lerobot-train` + `--method.*`: wraps `make_train_eval_datasets` (capture dataset, halve chunk) and `make_pre_post_processors` (insert method steps pre-normalizer); calls `train.__wrapped__` (subclass fails upstream's identity check) |
| `eval/run_libero.py` | draccus eval runner, one task per output dir: method steps attached, actuator per method (PACE per-step / DemoSpeedup constant / none), env↔checkpoint ImageNet dedupe, IDENTITY-stats guard, sim-time recording, `run_config.yaml` |
| `eval/bspline_policy.py` | `attach_bspline`: the same instance-level rebinding as `attach_pace`, and for the same reason (upstream's evaluator asserts `isinstance(policy, PreTrainedPolicy)`, so a wrapper is refused). A B-spline checkpoint predicts curve parameters rather than actions, so `select_action` decodes one predicted chunk into `num_actions` executable actions, serves them one per call, and logs the realised rate per query |
| `eval/pace_policy.py` | `attach_pace`: instance-level select_action/reset rebinding (upstream rejects wrapper objects), speed queue, applied-speed log |
| `eval/sim_time.py` | `SimTimeRecorder` + vector-env re-arm (autoreset-safe per-episode sim durations) |
| `real/` (repo root, **not** under `src/`) | deploy env: pixi manifest + lock pinning lerobot @ the shared SHA, crisp fork SHAs, numpy<2 override; site network scripts incl. the cv2 libjpeg preload |

Outside this repo, load-bearing and NOT under pace_bench's git:
- crisp forks `ChazzKemal/crisp_{gym,py}` branch `robot-stack-pin`
  (`45dbb06` / `47b3d23`): byte-verified lab snapshots, pinned by `real/pixi.toml`.
  (Verified 2026-08-29: both branch heads still match those SHAs exactly.)

**The `lerobot_uncertainty` fork is no longer needed for the ACT pipeline.** Its
labelling stage — `utils/entropy.py`, `configs/label.py`, `scripts/lerobot_label.py`,
and the `sample_action_chunks` / `forward_with_latent` methods it added to LeRobot's
policy classes — is reimplemented here from the *original* DemoSpeedup instead
(`lingxiao-guo/DemoSpeedup` @ `34bd43a`, user decision 2026-08-29). That repo is
not depended on, fetched or expected on disk anywhere — it is not installable in
any case: no root package, `robobase` resolves through an SSH-only Gymnasium fork
whose manifest will not parse and pins numpy<2, and `aloha`'s `setup.py` needs
`pkg_resources` at build time without declaring it. Both pipeline scripts now run every stage in this
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
- **Diffusion labelling needs an `n_obs_steps=1` proxy** — resolved 2026-08-29 by
  passing `--policy.n_obs_steps=1` in the cups diffusion stage, which is both the
  upstream configuration and what makes that baseline its own DP oracle.
  `DiffusionChunkSampler` refuses anything else at construction: per-frame labelling
  has no observation history to give, and the runner reads one frame per query.
  `DiffusionConfig`'s default is **2**, so any new diffusion stage must override it.
  Upstream is single-frame in both instantiations: robobase (which has priority) sets
  `frame_stack: 1` in `cfgs/launch/dp_pixel_bigym.yaml` and has no per-policy obs-step
  notion — history there would be a FrameStack env wrapper folded into channels
  (`method/utils.py:86`), supplied for free because `Workspace.label()` replays each
  demo through a wrapped env rather than indexing frames; aloha says it outright
  (`n_obs_steps: 1  # Now only support for 1 obs_step`, with the time-axis slice
  commented out of `diffusion_unet_hybrid_image_policy.py:177`). Reading a 2-frame
  window in the runner remains possible but would be a LeRobot-native choice, not
  upstream parity. Caveat: the wider DP literature uses 2 frames, so this baseline is
  slightly weaker than a stock DP.
- **Diffusion is training for the first time.** Smoked 2026-08-29 on stack_cups: the
  baseline config trains (278M params), an `n_obs_steps=1` proxy labels episode 0
  end-to-end through `run_label` (437 frames, 0.13 s/frame — the first time the
  diffusion sampler ran on a real checkpoint rather than in unit tests), and
  DemoSpeedup+diffusion trains with `pad_mode=hold`. `pad_mode=zero` is rejected at
  construction: Diffusion's `do_mask_loss_for_padding=False` would train the zero
  tail as real targets, so the ACT scripts' `pad_mode=zero` does not transfer.
  `pickplace_diffusion_base` is the first diffusion run to go past a smoke test and
  is in flight now (see below); no diffusion policy has yet reached a checkpoint at
  its full step budget.
- **The controlled cross-oracle comparison is now running, not yet available.**
  ACT is trained on stack_cups and xVLA on LIBERO, so their precision fractions
  (79.6% / 84.1%, episode 0, `rule=mean`) differ by dataset as much as by policy.
  Comparing oracles properly needs two families trained on one dataset, which is
  exactly what `run_demospeedup_pickplace.sh` schedules: the ACT half is done
  (17.1% non-precision) and the diffusion half is training. Until its labelling
  stage runs, no two oracles have been compared on one dataset.
- **LIBERO's labels are the one stage still taken on trust from the fork.**
  `data/labels/xvla_libero10_ee6d/speedup_labels` holds all 400 episodes
  (`episode_<i>.npy` + `entropy_<i>.npy`), produced by the fork's stage-2 run;
  `run_demospeedup_libero10.sh:32` consumes them via `--method.labels_path` behind a
  `require` guard, and the completed LIBERO A/B rests on them. So nothing is blocked
  operationally — every other LIBERO stage runs in this repo, and this one has its
  input on disk. What is missing is the check that pace_bench's own `run_label`
  reproduces them: the real pipeline has been end-to-end verified on stack_cups and
  pickplace, never on LIBERO.
- **Regenerating those labels is expensive, because xVLA still encodes per *sample*.**
  Two wins were available and one has been taken. Taken (`b53798a`): `--batch_frames`
  batches several *frames* into one policy call, and diffusion additionally got a
  specialised path that encodes each frame once instead of once per sample — worth
  ~6× in memory, which is what makes a wide frame batch fit. That took pickplace
  diffusion labelling from 4.8 h at `--batch_frames=1` to **1.4 h** at 32, plateauing
  past ~32 because the denoiser is latency- not throughput-bound. Not taken: xVLA
  still lets Florence2 encode all N rows (`XVLAChunkSampler` inherits the generic
  `sample_frames` and adds nothing). The fix remains a ~5-line `forward_vlm` override
  on an `XVLAModel` subclass (encode row 0, expand to N), leaving upstream's
  flow-matching loop untouched. At the measured 0.60 s/frame, re-deriving the 400
  ee6d episodes (102033 frames) is ~17 h — an overnight job, not an interactive one,
  which is why it keeps not happening. **xVLA's gain from `--batch_frames` alone has
  not been measured** — that timing predates the flag — so how much the encoder
  override would add on top is unquantified.
- **No real-robot arm has a number, and the offline substitute was rejected**
  (user decision 2026-08-30). There is no UR10e simulator, so the real arms cannot be
  scored without the robot. An open-loop evaluator was built and then dropped
  (`eval/run_offline.py`, `tests/test_offline_eval.py` and the
  `outputs/eval/pickplace_act/` results, all deleted): it scored each policy's
  predicted chunk against the demonstrations, and every one of those demonstrations
  was in that policy's training set. With no held-out split it largely measures how
  well each arm memorised its own targets, so it cannot separate a policy that
  generalises from one that overfits — and "path deviation" reads as a
  trajectory-fidelity claim that the construction does not support. Neither does it
  say anything about the thing the benchmark is for: whether the task succeeds when
  the robot runs faster. **Do not rebuild it.** The real-arm question stays open until
  the UR10e runs both arms and success rates are counted; that is deploy-day work.
- **B-spline trains on every dataset and policy family in scope.** Verified by
  running: pickplace (cart7, real) and libero_10_ee6d (ee6d20, sim) × ACT and
  Diffusion, plus xVLA on LIBERO. Decode is implemented, so a checkpoint is
  executable. Three hooks were needed, each because the action *space* changes --
  something no other method here does:
  `adjust_dataset` (`make_policy` overwrites `cfg.output_features` from
  `ds_meta.features` and takes normalization from `ds_meta.stats`, so `adjust_policy`
  alone cannot change the action width); `adjust_processors` (a policy loaded from a
  checkpoint carries its own saved normalizer, which `adjust_dataset` cannot reach);
  and the arrangement/action-space pair for xVLA below. ACT's
  `actions.sum(-1) == 0` pad heuristic is moot -- a B-spline chunk is fixed-size by
  construction and the step masks nothing.
- **A B-spline xVLA arm is NOT comparable to the PACE and DemoSpeedup xVLA arms**
  (raised by the user 2026-08-30; decision pending). Every LIBERO arm starts from a
  checkpoint pretrained on *dense action chunks*, and the three methods ask different
  amounts of that checkpoint. PACE does not touch training targets at all. DemoSpeedup
  keeps the action space exactly -- slot k still means "ee6d pose at step k", only
  which demonstration frames fill the slots changes -- so its head starts aligned.
  B-spline reinterprets the tokens: the chunk axis stops indexing timesteps and starts
  indexing control points, and one channel becomes a time. **Weakened by the
  domain-0 finding below**: the action head is random in every arm, so no arm inherits
  an aligned decoder and the asymmetry is confined to the *trunk* — LoRA-adapted
  blocks and a frozen `pos_emb`, both pretrained on dense action chunks. Real, but
  much milder than "B-spline must repurpose a pretrained head" as first stated here. Two mitigations: the
  reinterpretation is not arbitrary (control point k is roughly where the arm is
  around the k-th knot, and channels 0-9 keep their exact meaning), and the paper
  never makes this claim -- it validates B-spline on Diffusion Policy and ACT, both
  from scratch, so a pretrained VLA is our extrapolation. Options, ranked:
  (1) make the **real-robot** comparison the headline, where ACT and Diffusion train
  from scratch and no pretraining asymmetry exists, and report LIBERO B-spline
  separately if at all; (2) equalise the handicap by reinitialising
  `action_encoder`/`action_decoder` for *every* LIBERO arm, so all three start from a
  pretrained trunk plus a fresh head -- a small change, since those modules are
  already in `full_training_modules`, and it costs the baseline absolute SR in
  exchange for comparability; (3) drop B-spline from the LIBERO axis entirely.
- **Every xVLA arm trains its action head from random init — including the baseline**
  (verified against the checkpoint 2026-08-30). `action_encoder`, `action_decoder` and
  `soft_prompt_hub` are `DomainAwareLinear` / `nn.Embedding` *tables* of 30 domains,
  indexed per sample by `domain_id`. `lerobot/xvla-libero` has 9 genuinely pretrained
  domains — rows 3 and 10-17, weight norms 11.6-25.8 with non-zero biases — and
  **row 0 is at xavier initialization**: its biases are exactly 0.0 (the initializer)
  and its weight norms sit at the median of the untouched rows. The checkpoint sets
  `domain_feature_key=None` and `libero_10_ee6d` carries no `domain_id`, so
  `_get_domain_id` returns 0 for every sample and every arm trains that untrained row.
  It works: the baseline reached 92.0% SR that way. Consequences: (a) B-spline is
  **not** disadvantaged at the head — no arm inherits an aligned action head, all three
  learn what the tokens mean from scratch, which weakens the comparability objection
  below to the trunk alone; (b) "give the B-spline arm its own domain" is moot, since
  domain 0 is already a free row; (c) 2.8M parameters of pretrained action head sit
  unused in every run.
  **Untested:** whether the head learns the knot channel. 300 steps at batch 2 (0.6%
  of an epoch, still in LR warmup) left `knot_loss / KNOT_SCALE` at ~1.34 MSE against
  a target variance of ~0.94 — no better than predicting the mean knot.
- **xVLA needed a second construction, and one number in it is a guess.** xVLA reads
  its action vector *structurally* -- `POS_IDX_1 = (0,1,2)`, `ROT_IDX_1 = (3..8)`,
  gripper at 9 and 19, with per-group scales -- so upstream's knot-first matrix trains
  a time as an x-coordinate (position loss 122840 against a rotation loss of 6.3).
  Two fixes, both ours since the paper uses only Diffusion and ACT: the
  `xvla_ee6d20` arrangement puts the control point in slots 0-9 and the knot in 10,
  and `ee6d_bspline` (registered in xVLA's own registry) scores the knot as a fourth
  term. It also does not *normalize*: `XVLAConfig.normalization_mapping` is IDENTITY
  throughout, so raw magnitudes reach the loss and knots in frames (~50) swamped
  positions in metres (~1.3). The arrangement therefore carries knots in **seconds**
  (`knot_scale = 1/fps`), inverted on decode. **`KNOT_SCALE = 10.0` is untuned** --
  there is no reference to inherit it from -- and is the first thing to sweep if a
  B-spline xVLA arm tracks poorly in time. The paper (§5) integrates with **both**
  Diffusion Policy ("Diff.+BSP") and ACT ("Reg.+BSP"), so batch 7 is a reproduction;
  only the released *code* is diffusion-only. xVLA (batch 8) is ours.
  **The gripper is not special-cased**, which is upstream's own choice
  (`_preprocess_chunks` fits `episode_actions[:, :n_dims]`, gripper included). It is
  safe because the fit tolerance is a max over every element, so a gripper edge is a
  hard constraint on the knot search: at `max_error=0.01` with a gripper in [0, 1]
  the transition is tracked to 1% of stroke. Gripper edges therefore *drive* knot
  density, which is what makes knot density a precision signal in its own right --
  the quantity DemoSpeedup needs a policy and an entropy estimate to obtain.
- **B-spline needs no labelling stage** (user decision 2026-08-30). Its labels are
  the fitted spline parameters, not anything the dataset carries — but unlike
  DemoSpeedup's, they need no policy to produce, only geometry, so they are cheap
  enough to build in the preprocessor at training time. Cheap because nothing is
  *optimised*: the paper's Algorithm 1 places knots by the classical FITPACK greedy
  criterion (insert where the residual is worst) and, with the knots fixed, the
  control points are a plain linear least-squares solve. Measured, that is 17-21
  candidate knot vectors per episode at 0.23 ms (libero, 293 frames) to 1.05 ms
  (pickplace, 1505 frames) per solve. scipy's `generate_knots` inserts several knots
  per step rather than the paper's one — 8, 9, 11, 12, 14, 18, ... 73 — reaching the
  same tolerance in ~20 candidates instead of ~70. Most of the wall clock is not even
  the solve (4% of it on pickplace) but `generate_knots`' own residual analysis. Measured full passes:
  **libero_10_ee6d 3.1 s** (400 eps, 102033 frames, 0.03 ms/frame) and
  **pickplace 50.3 s** (45 eps, 31178 frames, 1.61 ms/frame), zero episodes missing
  `max_error=0.01`. Against 1.9 h and 5.9 h of training that is free. So: no
  `run_label` stage, no labels on disk (117 MB and 36 MB not written), and no
  possibility of the labels-do-not-match-this-dataset class of bug that the
  DemoSpeedup path has to guard against — the fit parameters live in the method
  config and cannot drift from the run. Upstream caches to npz
  (`make_bspline_sampler_cache_path`) because it hashes the settings; at these
  timings a cache would cost more than it saves. Fit once at startup, the way
  `config.py` already preloads episode action tables, and keep the per-episode
  chunks plus a frame→chunk index rather than a matrix per frame (~12 MB for
  pickplace instead of 36 MB, and it is what upstream's `all_actions` /
  `timestep_to_chunk` pair does).
- **Knots are absolute offsets from the current frame** (user decision 2026-08-30),
  matching every shipped upstream config and the recorded dataset. Two things get
  called "relative" here and only one is a choice: the control points are absolute
  poses always, and the knot column is relative to the sample's own frame always
  (upstream shifts it per sample; the recorded metadata says `"knot_units": "source
  frames, relative to the current frame"`). What `relative_knots` selects is whether
  that time column holds those offsets (-7, 0, 4, ... 51) or their consecutive
  differences. Offsets normalise to about [-1.6, +1.4], which is fine; the cost is
  that one per-column statistic spans the row-to-row ramp and attenuates the
  per-sample knot signal ~2.3x against the control points. The flag remains, off.
- **A fixed `num_actions` gives a *variable* speed-up.** The lever is decode-time
  and needs no retraining, but a chunk's span is whatever the fit chose -- on
  pickplace episode 0, 18 to 64 source frames for the same 10-span chunk -- so one
  `num_actions` yields 1.2x at one frame and 4.3x at another. The realised factor is
  published per sample as `bspline_rate`; any reported speed-up has to come from that
  and not from the config. Decoding at one sample per source frame reproduces the
  demonstration to 0.31-0.77 mm, so the error budget is the fit's, not the decode's.
- **Diffusion constrains the matrix width, and LeRobot's own check cannot catch it.**
  The temporal U-Net halves the horizon once per `down_dims` stage, so the width must
  be a multiple of `2**len(down_dims)` = 8. `DiffusionConfig.__post_init__` validates
  exactly this, but it has already run by the time a method mutates the config — so
  the first attempt died mid-forward with "Sizes of tensors must match except in
  dimension 1", naming neither the horizon nor the method. `adjust_policy` now checks
  it and lists the usable `chunk_size` values. The default is `chunk_size=10` (width
  16), which is upstream's own real-robot horizon and suits every family; the recorded
  dataset's 20 (width 26) is fine for ACT and xVLA but Diffusion rejects it.
- **Superseded: normalisation via `relative_knots=True`** (2026-08-30).
  It looked as though the B-spline step would have to own normalisation, because
  LeRobot computes one statistic per action *dimension* while upstream computes one
  per *element* of the matrix — and the knot column badly needs the latter, its value
  being mostly decided by which row it sits in (measured on the recorded dataset:
  mean -7.66 at row 0, +50.91 at row 25; row-to-row spread 1.98x the within-row
  spread, against 0.02x for a control-point column). One per-column statistic
  therefore leaves a ~2.9-normalised-unit deterministic ramp in the target with only
  0.44 of a unit of real signal on top. But upstream already has the fix and it is a
  flag, not an architecture: `relative_knots=True` stores the column as consecutive
  differences, which drops the ratio to 0.23 and makes the stock per-dim normaliser
  correct. LeRobot's normaliser stays in charge. **Overtaken by the decision above**:
  the ramp is real but its cost is a ~2.3x attenuation, not a defect, and upstream
  ships absolute. Kept here because the measurement is the reason the flag exists.
- **The interpolable action layout is per dataset, not per method.** `to_spline_actions`
  converts cart7 (xyz + angle-axis + gripper) to xyz + rot6d + gripper because
  angle-axis cannot be interpolated across the pi wrap. But `libero_10_ee6d` is
  *already* xyz + rot6d + gripper in its first 10 dims, followed by 10 zero-pad dims
  that must be dropped before fitting and restored after decoding; a joint-space
  dataset would need no conversion at all. So the layout is a dataset-level config
  choice that every B-spline arm has to name, and getting it wrong is silent — a
  transposed rotation still fits, still decodes, and is still wrong.
- **Fake-mode dataset diff** — deploy-day gate on the lab machine, where the
  baseline env exists (user decision 2026-08-28).
- **Real-inference gripper postprocessors** — PACE: speed=1 during gripper motion;
  DemoSpeedup: repeat gripper-moving actions ×low_v, truncate. Config-gated,
  never serialized into checkpoints. See memory note; not implemented. Upstream
  B-spline reaches the same fix independently and is worth copying rather than
  reinventing (`scripts/policy_local_bspline.py`): on
  `|gripper - last_gripper|.max() > 0.08` it sets a 7-step counter and ramps the
  speed-up *linearly* back from 1.0 to full — `1 + (steps - remaining)/steps *
  (speed_up_times - 1)` — scaling the time advance rather than the actions. A ramp,
  not a step, and off by default (`gripper_slowdown_enabled=False`). It also drops
  the gripper from the spline time-alignment distance unless
  `consider_gripper_during_align` is set, because a near-binary channel is
  misleading for "where am I along this path".
- **Batch 2's 3-seed PACE gate** — parked; upstream xVLA has a known ~10pp task-1
  deficit vs the fork that would confound absolute-SR comparison.

## Experiments state (2026-08-30 ~12:40; diffusion baseline RUNNING)

- **LIBERO A/B (xVLA)**: complete. Baseline 92.0% SR / 13.28 s vs DemoSpeedup
  86.5% / 6.99 s = 1.90× at −5.5 pp; task 2 (−30 pp) is the known speed-intolerant
  task. `outputs/eval/ds_libero10_*`.
- **pickplace ACT: both arms trained, neither evaluated.** 100k × 32 in bf16
  (`pickplace_act_base` finished 05:27, `pickplace_act_speedup` 11:44), labels from
  the ACT baseline's own checkpoint (45 episodes, 31178 frames, **17.1%
  non-precision**, mean fast-run 11.97 frames vs 1.21 if random — real signal, same
  shape as cups). The retiming reaches ~2.16 raw frames per executed step, which is
  what the labels imply and is a property of the labels, not a result. There is no
  score for either arm: the offline evaluator that produced one was rejected as
  unprincipled (see gaps), and the honest statement is that these checkpoints are
  waiting on the robot.
- **pickplace diffusion: baseline in flight.** `pickplace_diffusion_base`
  (`--policy.n_obs_steps=1`, bf16, 100k × 32) started ~11:45, at ~19k/100k as of
  12:38, 6.5 step/s, ETA ~16:10. The script then labels from it (`pickplace_dp`) and
  trains `pickplace_diffusion_speedup`, completing the 2×2 and with it the
  cross-oracle comparison. Nothing evaluates those arms yet — see gaps.
- **stack_cups**: ACT baseline ✅ (`outputs/train/cups_act_base`, 30k), labels ✅
  (12/12, 18.4% non-precision, run-length 13.9 vs 1.23 random), Diffusion baseline
  ❌ not trained. **DemoSpeedup ACT ❌ still not trained**: the 2026-08-29 23:36
  attempt died with a CUDA OOM (`logs/cups_act_speedup.log`) — 23.5 GB card, another
  process holding 14.2 GB, so it is a scheduling collision, not a config fault.
  `outputs/train/cups_act_speedup/` exists with no checkpoint, so the script's
  skip-guard will correctly retry it. Resume: `./run_demospeedup_stackcups.sh`, but
  not while pickplace has the card.

Runtimes for every stage above are tabulated in the README (`## Runtimes`).

## Conventions that bit us (do not relearn)

- wandb naming (user decision 2026-08-29): project `pace_benchmark_<task>`
  (`stack_cups` / `pickplace` / `libero10`), run `<policy>_<addon>` —
  `act_demospeedup`, `xvla_baseline`, `diffusion_baseline`. Run names are kept
  independent of the output dirs, which keep their old names: `--job_name` is
  what wandb shows, and coupling the two means a rename orphans a checkpoint.
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
- An equivalence test between two batch widths cannot assert bit-equality. The
  diffusion encoder-dedup test asserted `atol=1e-6` and failed 3 of 8 full-suite runs
  on arithmetic noise alone (measured gap 1e-6 to 3e-5): float32 conv/GEMM results
  depend on batch width, and a denoising loop amplifies that. Size such a tolerance
  from the defect being caught, not from zero — a swapped frame/sample axis moves
  whole actions, so 1e-4 catches it and noise does not trip it. Verified by mutation:
  tiling instead of interleaving still fails at the looser tolerance.
- A B-spline chunk's valid domain starts at `knots[degree]`, not at the sample's own
  frame: everything before that needs knots the window does not carry. Comparing a
  decode against `raw[frame + 0 ...]` instead of `raw[frame + knots[degree] ...]`
  looks like a 10 cm error and is a bug in the test, not the code.
- A smooth synthetic trajectory is not a demonstration, and some properties can
  only be tested on real data. The relative-knot claim inverts on synthetic
  sinusoids — they fit with near-uniform knots, whose differences have almost no
  within-row spread, so the ratio the property is stated in terms of becomes
  meaningless. That test lives with the recorded dataset for this reason.
- The B-spline fit is the slowest thing in the suite: ~1.4 s per episode, because
  `generate_knots` fits a spline per candidate knot vector. Fit once and reuse
  (`fit_episode` + `chunk_parameters` + `assign_chunks_to_frames`) rather than
  calling `episode_parameter_chunks` twice — that alone halved the recorded-dataset
  gate from 195 s to 97 s.
- A heredoc binds to the *last* command of a pipeline. `"$PY" - ... | tee log <<'EOF'`
  feeds the script to `tee`, which echoes it, while python reads the real stdin
  (`/dev/null` under nohup), sees EOF and runs nothing — silently, if the check
  gates no exit. Put the `<<'EOF'` on the command that must read it. Cost us the
  pickplace label-signal check, whose log holds its own source instead of an answer.
- Background helpers must run under `$PY`, not `python3`: this box's `python3` is
  3.8 and the checkpoint pruner annotates with PEP 585 generics. Backgrounded, its
  `ImportError` surfaces nowhere and the first symptom is a full disk hours later.
- Batch shape cannot be inferred by counting dimensions: a batched camera image and
  an unbatched action chunk are both 3-D. Reshape against the policy config's
  declared feature shapes, the same way chunk fields come from
  `POLICY_CHUNK_FIELDS`.
