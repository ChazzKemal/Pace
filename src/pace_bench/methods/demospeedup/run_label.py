"""Label a dataset's frames precision / non-precision, using a trained proxy policy.
This is DemoSpeedup's second stage and the input to its third. Stage 1 trains an
ordinary policy on the demonstrations; this stage asks that policy, at every frame,
for several action chunks and measures how much they disagree
(:mod:`.entropy`); stage 3 retimes the
demonstrations, taking long strides where the disagreement was low
(:mod:`.retime`).

    python -m pace_bench.methods.demospeedup.run_label \\
        --policy_path=outputs/train/cups_act_base/checkpoints/last/pretrained_model \\
        --dataset_repo_id=local/stack_cups --dataset_root=/path/to/stack_cups \\
        --out=outputs/label/cups

Writes ``<out>/speedup_labels/episode_<i>.npy`` -- exactly what
``--method.labels_path`` reads at training time -- plus the raw entropy trace per
episode. The trace is kept because segmentation is lossy and its knobs are a choice:
re-segmenting a labelling run is cheap, re-running the policy over the dataset is not.
"""


import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

from pace_bench.methods.demospeedup.entropy import kde_entropy
from pace_bench.methods.demospeedup.sampler import (
    ACTChunkSampler,
    DiffusionChunkSampler,
    XVLAChunkSampler,
)
from pace_bench.methods.demospeedup.segment import Rule, segment

logger = logging.getLogger(__name__)

#: Which policy families can be sampled from. Adding one means implementing a
#: :class:`~pace_bench.methods.demospeedup.sampler.ChunkSampler` for it, not
#: probing the policy object for whatever randomness it happens to expose.
SAMPLERS = {
    "act": ACTChunkSampler,
    "xvla": XVLAChunkSampler,
    "diffusion": DiffusionChunkSampler,
}


@dataclass
class LabelConfig:
    """Everything one labelling run needs."""

    policy_path: str = ""
    dataset_repo_id: str = ""
    dataset_root: str | None = None
    out: Path = Path("outputs/label")

    # Episodes to label. None means all of them.
    episodes: list[int] | None = None
    device: str | None = None
    seed: int = 1000

    # --- measurement ---
    # Chunks drawn per frame. Upstream's default; the cost of the run is linear in it.
    num_action_samples: int = 10
    kde_bandwidth: float = 1.0
    # Pool every chunk that covers a frame, not just the one that starts on it. A
    # frame predicted the same way from ten different observations is genuinely
    # unambiguous, which is the property the labels are meant to capture.
    temporal_aggregation: bool = True
    # Frames sampled per policy call. 1 reproduces the original one-frame-at-a-time
    # loop exactly. Higher only helps a family whose sampler implements
    # `sample_frames` (diffusion today): a diffusion chunk costs 100 sequential
    # denoising steps whatever the batch width, so the card sits idle at width 10.
    # Measured on pickplace, 31k frames: 4.76h at 1, 1.35h at 32. Above ~32 it
    # plateaus -- the denoiser is latency-bound, not throughput-bound.
    batch_frames: int = 32

    # --- segmentation --- see segment.Rule; "upstream" is the reference behaviour.
    rule: Rule = "upstream"
    min_cluster_size: int = 5
    max_cluster_size: int = 25
    outlier_contamination: float = 0.1

    rename_map: dict[str, str] = field(default_factory=dict)


def _sample_stream(dataset, sampler, preprocessor, start, length, batch_frames):
    """Yield each frame's ``(num_samples, chunk, dim)`` stack, in frame order.

    Sampling is batched where the family supports it; the frames still come out one
    at a time, so the aggregation below is unchanged by how they were produced. A
    sampler without ``sample_frames`` (ACT, which drives the model directly and is
    fast enough not to need this) takes the original path.
    """
    batched = getattr(sampler, "sample_frames", None)
    if batched is None or batch_frames <= 1:
        for t in range(length):
            yield sampler(preprocessor(dataset[start + t]))
        return
    for begin in range(0, length, batch_frames):
        block = [
            preprocessor(dataset[start + t])
            for t in range(begin, min(begin + batch_frames, length))
        ]
        stacked = batched(block)
        for i in range(len(block)):
            yield stacked[i]


def episode_entropy(
    dataset: LeRobotDataset,
    sampler,
    preprocessor,
    start: int,
    length: int,
    *,
    bandwidth: float,
    temporal_aggregation: bool,
    batch_frames: int = 1,
) -> np.ndarray:
    """The entropy trace of one episode, one value per frame.

    With ``temporal_aggregation`` the samples pooled at frame ``t`` come from every
    chunk still covering it -- the one that started at ``t`` and the chunk-length
    minus one before it, each contributing the row that predicts ``t``. Without it,
    only the chunk starting at ``t`` is used, which is noisier.

    The window comes from the chunks the sampler actually returns, not from the
    policy config: ACT hands back ``chunk_size`` steps but a diffusion policy hands
    back ``n_action_steps``, and the pool has to match what arrived.
    """
    trace = np.zeros(length, dtype=np.float64)
    recent: deque[torch.Tensor] | None = None

    stream = _sample_stream(dataset, sampler, preprocessor, start, length, batch_frames)
    for t, samples in enumerate(stream):  # samples: (num_samples, chunk_length, action_dim)
        if not temporal_aggregation:
            pooled = samples[:, 0, :]
        else:
            if recent is None:
                recent = deque(maxlen=samples.shape[1])
            recent.append(samples)
            first_start = t - len(recent) + 1
            covering = [
                chunk[:, t - (first_start + offset), :]
                for offset, chunk in enumerate(recent)
                if 0 <= t - (first_start + offset) < chunk.shape[1]
            ]
            pooled = torch.cat(covering, dim=0)

        trace[t] = kde_entropy(pooled.unsqueeze(0).float(), bandwidth).item()
    return trace


@draccus.wrap()
def main(cfg: LabelConfig) -> None:
    # force=True: importing lerobot installs a root handler, and without this
    # basicConfig would quietly do nothing and the run would print no progress.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    set_seed(cfg.seed)

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    if cfg.device:
        policy_cfg.device = cfg.device
    device = get_safe_torch_device(policy_cfg.device, log=True)

    sampler_cls = SAMPLERS.get(policy_cfg.type)
    if sampler_cls is None:
        raise ValueError(
            f"no chunk sampler for policy type {policy_cfg.type!r}; "
            f"labelling supports {sorted(SAMPLERS)}. Add one in methods/demospeedup/sampler.py."
        )

    dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, episodes=cfg.episodes)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    policy.to(device)

    # The checkpoint's own preprocessor, not a hand-rolled one: it carries the
    # normalization the policy was trained under, and measuring entropy through a
    # different one would measure a mis-conditioned policy.
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    logger.info("preprocessor: %s", [type(s).__name__ for s in preprocessor.steps])

    sampler = sampler_cls(policy, num_samples=cfg.num_action_samples)

    labels_dir = cfg.out / "speedup_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out / "run_config.yaml", "w") as f:
        draccus.dump(cfg, f)

    logger.info(
        "labelling %d episodes | %d samples/frame | %s oracle | rule=%s",
        dataset.num_episodes, cfg.num_action_samples, policy_cfg.type, cfg.rule,
    )

    for episode in range(dataset.num_episodes):
        # Re-seeded per episode so one episode can be re-labelled on its own and
        # land on the same answer it did inside a full run.
        set_seed(cfg.seed + episode)
        meta = dataset.meta.episodes[episode]
        start = meta["dataset_from_index"]
        length = meta["dataset_to_index"] - start

        trace = episode_entropy(
            dataset, sampler, preprocessor, start, length,
            bandwidth=cfg.kde_bandwidth,
            temporal_aggregation=cfg.temporal_aggregation,
            batch_frames=cfg.batch_frames,
        )
        np.save(labels_dir / f"entropy_{episode}.npy", trace)

        labels = segment(
            trace,
            rule=cfg.rule,
            min_cluster_size=cfg.min_cluster_size,
            max_cluster_size=cfg.max_cluster_size,
            contamination=cfg.outlier_contamination,
            seed=cfg.seed,
        )
        np.save(labels_dir / f"episode_{episode}.npy", labels)
        logger.info(
            "episode %d: %d frames, %.1f%% precision",
            episode, length, 100.0 * (labels == 0).mean(),
        )

    logger.info("labels written to %s", labels_dir)


if __name__ == "__main__":
    main()
