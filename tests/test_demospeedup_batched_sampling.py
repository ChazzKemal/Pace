"""Batched labelling must land each frame's chunks in that frame's slot.

Speed is why `sample_frames` exists, but the property worth pinning is the boring
one: widening the batch must not shuffle which observation a chunk answers. A
tiling bug instead of an interleaving one returns exactly the right chunks in the
wrong rows -- every downstream stage still runs, the entropy trace is simply
measured against the wrong frames, and the labels are quietly wrong.
"""

import torch

from pace_bench.methods.demospeedup.run_label import _sample_stream
from pace_bench.methods.demospeedup.sampler import (
    _BroadcastChunkSampler,
    collate,
    repeat_frames,
)

CHUNK, DIM, SAMPLES = 4, 3, 5


class EchoPolicy:
    """Returns each row's own state value, so a chunk names the frame it came from."""

    class _Cfg:
        n_obs_steps = 1

    config = _Cfg()

    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        self.calls += 1
        marker = batch["observation.state"][:, :1]  # (rows, 1)
        return marker.reshape(-1, 1, 1).expand(-1, CHUNK, DIM).clone()


def frame_batch(value: float) -> dict[str, torch.Tensor]:
    """One preprocessed observation, batched to a single row and tagged with `value`."""
    return {
        "observation.state": torch.full((1, DIM), value),
        "index": torch.tensor(7),  # a scalar the batch carries; must survive untouched
    }


def test_collate_stacks_rows_and_leaves_scalars_alone():
    batch = collate([frame_batch(0.0), frame_batch(1.0), frame_batch(2.0)])
    assert batch["observation.state"].shape == (3, DIM)
    assert torch.equal(batch["observation.state"][:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert batch["index"].ndim == 0


def test_repeat_frames_interleaves_rather_than_tiles():
    rows = torch.tensor([[0.0], [1.0], [2.0]])
    out = repeat_frames({"x": rows}, 2)["x"]
    # interleaved: each frame's copies adjacent. Tiling would give 0,1,2,0,1,2.
    assert torch.equal(out[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]))


def test_sample_frames_puts_each_frame_in_its_own_slot():
    sampler = _BroadcastChunkSampler(EchoPolicy(), num_samples=SAMPLES)
    values = [0.0, 1.0, 2.0, 3.0]
    out = sampler.sample_frames([frame_batch(v) for v in values])

    assert out.shape == (len(values), SAMPLES, CHUNK, DIM)
    for i, value in enumerate(values):
        assert torch.all(out[i] == value), f"frame {i} carries another frame's chunks"
    assert sampler.policy.calls == 1, "the point is one policy call, not one per frame"


def test_stream_agrees_with_the_one_frame_path():
    """Batched and unbatched streams must yield the same frames in the same order."""

    class Dataset:
        def __getitem__(self, i):
            return frame_batch(float(i))

    sampler = _BroadcastChunkSampler(EchoPolicy(), num_samples=SAMPLES)
    identity = lambda item: item  # noqa: E731 -- the real preprocessor is applied per frame

    length = 7  # deliberately not a multiple of batch_frames: the tail block is short
    single = list(_sample_stream(Dataset(), sampler, identity, 10, length, batch_frames=1))
    batched = list(_sample_stream(Dataset(), sampler, identity, 10, length, batch_frames=3))

    assert len(single) == len(batched) == length
    for t, (a, b) in enumerate(zip(single, batched)):
        assert torch.equal(a, b), f"frame {t} differs between the batched and single paths"
        assert torch.all(a == 10 + t), f"frame {t} is not the observation at start+t"


def test_stream_falls_back_when_a_sampler_cannot_batch():
    """ACT's sampler has no sample_frames; it must keep taking the per-frame path."""

    class PerFrameOnly:
        def __call__(self, batch):
            return batch["observation.state"][:1].reshape(1, 1, DIM).expand(SAMPLES, CHUNK, DIM)

    class Dataset:
        def __getitem__(self, i):
            return frame_batch(float(i))

    out = list(_sample_stream(Dataset(), PerFrameOnly(), lambda x: x, 0, 5, batch_frames=32))
    assert len(out) == 5
    for t, chunk in enumerate(out):
        assert torch.all(chunk == t)


# --- the diffusion dedup path, against upstream's own generate_actions ---------

import pytest
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from pace_bench.methods.demospeedup.sampler import DiffusionChunkSampler

HORIZON, D_ACT, D_STATE = 16, 4, 5


@pytest.fixture(scope="module")
def dp_policy():
    """A small n_obs_steps=1 diffusion policy: the geometry, none of the cost."""
    cfg = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(D_STATE,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(D_ACT,))},
        normalization_mapping=dict.fromkeys(
            ("STATE", "ACTION", "VISUAL"), NormalizationMode.IDENTITY
        ),
        n_obs_steps=1,
        horizon=HORIZON,
        n_action_steps=8,
        down_dims=(32, 64, 128),
        num_train_timesteps=4,      # the denoiser is exercised, not benchmarked
        num_inference_steps=4,
        crop_shape=None,
        device="cpu",
    )
    return DiffusionPolicy(cfg).eval()


def dp_frame(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.state": torch.rand(1, D_STATE, generator=g),
        "observation.images.cam": torch.rand(1, 3, 96, 96, generator=g),
    }


def test_dedup_matches_generate_actions_exactly(dp_policy):
    """Encoding once per frame must not change the answer, only the cost.

    Both paths are seeded identically rather than handed a fixed initial noise:
    DDPM draws *fresh* variance noise inside `noise_scheduler.step` at every one of
    the denoising iterations, so pinning only the prior leaves the two runs on
    different points of the RNG stream. Seeded alike they draw the same
    `randn(frames * samples, horizon, dim)` prior and the same per-step noise, so
    any remaining difference is the dedup's own doing -- a wrong slice, a tiled
    instead of interleaved repeat, a dropped observation step.
    """
    samples, frames = 3, 2
    batches = [dp_frame(i) for i in range(frames)]
    sampler = DiffusionChunkSampler(dp_policy, num_samples=samples)

    from lerobot.utils.constants import OBS_IMAGES

    from pace_bench.methods.demospeedup.sampler import collate, repeat_frames

    # upstream's path: the observation repeated per sample, stacked as predict_action_chunk does
    wide = dict(repeat_frames(sampler.prepare(collate([dict(b) for b in batches])), samples))
    for key in dp_policy.config.image_features:
        if wide[key].ndim == 4:
            wide[key] = wide[key].unsqueeze(1)
    wide[OBS_IMAGES] = torch.stack([wide[k] for k in dp_policy.config.image_features], dim=-4)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = dp_policy.diffusion.generate_actions(wide)

    torch.manual_seed(0)
    out = sampler.sample_frames([dict(b) for b in batches])

    assert out.shape == (frames, samples, dp_policy.config.n_action_steps, D_ACT)
    assert torch.allclose(out.reshape(frames * samples, -1, D_ACT), reference, atol=1e-6), (
        "encoder-dedup diverged from generate_actions on an identical RNG stream"
    )
