"""Sampling action chunks: diversity comes from the VAE latent, and nothing else moves."""

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy

from robot_stack.methods.demospeedup.run_label import episode_entropy
from robot_stack.methods.demospeedup.sampler import (
    ACTChunkSampler,
    DiffusionChunkSampler,
    LatentSamplingACT,
    XVLAChunkSampler,
    broadcast,
)

CHUNK = 20
ACTION_DIM = 7


def make_config(**overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "normalization_mapping": dict.fromkeys(("STATE", "ACTION", "VISUAL"), NormalizationMode.IDENTITY),
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "dim_model": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_heads": 2,
        "dim_feedforward": 128,
        "vision_backbone": "resnet18",
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


@pytest.fixture(scope="module")
def policy() -> ACTPolicy:
    torch.manual_seed(0)
    p = ACTPolicy(make_config())
    p.eval()
    return p


@pytest.fixture
def observation() -> dict:
    return {
        "observation.state": torch.zeros(1, ACTION_DIM),
        "observation.images.cam": torch.zeros(1, 3, 96, 96),
        # LeRobotDataset puts the task string in the batch; ACT never reads it.
        "task": ["put the cup on the stack"],
    }


def test_returns_one_chunk_per_sample(policy, observation):
    out = ACTChunkSampler(policy, num_samples=8)(observation)
    assert out.shape == (8, CHUNK, ACTION_DIM)


def test_samples_are_diverse(policy, observation):
    """The point of the whole module: identical samples would make entropy constant."""
    out = ACTChunkSampler(policy, num_samples=8)(observation)
    for i in range(1, 8):
        assert not torch.allclose(out[0], out[i]), f"sample {i} is identical to sample 0"
    assert out.std(dim=0).mean().item() > 1e-3


def test_the_policy_is_left_alone(policy, observation):
    """The sampler copies weights; it must not make the caller's policy stochastic."""
    before = policy.predict_action_chunk(dict(observation))
    ACTChunkSampler(policy, num_samples=4)(observation)
    after = policy.predict_action_chunk(dict(observation))
    assert torch.allclose(before, after)
    # And the deterministic path is still deterministic.
    assert torch.allclose(after, policy.predict_action_chunk(dict(observation)))


def test_trained_weights_load_without_remapping(policy):
    """A forward-pre-hook adds no parameters, so state_dict keys must be unchanged."""
    model = LatentSamplingACT(policy.config)
    missing, unexpected = model.load_state_dict(policy.model.state_dict(), strict=True)
    assert not missing and not unexpected
    assert set(model.state_dict()) == set(ACT(policy.config).state_dict())


def test_latent_is_the_only_source_of_diversity(policy, observation):
    """Seeding torch makes the sampler reproducible -- nothing else is stochastic."""
    sampler = ACTChunkSampler(policy, num_samples=6)
    torch.manual_seed(123)
    first = sampler(observation)
    torch.manual_seed(123)
    assert torch.allclose(first, sampler(observation))


def test_non_vae_policy_is_refused():
    """Without a VAE every sample would be identical and the entropy a constant 0."""
    with pytest.raises(ValueError, match="use_vae=False"):
        LatentSamplingACT(make_config(use_vae=False))


def test_too_few_samples_is_refused(policy):
    with pytest.raises(ValueError, match="at least 2 samples"):
        ACTChunkSampler(policy, num_samples=1)


# --- the driver's glue ------------------------------------------------------


def test_accepts_the_observation_batched_or_not(policy):
    """A checkpoint's saved preprocessor may or may not add a batch dimension."""
    sampler = ACTChunkSampler(policy, num_samples=4)
    unbatched = {
        "observation.state": torch.zeros(ACTION_DIM),
        "observation.images.cam": torch.zeros(3, 96, 96),
    }
    batched = {k: v.unsqueeze(0) for k, v in unbatched.items()}
    torch.manual_seed(0)
    a = sampler(unbatched)
    torch.manual_seed(0)
    b = sampler(batched)
    assert a.shape == (4, CHUNK, ACTION_DIM)
    assert torch.allclose(a, b)


def test_extra_batch_keys_are_ignored(policy, observation):
    """A dataset frame carries the recorded action too -- same ndim as an image.

    Forwarding by declared feature rather than by counting dimensions is what keeps
    that from being tiled into the model as if it were an observation.
    """
    noisy = dict(observation)
    noisy["action"] = torch.zeros(1, CHUNK, ACTION_DIM)
    noisy["action_is_pad"] = torch.zeros(1, CHUNK, dtype=torch.bool)
    noisy["index"] = torch.tensor([3])
    sampler = ACTChunkSampler(policy, num_samples=4)
    torch.manual_seed(0)
    with_extras = sampler(noisy)
    torch.manual_seed(0)
    assert torch.allclose(with_extras, sampler(observation))


def test_wrong_feature_shape_is_refused(policy):
    sampler = ACTChunkSampler(policy, num_samples=4)
    with pytest.raises(ValueError, match="expected a feature shaped"):
        sampler({
            "observation.state": torch.zeros(3, ACTION_DIM),
            "observation.images.cam": torch.zeros(1, 3, 96, 96),
        })


class FakeDataset:
    """One frame per index; the sampler below ignores the content."""

    def __getitem__(self, index):
        return {"observation.state": torch.zeros(1, ACTION_DIM)}


def constant_sampler(_batch):
    """Every chunk identical, so entropy is 0 wherever the pooling is well-formed."""
    return torch.zeros(4, CHUNK, ACTION_DIM)


def test_temporal_aggregation_pools_every_covering_chunk():
    """At frame t the pool is (chunks covering t) x (samples per chunk).

    Frame 0 is covered by 1 chunk, frame 5 by 6, and from `chunk_size` on by
    `chunk_size` -- the rolling buffer's length. Recorded through a sampler that
    returns a known number of rows.
    """
    seen = []

    def recording_sampler(batch):
        return torch.randn(4, CHUNK, ACTION_DIM)

    def spy(x, bandwidth=1.0):
        seen.append(x.shape[1])
        return torch.zeros(x.shape[0])

    from robot_stack.methods.demospeedup import run_label

    original = run_label.kde_entropy
    run_label.kde_entropy = spy
    try:
        episode_entropy(
            FakeDataset(), recording_sampler, lambda b: b, 0, 30,
            bandwidth=1.0, temporal_aggregation=True,
        )
    finally:
        run_label.kde_entropy = original

    assert seen[0] == 4, "frame 0 is covered by one chunk of 4 samples"
    assert seen[5] == 6 * 4, "frame 5 is covered by six chunks"
    assert seen[25] == CHUNK * 4, "past chunk_size the buffer is full and stays full"


def test_without_temporal_aggregation_only_the_starting_chunk_is_used():
    trace = episode_entropy(
        FakeDataset(), constant_sampler, lambda b: b, 0, 10,
        bandwidth=1.0, temporal_aggregation=False,
    )
    assert trace.shape == (10,)
    # Identical samples everywhere -> zero entropy everywhere.
    assert np.allclose(trace, 0.0, atol=1e-6)


# --- policies that are already stochastic: xVLA and Diffusion ---------------


def make_diffusion(**overrides):
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    kwargs = {
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "normalization_mapping": dict.fromkeys(("STATE", "ACTION", "VISUAL"), NormalizationMode.IDENTITY),
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 1,
        "crop_shape": (84, 84),
        "num_inference_steps": 4,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


@pytest.fixture(scope="module")
def diffusion_policy():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    torch.manual_seed(0)
    p = DiffusionPolicy(make_diffusion())
    p.eval()
    return p


def test_diffusion_samples_are_diverse(diffusion_policy, observation):
    """Diffusion keeps its randomness at inference, so nothing has to be injected."""
    out = DiffusionChunkSampler(diffusion_policy, num_samples=6)(observation)
    assert out.shape == (6, 8, ACTION_DIM)  # n_action_steps, not horizon
    for i in range(1, 6):
        assert not torch.allclose(out[0], out[i])
    assert out.std(dim=0).mean().item() > 1e-3


def test_diffusion_needs_a_single_observation_step():
    """Per-frame labelling has no observation history to condition on."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy(make_diffusion(n_obs_steps=2))
    with pytest.raises(ValueError, match="no history to give it"):
        DiffusionChunkSampler(policy, num_samples=4)


class StochasticStandIn:
    """A policy shaped like xVLA's: fresh noise per batch row, queues cleared by reset.

    xVLA itself is not constructible in a unit test -- it pulls a Florence2
    checkpoint -- so the shared machinery is exercised here and the family-specific
    part (flow matching from `x1 = randn(batch_size, ...)`) is upstream's own code.
    """

    def __init__(self):
        self.config = type("cfg", (), {"n_obs_steps": 1})()
        self.resets = 0
        self.seen = None

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        self.seen = batch
        batch_size = batch["observation.state"].shape[0]
        return torch.randn(batch_size, CHUNK, ACTION_DIM)


def test_broadcast_sampler_repeats_the_observation(observation):
    policy = StochasticStandIn()
    out = XVLAChunkSampler(policy, num_samples=5)(observation)
    assert out.shape == (5, CHUNK, ACTION_DIM)
    assert policy.seen["observation.state"].shape == (5, ACTION_DIM)
    assert policy.seen["observation.images.cam"].shape == (5, 3, 96, 96)
    # Every row is the same observation; only the policy's noise differs.
    assert torch.equal(policy.seen["observation.state"][0], policy.seen["observation.state"][4])


def test_broadcast_sampler_resets_before_every_query(observation):
    """Otherwise a frame's entropy would depend on which frames came before it."""
    policy = StochasticStandIn()
    sampler = XVLAChunkSampler(policy, num_samples=3)
    for _ in range(4):
        sampler(observation)
    assert policy.resets == 4


def test_broadcast_sampler_leaves_the_caller_batch_alone(observation):
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else list(v)) for k, v in observation.items()}
    XVLAChunkSampler(StochasticStandIn(), num_samples=3)(observation)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(observation[key], value)
        else:
            assert observation[key] == value


# --- broadcast itself -------------------------------------------------------


def test_broadcast_expands_tensors_and_repeats_metadata():
    out = broadcast(
        {
            "state": torch.zeros(1, 7),
            "images": [torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8)],
            "task": ["stack the cups"],
            "steps": 4,
        },
        num_samples=3,
    )
    assert out["state"].shape == (3, 7)
    assert [t.shape for t in out["images"]] == [(3, 3, 8, 8), (3, 3, 8, 8)]
    assert out["task"] == ["stack the cups"] * 3
    assert out["steps"] == 4


def test_broadcast_is_idempotent_on_already_broadcast_tensors():
    assert broadcast(torch.zeros(3, 7), num_samples=3).shape == (3, 7)


def test_broadcast_refuses_a_real_multi_row_batch():
    """Labelling asks about one frame; a batch of 2 would silently mix observations."""
    with pytest.raises(ValueError, match="expected a batch of 1"):
        broadcast(torch.zeros(2, 7), num_samples=3)


def test_broadcast_leaves_scalars_alone():
    """A dataset frame carries 0-dim tensors -- frame index, timestamp, domain id.

    They are not indexed by batch element, and policies that read one broadcast it
    themselves (xVLA's `_get_domain_id`). Found by running a real LIBERO episode.
    """
    out = broadcast({"state": torch.zeros(1, 7), "domain_id": torch.tensor(3)}, num_samples=4)
    assert out["state"].shape == (4, 7)
    assert out["domain_id"].shape == ()
    assert out["domain_id"].item() == 3
