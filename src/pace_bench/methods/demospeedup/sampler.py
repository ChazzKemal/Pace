"""Draw several action chunks from one observation, so entropy has something to measure.

:mod:`pace_bench.methods.demospeedup.entropy` needs ``num_samples`` *different*
answers to the same question. Getting them means reaching for whatever randomness a
policy family already has and that its deterministic inference path suppresses --
ACT's VAE latent, a diffusion policy's noise, a flow-matching policy's initial
sample. There is no shared interface for that in LeRobot, so this module defines the
small one labelling needs (:class:`ChunkSampler`) and implements it for ACT.

ACT is the awkward one, and the only one that needs machinery here. Its inference
path *deletes* its randomness -- the CVAE latent is pinned to ``z = 0``, the prior's
mode -- so sampling means putting it back. Diffusion and xVLA keep theirs: both draw
fresh noise inside ``predict_action_chunk`` on every call, one draw per batch row. So
their samplers do nothing but repeat the observation and ask, through the policy's
public API, and the diversity falls out of upstream's own code.

That split is why the two kinds of sampler are shaped differently. ACT's drives the
*model* directly, so it builds the model's input itself from the config's declared
features. The other two drive the *policy*, so they broadcast the batch and let the
policy pick the keys it wants out of it.

Nothing here edits or wraps LeRobot. ``LatentSamplingACT`` is an ordinary subclass
that loads a trained model's weights, which is why the pinned dependency stays
untouched and a checkpoint needs no preparation to be labelled.

Reference: ``lingxiao-guo/DemoSpeedup`` @ ``34bd43a`` samples ACT by calling the
policy with a latent drawn from the prior instead of the ``z = 0`` mode.
"""

from __future__ import annotations

from typing import Protocol

import torch
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor


class ChunkSampler(Protocol):
    """Callable turning one preprocessed observation into a stack of action chunks."""

    def __call__(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns ``(num_samples, chunk_size, action_dim)``."""
        ...


def _draw_prior_latent(module, args: tuple) -> tuple:
    """Forward-pre-hook: replace the incoming latent with a draw from ``N(0, I)``."""
    (latent,) = args
    return (torch.randn_like(latent),)


class LatentSamplingACT(ACT):
    """ACT that decodes from a fresh VAE latent instead of the ``z = 0`` mode.

    Upstream builds ``latent_sample`` inline in :meth:`ACT.forward` -- the
    reparameterized posterior while training, all zeros otherwise -- and offers no
    hook to change it. Copying ``forward`` to swap one tensor would duplicate the
    whole encoder-input construction (backbone, camera tokens, position embeddings)
    and leave that copy to rot silently the next time LeRobot touches ACT.

    So this intercepts the one module the latent passes through instead. A
    forward-pre-hook on ``encoder_latent_input_proj`` swaps the zeros for a fresh
    ``N(0, I)`` sample of exactly the same shape, dtype and device, and the
    unmodified ``forward`` runs on top. A hook registers no parameters, so
    ``state_dict`` keys are identical to a plain :class:`ACT` and a trained model
    loads with ``strict=True``.
    """

    def __init__(self, config):
        if not config.use_vae:
            raise ValueError(
                "LatentSamplingACT needs a VAE-trained ACT: the latent is the only "
                "stochastic source, and with use_vae=False every sample would be "
                "identical, making the measured entropy a constant zero."
            )
        super().__init__(config)
        self.encoder_latent_input_proj.register_forward_pre_hook(_draw_prior_latent)


class ACTChunkSampler:
    """Sample action chunks from a trained :class:`ACTPolicy`.

    The policy's weights are copied into a :class:`LatentSamplingACT` once, at
    construction; the original policy is left exactly as it was, so a caller can go
    on using it for anything else.
    """

    def __init__(self, policy, num_samples: int = 10):
        """
        Args:
            policy: A trained ``ACTPolicy``.
            num_samples: Chunks drawn per observation. Upstream's default is 10.
        """
        if num_samples < 2:
            raise ValueError(f"entropy needs at least 2 samples, got {num_samples}")
        self.config = policy.config
        self.num_samples = num_samples

        model = LatentSamplingACT(policy.config)
        model.load_state_dict(policy.model.state_dict())
        model.to(next(policy.model.parameters()).device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> Tensor:
        """One observation in, ``(num_samples, chunk_size, action_dim)`` out.

        Only the features the config declares are forwarded, each reshaped from the
        shape that config declares. Everything else a dataset batch carries -- the
        task string, the recorded action, frame indices -- is left behind rather
        than guessed at: an action chunk and a batched camera image have the same
        number of dimensions, so any rule based on counting them is a coin flip.
        """
        config = self.config
        model_batch: dict = {}
        if config.robot_state_feature:
            model_batch[OBS_STATE] = self._tile(batch[OBS_STATE], config.robot_state_feature.shape)
        if config.env_state_feature:
            model_batch[OBS_ENV_STATE] = self._tile(batch[OBS_ENV_STATE], config.env_state_feature.shape)
        if config.image_features:
            model_batch[OBS_IMAGES] = [
                self._tile(batch[key], feature.shape) for key, feature in config.image_features.items()
            ]
        # One forward over `num_samples` copies of the observation: the hook draws an
        # independent latent per row, so the copies come back different.
        return self.model(model_batch)[0]

    def _tile(self, value: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Repeat one observation feature into ``(num_samples, *shape)``.

        Accepts it batched or not: whether a checkpoint's saved preprocessor adds a
        batch dimension depends on when it was written, and that is not a question
        the caller should have to answer.
        """
        expected = tuple(shape)
        actual = tuple(value.shape)
        if actual == expected:
            value = value.unsqueeze(0)
        elif actual != (1, *expected):
            raise ValueError(
                f"expected a feature shaped {expected} or {(1, *expected)}, got {actual}"
            )
        return value.expand(self.num_samples, *expected)


def broadcast(value, num_samples: int):
    """Repeat a single-observation batch ``num_samples`` times along the batch dim.

    Every tensor a preprocessed observation carries is batched to one row, and one
    row per sample is what the stochastic decoders need: they draw their noise as
    ``randn(batch_size, ...)``, so N rows of the same observation come back as N
    independent samples.

    Lists are followed into (a policy may hold its cameras in one), and a
    single-element list of non-tensors -- the task string a ``LeRobotDataset`` batch
    carries -- is repeated so it still lines up row for row.
    """
    if isinstance(value, Tensor):
        if value.ndim == 0:
            # A scalar is not indexed by batch element -- a dataset frame carries
            # several (frame index, timestamp, task/domain id). Policies that read
            # one broadcast it themselves; xVLA's `_get_domain_id` does exactly that.
            return value
        if value.shape[0] == num_samples:
            return value
        if value.shape[0] != 1:
            raise ValueError(
                f"expected a batch of 1 to broadcast, got leading dimension {value.shape[0]}. "
                "Labelling asks about one observation at a time."
            )
        return value.expand(num_samples, *value.shape[1:])
    if isinstance(value, list):
        if len(value) == 1 and not isinstance(value[0], Tensor):
            return value * num_samples
        return [broadcast(v, num_samples) for v in value]
    if isinstance(value, dict):
        return {k: broadcast(v, num_samples) for k, v in value.items()}
    return value


def collate(batches: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Stack single-observation batches into one batch of ``len(batches)`` rows.

    Each input is what the preprocessor returns for one frame: every tensor batched
    to a single row. Concatenating along that dim gives the policy several frames in
    one call, which is the whole point -- a diffusion chunk costs 100 sequential
    denoising steps no matter how wide the batch is, so the only way to use the card
    is to make it wider.
    """
    if len(batches) == 1:
        return batches[0]
    out: dict[str, Tensor] = {}
    for key in batches[0]:
        values = [b[key] for b in batches]
        first = values[0]
        if isinstance(first, Tensor):
            # Scalars are not indexed by batch element (frame index, timestamp, a
            # task id); broadcast() already passes them through, so keep the first.
            out[key] = first if first.ndim == 0 else torch.cat(values, dim=0)
        elif isinstance(first, list):
            out[key] = [v for value in values for v in value]
        else:
            out[key] = first
    return out


def repeat_frames(value, num_samples: int):
    """Repeat every row ``num_samples`` times, keeping each frame's copies adjacent.

    ``repeat_interleave`` rather than ``repeat``: row order decides which frame a
    returned chunk belongs to, and the caller reshapes the result to
    ``(frames, num_samples, ...)``. Tiling instead of interleaving would still
    return the right chunks in the wrong slots -- entropy would be measured against
    the wrong observation, and nothing downstream would notice.
    """
    if isinstance(value, Tensor):
        return value if value.ndim == 0 else value.repeat_interleave(num_samples, dim=0)
    if isinstance(value, list):
        if value and not isinstance(value[0], Tensor):
            return [v for v in value for _ in range(num_samples)]
        return [repeat_frames(v, num_samples) for v in value]
    if isinstance(value, dict):
        return {k: repeat_frames(v, num_samples) for k, v in value.items()}
    return value


class _BroadcastChunkSampler:
    """Sampler for policies whose ``predict_action_chunk`` is already stochastic.

    The observation is repeated ``num_samples`` times and handed to the policy once.
    Upstream draws its own noise per row -- flow matching's initial sample for xVLA,
    the denoising prior for Diffusion -- so the rows come back different without
    anything here touching the model.
    """

    def __init__(self, policy, num_samples: int = 10):
        if num_samples < 2:
            raise ValueError(f"entropy needs at least 2 samples, got {num_samples}")
        self.policy = policy
        self.config = policy.config
        self.num_samples = num_samples

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> Tensor:
        # Reset first, so each frame is judged on its own. Diffusion keeps a queue of
        # the last n_obs_steps observations for the online control loop, and
        # predict_action_chunk reads its input from that queue whenever it is
        # non-empty -- so a carried-over queue would make the entropy at one frame
        # depend on which frames happened to be visited before it, and a labelling
        # pass visits every frame. xVLA queues only actions (it conditions on a
        # single frame), so the reset is a no-op there; it costs nothing and keeps
        # one rule for every family.
        self.policy.reset()
        batch = self.prepare(dict(batch))
        return self.policy.predict_action_chunk(broadcast(batch, self.num_samples))

    @torch.no_grad()
    def sample_frames(self, batches: list[dict[str, Tensor]]) -> Tensor:
        """Sample for several frames at once. Returns ``(frames, num_samples, chunk, dim)``.

        Same answer as calling ``__call__`` per frame -- the rows are independent
        either way -- but one policy call instead of ``len(batches)`` of them. The
        draws differ from the one-at-a-time path because the noise comes from one
        wider ``randn``, which is a different point in the RNG stream, not a
        different distribution.
        """
        self.policy.reset()
        frames = len(batches)
        batch = self.prepare(collate(batches))
        chunks = self.policy.predict_action_chunk(repeat_frames(batch, self.num_samples))
        return chunks.reshape(frames, self.num_samples, *chunks.shape[1:])

    def prepare(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Last shaping before broadcasting. Identity unless a family needs more."""
        return batch


class XVLAChunkSampler(_BroadcastChunkSampler):
    """Sample action chunks from a trained ``XVLAPolicy``.

    xVLA denoises by flow matching from ``x1 = randn(batch_size, chunk, action_dim)``
    (``XVLAModel.generate_actions``), so a batch of N identical observations is N
    trajectories. The Florence2 encoder runs over all N rows rather than once with
    its output broadcast; that is upstream's own path, and worth revisiting only if
    profiling a real labelling run shows the encoder dominating.
    """


class DiffusionChunkSampler(_BroadcastChunkSampler):
    """Sample action chunks from a trained ``DiffusionPolicy``.

    ``conditional_sample`` starts from ``randn`` whenever no noise is passed in, so
    the N rows denoise from N different priors. ``predict_action_chunk`` takes its
    offline path here -- queues empty, observations read straight from the batch --
    which the reset above guarantees.
    """

    def __init__(self, policy, num_samples: int = 10):
        super().__init__(policy, num_samples)
        if policy.config.n_obs_steps != 1:
            raise ValueError(
                f"this policy conditions on {policy.config.n_obs_steps} observation steps, but "
                "labelling asks about one frame at a time and has no history to give it. "
                "Label with an n_obs_steps=1 proxy, or extend the runner to read a "
                "window of frames per query."
            )

    @torch.no_grad()
    def sample_frames(self, batches: list[dict[str, Tensor]]) -> Tensor:
        """Batched sampling that runs the vision encoder once per *frame*.

        The ``num_samples`` chunks of one frame condition on the same observation,
        so the generic path -- repeat the observation, hand the policy N identical
        images -- pays for the ResNet N times over. Only the denoising is per-sample.

        This splits ``generate_actions`` at its own seam: encode the distinct frames
        into ``(frames, cond_dim)``, repeat *that* into the denoiser, and slice the
        result exactly as upstream does. Measured on the pickplace card it is worth
        little in wall-clock (100 sequential DDPM steps dominate) but ~6x in memory,
        which is what makes a wide frame batch affordable at all.

        The two bypassed lines of ``predict_action_chunk`` are reproduced here: the
        queue branch (dead -- ``reset()`` empties them) and the image stacking.
        """
        self.policy.reset()
        frames = len(batches)
        batch = self.prepare(collate(batches))

        config = self.policy.config
        if config.image_features:
            batch = dict(batch)
            for key in config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in config.image_features], dim=-4)

        model = self.policy.diffusion
        global_cond = model._prepare_global_conditioning(batch)  # (frames, cond_dim)
        chunks = model.conditional_sample(
            frames * self.num_samples,
            global_cond=global_cond.repeat_interleave(self.num_samples, dim=0),
        )
        # generate_actions' own slice: the chunk starts at the current observation.
        start = config.n_obs_steps - 1
        chunks = chunks[:, start : start + config.n_action_steps]
        return chunks.reshape(frames, self.num_samples, *chunks.shape[1:])

    def prepare(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Give the state an observation-step axis.

        ``generate_actions`` wants ``(batch, n_obs_steps, ...)`` and reads the step
        count off the state. ``predict_action_chunk`` inserts that axis for images
        on its offline path but not for the state, because a dataloader batch
        already carries it; a single frame does not.
        """
        for key in (OBS_STATE, OBS_ENV_STATE):
            value = batch.get(key)
            if isinstance(value, Tensor) and value.ndim == 2:
                batch[key] = value.unsqueeze(1)
        return batch
