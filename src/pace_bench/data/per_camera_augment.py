#!/usr/bin/env python3
"""Per-camera image augmentation for LeRobot v3 ACT training.

STATUS: written on request but NOT wired into training yet. The active setup
uses lerobot's built-in photometric-only augmentation (see the bottom of this
file for the equivalent CLI). This module exists for when per-camera control
is wanted later.

Why per-camera
--------------
The policy regresses end-effector poses, so any *geometric* image transform
that moves pixels without moving the action label can teach the wrong thing.
The two cameras differ:

  * ``observation.images.camera`` -- fixed TABLE camera. A *minimal* random
    crop is a safe regulariser: the viewpoint is static, so a small crop just
    discourages memorising exact pixel positions.
  * ``observation.images.d405`` -- WRIST camera, rigidly bolted to the EE.
    There is a fixed geometric relationship between what it sees and the
    action, so it gets photometric augmentation ONLY -- no crop, no affine.

lerobot's built-in ``ImageTransforms`` applies one config to every camera and
never sees the camera key, so per-camera behaviour needs this separate path.

Photometric augmentation (brightness/contrast/saturation/hue/sharpness) is
applied to BOTH cameras -- it is geometry-preserving and therefore always safe.

Integration
-----------
``lerobot-train`` builds the dataset internally, so the only way to inject this
without editing lerobot is to monkeypatch ``LeRobotDataset.__getitem__``. Call
``enable_per_camera_augmentation(...)`` once before training starts (e.g. from a
thin wrapper script around ``lerobot.scripts.train.train``).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

# Reuse lerobot's own building blocks so behaviour matches the built-in path.
from lerobot.transforms import RandomSubsetApply, SharpnessJitter

DEFAULT_TABLE_KEYS = ("observation.images.camera",)
DEFAULT_WRIST_KEYS = ("observation.images.d405",)


def make_photometric_transform() -> RandomSubsetApply:
    """Geometry-preserving colour/sharpness jitter, safe for every camera.

    Ranges are slightly tighter than lerobot's defaults: saturation and hue are
    narrowed because large shifts can recolour task objects misleadingly.
    Up to 3 of these are sampled per image (matching ``max_num_transforms=3``).
    """
    transforms = [
        v2.ColorJitter(brightness=(0.8, 1.2)),
        v2.ColorJitter(contrast=(0.8, 1.2)),
        v2.ColorJitter(saturation=(0.8, 1.2)),  # tightened from (0.5, 1.5)
        v2.ColorJitter(hue=(-0.03, 0.03)),      # tightened from (-0.05, 0.05)
        SharpnessJitter(sharpness=(0.5, 1.5)),  # <1 = blur, helps wrist motion blur
    ]
    return RandomSubsetApply(transforms, p=None, n_subset=3, random_order=True)


class MinimalTableCrop(torch.nn.Module):
    """A *really minimal* random crop+resize, for the static table camera only.

    Crops to ``scale`` of the area (kept very close to 1.0), aspect ratio fixed,
    then resizes back to the original resolution. Applied with probability
    ``crop_prob``; otherwise the image passes through untouched.
    """

    def __init__(self, scale: tuple[float, float] = (0.95, 1.0), crop_prob: float = 0.5):
        super().__init__()
        self.scale = scale
        self.crop_prob = crop_prob

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.crop_prob:
            return img
        h, w = img.shape[-2:]
        crop = v2.RandomResizedCrop(
            size=(h, w), scale=self.scale, ratio=(1.0, 1.0), antialias=True
        )
        return crop(img)


class PerCameraImageTransforms:
    """Applies photometric jitter to all cameras, a minimal crop to table cams.

    ``__call__`` takes a full LeRobot item dict and mutates the image entries
    in place (image tensors are CHW float in [0, 1]).
    """

    def __init__(
        self,
        table_camera_keys: tuple[str, ...] = DEFAULT_TABLE_KEYS,
        wrist_camera_keys: tuple[str, ...] = DEFAULT_WRIST_KEYS,
        crop_scale: tuple[float, float] = (0.95, 1.0),
        crop_prob: float = 0.5,
    ):
        self.table_camera_keys = set(table_camera_keys)
        self.wrist_camera_keys = set(wrist_camera_keys)
        self.photometric = make_photometric_transform()
        self.table_crop = MinimalTableCrop(scale=crop_scale, crop_prob=crop_prob)

    def __call__(self, item: dict) -> dict:
        for key in self.table_camera_keys | self.wrist_camera_keys:
            if key not in item:
                continue
            img = self.photometric(item[key])
            if key in self.table_camera_keys:
                img = self.table_crop(img)  # geometric: table camera ONLY
            item[key] = img
        return item


def enable_per_camera_augmentation(
    table_camera_keys: tuple[str, ...] = DEFAULT_TABLE_KEYS,
    wrist_camera_keys: tuple[str, ...] = DEFAULT_WRIST_KEYS,
    crop_scale: tuple[float, float] = (0.95, 1.0),
    crop_prob: float = 0.5,
) -> None:
    """Monkeypatch ``LeRobotDataset.__getitem__`` to use per-camera transforms.

    Disables lerobot's built-in (key-agnostic) ``image_transforms`` for the call
    and applies :class:`PerCameraImageTransforms` instead. Call once before
    constructing the training dataloader.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    transform = PerCameraImageTransforms(
        table_camera_keys, wrist_camera_keys, crop_scale, crop_prob
    )
    original_getitem = LeRobotDataset.__getitem__

    def patched_getitem(self, idx):
        saved = self.image_transforms
        self.image_transforms = None  # skip the built-in key-agnostic loop
        try:
            item = original_getitem(self, idx)
        finally:
            self.image_transforms = saved
        return transform(item)

    LeRobotDataset.__getitem__ = patched_getitem


# ---------------------------------------------------------------------------
# ACTIVE setup (option 1) -- photometric-only, no per-camera code needed.
# Pass these flags to `lerobot-train` directly:
#
#   --dataset.image_transforms.enable=true
#   --dataset.image_transforms.max_num_transforms=3
#   --dataset.image_transforms.tfs.affine.weight=0.0
#
# That enables brightness/contrast/saturation/hue/sharpness on both cameras and
# disables the geometric (RandomAffine) transform.
# ---------------------------------------------------------------------------
