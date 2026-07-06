"""Tests for the training image resampling behaviour of ``CacheDataloader``.

Regression coverage for the ``FlexibleDataManagerConfig`` default that used to cache a
single training image forever (see issue #18).
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from sdfstudio.data.datamanagers.base_datamanager import FlexibleDataManagerConfig
from sdfstudio.data.utils.dataloaders import CacheDataloader


class _IndexDataset(Dataset):
    """Minimal dataset returning its item index, enough to exercise CacheDataloader."""

    def __init__(self, num_images: int) -> None:
        self.num_images = num_images

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image_idx": torch.tensor(idx)}


def _sampled_indices(dataloader: CacheDataloader, num_iters: int) -> list[int]:
    iterator = iter(dataloader)
    return [int(next(iterator)["image_idx"].item()) for _ in range(num_iters)]


def test_never_resample_caches_single_image() -> None:
    """With num_times_to_repeat_images=-1 the same image is served forever."""
    random.seed(0)
    dataloader = CacheDataloader(
        _IndexDataset(num_images=5),
        num_images_to_sample_from=1,
        num_times_to_repeat_images=-1,
    )
    assert len(set(_sampled_indices(dataloader, num_iters=50))) == 1


def test_resample_every_iteration_rotates_images() -> None:
    """With num_times_to_repeat_images=0 a new image is resampled each iteration."""
    random.seed(0)
    dataloader = CacheDataloader(
        _IndexDataset(num_images=5),
        num_images_to_sample_from=1,
        num_times_to_repeat_images=0,
    )
    assert len(set(_sampled_indices(dataloader, num_iters=50))) > 1


def test_flexible_datamanager_resamples_by_default() -> None:
    """The Flexible config must resample so it does not overfit a single image."""
    config = FlexibleDataManagerConfig()
    assert config.train_num_images_to_sample_from == 1
    assert config.train_num_times_to_repeat_images == 0
