# This file is part of SNN2Bitstream.
# Copyright (C) 2026 Xindan Zhang, Sorbonne Université, CNRS, LIP6

# SNN2Bitstream is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SNN2Bitstream is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import Dataset, DataLoader, random_split
from frontend.datasets import pad_collate


DEFAULT_ENCODING = "temporal"


class AugmentedDataset(Dataset):
    """Wraps a dataset with spatial augmentations on frame tensors: random rotation +/-10 degrees, random crop with padding=4."""
    def __init__(self, dataset, output_size=48, pad=4, max_angle=10.0):
        self.dataset = dataset
        self.output_size = output_size
        self.pad = pad
        self.max_angle = max_angle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = torch.as_tensor(x, dtype=torch.float32)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-self.max_angle, self.max_angle)
            x = TF.rotate(x, angle)

        # Random crop with padding
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        t, c, h, w = x.shape
        i = random.randint(0, h - self.output_size)
        j = random.randint(0, w - self.output_size)
        x = x[:, :, i:i+self.output_size, j:j+self.output_size]

        return x, y


def get_loaders(dataset_cfg, batch_size, timesteps):
    """Load CIFAR10-DVS dataset (temporal event-based).

    Uses n_time_bins mode: total event stream divided into fixed number of bins.
    Default: 48x48 spatial resolution.

    Config keys:
        path:         dataset storage path (default: "data")
        target_h:     spatial height after downsample (default: 48)
        target_w:     spatial width after downsample (default: 48)
        seed:         random seed for train/test split (default: 42)
        train_ratio:  fraction for training (default: 0.9)
        augment:      enable data augmentation (default: false)

    Returns:
        (train_loader, test_loader, input_dim, num_classes)
        - input_dim: (2, target_h, target_w)
        - num_classes: 10
    """
    path = dataset_cfg["path"]
    target_h = int(dataset_cfg.get("target_h", 48))
    target_w = int(dataset_cfg.get("target_w", 48))
    seed = int(dataset_cfg.get("seed", 42))
    train_ratio = float(dataset_cfg.get("train_ratio", 0.9))
    augment = dataset_cfg.get("augment", False)

    raw_sensor_size = tonic.datasets.CIFAR10DVS.sensor_size  # (128, 128, 2)
    target_sensor_size = (target_h, target_w, 2)

    transform = transforms.Compose([
        transforms.Downsample(
            sensor_size=raw_sensor_size,
            target_size=(target_h, target_w)),
        transforms.ToFrame(
            sensor_size=target_sensor_size,
            n_time_bins=timesteps),
    ])

    os.makedirs(path, exist_ok=True)
    # Fix Figshare download URL (AWS WAF blocks the default URL)
    tonic.datasets.CIFAR10DVS.url = "https://ndownloader.figshare.com/files/38023437"

    cached_ds = DiskCachedDataset(
        tonic.datasets.CIFAR10DVS(save_to=path, transform=transform),
        cache_path=os.path.join(path, "cache",
                                f"cifar10dvs_{target_h}x{target_w}_t{timesteps}")
    )

    # CIFAR10-DVS has no separate train/test; use random split
    gen = torch.Generator().manual_seed(seed)
    train_size = int(train_ratio * len(cached_ds))
    test_size = len(cached_ds) - train_size
    train_ds, test_ds = random_split(cached_ds, [train_size, test_size],
                                     generator=gen)

    # Apply augmentation to training set only
    if augment:
        train_ds = AugmentedDataset(train_ds, output_size=target_h)
        print(f"[Info] Data augmentation enabled (rotation +/-10deg, random crop pad=4)")

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=pad_collate, num_workers=num_workers,
                             pin_memory=True)

    input_dim = (2, target_h, target_w)
    num_classes = 10
    return train_loader, test_loader, input_dim, num_classes
