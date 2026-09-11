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

# frontend/nmnist.py
import os
import numpy as np
import torch
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from frontend.datasets import pad_collate


DEFAULT_ENCODING = "temporal"


def _binarize_frames(frames):
    """Spike input: collapse ToFrame event counts to {0,1}. Applied after ToFrame."""
    return (np.asarray(frames) > 0).astype(np.float32)


def get_loaders(dataset_cfg, batch_size, timesteps):
    path = dataset_cfg["path"]
    # Frame window must match the deploy-time bin export (export_nmnist_bin.py)
    time_window_us = int(dataset_cfg.get("time_window_us", 300000 // max(int(timesteps), 1)))
    denoise_us     = int(dataset_cfg.get("denoise_us", 10000))
    augment        = dataset_cfg.get("augment", False)
    # Spike vs count input knob: binarize the count frames to {0,1}
    binarize = bool(dataset_cfg.get("binary_input", False)) or \
        str(dataset_cfg.get("input_encoding", "")).lower() == "spike"

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (H, W, C) = (34,34,2)

    def _frame_pipeline(with_flip):
        tf = [transforms.Denoise(filter_time=denoise_us)]
        if with_flip:
            tf.append(transforms.RandomFlipLR(sensor_size=sensor_size, p=0.5))
        tf.append(transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us))
        if binarize:
            tf.append(_binarize_frames)
        return transforms.Compose(tf)

    base_transform  = _frame_pipeline(with_flip=False)
    train_transform = _frame_pipeline(with_flip=True) if augment else base_transform
    test_transform  = base_transform

    train_set = tonic.datasets.NMNIST(save_to=path, train=True,  transform=train_transform)
    test_set  = tonic.datasets.NMNIST(save_to=path, train=False, transform=test_transform)

    # Cache keyed on (time_window, encoding): different T or spike/count must not share
    cache_tag  = f"tw{time_window_us}" + ("_bin" if binarize else "")
    cache_path = os.path.join(path, "cache", cache_tag)
    if augment:
        # Skip training cache
        test_set = DiskCachedDataset(test_set, cache_path=os.path.join(cache_path, "test"))
    else:
        train_set = DiskCachedDataset(train_set, cache_path=os.path.join(cache_path, "train"))
        test_set  = DiskCachedDataset(test_set,  cache_path=os.path.join(cache_path, "test"))
        print(f"[Info] DiskCache at {cache_path}")

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)

    input_dim = (sensor_size[2], sensor_size[0], sensor_size[1])  # (2,34,34)
    num_classes = 10
    return train_loader, test_loader, input_dim, num_classes
