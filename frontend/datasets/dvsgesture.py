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

# frontend/dvsgesture.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from frontend.datasets import pad_collate


DEFAULT_ENCODING = "temporal"


class DVSGestureFullDataset(Dataset):
    """Load pre-processed DVS Gesture .npy frames from dvsgesture_full/.

    Each subdirectory = one recording session, containing .npy files per gesture.
    Filename convention: {class_id}.npy or {class_id}b.npy (for duplicates).
    """
    def __init__(self, root_dir, max_frames=0, merge_polarity=False):
        self.max_frames = max_frames
        self.merge_polarity = merge_polarity
        self.samples = []  # list of (npy_path, label)
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in sorted(os.listdir(subdir_path)):
                if not fname.endswith('.npy'):
                    continue
                label = int(fname.replace('b.npy', '.npy').replace('.npy', ''))
                self.samples.append((os.path.join(subdir_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path)  # already processed: (T, C, H, W)
        if self.merge_polarity and frames.ndim == 4 and frames.shape[1] > 1:
            frames = frames.sum(axis=1, keepdims=True)
        if self.max_frames > 0 and frames.shape[0] > self.max_frames:
            frames = frames[:self.max_frames]
        return frames, label


def get_loaders(dataset_cfg, batch_size, timesteps):
    """Load IBM DVS Gesture dataset.

    Uses pre-processed full dataset (1176 train + 288 test) from
    data/dvsgesture_full/ if available, otherwise falls back to tonic.

    Config options:
        max_frames (int): Truncate frame sequences to this length. Default 0 (no truncation).
        use_tonic (bool): Force use of tonic dataset. Default False.

    Returns:
        (train_loader, test_loader, input_dim, num_classes)
    """
    path = dataset_cfg["path"]
    spatial_factor = float(dataset_cfg.get("spatial_factor", 0.25))
    max_frames = int(dataset_cfg.get("max_frames", 0))
    merge_polarity = bool(dataset_cfg.get("merge_polarity", False))
    use_tonic = bool(dataset_cfg.get("use_tonic", False))

    ds_h = int(128 * spatial_factor)
    ds_w = int(128 * spatial_factor)

    # Check for pre-processed full dataset
    full_data_dir = os.path.join(path, "dvsgesture_full")
    full_train_dir = os.path.join(full_data_dir, "ibmGestureTrain")
    full_test_dir = os.path.join(full_data_dir, "ibmGestureTest")

    if os.path.isdir(full_train_dir) and os.path.isdir(full_test_dir) and not use_tonic:
        print(f"[Info] Using pre-processed full DVS Gesture dataset from {full_data_dir}")
        train_set = DVSGestureFullDataset(full_train_dir, max_frames=max_frames,
                                          merge_polarity=merge_polarity)
        test_set = DVSGestureFullDataset(full_test_dir, max_frames=max_frames,
                                         merge_polarity=merge_polarity)
        print(f"[Info] Train: {len(train_set)}, Test: {len(test_set)}")
        # Detect actual channel count from a sample (files may be 1- or 2-channel)
        sample_frames, _ = train_set[0]
        n_channels = sample_frames.shape[1]
    else:
        print(f"[Info] Falling back to tonic DVS Gesture dataset")
        return _get_loaders_tonic(dataset_cfg, batch_size, timesteps)

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=pad_collate, num_workers=num_workers,
                             pin_memory=True)

    input_dim = (n_channels, ds_h, ds_w)
    num_classes = 11
    print(f"[Info] Input dim: {input_dim}, num_classes: {num_classes}")
    return train_loader, test_loader, input_dim, num_classes


def _get_loaders_tonic(dataset_cfg, batch_size, timesteps):
    """Fallback: load via tonic (264 test samples)."""
    import tonic
    import tonic.transforms as transforms
    from tonic import DiskCachedDataset

    path = dataset_cfg["path"]
    time_window_us = int(dataset_cfg.get("time_window_us", 50000))
    denoise_us = int(dataset_cfg.get("denoise_us", 10000))
    augment = dataset_cfg.get("augment", False)
    spatial_factor = float(dataset_cfg.get("spatial_factor", 0.25))
    merge_polarity = bool(dataset_cfg.get("merge_polarity", False))
    max_frames = int(dataset_cfg.get("max_frames", 0))
    binary_input = bool(dataset_cfg.get("binary_input", False))

    raw_sensor_size = tonic.datasets.DVSGesture.sensor_size
    ds_h = int(raw_sensor_size[0] * spatial_factor)
    ds_w = int(raw_sensor_size[1] * spatial_factor)
    ds_sensor_size = (ds_h, ds_w, 2)

    class PostFrameTransform:
        def __init__(self, merge_pol, max_t, binary):
            self.merge_pol = merge_pol
            self.max_t = max_t
            self.binary = binary
        def __call__(self, frames):
            if self.merge_pol:
                frames = frames.sum(axis=1, keepdims=True)
            if self.binary:
                frames = (frames > 0).astype(frames.dtype)
            if self.max_t > 0 and frames.shape[0] > self.max_t:
                frames = frames[:self.max_t]
            return frames

    need_post = merge_polarity or max_frames > 0 or binary_input
    post_transform = PostFrameTransform(merge_polarity, max_frames, binary_input) if need_post else None

    transform_list = []
    if denoise_us > 0:
        transform_list.append(transforms.Denoise(filter_time=denoise_us))
    if spatial_factor != 1.0:
        transform_list.append(transforms.Downsample(spatial_factor=spatial_factor))

    base_transform_list = transform_list + [
        transforms.ToFrame(sensor_size=ds_sensor_size, time_window=time_window_us),
    ]
    if post_transform is not None:
        base_transform_list.append(post_transform)
    base_transform = transforms.Compose(base_transform_list)

    if augment:
        aug_list = list(transform_list)
        aug_list.append(transforms.RandomFlipLR(sensor_size=ds_sensor_size, p=0.5))
        aug_list.append(transforms.ToFrame(sensor_size=ds_sensor_size, time_window=time_window_us))
        if post_transform is not None:
            aug_list.append(post_transform)
        train_transform = transforms.Compose(aug_list)
    else:
        train_transform = base_transform

    train_set = tonic.datasets.DVSGesture(save_to=path, train=True, transform=train_transform)
    test_set = tonic.datasets.DVSGesture(save_to=path, train=False, transform=base_transform)

    cache_path = os.path.join(path, "cache", "dvsgesture")
    cache_suffix = f"_sf{spatial_factor}_tw{time_window_us}"
    if merge_polarity:
        cache_suffix += "_mp"
    if max_frames > 0:
        cache_suffix += f"_mf{max_frames}"
    if binary_input:
        cache_suffix += "_bin"

    if augment:
        test_set = DiskCachedDataset(test_set, cache_path=os.path.join(cache_path, f"test{cache_suffix}"))
    else:
        train_set = DiskCachedDataset(train_set, cache_path=os.path.join(cache_path, f"train{cache_suffix}"))
        test_set = DiskCachedDataset(test_set, cache_path=os.path.join(cache_path, f"test{cache_suffix}"))

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)

    n_channels = 1 if merge_polarity else 2
    input_dim = (n_channels, ds_h, ds_w)
    num_classes = 11
    print(f"[Info] Input dim: {input_dim}, num_classes: {num_classes}")
    return train_loader, test_loader, input_dim, num_classes
