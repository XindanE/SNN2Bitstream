#!/usr/bin/env python3
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

"""Export CIFAR10-DVS test data for SD card (ZCU104 Vitis).

Output matches main_sd.c and main_test.c: fall.bin (T * D_flat float32 per sample, all concatenated) and labels.bin (N int32 labels). Temporal encoding gives each sample T frames of C*H*W float32 values; data is stored as float32 and converted to input_t at runtime. File names are kept <=8 chars for FAT32 compatibility.

Usage:
    python tools/export_cifar10dvs_bin.py <output_dir> [--data-path data] [--synthetic]
"""

import os
import sys
import struct
import argparse
import numpy as np


TIMESTEPS = 16
INPUT_C, INPUT_H, INPUT_W = 2, 48, 48
DIM_FLAT = INPUT_C * INPUT_H * INPUT_W  # 4608
NUM_CLASSES = 10


def load_cifar10dvs_test(data_path, seed, target_h, target_w, timesteps):
    """Load CIFAR10-DVS test split via tonic."""
    import torch
    import tonic
    from torch.utils.data import random_split

    transform = tonic.transforms.Compose([
        tonic.transforms.Downsample(
            sensor_size=tonic.datasets.CIFAR10DVS.sensor_size,
            target_size=(target_h, target_w)),
        tonic.transforms.ToFrame(
            sensor_size=(target_h, target_w, INPUT_C),
            n_time_bins=timesteps)
    ])

    os.makedirs(data_path, exist_ok=True)
    tonic.datasets.CIFAR10DVS.url = "https://ndownloader.figshare.com/files/38023437"
    cached_ds = tonic.DiskCachedDataset(
        tonic.datasets.CIFAR10DVS(save_to=data_path, transform=transform),
        cache_path=os.path.join(data_path, "../cache/cifar10dvs_sd_export")
    )

    # Same split as training code (seed=42, train_ratio=0.9)
    gen = torch.Generator().manual_seed(seed)
    train_size = int(0.9 * len(cached_ds))
    test_size = len(cached_ds) - train_size
    _, test_ds = random_split(cached_ds, [train_size, test_size], generator=gen)

    return test_ds


def main():
    parser = argparse.ArgumentParser(description="Export CIFAR10-DVS test data for SD card")
    parser.add_argument("output_dir", help="Output directory (e.g., <sd_mount>/cifarate or output_c/<project>)")
    parser.add_argument("--data-path", default="data", help="Dataset storage path (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split (default: 42)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Max samples to export (0 = all test samples)")
    parser.add_argument("--target-h", type=int, default=INPUT_H, help=f"Target height (default: {INPUT_H})")
    parser.add_argument("--target-w", type=int, default=INPUT_W, help=f"Target width (default: {INPUT_W})")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help=f"Timesteps (default: {TIMESTEPS})")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data (skip dataset)")
    args = parser.parse_args()

    T = args.timesteps
    H, W = args.target_h, args.target_w
    dim_flat = INPUT_C * H * W
    sample_floats = T * dim_flat  # floats per sample
    sample_bytes = sample_floats * 4  # bytes per sample

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    fall_path = os.path.join(out_dir, "fall.bin")
    labels_path = os.path.join(out_dir, "labels.bin")

    if args.synthetic:
        n = args.num_samples if args.num_samples > 0 else 100
        print(f"Generating {n} synthetic CIFAR10-DVS samples...")
        print(f"  T={T}, C={INPUT_C}, H={H}, W={W}, dim_flat={dim_flat}")

        rng = np.random.RandomState(args.seed)
        all_labels = []

        with open(fall_path, "wb") as f:
            for i in range(n):
                # DVS-like sparse data: ~5% event rate
                x = np.zeros((T, INPUT_C, H, W), dtype=np.float32)
                mask = rng.random(x.shape) < 0.05
                x[mask] = rng.randint(1, 4, size=mask.sum()).astype(np.float32)
                f.write(x.tobytes())
                all_labels.append(rng.randint(0, NUM_CLASSES))
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1:5d}/{n}]")
    else:
        print(f"Loading CIFAR10-DVS test set...")
        print(f"  T={T}, C={INPUT_C}, H={H}, W={W}, dim_flat={dim_flat}")
        test_ds = load_cifar10dvs_test(args.data_path, args.seed, H, W, T)
        n = len(test_ds)
        if args.num_samples > 0:
            n = min(n, args.num_samples)
        print(f"  Test samples: {len(test_ds)}, exporting: {n}")

        all_labels = []
        with open(fall_path, "wb") as f:
            for i in range(n):
                x, y = test_ds[i]
                x = np.asarray(x, dtype=np.float32)  # (T, C, H, W)
                assert x.shape == (T, INPUT_C, H, W), \
                    f"Unexpected shape {x.shape}, expected ({T}, {INPUT_C}, {H}, {W})"
                f.write(x.tobytes())
                all_labels.append(int(y))
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1:5d}/{n}]")

    # Write labels.bin
    labels_arr = np.array(all_labels, dtype=np.int32)
    with open(labels_path, "wb") as f:
        f.write(labels_arr.tobytes())

    fall_size = os.path.getsize(fall_path)
    labels_size = os.path.getsize(labels_path)

    print(f"\nDone! Files written to {out_dir}/")
    print(f"  fall.bin   - {fall_size:,} bytes ({fall_size / 1024 / 1024:.1f} MB)")
    print(f"             ({n} samples x {T} timesteps x {dim_flat} dims x 4 bytes)")
    print(f"  labels.bin - {labels_size:,} bytes ({n} labels x 4 bytes)")
    print(f"\n  Per sample: {sample_bytes:,} bytes ({sample_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
