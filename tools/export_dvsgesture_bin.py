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

"""Export DVS Gesture test data for SD card (ZCU104 Vitis).

Output matches main_sd.c and main_test.c: fall.bin (T * D_flat float32 per sample, all concatenated) and labels.bin (N int32 labels). Each sample defaults to T=60 frames of 1024 float32 values (32x32, polarity merged); data is stored as float32 and converted to input_t at runtime. File names are kept <=8 chars for FAT32 compatibility.

Usage:
    python tools/export_dvsgesture_bin.py <output_dir> [--time-window US] [--max-frames N] [--num-samples N]
"""

import os
import sys
import argparse
import numpy as np

# Defaults matching configs/dvsgesture_fcn.toml
TIMESTEPS = 60
SPATIAL_FACTOR = 0.25   # 128->32
TIME_WINDOW_US = 25000  # 25ms
DENOISE_US = 10000
MERGE_POLARITY = True
NUM_CLASSES = 11


def load_dvsgesture_test_full(data_path, spatial_factor, merge_polarity):
    """Load pre-processed full DVS Gesture test set (288 samples) from .npy files."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.datasets.dvsgesture import DVSGestureFullDataset

    full_test_dir = os.path.join(data_path, "dvsgesture_full", "ibmGestureTest")
    test_ds = DVSGestureFullDataset(full_test_dir)

    ds_h = int(128 * spatial_factor)
    ds_w = int(128 * spatial_factor)
    n_ch = 1 if merge_polarity else 2
    dim_flat = n_ch * ds_h * ds_w

    return test_ds, ds_h, ds_w, n_ch, dim_flat


def load_dvsgesture_test_tonic(data_path, spatial_factor, time_window_us, denoise_us,
                               merge_polarity, max_frames):
    """Fallback: load DVS Gesture test set via tonic (264 samples)."""
    import tonic
    import tonic.transforms as transforms

    raw_sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
    ds_h = int(raw_sensor_size[0] * spatial_factor)
    ds_w = int(raw_sensor_size[1] * spatial_factor)
    ds_sensor_size = (ds_h, ds_w, 2)

    transform_list = []
    if denoise_us > 0:
        transform_list.append(transforms.Denoise(filter_time=denoise_us))
    if spatial_factor != 1.0:
        transform_list.append(transforms.Downsample(spatial_factor=spatial_factor))
    transform_list.append(
        transforms.ToFrame(sensor_size=ds_sensor_size, time_window=time_window_us)
    )
    transform = transforms.Compose(transform_list)

    test_ds = tonic.datasets.DVSGesture(save_to=data_path, train=False,
                                        transform=transform)

    # Cache for speed
    cache_path = os.path.join(data_path, "cache", "dvsgesture",
                              f"test_sf{spatial_factor}_tw{time_window_us}_sd")
    cached_ds = tonic.DiskCachedDataset(test_ds, cache_path=cache_path)

    n_ch = 1 if merge_polarity else 2
    dim_flat = n_ch * ds_h * ds_w

    return cached_ds, ds_h, ds_w, n_ch, dim_flat


def process_sample(frames, merge_polarity, max_frames, binary_input=False):
    """Merge polarity, truncate/pad to max_frames, optionally binarize."""
    # frames: (T_var, 2, H, W) numpy
    frames = np.asarray(frames, dtype=np.float32)

    if merge_polarity:
        frames = frames.sum(axis=1, keepdims=True)  # (T, 1, H, W)

    if binary_input:
        frames = (frames > 0).astype(np.float32)

    T = frames.shape[0]
    if T > max_frames:
        frames = frames[:max_frames]
    elif T < max_frames:
        pad_shape = (max_frames - T,) + frames.shape[1:]
        frames = np.concatenate([frames, np.zeros(pad_shape, dtype=np.float32)], axis=0)

    return frames  # (max_frames, C, H, W)


def main():
    parser = argparse.ArgumentParser(description="Export DVS Gesture test data for SD card")
    parser.add_argument("output_dir",
                        help="Output directory (e.g., <sd_mount>/dvgestur or output_c/<project>)")
    parser.add_argument("--data-path", default="data",
                        help="Dataset storage path (default: data)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Max samples to export (0 = all)")
    parser.add_argument("--time-window", type=int, default=TIME_WINDOW_US,
                        help=f"Time window in us (default: {TIME_WINDOW_US})")
    parser.add_argument("--max-frames", type=int, default=TIMESTEPS,
                        help=f"Max frames per sample (default: {TIMESTEPS})")
    parser.add_argument("--spatial-factor", type=float, default=SPATIAL_FACTOR,
                        help=f"Spatial downsample factor (default: {SPATIAL_FACTOR})")
    parser.add_argument("--no-merge-polarity", action="store_true",
                        help="Keep 2 polarity channels (default: merge to 1)")
    parser.add_argument("--binary", action="store_true",
                        help="Binarize frames: count > 0 -> 1 (event-driven style)")
    parser.add_argument("--use-tonic", action="store_true",
                        help="Force use of tonic dataset (264 samples) instead of full (288)")
    args = parser.parse_args()

    merge_pol = not args.no_merge_polarity
    max_T = args.max_frames

    print(f"Loading DVS Gesture test set...")
    print(f"  spatial_factor={args.spatial_factor}, time_window={args.time_window}us")
    print(f"  merge_polarity={merge_pol}, max_frames={max_T}, binary={args.binary}")

    # Try full dataset first (288 samples), fall back to tonic (264)
    full_test_dir = os.path.join(args.data_path, "dvsgesture_full", "ibmGestureTest")
    use_full = os.path.isdir(full_test_dir) and not args.use_tonic

    if use_full:
        print(f"  Using pre-processed full dataset from {full_test_dir}")
        test_ds, ds_h, ds_w, n_ch, dim_flat = load_dvsgesture_test_full(
            args.data_path, args.spatial_factor, merge_pol
        )
        preprocessed = True
    else:
        print(f"  Using tonic dataset (fallback)")
        test_ds, ds_h, ds_w, n_ch, dim_flat = load_dvsgesture_test_tonic(
            args.data_path, args.spatial_factor, args.time_window,
            DENOISE_US, merge_pol, max_T
        )
        preprocessed = False

    n = len(test_ds)
    if args.num_samples > 0:
        n = min(n, args.num_samples)

    sample_floats = max_T * dim_flat
    sample_bytes = sample_floats * 4
    print(f"  Test samples: {len(test_ds)}, exporting: {n}")
    print(f"  Per sample: T={max_T}, C={n_ch}, H={ds_h}, W={ds_w}, "
          f"D_flat={dim_flat}, bytes={sample_bytes}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    fall_path = os.path.join(out_dir, "fall.bin")
    labels_path = os.path.join(out_dir, "labels.bin")

    all_labels = []
    short_count = 0

    with open(fall_path, "wb") as f:
        for i in range(n):
            raw_frames, y = test_ds[i]
            raw_frames = np.asarray(raw_frames, dtype=np.float32)
            raw_T = raw_frames.shape[0]

            if preprocessed:
                # .npy files already have correct shape (T, C, H, W)
                frames = raw_frames
                if args.binary:
                    frames = (frames > 0).astype(np.float32)
            else:
                frames = process_sample(raw_frames, merge_pol, max_T, binary_input=args.binary)

            assert frames.shape == (max_T, n_ch, ds_h, ds_w), \
                f"Sample {i}: unexpected shape {frames.shape}"

            f.write(frames.tobytes())
            all_labels.append(int(y))

            if raw_T < max_T:
                short_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"  [{i+1:4d}/{n}] label={y}, raw_T={raw_T}")

    # Write labels
    labels_arr = np.array(all_labels, dtype=np.int32)
    with open(labels_path, "wb") as f:
        f.write(labels_arr.tobytes())

    fall_size = os.path.getsize(fall_path)
    labels_size = os.path.getsize(labels_path)

    print(f"\nDone! Files written to {out_dir}/")
    print(f"  fall.bin   - {fall_size:,} bytes ({fall_size / 1024 / 1024:.1f} MB)")
    print(f"  labels.bin - {labels_size:,} bytes ({n} labels x 4 bytes)")
    if short_count > 0:
        print(f"  NOTE: {short_count}/{n} samples shorter than {max_T} frames (zero-padded)")


if __name__ == "__main__":
    main()
