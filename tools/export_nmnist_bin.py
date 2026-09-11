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

"""Export N-MNIST test data as fall.bin + labels.bin.

Output (compatible with main_test.c and main_sd.c): fall.bin holds N * (T * 2 * 34 * 34) float32 values, labels.bin holds N int32 labels. Samples are fixed to T timesteps (zero-padded if shorter, truncated if longer). N-MNIST raw events span ~300 ms, so default T=10 gives 30 ms per frame.

Usage:
    python tools/export_nmnist_bin.py <data_root> <out_dir>
                                     [--timesteps N] [--time-window-us US]
"""

import argparse
import os
import numpy as np

import tonic
from tonic.transforms import ToFrame, Denoise


def to_chw(frames_np):
    """Normalize frame array to shape (T, 2, 34, 34)."""
    arr = frames_np
    if arr.ndim == 4:
        if arr.shape[1] == 2:
            return arr.astype(np.float32, copy=False)
        if arr.shape[3] == 2:
            return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 2:
        # (2, H, W) -> (1, 2, H, W)
        return arr[None, ...].astype(np.float32, copy=False)
    raise ValueError(f"Unexpected frame shape: {arr.shape}")


def get_nmnist(path, train, transform):
    """Load N-MNIST, compatible with different tonic API versions."""
    last_err = None
    for kw in ("save_to", "root", "location"):
        try:
            return tonic.datasets.NMNIST(**{kw: path}, train=train, transform=transform)
        except TypeError as e:
            last_err = e
    raise last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", help="N-MNIST raw data root (tonic save_to)")
    ap.add_argument("out_dir", help="output directory for fall.bin + labels.bin")
    ap.add_argument("--timesteps", type=int, default=10,
                    help="number of frames per sample (default: 10)")
    ap.add_argument("--time-window-us", type=int, default=None,
                    help="event bin width in microseconds. "
                         "Defaults to 300000/timesteps so total span stays ~300ms.")
    ap.add_argument("--filter-time-us", type=int, default=10000,
                    help="Denoise filter window (default: 10000)")
    ap.add_argument("--binarize", action="store_true",
                    help="Collapse ToFrame event counts to {0,1} for a true spike model. Must "
                         "match training (dataset input_encoding='spike' / binary_input=true).")
    args = ap.parse_args()

    data_root = args.data_root
    out_dir = args.out_dir
    timesteps = args.timesteps
    time_window_us = args.time_window_us if args.time_window_us is not None \
        else 300000 // timesteps
    filter_time_us = args.filter_time_us

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (H, W, 2), typically (34, 34, 2)

    print(f"data_root    = {data_root}")
    print(f"out_dir      = {out_dir}")
    print(f"timesteps    = {timesteps}, time_window = {time_window_us}us, "
          f"filter_time = {filter_time_us}us")
    print(f"sensor_size  = {sensor_size}")

    steps = [
        Denoise(filter_time=filter_time_us),
        ToFrame(sensor_size=sensor_size, time_window=time_window_us),
    ]
    if args.binarize:
        # Same {0,1} collapse the training loader applies for a spike model.
        steps.append(lambda frames: (np.asarray(frames) > 0).astype(np.float32))
        print("binarize     = True (event counts -> {0,1})")
    transform = tonic.transforms.Compose(steps)

    test_set = get_nmnist(data_root, train=False, transform=transform)
    total = len(test_set)
    print(f"Exporting {total} test samples...")

    os.makedirs(out_dir, exist_ok=True)
    fall_path = os.path.join(out_dir, "fall.bin")
    labels_path = os.path.join(out_dir, "labels.bin")

    labels = []
    floats_per_sample = timesteps * 2 * 34 * 34

    with open(fall_path, "wb") as f:
        for idx in range(total):
            frames, label = test_set[idx]

            if hasattr(frames, "numpy"):
                arr = frames.numpy()
            else:
                arr = np.array(frames)
            arr = to_chw(arr)  # (T, 2, 34, 34)

            # Truncate or zero-pad to fixed timesteps
            T = arr.shape[0]
            if T < timesteps:
                pad = np.zeros((timesteps - T, 2, 34, 34), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            elif T > timesteps:
                arr = arr[:timesteps]

            flat = arr.reshape(-1).astype(np.float32, copy=False)
            assert len(flat) == floats_per_sample, \
                f"Sample {idx}: expected {floats_per_sample} floats, got {len(flat)}"

            f.write(flat.tobytes())
            labels.append(int(label))

            if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                print(f"  [{idx+1:5d}/{total}]")

    # Write labels.bin
    np.array(labels, dtype=np.int32).tofile(labels_path)

    fall_size = os.path.getsize(fall_path)
    labels_size = os.path.getsize(labels_path)

    print(f"\nDone! Files written to {out_dir}/")
    print(f"  fall.bin   - {fall_size:,} bytes ({fall_size / 1024 / 1024:.1f} MB)")
    print(f"             ({total} samples x {timesteps} timesteps x {2*34*34} dims x 4 bytes)")
    print(f"  labels.bin - {labels_size:,} bytes ({total} labels x 4 bytes)")


if __name__ == "__main__":
    main()
