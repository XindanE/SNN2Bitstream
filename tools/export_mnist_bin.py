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

"""Generate MNIST test data for SD card (ZCU104 Vitis).

Output matches main_sd.c: fall.bin (all samples concatenated) and labels.bin (int32 labels).
File names are kept <=8 chars for FAT32 compatibility.

Encoding modes (--encoding):
  repeat: D normalized values per sample, (x/255 - 0.1307) / 0.3081; HLS IP reuses per timestep.
  rate:   T*D binary Poisson spikes per sample; pixel intensity is the spike probability.

Usage:
    python tools/export_mnist_bin.py <output_dir> --encoding {repeat|rate}
"""

import os
import sys
import struct
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export MNIST test data for SD card")
    parser.add_argument("output_dir", help="Output directory (e.g., SD card mount point or output_c/<project>)")
    parser.add_argument("--encoding", choices=["repeat", "rate"], default="repeat",
                        help="Encoding mode: repeat (D values) or rate (T*D spikes)")
    parser.add_argument("--timesteps", type=int, default=10, help="Number of timesteps (default: 10)")
    parser.add_argument("--input-dim-flat", type=int, default=784, help="Flattened input dim (default: 784)")
    parser.add_argument("--data-path", default="data", help="MNIST data root (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for rate encoding (default: 42)")
    args = parser.parse_args()

    T = args.timesteps
    D = args.input_dim_flat
    out_dir = args.output_dir
    encoding = args.encoding

    # Read raw MNIST test images and labels
    mnist_dir = os.path.join(args.data_path, "MNIST", "raw")
    image_file = os.path.join(mnist_dir, "t10k-images-idx3-ubyte")
    label_file = os.path.join(mnist_dir, "t10k-labels-idx1-ubyte")

    if not os.path.exists(image_file):
        print(f"[Error] MNIST image file not found: {image_file}")
        print("Run: python tools/download_data.py mnist")
        sys.exit(1)

    # Read labels
    with open(label_file, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad label magic: {magic}"
        raw_labels = np.frombuffer(f.read(num), dtype=np.uint8)

    # Read images
    with open(image_file, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad image magic: {magic}"
        raw_images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        raw_images = raw_images.reshape(num, rows * cols)

    print(f"Loaded {num} MNIST test images ({rows}x{cols})")
    print(f"Encoding: {encoding}, Timesteps: {T}, Input dim flat: {D}")

    # Verify dimensions
    assert raw_images.shape[1] == D, \
        f"Image dim {raw_images.shape[1]} != input_dim_flat {D}"

    os.makedirs(out_dir, exist_ok=True)
    fall_path = os.path.join(out_dir, "fall.bin")
    labels_path = os.path.join(out_dir, "labels.bin")

    if encoding == "repeat":
        # D normalized values per sample; HLS IP reuses them per timestep (INPUT_ENCODING_REPEAT=1).
        images_float = (raw_images.astype(np.float32) / 255.0 - 0.1307) / 0.3081
        sample_size = D  # only D values (no replication)

        print(f"Writing {fall_path} ({num} samples, {sample_size * 4} bytes each)...")
        with open(fall_path, "wb") as f:
            for i in range(num):
                f.write(images_float[i].tobytes())
                if (i + 1) % 2000 == 0:
                    print(f"  [{i+1:5d}/{num}]")

    elif encoding == "rate":
        # T*D binary Poisson spikes per sample; pixel intensity [0,1] is the spike probability.
        rng = np.random.RandomState(args.seed)
        probs = raw_images.astype(np.float32) / 255.0  # [num, D] in [0,1]
        sample_size = T * D  # T*D values per sample

        print(f"Writing {fall_path} ({num} samples, {sample_size * 4} bytes each)...")
        with open(fall_path, "wb") as f:
            for i in range(num):
                # Generate T frames of Poisson spikes
                for t in range(T):
                    spikes = (rng.rand(D) < probs[i]).astype(np.float32)
                    f.write(spikes.tobytes())
                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1:5d}/{num}]")

    total_size = os.path.getsize(fall_path)
    print(f"  fall.bin: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")

    # Write labels.bin: int32 array
    print(f"Writing {labels_path}...")
    labels_int32 = raw_labels.astype(np.int32)
    with open(labels_path, "wb") as f:
        f.write(labels_int32.tobytes())

    labels_size = os.path.getsize(labels_path)
    print(f"  labels.bin: {labels_size:,} bytes")

    print(f"\nDone! Files written to {out_dir}/")
    if encoding == "repeat":
        print(f"  fall.bin   - {total_size:,} bytes ({num} samples x {D} dims x 4 bytes)")
    else:
        print(f"  fall.bin   - {total_size:,} bytes ({num} samples x {T} timesteps x {D} dims x 4 bytes)")
    print(f"  labels.bin - {labels_size:,} bytes ({num} labels x 4 bytes)")


if __name__ == "__main__":
    main()
