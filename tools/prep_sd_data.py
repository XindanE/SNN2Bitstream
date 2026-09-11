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

"""Stage a project's board test data for the SD card, splitting fall.bin to fit FAT32.

Reads SD_DATA_DIR / SD_SAMPLES_PER_FILE / SD_NUM_SAMPLES from the project's generated
model.h, takes the local single test_data/<dataset>/fall.bin (+ labels.bin), and writes
an SD-ready tree under <out>/<SD_DATA_DIR>/:

  * SD_SAMPLES_PER_FILE == 0 -> a single fall.bin (dataset already fits under 4 GiB).
  * SD_SAMPLES_PER_FILE  > 0 -> fall0.bin, fall1.bin, ... each holding that many samples,
    so every chunk stays under the FAT32 4 GiB/file limit.

labels.bin is small and never split. The FatFs harness (main_sd_streaming.c) reads the
matching fall{k}.bin automatically via the same SD_SAMPLES_PER_FILE value.

Usage:
    python tools/prep_sd_data.py <project_name> [--test-data-dir test_data]
                                 [--out sd_staging] [--sd /media/user/LABEL]
"""
import argparse
import os
import re
import shutil
import sys

_COPY_BLOCK = 64 * 1024 * 1024  # 64 MiB streaming copy blocks (never load the whole file)


def _read_defines(model_h):
    """Pull the SD_* / INPUT / TIMESTEPS integer + string defines out of model.h."""
    text = open(model_h).read()
    out = {}
    for name in ("SD_DATA_DIR",):
        m = re.search(r'#define\s+%s\s+"([^"]*)"' % name, text)
        if m:
            out[name] = m.group(1)
    for name in ("SD_SAMPLES_PER_FILE", "SD_NUM_SAMPLES", "TIMESTEPS", "INPUT_DIM_FLAT"):
        m = re.search(r'#define\s+%s\s+(\d+)' % name, text)
        if m:
            out[name] = int(m.group(1))
    return out


def _copy_range(src_f, dst_path, start, length):
    """Stream `length` bytes from an open file at `start` into dst_path."""
    src_f.seek(start)
    remaining = length
    with open(dst_path, "wb") as dst:
        while remaining > 0:
            buf = src_f.read(min(_COPY_BLOCK, remaining))
            if not buf:
                break
            dst.write(buf)
            remaining -= len(buf)
    return length - remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", help="project name under backend_projects/")
    ap.add_argument("--test-data-dir", default="test_data",
                    help="root holding the SW flow's test-data directories (default: test_data)")
    ap.add_argument("--out", default="sd_staging",
                    help="output staging root; SD tree is written under it (default: sd_staging)")
    ap.add_argument("--sd", default=None,
                    help="also copy the staged tree straight to this SD mount root")
    args = ap.parse_args()

    model_h = os.path.join("backend_projects", args.project, "cpp", "model.h")
    if not os.path.exists(model_h):
        sys.exit(f"[Error] {model_h} not found; generate the project first")
    d = _read_defines(model_h)

    sd_data_dir = d.get("SD_DATA_DIR", "").lstrip("0:").lstrip("/")   # 0:/nmnist/spike/t100 -> nmnist/spike/t100
    per_file    = d.get("SD_SAMPLES_PER_FILE", 0)
    if not sd_data_dir:
        sys.exit("[Error] SD_DATA_DIR missing from model.h")
    sd_dataset = sd_data_dir.split("/")[0]
    # SD_DATA_DIR uses FAT 8.3 short names (cifar10dvs->cifardvs, dvsgesture->dvsgest), but the
    # local test_data/ is keyed on the full dataset name. Reverse-map so the lookup still finds it.
    _SD_TO_FULL = {"cifardvs": "cifar10dvs", "dvsgest": "dvsgesture"}
    dataset = _SD_TO_FULL.get(sd_dataset, sd_dataset)

    # The SW flow caches under <dataset>_T<timesteps>_<encoding>, which SD_DATA_DIR
    # already spells out as <dataset>/<encoding>/t<timesteps>. Fall back to a flat
    # <dataset>/ for data exported by hand, as MNIST needs (the SW flow reads the
    # raw IDX files for MNIST and never writes fall.bin for it).
    parts = sd_data_dir.split("/")
    candidates = []
    if len(parts) >= 3:
        candidates.append(f"{dataset}_T{parts[2].lstrip('t')}_{parts[1]}")
    candidates.append(dataset)

    for name in candidates:
        src_dir    = os.path.join(args.test_data_dir, name)
        fall_path  = os.path.join(src_dir, "fall.bin")
        label_path = os.path.join(src_dir, "labels.bin")
        if os.path.exists(fall_path) and os.path.exists(label_path):
            break
    else:
        tried = ", ".join(os.path.join(args.test_data_dir, c) for c in candidates)
        sys.exit(f"[Error] no fall.bin + labels.bin in {tried}; run the SW flow to generate test data first")

    num_samples = os.path.getsize(label_path) // 4          # int32 labels
    fall_bytes  = os.path.getsize(fall_path)
    if num_samples == 0:
        sys.exit("[Error] labels.bin is empty")
    if fall_bytes % num_samples != 0:
        print(f"[Warn] fall.bin ({fall_bytes} B) not a clean multiple of {num_samples} samples")
    bytes_per_sample = fall_bytes // num_samples
    if "SD_NUM_SAMPLES" in d and d["SD_NUM_SAMPLES"] != num_samples:
        print(f"[Warn] labels.bin has {num_samples} samples but model.h SD_NUM_SAMPLES="
              f"{d['SD_NUM_SAMPLES']}")

    dst_dir = os.path.join(args.out, sd_data_dir)
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[prep_sd_data] {args.project}")
    print(f"  SD path:          0:/{sd_data_dir}")
    print(f"  samples:          {num_samples}  ({bytes_per_sample} B/sample)")
    print(f"  split per file:   {per_file if per_file else '(none, single fall.bin)'}")

    with open(fall_path, "rb") as src_f:
        if per_file and num_samples > per_file:
            n_files = (num_samples + per_file - 1) // per_file
            for k in range(n_files):
                cnt   = min(per_file, num_samples - k * per_file)
                start = k * per_file * bytes_per_sample
                length = cnt * bytes_per_sample
                out_path = os.path.join(dst_dir, f"fall{k}.bin")
                _copy_range(src_f, out_path, start, length)
                gib = length / 1024 ** 3
                flag = "OK" if length < 4 * 1024 ** 3 else "!! >4GiB"
                print(f"  fall{k}.bin: {cnt} samples, {gib:.2f} GiB [{flag}]")
        else:
            _copy_range(src_f, os.path.join(dst_dir, "fall.bin"), 0, fall_bytes)
            print(f"  fall.bin: {num_samples} samples, {fall_bytes / 1024 ** 3:.2f} GiB")

    shutil.copyfile(label_path, os.path.join(dst_dir, "labels.bin"))
    print(f"  labels.bin: {num_samples} labels")
    print(f"[prep_sd_data] staged at {dst_dir}")

    if args.sd:
        sd_root = args.sd.rstrip("/")
        print(f"[prep_sd_data] copying to SD {sd_root} ...")
        for name in sorted(os.listdir(dst_dir)):
            dst = os.path.join(sd_root, sd_data_dir)
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(os.path.join(dst_dir, name), os.path.join(dst, name))
            print(f"    -> {sd_data_dir}/{name}")
        print("[prep_sd_data] SD copy done")
    else:
        top = args.out.rstrip("/") + "/" + sd_dataset
        print(f"\nCopy to SD with:\n  cp -r {top} <SD_ROOT>/")


if __name__ == "__main__":
    main()
