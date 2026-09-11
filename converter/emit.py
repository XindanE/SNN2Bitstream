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

"""Code generation: C-header weight/bias arrays and Jinja template rendering.

Streaming code generation lives in emit_streaming.py.
"""
import os
import numpy as np


def _write_array(path, name, arr, dtype="float"):
    """Write a numpy array (1-D through 4-D) as an auto-generated C/C++ header.
    For ap_int / ap_fixed dtypes, emits a guarded ``#ifdef __SYNTHESIS__`` block
    """
    is_ap   = "ap_int" in dtype or "ap_fixed" in dtype
    is_flt  = "float"  in dtype or "fixed"    in dtype  # values printed in float format

    def _body_1d(f, dtype_str, flt):
        _fmt = (lambda v: f"{v:.8e}f") if flt else (lambda v: f"{int(v):d}")
        flat = arr.reshape(-1)
        n    = flat.size
        f.write(f"const {dtype_str} {name}[{n}] = {{\n")
        for i, v in enumerate(flat):
            if i % 8 == 0:
                f.write("  ")
            f.write(_fmt(v))
            if i != n - 1:
                f.write(", ")
            if (i + 1) % 8 == 0 or i == n - 1:
                f.write("\n")
        f.write("};\n")

    def _body_2d(f, dtype_str, flt):
        _fmt = (lambda v: f"{v:.8e}f") if flt else (lambda v: f"{int(v):d}")
        O, I = arr.shape
        f.write(f"const {dtype_str} {name}[{O}][{I}] = {{\n")
        for o in range(O):
            f.write("  { " + ", ".join(_fmt(x) for x in arr[o]) + " },\n")
        f.write("};\n")

    def _body_3d(f, dtype_str, flt):
        _fmt = (lambda v: f"{v:.8e}f") if flt else (lambda v: f"{int(v):d}")
        C, KH, KW = arr.shape
        f.write(f"const {dtype_str} {name}[{C}][{KH}][{KW}] = {{\n")
        for c in range(C):
            f.write("  {\n")
            for kh in range(KH):
                f.write("    { " + ", ".join(_fmt(x) for x in arr[c, kh]) + " },\n")
            f.write("  },\n")
        f.write("};\n")

    def _body_4d(f, dtype_str, flt):
        _fmt = (lambda v: f"{v:.8e}f") if flt else (lambda v: f"{int(v):d}")
        OC, IC, KH, KW = arr.shape
        f.write(f"const {dtype_str} {name}[{OC}][{IC}][{KH}][{KW}] = {{\n")
        for oc in range(OC):
            f.write("  {\n")
            for ic in range(IC):
                f.write("    {\n")
                for kh in range(KH):
                    f.write("      { " + ", ".join(_fmt(x) for x in arr[oc, ic, kh]) + " },\n")
                f.write("    },\n")
            f.write("  },\n")
        f.write("};\n")

    body = {1: _body_1d, 2: _body_2d, 3: _body_3d, 4: _body_4d}[arr.ndim]
    fallback = "int32_t" if "ap_int" in dtype else "float"

    with open(path, "w") as f:
        f.write("// auto-generated\n")
        if is_ap:
            f.write("#ifdef __SYNTHESIS__\n#include <ap_int.h>\n#include <ap_fixed.h>\n")
            body(f, dtype, "ap_fixed" in dtype)
            f.write("#else\n#include <stdint.h>\n")
            body(f, fallback, fallback == "float")
            f.write("#endif\n")
        else:
            if "int" in dtype:
                f.write("#include <stdint.h>\n")
            body(f, dtype, is_flt)


WEIGHT_SPLIT_THRESHOLD = 300000  # max elements per sub-array to avoid an HLS clang crash


def _write_array_2d_split(path, name, arr, dtype="float"):
    """Split large 2-D weight arrays into sub-arrays for HLS compatibility.
    When O*I > WEIGHT_SPLIT_THRESHOLD, splits on the output (row) dimension.
    Returns number of chunks (1 = no split).
    """
    O, I = arr.shape
    total = O * I
    if total <= WEIGHT_SPLIT_THRESHOLD:
        _write_array(path, name, arr, dtype)
        return 1

    num_chunks = (total + WEIGHT_SPLIT_THRESHOLD - 1) // WEIGHT_SPLIT_THRESHOLD
    chunk_rows = (O + num_chunks - 1) // num_chunks

    for c in range(num_chunks):
        r0 = c * chunk_rows
        r1 = min(r0 + chunk_rows, O)
        _write_array(path.replace(".h", f"_{c}.h"), f"{name}_{c}", arr[r0:r1], dtype)

    is_ap    = "ap_int" in dtype or "ap_fixed" in dtype
    fallback = "int32_t" if "ap_int" in dtype else "float" if is_ap else dtype

    with open(path, "w") as f:
        f.write(f"// auto-generated - split {name}[{O}][{I}] into {num_chunks} chunks\n")
        for c in range(num_chunks):
            f.write(f'#include "{name}_{c}.h"\n')
        f.write(f"\n#define {name.upper()}_SPLIT {num_chunks}\n")
        f.write(f"#define {name.upper()}_CHUNK_ROWS {chunk_rows}\n\n")

        for guard, ret_type in (("#ifdef __SYNTHESIS__\n#pragma HLS INLINE\n", dtype),
                                ("#else\n", fallback)):
            if guard.startswith("#ifdef"):
                f.write(f"#ifdef __SYNTHESIS__\n")
                f.write(f"static inline {ret_type} {name}_get(int o, int i) {{\n")
                f.write(f"#pragma HLS INLINE\n")
            else:
                f.write(f"#else\n")
                f.write(f"static inline {ret_type} {name}_get(int o, int i) {{\n")
            for c in range(num_chunks):
                kw = "if" if c == 0 else "else if"
                r0 = c * chunk_rows
                r1 = min(r0 + chunk_rows, O)
                f.write(f"    {kw} (o < {r1}) return {name}_{c}[o - {r0}][i];\n")
            f.write("    return 0;\n}\n")
        f.write("#endif\n")

    print(f"[SPLIT] {name}[{O}][{I}] ({total} params) -> {num_chunks} chunks of {chunk_rows} rows")
    return num_chunks


# Dataset directory names must be <=8 chars (FAT 8.3); only the long ones need shortening.
_SD_DATASET_DIR = {"cifar10dvs": "cifardvs", "dvsgesture": "dvsgest", "ninapro_db5": "ninapro"}
# Encoding directory names describe the value type, not the time axis (everything is temporal).
# DVS ToFrame values are event *counts*, so 'temporal' -> 'count'; 'rate'/'repeat' stay as-is.
_SD_ENCODING_DIR = {"temporal": "count"}
_SD_TEST_SIZES  = {"mnist": 10000, "nmnist": 10000, "cifar10dvs": 1000, "dvsgesture": 288, "ninapro_db5": 4581}

def _sd_data_dir(dataset_kind, encoding, timesteps, sd_encoding=None, input_is_binary=None):
    """Nested FAT32 path for main_sd.c SD_DATA_DIR: <dataset>/<encoding>/t<T>.

    Each component is kept <=8 chars (FAT 8.3 short names). Splitting by dataset,
    encoding, and timesteps lets test sets for different T or encodings of the same
    dataset coexist on the SD card instead of colliding on one flat directory
    (e.g. nmnist/count/t10 vs nmnist/spike/t100).

    The encoding component is the frame *value type* (spike vs count), not the time axis
    (rate/temporal/repeat). Priority: explicit sd_encoding (TOML/CLI) > the resolved
    input_is_binary flag (spike when binary, count otherwise; the same signal that gates
    fabric, so a rate/binarized input correctly becomes 'spike') > legacy encoding fallback.
    """
    ds  = _SD_DATASET_DIR.get(dataset_kind, dataset_kind)[:8]
    if sd_encoding:
        enc = str(sd_encoding).strip().lower()[:8]
    elif input_is_binary is not None:
        enc = "spike" if input_is_binary else "count"
    else:
        enc = _SD_ENCODING_DIR.get(encoding, encoding or "repeat")[:8]
    return f"{ds}/{enc}/t{int(timesteps)}"

# SD card FAT32
_SD_FAT32_MAX_BYTES = int(3.7 * 1024**3)

def _sd_num_samples(dataset_kind):
    """Full number of board test samples for the dataset (the test set is never truncated,
    when it exceeds 4 GiB it is split across fall0.bin, fall1.bin, ... instead)."""
    return _SD_TEST_SIZES.get(dataset_kind, 10000)

def _sd_samples_per_file(bytes_per_sample, num_samples):
    """Samples per fall*.bin chunk so each stays under the FAT32 4 GiB/file limit. Returns 0
    when the whole test set already fits in one fall.bin (no split needed)."""
    if not bytes_per_sample or num_samples * bytes_per_sample <= _SD_FAT32_MAX_BYTES:
        return 0
    return _SD_FAT32_MAX_BYTES // int(bytes_per_sample)


def emit_param_headers(ir_path, ir, out_dir, fixed_config):
    """Load weight/bias CSVs from *ir* and write per-layer C header files."""
    base      = os.path.dirname(ir_path)
    use_fixed = fixed_config.get("use_fixed", False)
    os.makedirs(out_dir, exist_ok=True)

    for L in ir["layers"]:
        if L["type"] not in ("Linear", "Conv2d", "DepthwiseConv2d"):
            continue
        p     = L.get("pair_idx")
        w_csv = L.get("weight")
        b_csv = L.get("bias")
        if p is None or not w_csv:
            continue

        is_quant = "quant_weight" in L
        if is_quant:
            bit_width = L["quant_weight"]["bit_width"]
            if use_fixed:

                if bit_width <= 8:    dtype_c_w = "signed char"
                elif bit_width <= 16: dtype_c_w = "short"
                elif bit_width <= 32: dtype_c_w = "int"
                else:                 dtype_c_w = f"ap_int<{bit_width}>"
            else:
                dtype_c_w = "int32_t"
            dtype_c_b = "int32_t"
            dtype_np  = np.int32
        else:
            if use_fixed:
                w_bits, i_bits = fixed_config["width"], fixed_config["int"]
                dtype_c_w = dtype_c_b = f"ap_fixed<{w_bits},{i_bits}>"
            else:
                dtype_c_w = dtype_c_b = "float"
            dtype_np = np.float32

        w_path = os.path.join(base, w_csv)
        if not os.path.isfile(w_path):
            raise SystemExit(f"[Error] weight CSV not found: {w_path} (IR dir incomplete; re-run export_ir)")
        w = np.loadtxt(w_path, delimiter=",", dtype=dtype_np)
        b = np.loadtxt(os.path.join(base, b_csv), delimiter=",", dtype=dtype_np) if b_csv else None

        # Stash the bias array on the layer so convert_model can precompute bias_dequant_table for the LUT path that fixes bias_scale precision loss in the (acc_t) cast.
        if b is not None:
            L["_bias_loaded"] = b.flatten().tolist()

        match L["type"]:
            case "Linear":
                O, I = int(L["out_dim"]), int(L["prev_out_dim"])
                n_chunks = _write_array_2d_split(
                    os.path.join(out_dir, f"weights{p}.h"), f"weights{p}",
                    w.reshape(O, I), dtype=dtype_c_w)
                if n_chunks > 1:
                    L["_weight_split_chunks"]    = n_chunks
                    L["_weight_split_chunk_rows"] = (O + n_chunks - 1) // n_chunks
                if b is not None:
                    _write_array(os.path.join(out_dir, f"biases{p}.h"),
                                 f"biases{p}", b.reshape(O), dtype_c_b)

            case "DepthwiseConv2d":
                C, K = int(L["channels"]), int(L["kernel_size"])
                _write_array(os.path.join(out_dir, f"weights{p}.h"),
                             f"weights{p}", w.reshape(C, K, K), dtype_c_w)
                if b is not None:
                    _write_array(os.path.join(out_dir, f"biases{p}.h"),
                                 f"biases{p}", b.reshape(C), dtype_c_b)

            case "Conv2d":
                OC, IC, K = int(L["out_ch"]), int(L["in_ch"]), int(L["kernel_size"])
                _write_array(os.path.join(out_dir, f"weights{p}.h"),
                             f"weights{p}", w.reshape(OC, IC, K, K), dtype_c_w)
                if b is not None:
                    _write_array(os.path.join(out_dir, f"biases{p}.h"),
                                 f"biases{p}", b.reshape(OC), dtype_c_b)

        if is_quant:
            print(f"[Done] Quantized headers: {L.get('name', 'unknown')} (bit_width={bit_width})")
        if use_fixed:
            print(f"[Done] Fixed-point type {dtype_c_w}: {L.get('name', 'unknown')}")


def render_template(env, template_name, context, out_dir, out_name=None):
    rendered = env.get_template(template_name).render(context)
    out_path = os.path.join(out_dir, out_name or template_name.replace(".j2", ""))
    with open(out_path, "w") as f:
        f.write(rendered)


