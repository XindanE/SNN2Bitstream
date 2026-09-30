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
import json
import math

import torch
import numpy as np
import argparse


# HLS fixed-point type helpers. The FPGA has no native float, so weights and
# activations use fixed-point. ap_fixed<W, I> = W total bits, I integer bits, W-I fraction bits.


def _round_up_4(n):
    """Round up to the nearest multiple of 4, minimum 4. HLS DSP slices align to 4-bit boundaries."""
    return max(((n + 3) // 4) * 4, 4)

def derive_type_config(min_val, max_val, is_integer, is_signed, frac_bits=8):
    """Derive the HLS type config from the data range. Returns {"type_class", "width", "int"}."""
    if is_integer and not is_signed:
        bits = max(math.ceil(math.log2(max(int(max_val), 1) + 1)), 1)
        w = _round_up_4(bits)
        return {"type_class": "uint", "width": w, "int": w}

    elif is_integer and is_signed:
        abs_max = max(abs(int(min_val)), abs(int(max_val)), 1)
        bits = math.ceil(math.log2(abs_max + 1)) + 1  # +1 sign bit
        w = _round_up_4(bits)
        return {"type_class": "int", "width": w, "int": w}

    elif not is_signed:
        i = max(math.ceil(math.log2(max(max_val, 1e-10) + 1)), 1)
        return {"type_class": "ufixed", "width": i + frac_bits, "int": i}

    else:
        abs_max = max(abs(min_val), abs(max_val), 1e-10)
        i = math.ceil(math.log2(abs_max + 1)) + 1
        return {"type_class": "fixed", "width": i + frac_bits, "int": i}
    

def calculate_scale_bits(min_scale, safety_margin=2):
    """Return (total_width, int_width, frac_bits) needed to represent a quantization scale value."""
    if min_scale <= 0 or min_scale >= 1.0:
        return 16, 8, 8  # default ap_fixed<16, 8>

    # safety_margin keeps a few bits below the smallest scale so it isn't rounded toward 0.
    frac_bits = math.ceil(-math.log2(min_scale)) + safety_margin
    frac_bits = ((frac_bits + 7) // 8) * 8  # round up to multiple of 8
    frac_bits = min(max(frac_bits, 8), 32)  # clamp to [8, 32]

    int_width = 8  # scale values are always < 256
    return int_width + frac_bits, int_width, frac_bits

def analyze_scales_for_ir(layers):
    """Scan all quantized layers, find the smallest scale, return ap_fixed config dict for it."""
    all_scales = []
    for layer in layers:
        for key in ("quant_weight", "quant_bias"):
            if layer.get(key):
                scale = layer[key].get("scale", 1.0)
                if scale > 0:
                    all_scales.append(scale)

    if not all_scales:
        return {"auto_detect": False, "width": 16, "int": 8, "frac": 8,
                "min_scale": None, "max_scale": None}

    min_scale = min(all_scales)
    max_scale = max(all_scales)
    total_width, int_width, frac_bits = calculate_scale_bits(min_scale)

    return {
        "auto_detect": True,
        "width":       total_width,
        "int":         int_width,
        "frac":        frac_bits,
        "min_scale":   min_scale,
        "max_scale":   max_scale,
    }



def quantize_tensor(tensor, bit_width=8, symmetric=True, calibration_data=None):
    """Quantize a float tensor to integers. Returns (q_tensor, scale, zero_point).
    symmetric=True for weights (zero-centered range), False for biases/activations.
    """
    if calibration_data is not None:
        min_val = min(d.min().item() for d in calibration_data)
        max_val = max(d.max().item() for d in calibration_data)
    else:
        min_val = tensor.min().item()
        max_val = tensor.max().item()

    if bit_width == 1:
        # Binary: map all values to {-1, +1}, treating 0 as +1.
        max_abs = max(abs(min_val), abs(max_val))
        scale      = max_abs if max_abs > 0 else 1.0
        zero_point = 0
        q_tensor = torch.sign(tensor).to(torch.int32)
        q_tensor[q_tensor == 0] = 1
        clamp_ratio = 0.0

    elif symmetric:
        # Signed range [-2^(b-1), 2^(b-1)-1], zero_point = 0.
        max_abs = max(abs(min_val), abs(max_val))
        scale      = max_abs / (2 ** (bit_width - 1) - 1) if max_abs > 0 else 1.0
        zero_point = 0
        qmin = -(2 ** (bit_width - 1))
        qmax =   2 ** (bit_width - 1) - 1
        q_tensor = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int32)
        clamp_ratio = ((q_tensor == qmin) | (q_tensor == qmax)).float().mean().item()

    else:
        # Unsigned range [0, 2^b - 1], zero_point shifts the origin.
        qmin = 0
        qmax = 2 ** bit_width - 1
        scale      = (max_val - min_val) / (qmax - qmin) if (max_val - min_val) > 0 else 1.0
        zero_point = int(round(qmin - min_val / scale))
        q_tensor = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.int32)
        clamp_ratio = ((q_tensor == qmin) | (q_tensor == qmax)).float().mean().item()

    if clamp_ratio > 0.01:
        print(f"[Warn] Quantization endpoint occupancy {clamp_ratio:.2%} for shape {tensor.shape} - may lose precision")

    return q_tensor, scale, zero_point


# (bn_key_prefix, weight_key_prefix, scale_view_shape)
_BN_FOLD_TARGETS = [
    ("conv_bn_layers", "conv_layers",    (-1, 1, 1, 1)),
    ("fc_bn_layers",   "fc_layers",      (-1, 1)),
    ("dw_bn_layers",   "dw_conv_layers", (-1, 1, 1, 1)),
    ("pw_bn_layers",   "pw_conv_layers", (-1, 1, 1, 1)),
]

def _fold_bn_into_weights(sd):
    """Fold BatchNorm parameters into Conv/FC weights in a state_dict.

    Operates on a plain dict of tensors loaded from a checkpoint, unlike standardize_model.py which folds BN into nn.Module objects.
    """
    bn_prefixes = {bn for bn, _, _ in _BN_FOLD_TARGETS}
    if not any(k.split(".")[0] in bn_prefixes for k in sd):
        return sd

    sd = dict(sd)  # shallow copy: do not mutate the original
    eps = 1e-5
    print("[BN-Fold] Folding BatchNorm into weights ...")

    for bn_prefix, w_prefix, view_shape in _BN_FOLD_TARGETS:
        i = 0
        while f"{bn_prefix}.{i}.weight" in sd:
            gamma   = sd.pop(f"{bn_prefix}.{i}.weight")
            beta_bn = sd.pop(f"{bn_prefix}.{i}.bias")
            mean    = sd.pop(f"{bn_prefix}.{i}.running_mean")
            var     = sd.pop(f"{bn_prefix}.{i}.running_var")
            sd.pop(f"{bn_prefix}.{i}.num_batches_tracked", None)

            scale = gamma / torch.sqrt(var + eps)

            w_key = f"{w_prefix}.{i}.weight"
            b_key = f"{w_prefix}.{i}.bias"
            W = sd[w_key]
            b = sd.get(b_key, torch.zeros(W.shape[0]))

            sd[w_key] = W * scale.view(view_shape)
            sd[b_key] = (b - mean) * scale + beta_bn

            print(f"  {bn_prefix}.{i} -> {w_prefix}.{i}  "
                  f"(scale [{scale.min():.4f}, {scale.max():.4f}])")
            i += 1

    print("[BN-Fold] Done.")
    return sd


def _quantize_and_save(tensor, out_path, row_dim, quant_bits, symmetric,
                       use_qat, qat_scales, qat_prefix, suffix, cal_data=None):
    """Quantize tensor and save as CSV. Returns quant_info dict, or None if not quantizing.

    row_dim: number of rows in the saved CSV (tensor is reshaped to [row_dim, -1]).
    """
    if not quant_bits:
        np.savetxt(out_path, tensor.numpy().reshape(row_dim, -1), delimiter=",", fmt="%.6f")
        return None

    scale_key = f"{qat_prefix}_{suffix}_scale"
    if use_qat and qat_scales and scale_key in qat_scales:
        scale = qat_scales[scale_key]
        zero_point    = qat_scales.get(f"{qat_prefix}_{suffix}_zero_point", 0)
        qmin  = -(2**(quant_bits-1)) if symmetric else 0
        qmax  =   2**(quant_bits-1)-1 if symmetric else 2**quant_bits-1
        q     = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.int32)
        print(f"  [{qat_prefix}] {suffix}: QAT scale={float(scale):.6f}")
    else:
        q, scale, zero_point = quantize_tensor(tensor, bit_width=quant_bits, symmetric=symmetric,
                                       calibration_data=cal_data)

    np.savetxt(out_path, q.numpy().reshape(row_dim, -1), delimiter=",", fmt="%d")
    return {"bit_width": quant_bits, "scale": float(scale), "zero_point": int(zero_point), "symmetric": symmetric}



def export_ir_from_state_dict(sd, timesteps, out_dir, layer_defs, input_dim, output_dim,
                            default_beta=0.9, default_threshold=1.0,
                            quant_bits=None, quant_mode="none", fixed_width=32, fixed_int=16,
                            use_qat=False, qat_scales=None,
                            input_fixed_width=16, input_fixed_int=None, encoding="repeat",
                            input_is_binary=False, training_method=None):
    """Export IR JSON + CSV weight files from a standardized state_dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    sd = _fold_bn_into_weights(sd)

    use_qat_scales = use_qat and bool(qat_scales)
    if use_qat_scales:
        print("[QAT] Using pre-trained quantization scales")

    layers = []
    if isinstance(input_dim, (list, tuple)):
        cur_c, cur_h, cur_w = int(input_dim[0]), int(input_dim[1]), int(input_dim[2])
    else:
        cur_c = cur_h = cur_w = None

    # Detect DSC pointwise layers among the Conv2d entries.
    has_pw = any(k.startswith("pw_conv_layers.") and k.endswith(".weight") for k in sd)

    conv_idx = fc_idx = dw_conv_idx = pw_conv_idx = conv_lif_idx = fc_lif_idx = 0
    prev_fc_out = None  # out_dim of previous Linear, for dim consistency check

    for ldef in layer_defs:
        ltype = ldef["type"]

        # Per-layer weight bitwidth (mixed precision); falls back to the global quant_bits, stays None in float mode. Honored on the PTQ path only; QAT uses scales trained at the global bit width.
        layer_bits = (ldef.get("bitwidth") or quant_bits) if quant_bits else quant_bits

        match ltype:

            case "DepthwiseConv2d":
                w    = sd[f"dw_conv_layers.{dw_conv_idx}.weight"]  # (C, 1, K, K)
                b    = sd.get(f"dw_conv_layers.{dw_conv_idx}.bias")
                ch   = w.shape[0]
                tag  = f"dw_conv{dw_conv_idx + 1}"
                quant_prefix = f"dw_conv_{dw_conv_idx}"

                w_file  = f"{tag}_weight.csv"
                quant_w = _quantize_and_save(w, os.path.join(out_dir, w_file), ch,
                                             layer_bits, True, use_qat_scales, qat_scales, quant_prefix, "weight")
                layer = {
                    "type":        "DepthwiseConv2d",
                    "name":        tag,
                    "channels":    int(ch),
                    "bitwidth":    layer_bits,
                    "kernel_size": int(w.shape[2]),
                    "stride":      ldef.get("stride", 1),
                    "padding":     ldef.get("padding", 0),
                    "in_h":        cur_h,
                    "in_w":        cur_w,
                    "out_h":       int(ldef["out_h"]),
                    "out_w":       int(ldef["out_w"]),
                    "weight":      w_file,
                    "bias":        None,
                }
                if quant_w:
                    layer["quant_weight"] = quant_w
                if b is not None:
                    b_file  = f"{tag}_bias.csv"
                    quant_b = _quantize_and_save(b, os.path.join(out_dir, b_file), 1,
                                                 layer_bits, False, use_qat_scales, qat_scales, quant_prefix, "bias")
                    layer["bias"] = b_file
                    if quant_b:
                        layer["quant_bias"] = quant_b

                layers.append(layer)
                cur_c, cur_h, cur_w = int(ch), int(ldef["out_h"]), int(ldef["out_w"])
                dw_conv_idx += 1

            case "Conv2d":
                lname = ldef.get("name", "")
                if lname.startswith("pw_conv") and has_pw:
                    w    = sd[f"pw_conv_layers.{pw_conv_idx}.weight"]
                    b    = sd.get(f"pw_conv_layers.{pw_conv_idx}.bias")
                    quant_prefix = f"pw_conv_{pw_conv_idx}"
                    tag  = f"pw_conv{pw_conv_idx + 1}"
                    pw_conv_idx += 1
                else:
                    w    = sd[f"conv_layers.{conv_idx}.weight"]
                    b    = sd.get(f"conv_layers.{conv_idx}.bias")
                    quant_prefix = f"conv_{conv_idx}"
                    tag  = f"conv{conv_idx + 1}"

                out_ch = w.shape[0]
                in_ch  = w.shape[1]

                w_file  = f"{tag}_weight.csv"
                quant_w = _quantize_and_save(w, os.path.join(out_dir, w_file), out_ch,
                                             layer_bits, True, use_qat_scales, qat_scales, quant_prefix, "weight")
                layer = {
                    "type":        "Conv2d",
                    "name":        tag,
                    "in_ch":       int(in_ch),
                    "out_ch":      int(out_ch),
                    "bitwidth":    layer_bits,
                    "kernel_size": int(w.shape[2]),
                    "stride":      ldef.get("stride", 1),
                    "padding":     ldef.get("padding", 0),
                    "in_h":        cur_h,
                    "in_w":        cur_w,
                    "out_h":       int(ldef["out_h"]),
                    "out_w":       int(ldef["out_w"]),
                    "weight":      w_file,
                    "bias":        None,
                }
                if quant_w:
                    layer["quant_weight"] = quant_w
                if b is not None:
                    b_file  = f"{tag}_bias.csv"
                    quant_b = _quantize_and_save(b, os.path.join(out_dir, b_file), 1,
                                                 layer_bits, False, use_qat_scales, qat_scales, quant_prefix, "bias")
                    layer["bias"] = b_file
                    if quant_b:
                        layer["quant_bias"] = quant_b

                layers.append(layer)
                cur_c, cur_h, cur_w = int(out_ch), int(ldef["out_h"]), int(ldef["out_w"])
                if not lname.startswith("pw_conv"):
                    conv_idx += 1

            case "AvgPool2d" | "MaxPool2d":
                out_h = int(ldef["out_h"])
                out_w = int(ldef["out_w"])
                layers.append({
                    "type":        ltype,
                    "name":        ldef.get("name", f"pool{conv_idx}"),
                    "kernel_size": int(ldef["kernel_size"]),
                    "stride":      int(ldef.get("stride", ldef["kernel_size"])),
                    "padding":     int(ldef.get("padding", 0)),
                    "pool_type":   ldef.get("pool_type", "avg" if ltype == "AvgPool2d" else "max"),
                    "channels":    cur_c,
                    "in_h":        cur_h,
                    "in_w":        cur_w,
                    "out_h":       out_h,
                    "out_w":       out_w,
                    "out_dim":     int(cur_c * out_h * out_w),
                })
                cur_h, cur_w = out_h, out_w

            case "Linear":
                lname = ldef["name"]
                if lname == "fc_out":
                    w    = sd["fc_out.weight"]
                    b    = sd.get("fc_out.bias")
                    quant_prefix = "fc_out"
                else:
                    w    = sd[f"fc_layers.{fc_idx}.weight"]
                    b    = sd.get(f"fc_layers.{fc_idx}.bias")
                    quant_prefix = f"fc_{fc_idx}"

                in_dim     = w.shape[1]
                out_dim_fc = w.shape[0]

                if prev_fc_out is not None and prev_fc_out != in_dim:
                    raise ValueError(
                        f"Dim mismatch: '{lname}' in_dim={in_dim}, expected {prev_fc_out} from previous layer"
                    )

                # FCN models save QAT scales under the 'layer_{n}' prefix, but export uses 'fc_{n}' / 'fc_out'. Try both when QAT scales are present.
                quant_prefix_eff = quant_prefix
                if use_qat_scales and qat_scales and f"{quant_prefix}_weight_scale" not in qat_scales:
                    alt = f"layer_{fc_idx}" if quant_prefix == "fc_out" else quant_prefix.replace("fc_", "layer_")
                    if f"{alt}_weight_scale" in qat_scales:
                        quant_prefix_eff = alt

                w_file  = f"{lname}_weight.csv"
                quant_w = _quantize_and_save(w, os.path.join(out_dir, w_file), out_dim_fc,
                                             layer_bits, True, use_qat_scales, qat_scales, quant_prefix_eff, "weight")
                layer = {
                    "type":    "Linear",
                    "name":    lname,
                    "in_dim":  int(in_dim),
                    "out_dim": int(out_dim_fc),
                    "bitwidth": layer_bits,
                    "weight":  w_file,
                    "bias":    None,
                }
                if quant_w:
                    layer["quant_weight"] = quant_w
                if b is not None:
                    b_file  = f"{lname}_bias.csv"
                    quant_b = _quantize_and_save(b, os.path.join(out_dir, b_file), 1,
                                                 layer_bits, False, use_qat_scales, qat_scales, quant_prefix_eff, "bias")
                    layer["bias"] = b_file
                    if quant_b:
                        layer["quant_bias"] = quant_b

                layers.append(layer)
                prev_fc_out = out_dim_fc
                if lname != "fc_out":
                    fc_idx += 1

            case "LIF":
                lif_name = ldef["name"]

                # Determine the state_dict key prefix for this LIF.
                if lif_name.startswith("lif_conv"):
                    lif_prefix = f"conv_lif_layers.{conv_lif_idx}"
                    conv_lif_idx += 1
                elif lif_name.startswith("lif_fc"):
                    lif_prefix = f"fc_lif_layers.{fc_lif_idx}"
                    fc_lif_idx += 1
                elif lif_name == "lif_out":
                    lif_prefix = "lif_out"
                else:
                    lif_prefix = None

                beta_key = f"{lif_prefix}.beta" if lif_prefix else None
                beta = float(sd[beta_key]) if beta_key and beta_key in sd else default_beta

                beta = min(max(beta, 0.0), 1.0)

                thr_key = f"{lif_prefix}.threshold" if lif_prefix else None
                threshold = float(sd[thr_key]) if thr_key and thr_key in sd \
                    else float(ldef.get("threshold", default_threshold))

                in_dim_lif  = ldef.get("in_dim", cur_c * cur_h * cur_w if cur_c is not None else 0)
                out_dim_lif = ldef.get("out_dim", in_dim_lif)

                if lif_prefix and f"{lif_prefix}.a" in sd:
                    neuron_model = "PLIF"
                elif abs(beta - 1.0) < 1e-3:
                    neuron_model = "IF"
                else:
                    neuron_model = "LIF"

                lif_entry = {
                    "type":             "LIF",
                    "neuron_model":     neuron_model,
                    "name":             lif_name,
                    "beta":             float(beta),
                    "threshold":        float(threshold),
                    "reset_mechanism":  ldef.get("reset_mechanism", "subtract"),
                    "in_dim":           int(in_dim_lif),
                    "out_dim":          int(out_dim_lif),
                }

                layers.append(lif_entry)

    # Assemble IR.
    resolved_int = input_fixed_int if input_fixed_int is not None else 8
    input_type   = {"type_class": "fixed", "width": input_fixed_width, "int": resolved_int}

    acc_type    = derive_type_config(0, timesteps, is_integer=True, is_signed=False)
    # SNN output is spike count / T, always in [0, 1]: keep fixed 16-bit to match converter.py contract.
    output_type = {"type_class": "ufixed", "width": 16, "int": 2}

    has_conv = any(ld["type"] in ("Conv2d", "DepthwiseConv2d") for ld in layer_defs)

    ir = {
        "model":               "csnn" if has_conv else "fcsnn",
        "timesteps":           timesteps,
        "input_dim":           list(input_dim) if isinstance(input_dim, (list, tuple)) else input_dim,
        "output_dim":          output_dim,
        "layers":              layers,
        "quant_mode":          quant_mode,
        "training_method":     training_method or ("QAT" if use_qat else ("PTQ" if quant_bits else "FP32")),
        "fixed_config":        {"use_fixed": quant_mode == "int8_fixed", "width": fixed_width, "int": fixed_int},
        "scale_config":        analyze_scales_for_ir(layers),
        "input_fixed_config":  input_type,
        "output_fixed_config": output_type,
        "acc_fixed_config":    acc_type,
        "encoding":            encoding,
        "input_is_binary":     bool(input_is_binary),
        # Consolidated data-identity view (time axis + value axis + window)
        "data_identity": {
            "encoding":        encoding,
            "value_coding":    "spike" if input_is_binary else "count",
            "input_is_binary": bool(input_is_binary),
            "timesteps":       timesteps,
            "window":          None,   
            "bin_factor":      None,   
        },
    }

    with open(os.path.join(out_dir, "ir.json"), "w") as f:
        json.dump(ir, f, indent=2)

    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Model: {ir['model']}\n")
        f.write(f"Timesteps: {ir['timesteps']}\n")
        f.write(f"Input dim: {ir['input_dim']}\n")
        f.write(f"Output dim: {ir['output_dim']}\n")
        f.write(f"Layers ({len(ir['layers'])}):\n")
        for i, layer in enumerate(ir["layers"]):
            f.write(f"  [{i}] {layer['type']}  {layer.get('name', '')}\n")

    print(f"[Done] IR exported to {out_dir}")
    if quant_bits:
        print(f"[Done] PTQ {quant_bits}-bit quantization applied")
    return ir


def export_ir_from_pth(pth_path, timesteps, out_dir,
                        default_beta=0.9, default_threshold=1.0,
                        quant_bits=None, quant_mode="none", fixed_width=32, fixed_int=16,
                        use_qat=False,
                        input_fixed_width=16, input_fixed_int=None, encoding_override=None,
                        input_is_binary_override=None):
    """Load a standardized .pth checkpoint and call export_ir_from_state_dict."""
    checkpoint = torch.load(pth_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd            = checkpoint["state_dict"]
        training_meta = checkpoint.get("training_meta", {})
        qat_scales    = checkpoint.get("qat_scales", None)
        layer_defs    = checkpoint.get("layer_defs", None)
        encoding      = checkpoint.get("encoding", "repeat")
        input_is_binary = bool(checkpoint.get("input_is_binary", False))
    else:  # raw state_dict, no metadata
        sd            = checkpoint
        training_meta = {}
        qat_scales    = None
        layer_defs    = None
        input_is_binary = False
        encoding      = "repeat"

    if encoding_override is not None:
        encoding = encoding_override


    if input_is_binary_override is not None:
        input_is_binary = bool(input_is_binary_override)
        print(f"[Info] input_is_binary overridden to {input_is_binary} (explicit spike/count declaration)")
    elif not input_is_binary and encoding == "rate":
        input_is_binary = True
        print("[Info] input_is_binary derived True from encoding='rate' (legacy checkpoint, no field)")

    if layer_defs is None:
        raise ValueError(
            "Checkpoint does not contain 'layer_defs'. "
            "Run standardize_model.py first to produce a standardized checkpoint."
        )

    # Recover input_dim from layer_defs and weight shapes.
    has_conv = any(ld["type"] in ("Conv2d", "DepthwiseConv2d") for ld in layer_defs)
    if has_conv:
        first_conv = next(ld for ld in layer_defs if ld["type"] in ("Conv2d", "DepthwiseConv2d"))
        if first_conv["type"] == "DepthwiseConv2d":
            in_ch = sd["dw_conv_layers.0.weight"].shape[0]
        else:
            in_ch = sd["conv_layers.0.weight"].shape[1]
        k    = first_conv.get("kernel_size", 3)
        s    = first_conv.get("stride", 1)
        p    = first_conv.get("padding", 0)
        if "in_h" in first_conv and "in_w" in first_conv:

            in_h = int(first_conv["in_h"])
            in_w = int(first_conv["in_w"])
        else:
            # reverse of: out = (in + 2*p - k) // s + 1
            in_h = (first_conv["out_h"] - 1) * s + k - 2 * p
            in_w = (first_conv["out_w"] - 1) * s + k - 2 * p
        input_dim = (int(in_ch), int(in_h), int(in_w))
    else:  # FCN: flat input
        first_linear = next(ld for ld in layer_defs if ld["type"] == "Linear")
        input_dim = int(first_linear["in"])

    last_linear = [ld for ld in layer_defs if ld["type"] == "Linear"][-1]
    output_dim  = int(last_linear["out"])

    # Auto-detect QAT from training_meta
    if not use_qat and training_meta.get("training_method") in ("QAT", "QAT-FT"):
        use_qat = True
        print(f"[Info] Auto-detected QAT from checkpoint metadata")

    if qat_scales:
        print(f"[Info] Found {len(qat_scales)} QAT scales in checkpoint")

    ir = export_ir_from_state_dict(
        sd,
        timesteps=timesteps,
        out_dir=out_dir,
        layer_defs=layer_defs,
        input_dim=input_dim,
        output_dim=output_dim,
        default_beta=default_beta,
        default_threshold=default_threshold,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
        fixed_width=fixed_width,
        fixed_int=fixed_int,
        use_qat=use_qat,
        qat_scales=qat_scales,
        input_fixed_width=input_fixed_width,
        input_fixed_int=input_fixed_int,
        training_method=training_meta.get("training_method") if training_meta else None,
        encoding=encoding,
        input_is_binary=input_is_binary,
    )

    if training_meta:
        ir["training_meta"] = {
            "epochs":          training_meta.get("epochs"),
            "pretrain_epochs": training_meta.get("pretrain_epochs"),
            "final_train_acc": training_meta.get("final_train_acc"),
            "final_test_acc":  training_meta.get("final_test_acc"),
            "best_test_acc":   training_meta.get("best_test_acc"),
            "best_test_epoch": training_meta.get("best_test_epoch"),
            "pretrained_acc":  training_meta.get("pretrained_acc"),
            "weight_decay":    training_meta.get("weight_decay"),
            "lr":              training_meta.get("lr"),
            "batch_size":      training_meta.get("batch_size"),
        }
        with open(os.path.join(out_dir, "ir.json"), "w") as f:
            json.dump(ir, f, indent=2)
        best = training_meta.get("best_test_acc")
        if best:
            print(f"[Info] Training metadata: epochs={training_meta.get('epochs')}, best_acc={best:.2f}%")

    return ir


def main():
    parser = argparse.ArgumentParser(description="Export SNN IR from a standardized .pth checkpoint")
    parser.add_argument("pth_path",
                        help="Standardized checkpoint produced by standardize_model.py")
    parser.add_argument("--timesteps",           type=int, required=True,
                        help="Number of simulation timesteps")
    parser.add_argument("--out-dir",             default=None,
                        help="Output directory (default: ir_output/<stem>)")
    parser.add_argument("--quant-bits",          type=int, default=None,
                        help="PTQ bit width (e.g. 8); omit for float export")
    parser.add_argument("--quant-mode",          default="none",
                        choices=["none", "int8_float", "int8_fixed"],
                        help="none: float  int8_float: int8 weights + float accumulation  int8_fixed: full fixed-point")
    parser.add_argument("--fixed-width",         type=int, default=16,
                        help="HLS ap_fixed total bit width (default 16)")
    parser.add_argument("--fixed-int",           type=int, default=8,
                        help="HLS ap_fixed integer bits (default 8)")
    parser.add_argument("--qat",                 action="store_true",
                        help="Use QAT scales stored in the checkpoint")
    parser.add_argument("--input-fixed-width",   type=int, default=16,
                        help="Input fixed-point total bits (default 16)")
    parser.add_argument("--input-fixed-int",     type=int, default=None,
                        help="Input fixed-point integer bits (default: auto)")
    parser.add_argument("--encoding",            default=None,
                        choices=["repeat", "rate", "temporal"],
                        help="Override input encoding (repeat: static image  rate: spike  temporal: DVS)")
    parser.add_argument("--dataset-kind",        default=None,
                        help="Dataset kind (mnist|nmnist|cifar10dvs|dvsgesture); written into ir.json for SD card path generation")
    parser.add_argument("--input-is-binary",     default="auto",
                        choices=["auto", "true", "false"],
                        help="Input value type: binary/spike (add-only first-layer MAC, fabric-safe) vs "
                             "count/graded. auto: use the checkpoint's data-detected field (rate encoding "
                             "→ binary). true/false: force it; pass true when the training toml declared a "
                             "spike/binarized input but the checkpoint predates the input_is_binary field.")
    args = parser.parse_args()

    if not os.path.isfile(args.pth_path):
        raise SystemExit(f"Error: checkpoint not found: {args.pth_path}. "
                         f"Run standardize_model.py or train via configs/ first.")

    out_dir = args.out_dir
    if out_dir is None:
        stem    = os.path.splitext(os.path.basename(args.pth_path))[0]
        out_dir = os.path.join("ir_output", stem)

    quant_bits = args.quant_bits
    if args.quant_mode in ("int8_float", "int8_fixed") and quant_bits is None:
        quant_bits = 8


    encoding_override = args.encoding
    if encoding_override is None and args.dataset_kind:
        try:
            from frontend.datasets import default_encoding as _ds_default_encoding
        except ImportError:
            _ds_default_encoding = lambda _kind: None
        declared = _ds_default_encoding(args.dataset_kind)
        if declared:
            encoding_override = declared
            print(f"[Info] encoding taken from the '{args.dataset_kind}' dataset module: "
                  f"'{declared}' (pass --encoding to override)")
        else:
            print(f"[Warn] no encoding given and dataset '{args.dataset_kind}' declares none; "
                  f"falling back to the checkpoint's own field, then to 'repeat'. Pass "
                  f"--encoding if the deployed accuracy looks random.")

    export_ir_from_pth(
        args.pth_path,
        timesteps=args.timesteps,
        out_dir=out_dir,
        quant_bits=quant_bits,
        quant_mode=args.quant_mode,
        fixed_width=args.fixed_width,
        fixed_int=args.fixed_int,
        use_qat=args.qat,
        input_fixed_width=args.input_fixed_width,
        input_fixed_int=args.input_fixed_int,
        encoding_override=encoding_override,
        input_is_binary_override={"auto": None, "true": True, "false": False}[args.input_is_binary],
    )

    if args.dataset_kind:
        ir_json_path = os.path.join(out_dir, "ir.json")
        if os.path.exists(ir_json_path):
            import json as _json
            with open(ir_json_path, "r") as f:
                ir_data = _json.load(f)
            if not ir_data.get("dataset_kind"):
                ir_data["dataset_kind"] = args.dataset_kind.lower()
                with open(ir_json_path, "w") as f:
                    _json.dump(ir_data, f, indent=2)
                print(f"[Info] Injected dataset_kind='{args.dataset_kind}' into IR")


if __name__ == "__main__":
    main()
