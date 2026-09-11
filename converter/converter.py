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
import glob as _glob
import jinja2
import numpy as np
import argparse
from datetime import datetime

from hls_types import (hls_typedef, gcc_typedef, compute_mac_acc_width,
                       compute_lif_mem_width_profiled)
from ir_checks import sanity_check_ir, validate_ir_schema
from ir_utils import compute_layer_dims, compute_pair_indices, compute_stages
from emit import (emit_param_headers, render_template,
                  _sd_data_dir, _sd_num_samples, _sd_samples_per_file)
from emit_streaming import emit_streaming_stages

TEMPLATE_STAGE_DIR = os.path.join(os.path.dirname(__file__), "templates_stage")
TEMPLATE_BAMBU_DIR = os.path.join(os.path.dirname(__file__), "templates_bambu")
# Output root for generated projects. Override with SNN2B_OUTPUT_DIR to keep large batch
# sweeps out of backend_projects, which gets slow to load when it fills up.
BASE_OUTPUT_DIR = os.environ.get("SNN2B_OUTPUT_DIR", "backend_projects")


def _beta_to_shift(beta):
    """Snap a LIF leak beta to a bit-shift form: beta*mem -> mem - (mem>>k) = mem*(1-2^-k).
    Returns k in 1..8, or None when beta is within half an LSB of 1 (no leak: m = mem).
    Lets the LIF leak run with a shift-add instead of a DSP multiply (requires the model
    to be (re)trained with beta near a 1-2^-k value to keep accuracy)."""
    beta = float(beta)
    if beta >= 0.998:
        return None
    return min(range(1, 9), key=lambda k: abs(beta - (1.0 - 2.0 ** -k)))


def convert_model(ir_path, use_fixed=False, use_pragma=False,
                  parallel_factor=8, opt_tags=None,
                  project_name=None, use_sparse=False,
                  streaming=False,
                  conv_oc_factor="0", conv_oc_max=16,
                  per_layer_scale=True, per_layer_acc=True, per_layer_lif_mem=True,
                  lif_mem_profile=None, lif_mem_slack=2.0,
                  backend="vitis", bambu_opt=None, bambu_extra=None,
                  fold_dequant=False, bit_shift_beta=False, input_is_binary=False,
                  data_width=None, data_int=None, mul_impl_fabric=False,
                  sd_encoding=None):
    # Bambu emits float, plain C++: no ap_fixed, no HLS pragmas
    if backend == "bambu":
        use_fixed = False
        use_pragma = False

    # Resolve opt_tags
    VALID_OPT_TAGS   = {"conv_kernel", "conv_ic", "dataflow"}
    OPT_TAG_REQUIRES = {"conv_ic": ["conv_kernel"]}

    # Normalize conv_oc_factor: "auto" stays; numeric string -> int; None -> 0
    if conv_oc_factor is None:
        conv_oc_factor = 0
    elif isinstance(conv_oc_factor, str):
        s = conv_oc_factor.strip().lower()
        if s == "auto":
            conv_oc_factor = "auto"
        else:
            try:
                conv_oc_factor = int(s)
            except ValueError:
                print(f"[Warn] Invalid --conv-oc-factor '{conv_oc_factor}', falling back to 0")
                conv_oc_factor = 0

    opt_tags = set(opt_tags or [])
    unknown  = opt_tags - VALID_OPT_TAGS
    if unknown:
        print(f"[Warn] Unknown opt tags ignored: {unknown}")
        opt_tags -= unknown
    for tag, deps in OPT_TAG_REQUIRES.items():
        if tag in opt_tags:
            for dep in deps:
                if dep not in opt_tags:
                    print(f"[Info] Opt tag '{tag}' requires '{dep}', adding automatically")
                    opt_tags.add(dep)

    # Load IR
    with open(ir_path) as f:
        ir = json.load(f)
    validate_ir_schema(ir)   # fatal on malformed structure, warns on out-of-range opt/identity
    compute_layer_dims(ir)
    compute_pair_indices(ir["layers"])

    # Read-back
    _ir_opt = ir.get("optimization", {})
    if not opt_tags and _ir_opt.get("opt"):
        opt_tags = set(_ir_opt["opt"])
    if data_width is None and _ir_opt.get("data_width"):
        data_width = _ir_opt["data_width"]
        if data_int is None:
            data_int = _ir_opt.get("data_int")

    # Promote any per-layer optimization onto the layer dicts and lock those keys
    for layer in ir["layers"]:
        locked = layer.pop("opt", None)
        if locked:
            layer.update(locked)
            layer["_opt_locked"] = set(locked.keys())

    sanity_check_ir(ir, os.path.dirname(ir_path))

    # Fixed-point / type resolution
    if use_fixed:
        fixed_config = ir.get("fixed_config", {"width": 32, "int": 16})
        fixed_config["use_fixed"] = True
        if data_width is not None:
            # Explicit data_t width override, bypassing the legacy upgrade below.
            # AP_SAT clamps on overflow instead of wrapping, so a too-narrow width
            # costs accuracy through saturation rather than a catastrophic wrap; verify with csim.
            fixed_config["width"] = data_width
            fixed_config["int"] = data_int if data_int is not None else data_width // 2
            print(f"[Info] data_t width overridden to <{fixed_config['width']}, {fixed_config['int']}>")
        # Widen legacy data_t configurations to reduce saturation risk.
        elif fixed_config.get("width", 16) < 32:
            print(f"[Info] Upgrading legacy fixed_config <{fixed_config['width']}, {fixed_config['int']}> -> <32, 16> to reduce saturation risk")
            fixed_config["width"] = 32
            fixed_config["int"] = 16
    else:
        fixed_config = {"use_fixed": False, "width": 32, "int": 16}

    input_fixed_config  = ir.get("input_fixed_config",  {"width": 16, "int": 8})
    output_fixed_config = ir.get("output_fixed_config", {"width": 16, "int": 2})
    acc_fixed_config    = ir.get("acc_fixed_config", None)

    input_fixed_config.setdefault("type_class",  "fixed")
    output_fixed_config.setdefault("type_class", "fixed")

    if acc_fixed_config is None:
        ts = ir.get("timesteps", 10)
        w  = max(((max(math.ceil(math.log2(ts + 1)), 1) + 3) // 4) * 4, 4)
        acc_fixed_config = {"type_class": "uint", "width": w, "int": w}
    else:
        acc_fixed_config.setdefault("type_class", "uint")

    input_hls_type  = hls_typedef(input_fixed_config)
    input_gcc_type  = gcc_typedef(input_fixed_config)
    output_hls_type = hls_typedef(output_fixed_config)
    output_gcc_type = gcc_typedef(output_fixed_config)
    acc_hls_type    = hls_typedef(acc_fixed_config)
    acc_gcc_type    = gcc_typedef(acc_fixed_config)

    quant_mode   = ir.get("quant_mode", "none")
    scale_config = ir.get("scale_config", {
        "auto_detect": False, "width": 32, "int": 8, "frac": 24,
        "min_scale": None, "max_scale": None,
    })

    # Scan all weight and bias scales; widen scale_config if its frac precision cannot represent the smallest scale.
    if use_fixed:
        actual_scales = []
        for L in ir["layers"]:
            qw = L.get("quant_weight") or {}
            qb = L.get("quant_bias")   or {}
            for k in ("scale",):
                v = qw.get(k)
                if v is not None and v > 0:
                    actual_scales.append(float(v))
                v = qb.get(k)
                if v is not None and v > 0:
                    actual_scales.append(float(v))
        if actual_scales:
            true_min = min(actual_scales)
            true_max = max(actual_scales)
            sc_frac  = scale_config.get("frac", scale_config["width"] - scale_config["int"])
            sc_prec  = 2.0 ** (-sc_frac)
            # Widen if scale_t cannot resolve the smallest scale to within ~4 LSBs, else it rounds toward 0.
            if true_min < sc_prec * 4:
                needed_frac = math.ceil(-math.log2(true_min)) + 2  # keep a couple bits below its MSB
                needed_frac = max(needed_frac, sc_frac)
                needed_frac = ((needed_frac + 7) // 8) * 8  # round to multiple of 8
                needed_frac = min(needed_frac, 32)           # cap at 32
                needed_int  = max(scale_config["int"], math.ceil(math.log2(max(true_max, 1.0)) + 2))
                needed_int  = ((needed_int + 7) // 8) * 8
                new_width   = needed_int + needed_frac
                if new_width > scale_config["width"]:
                    print(f"[Info] Widening scale_config: <{scale_config['width']}, {scale_config['int']}> "
                          f"-> <{new_width}, {needed_int}> (true_min_scale={true_min:.2e}, "
                          f"old precision {sc_prec:.2e} would quantize small scales to 0)")
                    scale_config = dict(scale_config)
                    scale_config["width"] = new_width
                    scale_config["int"]   = needed_int
                    scale_config["frac"]  = needed_frac
                    scale_config["min_scale"] = true_min
                    scale_config["max_scale"] = true_max

    # Config summary
    config_name = "S" + ("P" if use_pragma else "") + ("Q" if use_fixed else "")
    print(f"[Info] Configuration: {config_name}")
    print(f"[Info] Fixed-point: {use_fixed}  Pragma: {use_pragma}  QMode: {quant_mode}")
    print(f"[Info] Parallel factor: {parallel_factor}")
    if opt_tags:
        print(f"[Info] Opt tags: {', '.join(sorted(opt_tags))}")
    if use_sparse:
        print("[Info] Sparse FC: enabled (skip zero-spike MACs)")
    if conv_oc_factor == "auto":
        print(f"[Info] Conv OC unroll: auto (per-layer; OC<={conv_oc_max} -> full unroll + lane-sep + block partition)")
    elif isinstance(conv_oc_factor, int) and conv_oc_factor > 0:
        print(f"[Info] Conv OC unroll: forced factor={conv_oc_factor} (per layer, capped at OC)")
    else:
        print("[Info] Conv OC unroll: disabled (legacy code path)")
    if use_fixed:
        print(f"[Info] Weight type: ap_fixed<{fixed_config['width']}, {fixed_config['int']}>")
        print(f"[Info] Input:  HLS={input_hls_type}  GCC={input_gcc_type}")
        print(f"[Info] Output: HLS={output_hls_type}  GCC={output_gcc_type}")
        print(f"[Info] Acc:    HLS={acc_hls_type}  GCC={acc_gcc_type}")
    if scale_config["auto_detect"]:
        print(f"[Info] Scale: ap_fixed<{scale_config['width']}, {scale_config['int']}> "
              f"(min={scale_config['min_scale']:.2e}, frac={scale_config['frac']})")

    # Output directory
    if project_name:
        subdir = project_name
    elif "dataset" in ir and "model" in ir:
        subdir = f"{ir['dataset'].lower()}_{ir['model'].lower()}"
    else:
        subdir = os.path.basename(os.path.dirname(ir_path))
    # One folder per project: cpp holds the code, xilinx/bambu holds the build, plus build_information.txt.
    project_dir = os.path.join(BASE_OUTPUT_DIR, subdir)
    output_dir = os.path.join(project_dir, "cpp")
    os.makedirs(output_dir, exist_ok=True)

    # clean
    for pattern in ["fc_layer*", "neuron_layer*", "conv_layer*", "dw_conv_layer*",
                    "pool_layer*", "model.cpp", "model.h", "reset_node.*",
                    "weights*.h", "biases*.h"]:
        for stale in _glob.glob(os.path.join(output_dir, pattern)):
            os.remove(stale)

    emit_param_headers(ir_path, ir, output_dir, fixed_config)

    # LIF parameters
    lif_betas = [float(L.get("beta",      0.9)) for L in ir["layers"] if L["type"] == "LIF"]

    # Fold dequant scale into the threshold
    # Restricted to Linear->LIF 
    if fold_dequant:
        last_scale = None
        last_wtype = None
        for L in ir["layers"]:
            qw = L.get("quant_weight")
            if isinstance(qw, dict) and qw.get("scale"):
                last_scale = float(qw["scale"])
                last_wtype = L["type"]
            if L["type"] == "LIF" and last_scale and last_wtype == "Linear":
                L["threshold"] = float(L.get("threshold", 1.0)) / last_scale

    lif_vths  = [float(L.get("threshold", 1.0)) for L in ir["layers"] if L["type"] == "LIF"]

    # Per-LIF mem width defaults to wide (lif_mem_t = data_t). Narrowing happens only when a measured
    # profile is supplied via --lif-mem-profile; --no-per-layer-lif-mem forces wide even with a profile.
    lif_mem_narrowed = use_fixed and per_layer_lif_mem and lif_mem_profile is not None
    if lif_mem_narrowed:
        # Profile JSON is keyed by snnTorch layer name in LIF order, so map positionally.
        lif_layers_in_order = [L for L in ir["layers"] if L["type"] == "LIF"]
        prof_names = list(lif_mem_profile.keys())
        if len(prof_names) != len(lif_layers_in_order):
            print(f"[Warn] LIF profile has {len(prof_names)} entries but IR has "
                  f"{len(lif_layers_in_order)} LIF layers; keeping all LIF mem wide (data_t)")
            lif_mem_narrowed = False
    if lif_mem_narrowed:
        for idx, L in enumerate(lif_layers_in_order):
            stats = lif_mem_profile[prof_names[idx]]
            w, i_, narrow = compute_lif_mem_width_profiled(
                mem_max_abs=stats["max_abs"],
                data_t_width=fixed_config["width"],
                data_t_int=fixed_config["int"],
                slack=lif_mem_slack,
            )
            L["lif_mem_width"]  = w
            L["lif_mem_int"]    = i_
            L["lif_mem_narrow"] = narrow
            print(f"[LIF-MEM] L{L.get('global_idx','?')} profile({prof_names[idx]} "
                  f"max={stats['max_abs']:.2f} x {lif_mem_slack}) -> width={w} int={i_} narrow={narrow}")

    # Build parallel array indexed by lif_order_idx for streaming code path
    per_lif_mem_widths = None
    if lif_mem_narrowed:
        per_lif_mem_widths = []
        for L in ir["layers"]:
            if L["type"] != "LIF":
                continue
            per_lif_mem_widths.append((
                L.get("lif_mem_width", fixed_config["width"]),
                L.get("lif_mem_int",   fixed_config["int"]),
                L.get("lif_mem_narrow", False),
            ))

    # Jinja2 environment
    template_dir = TEMPLATE_BAMBU_DIR if backend == "bambu" else TEMPLATE_STAGE_DIR
    print(f"[Info] Templates: {template_dir}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True, lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    # Stage + spike-input computation
    pool_after_lif = ir.get("pool_after_lif", False)
    stages         = compute_stages(ir["layers"], pool_after_lif=pool_after_lif)

    encoding = ir.get("encoding") or "rate"
    if not ir.get("encoding"):
        print(f"[Warn] IR missing 'encoding' - defaulting to '{encoding}'.")

    # Whether the first layer's input is binary {0,1}: if so it can skip the input
    # multiply (spike-driven add), like the hidden layers. 
    input_is_binary = input_is_binary or bool(ir.get("input_is_binary")) or (encoding == "rate")

    # Safety gate for fabric: 'config_op mul -impl fabric' forces every multiply to LUT. 
    # Disable fabric unless the input is binary.
    if mul_impl_fabric and not input_is_binary:
        print("[Warn] --mul-impl-fabric ignored: input is not binary/spike, so the first layer "
              "does real W*in multiplies that would explode in LUT fabric. Fabric only helps "
              "spike-input designs (where the MAC is add-only).")
        mul_impl_fabric = False

    spike_input_layers: set[int] = set()
    for stage in stages:
        stage_idx = stage["stage_idx"]
        first_non_lif = stage["non_lif_layers"][0] if stage["non_lif_layers"] else None
        if stage_idx == 0:
            if input_is_binary and first_non_lif:
                spike_input_layers.add(first_non_lif["global_idx"])
            stage["input_is_spike"] = bool(input_is_binary)
            stage["input_is_narrow_data"] = False
        else:
            prev = stages[stage_idx - 1]
            if prev.get("output_is_spike", True) and first_non_lif:
                spike_input_layers.add(first_non_lif["global_idx"])
            stage["input_is_spike"] = prev.get("output_is_spike", True)
            # If the previous stage output is narrow (AvgPool yields [0,1]), this stage reads narrow.
            stage["input_is_narrow_data"] = prev.get("output_is_narrow_data", False)

    if pool_after_lif:
        for stage in stages:
            lif_seen = False
            for layer in stage["all_layers"]:
                if layer["type"] == "LIF":
                    lif_seen = True
                elif lif_seen and layer["type"] in ("AvgPool2d", "MaxPool2d"):
                    # Only the first pool right after a LIF consumes spikes. Further pools in a
                    # chain consume the (narrow) pooled data, so they must not be marked here.
                    spike_input_layers.add(layer["global_idx"])
                    lif_seen = False
                elif layer["type"] in ("Conv2d", "DepthwiseConv2d", "Linear"):
                    lif_seen = False

    # Propagate spike/narrow flow through each stage so mid-stage layers get the right input type.
    # MaxPool keeps spikes binary; AvgPool of spikes yields narrow data [0,1]; Conv/Linear consume it then output wide data_t.
    if use_fixed:
        for stage in stages:
            cur_is_spike = stage.get("input_is_spike", False)
            cur_is_narrow = stage.get("input_is_narrow_data", False)
            for layer in stage["all_layers"]:
                layer_type = layer["type"]
                global_idx = layer["global_idx"]
                if layer_type in ("Conv2d", "DepthwiseConv2d", "Linear"):
                    if cur_is_spike:
                        spike_input_layers.add(global_idx)
                    if cur_is_narrow:
                        layer["input_is_narrow_data"] = True
                    cur_is_spike = False
                    cur_is_narrow = False  # compute-layer output is wide data_t
                elif layer_type == "LIF":
                    cur_is_spike = True
                    cur_is_narrow = False
                elif layer_type == "MaxPool2d":
                    if cur_is_spike:
                        spike_input_layers.add(global_idx)   # spike in -> spike out
                    elif cur_is_narrow:
                        # Max of narrow [0,1] values stays narrow.
                        layer["input_is_narrow_data"] = True
                        layer["output_is_narrow"] = True
                elif layer_type == "AvgPool2d":
                    if cur_is_spike:
                        # AvgPool of spikes produces narrow data in [0,1].
                        spike_input_layers.add(global_idx)
                        layer["output_is_narrow"] = True
                        cur_is_narrow = True
                    elif cur_is_narrow:
                        # AvgPool of narrow [0,1] stays narrow.
                        layer["input_is_narrow_data"] = True
                        layer["output_is_narrow"] = True
                    cur_is_spike = False

    # Minimum elements per ARRAY_PARTITION bank for LUTRAM mapping. Below this threshold HLS maps to registers plus MUX, causing LUT explosion.
    MIN_BANK_DEPTH = 64

    layer_sources, layer_headers = [], []

    # Compute per-Conv2d oc_factor in place, read by both the stage and streaming templates.
    # 0 = no OC unroll; OC = full unroll with lane-separated accumulators. Partial unroll falls back to 0 with a warning.
    # Gated on the conv_kernel opt_tag so the baseline keeps the default pragmas and conv_kernel opts in for OC <= conv_oc_max layers.
    conv_kernel_on = "conv_kernel" in opt_tags
    for layer in ir["layers"]:
        layer.setdefault("oc_factor", 0)
        if "oc_factor" in layer.get("_opt_locked", ()):
            continue  # honor IR-provided oc_factor
        if not use_pragma or not conv_kernel_on or layer["type"] != "Conv2d":
            continue
        oc = int(layer.get("out_ch", 0))
        if oc <= 0:
            continue
        if conv_oc_factor == "auto":
            chosen = oc if oc <= conv_oc_max else 0
        elif isinstance(conv_oc_factor, int) and conv_oc_factor > 0:
            if conv_oc_factor >= oc:
                chosen = oc
            else:
                print(f"[Warn] Conv layer with OC={oc} requested OC unroll factor={conv_oc_factor}; "
                      f"partial OC unroll not yet supported, disabling for this layer")
                chosen = 0
        else:
            chosen = 0
        layer["oc_factor"] = chosen
        if chosen > 0:
            print(f"[Info]   Conv layer OC={oc}: full OC unroll enabled (oc_factor={chosen})")

    for i, layer in enumerate(ir["layers"]):
        ltype = layer["type"]

        conv_partition_factor = 0
        if use_pragma and "conv_kernel" in opt_tags and ltype in ("Conv2d", "DepthwiseConv2d"):
            if ltype == "Conv2d":
                ic, local_in_ch = layer.get("in_ch", 1), layer.get("in_ch", 1)
            else:
                ic, local_in_ch = 1, layer.get("channels", 1)
            ih, iw, k  = layer.get("in_h", 1), layer.get("in_w", 1), layer.get("kernel_size", 1)
            local_in_sz = local_in_ch * ih * iw
            max_safe    = max(1, local_in_sz // MIN_BANK_DEPTH)
            desired     = (ic * k * k) if ("conv_ic" in opt_tags and ltype == "Conv2d") else (k * k)
            conv_partition_factor = min(desired, max_safe)
            if conv_partition_factor < 2:
                conv_partition_factor = 0
            if 0 < conv_partition_factor < desired:
                ch_label = f"IC={ic}" if ltype == "Conv2d" else f"C={local_in_ch}"
                print(f"[Info] Layer {i} ({ltype} {ch_label} {ih}x{iw} K={k}): "
                      f"partition {desired} -> {conv_partition_factor} "
                      f"(capped, {local_in_sz // max(conv_partition_factor, 1)} elem/bank)")

        # Per-layer scale_t sizing: each layer's own scale magnitudes drive precision.
        # Disabled via --no-per-layer-scale, which falls back to the global scale_config.
        if per_layer_scale and ltype in ("Linear", "Conv2d", "DepthwiseConv2d"):
            qw = layer.get("quant_weight") or {}
            qb = layer.get("quant_bias")   or {}
            own_scales = [s for s in (qw.get("scale"), qb.get("scale")) if s and s > 0]
            if own_scales and "scale_width" not in layer.get("_opt_locked", ()):
                own_min = min(own_scales)
                own_max = max(own_scales)
                # frac bits: enough precision for the smallest scale, with a safety margin.
                needed_frac = max(8, math.ceil(-math.log2(own_min)) + 2) if own_min < 1.0 else 8
                needed_frac = ((needed_frac + 7) // 8) * 8
                needed_frac = min(needed_frac, 32)
                needed_int  = max(8, math.ceil(math.log2(max(own_max, 1.0)) + 2))
                needed_int  = ((needed_int + 7) // 8) * 8
                layer["scale_width"] = needed_int + needed_frac
                layer["scale_int"]   = needed_int

        # Per-layer acc_t sizing: tight MAC width instead of the blunt width*2 formula.
        # Disabled via --no-per-layer-acc, which falls back to width*2, int*2.
        if (per_layer_acc and ltype in ("Linear", "Conv2d", "DepthwiseConv2d")
                and "acc_width" not in layer.get("_opt_locked", ())):
            qw = layer.get("quant_weight") or {}
            w_int_bits = qw.get("bit_width", fixed_config.get("int", 8))
            # Input int bits: 1 for spike or narrow input, otherwise data_t.int.
            if i in spike_input_layers:
                in_int_bits = 1
            elif layer.get("input_is_narrow_data"):
                in_int_bits = 1
            else:
                in_int_bits = fixed_config.get("int", 16)
            # N_MAC per output
            if ltype == "Linear":
                n_mac = layer.get("in_features", layer.get("prev_out_dim", 1))
            elif ltype == "Conv2d":
                # IR uses 'in_ch'; older snapshots may use 'ic' or 'in_channels'.
                ic = layer.get("in_ch", layer.get("ic", layer.get("in_channels", 1)))
                k  = layer.get("kernel_size", 1)
                n_mac = ic * k * k
            else:  # DepthwiseConv2d
                k  = layer.get("kernel_size", 1)
                n_mac = k * k
            data_t_frac = fixed_config.get("width", 32) - fixed_config.get("int", 16)
            layer["acc_width"], layer["acc_int"] = compute_mac_acc_width(
                n_mac, w_int_bits, in_int_bits, data_t_frac
            )

        # Per-AvgPool acc_t: just sums K*K data_t values, so int bits = data_t int + log2(K*K).
        # Tighter than width*2 since a pool never needs the full MAC headroom.
        if (per_layer_acc and ltype == "AvgPool2d"
                and "acc_width" not in layer.get("_opt_locked", ())):
            kk = max(layer.get("kernel_size", 1) ** 2, 1)
            data_t_int  = fixed_config.get("int", 16)
            data_t_frac = fixed_config.get("width", 32) - data_t_int
            acc_int   = data_t_int + (math.ceil(math.log2(kk)) if kk > 1 else 0)
            acc_width = ((acc_int + data_t_frac + 7) // 8) * 8   # round to multiple of 8
            layer["acc_width"], layer["acc_int"] = acc_width, acc_width - data_t_frac

        # Pre-compute bias_dequant[oc] = (bias_int - bias_zero_point) * bias_scale in Python double.
        # Avoids precision loss when the (acc_t)bias_scale cast lands in an acc_t whose LSB exceeds the tiny physical scale at bw>=16.
        bias_dequant_table = None
        qb_now = layer.get("quant_bias")
        bias_loaded = layer.get("_bias_loaded")
        if qb_now and bias_loaded is not None:
            bias_scale_py = float(qb_now["scale"])
            bias_zero_point_py    = int(qb_now.get("zero_point", 0))
            bias_dequant_table = [float((int(b) - bias_zero_point_py) * bias_scale_py) for b in bias_loaded]

            # Fold dequant
            if fold_dequant and layer.get("type") == "Linear":
                qw_now = layer.get("quant_weight")
                if isinstance(qw_now, dict) and qw_now.get("scale"):
                    sw = float(qw_now["scale"])
                    bias_dequant_table = [v / sw for v in bias_dequant_table]

            # Warn if any bias_dequant is below data_t precision
            if bias_dequant_table and use_fixed:
                data_t_lsb = 2 ** -(fixed_config.get("width", 32) - fixed_config.get("int", 16))
                nonzero = [v for v in bias_dequant_table if abs(v) > 0]
                if nonzero:
                    smallest = min(abs(v) for v in nonzero)
                    if smallest < data_t_lsb / 2:
                        print(f"[SANITY] L{i} {ltype}: bias_dequant min={smallest:.2e} < data_t.LSB/2={data_t_lsb/2:.2e} - conversion may lose precision")
                largest = max((abs(v) for v in bias_dequant_table), default=0)
                data_t_max = 2 ** (fixed_config.get("int", 16) - 1)
                if largest > data_t_max * 0.9:
                    print(f"[SANITY] L{i} {ltype}: bias_dequant max={largest:.2e} >= 0.9xdata_t.max - saturation risk")

        ctx = {
            "index": i, "layer": layer,
            "quant_weight": layer.get("quant_weight"),
            "quant_bias":   layer.get("quant_bias"),
            "bias_dequant_table":  bias_dequant_table,
            "fixed_config": fixed_config, "scale_config": scale_config,
            "use_fixed": use_fixed, "use_pragma": use_pragma,
            "parallel_factor": parallel_factor, "opt_tags": opt_tags,
            "conv_partition_factor": conv_partition_factor,
            "input_is_spike": i in spike_input_layers,
            "use_sparse": use_sparse and ltype == "Linear",   # FC-only; Conv if-skip gave ~0% and is dropped
            "act_quant": layer.get("act_quant"),
            "fold_dequant": fold_dequant,
            "bit_shift_beta": bit_shift_beta,
            "beta_shift_k": _beta_to_shift(layer.get("beta", 0.9)) if ltype == "LIF" else None,
        }

        match ltype:
            case "Linear":
                render_template(env, "fc_layer.cpp.j2",  ctx, output_dir, f"fc_layer{i}.cpp")
                render_template(env, "fc_layer.h.j2",    ctx, output_dir, f"fc_layer{i}.h")
                layer_sources.append(f"fc_layer{i}.cpp")
                layer_headers.append(f"fc_layer{i}.h")

            case "LIF":
                render_template(env, "neuron_layer.cpp.j2", ctx, output_dir, f"neuron_layer{i}.cpp")
                render_template(env, "neuron_layer.h.j2",   ctx, output_dir, f"neuron_layer{i}.h")
                layer_sources.append(f"neuron_layer{i}.cpp")
                layer_headers.append(f"neuron_layer{i}.h")

            case "DepthwiseConv2d":
                render_template(env, "dw_conv_layer.cpp.j2", ctx, output_dir, f"dw_conv_layer{i}.cpp")
                render_template(env, "dw_conv_layer.h.j2",   ctx, output_dir, f"dw_conv_layer{i}.h")
                layer_sources.append(f"dw_conv_layer{i}.cpp")
                layer_headers.append(f"dw_conv_layer{i}.h")

            case "Conv2d":
                render_template(env, "conv_layer.cpp.j2", ctx, output_dir, f"conv_layer{i}.cpp")
                render_template(env, "conv_layer.h.j2",   ctx, output_dir, f"conv_layer{i}.h")
                layer_sources.append(f"conv_layer{i}.cpp")
                layer_headers.append(f"conv_layer{i}.h")

            case "AvgPool2d" | "MaxPool2d":
                ctx["pool_type"] = layer.get("pool_type", "avg" if ltype == "AvgPool2d" else "max")
                render_template(env, "pool_layer.cpp.j2", ctx, output_dir, f"pool_layer{i}.cpp")
                render_template(env, "pool_layer.h.j2",   ctx, output_dir, f"pool_layer{i}.h")
                layer_sources.append(f"pool_layer{i}.cpp")
                layer_headers.append(f"pool_layer{i}.h")

            case "ReLU":
                pass  # inlined in the model template

            case _:
                print(f"[!] Layer type '{ltype}' not supported, skipped.")

    # Model-level context
    if isinstance(ir["input_dim"], (list, tuple)):
        input_c, input_h, input_w = (int(x) for x in ir["input_dim"])
        input_dim_flat = input_c * input_h * input_w
    else:
        input_c = input_h = input_w = None
        input_dim_flat = int(ir["input_dim"])

    lif_layers     = [{"global_idx": i, "out_dim": l["out_dim"],
                       "lif_mem_narrow": l.get("lif_mem_narrow", False),
                       "lif_mem_width":  l.get("lif_mem_width"),
                       "lif_mem_int":    l.get("lif_mem_int")}
                      for i, l in enumerate(ir["layers"]) if l["type"] == "LIF"]
    num_lif_layers = len(lif_layers)

    training_meta = ir.get("training_meta", {})
    model_meta = {
        "ir_path":        os.path.abspath(ir_path),
        "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type":     ir.get("model", "unknown"),
        "config_name":    config_name,
        "quant_mode":     quant_mode,
        "training_method": ir.get("training_method", "FP32"),
        "timesteps":      ir["timesteps"],
        "input_dim":      ir["input_dim"],
        "input_dim_flat": input_dim_flat,
        "output_dim":     ir["output_dim"],
        "num_layers":     len(ir["layers"]),
        "num_linear":     sum(1 for l in ir["layers"] if l["type"] == "Linear"),
        "num_lif":        num_lif_layers,
        "num_conv":       sum(1 for l in ir["layers"] if l["type"] in ("Conv2d", "DepthwiseConv2d")),
        "use_fixed":      use_fixed,
        "use_pragma":     use_pragma,
        "use_stage":      True,
        "epochs":         training_meta.get("epochs"),
        "pretrain_epochs": training_meta.get("pretrain_epochs"),
        "best_test_acc":  training_meta.get("best_test_acc"),
        "best_test_epoch": training_meta.get("best_test_epoch"),
        "pretrained_acc": training_meta.get("pretrained_acc"),
        "final_test_acc": training_meta.get("final_test_acc"),
        "lr":             training_meta.get("lr"),
        "weight_decay":   training_meta.get("weight_decay"),
    }
    if use_fixed:
        model_meta["fixed_width"] = fixed_config["width"]
        model_meta["fixed_int"]   = fixed_config["int"]

    layer_summary = []
    for l in ir["layers"]:
        match l["type"]:
            case "Linear":
                layer_summary.append(f"Linear({l.get('in_dim','?')} -> {l.get('out_dim','?')})")
            case "LIF":
                layer_summary.append(f"LIF(beta={l.get('beta',0.9):.2f})")
            case "DepthwiseConv2d":
                layer_summary.append(f"DWConv2d(ch={l.get('channels','?')}, k={l.get('kernel_size','?')})")
            case "Conv2d":
                k = l.get('kernel_size','?')
                layer_summary.append(f"Conv2d({l.get('in_ch','?')}x{k}x{k} -> {l.get('out_ch','?')})")
            case "AvgPool2d" | "MaxPool2d":
                layer_summary.append(
                    f"{l['type']}(k={l.get('kernel_size','?')}, "
                    f"{l.get('in_h','?')}x{l.get('in_w','?')} -> "
                    f"{l.get('out_h','?')}x{l.get('out_w','?')})")

    ctx_model = {
        "layer_headers": layer_headers,
        "input_dim": ir["input_dim"], "timesteps": ir["timesteps"], "output_dim": ir["output_dim"],
        # Output time-average as a power-of-two right shift instead of /TIMESTEPS
        "output_shift": max(0, math.ceil(math.log2(ir["timesteps"]))),
        "input_dim_flat": input_dim_flat, "input_c": input_c, "input_h": input_h, "input_w": input_w,
        "lif_betas": lif_betas, "lif_vths": lif_vths,
        "fixed_config": fixed_config, "use_fixed": use_fixed, "use_pragma": use_pragma,
        "stages": stages,
        "lif_layers": lif_layers, "num_lif_layers": num_lif_layers,
        "model_meta": model_meta, "layer_summary": layer_summary,
        "input_fixed_config": input_fixed_config, "output_fixed_config": output_fixed_config,
        "acc_fixed_config": acc_fixed_config,
        "input_hls_type": input_hls_type,   "input_gcc_type": input_gcc_type,
        "output_hls_type": output_hls_type, "output_gcc_type": output_gcc_type,
        "acc_hls_type": acc_hls_type,       "acc_gcc_type": acc_gcc_type,
        "opt_tags": opt_tags, "parallel_factor": parallel_factor,
        "encoding": encoding,
        "input_is_repeat": encoding == "repeat",
        # Use the effective binary flag (set by encoding=="rate" OR the --input-is-binary override)
        # see stage["input_is_spike"]
        "input_is_binary": bool(input_is_binary),
        "spike_input_layers": spike_input_layers,
        # SD card fields used by model.h SD_DATA_DIR / SD_NUM_SAMPLES.
        "sd_data_dir":    _sd_data_dir(ir.get("dataset_kind", "nmnist"), encoding, ir["timesteps"],
                                       sd_encoding, input_is_binary=input_is_binary),
        "sd_num_samples": _sd_num_samples(ir.get("dataset_kind", "nmnist")),
        "sd_samples_per_file": _sd_samples_per_file(input_dim_flat * ir["timesteps"] * 4,
                                                    _sd_num_samples(ir.get("dataset_kind", "nmnist"))),
    }

    render_template(env, "model.cpp.j2",    ctx_model, output_dir, "model.cpp")
    render_template(env, "model.h.j2",      ctx_model, output_dir, "model.h")

    # Per-project mono HLS op directives. --mul-impl-fabric
    if mul_impl_fabric:
        with open(os.path.join(output_dir, "hls_directives.tcl"), "w") as f:
            f.write("# Auto-generated by converter for --mul-impl-fabric (binary/spike input).\n")
            f.write("config_op mul -impl fabric\n")
            f.write("config_op add -impl fabric\n")
    render_template(env, "reset_node.cpp.j2",
                    {"use_fixed": use_fixed, "use_pragma": use_pragma, "fixed_config": fixed_config},
                    output_dir, "reset_node.cpp")
    render_template(env, "reset_node.h.j2",
                    {"use_fixed": use_fixed, "use_pragma": use_pragma, "fixed_config": fixed_config},
                    output_dir, "reset_node.h")

    # Bambu backend: render run_bambu.sh, Makefile, AXI wrapper; copy build TCL.
    if backend == "bambu":
        import shutil
        bambu_ctx = {
            "project_name": subdir,
            "layer_sources": layer_sources,
            "bambu_opt": bambu_opt or "",
            "bambu_extra": bambu_extra or "",
        }
        render_template(env, "run_bambu.sh.j2", bambu_ctx, output_dir, "run_bambu.sh")
        render_template(env, "Makefile.j2", bambu_ctx, output_dir, "Makefile")
        os.chmod(os.path.join(output_dir, "run_bambu.sh"), 0o755)

        wrapper_ctx = {
            "input_words": input_dim_flat if ir.get("encoding", "rate") == "repeat"
                           else ir["timesteps"] * input_dim_flat,
            "output_words": ir["output_dim"],
            "dut_module_name": "_Z9inferencePKfPf",  # mangled name of inference(const float*, float*)
        }
        render_template(env, "bambu_inference_axi_wrapper.v.j2", wrapper_ctx,
                        output_dir, "bambu_inference_axi_wrapper.v")

        xilinx_tpl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "frontend", "xilinx_templates")
        for tcl_file in ("build_block_design_bambu.tcl", "run_vitis_bambu.tcl"):
            tcl_src = os.path.join(xilinx_tpl_dir, tcl_file)
            if os.path.isfile(tcl_src):
                shutil.copy2(tcl_src, os.path.join(output_dir, tcl_file))
            else:
                print(f"[Warn] TCL template not found: {tcl_src}")
        print("[Done] Bambu: run_bambu.sh, Makefile, AXI wrapper, TCL scripts")

    # Provenance: codegen step. The SW scripts append checkpoint / TOML / training info.
    import sys
    with open(os.path.join(project_dir, "build_information.txt"), "w") as f:
        f.write("build_information\n")
        f.write(f"generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ir_source:  {ir_path}\n")
        f.write(f"config:     {config_name}\n")
        f.write(f"backend:    {backend}\n")
        f.write(f"sparse:     {use_sparse}\n")
        f.write(f"converter:  python {' '.join(sys.argv)}\n")

    print(f"[Done] Conversion completed. Output_dir: {output_dir}")

    # Streaming per-stage generation
    if streaming:
        emit_streaming_stages(
            output_dir, stages, ir, ir_path,
            lif_betas, lif_vths,
            fixed_config, use_fixed, use_pragma,
            parallel_factor, opt_tags, spike_input_layers,
            layer_sources, layer_headers,
            input_hls_type, input_gcc_type,
            output_hls_type, output_gcc_type,
            acc_hls_type, acc_gcc_type, encoding,
            per_layer_lif_mem=per_layer_lif_mem,
            per_lif_mem_widths=per_lif_mem_widths,
            mul_impl_fabric=mul_impl_fabric,
        )


    with open(ir_path) as f:
        clean_ir = json.load(f)
    clean_ir["optimization"] = {
        "config":          config_name,
        "opt":             sorted(opt_tags),
        "sparse":          bool(use_sparse),
        "quant_mode":      quant_mode,
        "data_width":      fixed_config.get("width") if use_fixed else None,
        "data_int":        fixed_config.get("int") if use_fixed else None,
        "conv_oc_factor":  conv_oc_factor,
        "conv_oc_max":     conv_oc_max,
        "lif_mem_profile": lif_mem_profile is not None,
        "lif_mem_slack":   lif_mem_slack,
        "backend":         backend,
        "bambu_opt":       bambu_opt,
        "bambu_extra":     bambu_extra,
        "fold_dequant":    bool(fold_dequant),
        "bit_shift_beta":  bool(bit_shift_beta),
        "mul_impl_fabric": bool(mul_impl_fabric),
        "parallel_factor": parallel_factor,
        "input_is_binary": bool(input_is_binary),
    }
    _PER_LAYER_OPT_KEYS = ("oc_factor", "acc_width", "acc_int", "scale_width",
                           "scale_int", "lif_mem_width", "lif_mem_int", "lif_mem_narrow")
    for cl, ml in zip(clean_ir["layers"], ir["layers"]):
        per = {k: ml[k] for k in _PER_LAYER_OPT_KEYS if k in ml}
        if per:
            cl["opt"] = per
    with open(ir_path, "w") as f:
        json.dump(clean_ir, f, indent=2)

    return output_dir


# CLI

def parse_config(config_str):
    """Map a preset string (S / SQ / SP / SPQ) to (use_fixed, use_pragma).

    Unknown presets are rejected so a silent fall-through cannot build the wrong configuration.
    """
    s = config_str.upper()
    if s not in ("S", "SQ", "SP", "SPQ"):
        raise SystemExit(f"Error: invalid --config '{config_str}'. Choose from S, SQ, SP, SPQ.")
    return ('Q' in s), ('P' in s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IR JSON to HLS C++ via Jinja2 templates")
    parser.add_argument("ir_json",           help="Path to IR JSON file")
    parser.add_argument("--fixed",           action="store_true", help="Use ap_fixed types")
    parser.add_argument("--pragma",          action="store_true", help="Enable HLS pragmas")
    parser.add_argument("--config",          type=str, default=None,
                        help="Preset: S | SQ | SP | SPQ  (overrides --fixed / --pragma)")
    parser.add_argument("--parallel-factor", type=int, default=None,
                        help="Unroll factor for large FC/Conv loops (default: 8)")
    parser.add_argument("--unroll",          type=str, default=None,
                        help="Convolution unroll tags: ck (kernel loops), ic (input-channel), "
                             "oc (output-channel). ic and oc imply ck.")
    parser.add_argument("--dataflow",        action="store_true",
                        help="Enable the DATAFLOW pragma on the timestep loop with inter-stage PIPO.")
    parser.add_argument("--sparse",          nargs="?", const="sp", default=None,
                        help="Spike-driven sparsity in FC layers (skip zero-spike MACs). "
                             "Use bare --sparse or --sparse sp.")
    parser.add_argument("--project",         type=str, default=None,
                        help="Output project name; files go to backend_projects/<project>/cpp/")
    parser.add_argument("--streaming",       action="store_true",
                        help="Generate per-stage streaming HLS projects (additive)")
    parser.add_argument("--conv-oc-factor",  type=str, default=None,
                        help="Conv2d OC unroll: '0' (off, default), 'auto' (full unroll where "
                             "OC <= --conv-oc-max; large OC wrecks HLS scheduling), or N.")
    parser.add_argument("--conv-oc-max",     type=int, default=None,
                        help="In auto mode, Conv2d layers with OC<=N get full OC unroll (default: 16)")
    # Per-layer bitwidth sizing is on by default; these flags fall back to global widths.
    parser.add_argument("--no-per-layer-scale", action="store_true",
                        help="Disable per-layer scale_t sizing (fall back to global scale_config)")
    parser.add_argument("--no-per-layer-acc", action="store_true",
                        help="Disable per-layer MAC acc_t sizing (fall back to width*2, int*2 formula)")
    parser.add_argument("--no-per-layer-lif-mem", action="store_true",
                        help="Force lif_mem_t = data_t everywhere, even when a --lif-mem-profile is given")
    parser.add_argument("--lif-mem-profile",  type=str, default=None,
                        help="JSON from tools/profile_lif_mem.py; narrows each lif_mem_t from the "
                             "measured |mem|_max. Without it lif_mem_t stays data_t.")
    parser.add_argument("--lif-mem-slack",    type=float, default=2.0,
                        help="Slack multiplier on profiled |mem|_max (default 2.0)")
    parser.add_argument("--backend",          type=str, default=None,
                        choices=["vitis", "bambu"],
                        help="HLS backend: vitis (ap_fixed + pragmas) or bambu (float, plain C++)")
    parser.add_argument("--bambu-opt",        type=str, default=None,
                        help="Bambu optimization level baked into run_bambu.sh (e.g. -O2, -O3)")
    parser.add_argument("--bambu-extra",      type=str, default=None,
                        help="Extra Bambu flags for run_bambu.sh (e.g. --pipelining=inference)")
    parser.add_argument("--data-width",       type=int, default=None,
                        help="Override the data_t total width instead of the default upgrade to 32 "
                             "bits. Narrower is cheaper in DSPs but can saturate; check with csim.")
    parser.add_argument("--data-int-width", "--data-int", dest="data_int", type=int, default=None,
                        help="Override I in ap_fixed<W,I> (default: data_width // 2).")
    parser.add_argument("--fold-dequant",     action="store_true",
                        help="Fold the weight dequant scale into the LIF threshold, dropping the "
                             "per-output multiply. Needs a wide data_t. Linear->LIF only.")
    parser.add_argument("--bit-shift-beta",   action="store_true",
                        help="Snap each beta to 1-2^-k and run the leak as mem-(mem>>k). Removes the "
                             "leak DSP; best when beta trains near a power of two.")
    parser.add_argument("--mul-impl-fabric",  action="store_true",
                        help="Map multiplies to LUT fabric instead of DSP48. Only safe for "
                             "binary-input FCN; blows up LUT on conv, do NOT use for CSNN.")
    parser.add_argument("--sd-encoding",      default=None,
                        help="Encoding label in SD_DATA_DIR = 0:/<dataset>/<sd-encoding>/t<T>, so "
                             "several encodings coexist on the card. Defaults to the IR encoding.")
    parser.add_argument("--input-is-binary",  action="store_true",
                        help="Treat the input as binary {0,1} so the first layer adds instead of "
                             "multiplies. Custom route only; the config route reads it from the IR.")

    args = parser.parse_args()
    conv_oc_factor_explicit = args.conv_oc_factor is not None

    if not os.path.isfile(args.ir_json):
        raise SystemExit(f"Error: IR file not found: {args.ir_json}. Run export_ir.py first.")

    # An IR-recorded optimization block fills in whatever the CLI did not set.
    with open(args.ir_json) as f:
        _ir_opt = json.load(f).get("optimization", {})
    if _ir_opt:
        if args.config is None and not args.fixed and not args.pragma and _ir_opt.get("config"):
            args.config = _ir_opt["config"]
        if args.sparse is None and _ir_opt.get("sparse"):
            args.sparse = "sp"
        if args.bambu_opt is None and _ir_opt.get("bambu_opt"):
            args.bambu_opt = _ir_opt["bambu_opt"]
        if args.bambu_extra is None and _ir_opt.get("bambu_extra"):
            args.bambu_extra = _ir_opt["bambu_extra"]
        for _flag in ("streaming", "fold_dequant", "bit_shift_beta", "mul_impl_fabric", "input_is_binary"):
            if not getattr(args, _flag) and _ir_opt.get(_flag):
                setattr(args, _flag, True)

    # Explicit CLI values take precedence over IR values and built-in defaults.
    for name, default in (("conv_oc_factor", "0"), ("conv_oc_max", 16),
                          ("parallel_factor", 8), ("backend", "vitis")):
        if getattr(args, name) is None:
            value = _ir_opt.get(name)
            setattr(args, name, default if value is None else value)
    args.conv_oc_factor = str(args.conv_oc_factor)

    if args.config:
        use_fixed, use_pragma = parse_config(args.config)
    else:
        use_fixed, use_pragma = args.fixed, args.pragma

    # Map the public --unroll/--dataflow flags onto the internal opt-tag set.
    opt_tags = []
    unroll = {t.strip().lower() for t in args.unroll.split(",") if t.strip()} if args.unroll else set()
    unknown_unroll = unroll - {"ck", "ic", "oc"}
    if unknown_unroll:
        raise SystemExit(f"Error: invalid --unroll tag(s) {unknown_unroll}. Choose from ck, ic, oc.")
    if "ck" in unroll:
        opt_tags.append("conv_kernel")
    if "ic" in unroll:
        opt_tags.append("conv_ic")     # convert_model auto-adds conv_kernel as its dependency
    if args.dataflow:
        opt_tags.append("dataflow")
    opt_tags = opt_tags or None
    # --unroll oc enables auto OC-unroll unless an explicit --conv-oc-factor was passed.
    conv_oc_factor = args.conv_oc_factor
    if "oc" in unroll and not conv_oc_factor_explicit and conv_oc_factor.strip().lower() in ("0", "none", ""):
        conv_oc_factor = "auto"

    profile_data = None
    if args.lif_mem_profile:
        if not os.path.isfile(args.lif_mem_profile):
            raise SystemExit(f"Error: --lif-mem-profile file not found: {args.lif_mem_profile}")
        with open(args.lif_mem_profile) as f:
            profile_data = json.load(f)

    convert_model(args.ir_json, use_fixed=use_fixed, use_pragma=use_pragma,
                  parallel_factor=args.parallel_factor, opt_tags=opt_tags,
                  project_name=args.project, use_sparse=args.sparse is not None,
                  streaming=args.streaming,
                  conv_oc_factor=conv_oc_factor, conv_oc_max=args.conv_oc_max,
                  per_layer_scale=not args.no_per_layer_scale,
                  per_layer_acc=not args.no_per_layer_acc,
                  per_layer_lif_mem=not args.no_per_layer_lif_mem,
                  lif_mem_profile=profile_data,
                  lif_mem_slack=args.lif_mem_slack,
                  backend=args.backend, bambu_opt=args.bambu_opt, bambu_extra=args.bambu_extra,
                  fold_dequant=args.fold_dequant, bit_shift_beta=args.bit_shift_beta,
                  input_is_binary=args.input_is_binary,
                  data_width=args.data_width, data_int=args.data_int,
                  mul_impl_fabric=args.mul_impl_fabric,
                  sd_encoding=args.sd_encoding)
