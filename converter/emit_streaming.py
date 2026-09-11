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

"""Streaming code generation: per-stage AXI-Stream HLS projects + Vivado/Vitis TCL."""
import os
import glob as _glob
import math
import shutil
import jinja2

from emit import render_template

TEMPLATE_STREAMING_DIR = os.path.join(os.path.dirname(__file__), "templates_streaming")


def emit_streaming_stages(output_dir, stages, ir, ir_path,
                          lif_betas, lif_vths,
                          fixed_config, use_fixed, use_pragma,
                          parallel_factor, opt_tags, spike_input_layers,
                          layer_sources, layer_headers,
                          input_hls_type, input_gcc_type,
                          output_hls_type, output_gcc_type,
                          acc_hls_type, acc_gcc_type, encoding,
                          per_layer_lif_mem=True,
                          per_lif_mem_widths=None,
                          mul_impl_fabric=False):
    """Generate streaming code: C++ stages under output_dir/streaming/ (cpp/streaming/),
    build TCL + host harness under the sibling xilinx/streaming/."""

    streaming_dir = os.path.join(output_dir, "streaming")  # cpp/streaming/, C++ sources only
    os.makedirs(streaming_dir, exist_ok=True)
    # Build scripts (TCL) and the host harness go to the sibling xilinx/ tree
    xilinx_streaming_dir = os.path.normpath(
        os.path.join(output_dir, os.pardir, "xilinx", "streaming"))
    os.makedirs(xilinx_streaming_dir, exist_ok=True)
    # Drop stale generated TCL so a regenerate never leaves an inconsistent build tree.
    for stale in _glob.glob(os.path.join(xilinx_streaming_dir, "vivado_streaming_*.tcl")):
        os.remove(stale)

    env_s = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_STREAMING_DIR),
        trim_blocks=True, lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    num_stages = len(stages)

    for stage in stages:
        stage_idx = stage["stage_idx"]
        stage_dir = os.path.join(streaming_dir, f"stage{stage_idx}")
        os.makedirs(stage_dir, exist_ok=True)
        # Per-stage TCL goes to the xilinx build tree; C++ stays in cpp/streaming/
        xilinx_stage_dir = os.path.join(xilinx_streaming_dir, f"stage{stage_idx}")
        os.makedirs(xilinx_stage_dir, exist_ok=True)
        # Clean stale sources here, and stale TCL in the xilinx stage dir
        for pattern in ["*.cpp", "*.h", "*.tcl"]:
            for stale in _glob.glob(os.path.join(stage_dir, pattern)):
                os.remove(stale)
        for stale in _glob.glob(os.path.join(xilinx_stage_dir, "*.tcl")):
            os.remove(stale)

        is_first = (stage_idx == 0)
        is_last  = (stage_idx == num_stages - 1)

        # Input/output dimensions
        if is_first:
            if isinstance(ir["input_dim"], (list, tuple)):
                in_dim = int(ir["input_dim"][0]) * int(ir["input_dim"][1]) * int(ir["input_dim"][2])
            else:
                in_dim = int(ir["input_dim"])
        else:
            in_dim = stages[stage_idx - 1]["out_dim"]
        out_dim = stage["out_dim"]

        # Layer info
        all_layers = stage["all_layers"]
        non_lif_layers = stage["non_lif_layers"]
        lif_layer = next((l for l in all_layers if l["type"] == "LIF"), None)
        lif_out_dim = lif_layer["out_dim"] if lif_layer else out_dim

        # Collect files to copy
        stage_layer_headers = []
        files_to_copy = {"reset_node.cpp", "reset_node.h"}

        for layer in all_layers:
            global_idx = layer["global_idx"]
            layer_type = layer["type"]
            match layer_type:
                case "Linear":
                    files_to_copy.update([f"fc_layer{global_idx}.cpp", f"fc_layer{global_idx}.h"])
                    stage_layer_headers.append(f"fc_layer{global_idx}.h")
                    pair_idx = layer.get("pair_idx")
                    if pair_idx is not None:
                        files_to_copy.add(f"weights{pair_idx}.h")
                        for chunk_h in _glob.glob(os.path.join(output_dir, f"weights{pair_idx}_*.h")):
                            files_to_copy.add(os.path.basename(chunk_h))
                        if layer.get("bias"):
                            files_to_copy.add(f"biases{pair_idx}.h")
                case "LIF":
                    files_to_copy.update([f"neuron_layer{global_idx}.cpp", f"neuron_layer{global_idx}.h"])
                    stage_layer_headers.append(f"neuron_layer{global_idx}.h")
                case "Conv2d":
                    files_to_copy.update([f"conv_layer{global_idx}.cpp", f"conv_layer{global_idx}.h"])
                    stage_layer_headers.append(f"conv_layer{global_idx}.h")
                    pair_idx = layer.get("pair_idx")
                    if pair_idx is not None:
                        files_to_copy.add(f"weights{pair_idx}.h")
                        if layer.get("bias", True):
                            files_to_copy.add(f"biases{pair_idx}.h")
                case "DepthwiseConv2d":
                    files_to_copy.update([f"dw_conv_layer{global_idx}.cpp", f"dw_conv_layer{global_idx}.h"])
                    stage_layer_headers.append(f"dw_conv_layer{global_idx}.h")
                    pair_idx = layer.get("pair_idx")
                    if pair_idx is not None:
                        files_to_copy.add(f"weights{pair_idx}.h")
                        if layer.get("bias", True):
                            files_to_copy.add(f"biases{pair_idx}.h")
                case "AvgPool2d" | "MaxPool2d":
                    files_to_copy.update([f"pool_layer{global_idx}.cpp", f"pool_layer{global_idx}.h"])
                    stage_layer_headers.append(f"pool_layer{global_idx}.h")

        for fname in files_to_copy:
            src = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, stage_dir)
            else:
                print(f"[Warn] streaming stage{stage_idx}: file not found: {fname}")

        # Template context
        ctx = {
            "stage_idx":       stage_idx,
            "description":     stage["description"],
            "in_dim":          in_dim,
            "out_dim":         out_dim,
            "timesteps":       ir["timesteps"],
            # Output time-average uses a power-of-two divide (right shift) instead of /TIMESTEPS
            "output_shift":    max(0, math.ceil(math.log2(ir["timesteps"]))),
            "encoding":        encoding,
            "is_first_stage":  is_first,
            "is_last_stage":   is_last,
            "lif_beta":        lif_betas[stage["lif_order_idx"]],
            "lif_vth":         lif_vths[stage["lif_order_idx"]],
            "lif_out_dim":     lif_out_dim,
            "all_layers":      all_layers,
            "non_lif_layers":  non_lif_layers,
            "stage_layer_headers": stage_layer_headers,
            "fixed_config":    fixed_config,
            "use_fixed":       use_fixed,
            "use_pragma":      use_pragma,
            "parallel_factor": parallel_factor,
            "opt_tags":        opt_tags,
            "mul_impl_fabric": mul_impl_fabric,
            "spike_input_layers": spike_input_layers,
            "input_is_spike":  stage.get("input_is_spike", False),
            "output_is_spike": stage.get("output_is_spike", True),
            "input_is_narrow_data":  stage.get("input_is_narrow_data", False),
            "output_is_narrow_data": stage.get("output_is_narrow_data", False),
            # Per-stage LIF mem sizing
            **{k: v for k, v in zip(
                ["lif_mem_width", "lif_mem_int", "lif_mem_narrow"],
                (per_lif_mem_widths[stage["lif_order_idx"]]
                 if per_layer_lif_mem and per_lif_mem_widths is not None
                 else (fixed_config["width"], fixed_config["int"], False)),
            )},
            # HLS/GCC type strings for the first/last stage DDR interface
            "input_hls_type":  input_hls_type,
            "input_gcc_type":  input_gcc_type,
            "output_hls_type": output_hls_type,
            "output_gcc_type": output_gcc_type,
            "acc_hls_type":    acc_hls_type,
            "acc_gcc_type":    acc_gcc_type,
        }

        render_template(env_s, "stage_top.cpp.j2", ctx, stage_dir, f"stage{stage_idx}_top.cpp")
        render_template(env_s, "stage_top.h.j2",   ctx, stage_dir, f"stage{stage_idx}_top.h")
        render_template(env_s, "run_hls_stage.tcl.j2", ctx, xilinx_stage_dir, "run_hls_stage.tcl")

        role = "FIRST" if is_first else ("LAST" if is_last else "MID")
        print(f"[STREAMING] Stage {stage_idx} ({role}): {stage['description']}  "
              f"(in={in_dim}, out={out_dim}, files={len(files_to_copy)})")

    # Vivado axis_data_fifo accepts a power of two from 16 to 32768
    AXIS_FIFO_DEPTH_MIN = 16
    AXIS_FIFO_DEPTH_MAX = 32768
    fifo_depths = []
    for stage_idx in range(num_stages - 1):
        # AXI-Stream FIFO depth must be a power of 2; round up
        raw = stages[stage_idx]["out_dim"]
        depth = 1
        while depth < raw:
            depth <<= 1
        if depth < AXIS_FIFO_DEPTH_MIN:
            depth = AXIS_FIFO_DEPTH_MIN
        if depth > AXIS_FIFO_DEPTH_MAX:
            print(f"[Warn] Stage {stage_idx} FIFO depth {depth} (raw out_dim={raw}) "
                  f"exceeds Vivado axis_data_fifo cap; capping to {AXIS_FIFO_DEPTH_MAX}. "
                  f"If runtime backpressure causes hang, consider BRAM-backed FIFO.")
            depth = AXIS_FIFO_DEPTH_MAX
        fifo_depths.append(depth)

    stage_has_gmem = []
    for stage_idx in range(num_stages):
        has_gmem = (stage_idx == 0) or (stage_idx == num_stages - 1)
        stage_has_gmem.append(has_gmem)

    vivado_ctx = {
        "num_stages":    num_stages,
        "fifo_depths":   fifo_depths,
        "stage_has_gmem": stage_has_gmem,
    }
    render_template(env_s, "vivado_streaming_zcu104.tcl.j2", vivado_ctx,
                    xilinx_streaming_dir, "vivado_streaming_zcu104.tcl")
    render_template(env_s, "run_vitis_streaming.tcl.j2", vivado_ctx,
                    xilinx_streaming_dir, "run_vitis_streaming.tcl")
    render_template(env_s, "main_sd_streaming.c.j2", vivado_ctx,
                    xilinx_streaming_dir, "main_sd_streaming.c")

    print(f"[STREAMING] C++ stages in {streaming_dir}")
    print(f"[STREAMING] Build scripts in {xilinx_streaming_dir}")
    print(f"[STREAMING] Vivado Tcl:  {xilinx_streaming_dir}/vivado_streaming_zcu104.tcl")
    print(f"[STREAMING] Vitis Tcl:   {xilinx_streaming_dir}/run_vitis_streaming.tcl")
