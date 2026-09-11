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

"""IR layer-dimension propagation, pair indexing, and pipeline-stage partitioning."""


def compute_layer_dims(ir):
    """Propagate spatial dimensions and prev_out_dim through each IR layer in-place."""
    if isinstance(ir["input_dim"], (list, tuple)):
        cur_c, cur_h, cur_w = (int(x) for x in ir["input_dim"])
        prev_out_dim = cur_c * cur_h * cur_w
    else:
        prev_out_dim = ir["input_dim"]
        cur_c = cur_h = cur_w = None

    for layer in ir["layers"]:
        layer["prev_out_dim"] = prev_out_dim

        match layer["type"]:
            case "DepthwiseConv2d":
                if "in_h" not in layer and cur_h is not None:
                    layer["in_h"], layer["in_w"] = cur_h, cur_w
                if "out_h" not in layer and cur_h is not None:
                    k, s, p = layer["kernel_size"], layer.get("stride", 1), layer.get("padding", 0)
                    layer["out_h"] = (cur_h + 2*p - k) // s + 1
                    layer["out_w"] = (cur_w + 2*p - k) // s + 1
                ch = layer["channels"]
                layer["out_dim"] = ch * layer["out_h"] * layer["out_w"]
                cur_c, cur_h, cur_w = ch, layer["out_h"], layer["out_w"]

            case "Conv2d":
                if "in_h" not in layer and cur_h is not None:
                    layer["in_h"], layer["in_w"] = cur_h, cur_w
                if "out_h" not in layer and cur_h is not None:
                    k, s, p = layer["kernel_size"], layer.get("stride", 1), layer.get("padding", 0)
                    layer["out_h"] = (cur_h + 2*p - k) // s + 1
                    layer["out_w"] = (cur_w + 2*p - k) // s + 1
                layer["out_dim"] = layer["out_ch"] * layer["out_h"] * layer["out_w"]
                cur_c, cur_h, cur_w = layer["out_ch"], layer["out_h"], layer["out_w"]

            case "AvgPool2d" | "MaxPool2d":
                if cur_c is not None:
                    layer.setdefault("channels", cur_c)
                    if "in_h" not in layer:
                        layer["in_h"], layer["in_w"] = cur_h, cur_w
                    k, s, p = layer["kernel_size"], layer.get("stride", layer["kernel_size"]), layer.get("padding", 0)
                    if "out_h" not in layer:
                        layer["out_h"] = (cur_h + 2*p - k) // s + 1
                        layer["out_w"] = (cur_w + 2*p - k) // s + 1
                    layer["out_dim"] = layer["channels"] * layer["out_h"] * layer["out_w"]
                    cur_h, cur_w = layer["out_h"], layer["out_w"]
                else:
                    layer["out_dim"] = layer["prev_out_dim"] // (layer["kernel_size"] ** 2)

            case "Linear":
                cur_c = cur_h = cur_w = None  # data is now flat

            # LIF, ReLU, etc.: out_dim already set by export_ir, nothing to propagate.

        prev_out_dim = layer["out_dim"]


def compute_pair_indices(layers):
    """Assign a sequential pair_idx to every weight-bearing layer."""
    pair_idx = 0
    for layer in layers:
        if layer["type"] in ("Linear", "Conv2d", "DepthwiseConv2d"):
            layer["pair_idx"] = pair_idx
            pair_idx += 1


def compute_stages(layers, pool_after_lif=False):
    """Partition layers into pipeline stages.
    Returns a list of stage dicts, each containing:
        stage_idx, description, all_layers, non_lif_layers,
        lif_layer_idx, lif_order_idx, out_dim, output_is_spike, pool_after_lif
    """
    stages, current, lif_order_idx = [], [], 0

    for i, layer in enumerate(layers):
        layer["global_idx"] = i
        current.append(layer)

        if pool_after_lif:
            is_pool     = layer["type"] in ("AvgPool2d", "MaxPool2d")
            next_is_pool = (i + 1 < len(layers) and
                            layers[i + 1]["type"] in ("AvgPool2d", "MaxPool2d"))
            end_stage = is_pool or (layer["type"] == "LIF" and not next_is_pool)
        else:
            end_stage = layer["type"] == "LIF"

        if not end_stage:
            continue

        lif_layer = next((l for l in current if l["type"] == "LIF"), None)
        if lif_layer is None:
            continue  # pool without a preceding LIF; skip

        last = current[-1]
        for j, l in enumerate(current):
            l["is_last_in_stage"] = (j == len(current) - 1)

        stages.append({
            "stage_idx":      len(stages),
            "description":    " -> ".join(l["type"] for l in current),
            "all_layers":     current.copy(),
            "non_lif_layers": [l for l in current if l["type"] != "LIF"],
            "lif_layer_idx":  lif_layer["global_idx"],
            "lif_order_idx":  lif_order_idx,
            "out_dim":        last.get("out_dim", lif_layer["out_dim"]),
            # Stage output type must match the pool template's actual signature:
            # MaxPool or LIF output spike_t (binary); AvgPool of spikes and any data_t-input pool output data_t.
            "output_is_spike": (
                last["type"] == "LIF" or
                (pool_after_lif and last["type"] == "MaxPool2d")
            ),
            # AvgPool of binary spikes outputs values in [0,1], so narrow data_t saves resources.
            "output_is_narrow_data": (
                pool_after_lif and last["type"] == "AvgPool2d"
            ),
            # Raw beta/vth/reset for the LIF neuron update. Mem storage width is set later in the caller.
            "lif_beta_raw":      float(lif_layer.get("beta", 1.0))      if lif_layer is not None else 1.0,
            "lif_vth_raw":       float(lif_layer.get("threshold", 1.0)) if lif_layer is not None else 1.0,
            "lif_reset_mech":    lif_layer.get("reset_mechanism", "subtract") if lif_layer is not None else "subtract",
            "pool_after_lif": pool_after_lif,
        })
        current = []
        lif_order_idx += 1

    return stages
