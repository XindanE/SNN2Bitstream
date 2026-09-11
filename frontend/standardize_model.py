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

import sys
import torch
import torch.nn as nn
from collections import OrderedDict
import importlib
import argparse


def detect_structure(model):
    """Scan model modules and return flags indicating which layer types are present."""
    flags = {"has_conv": False, "has_pool": False, "has_linear": False, "has_lif": False}
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            flags["has_conv"] = True
        elif isinstance(m, (nn.AvgPool2d, nn.MaxPool2d)):
            flags["has_pool"] = True
        elif isinstance(m, nn.Linear):
            flags["has_linear"] = True
        elif _is_lif(m):
            flags["has_lif"] = True
    return flags


def _is_lif(m):
    """Return True if m is a LIF neuron (has beta + threshold but is not a model container)."""
    if not (hasattr(m, "beta") and hasattr(m, "threshold")):
        return False
    if hasattr(m, "layers") and hasattr(m, "neurons"):
        return False
    return True

def _apply_bn_fold_conv(conv, bns):
    """Fold one or more BatchNorm2d modules into a Conv2d weight/bias in-place."""
    eps = 1e-5
    n = len(bns)

    if n == 1:
        bn = bns[0]
        gamma = bn.weight.data if bn.weight is not None else torch.ones(bn.num_features)
        beta_bn = bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features)
        mean = bn.running_mean
        var = bn.running_var
    else:
        # BatchNormTT2d: fold by averaging the BNs.
        print(f"[Warn] Folding {n} BNs (BatchNormTT2d) by averaging - approximate, may affect accuracy")
        gamma = torch.stack([bn.weight.data if bn.weight is not None else torch.ones(bn.num_features) for bn in bns]).mean(0)
        beta_bn = torch.stack([bn.bias.data if bn.bias is not None else torch.zeros(bn.num_features) for bn in bns]).mean(0)
        mean = torch.stack([bn.running_mean for bn in bns]).mean(0)
        var = torch.stack([bn.running_var for bn in bns]).mean(0)

    # gamma * (x - mean) / sqrt(var + eps)
    scale = gamma / torch.sqrt(var + eps)
    conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
    if conv.bias is not None:
        conv.bias.data = (conv.bias.data - mean) * scale + beta_bn
    else:
        conv.bias = nn.Parameter((-mean) * scale + beta_bn)


def _fold_bn_into_model(model):
    """Walk model modules and fold all BatchNorm2d layers into preceding Conv2d in-place."""
    prev_conv = None
    pending_bns = []
    folded_count = 0

    def _try_fold():
        nonlocal prev_conv, folded_count
        if prev_conv is not None and pending_bns:
            if all(bn.num_features == prev_conv.out_channels for bn in pending_bns):
                _apply_bn_fold_conv(prev_conv, pending_bns)
                n = len(pending_bns)
                print(f"[Info] Folded {'BN' if n == 1 else f'{n} BN(s)'} into Conv2d(out_ch={prev_conv.out_channels})")
                folded_count += n
            else:
                print(f"[Warn] Skipping BN fold: num_features mismatch "
                      f"(BN={pending_bns[0].num_features}, Conv out_ch={prev_conv.out_channels})")
        pending_bns.clear()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            _try_fold()
            prev_conv = m
        elif isinstance(m, nn.BatchNorm2d):
            if prev_conv is not None:
                pending_bns.append(m)
        elif isinstance(m, nn.Linear) or _is_lif(m):
            _try_fold()
            if isinstance(m, nn.Linear):
                prev_conv = None

    _try_fold()

    if folded_count > 0:
        print(f"[Info] Total BN modules folded: {folded_count}")
    return folded_count


def standardize_state_dict(model, input_shape):
    """Walk model modules and produce a standardized state_dict + layer_defs list.

    input_shape: (C, H, W) for CSNN models; None for FCN models.
    Returns (state_dict, layer_defs).
    """
    # Template models (CSNNNet) build self.layer_defs in __init__ with correct spatial tracking. Re-deriving via model.modules() is wrong: pool_layers are registered before conv_layers, so traversal visits pools first and corrupts cur_h/cur_w before conv dims are computed. Use the pre-built layer_defs and already-standardized keys.
    if getattr(model, 'layer_defs', None):
        state_dict = OrderedDict(
            (k, v.detach().cpu()) for k, v in model.state_dict().items()
        )
        return state_dict, list(model.layer_defs)

    state_dict = OrderedDict()
    layer_defs = []

    # Pre-scan: count Linear and LIF modules to identify the last of each.
    num_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    num_lifs    = sum(1 for m in model.modules() if _is_lif(m))

    if input_shape is not None:
        cur_c, cur_h, cur_w = input_shape
    else:
        cur_c, cur_h, cur_w = 0, 0, 0

    # layer counters
    conv_idx      = 0
    dw_conv_idx   = 0
    pw_conv_idx   = 0
    fc_idx        = 0
    conv_lif_idx  = 0
    fc_lif_idx    = 0
    lif_idx       = 0
    linear_count  = 0

    # state flags
    in_fc_phase      = False
    expect_pointwise = False

    for m in model.modules():

        # Conv2d
        # out = (in + 2*padding - kernel) // stride + 1
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            s = m.stride[0]      if isinstance(m.stride, tuple)      else m.stride
            p = m.padding[0]     if isinstance(m.padding, tuple)     else m.padding

            is_depthwise = (m.groups > 1 and m.groups == m.in_channels)
            is_pointwise = (expect_pointwise and k == 1 and m.groups == 1)

            if is_depthwise:
                state_dict[f"dw_conv_layers.{dw_conv_idx}.weight"] = m.weight.detach().cpu()
                if m.bias is not None:
                    state_dict[f"dw_conv_layers.{dw_conv_idx}.bias"] = m.bias.detach().cpu()

                out_h = (cur_h + 2 * p - k) // s + 1
                out_w = (cur_w + 2 * p - k) // s + 1
                layer_defs.append({
                    "type":        "DepthwiseConv2d",
                    "name":        f"dw_conv{dw_conv_idx + 1}",
                    "channels":    m.in_channels,
                    "kernel_size": k,
                    "stride":      s,
                    "padding":     p,
                    "out_h":       out_h,
                    "out_w":       out_w,
                })
                cur_h, cur_w = out_h, out_w
                dw_conv_idx += 1
                expect_pointwise = True

            elif is_pointwise:
                state_dict[f"pw_conv_layers.{pw_conv_idx}.weight"] = m.weight.detach().cpu()
                if m.bias is not None:
                    state_dict[f"pw_conv_layers.{pw_conv_idx}.bias"] = m.bias.detach().cpu()

                layer_defs.append({
                    "type":        "Conv2d",
                    "name":        f"pw_conv{pw_conv_idx + 1}",
                    "in_ch":       cur_c,
                    "out_ch":      m.out_channels,
                    "kernel_size": 1,
                    "stride":      1,
                    "padding":     0,
                    "out_h":       cur_h,
                    "out_w":       cur_w,
                })
                cur_c = m.out_channels
                pw_conv_idx += 1
                expect_pointwise = False

            else:
                state_dict[f"conv_layers.{conv_idx}.weight"] = m.weight.detach().cpu()
                if m.bias is not None:
                    state_dict[f"conv_layers.{conv_idx}.bias"] = m.bias.detach().cpu()

                out_h = (cur_h + 2 * p - k) // s + 1
                out_w = (cur_w + 2 * p - k) // s + 1
                layer_defs.append({
                    "type":        "Conv2d",
                    "name":        f"conv{conv_idx + 1}",
                    "in_ch":       cur_c,
                    "out_ch":      m.out_channels,
                    "kernel_size": k,
                    "stride":      s,
                    "padding":     p,
                    "out_h":       out_h,
                    "out_w":       out_w,
                })
                cur_c = m.out_channels
                cur_h, cur_w = out_h, out_w
                conv_idx += 1
                expect_pointwise = False

        # Pooling
        elif isinstance(m, (nn.AvgPool2d, nn.MaxPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride      if isinstance(m.stride, int)      else m.stride[0]
            if s is None:
                s = k
            p = m.padding if isinstance(m.padding, int) else m.padding[0]

            out_h = (cur_h + 2 * p - k) // s + 1
            out_w = (cur_w + 2 * p - k) // s + 1

            is_avg = isinstance(m, nn.AvgPool2d)
            layer_defs.append({
                "type":        "AvgPool2d" if is_avg else "MaxPool2d",
                "name":        f"pool{conv_idx}",
                "pool_type":   "avg" if is_avg else "max",
                "kernel_size": k,
                "stride":      s,
                "padding":     p,
                "out_h":       out_h,
                "out_w":       out_w,
            })
            cur_h, cur_w = out_h, out_w

        # Linear
        elif isinstance(m, nn.Linear):
            in_fc_phase = True
            linear_count += 1
            is_last_linear = (linear_count == num_linears)

            if is_last_linear:
                state_dict["fc_out.weight"] = m.weight.detach().cpu()
                if m.bias is not None:
                    state_dict["fc_out.bias"] = m.bias.detach().cpu()
                layer_defs.append({
                    "type": "Linear",
                    "name": "fc_out",
                    "in":   m.in_features,
                    "out":  m.out_features,
                    "bitwidth": getattr(m, "_bitwidth", None),
                })
            else:
                state_dict[f"fc_layers.{fc_idx}.weight"] = m.weight.detach().cpu()
                if m.bias is not None:
                    state_dict[f"fc_layers.{fc_idx}.bias"] = m.bias.detach().cpu()
                layer_defs.append({
                    "type": "Linear",
                    "name": f"fc{conv_idx + fc_idx + 1}",
                    "in":   m.in_features,
                    "out":  m.out_features,
                    "bitwidth": getattr(m, "_bitwidth", None),
                })
                fc_idx += 1

        # LIF
        elif _is_lif(m):
            lif_idx += 1
            is_last_lif = (lif_idx == num_lifs)

            beta = m.beta
            thr  = m.threshold
            if torch.is_tensor(beta):
                beta = beta.detach().cpu()
            if torch.is_tensor(thr):
                thr = thr.detach().cpu()

            if in_fc_phase:
                lif_dim = layer_defs[-1].get("out", 0)
                if is_last_lif:
                    state_dict["lif_out.beta"]      = torch.as_tensor(beta)
                    state_dict["lif_out.threshold"] = torch.as_tensor(thr)
                    layer_defs.append({
                        "type":      "LIF",
                        "name":      "lif_out",
                        "in_dim":    lif_dim,
                        "out_dim":   lif_dim,
                        "threshold": float(thr),
                        "reset_mechanism": getattr(m, "reset_mechanism", "subtract"),
                    })
                else:
                    state_dict[f"fc_lif_layers.{fc_lif_idx}.beta"]      = torch.as_tensor(beta)
                    state_dict[f"fc_lif_layers.{fc_lif_idx}.threshold"] = torch.as_tensor(thr)
                    layer_defs.append({
                        "type":      "LIF",
                        "name":      f"lif_fc{conv_idx + fc_lif_idx + 1}",
                        "in_dim":    lif_dim,
                        "out_dim":   lif_dim,
                        "threshold": float(thr),
                        "reset_mechanism": getattr(m, "reset_mechanism", "subtract"),
                    })
                    fc_lif_idx += 1
            else:
                lif_dim = cur_c * cur_h * cur_w
                state_dict[f"conv_lif_layers.{conv_lif_idx}.beta"]      = torch.as_tensor(beta)
                state_dict[f"conv_lif_layers.{conv_lif_idx}.threshold"] = torch.as_tensor(thr)
                layer_defs.append({
                    "type":      "LIF",
                    "name":      f"lif_conv{conv_lif_idx + 1}",
                    "in_dim":    lif_dim,
                    "out_dim":   lif_dim,
                    "threshold": float(thr),
                    "reset_mechanism": getattr(m, "reset_mechanism", "subtract"),
                })
                conv_lif_idx += 1

    # Fix FCN layer ordering: for ModuleList-based FCN models, model.modules() visits all Linears before all LIFs, producing [L0, L1, LIF0, LIF1] instead of the correct interleaved [L0, LIF0, L1, LIF1].
    has_conv_pool = any(
        d["type"] in ("Conv2d", "DepthwiseConv2d", "AvgPool2d", "MaxPool2d")
        for d in layer_defs
    )
    if not has_conv_pool:
        linears = [d for d in layer_defs if d["type"] == "Linear"]
        lifs    = [d for d in layer_defs if d["type"] == "LIF"]
        if len(linears) == len(lifs):
            new_defs = []
            for lin, lif in zip(linears, lifs):
                lif["in_dim"]  = lin["out"]
                lif["out_dim"] = lin["out"]
                new_defs.append(lin)
                new_defs.append(lif)
            layer_defs = new_defs

    return state_dict, layer_defs

def standardize_checkpoint(model_class_path, ckpt_path, out_path,
                           model_kwargs=None, input_shape=None):
    """Load a model checkpoint, standardize it, and save to out_path.
    input_shape: (C, H, W) required for Conv2d models; None for FCN
    """
    module_name, class_name = model_class_path.rsplit(".", 1)
    try:
        mdl = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"[Error] cannot import model module '{module_name}': {e}")
    try:
        model_cls = getattr(mdl, class_name)
    except AttributeError:
        raise SystemExit(f"[Error] class '{class_name}' not found in module '{module_name}'")

    if model_kwargs:
        model = model_cls(**model_kwargs)
    else:
        model = model_cls()

    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = raw_ckpt
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        raw_sd = raw_ckpt["state_dict"]
    model.load_state_dict(raw_sd)


    passthrough = {}
    if isinstance(raw_ckpt, dict):
        for k in ("encoding", "input_is_binary"):
            if raw_ckpt.get(k) is not None:
                passthrough[k] = raw_ckpt[k]

    structure = detect_structure(model)
    print(f"[Info] Detected structure: {structure}")

    if structure["has_conv"]:
        if input_shape is None:
            raise ValueError(
                "Model has Conv2d layers but input_shape was not provided. "
                "Pass input_shape as (C, H, W), e.g. (2, 34, 34)."
            )
        _fold_bn_into_model(model)

    state_dict, layer_defs = standardize_state_dict(model, input_shape)

    print(f"[Info] layer_defs ({len(layer_defs)} layers):")
    for i, ld in enumerate(layer_defs):
        print(f"  [{i}] {ld['type']}  {ld.get('name', '')}")

    save_dict = {"state_dict": state_dict, "layer_defs": layer_defs, **passthrough}
    if passthrough:
        print(f"[Info] Preserved metadata: {passthrough}")
    torch.save(save_dict, out_path)
    print(f"[Done] Saved to: {out_path}")


def main():
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model class path, e.g. user_model.nmnist_small.FCSNN")
    parser.add_argument("--input", required=True, help="Input checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Output path (.pth)")
    parser.add_argument("--model-kwargs", default=None,
                        help='Model constructor args as JSON, e.g. \'{"input_dim": [2,34,34]}\'')
    parser.add_argument("--input-shape", default=None,
                        help="Input shape C,H,W e.g. 2,34,34 - required for CSNN models")
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None

    input_shape = None
    if args.input_shape:
        parts = [int(x.strip()) for x in args.input_shape.split(",")]
        if len(parts) != 3:
            print(f"[Error] --input-shape must be C,H,W (3 values), got {len(parts)}")
            sys.exit(1)
        input_shape = tuple(parts)

    standardize_checkpoint(args.model, args.input, args.output,
                           model_kwargs=model_kwargs, input_shape=input_shape)


if __name__ == "__main__":
    main()
