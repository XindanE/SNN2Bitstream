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

"""Profile per-layer LIF membrane |mem|_max for fixed-point width sizing.

Rebuilds the network from a standardized checkpoint (state_dict + layer_defs, the format export_ir.py consumes) and runs the test set, recording max |membrane| per LIF layer. The output JSON feeds converter.py --lif-mem-profile, which sizes each lif_mem_t from the measured max times a safety factor. Entries are emitted in IR LIF order so the converter maps them positionally.

Usage:
    python tools/profile_lif_mem.py checkpoints/<project>.pth \
        --toml configs/<project>.toml \
        --out logs/lif_mem_profile_<project>.json \
        --max-samples 2000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import toml
import snntorch as snn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from frontend.train_model import get_loaders, encode_input_fcn, encode_input_csnn
from frontend.export_ir import _fold_bn_into_weights


def _make_leaky(beta, threshold, reset_mechanism):
    """snn.Leaky for inference; older snntorch drops reset_mechanism kwarg."""
    try:
        return snn.Leaky(beta=beta, threshold=threshold,
                         reset_mechanism=reset_mechanism, init_hidden=False)
    except TypeError:
        return snn.Leaky(beta=beta, threshold=threshold, init_hidden=False)


def build_layers(sd, layer_defs, device, default_beta=0.9, default_threshold=1.0):
    """Rebuild an ordered layer list from standardized state_dict + layer_defs.

    BN is already folded in a standardized checkpoint; _fold_bn_into_weights is
    a no-op if no BN keys remain. Returns a list of (kind, module, name) where
    kind in {conv, dwconv, pwconv, pool, linear, lif}, mirroring export_ir's walk.
    """
    sd = _fold_bn_into_weights(sd)
    layers = []
    conv_i = dw_i = pw_i = fc_i = conv_lif_i = fc_lif_i = 0

    for ld in layer_defs:
        t = ld["type"]

        if t == "DepthwiseConv2d":
            w = sd[f"dw_conv_layers.{dw_i}.weight"]
            b = sd.get(f"dw_conv_layers.{dw_i}.bias")
            dw_i += 1
            ch, k = w.shape[0], w.shape[2]
            m = nn.Conv2d(ch, ch, k, stride=ld.get("stride", 1),
                          padding=ld.get("padding", 0), groups=ch, bias=b is not None)
            m.weight.data = w
            if b is not None:
                m.bias.data = b
            layers.append(("dwconv", m.to(device).eval(), ld.get("name")))

        elif t == "Conv2d":
            name = ld.get("name", "")
            if name.startswith("pw_conv"):
                w = sd[f"pw_conv_layers.{pw_i}.weight"]
                b = sd.get(f"pw_conv_layers.{pw_i}.bias")
                pw_i += 1
                kind = "pwconv"
            else:
                w = sd[f"conv_layers.{conv_i}.weight"]
                b = sd.get(f"conv_layers.{conv_i}.bias")
                conv_i += 1
                kind = "conv"
            oc, ic, k = w.shape[0], w.shape[1], w.shape[2]
            m = nn.Conv2d(ic, oc, k, stride=ld.get("stride", 1),
                          padding=ld.get("padding", 0), bias=b is not None)
            m.weight.data = w
            if b is not None:
                m.bias.data = b
            layers.append((kind, m.to(device).eval(), name))

        elif t in ("AvgPool2d", "MaxPool2d"):
            k = ld["kernel_size"]
            s = ld.get("stride", k)
            p = ld.get("padding", 0)
            m = (nn.AvgPool2d(k, stride=s, padding=p) if t == "AvgPool2d"
                 else nn.MaxPool2d(k, stride=s, padding=p))
            layers.append(("pool", m, ld.get("name")))

        elif t == "Linear":
            name = ld["name"]
            if name == "fc_out":
                w = sd["fc_out.weight"]
                b = sd.get("fc_out.bias")
            else:
                w = sd[f"fc_layers.{fc_i}.weight"]
                b = sd.get(f"fc_layers.{fc_i}.bias")
                fc_i += 1
            out_f, in_f = w.shape
            m = nn.Linear(in_f, out_f, bias=b is not None)
            m.weight.data = w
            if b is not None:
                m.bias.data = b
            layers.append(("linear", m.to(device).eval(), name))

        elif t == "LIF":
            name = ld["name"]
            if name.startswith("lif_conv"):
                beta_key = f"conv_lif_layers.{conv_lif_i}.beta"
                conv_lif_i += 1
            elif name.startswith("lif_fc"):
                beta_key = f"fc_lif_layers.{fc_lif_i}.beta"
                fc_lif_i += 1
            elif name == "lif_out":
                beta_key = "lif_out.beta"
            else:
                beta_key = None
            beta = float(sd[beta_key]) if beta_key and beta_key in sd else default_beta
            thr = float(ld.get("threshold", default_threshold))
            reset = ld.get("reset_mechanism", "subtract")
            layers.append(("lif", _make_leaky(beta, thr, reset).to(device), name))

    return layers


def measure_lif_mem(layers, loader, model_kind, timesteps, encoding, device, max_samples):
    """Run the test set, return (stats, accuracy_pct).

    stats: {lif_name: {max_abs, p99, p999, p9999, n_obs}} in layer order.
    accuracy is a forward-correctness sanity check (float, should track golden).
    """
    lif_names = [name for (k, _, name) in layers if k == "lif"]
    acc = {n: {"max_abs": 0.0, "values": []} for n in lif_names}

    correct = seen = 0
    with torch.no_grad():
        for data, targets in loader:
            if seen >= max_samples:
                break
            data = data.to(device)
            targets = targets.to(device)
            if model_kind == "fcn":
                x = encode_input_fcn(data, timesteps, encoding)   # [T,B,D]
                B = x.size(1)
                step = lambda t: x[t]
            else:
                x = encode_input_csnn(data, timesteps, encoding)  # [B,T,C,H,W]
                B = x.size(0)
                step = lambda t: x[:, t]

            mems = {name: m.init_leaky() for (k, m, name) in layers if k == "lif"}
            spk_sum = None
            for t in range(timesteps):
                z = step(t)
                for kind, m, name in layers:
                    if kind == "linear" and z.dim() > 2:
                        z = z.reshape(z.size(0), -1)
                    if kind == "lif":
                        spk, mems[name] = m(z, mems[name])
                        mabs = mems[name].detach().abs()
                        acc[name]["max_abs"] = max(acc[name]["max_abs"], float(mabs.max()))
                        flat = mabs.flatten()
                        if flat.numel() > 1024:
                            idx = torch.randperm(flat.numel(), device=device)[:1024]
                            flat = flat[idx]
                        acc[name]["values"].append(flat.cpu().numpy())
                        z = spk
                    else:
                        z = m(z)
                spk_sum = z if spk_sum is None else spk_sum + z

            correct += int((spk_sum.argmax(-1) == targets).sum())
            seen += B

    out = {}
    for name, s in acc.items():
        if s["values"]:
            allv = np.concatenate(s["values"])
            out[name] = {
                "max_abs": float(s["max_abs"]),
                "p99":   float(np.percentile(allv, 99)),
                "p999":  float(np.percentile(allv, 99.9)),
                "p9999": float(np.percentile(allv, 99.99)),
                "n_obs": int(allv.size),
            }
        else:
            out[name] = {"max_abs": 0.0, "p99": 0.0, "p999": 0.0, "p9999": 0.0, "n_obs": 0}
    accuracy = 100.0 * correct / seen if seen else 0.0
    return out, accuracy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="Standardized checkpoint (state_dict + layer_defs)")
    ap.add_argument("--toml", required=True,
                    help="Training config TOML - supplies dataset, timesteps, encoding")
    ap.add_argument("--out", required=True, help="Output profile JSON path")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "layer_defs" not in ckpt:
        raise ValueError(
            f"{args.checkpoint} is not a standardized checkpoint (need 'state_dict' "
            f"+ 'layer_defs'). Run standardize_model.py / train via configs/ first.")
    sd = ckpt["state_dict"]
    layer_defs = ckpt["layer_defs"]

    cfg = toml.load(args.toml)
    dataset_cfg = cfg["dataset"]
    params = cfg["model_template"]["params"]
    timesteps = int(params["timesteps"])
    encoding = params.get("encoding", "repeat")

    model_kind = "csnn" if any(
        ld["type"] in ("Conv2d", "DepthwiseConv2d") for ld in layer_defs) else "fcn"

    device = torch.device(args.device)
    _, test_loader, _, _ = get_loaders(dataset_cfg, args.batch, timesteps, encoding)
    layers = build_layers(sd, layer_defs, device)
    n_lif = sum(1 for k, _, _ in layers if k == "lif")
    print(f"[Info] {model_kind} model, {len(layers)} layers ({n_lif} LIF), "
          f"T={timesteps}, encoding={encoding}, device={device}")

    print(f"[Info] profiling on max {args.max_samples} test samples...")
    stats, accuracy = measure_lif_mem(layers, test_loader, model_kind, timesteps,
                              encoding, device, args.max_samples)

    print(f"\n[CHECK] float forward accuracy = {accuracy:.2f}% "
          f"(should track the GCC golden for this config)")
    print("\nPer-LIF |mem| statistics:")
    print(f"  {'name':12s} {'max':>10s} {'p99.99':>10s} {'p99.9':>10s} {'p99':>10s}")
    for name, s in stats.items():
        print(f"  {name:12s} {s['max_abs']:10.4f} {s['p9999']:10.4f} "
              f"{s['p999']:10.4f} {s['p99']:10.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Done] saved {len(stats)} LIF entries to {args.out}")


if __name__ == "__main__":
    main()
