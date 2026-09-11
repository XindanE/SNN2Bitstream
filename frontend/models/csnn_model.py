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

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from frontend.models import _as_list
from frontend.models.neurons import PLIFLeaky


class CSNNNet(nn.Module):
    """Convolutional spiking network: (Conv -> Pool -> LIF) x N -> (FC -> LIF) x M -> output.
    Per-stage params (strides, betas, thresholds, reset_mechanisms, bitwidths) accept a scalar or a per-stage list
    """

    def __init__(self, input_dim, output_dim,
                 conv_channels=[16, 32], kernel_sizes=[5, 3],
                 paddings=None, strides=None,
                 pool_sizes=[2, 2], pool_types=None, pool_strides=None,
                 fc_units=[128], beta=0.9, timesteps=10, learn_beta=False,
                 threshold=1.0, dropout=0.0,
                 betas=None, thresholds=None, learn_thresholds=None,
                 reset_mechanisms=None, bitwidths=None,
                 use_bn=False, use_dsc=False, dsc_start_layer=0,
                 neuron="lif", init_tau=2.0, weight_init="default",
                 surrogate_grad="atan", surrogate_slope=1.0,
                 layer_seq=None, pool_before_lif=False
                 ):
        super().__init__()
        # weight_init: "default" = PyTorch's Conv2d/Linear init (kaiming_uniform a=sqrt5,
        # gain ~0.577); "kaiming" = kaiming_normal with relu gain (~1.41, ~2.4x larger).
        # Applied at the end of __init__.
        self.weight_init = str(weight_init).lower()

        n_conv = len(conv_channels)

        self.beta = beta
        self.timesteps = timesteps
        self.learn_beta = learn_beta
        self.threshold = threshold
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else None

        self.use_bn = use_bn
        self.use_dsc = use_dsc
        self.dsc_start_layer = dsc_start_layer if use_dsc else n_conv

        self.neuron = str(neuron).lower()
        self.init_tau = float(init_tau)

        # Build the surrogate gradient from the config
        _sg = str(surrogate_grad).lower(); _sl = float(surrogate_slope)
        if _sg == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=_sl)
        elif _sg in ("ste", "straight_through_estimator"):
            spike_grad = surrogate.straight_through_estimator()
        else:  # atan (default)
            spike_grad = surrogate.atan(alpha=_sl)
        self.output_mem = False
        self.layer_defs = []

        # Sequence-driven mode: build the exact configured layer order (conv->lif->pool, with
        # consecutive pools and pool-of-spikes)
        self._seq_mode = layer_seq is not None
        self.pool_before_lif = bool(pool_before_lif)
        if self._seq_mode:
            self._build_from_seq(input_dim, output_dim, layer_seq, spike_grad)
            self._apply_weight_init()
            return

        in_ch, cur_h, cur_w = input_dim

        if paddings is None:    paddings    = [0] * n_conv
        if pool_types is None:  pool_types  = ["avg"] * n_conv
        strides       = _as_list(strides, n_conv, 1)
        pool_strides_ = _as_list(pool_strides, n_conv, None)
        for i in range(n_conv):
            if pool_strides_[i] is None:
                pool_strides_[i] = pool_sizes[i]
        conv_betas      = _as_list(betas,            n_conv, beta)
        conv_thrs       = _as_list(thresholds,       n_conv, threshold)
        conv_learn_thrs = _as_list(learn_thresholds, n_conv, False)
        # Per-stage reset mechanism and weight bitwidth.
        conv_resets = _as_list(reset_mechanisms, n_conv, "subtract")
        conv_bws    = _as_list(bitwidths,        n_conv, None)

        fc_beta, fc_thr, fc_learn_thr = beta, threshold, False
        fc_reset, fc_bw = "subtract", None
        if isinstance(betas, list) and len(betas) > n_conv:
            fc_beta = betas[n_conv]
        if isinstance(thresholds, list) and len(thresholds) > n_conv:
            fc_thr = thresholds[n_conv]
        if isinstance(learn_thresholds, list) and len(learn_thresholds) > n_conv:
            fc_learn_thr = learn_thresholds[n_conv]
        if isinstance(reset_mechanisms, list) and len(reset_mechanisms) > n_conv:
            fc_reset = reset_mechanisms[n_conv]
        if isinstance(bitwidths, list) and len(bitwidths) > n_conv:
            fc_bw = bitwidths[n_conv]

        self.pool_layers = nn.ModuleList()
        self.conv_lif_layers = nn.ModuleList()
        self._stage_is_dsc = []

        self.conv_layers = nn.ModuleList()
        self.conv_bn_layers = nn.ModuleList()
        self.dw_conv_layers = nn.ModuleList()
        self.pw_conv_layers = nn.ModuleList()
        self.dw_bn_layers = nn.ModuleList()
        self.pw_bn_layers = nn.ModuleList()

        for i in range(n_conv):
            out_ch = conv_channels[i]
            k = kernel_sizes[i]
            pad = paddings[i]
            stride = strides[i]
            pool_size = pool_sizes[i]
            pool_stride = pool_strides_[i]
            pool_type = str(pool_types[i]).lower()
            stage_beta = conv_betas[i]
            stage_thr = conv_thrs[i]
            stage_learn_thr = conv_learn_thrs[i]
            stage_reset = conv_resets[i]
            stage_bw = conv_bws[i]

            use_dsc_this_layer = self.use_dsc and i >= self.dsc_start_layer
            self._stage_is_dsc.append(use_dsc_this_layer)

            if use_dsc_this_layer:
                dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, groups=in_ch, padding=pad, bias=True)
                dw._bitwidth = stage_bw
                self.dw_conv_layers.append(dw)
                if self.use_bn: self.dw_bn_layers.append(nn.BatchNorm2d(in_ch))
                dw_out_h = (cur_h + 2 * pad - k) // stride + 1
                dw_out_w = (cur_w + 2 * pad - k) // stride + 1
                self.layer_defs.append({
                    "type": "DepthwiseConv2d", "name": f"dw_conv{i+1}",
                    "channels": in_ch, "kernel_size": k,
                    "stride": stride, "padding": pad,
                    "in_h": cur_h, "in_w": cur_w,
                    "out_h": dw_out_h, "out_w": dw_out_w,
                    "bitwidth": stage_bw,
                })

                pw = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=True)
                pw._bitwidth = stage_bw
                self.pw_conv_layers.append(pw)
                if self.use_bn: self.pw_bn_layers.append(nn.BatchNorm2d(out_ch))
                self.layer_defs.append({
                    "type": "Conv2d", "name": f"pw_conv{i+1}",
                    "in_channels": in_ch, "out_channels": out_ch, "kernel_size": 1,
                    "out_h": dw_out_h, "out_w": dw_out_w,
                    "bitwidth": stage_bw,
                })

                conv_out_h, conv_out_w = dw_out_h, dw_out_w
            else:
                conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=True)
                conv._bitwidth = stage_bw  # per-layer QAT bitwidth, read by convert_to_qat_model
                self.conv_layers.append(conv)
                if self.use_bn: self.conv_bn_layers.append(nn.BatchNorm2d(out_ch))

                conv_out_h = (cur_h + 2 * pad - k) // stride + 1
                conv_out_w = (cur_w + 2 * pad - k) // stride + 1
                self.layer_defs.append({
                    "type": "Conv2d", "name": f"conv{i+1}",
                    "in_channels": int(in_ch), "out_channels": int(out_ch), "kernel_size": k,
                    "stride": stride, "padding": pad,
                    "in_h": cur_h, "in_w": cur_w,
                    "out_h": conv_out_h, "out_w": conv_out_w,
                    "bitwidth": stage_bw,
                })

            skip_pool = (pool_size in (0, None)) or (pool_size == 1 and pool_stride == 1)
            if skip_pool:
                self.pool_layers.append(nn.Identity())
                pool_out_h, pool_out_w = conv_out_h, conv_out_w
            else:
                if pool_type == "max":
                    self.pool_layers.append(nn.MaxPool2d(pool_size, stride=pool_stride))
                    pool_type_name = "MaxPool2d"
                else:
                    self.pool_layers.append(nn.AvgPool2d(pool_size, stride=pool_stride))
                    pool_type_name = "AvgPool2d"
                pool_out_h = (conv_out_h - pool_size) // pool_stride + 1
                pool_out_w = (conv_out_w - pool_size) // pool_stride + 1
                self.layer_defs.append({
                    "type": pool_type_name, "name": f"pool{i+1}",
                    "kernel_size": pool_size, "stride": pool_stride, "pool_type": pool_type,
                    "out_h": pool_out_h, "out_w": pool_out_w,
                })

            lif = self._make_lif(stage_beta, stage_thr, spike_grad,
                                 learn_threshold=stage_learn_thr, reset_mechanism=stage_reset)
            self.conv_lif_layers.append(lif)
            self.layer_defs.append({
                "type": "LIF", "name": f"lif_conv{i+1}",
                "beta": stage_beta, "threshold": stage_thr,
                "reset_mechanism": stage_reset,
                "in_dim": out_ch * pool_out_h * pool_out_w,
                "out_dim": out_ch * pool_out_h * pool_out_w,
            })

            in_ch = out_ch
            cur_h, cur_w = pool_out_h, pool_out_w

        self.fc_layers = nn.ModuleList()
        self.fc_lif_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()

        in_dim = in_ch * cur_h * cur_w

        for i, hidden in enumerate(fc_units):
            self.fc_layers.append(nn.Linear(in_dim, hidden))
            if self.use_bn:
                self.fc_bn_layers.append(nn.BatchNorm1d(hidden))
            self.layer_defs.append({
                "type": "Linear", "name": f"fc{i+1}",
                "in": in_dim, "out": hidden
            })
            self.fc_lif_layers.append(self._make_lif(beta, threshold, spike_grad))
            self.layer_defs.append({
                "type": "LIF", "name": f"lif_fc{i+1}",
                "beta": beta, "threshold": threshold,
                "in_dim": hidden, "out_dim": hidden,
            })
            in_dim = hidden

        self.fc_out = nn.Linear(in_dim, output_dim)
        self.fc_out._bitwidth = fc_bw
        self.layer_defs.append({
            "type": "Linear", "name": "fc_out",
            "in": in_dim, "out": output_dim,
            "bitwidth": fc_bw,
        })

        self.lif_out = self._make_lif(fc_beta, fc_thr, spike_grad,
                                      learn_threshold=fc_learn_thr, reset_mechanism=fc_reset)
        self.layer_defs.append({
            "type": "LIF", "name": "lif_out",
            "beta": fc_beta, "threshold": fc_thr,
            "reset_mechanism": fc_reset,
            "in_dim": output_dim, "out_dim": output_dim,
        })

        self._apply_weight_init()

    def _apply_weight_init(self):
        # "default" leaves PyTorch's built-in init. kaiming/xavier follow the configured
        # weight_init; the choice changes training dynamics, so it must be honored.
        if self.weight_init == "kaiming":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif self.weight_init == "xavier":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _make_lif(self, beta, threshold, spike_grad,
                  learn_threshold=False, reset_mechanism="subtract"):
        """Build one spiking neuron"""
        if self.neuron == "plif":
            return PLIFLeaky(init_tau=self.init_tau, threshold=threshold, spike_grad=spike_grad,
                             learn_threshold=bool(learn_threshold), reset_mechanism=reset_mechanism)
        return snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad,
                         learn_beta=self.learn_beta, learn_threshold=bool(learn_threshold),
                         reset_mechanism=reset_mechanism)

    def _build_from_seq(self, input_dim, output_dim, layer_seq, spike_grad):
        """Build the model from an explicit layer sequence"""
        self.conv_layers    = nn.ModuleList(); self.conv_bn_layers = nn.ModuleList()
        self.dw_conv_layers = nn.ModuleList(); self.dw_bn_layers   = nn.ModuleList()
        self.pw_conv_layers = nn.ModuleList(); self.pw_bn_layers   = nn.ModuleList()
        self.pools          = nn.ModuleList()
        self.conv_lif_layers = nn.ModuleList()
        self.fc_layers      = nn.ModuleList(); self.fc_bn_layers = nn.ModuleList()
        self.fc_lif_layers  = nn.ModuleList()
        self.fc_out = None; self.lif_out = None
        self._seq = []

        def _lif_of(spec):
            return self._make_lif(spec.get("beta", self.beta),
                                  spec.get("threshold", self.threshold), spike_grad,
                                  learn_threshold=spec.get("learn_threshold", False),
                                  reset_mechanism=spec.get("reset_mechanism", spec.get("reset", "subtract")))

        in_ch, cur_h, cur_w = input_dim
        n_linear = sum(1 for s in layer_seq if s["type"] == "linear")
        seen_linear = 0
        flat = False

        def _emit_conv_lif(spec):
            li = len(self.conv_lif_layers); self.conv_lif_layers.append(_lif_of(spec))
            self._seq.append(("lif_conv", li))
            self.layer_defs.append({"type": "LIF", "name": f"lif_conv{li+1}",
                "beta": spec.get("beta", self.beta), "threshold": spec.get("threshold", self.threshold),
                "reset_mechanism": spec.get("reset_mechanism", spec.get("reset", "subtract")),
                "in_dim": in_ch * cur_h * cur_w, "out_dim": in_ch * cur_h * cur_w})

        # pool_before_lif
        pending_lif = None
        for spec in layer_seq:
            t = spec["type"]
            if t == "conv":
                if pending_lif is not None:  # previous conv's deferred LIF fires before this conv
                    _emit_conv_lif(pending_lif); pending_lif = None
                bw = spec.get("bitwidth"); k = spec["kernel_size"]
                stride = spec.get("stride", 1); pad = spec.get("padding", 0)
                out_ch = spec["out_channels"]
                oh = (cur_h + 2 * pad - k) // stride + 1
                ow = (cur_w + 2 * pad - k) // stride + 1
                if spec.get("use_dsc", False):
                    dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, groups=in_ch, padding=pad, bias=True)
                    dw._bitwidth = bw; di = len(self.dw_conv_layers); self.dw_conv_layers.append(dw)
                    if self.use_bn: self.dw_bn_layers.append(nn.BatchNorm2d(in_ch))
                    self._seq.append(("dwconv", di))
                    self.layer_defs.append({"type": "DepthwiseConv2d", "name": f"dw_conv{di+1}",
                        "channels": in_ch, "kernel_size": k, "stride": stride, "padding": pad,
                        "in_h": cur_h, "in_w": cur_w, "out_h": oh, "out_w": ow, "bitwidth": bw})
                    pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)
                    pw._bitwidth = bw; pi = len(self.pw_conv_layers); self.pw_conv_layers.append(pw)
                    if self.use_bn: self.pw_bn_layers.append(nn.BatchNorm2d(out_ch))
                    self._seq.append(("pwconv", pi))
                    self.layer_defs.append({"type": "Conv2d", "name": f"pw_conv{pi+1}",
                        "in_channels": in_ch, "out_channels": out_ch, "kernel_size": 1,
                        "in_h": oh, "in_w": ow, "out_h": oh, "out_w": ow, "bitwidth": bw})
                else:
                    conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=True)
                    conv._bitwidth = bw; ci = len(self.conv_layers); self.conv_layers.append(conv)
                    if self.use_bn: self.conv_bn_layers.append(nn.BatchNorm2d(out_ch))
                    self._seq.append(("conv", ci))
                    self.layer_defs.append({"type": "Conv2d", "name": f"conv{ci+1}",
                        "in_channels": int(in_ch), "out_channels": int(out_ch), "kernel_size": k,
                        "stride": stride, "padding": pad, "in_h": cur_h, "in_w": cur_w,
                        "out_h": oh, "out_w": ow, "bitwidth": bw})
                in_ch, cur_h, cur_w = out_ch, oh, ow
                if self.pool_before_lif:
                    pending_lif = spec  # emit after the following pools (pool->lif)
                else:
                    _emit_conv_lif(spec)
            elif t == "pool":
                pt = str(spec.get("pool_type", "avg")).lower()
                ks = spec["kernel_size"]; ps = spec.get("stride", ks)
                if pt == "max":
                    self.pools.append(nn.MaxPool2d(ks, stride=ps)); pname = "MaxPool2d"
                else:
                    self.pools.append(nn.AvgPool2d(ks, stride=ps)); pname = "AvgPool2d"
                pj = len(self.pools) - 1
                cur_h = (cur_h - ks) // ps + 1; cur_w = (cur_w - ks) // ps + 1
                self._seq.append(("pool", pj))
                self.layer_defs.append({"type": pname, "name": f"pool{pj+1}",
                    "kernel_size": ks, "stride": ps, "pool_type": pt, "out_h": cur_h, "out_w": cur_w})
            elif t == "linear":
                if pending_lif is not None:  # last conv's deferred LIF fires before the classifier
                    _emit_conv_lif(pending_lif); pending_lif = None
                if not flat:
                    self._seq.append(("flatten", 0)); flat = True
                    in_dim = in_ch * cur_h * cur_w
                bw = spec.get("bitwidth"); out = spec["out"]
                seen_linear += 1
                is_out = spec.get("is_output", seen_linear == n_linear)
                if is_out:
                    self.fc_out = nn.Linear(in_dim, out); self.fc_out._bitwidth = bw
                    self._seq.append(("fc_out", 0))
                    self.layer_defs.append({"type": "Linear", "name": "fc_out",
                        "in": in_dim, "out": out, "bitwidth": bw})
                    self.lif_out = _lif_of(spec)
                    self._seq.append(("lif_out", 0))
                    self.layer_defs.append({"type": "LIF", "name": "lif_out",
                        "beta": spec.get("beta", self.beta), "threshold": spec.get("threshold", self.threshold),
                        "reset_mechanism": spec.get("reset_mechanism", spec.get("reset", "subtract")),
                        "in_dim": out, "out_dim": out})
                else:
                    fi = len(self.fc_layers); fc = nn.Linear(in_dim, out); fc._bitwidth = bw
                    self.fc_layers.append(fc)
                    if self.use_bn: self.fc_bn_layers.append(nn.BatchNorm1d(out))
                    self._seq.append(("fc", fi))
                    self.layer_defs.append({"type": "Linear", "name": f"fc{fi+1}", "in": in_dim, "out": out, "bitwidth": bw})
                    lj = len(self.fc_lif_layers); self.fc_lif_layers.append(_lif_of(spec))
                    self._seq.append(("lif_fc", lj))
                    self.layer_defs.append({"type": "LIF", "name": f"lif_fc{lj+1}",
                        "beta": spec.get("beta", self.beta), "threshold": spec.get("threshold", self.threshold),
                        "in_dim": out, "out_dim": out})
                in_dim = out
            else:
                raise ValueError(f"Unknown layer_seq type: {t!r}")
        if pending_lif is not None:
            _emit_conv_lif(pending_lif)

        if self.fc_out is None:
            raise ValueError("layer_seq must contain a terminal linear (classifier head).")

    def _forward_seq(self, x):
        B, T = x.shape[0], x.shape[1]
        conv_mems = [lif.init_leaky() for lif in self.conv_lif_layers]
        fc_mems   = [lif.init_leaky() for lif in self.fc_lif_layers]
        out_mem   = self.lif_out.init_leaky()
        spk_rec, mem_rec = [], ([] if self.output_mem else None)
        for t in range(T):
            z = x[:, t]
            for op, idx in self._seq:
                if op == "conv":
                    z = self.conv_layers[idx](z)
                    if self.use_bn: z = self.conv_bn_layers[idx](z)
                elif op == "dwconv":
                    z = self.dw_conv_layers[idx](z)
                    if self.use_bn: z = self.dw_bn_layers[idx](z)
                elif op == "pwconv":
                    z = self.pw_conv_layers[idx](z)
                    if self.use_bn: z = self.pw_bn_layers[idx](z)
                elif op == "pool":
                    z = self.pools[idx](z)
                elif op == "lif_conv":
                    z, conv_mems[idx] = self.conv_lif_layers[idx](z, conv_mems[idx])
                elif op == "flatten":
                    z = z.view(B, -1)
                elif op == "fc":
                    z = self.fc_layers[idx](z)
                    if self.use_bn: z = self.fc_bn_layers[idx](z)
                    if self.dropout is not None: z = self.dropout(z)
                elif op == "lif_fc":
                    z, fc_mems[idx] = self.fc_lif_layers[idx](z, fc_mems[idx])
                elif op == "fc_out":
                    z = self.fc_out(z)
                elif op == "lif_out":
                    z, out_mem = self.lif_out(z, out_mem)
            spk_rec.append(z)
            if self.output_mem: mem_rec.append(out_mem)
        spikes = torch.stack(spk_rec, dim=1)
        if self.output_mem:
            return spikes, torch.stack(mem_rec, dim=1)
        return spikes

    def forward(self, x):
        """Input: x of shape [B, T, C, H, W] (batch-first).
        Output: spike tensor of shape [B, T, num_classes]; if output_mem also membrane.
        """
        if getattr(self, "_seq_mode", False):
            return self._forward_seq(x)
        B, T = x.shape[0], x.shape[1]
        conv_mems = [lif.init_leaky() for lif in self.conv_lif_layers]
        fc_mems   = [lif.init_leaky() for lif in self.fc_lif_layers]
        out_mem = self.lif_out.init_leaky()

        spk_rec = []
        mem_rec = [] if self.output_mem else None
        for t in range(T):
            z = x[:, t]

            std_idx, dsc_idx = 0, 0
            for i, (pool, lif) in enumerate(zip(self.pool_layers, self.conv_lif_layers)):
                if self._stage_is_dsc[i]:
                    z = self.dw_conv_layers[dsc_idx](z)
                    if self.use_bn: z = self.dw_bn_layers[dsc_idx](z)
                    z = self.pw_conv_layers[dsc_idx](z)
                    if self.use_bn: z = self.pw_bn_layers[dsc_idx](z)
                    dsc_idx += 1
                else:
                    z = self.conv_layers[std_idx](z)
                    if self.use_bn: z = self.conv_bn_layers[std_idx](z)
                    std_idx += 1
                z = pool(z)
                z, conv_mems[i] = lif(z, conv_mems[i])
            z = z.view(B, -1)

            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.fc_lif_layers)):
                z = fc(z)
                if self.use_bn: z = self.fc_bn_layers[i](z)
                if self.dropout is not None: z = self.dropout(z)
                z, fc_mems[i] = lif(z, fc_mems[i])

            z = self.fc_out(z)
            spk, out_mem = self.lif_out(z, out_mem)
            spk_rec.append(spk)
            if self.output_mem:
                mem_rec.append(out_mem)

        spikes = torch.stack(spk_rec, dim=1)
        if self.output_mem:
            return spikes, torch.stack(mem_rec, dim=1)
        return spikes
