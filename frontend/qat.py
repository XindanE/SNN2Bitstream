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

# QAT infrastructure: fake-quantize classes, model conversion, BN folding.
import torch
import torch.nn as nn


class FakeQuantize(torch.autograd.Function):
    """Straight-through estimator for fake quantization during training."""
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged.
        return grad_output, None, None, None, None


class LearnedFakeQuantize(nn.Module):
    """Fake quantization with EMA-updated scale for stable QAT training."""
    def __init__(self, bit_width=8, symmetric=True, ema_momentum=0.1):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.ema_momentum = ema_momentum

        if bit_width == 1:
            # Binary quantization: {-1, +1}
            self.qmin = -1
            self.qmax = 1
        elif symmetric:
            self.qmin = -(2**(bit_width-1))
            self.qmax = 2**(bit_width-1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bit_width - 1

        # EMA buffers for scale; scalar for stability.
        self.register_buffer('ema_scale', torch.ones(1))
        self.register_buffer('ema_min', torch.zeros(1))
        self.register_buffer('ema_max', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                cur_min = x.detach().min().item()
                cur_max = x.detach().max().item()
                if self.num_batches_tracked == 0:
                    new_min = cur_min
                    new_max = cur_max
                else:
                    new_min = self.ema_min.item() * (1 - self.ema_momentum) + cur_min * self.ema_momentum
                    new_max = self.ema_max.item() * (1 - self.ema_momentum) + cur_max * self.ema_momentum
                self.ema_min.fill_(new_min)
                self.ema_max.fill_(new_max)
                self.num_batches_tracked.add_(1)
                if self.symmetric:
                    max_abs = max(abs(new_min), abs(new_max))
                    new_scale = max(max_abs / self.qmax, 1e-8)
                else:
                    new_scale = max((new_max - new_min) / (self.qmax - self.qmin), 1e-8)
                self.ema_scale.fill_(new_scale)
        # Scale is detached: gradients flow through x, not through the scale estimate.
        scale = self.ema_scale.detach()
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = self.qmin - torch.round(self.ema_min.detach() / scale)
        return FakeQuantize.apply(x, scale, zero_point, self.qmin, self.qmax)


class QATLinear(nn.Module):
    """Linear layer with EMA-based weight quantization for QAT."""
    def __init__(self, linear_layer, bit_width=8):
        super().__init__()
        self.linear = linear_layer
        self.bit_width = bit_width
        self.weight_quantizer = LearnedFakeQuantize(
            bit_width=bit_width, symmetric=True, ema_momentum=0.1
        )
        if linear_layer.bias is not None:
            self.bias_quantizer = LearnedFakeQuantize(
                bit_width=bit_width, symmetric=False, ema_momentum=0.1
            )
        else:
            self.bias_quantizer = None

    def forward(self, x):
        w_q = self.weight_quantizer(self.linear.weight)
        b_q = self.bias_quantizer(self.linear.bias) if self.bias_quantizer is not None else None
        return nn.functional.linear(x, w_q, b_q)


class QATConv2d(nn.Module):
    """Conv2d layer with EMA-based weight quantization for QAT."""
    def __init__(self, conv_layer, bit_width=8):
        super().__init__()
        self.conv = conv_layer
        self.bit_width = bit_width
        self.weight_quantizer = LearnedFakeQuantize(
            bit_width=bit_width, symmetric=True, ema_momentum=0.1
        )
        if conv_layer.bias is not None:
            self.bias_quantizer = LearnedFakeQuantize(
                bit_width=bit_width, symmetric=False, ema_momentum=0.1
            )
        else:
            self.bias_quantizer = None

    def forward(self, x):
        w_q = self.weight_quantizer(self.conv.weight)
        b_q = self.bias_quantizer(self.conv.bias) if self.bias_quantizer is not None else None
        return nn.functional.conv2d(
            x, w_q, b_q,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )


class QATActivation(nn.Module):
    """Quantize activations after LIF layer for full QAT."""
    def __init__(self, bit_width=8):
        super().__init__()
        self.quantizer = LearnedFakeQuantize(
            bit_width=bit_width,
            symmetric=False,  # Activations are typically non-negative after LIF
            ema_momentum=0.1
        )

    def forward(self, x):
        return self.quantizer(x)


# (model_attr, orig_cls, qat_cls, inner_attr, prefix_fmt). Add an entry to support another layer type.
_QAT_MODULELIST_MAP = [
    ('layers',         nn.Linear, QATLinear, 'linear', 'layer_{}'),
    ('conv_layers',    nn.Conv2d, QATConv2d, 'conv',   'conv_{}'),
    ('dw_conv_layers', nn.Conv2d, QATConv2d, 'conv',   'dw_conv_{}'),
    ('pw_conv_layers', nn.Conv2d, QATConv2d, 'conv',   'pw_conv_{}'),
    ('fc_layers',      nn.Linear, QATLinear, 'linear', 'fc_{}'),
]


def convert_to_qat_model(model, bit_width=8, quantize_activations=True):
    """Wrap Linear/Conv2d layers with QAT and optionally add activation quantization.

    A layer carrying a ._bitwidth attribute is fake-quantized at that bitwidth; layers without it fall back to the global bit_width. Activation quantizers stay at the global bit_width.
    """
    def _bits(layer):
        return getattr(layer, '_bitwidth', None) or bit_width
    for attr, orig_cls, qat_cls, _, _ in _QAT_MODULELIST_MAP:
        lst = getattr(model, attr, None)
        if isinstance(lst, nn.ModuleList):
            for i, layer in enumerate(lst):
                if isinstance(layer, orig_cls):
                    lst[i] = qat_cls(layer, _bits(layer))
    if hasattr(model, 'fc_out') and isinstance(model.fc_out, nn.Linear):
        model.fc_out = QATLinear(model.fc_out, _bits(model.fc_out))
    if quantize_activations and hasattr(model, 'neurons') and not hasattr(model, 'act_quantizers'):
        model.act_quantizers = nn.ModuleList([QATActivation(bit_width) for _ in model.neurons])
        model._use_act_quant = True
    return model


def _extract_qat_layer_scales(layer, prefix, qat_scales):
    """Extract QAT scales from a QATLinear or QATConv2d layer."""
    w_scale = layer.weight_quantizer.ema_scale.item()
    qat_scales[f"{prefix}_weight_scale"] = w_scale
    qat_scales[f"{prefix}_weight_symmetric"] = True
    qat_scales[f"{prefix}_weight_zero_point"] = 0
    if layer.bias_quantizer is not None:
        b_scale = layer.bias_quantizer.ema_scale.item()
        b_min = layer.bias_quantizer.ema_min.item()
        bias_zero_point = int(round(-b_min / b_scale)) if b_scale > 0 else 0
        qat_scales[f"{prefix}_bias_scale"] = b_scale
        qat_scales[f"{prefix}_bias_symmetric"] = False
        qat_scales[f"{prefix}_bias_zero_point"] = bias_zero_point


def extract_qat_scales(model):
    """Extract learned quantization scales from QAT model."""
    qat_scales = {}
    for attr, _, qat_cls, _, prefix_fmt in _QAT_MODULELIST_MAP:
        lst = getattr(model, attr, None)
        if isinstance(lst, nn.ModuleList):
            for i, layer in enumerate(lst):
                if isinstance(layer, qat_cls):
                    _extract_qat_layer_scales(layer, prefix_fmt.format(i), qat_scales)
    if hasattr(model, 'fc_out') and isinstance(model.fc_out, QATLinear):
        _extract_qat_layer_scales(model.fc_out, "fc_out", qat_scales)
    return qat_scales


def convert_from_qat_model(model):
    """Unwrap QAT layers back to regular Linear/Conv2d for export."""
    for attr, _, qat_cls, inner_attr, _ in _QAT_MODULELIST_MAP:
        lst = getattr(model, attr, None)
        if isinstance(lst, nn.ModuleList):
            for i, layer in enumerate(lst):
                if isinstance(layer, qat_cls):
                    lst[i] = getattr(layer, inner_attr)
    if hasattr(model, 'fc_out') and isinstance(model.fc_out, QATLinear):
        model.fc_out = model.fc_out.linear
    for attr in ('act_quantizers', '_use_act_quant'):
        if hasattr(model, attr):
            delattr(model, attr)
    return model


# (bn_attr, layer_attr, weight_view_shape); weight_view_shape broadcasts the per-channel BN scale onto the weight tensor.
_BN_FOLD_MAP = [
    ('conv_bn_layers', 'conv_layers',    (-1, 1, 1, 1)),
    ('dw_bn_layers',   'dw_conv_layers', (-1, 1, 1, 1)),
    ('pw_bn_layers',   'pw_conv_layers', (-1, 1, 1, 1)),
    ('fc_bn_layers',   'fc_layers',      (-1, 1)),
]


def _fold_bn_block(layer, bn, weight_view, eps=1e-5):
    scale = bn.weight.data / torch.sqrt(bn.running_var + eps)
    layer.weight.data *= scale.view(weight_view)
    bias = layer.bias.data if layer.bias is not None else torch.zeros_like(bn.running_mean)
    layer.bias = nn.Parameter((bias - bn.running_mean) * scale + bn.bias.data)


def fold_bn_into_model(model):
    """Fold BatchNorm into Conv/FC weights before QAT. Removes BN layers in-place."""
    if not getattr(model, 'use_bn', False):
        return model
    for bn_attr, layer_attr, weight_view in _BN_FOLD_MAP:
        if hasattr(model, bn_attr) and hasattr(model, layer_attr):
            for layer, bn in zip(getattr(model, layer_attr), getattr(model, bn_attr)):
                _fold_bn_block(layer, bn, weight_view)
            delattr(model, bn_attr)
    model.use_bn = False
    print("[BN Fold] Folded BatchNorm into Conv/FC weights")
    return model
