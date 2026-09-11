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
import numpy as np

from frontend.models import _as_list
from frontend.models.neurons import PLIFLeaky


class FCNNet(nn.Module):
    """Fully-connected SNN for flat inputs (MNIST, N-MNIST, etc.).
    Input:  [T, B, D]  (time-major)
    Output: [T, B, num_classes] spikes
    """

    def __init__(self,input_dim, output_dim, hidden, beta=0.9, learn_beta=False,
                 threshold=1.0, dropout=0.0, bias=True, neuron="lif", init_tau=2.0,
                 reset_mechanisms=None, bitwidths=None, **kwargs):
        super().__init__()

        # **kwargs absorbs extra TOML params silently.

        if isinstance(input_dim, (tuple, list)):
            input_dim = int(np.prod(input_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta
        self.learn_beta = learn_beta
        self.threshold = threshold
        self.neuron = str(neuron).lower()
        self.init_tau = float(init_tau)
        # Dropout between hidden layers (not after output)
        self.dropout = nn.Dropout((dropout)) if float(dropout) > 0 else None
        self.bias = bias

        # Surrogate gradient
        spike_grad = surrogate.atan()

        self.output_mem = False  # set True for mem_ce training

        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()

        n_total = len(hidden) + 1
        layer_resets = _as_list(reset_mechanisms, n_total, "subtract")
        layer_bitwidths = _as_list(bitwidths, n_total, None)

        prev = input_dim
        for idx, hidden_dim in enumerate(hidden):
            fc = nn.Linear(prev, hidden_dim, bias=self.bias)
            fc._bitwidth = layer_bitwidths[idx]
            self.layers.append(fc)
            self.neurons.append(self._make_neuron(spike_grad, features=hidden_dim,
                                                  reset_mechanism=layer_resets[idx]))
            prev = hidden_dim

        fc = nn.Linear(prev, output_dim, bias=self.bias)
        fc._bitwidth = layer_bitwidths[len(hidden)]
        self.layers.append(fc)
        self.neurons.append(self._make_neuron(spike_grad, features=output_dim,
                                              reset_mechanism=layer_resets[len(hidden)]))

    def _make_neuron(self, spike_grad, features=None, reset_mechanism="subtract"):
        if self.neuron == "plif":
            # PLIF learns the leak through `a`; beta follows from it on every forward, so
            # standardization picks up the trained value and deployment sees a plain LIF.
            return PLIFLeaky(init_tau=self.init_tau, threshold=self.threshold,
                             spike_grad=spike_grad, reset_mechanism=reset_mechanism)
        return snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=spike_grad,
                         learn_beta=self.learn_beta, reset_mechanism=reset_mechanism)
    

    def forward(self, x):
        """x: [T,B,D] (time-major). Returns [T,B,num_classes] spikes."""
        assert x.dim() >= 3, "expect time-major input [T,B,...]"
        T = x.size(0)
        B = x.size(1)

        mem_states = []
        for lif in self.neurons: mem_states.append(lif.init_leaky())

        use_act_quant = getattr(self, 'use_act_quant', False) and hasattr(self, 'act_quantizers')

        spk_outputs = []
        mem_outputs = [] if self.output_mem else None
        n_layers = len(self.layers)

        for t in range(T):
            z = x[t]    #[B,D] / [B,C,H,W]
            if z.dim() > 2:
                z = z.view(B, -1)
            # Linear -> LIF, repeated per layer
            for i, (fc, lif) in enumerate(zip(self.layers, self.neurons)):
                z = fc(z)
                z, mem_states[i] = lif(z, mem_states[i])
                if use_act_quant:
                    z= self.act_quantizers[i](z)
                if self.dropout is not None and i < n_layers - 1: # hidden layer only
                    z = self.dropout(z)

            spk_outputs.append(z) #[B,C]
            if self.output_mem:
                mem_outputs.append(mem_states[-1])  # output layer membrane [B,C]

        spikes = torch.stack(spk_outputs, dim=0)  # [T,B,C]
        if self.output_mem:
            return spikes, torch.stack(mem_outputs, dim=0)  # [T,B,C], [T,B,C]
        return spikes
