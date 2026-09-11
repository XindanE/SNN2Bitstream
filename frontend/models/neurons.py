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

import math

import torch
import torch.nn as nn
import snntorch as snn


class PLIFLeaky(snn.Leaky):

    def __init__(self, init_tau=2.0, beta_min=0.01, beta_max=0.99, **kwargs):
        # beta here is a placeholder; the real leak is driven by `a`. snn's own learn_beta would
        # register a second, conflicting beta parameter, so it must stay off.
        kwargs.pop("learn_beta", None)
        super().__init__(beta=0.5, **kwargs)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        init_a = -math.log(float(init_tau) - 1.0)
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))

    def forward(self, input_, mem=None):
        self.beta = torch.clamp(1.0 - torch.sigmoid(self.a), self.beta_min, self.beta_max)
        return super().forward(input_, mem)
