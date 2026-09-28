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

class SurrogateCSNN(nn.Module):
    def __init__(self, beta=0.9, spike_grad=surrogate.atan()):
        super(SurrogateCSNN, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1)  
        self.pool1 = nn.AvgPool2d(2)                         
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1) 
        self.pool2 = nn.MaxPool2d(2)                         
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.fc1   = nn.Linear(1152, 128)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.fc2   = nn.Linear(128, 10)
        self.lif4  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        T, B, C, H, W = x.size()
        spk4_rec = []

        for step in range(T):
            cur1 = self.conv1(x[step])
            spk1 = self.lif1(self.pool1(cur1))

            cur2 = self.conv2(spk1)
            spk2 = self.lif2(self.pool2(cur2))

            flat_spk2 = spk2.view(B, -1)  # 
            
            cur3 = self.fc1(flat_spk2)
            spk3 = self.lif3(cur3)

            cur4 = self.fc2(spk3)
            spk4, _ = self.lif4(cur4)
            spk4_rec.append(spk4)

        return torch.stack(spk4_rec)

class FCSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        num_inputs = 2 * 34 * 34
        num_hidden = 128
        num_outputs = 10
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True, output=True)

    def forward(self, x):
        T, B, C, H, W = x.size()
        spk_out_rec = []

        for step in range(T):
            
            flat_x = x[step].view(B, -1)

            cur1 = self.fc1(flat_x)
            spk1 = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _ = self.lif2(cur2)
            
            spk_out_rec.append(spk2)

        return torch.stack(spk_out_rec)