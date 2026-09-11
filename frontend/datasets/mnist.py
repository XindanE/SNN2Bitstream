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
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEFAULT_ENCODING = "rate"


def get_loaders(dataset_cfg, batch_size, encoding="repeat"):
    path = dataset_cfg.get("path", "./data")

    if encoding == "rate":
        # Rate coding: pixel values as spike probabilities, no normalization
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1] range
        ])
    else:
        # Repeat encoding: standard normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(path, train=False, download=True, transform=transform)

    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    input_dim = (1, 28, 28)
    num_classes = 10
    return train_loader, test_loader, input_dim, num_classes
