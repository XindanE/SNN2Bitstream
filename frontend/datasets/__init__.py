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

import importlib

import torch
from torch.nn.utils.rnn import pad_sequence


def module_name(kind):
    return str(kind).lower().replace("-", "")


def default_encoding(kind):
    """Dataset's DEFAULT_ENCODING, or None if it has no module here."""
    try:
        mod = importlib.import_module(f"frontend.datasets.{module_name(kind)}")
    except ModuleNotFoundError:
        return None
    return getattr(mod, "DEFAULT_ENCODING", None)


def pad_collate(batch):
    """
    Custom collate function that pads variable-length sequences.
    Converts to float32 and pads along time dimension.
    """
    data = [torch.as_tensor(item[0], dtype=torch.float32) for item in batch]
    targets = torch.LongTensor([item[1] for item in batch])
    # pad_sequence expects [T, ...] tensors, output is [T_max, B, ...]
    # We want [B, T_max, ...] so use batch_first=True
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)
    return padded_data, targets
