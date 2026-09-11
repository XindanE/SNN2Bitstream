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

"""HLS/GCC fixed-point type strings and bit-width sizing for the converter."""
import math


def hls_typedef(config):
    """HLS type string from a fixed-point config dict."""
    type_class = config.get("type_class", "fixed")
    w  = config["width"]
    i  = config.get("int", w)
    match type_class:
        case "uint":   return f"ap_uint<{w}>"
        case "int":    return f"ap_int<{w}>"
        case "ufixed": return f"ap_ufixed<{w}, {i}, AP_TRN, AP_SAT>"
        case _:        return f"ap_fixed<{w}, {i}, AP_TRN, AP_SAT>"


def gcc_typedef(config):
    """GCC fallback type string from a fixed-point config dict."""
    type_class = config.get("type_class", "fixed")
    w  = config["width"]
    match type_class:
        case "uint": return "uint8_t" if w <= 8 else ("uint16_t" if w <= 16 else "uint32_t")
        case "int":  return "int8_t"  if w <= 8 else ("int16_t"  if w <= 16 else "int32_t")
        case _:      return "float"


def compute_mac_acc_width(n_mac, w_int_bits, in_int_bits, data_t_frac, cap_width=48):
    """Compute minimum acc_t (width, int_bits) for a Conv/FC/DwConv MAC accumulator.

    Worst case: per-MAC product magnitude <= 2^(w_int_bits + in_int_bits), N-MAC sum <= N x that, so signed int bits = ceil(log2(N x per_mac_max)) + 1.
    Returns (acc_width, acc_int_bits) rounded up to a multiple of 8 and capped at cap_width. Frac bits = data_t_frac to preserve dequant precision.
    """
    per_mac_int = w_int_bits + in_int_bits
    sum_int     = per_mac_int + max(1, math.ceil(math.log2(max(n_mac, 2))))
    acc_int     = sum_int + 1            # sign bit
    acc_frac    = max(8, min(16, data_t_frac))
    acc_width   = acc_int + acc_frac
    # Round up to a multiple of 8. If the needed width exceeds cap_width the accumulator is capped; acc_t is AP_SAT so it clamps instead of wrapping, but the sum is no longer exact, so warn.
    acc_width_needed = ((acc_width + 7) // 8) * 8
    if acc_width_needed > cap_width:
        print(f"[Warn] MAC acc_t needs {acc_width_needed} bits (N_MAC={n_mac}) but is capped "
              f"at {cap_width}; AP_SAT will clamp on overflow - raise cap_width if accuracy drops.")
    acc_width = min(cap_width, acc_width_needed)
    acc_int   = acc_width - acc_frac
    return acc_width, acc_int


def compute_lif_mem_width_profiled(mem_max_abs, data_t_width, data_t_int, slack=2.0, frac=8):
    """Profile-based mem width sizing.

    Take measured |mem|_max from a reference inference run, multiply by slack, and round up to a clean ap_fixed width. This avoids guessing bounds from BETA/VTH, which ignores input transients.
    Returns (width, int_bits, is_narrow).
    """
    bound      = max(mem_max_abs * slack, 0.25)
    needed_int = max(2, math.ceil(math.log2(max(bound * 2, 4))))  # *2 bakes in the sign bit
    width      = needed_int + frac
    width      = ((width + 3) // 4) * 4
    int_bits   = width - frac
    if width >= data_t_width:
        return data_t_width, data_t_int, False
    return width, int_bits, True
