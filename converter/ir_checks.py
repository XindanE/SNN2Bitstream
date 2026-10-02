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

"""IR validation and heuristic warnings."""
import os


def sanity_check_ir(ir, ir_dir, thr_eps=1e-3, sat_warn_threshold=0.5):
    """Warn on near-zero subtractive thresholds and high weight endpoint occupancy.

    Returns (code, message) pairs without modifying the IR.
    """
    import csv
    warnings_out = []


    for L in ir.get("layers", []):
        if L.get("type") == "LIF":
            thr = L.get("threshold", 0.0)
            rst = L.get("reset_mechanism", "subtract")
            if abs(thr) < thr_eps and rst == "subtract":
                init_thr = L.get("init_threshold")
                learn_thr = L.get("learn_threshold")
                if init_thr is not None and learn_thr is not None:
                    if init_thr > thr_eps and learn_thr:
                        msg = (f"LIF '{L.get('name','?')}': threshold={thr:.4g}, reset=subtract "
                               f"(init={init_thr:.4g}, learn_threshold=True). "
                               f"Near-zero reset decrement; check firing activity and accuracy.")
                        warnings_out.append(("DRIFTED_DEGEN_LIF", msg))
                    else:
                        msg = (f"LIF '{L.get('name','?')}': threshold={thr:.4g}, reset=subtract "
                               f"(init={init_thr:.4g}, learn_threshold={learn_thr}). "
                               f"Near-zero reset decrement; check firing activity and accuracy.")
                        warnings_out.append(("DESIGNED_DEGEN_LIF", msg))
                else:
                    msg = (f"LIF '{L.get('name','?')}': threshold={thr:.4g}, reset=subtract. "
                           f"Near-zero reset decrement; check firing activity and accuracy.")
                    warnings_out.append(("DEGENERATE_LIF", msg))

    # Endpoint occupancy (assumes per-tensor symmetric quantization).
    for L in ir.get("layers", []):
        if L.get("type") not in ("Conv2d", "Linear"):
            continue
        qw = L.get("quant_weight")
        if not qw or "bit_width" not in qw:
            continue
        bw = qw["bit_width"]
        if bw >= 16:
            continue  
        weight_csv = L.get("weight")
        if not weight_csv:
            continue
        csv_path = os.path.join(ir_dir, weight_csv) if ir_dir else weight_csv
        if not os.path.exists(csv_path):
            continue
        try:
            qmax = (1 << (bw - 1)) - 1
            qmin = -(qmax + 1)
            sat, total = 0, 0
            with open(csv_path) as f:
                for row in csv.reader(f):
                    for v in row:
                        try:
                            vi = int(v)
                        except ValueError:
                            continue
                        total += 1
                        if vi == qmax or vi == qmin:
                            sat += 1
            if total > 0 and sat / total > sat_warn_threshold:
                msg = (f"{L['type']} '{L.get('name','?')}': bw={bw}, "
                       f"{sat}/{total} ({100*sat/total:.1f}%) weights at integer endpoints "
                       f"[{qmin}, {qmax}]. Check weight distribution and quantization scale.")
                warnings_out.append(("WEIGHT_SATURATION", msg))
        except Exception as e:
            warnings_out.append(("CHECK_FAIL", f"could not scan {weight_csv}: {e}"))

    if warnings_out:
        print(f"[SANITY] {len(warnings_out)} warning(s) on IR - review before deployment:")
        for sev, m in warnings_out:
            print(f"  [{sev}] {m}")
    else:
        print("[SANITY] No warnings from available checks; skipped data is not validated.")
    return warnings_out


# Schema for the optimization / data_identity / per-layer opt sections.
VALID_LAYER_TYPES  = {"Conv2d", "DepthwiseConv2d", "AvgPool2d", "MaxPool2d", "Linear", "LIF"}
VALID_OPT_TAGS     = {"conv_kernel", "conv_ic", "pack_spikes", "dataflow"}
VALID_CONFIGS      = {"S", "SP", "SQ", "SPQ"}
VALID_ENCODINGS    = {"repeat", "rate", "temporal", "delta"}
VALID_VALUE_CODING = {"spike", "count"}
PER_LAYER_OPT_KEYS = {"oc_factor", "acc_width", "acc_int", "scale_width", "scale_int",
                      "lif_mem_width", "lif_mem_int", "lif_mem_narrow"}


def validate_ir_schema(ir):
    """Validate IR structure and the optimization / data_identity / per-layer opt sections.

    Structurally fatal problems (no layers, unknown layer type) raise ValueError, since a
    malformed IR would otherwise crash codegen with a cryptic error. Softer issues (unknown
    opt tag/key, config/encoding out of range) are printed as warnings. The optimization and
    data_identity sections are optional; absent sections are skipped (backward compatible).
    Returns the list of (severity, msg).
    """
    issues, fatal = [], []

    layers = ir.get("layers")
    if not isinstance(layers, list) or not layers:
        fatal.append(("NO_LAYERS", "IR has no non-empty 'layers' list"))
        layers = []
    for i, L in enumerate(layers):
        t = L.get("type")
        if t not in VALID_LAYER_TYPES:
            fatal.append(("BAD_LAYER_TYPE", f"layer[{i}] type={t!r} not in {sorted(VALID_LAYER_TYPES)}"))
        opt = L.get("opt")
        if isinstance(opt, dict):
            for k in opt:
                if k not in PER_LAYER_OPT_KEYS:
                    issues.append(("UNKNOWN_LAYER_OPT",
                                   f"layer[{i}] '{L.get('name','?')}' opt has unknown key '{k}'"))
            if opt.get("oc_factor") and t not in ("Conv2d", "DepthwiseConv2d"):
                issues.append(("OC_ON_NONCONV",
                               f"layer[{i}] '{L.get('name','?')}' has non-zero oc_factor on {t}"))

    opt_g = ir.get("optimization")
    if isinstance(opt_g, dict):
        cfg = opt_g.get("config")
        if cfg is not None and cfg not in VALID_CONFIGS:
            issues.append(("BAD_CONFIG", f"optimization.config={cfg!r} not in {sorted(VALID_CONFIGS)}"))
        for tag in opt_g.get("opt", []) or []:
            if tag not in VALID_OPT_TAGS:
                issues.append(("BAD_OPT_TAG", f"optimization.opt has unknown tag '{tag}'"))

    di = ir.get("data_identity")
    if isinstance(di, dict):
        enc = di.get("encoding")
        if enc is not None and enc not in VALID_ENCODINGS:
            issues.append(("BAD_ENCODING", f"data_identity.encoding={enc!r} not in {sorted(VALID_ENCODINGS)}"))
        vc = di.get("value_coding")
        if vc is not None and vc not in VALID_VALUE_CODING:
            issues.append(("BAD_CODING", f"data_identity.value_coding={vc!r} not in {sorted(VALID_VALUE_CODING)}"))

    all_issues = fatal + issues
    for sev, m in all_issues:
        print(f"  [SCHEMA:{sev}] {m}")
    if fatal:
        raise ValueError(f"IR schema validation failed with {len(fatal)} fatal issue(s); see above.")
    return all_issues
