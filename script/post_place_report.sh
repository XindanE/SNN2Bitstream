#!/usr/bin/env bash
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

set -euo pipefail

# Extract HLS latency and Vivado post-implementation resource, timing, and power
# for one or more projects. Run with no args for usage.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# SNN2B_OUTPUT_DIR overrides the project base (matches run_xilinx.sh / converter.py).
PROJECTS_DIR="${SNN2B_OUTPUT_DIR:-backend_projects}"
[[ "$PROJECTS_DIR" = /* ]] || PROJECTS_DIR="${ROOT_DIR}/${PROJECTS_DIR}"

CSV_MODE=false
PROJECTS=()

# Vivado artifacts land in one of three layouts depending on the flow that built
# them. Search order matters only when a project was built more than one way, in
# which case the more specialised flow is the one we report.
find_rpt() {
    local proj="$1" name="$2" d hit
    for d in "${PROJECTS_DIR}/${proj}/xilinx/streaming" \
             "${PROJECTS_DIR}/${proj}/bambu" \
             "${PROJECTS_DIR}/${proj}/xilinx"; do
        hit=$(ls "$d"/vivado_*_zcu104/*_zcu104.runs/impl_1/design_1_wrapper_"${name}".rpt 2>/dev/null | head -1 || true)
        if [[ -n "$hit" ]]; then echo "$hit"; return; fi
    done
}

for arg in "$@"; do
    case "$arg" in
        --csv) CSV_MODE=true ;;
        --all)
            for d in "${PROJECTS_DIR}"/*/; do
                proj=$(basename "$d")
                if [[ -n "$(find_rpt "$proj" utilization_placed)" ]]; then
                    PROJECTS+=("$proj")
                fi
            done
            ;;
        *) PROJECTS+=("$arg") ;;
    esac
done

if [[ ${#PROJECTS[@]} -eq 0 ]]; then
    cat <<EOF
Usage: $0 [--csv] [--all] <project_name> [project_name ...]

Extract post-implementation resource utilization, HLS latency, timing, and power
from Vivado reports. Use these numbers for reporting.

Options:
  --csv   Output as CSV (tab-separated, for spreadsheet)
  --all   Report all projects that have completed Vivado implementation

Examples:
  $0 nmnist_csnn_tiny_dsc_t10_c8_16_qat_ft_p0p1_opt2_dataflow
  $0 --csv --all
  $0 --csv proj_a proj_b proj_c > results.tsv
EOF
    exit 1
fi

# Parse a Vivado table line like "| CLB LUTs | 20667 | 0 | 230400 | 8.97 |".
# Field layout (split on '|'): empty | Name | Used | Fixed | Available | Util% | empty
_parse_util_line() {
    local file="$1" pattern="$2" field="$3"
    { grep "$pattern" "$file" | head -1 | awk -F'|' -v f="$field" '{gsub(/^[ \t]+|[ \t]+$/, "", $f); print $f}'; } || true
}

# Parse power report Section 1 summary table: "| Dynamic (W)  | 2.960 |"
# Field layout: empty | Name | Value | empty
_parse_power_summary() {
    local file="$1" pattern="$2"
    { grep "$pattern" "$file" | head -1 | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}'; } || true
}

# Parse power report Section 1.1 on-chip table: "| Clocks  | 0.032 | 3 | --- | --- |"
# Field layout: empty | Name | Power(W) | Used | Available | Util% | empty
_parse_power_onchip() {
    local file="$1" pattern="$2"
    { grep "$pattern" "$file" | head -1 | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}'; } || true
}

# Parse power report Section 3.1 hierarchy table: "| inference_0  | 0.093 |"
# Field layout: empty | Name | Power(W) | empty
_parse_power_hierarchy() {
    local file="$1" pattern="$2"
    { grep "$pattern" "$file" | head -1 | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}'; } || true
}

extract_one() {
    local proj="$1"
    # The HLS csynth reports are not covered by find_rpt: their paths differ per
    # flow rather than only in the directory that holds the Vivado run.
    local proj_dir="${PROJECTS_DIR}/${proj}/xilinx"
    local stream_dir="${PROJECTS_DIR}/${proj}/xilinx/streaming"

    local util_rpt timing_rpt power_rpt hls_rpt
    util_rpt=$(find_rpt "$proj" utilization_placed)
    timing_rpt=$(find_rpt "$proj" timing_summary_routed)
    power_rpt=$(find_rpt "$proj" power_routed)

    # In streaming dataflow the slowest stage dominates, so iterate all stage<N>
    # top-level reports and take the max for per-inference latency. Matching only
    # the stage*_csynth.rpt top-level reports avoids picking a sub-function report
    # (e.g. conv_layer0_csynth.rpt) that measures one op invocation, not the stage.
    hls_rpts=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && hls_rpts+=("$f")
    done < <(ls "${stream_dir}"/stage*/stage*_hls/solution1/syn/report/stage*_csynth.rpt 2>/dev/null)
    if [[ ${#hls_rpts[@]} -gt 0 ]]; then
        hls_rpt="${hls_rpts[0]}"  # for clock parsing; latency below uses all stages
    else
        hls_rpt=$(ls "${proj_dir}"/inference_hls/solution1/syn/report/inference_csynth.rpt 2>/dev/null | head -1 || true)
    fi

    # Resource utilization (post-implementation).
    # Fields (split on '|'): 1=empty 2=Name 3=Used 4=Fixed 5=Available 6=Util%
    local lut="-" lut_pct="-" ff="-" ff_pct="-" bram="-" bram_pct="-" dsp="-" dsp_pct="-"
    if [[ -n "$util_rpt" ]]; then
        lut=$(_parse_util_line "$util_rpt" "CLB LUTs" 3)
        lut_pct=$(_parse_util_line "$util_rpt" "CLB LUTs" 6)
        # CLB Registers appears twice (section 1 and 2); use section 1 (first match)
        ff=$(_parse_util_line "$util_rpt" "CLB Registers" 3)
        ff_pct=$(_parse_util_line "$util_rpt" "CLB Registers" 6)
        bram=$(_parse_util_line "$util_rpt" "Block RAM Tile" 3)
        bram_pct=$(_parse_util_line "$util_rpt" "Block RAM Tile" 6)
        dsp=$(_parse_util_line "$util_rpt" "DSPs" 3)
        dsp_pct=$(_parse_util_line "$util_rpt" "DSPs" 6)
    fi

    # Power (post-route)
    local pwr_total="-" pwr_dynamic="-" pwr_static="-"
    local pwr_clocks="-" pwr_logic="-" pwr_signals="-" pwr_bram="-" pwr_dsp="-" pwr_ps8="-"
    local pwr_accel="-"
    if [[ -n "$power_rpt" ]]; then
        # Section 1: Summary
        pwr_total=$(_parse_power_summary "$power_rpt" "Total On-Chip Power")
        pwr_dynamic=$(_parse_power_summary "$power_rpt" "Dynamic (W)")
        pwr_static=$(_parse_power_summary "$power_rpt" "Device Static (W)")
        # Section 1.1: On-Chip Components
        pwr_clocks=$(_parse_power_onchip "$power_rpt" "| Clocks ")
        pwr_logic=$(_parse_power_onchip "$power_rpt" "| CLB Logic ")
        pwr_signals=$(_parse_power_onchip "$power_rpt" "| Signals ")
        pwr_bram=$(_parse_power_onchip "$power_rpt" "| Block RAM ")
        pwr_dsp=$(_parse_power_onchip "$power_rpt" "| DSPs ")
        pwr_ps8=$(_parse_power_onchip "$power_rpt" "| PS8 ")
        # Section 3.1 hierarchy: accelerator IP power. Monolithic flow uses a
        # single inference_0 IP; streaming flow has multiple stage<i>_0 IPs, summed.
        pwr_accel=$(_parse_power_hierarchy "$power_rpt" "inference_0")
        if [[ -z "$pwr_accel" ]]; then
            # Sum stage*_0 entries (streaming).
            pwr_accel=$(awk -F'|' '
                /\| *stage[0-9]+_0 +\|/ {
                    val = $3; gsub(/[ \t]/, "", val);
                    if (val ~ /^[0-9.]+$/) total += val + 0
                }
                END { if (total > 0) printf "%.3f", total; else print "" }
            ' "$power_rpt" 2>/dev/null || true)
        fi
        # Fallback
        [[ -z "$pwr_total" ]]   && pwr_total="-"
        [[ -z "$pwr_dynamic" ]] && pwr_dynamic="-"
        [[ -z "$pwr_static" ]]  && pwr_static="-"
        [[ -z "$pwr_clocks" ]]  && pwr_clocks="-"
        [[ -z "$pwr_logic" ]]   && pwr_logic="-"
        [[ -z "$pwr_signals" ]] && pwr_signals="-"
        [[ -z "$pwr_bram" ]]    && pwr_bram="-"
        [[ -z "$pwr_dsp" ]]     && pwr_dsp="-"
        [[ -z "$pwr_ps8" ]]     && pwr_ps8="-"
        [[ -z "$pwr_accel" ]]   && pwr_accel="-"
    fi

    # Timing (post-route): WNS is on the first data line after the header.
    local wns="-"
    if [[ -n "$timing_rpt" ]]; then
        wns=$(awk '/WNS\(ns\).*TNS\(ns\)/{found=1; next} found && /---/{found=2; next} found==2 && NF>0{print $1; exit}' "$timing_rpt")
        [[ -z "$wns" ]] && wns="-"
    fi

    # HLS latency and clock.
    # Latency fields (split on '|'): 1=empty 2=min_cyc 3=max_cyc 4=min_abs
    # 5=max_abs 6=min_int 7=max_int 8=type
    local min_cycles="-" max_cycles="-" min_latency="-" max_latency="-"
    local hls_target_ns="-" hls_estimated_ns="-" hls_freq_mhz="-"
    if [[ -n "$hls_rpt" ]]; then
        # HLS clock first (needed to turn summed cycles into absolute time).
        # |ap_clk  |  10.00 ns|  7.300 ns|     2.70 ns|
        local clk_line
        clk_line=$(awk '/\+ Timing/,/^\+--/' "$hls_rpt" | grep "ap_clk" | head -1)
        if [[ -n "$clk_line" ]]; then
            hls_target_ns=$(echo "$clk_line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); gsub(/ ns/, "", $3); print $3}')
            hls_estimated_ns=$(echo "$clk_line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); gsub(/ ns/, "", $4); print $4}')
            if [[ "$hls_target_ns" != "-" && -n "$hls_target_ns" ]]; then
                hls_freq_mhz=$(awk "BEGIN {printf \"%.1f\", 1000.0 / $hls_target_ns}")
            fi
        fi
        # Per-inference latency
        _top_latency() {
            awk '/^\+ Latency/,/^\+--/' "$1" | grep -E '^\s+\|' \
                | grep -v 'Latency\|min\|max\|---' | head -1 \
                | awk -F'|' '{gsub(/[ \t]/,"",$2); gsub(/[ \t]/,"",$3); print $2, $3}'
        }
        local best_min=0 best_max=0 n_stages=0
        # Streaming dataflow
        for rpt in "${hls_rpts[@]}"; do
            local c_min c_max
            read -r c_min c_max < <(_top_latency "$rpt")
            [[ "$c_min" =~ ^[0-9]+$ && "$c_max" =~ ^[0-9]+$ ]] || continue
            (( c_min > best_min )) && best_min=$c_min
            (( c_max > best_max )) && best_max=$c_max
            n_stages=$((n_stages + 1))
        done
        # Non-streaming (monolithic)
        if [[ "$n_stages" -eq 0 && -n "$hls_rpt" ]]; then
            local c_min c_max
            read -r c_min c_max < <(_top_latency "$hls_rpt")
            if [[ "$c_min" =~ ^[0-9]+$ && "$c_max" =~ ^[0-9]+$ ]]; then
                best_min=$c_min; best_max=$c_max; n_stages=1
            fi
        fi
        if [[ "$n_stages" -gt 0 ]]; then
            local per_ns="$hls_target_ns"
            [[ "$per_ns" == "-" || -z "$per_ns" ]] && per_ns=10
            min_cycles="$best_min"
            max_cycles="$best_max"
            min_latency=$(awk "BEGIN {printf \"%.3f ms\", $best_min * $per_ns / 1000000.0}")
            max_latency=$(awk "BEGIN {printf \"%.3f ms\", $best_max * $per_ns / 1000000.0}")
        fi
    fi

    # Convert a latency string to milliseconds (e.g. "0.493 sec" -> "493",
    # "2.462 ms" -> "2.462", "500 us" -> "0.5").
    _lat_to_ms() {
        local val unit
        val=$(echo "$1" | awk '{gsub(/[^0-9.]/, "", $1); print $1}')
        unit=$(echo "$1" | awk '{print $2}')
        case "$unit" in
            sec) awk "BEGIN {printf \"%.6f\", $val * 1000.0}" ;;
            ms)  echo "$val" ;;
            us)  awk "BEGIN {printf \"%.6f\", $val / 1000.0}" ;;
            *)   echo "$val" ;;  # assume ms if no unit
        esac
    }

    local min_lat_val="" max_lat_val=""
    [[ "$min_latency" != "-" ]] && min_lat_val=$(_lat_to_ms "$min_latency")
    [[ "$max_latency" != "-" ]] && max_lat_val=$(_lat_to_ms "$max_latency")
    local data_dependent=false
    if [[ -n "$min_lat_val" && -n "$max_lat_val" && "$min_lat_val" != "$max_lat_val" ]]; then
        data_dependent=true
    fi

    # Energy per inference (mJ) = accelerator power (W) x latency (ms).
    local energy_min="-" energy_max="-"
    if [[ "$pwr_accel" != "-" ]]; then
        [[ -n "$min_lat_val" ]] && energy_min=$(awk "BEGIN {printf \"%.4f\", $pwr_accel * $min_lat_val}")
        [[ -n "$max_lat_val" ]] && energy_max=$(awk "BEGIN {printf \"%.4f\", $pwr_accel * $max_lat_val}")
    fi
    # Throughput (inferences/sec) = 1000 / latency_ms
    local tp_min="-" tp_max="-"
    [[ -n "$max_lat_val" && "$max_lat_val" != "0" ]] && tp_min=$(awk "BEGIN {printf \"%.1f\", 1000.0 / $max_lat_val}")
    [[ -n "$min_lat_val" && "$min_lat_val" != "0" ]] && tp_max=$(awk "BEGIN {printf \"%.1f\", 1000.0 / $min_lat_val}")

    # Output
    if $CSV_MODE; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$proj" \
            "$lut" "$lut_pct" "$ff" "$ff_pct" "$bram" "$bram_pct" "$dsp" "$dsp_pct" \
            "$min_cycles" "$max_cycles" "$min_latency" "$max_latency" "$wns" \
            "$hls_freq_mhz" \
            "$pwr_total" "$pwr_dynamic" "$pwr_static" "$pwr_accel" \
            "$pwr_clocks" "$pwr_logic" "$pwr_signals" "$pwr_bram" "$pwr_dsp" \
            "$energy_min" "$energy_max" "$tp_min" "$tp_max" "$data_dependent"
    else
        
        echo "  Project: ${proj}"
        
        if [[ -z "$util_rpt" ]]; then
            echo "  [Warn] No post-implementation report found"
        else
            echo ""
            echo "  Resource Utilization (Vivado post-implementation)"
            echo "  +----------+----------+---------+"
            printf "  | %-8s | %8s | %6s%% |\n" "LUT"  "$lut"  "$lut_pct"
            printf "  | %-8s | %8s | %6s%% |\n" "FF"   "$ff"   "$ff_pct"
            printf "  | %-8s | %8s | %6s%% |\n" "BRAM" "$bram" "$bram_pct"
            printf "  | %-8s | %8s | %6s%% |\n" "DSP"  "$dsp"  "$dsp_pct"
            echo "  +----------+----------+---------+"
        fi

        echo ""
        if [[ -z "$power_rpt" ]]; then
            echo "  Power:  [no power report]"
        else
            echo "  Power (Vivado post-route)"
            echo "  +--------------------+-----------+"
            printf "  | %-18s | %7s W |\n" "Total On-Chip"    "$pwr_total"
            printf "  | %-18s | %7s W |\n" "Dynamic"          "$pwr_dynamic"
            printf "  | %-18s | %7s W |\n" "  Accelerator(PL)" "$pwr_accel"
            printf "  | %-18s | %7s W |\n" "  PS8 (ARM)"       "$pwr_ps8"
            printf "  | %-18s | %7s W |\n" "Static"           "$pwr_static"
            echo "  +--------------------+-----------+"
            printf "  | %-18s | %7s W |\n" "  Clocks"     "$pwr_clocks"
            printf "  | %-18s | %7s W |\n" "  CLB Logic"  "$pwr_logic"
            printf "  | %-18s | %7s W |\n" "  Signals"    "$pwr_signals"
            printf "  | %-18s | %7s W |\n" "  Block RAM"  "$pwr_bram"
            printf "  | %-18s | %7s W |\n" "  DSPs"       "$pwr_dsp"
            echo "  +--------------------+-----------+"
        fi

        echo ""
        if [[ -z "$hls_rpt" ]]; then
            echo "  HLS Latency:  [no HLS report]"
        else
            if $data_dependent; then
                echo "  HLS Latency (data-dependent):"
                echo "    Best:   ${min_cycles} cycles  (${min_latency})"
                echo "    Worst:  ${max_cycles} cycles  (${max_latency})"
            else
                echo "  HLS Latency:  ${max_cycles} cycles  (${max_latency})"
            fi
            if [[ "$hls_target_ns" != "-" ]]; then
                echo "  HLS Clock:    ${hls_target_ns} ns target, ${hls_estimated_ns} ns estimated  (${hls_freq_mhz} MHz)"
            fi
        fi
        local timing_status
        timing_status=$(echo "$wns" | awk '{if($1=="-") print ""; else if($1+0>=0) print "(PASSED)"; else print "(FAILED!)"}')
        echo "  Timing WNS:   ${wns} ns  ${timing_status}"

        # Derived metrics
        if [[ "$energy_max" != "-" || "$tp_min" != "-" ]]; then
            echo ""
            if $data_dependent; then
                echo "  Derived Metrics (data-dependent, showing best~worst)"
                [[ "$energy_min" != "-" ]] && echo "  Energy/Inference:  ${energy_min} ~ ${energy_max} mJ  (accelerator only)"
                [[ "$tp_min" != "-" ]]     && echo "  Throughput:        ${tp_max} ~ ${tp_min} inf/s"
            else
                echo "  Derived Metrics"
                [[ "$energy_max" != "-" ]] && echo "  Energy/Inference:  ${energy_max} mJ  (accelerator only)"
                [[ "$tp_min" != "-" ]]     && echo "  Throughput:        ${tp_min} inf/s"
            fi
        fi
        echo ""
    fi
}

# CSV header
if $CSV_MODE; then
    printf "Project\tLUT\tLUT%%\tFF\tFF%%\tBRAM\tBRAM%%\tDSP\tDSP%%\tCycles_Min\tCycles_Max\tLatency_Min\tLatency_Max\tWNS(ns)\tFreq(MHz)\tPwr_Total(W)\tPwr_Dynamic(W)\tPwr_Static(W)\tPwr_Accel(W)\tPwr_Clocks(W)\tPwr_Logic(W)\tPwr_Signals(W)\tPwr_BRAM(W)\tPwr_DSP(W)\tEnergy_Min(mJ)\tEnergy_Max(mJ)\tTP_Min(inf/s)\tTP_Max(inf/s)\tData_Dependent\n"
fi

for proj in "${PROJECTS[@]}"; do
    extract_one "$proj"
done
