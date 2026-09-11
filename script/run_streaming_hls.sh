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

# Streaming per-stage HLS: synthesize each stage independently and aggregate reports.

usage() {
    cat <<EOF
Usage: $0 <project_name> [streaming_dir]

Runs Vitis HLS synthesis on each Streaming stage in backend_projects/<project>/xilinx/streaming/stage*/
and aggregates per-stage resource and latency reports.
streaming_dir overrides the default build tree location.

Arguments:
  project_name  Name of the project (must have streaming/ subdirectory)

Example:
  $0 nmnist_csnn_tiny_dsc_qat_ft
EOF
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

PROJ_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Build tree (sources + TCL) lives under xilinx/streaming; caller may override via $2.
Streaming_DIR="${2:-${ROOT_DIR}/backend_projects/${PROJ_NAME}/xilinx/streaming}"
SUMMARY="${Streaming_DIR}/streaming_hls_summary.txt"

if [[ ! -d "$Streaming_DIR" ]]; then
    echo "[Error] Streaming directory not found: $Streaming_DIR"
    echo "Run converter with --backend streaming first."
    exit 1
fi

# Count stages
shopt -s nullglob
STAGE_DIRS=("$Streaming_DIR"/stage*/)
NUM_STAGES=${#STAGE_DIRS[@]}

if [[ "$NUM_STAGES" -eq 0 ]]; then
    echo "[Error] No stage directories found in $Streaming_DIR"
    exit 1
fi

echo "  Streaming Per-Stage HLS Synthesis"
echo "Project: ${PROJ_NAME}"
echo "Streaming dir: ${Streaming_DIR}"
echo "Stages: ${NUM_STAGES}"

# Initialize summary
{
    echo "=== Streaming Per-Stage HLS Summary ==="
    echo "Date: $(date)"
    echo "Project: ${PROJ_NAME}"
    echo "Stages: ${NUM_STAGES}"
    echo ""
    printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
        "Stage" "Description" "BRAM" "DSP" "FF" "LUT" "Latency"
    printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
        "-----" "-----------" "----" "---" "--" "---" "-------"
} > "$SUMMARY"

TOTAL_BRAM=0
TOTAL_DSP=0
TOTAL_FF=0
TOTAL_LUT=0
TOTAL_LATENCY=0
FAIL_COUNT=0

add_metric() {
    if [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]]; then
        echo "$((10#$1 + 10#$2))"
    else
        echo "N/A"
    fi
}

mark_totals_incomplete() {
    TOTAL_BRAM=N/A
    TOTAL_DSP=N/A
    TOTAL_FF=N/A
    TOTAL_LUT=N/A
    TOTAL_LATENCY=N/A
}

for stage_dir in "${STAGE_DIRS[@]}"; do
    stage_name=$(basename "$stage_dir")
    stage_idx=${stage_name#stage}

    echo -e "\n[Stage ${stage_idx}] Synthesizing..."

    cd "$stage_dir"

    if [[ ! -f "run_hls_stage.tcl" ]]; then
        echo "  SKIP: run_hls_stage.tcl not found"
        mark_totals_incomplete
        printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
            "$stage_idx" "NO TCL" "N/A" "N/A" "N/A" "N/A" "N/A" >> "$SUMMARY"
        continue
    fi

    # Run HLS synthesis
    if vitis_hls -f run_hls_stage.tcl 2>&1 | tee "${stage_dir}/hls_log.txt"; then
        echo "  [Done] Stage ${stage_idx} synthesis completed"
    else
        echo "  [FAIL] Stage ${stage_idx} synthesis failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        mark_totals_incomplete
        printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
            "$stage_idx" "FAILED" "N/A" "N/A" "N/A" "N/A" "N/A" >> "$SUMMARY"
        continue
    fi

    RPT_XML="${stage_dir}/stage${stage_idx}_hls/solution1/syn/report/csynth.xml"
    if [[ ! -f "$RPT_XML" ]]; then
        # Try alternative report path
        RPT_XML=$(find "${stage_dir}/stage${stage_idx}_hls/solution1/syn/report/" \
                  -name "*.xml" -type f 2>/dev/null | head -1) || RPT_XML=""
    fi

    if [[ -z "$RPT_XML" || ! -f "$RPT_XML" ]]; then
        echo "  [Warn] HLS report XML not found for stage ${stage_idx}"
        mark_totals_incomplete
        printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
            "$stage_idx" "NO REPORT" "N/A" "N/A" "N/A" "N/A" "N/A" >> "$SUMMARY"
        continue
    fi

    METRICS=$(python3 - "$RPT_XML" <<'PYXML'
import re
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
try:
    root = ET.parse(path).getroot()
except (OSError, ET.ParseError) as exc:
    print(f"[Warn] Cannot read HLS report {path}: {exc}", file=sys.stderr)
    print("N/A N/A N/A N/A N/A")
else:
    res = root.find(".//AreaEstimates/Resources")
    perf = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")

    def metric(parent, *tags):
        for tag in tags:
            value = parent.findtext(tag) if parent is not None else None
            if value is not None:
                value = value.strip()
                if re.fullmatch(r"[0-9]+", value):
                    return str(int(value))
                break
        print(f"[Warn] Missing or non-numeric {tags[0]} in {path}", file=sys.stderr)
        return "N/A"

    print(metric(res, "BRAM_18K"), metric(res, "DSP48E", "DSP"),
          metric(res, "FF"), metric(res, "LUT"),
          metric(perf, "Best-caseLatency", "Worst-caseLatency"))
PYXML
    )
    read -r BRAM DSP FF LUT LATENCY <<< "$METRICS"

    # Read description from stage top header
    DESC=$(head -3 "${stage_dir}/stage${stage_idx}_top.cpp" | grep "Stage" | sed 's/.*: //' || echo "")
    DESC="${DESC:0:40}"

    printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
        "$stage_idx" "$DESC" "$BRAM" "$DSP" "$FF" "$LUT" "$LATENCY" >> "$SUMMARY"

    TOTAL_BRAM=$(add_metric "$TOTAL_BRAM" "$BRAM")
    TOTAL_DSP=$(add_metric "$TOTAL_DSP" "$DSP")
    TOTAL_FF=$(add_metric "$TOTAL_FF" "$FF")
    TOTAL_LUT=$(add_metric "$TOTAL_LUT" "$LUT")
    TOTAL_LATENCY=$(add_metric "$TOTAL_LATENCY" "$LATENCY")

    echo "  BRAM=${BRAM} DSP=${DSP} FF=${FF} LUT=${LUT} Latency=${LATENCY}"
done

# Write totals
{
    printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
        "-----" "-----------" "----" "---" "--" "---" "-------"
    printf "%-8s %-40s %8s %8s %8s %8s %12s\n" \
        "TOTAL" "(sum of all stages)" \
        "$TOTAL_BRAM" "$TOTAL_DSP" "$TOTAL_FF" "$TOTAL_LUT" "$TOTAL_LATENCY"
    echo ""
    echo "N/A: metric unavailable for at least one stage."
    echo "Note: Total latency is per-timestep sum. Multiply by T for full inference."
    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        echo "WARNING: ${FAIL_COUNT} stage(s) failed synthesis."
    fi
} >> "$SUMMARY"

echo ""
echo "  Streaming HLS Summary"
cat "$SUMMARY"
echo ""
echo "Summary saved to: ${SUMMARY}"
