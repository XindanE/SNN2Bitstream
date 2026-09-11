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

# Xilinx flow (HLS, Vivado, Vitis) for all project types (config and custom route).
# Assumes the SW flow already generated code in backend_projects/<project_name>/cpp/.

usage() {
    cat <<EOF
Usage: $0 <project_name>

Copies generated C++ from backend_projects/<project_name>/cpp/ to backend_projects/<project_name>/xilinx/src/
and runs Vitis HLS + Vivado + Vitis to generate bitstream and ELF.

Works for all project types (config-route and custom-route).
Prerequisite: SW flow must have generated C++ code in backend_projects/<project_name>/cpp/.

Arguments:
  project_name  Name of the project (matches backend_projects/<project_name>/cpp/)

Examples:
  $0 nmnist_fcn_SPQ
  $0 myproj_scnn
EOF
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

PROJ_NAME="$1"

case "$PROJ_NAME" in
    -h|--help)
        usage
        exit 0
        ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SNN2B_OUTPUT_DIR selects the same project directory used by code generation.
PROJ_BASE="${SNN2B_OUTPUT_DIR:-backend_projects}"
[[ "$PROJ_BASE" = /* ]] || PROJ_BASE="${ROOT_DIR}/${PROJ_BASE}"
OUT_C_DIR="${PROJ_BASE}/${PROJ_NAME}/cpp"
XILINX_PROJ_DIR="${PROJ_BASE}/${PROJ_NAME}/xilinx"
BAMBU_BUILD_DIR="${PROJ_BASE}/${PROJ_NAME}/bambu"
XILINX_SRC_DIR="${XILINX_PROJ_DIR}/src"
HLS_TEMPLATE_DIR="${ROOT_DIR}/frontend/xilinx_templates"

# Board suffix used in generated Vivado project names. Set BOARD_TAG together with
# FPGA_PART / BOARD_MATCH to target a board other than the ZCU104
BOARD_TAG="${BOARD_TAG:-zcu104}"
export BOARD_TAG

require_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[Error] required tool '$1' not found in PATH ($2)"
        exit 1
    }
}

require_xilinx_tools() {
    require_tool vitis_hls "Xilinx Vitis HLS 2020.2"
    require_tool vivado    "Xilinx Vivado 2020.2"
    require_tool xsct      "Xilinx Vitis (xsct)"
}

# Bambu backend
run_bambu_flow() {
    echo -e "\nDetected Bambu backend (run_bambu.sh found)"
    require_tool bambu  "PandA Bambu HLS"
    require_tool vivado "Xilinx Vivado 2020.2"
    require_tool xsct   "Xilinx Vitis (xsct)"
    mkdir -p "${BAMBU_BUILD_DIR}"
    cp -r "${OUT_C_DIR}"/. "${BAMBU_BUILD_DIR}/"
 
    cp "${ROOT_DIR}/tools/xinference_bambu.h" "${BAMBU_BUILD_DIR}/" 2>/dev/null || true
    cd "${BAMBU_BUILD_DIR}"

    echo -e "\n[1/3] Running Bambu HLS synthesis ..."
    bash run_bambu.sh

    echo -e "\n[2/3] Running Vivado block design ..."
    vivado -mode batch -source build_block_design_bambu.tcl -tclargs "${PROJ_NAME}"

    local XSA_FILE="${BAMBU_BUILD_DIR}/vivado_${PROJ_NAME}_${BOARD_TAG}/${PROJ_NAME}_${BOARD_TAG}.xsa"
    if [[ ! -f "$XSA_FILE" ]]; then
        echo "[Error] Vivado produced no XSA; the block design failed - see the log above."
        exit 1
    fi

    echo -e "\n[3/3] Vitis (create ELF) ..."
    xsct run_vitis_bambu.tcl "${ROOT_DIR}/tools/main_sd.c"
    local ELF_FILE="${BAMBU_BUILD_DIR}/vitis_${PROJ_NAME}/SW_${PROJ_NAME}/Debug/SW_${PROJ_NAME}.elf"
    if [[ ! -s "$ELF_FILE" ]]; then
        echo "[Error] Vitis produced no ELF; see the build log above."
        exit 1
    fi

    echo ""
    echo "  Bambu Build Completed!"
    echo "Project: ${PROJ_NAME}"
    echo "Bitstream: ${BAMBU_BUILD_DIR}/vivado_${PROJ_NAME}_${BOARD_TAG}/${PROJ_NAME}_${BOARD_TAG}.runs/impl_1/design_1_wrapper.bit"
    echo "XSA: ${XSA_FILE}"
    echo "ELF: ${ELF_FILE}"
}

# Streaming backend
run_streaming_flow() {
    local SRC_STREAM_DIR="${OUT_C_DIR}/streaming"
    local STREAM_DIR="${XILINX_PROJ_DIR}/streaming"
    echo -e "\nDetected streaming backend (streaming/ found)"
    require_xilinx_tools

    echo -e "\n[0/3] Assembling xilinx streaming build tree ..."
    for sdir in "${SRC_STREAM_DIR}"/stage*/; do
        [ -d "$sdir" ] || continue
        local sname; sname=$(basename "$sdir")
        mkdir -p "${STREAM_DIR}/${sname}"
        cp "$sdir"/*.cpp "$sdir"/*.h "${STREAM_DIR}/${sname}/" 2>/dev/null || true
    done
    # Vitis host build reads ../model.h relative to the streaming dir.
    cp "${OUT_C_DIR}/model.h" "${XILINX_PROJ_DIR}/model.h" 2>/dev/null || true

    echo -e "\n[1/3] Per-stage HLS synthesis ..."
    bash "${ROOT_DIR}/script/run_streaming_hls.sh" "${PROJ_NAME}" "${STREAM_DIR}"

    # Each stage must have exported its IP before Vivado can integrate them.
    local n_stage n_ip
    n_stage=$(find "${STREAM_DIR}" -maxdepth 1 -type d -name 'stage*' | wc -l)
    n_ip=$(find "${STREAM_DIR}"/stage*/ip_export -name 'export.zip' 2>/dev/null | wc -l)
    if [[ "$n_ip" -lt "$n_stage" ]]; then
        echo "[Error] only ${n_ip}/${n_stage} streaming stages produced IP (export.zip missing); HLS failed - see ${STREAM_DIR}/stage*/hls_log.txt"
        exit 1
    fi

    echo -e "\n[2/3] Vivado streaming integration ..."
    cd "${STREAM_DIR}"
    vivado -mode batch -source vivado_streaming_zcu104.tcl -tclargs "${PROJ_NAME}"

    local BITSTREAM
    BITSTREAM=$(find "${STREAM_DIR}" -name '*_wrapper.bit' 2>/dev/null | head -1)
    if [[ -z "$BITSTREAM" ]]; then
        echo "[Error] Vivado produced no bitstream; streaming integration failed - see the log above."
        exit 1
    fi

    echo -e "\n[3/3] Vitis (create ELF) ..."
    if [[ ! -f run_vitis_streaming.tcl || ! -f main_sd_streaming.c ]]; then
        echo "[Error] Streaming Vitis files missing; regenerate the project with --streaming."
        exit 1
    fi
    xsct run_vitis_streaming.tcl "${STREAM_DIR}/main_sd_streaming.c"
    local ELF_FILE="${STREAM_DIR}/vitis_streaming/SW_streaming/Debug/SW_streaming.elf"
    if [[ ! -s "$ELF_FILE" ]]; then
        echo "[Error] Vitis produced no ELF; see the build log above."
        exit 1
    fi

    echo ""
    echo "  Streaming Build Completed!"
    echo "Project: ${PROJ_NAME}"
    echo "Bitstream: ${BITSTREAM}"
    echo "ELF: ${ELF_FILE}"
}

# Monolithic Vitis HLS backend: single inference IP, then Vivado, then Vitis ELF.
run_vitis_flow() {
    require_xilinx_tools
    mkdir -p "${XILINX_SRC_DIR}"

    echo -e "\n[1/4] Copying generated code to Xilinx project ..."
    rm -f "${XILINX_SRC_DIR}"/*.c "${XILINX_SRC_DIR}"/*.cpp "${XILINX_SRC_DIR}"/*.h 2>/dev/null || true
    # Copy only .cpp and .h; .c test harnesses like main_nmnist.c break HLS.
    cp -v "${OUT_C_DIR}"/*.cpp "${XILINX_SRC_DIR}/" 2>/dev/null || true
    cp -v "${OUT_C_DIR}"/*.h "${XILINX_SRC_DIR}/" 2>/dev/null || true
    # Per-project HLS op directives (converter emits it only for --mul-impl-fabric);
    # goes next to run_hls_inference.tcl so the script can source it before csynth.
    [ -f "${OUT_C_DIR}/hls_directives.tcl" ] && cp -v "${OUT_C_DIR}/hls_directives.tcl" "${XILINX_PROJ_DIR}/"
    # Copy HLS/Vivado/Vitis Tcl templates
    cp -v "${HLS_TEMPLATE_DIR}/run_hls_inference.tcl" \
          "${XILINX_PROJ_DIR}/run_hls_inference.tcl"
    cp -v "${HLS_TEMPLATE_DIR}/vivado_build_inference_zcu104.tcl" \
          "${XILINX_PROJ_DIR}/vivado_build_inference_zcu104.tcl"
    cp -v "${HLS_TEMPLATE_DIR}/run_vitis_app.tcl" \
          "${XILINX_PROJ_DIR}/run_vitis_app.tcl"

    echo -e "\n[2/4] Running Vitis HLS ..."
    cd "${XILINX_PROJ_DIR}"
    # Skip csynth if the IP is already built (Vivado-only rerun): lets a timed-out-in-Vivado
    # arch redo just the Vivado flow with a fresh time budget, without re-running HLS.
    if [[ -f "${XILINX_PROJ_DIR}/hls_ip/export.zip" ]]; then
        echo "     hls_ip/export.zip exists - skipping Vitis HLS (Vivado-only rerun)"
    else
        vitis_hls -f run_hls_inference.tcl
    fi

    # vitis_hls exits 0 even when csynth fails, so check it actually produced the IP.
    if [[ ! -f "${XILINX_PROJ_DIR}/hls_ip/export.zip" ]]; then
        echo "[Error] Vitis HLS produced no IP (hls_ip/export.zip missing); synthesis failed - see the log above."
        exit 1
    fi

    echo -e "\n[3/4] Running Vivado ..."
    vivado -mode batch -source vivado_build_inference_zcu104.tcl -tclargs "${PROJ_NAME}"

    local BITSTREAM XSA_FILE ELF_FILE XILINX_REPORT
    BITSTREAM="${XILINX_PROJ_DIR}/vivado_${PROJ_NAME}_${BOARD_TAG}/${PROJ_NAME}_${BOARD_TAG}.runs/impl_1/design_1_wrapper.bit"
    XSA_FILE="${XILINX_PROJ_DIR}/vivado_${PROJ_NAME}_${BOARD_TAG}/${PROJ_NAME}_${BOARD_TAG}.xsa"
    if [[ ! -f "$XSA_FILE" ]]; then
        echo "[Error] Vivado produced no XSA; implementation failed - see the log above."
        exit 1
    fi

    echo -e "\n[4/4] Vitis (create ELF) ..."
    xsct run_vitis_app.tcl "${ROOT_DIR}/tools/main_sd.c"
    ELF_FILE="${XILINX_PROJ_DIR}/vitis_${PROJ_NAME}/SW_${PROJ_NAME}/Debug/SW_${PROJ_NAME}.elf"
    if [[ ! -s "$ELF_FILE" ]]; then
        echo "[Error] Vitis produced no ELF; see the build log above."
        exit 1
    fi

    echo ""
    echo "  Xilinx Build Completed!"
    echo "Project: ${PROJ_NAME}"
    echo "Bitstream: ${BITSTREAM}"
    echo "XSA: ${XSA_FILE}"
    echo "ELF: ${ELF_FILE}"

    XILINX_REPORT="${XILINX_PROJ_DIR}/xilinx_report.txt"
    {
        echo "=== Xilinx Flow Report ==="
        echo "Date: $(date)"
        echo "Project: ${PROJ_NAME}"
        echo "Bitstream: ${BITSTREAM}"
        echo "XSA: ${XSA_FILE}"
        echo "ELF: ${ELF_FILE}"
    } > "${XILINX_REPORT}"
    echo "Xilinx Report: ${XILINX_REPORT}"
}

# Verify C++ code exists
if [[ ! -d "$OUT_C_DIR" ]] || ! ls "${OUT_C_DIR}"/*.cpp &>/dev/null; then
    echo "[Error] Generated C++ code not found at $OUT_C_DIR"
    echo "Run the SW flow first: ./script/run_config_sw.sh or run_custom_sw.sh"
    exit 1
fi

echo "  SNN to Bitstream - Xilinx Flow"
echo "PROJECT_NAME    = ${PROJ_NAME}"
echo "OUT_C_DIR       = ${OUT_C_DIR}"
echo "XILINX_PROJ_DIR = ${XILINX_PROJ_DIR}"

# Dispatch by backend: Bambu (run_bambu.sh) > streaming (streaming/) > monolithic.
if [ -f "${OUT_C_DIR}/run_bambu.sh" ]; then
    run_bambu_flow
elif [ -d "${OUT_C_DIR}/streaming" ]; then
    run_streaming_flow
else
    run_vitis_flow
fi

# Post-implementation resource/timing/power report, persisted next to the project.
"${ROOT_DIR}/script/post_place_report.sh" "${PROJ_NAME}" \
    | tee "${PROJ_BASE}/${PROJ_NAME}/post_place_report.txt"
