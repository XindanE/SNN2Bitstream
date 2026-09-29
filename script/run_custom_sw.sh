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
shopt -s nullglob

# Custom-route SW flow: standardize_model, export_ir, C++, GCC test (full dataset).
# Custom route always uses PTQ for quantization (no QAT support).
# Test data is cached in test_data/<dataset>/ and reused across projects.

usage() {
    cat <<EOF
Usage: $0 <model_class> <checkpoint> <project_name> <timesteps> [options]

Arguments:
  model_class   Python model class path (e.g., user_model.my_model.MyFCN)
  checkpoint    Path to PyTorch checkpoint file
  project_name  Name for the generated project
  timesteps     Number of timesteps for SNN inference

Options:
  --config        Configuration preset (default: SPQ):
                    S   - Stage structure only (float, no pragma)
                    SQ  - Stage + Quantization (8-bit PTQ + ap_fixed)
                    SP  - Stage + Pragma optimization (float + HLS directives)
                    SPQ - Stage + Pragma + Quantization (full optimization)

  --dataset       Dataset name for GCC test data (nmnist | cifar10dvs | dvsgesture | mnist).
                  Required for full GCC testing. If omitted, test step is skipped.

  --input-shape   Input shape C,H,W (e.g., 2,34,34). Required for CSNN (Conv2d) models.
                  FCN models auto-detect input dim and do not need this option.

  --toml          TOML file to read [codegen] settings from (optional).

  Note: Custom route always uses PTQ (Post-Training Quantization).

Examples:
  # FCN model on N-MNIST
  $0 user_model.my_model.MyFCN checkpoints/my_model.pt myproj 10 --dataset nmnist

  # CSNN model with full optimization
  $0 user_model.my_model.MyCSNN checkpoints/my_model.pt myproj_scnn 10 \\
      --config SPQ --input-shape 2,34,34 --dataset nmnist
EOF
}

if [ "$#" -lt 4 ]; then
    usage
    exit 1
fi

MODEL_CLASS="$1"
CKPT_PATH="$2"
PROJ_NAME="$3"
TIMESTEPS="$4"
shift 4

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "[Error] checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Defaults
CONFIG="SPQ"
INPUT_SHAPE=""
DATASET_KIND=""
PARALLEL_FACTOR=""
UNROLL_FLAG=""
DATAFLOW_FLAG=""
SPARSE_FLAG=""
FOLD_FLAG=""           # Fold dequant scale into LIF threshold
BSHIFT_FLAG=""         # Bit-shift LIF leak instead of beta multiply
BINARY_FLAG=""         # Treat input as binary {0,1} (custom route: no train-time detection)
CONV_OC_FACTOR_FLAG=""
CONV_OC_MAX_FLAG=""
USER_PROJECT=""
QUANT_BITS=""
CODEGEN_TOML=""
_CLI_CONFIG_SET="0"
MUL_FABRIC_FLAG=""
SD_ENC_FLAG=""
STREAMING_FLAG=""
DATA_WIDTH_FLAG=""
DATA_INT_FLAG=""
BACKEND_ARG=""
BAMBU_OPT_ARG=""
BAMBU_EXTRA_ARG=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --toml)
            CODEGEN_TOML="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; _CLI_CONFIG_SET="1"; shift 2 ;;
        --input-shape)
            INPUT_SHAPE="$2"; shift 2 ;;
        --dataset)
            DATASET_KIND="$2"; shift 2 ;;
        --parallel-factor)
            PARALLEL_FACTOR="--parallel-factor $2"; shift 2 ;;
        --unroll)
            UNROLL_FLAG="--unroll $2"; shift 2 ;;
        --dataflow)
            DATAFLOW_FLAG="--dataflow"; shift ;;
        --sparse)
            SPARSE_FLAG="--sparse"
            if [[ "${2:-}" == "sp" ]]; then shift 2; else shift; fi ;;
        --fold-dequant)
            FOLD_FLAG="--fold-dequant"; shift ;;
        --bit-shift-beta)
            BSHIFT_FLAG="--bit-shift-beta"; shift ;;
        --input-is-binary)
            BINARY_FLAG="--input-is-binary"; shift ;;
        --conv-oc-factor)
            CONV_OC_FACTOR_FLAG="--conv-oc-factor $2"; shift 2 ;;
        --conv-oc-max)
            CONV_OC_MAX_FLAG="--conv-oc-max $2"; shift 2 ;;
        --project)
            USER_PROJECT="$2"; shift 2 ;;
        --mul-impl-fabric)
            MUL_FABRIC_FLAG="--mul-impl-fabric"; shift ;;
        --sd-encoding)
            SD_ENC_FLAG="--sd-encoding $2"; shift 2 ;;
        --streaming)
            STREAMING_FLAG="--streaming"; shift ;;
        --data-width)
            DATA_WIDTH_FLAG="--data-width $2"; shift 2 ;;
        --data-int-width|--data-int)
            DATA_INT_FLAG="--data-int $2"; shift 2 ;;
        --backend)
            case "$2" in
                vitis) BACKEND_ARG="--backend vitis" ;;
                bambu) BACKEND_ARG="--backend bambu" ;;
                *) echo "[Error] unknown --backend '$2' (expected: vitis | bambu)" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --quant-bits)
            QUANT_BITS="$2"
            if ! [[ "$QUANT_BITS" =~ ^[0-9]+$ ]] || (( QUANT_BITS < 2 || QUANT_BITS > 16 )); then
                echo "[Error] --quant-bits must be an integer in 2..16 (got '$QUANT_BITS')." >&2
                exit 1
            fi
            if (( QUANT_BITS > 8 )); then
                echo "[Warn] --quant-bits $QUANT_BITS is outside the validated range (4/8)."
                echo "       Each extra weight bit widens the MAC accumulator, which caps at 48 bits."
                echo "       Watch the '[Warn] MAC acc_t' lines below and check the GCC accuracy."
            fi
            shift 2 ;;
        --csim-apfixed)
            CSIM_APFIXED=1; _CLI_CSIM_SET="1"; shift ;;
        --no-csim-apfixed)
            CSIM_APFIXED=0; _CLI_CSIM_SET="1"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "[Error] Unknown option $1"; usage; exit 1 ;;
    esac
done

# Read [codegen] from TOML if --toml given (provides defaults; CLI flags override).
if [[ -n "$CODEGEN_TOML" ]]; then
    eval $(python3 - "$CODEGEN_TOML" << 'PYEOF'
import sys, toml
cfg = toml.load(sys.argv[1])
cg = cfg.get("codegen", {})
g = cg.get
print("TOML_CONFIG=" + str(g("config", "")))
print("TOML_SPARSE=" + str(g("sparse", "")).lower())
print("TOML_FOLD=" + str(g("fold_dequant", "")).lower())
print("TOML_BSHIFT=" + str(g("bit_shift_beta", "")).lower())
print("TOML_BINARY=" + str(g("input_is_binary", "")).lower())
print("TOML_QUANT_BITS=" + str(g("quant_bits", "")))
unroll_val = g("unroll", "")
if isinstance(unroll_val, list):
    print("TOML_UNROLL=" + ",".join(unroll_val))
elif unroll_val:
    print("TOML_UNROLL=" + str(unroll_val))
else:
    print("TOML_UNROLL=")
print("TOML_DATAFLOW=" + str(g("dataflow", "")).lower())
print("TOML_PARALLEL_FACTOR=" + str(g("parallel_factor", "")))
print("TOML_CONV_OC_FACTOR=" + str(g("conv_oc_factor", "")))
print("TOML_CONV_OC_MAX=" + str(g("conv_oc_max", "")))
print("TOML_CSIM_APFIXED=" + str(g("csim_apfixed", "")).lower())
print("TOML_STREAMING=" + str(g("streaming", "")).lower())
print("TOML_DATA_WIDTH=" + str(g("data_width", "")))
print("TOML_SD_ENCODING=" + str(g("sd_encoding", "")))
bambu = cg.get("bambu", None)
if bambu is not None:
    print("TOML_BACKEND=bambu")
    print("TOML_BAMBU_OPT='" + str(bambu.get("opt", "")) + "'")
    print("TOML_BAMBU_EXTRA='" + str(bambu.get("extra", "")) + "'")
else:
    print("TOML_BACKEND=")
    print("TOML_BAMBU_OPT=")
    print("TOML_BAMBU_EXTRA=")
PYEOF
) || {
        echo "[Warn] failed to read [codegen] from $CODEGEN_TOML"
    }

    if [[ -n "${TOML_CONFIG:-}" && "$_CLI_CONFIG_SET" != "1" ]]; then CONFIG="$TOML_CONFIG"; fi
    if [[ "${TOML_SPARSE:-}" == "true" && -z "$SPARSE_FLAG" ]]; then SPARSE_FLAG="--sparse"; fi
    if [[ "${TOML_FOLD:-}" == "true" && -z "$FOLD_FLAG" ]]; then FOLD_FLAG="--fold-dequant"; fi
    if [[ "${TOML_BSHIFT:-}" == "true" && -z "$BSHIFT_FLAG" ]]; then BSHIFT_FLAG="--bit-shift-beta"; fi
    if [[ "${TOML_BINARY:-}" == "true" && -z "$BINARY_FLAG" ]]; then BINARY_FLAG="--input-is-binary"; fi
    if [[ -n "${TOML_QUANT_BITS:-}" && -z "$QUANT_BITS" ]]; then QUANT_BITS="$TOML_QUANT_BITS"; fi
    if [[ -n "${TOML_UNROLL:-}" && -z "$UNROLL_FLAG" ]]; then UNROLL_FLAG="--unroll $TOML_UNROLL"; fi
    if [[ "${TOML_DATAFLOW:-}" == "true" && -z "$DATAFLOW_FLAG" ]]; then DATAFLOW_FLAG="--dataflow"; fi
    if [[ -n "${TOML_PARALLEL_FACTOR:-}" && -z "$PARALLEL_FACTOR" ]]; then PARALLEL_FACTOR="--parallel-factor $TOML_PARALLEL_FACTOR"; fi
    if [[ -n "${TOML_CONV_OC_FACTOR:-}" && -z "$CONV_OC_FACTOR_FLAG" ]]; then CONV_OC_FACTOR_FLAG="--conv-oc-factor $TOML_CONV_OC_FACTOR"; fi
    if [[ -n "${TOML_CONV_OC_MAX:-}" && -z "$CONV_OC_MAX_FLAG" ]]; then CONV_OC_MAX_FLAG="--conv-oc-max $TOML_CONV_OC_MAX"; fi
    if [[ "$_CLI_CSIM_SET" != "1" && "${TOML_CSIM_APFIXED:-}" == "true" ]]; then CSIM_APFIXED=1; fi
    if [[ "${TOML_STREAMING:-}" == "true" && -z "$STREAMING_FLAG" ]]; then STREAMING_FLAG="--streaming"; fi
    if [[ -n "${TOML_DATA_WIDTH:-}" && -z "$DATA_WIDTH_FLAG" ]]; then DATA_WIDTH_FLAG="--data-width $TOML_DATA_WIDTH"; fi
    if [[ -n "${TOML_SD_ENCODING:-}" && -z "$SD_ENC_FLAG" ]]; then SD_ENC_FLAG="--sd-encoding $TOML_SD_ENCODING"; fi
    if [[ "${TOML_BACKEND:-}" == "bambu" && -z "$BACKEND_ARG" ]]; then
        BACKEND_ARG="--backend bambu"
        # '=' syntax required: -O2 / --pipelining=... look like flags to argparse
        if [[ -n "${TOML_BAMBU_OPT:-}" ]];   then BAMBU_OPT_ARG="--bambu-opt=${TOML_BAMBU_OPT}"; fi
        if [[ -n "${TOML_BAMBU_EXTRA:-}" ]]; then BAMBU_EXTRA_ARG="--bambu-extra=${TOML_BAMBU_EXTRA}"; fi
        echo "  Backend: bambu (opt='${TOML_BAMBU_OPT}' extra='${TOML_BAMBU_EXTRA}')"
    fi
fi

# Width and integer width are a pair: giving only the width leaves the converter's
# default of int = width // 2.
if [[ -n "$DATA_INT_FLAG" && -z "$DATA_WIDTH_FLAG" ]]; then
    echo "[Error] --data-int-width needs --data-width as well (they set ap_fixed<W,I> together)." >&2
    exit 1
fi

# Normalize and validate config
CONFIG=$(echo "$CONFIG" | tr '[:lower:]' '[:upper:]')
case "$CONFIG" in
    S|SQ|SP|SPQ) ;;
    *) echo "[Error] Invalid config '$CONFIG'. Must be S, SQ, SP, or SPQ."; exit 1 ;;
esac

USE_FIXED_QUANT="no"
if [[ "$CONFIG" == *"Q"* ]]; then USE_FIXED_QUANT="yes"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"
source "${ROOT_DIR}/script/lib_flow.sh"

STANDARDIZED_PATH="user_model/${PROJ_NAME}_standardized.pth"
IR_DIR="ir_output/${PROJ_NAME}"
IR_JSON="${IR_DIR}/ir.json"
if [[ -n "$USER_PROJECT" ]]; then
    OUT_C_DIR="backend_projects/${USER_PROJECT}/cpp"
else
    OUT_C_DIR="backend_projects/${PROJ_NAME}/cpp"
fi

echo "  SNN Custom-Route SW Flow"
echo "MODEL_CLASS     = ${MODEL_CLASS}"
echo "CKPT_PATH       = ${CKPT_PATH}"
echo "PROJECT_NAME    = ${PROJ_NAME}"
echo "TIMESTEPS       = ${TIMESTEPS}"
echo "CONFIG          = ${CONFIG}"
echo "USE_FIXED_QUANT = ${USE_FIXED_QUANT} (always PTQ)"
echo "INPUT_SHAPE     = ${INPUT_SHAPE:-auto}"
echo "DATASET         = ${DATASET_KIND:-not specified (test step will be skipped)}"
if [[ -n "$PARALLEL_FACTOR" ]]; then echo "PARALLEL_FACTOR = ${PARALLEL_FACTOR#--parallel-factor }"; fi
if [[ -n "$UNROLL_FLAG" ]]; then echo "UNROLL          = ${UNROLL_FLAG#--unroll }"; fi
if [[ -n "$DATAFLOW_FLAG" ]]; then echo "DATAFLOW        = enabled"; fi
if [[ -n "$SPARSE_FLAG" ]]; then echo "SPARSE          = enabled"; fi
if [[ -n "$QUANT_BITS" ]]; then echo "QUANT_BITS      = ${QUANT_BITS}"; fi
echo "STANDARDIZED_PATH  = ${STANDARDIZED_PATH}"
echo "IR_DIR          = ${IR_DIR}"
echo "OUT_C_DIR       = ${OUT_C_DIR}"

mkdir -p "$(dirname "${STANDARDIZED_PATH}")"
mkdir -p "${IR_DIR}"
mkdir -p "${OUT_C_DIR}"
start_log "${ROOT_DIR}/${OUT_C_DIR}/sw_flow.log"

# Step 1: standardize_model
echo -e "\n[1/5] Running standardize_model ..."
STANDARDIZE_ARGS=(--model "${MODEL_CLASS}" --input "${CKPT_PATH}" --output "${STANDARDIZED_PATH}")
if [ -n "${INPUT_SHAPE}" ]; then
    STANDARDIZE_ARGS+=(--input-shape "${INPUT_SHAPE}")
fi
run_logged python -m frontend.standardize_model "${STANDARDIZE_ARGS[@]}"

# Step 2: export_ir (PTQ if fixed-point mode, no QAT)
echo -e "\n[2/5] Running export_ir ..."
CUSTOM_QUANT_BITS="${QUANT_BITS:-8}"


# The dataset default is only a fallback: a standardized checkpoint that already carries an
# encoding wins, otherwise a rate-coded model would be re-exported as repeat and lose its
# spike input path.
CKPT_ENCODING=$(python -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(c.get('encoding', '') if isinstance(c, dict) else '')
" "${STANDARDIZED_PATH}" 2>/dev/null | tail -1)

ENCODING_FLAG=""
if [[ -n "$CKPT_ENCODING" ]]; then
    echo "     Input encoding: ${CKPT_ENCODING} (from checkpoint)"
else
    case "$DATASET_KIND" in
        nmnist|dvsgesture|cifar10dvs) ENCODING_FLAG="--encoding temporal" ;;
        mnist)                        ENCODING_FLAG="--encoding repeat" ;;
    esac
    [[ -n "$ENCODING_FLAG" ]] && echo "     Input encoding: ${ENCODING_FLAG#--encoding } (dataset default)"
fi

if [ "$USE_FIXED_QUANT" = "yes" ]; then
    echo "     Fixed-point mode: enabling ${CUSTOM_QUANT_BITS}-bit PTQ quantization"
    run_logged python "${ROOT_DIR}/frontend/export_ir.py" "${STANDARDIZED_PATH}" \
        --timesteps "${TIMESTEPS}" --out-dir "${IR_DIR}" \
        --quant-bits "${CUSTOM_QUANT_BITS}" --quant-mode int8_fixed $ENCODING_FLAG
else
    run_logged python "${ROOT_DIR}/frontend/export_ir.py" "${STANDARDIZED_PATH}" \
        --timesteps "${TIMESTEPS}" --out-dir "${IR_DIR}" $ENCODING_FLAG
fi

if [[ ! -f "$IR_JSON" ]]; then
    echo "[Error] IR not found at $IR_JSON"
    exit 1
fi

# Step 3: Convert to C++
echo -e "\n[3/5] Running converter (config=${CONFIG}) ..."
CONVERTER_PROJECT_ARG=""
if [[ -n "$USER_PROJECT" ]]; then CONVERTER_PROJECT_ARG="--project $USER_PROJECT"; fi
run_logged python "${ROOT_DIR}/converter/converter.py" "${IR_JSON}" --config "${CONFIG}" \
    $PARALLEL_FACTOR $UNROLL_FLAG $DATAFLOW_FLAG $SPARSE_FLAG $FOLD_FLAG $BSHIFT_FLAG \
    $BINARY_FLAG $MUL_FABRIC_FLAG $SD_ENC_FLAG $CONVERTER_PROJECT_ARG $STREAMING_FLAG \
    $CONV_OC_FACTOR_FLAG $CONV_OC_MAX_FLAG $DATA_WIDTH_FLAG $DATA_INT_FLAG \
    $BACKEND_ARG $BAMBU_OPT_ARG ${BAMBU_EXTRA_ARG:+"$BAMBU_EXTRA_ARG"}

# Step 4: Prepare test data (cached in test_data/<dataset>/)
echo -e "\n[4/5] Preparing test data..."

if [[ -z "$DATASET_KIND" ]]; then
    echo "  --dataset not specified. Skipping test data generation."
    echo "  To run GCC testing, re-run with --dataset <nmnist|cifar10dvs|dvsgesture|mnist>"
else
    TEST_DATA_DIR="${ROOT_DIR}/test_data/${DATASET_KIND}"

    GEN_META="${TEST_DATA_DIR}/.gen_meta"
    CACHED_T=""
    [[ -f "$GEN_META" ]] && CACHED_T=$(cat "$GEN_META" 2>/dev/null)
    if [[ "$DATASET_KIND" == "mnist" ]]; then
        echo "  MNIST: main_mnist.c reads raw IDX files directly."
        python3 tools/download_data.py mnist >> "$FLOW_LOG" 2>&1 || true
    elif [[ -f "${TEST_DATA_DIR}/fall.bin" && "$CACHED_T" == "$TIMESTEPS" ]]; then
        echo "  Test data already exists for T=${TIMESTEPS}: ${TEST_DATA_DIR}"
    else
        echo "  Generating test data for ${DATASET_KIND} (T=${TIMESTEPS}) -> ${TEST_DATA_DIR}..."
        mkdir -p "${TEST_DATA_DIR}"
        case "$DATASET_KIND" in
            nmnist)
                run_logged python3 tools/export_nmnist_bin.py data "${TEST_DATA_DIR}" --timesteps "$TIMESTEPS" ;;
            cifar10dvs)
                run_logged python3 tools/export_cifar10dvs_bin.py "${TEST_DATA_DIR}" --data-path data --timesteps "$TIMESTEPS" ;;
            dvsgesture)
                run_logged python3 tools/export_dvsgesture_bin.py "${TEST_DATA_DIR}" --data-path data --max-frames "$TIMESTEPS" ;;
            *)
                echo "[Warn] no export script for dataset '${DATASET_KIND}', skipping." ;;
        esac
        [[ -f "${TEST_DATA_DIR}/fall.bin" ]] && echo "$TIMESTEPS" > "$GEN_META"
    fi
fi

# Step 5: Compile and test
echo -e "\n[5/5] Compiling and testing C++ code..."

pushd "$OUT_C_DIR" >/dev/null

# ap_fixed csim flags: with these the GCC test uses the real quantized types and
# saturation; without them it runs in float and cannot catch width issues. Custom
# route defaults to float; --csim-apfixed or [codegen] csim_apfixed opts in.
CSIM_FLAGS=""
if [[ "${CSIM_APFIXED:-0}" == "1" ]]; then
    if [[ -n "${XILINX_HLS:-}" && -d "${XILINX_HLS}/include" ]]; then
        # -isystem rather than -I: the Vitis HLS 2020.2 headers emit thousands of
        # -Wmaybe-uninitialized warnings under modern GCC, drowning out real ones.
        CSIM_FLAGS="-DCSIM_APFIXED -isystem ${XILINX_HLS}/include"
        echo "  GCC test: ap_fixed mode (CSIM_APFIXED), headers from ${XILINX_HLS}/include"
    else
        echo "  [Warn] csim_apfixed is on but XILINX_HLS has no include/; the GCC test"
        echo "         falls back to float and will not exercise ap_fixed or saturation."
    fi
fi

SRC_FILES="model.cpp reset_node.cpp"
for f in fc_layer*.cpp neuron_layer*.cpp conv_layer*.cpp dw_conv_layer*.cpp pool_layer*.cpp; do
    [ -f "$f" ] && SRC_FILES="$SRC_FILES $f"
done

if [[ -z "$DATASET_KIND" || "$DATASET_KIND" != "mnist" ]]; then
    cp "${ROOT_DIR}/tools/main_test.c" .
    SRC_FILES="main_test.c $SRC_FILES"
    run_logged g++ -std=c++11 -O2 -Wall -Wextra $GCC_WARN_FLAGS $CSIM_FLAGS -o test_full $SRC_FILES -lm
    if [[ -n "$DATASET_KIND" && -f "${ROOT_DIR}/test_data/${DATASET_KIND}/fall.bin" ]]; then
        ./test_full "${ROOT_DIR}/test_data/${DATASET_KIND}"
        TEST_RESULT=$(grep "\[RESULT\]" sw_test.log | tail -1)
    else
        echo "  Compiled test_full but no test data available. Run manually:"
        echo "  cd ${OUT_C_DIR} && ./test_full <path_to_fall.bin_dir>"
        TEST_RESULT="(skipped: no test data)"
    fi
else
    cp "${ROOT_DIR}/tools/main_mnist.c" .
    SRC_FILES="main_mnist.c $SRC_FILES"
    run_logged g++ -std=c++11 -O2 -Wall -Wextra $GCC_WARN_FLAGS $CSIM_FLAGS -o test_mnist $SRC_FILES -lm
    ./test_mnist
    TEST_RESULT=$(grep "\[RESULT\]" sw_test.log | tail -1)
fi

popd >/dev/null

echo ""
echo "  Custom-Route SW Flow Completed!"
echo "Standardized: ${STANDARDIZED_PATH}"
echo "IR: ${IR_JSON}"
echo "C++ Code:  ${OUT_C_DIR}"
echo "GCC Result: ${TEST_RESULT:-N/A}"

FINAL_PROJ="${USER_PROJECT:-$PROJ_NAME}"

SW_REPORT="${OUT_C_DIR}/sw_report.txt"
{
    echo "=== SW Flow Report ==="
    echo "Date: $(date)"
    echo "Project: ${FINAL_PROJ}"
    echo "Model: ${MODEL_CLASS}"
    echo "Weights: ${CKPT_PATH}"
    echo "Config: ${CONFIG}"
    if [[ "$USE_FIXED_QUANT" == "yes" ]]; then echo "Quant Method: PTQ (${CUSTOM_QUANT_BITS}-bit)"; fi
    if [[ -n "$INPUT_SHAPE" ]]; then echo "Input Shape: ${INPUT_SHAPE}"; fi
    if [[ -n "$SPARSE_FLAG" ]]; then echo "Sparse: enabled"; fi
    if [[ "${CSIM_APFIXED:-0}" == "1" ]]; then echo "GCC Test: ap_fixed"; else echo "GCC Test: float"; fi
    echo "Standardized: ${STANDARDIZED_PATH}"
    echo "IR: ${IR_JSON}"
    echo "C++ Code:     ${OUT_C_DIR}"
    echo "Dataset: ${DATASET_KIND:-none} (T=${TIMESTEPS})"
    echo "GCC Result: ${TEST_RESULT:-N/A}"
    echo "Test Log: ${OUT_C_DIR}/sw_test.log"
    echo "Flow Log: ${OUT_C_DIR}/sw_flow.log"
} > "${SW_REPORT}"
echo "SW Report: ${SW_REPORT}"
report_warnings
echo "$FINAL_PROJ" > "${ROOT_DIR}/.last_sw_project"
