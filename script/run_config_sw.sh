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

# Config-route SW flow: TOML, train, IR, C++, GCC test (full dataset).
# The TOML determines the model architecture (FCN/CSNN); no need to choose a script.
# Test data is cached in test_data/<dataset>/<encoding>/t<timesteps>/ and reused across projects.
#
# Config: S (float, no pragma), SQ (+ fixed-point quant), SP (+ pragma opt),
# SPQ (pragma + quant, full optimization).
#
# Quant methods (SQ/SPQ): qat_ft (pretrain FP32 then QAT fine-tune, default),
# qat (direct QAT from scratch), ptq (post-training quantization).
#
# --pretrained <path>: existing FP32 checkpoint to use as base. PTQ quantizes it
# directly without training; QAT/QAT-FT use it as initialization for fine-tuning.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"
source "${ROOT_DIR}/script/lib_flow.sh"

# Default config files
CFG_FILE="configs/mnist_fcn_rate.toml"
QAT_FT_CFG=""  # auto-derived from CFG_FILE below

# Parse all arguments
CONFIG="S"
QUANT_METHOD="qat_ft"  # Default quantization method
USER_PRETRAINED=""     # User-specified pretrained model
PARALLEL_FACTOR=""     # Parallel factor for HLS
UNROLL_FLAG=""         # --unroll: convolution unroll tags (ck,ic,oc)
DATAFLOW_FLAG=""       # --dataflow: DATAFLOW pragma on the timestep loop
SPARSE_FLAG=""         # Sparse spike-driven computation
FOLD_FLAG=""           # Fold dequant scale into LIF threshold (drops dequant multiply)
BSHIFT_FLAG=""         # Bit-shift LIF leak (mem-(mem>>k) instead of beta multiply)
MUL_FABRIC_FLAG=""     # Map multiplies to LUT fabric instead of DSP48 (binary-input FCN only)
SD_ENC_FLAG=""         # SD test-data encoding label for SD_DATA_DIR (count/spike/rate/repeat)
USER_PROJECT=""        # Project name override for A/B experiments
QUANT_BITS=""          # Quantization bit width (default: 8)
STREAMING_FLAG=""      # --streaming: generate per-stage streaming HLS projects
CONV_OC_FACTOR_FLAG="" # --conv-oc-factor: OC unroll mode (auto/0/N) for Conv2d
DATA_WIDTH_FLAG=""     # --data-width: data_t width override (avoids the default 32-bit upgrade)
DATA_INT_FLAG=""       # --data-int-width: data_t integer width (pairs with --data-width)
MAX_SAMPLES=""         # --max-samples: cap the GCC test sample count (fast subset accuracy)
CONV_OC_MAX_FLAG=""    # --conv-oc-max: auto threshold for OC unroll
PACK_SPIKES_FLAG=""    # --pack-spikes: channel-packed spike input for conv layers
CONV_PARALLEL_MAX_FLAG="" # --conv-parallel-max: limit for --unroll ic
CLI_DATA_WIDTH=""      # CLI overrides for the TOML [codegen] equivalents below
CLI_DATA_INT=""
CLI_BACKEND=""
TOML_CKPT_OVERRIDE=""  # Checkpoint from TOML [codegen].checkpoint
# Sentinels to detect CLI-explicit flags (vs defaults)
_CLI_CONFIG_SET="0"
_CLI_QUANT_SET="0"
_CLI_SPARSE_SET=""     # empty = not set; "0" = explicit --no-sparse (future)
_CLI_CSIM_SET=""       # empty = not set via CLI; "1" = explicit --csim-apfixed/--no-csim-apfixed
INVOCATION="$0 $*"     # recorded into build_information.txt for provenance
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --toml)
            CFG_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            _CLI_CONFIG_SET="1"
            shift 2
            ;;
        --quant)
            QUANT_METHOD="$2"
            _CLI_QUANT_SET="1"
            shift 2
            ;;
        --pretrained)
            USER_PRETRAINED="$2"
            shift 2
            ;;
        --parallel-factor)
            PARALLEL_FACTOR="--parallel-factor $2"
            shift 2
            ;;
        --unroll)
            UNROLL_FLAG="--unroll $2"
            shift 2
            ;;
        --dataflow)
            DATAFLOW_FLAG="--dataflow"
            shift
            ;;
        --pack-spikes)
            PACK_SPIKES_FLAG="--pack-spikes"
            shift
            ;;
        --conv-parallel-max)
            CONV_PARALLEL_MAX_FLAG="--conv-parallel-max $2"
            shift 2
            ;;
        --sparse)
            SPARSE_FLAG="--sparse"
            # optional 'sp' tag (paper syntax); consume it if present
            if [[ "${2:-}" == "sp" ]]; then shift 2; else shift; fi
            ;;
        --fold-dequant)
            FOLD_FLAG="--fold-dequant"
            shift
            ;;
        --mul-impl-fabric)
            MUL_FABRIC_FLAG="--mul-impl-fabric"
            shift
            ;;
        --sd-encoding)
            SD_ENC_FLAG="--sd-encoding $2"
            shift 2
            ;;
        --bit-shift-beta)
            BSHIFT_FLAG="--bit-shift-beta"
            shift
            ;;
        --project)
            USER_PROJECT="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
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
            shift 2
            ;;
        --streaming)
            STREAMING_FLAG="--streaming"
            shift
            ;;
        --conv-oc-factor)
            CONV_OC_FACTOR_FLAG="--conv-oc-factor $2"
            shift 2
            ;;
        --conv-oc-max)
            CONV_OC_MAX_FLAG="--conv-oc-max $2"
            shift 2
            ;;
        --data-width)
            CLI_DATA_WIDTH="$2"
            shift 2
            ;;
        --data-int-width|--data-int)
            CLI_DATA_INT="$2"
            shift 2
            ;;
        --backend)
            CLI_BACKEND="$2"
            case "$CLI_BACKEND" in
                vitis|bambu) ;;
                *) echo "[Error] unknown --backend '$CLI_BACKEND' (expected: vitis | bambu)" >&2; exit 1 ;;
            esac
            shift 2
            ;;
        --ptq)
            # Legacy support
            QUANT_METHOD="ptq"
            shift
            ;;
        --qat)
            # Legacy support
            QUANT_METHOD="qat"
            shift
            ;;
        --csim-apfixed)
            CSIM_APFIXED=1
            _CLI_CSIM_SET="1"
            shift
            ;;
        --no-csim-apfixed)
            CSIM_APFIXED=0
            _CLI_CSIM_SET="1"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate the TOML exists before anything reads it. Without this a typo'd path is
# swallowed by the `2>/dev/null || defaults` below, silently falling back to
# nmnist/T=10 and training the wrong model.
if [[ ! -f "$CFG_FILE" ]]; then
    echo "[Error] TOML config not found at '$CFG_FILE'"
    exit 1
fi

# Auto-derive QAT_FT config path (needed even if not qat_ft, to know the filename)
if [[ -z "$QAT_FT_CFG" ]]; then
    QAT_FT_CFG="${CFG_FILE%.toml}_qat_ft.toml"
fi

# Normalize config
CONFIG=$(echo "$CONFIG" | tr '[:lower:]' '[:upper:]')

# Validate config
case "$CONFIG" in
    S|SQ|SP|SPQ) ;;
    *)
        echo "[Error] Invalid config '$CONFIG'. Must be S, SQ, SP, or SPQ."
        exit 1
        ;;
esac

# Validate quant method
QUANT_METHOD=$(echo "$QUANT_METHOD" | tr '[:upper:]' '[:lower:]')
case "$QUANT_METHOD" in
    qat_ft|qat|ptq) ;;
    *)
        echo "[Error] Invalid quant method '$QUANT_METHOD'. Must be qat_ft, qat, or ptq."
        exit 1
        ;;
esac

# Detect parameters from TOML (including project name + codegen)
echo "Detecting parameters from $CFG_FILE"
read -r PROJECT_NAME TIMESTEPS DATASET_KIND SD_BINARIZE RESOLVED_ENCODING <<< $(python3 -c '
import sys, toml
cfg = toml.load(sys.argv[1])
project_name = cfg.get("project", {}).get("name", "snn_project")
params = cfg.get("model_template", {}).get("params", {})
timesteps = params.get("timesteps", 10)
ds = cfg.get("dataset", {})
dataset_kind = ds.get("kind", "nmnist").lower()
# Same spike/binarize signal the training loader uses; drives export --binarize + cache key.
binarize = bool(ds.get("binary_input", False)) or str(ds.get("input_encoding", "")).lower() == "spike"
# Resolved once here so training and export cannot infer different values.
encoding = params.get("encoding", None)
if encoding is None:
    try:
        from frontend.datasets import default_encoding
        encoding = default_encoding(dataset_kind)
    except Exception:
        encoding = None
print(project_name, timesteps, dataset_kind, int(binarize), encoding or "")
' "$CFG_FILE" 2>/dev/null) || {
    PROJECT_NAME="snn_project"
    TIMESTEPS=10
    DATASET_KIND="nmnist"
    SD_BINARIZE=0
    RESOLVED_ENCODING=""
    echo "[Warn] failed to read params from TOML, using defaults"
}
# Encoding label for the test-data cache: spike (binarized) vs count must not share a cache dir.
INPUT_ENC_LABEL="count"; [[ "$SD_BINARIZE" == "1" ]] && INPUT_ENC_LABEL="spike"

# Read [codegen] section from TOML (provides defaults; CLI flags override)
eval $(python3 - "$CFG_FILE" << 'PYEOF'
import sys, toml
cfg = toml.load(sys.argv[1])
cg = cfg.get("codegen", {})
g = cg.get
print("TOML_CONFIG=" + str(g("config", "")))
print("TOML_SPARSE=" + str(g("sparse", "")).lower())
print("TOML_FOLD=" + str(g("fold_dequant", "")).lower())
print("TOML_BSHIFT=" + str(g("bit_shift_beta", "")).lower())
print("TOML_QUANT=" + str(g("quant", "")))
print("TOML_QUANT_BITS=" + str(g("quant_bits", "")))
unroll_val = g("unroll", "")
if isinstance(unroll_val, list):
    print("TOML_UNROLL=" + ",".join(unroll_val))
elif unroll_val:
    print("TOML_UNROLL=" + str(unroll_val))
else:
    print("TOML_UNROLL=")
print("TOML_DATAFLOW=" + str(g("dataflow", "")).lower())
print("TOML_CHECKPOINT=" + str(g("checkpoint", "")))
print("TOML_PARALLEL_FACTOR=" + str(g("parallel_factor", "")))
print("TOML_STREAMING=" + str(g("streaming", "")).lower())
print("TOML_CONV_OC_FACTOR=" + str(g("conv_oc_factor", "")))
print("TOML_CONV_OC_MAX=" + str(g("conv_oc_max", "")))
print("TOML_PACK_SPIKES=" + str(g("pack_spikes", "")).lower())
print("TOML_CONV_PARALLEL_MAX=" + str(g("conv_parallel_max", "")))
print("TOML_DATA_WIDTH=" + str(g("data_width", "")))
print("TOML_SD_ENCODING=" + str(g("sd_encoding", "")))
print("TOML_CSIM_APFIXED=" + str(g("csim_apfixed", "")).lower())
bambu = cg.get("bambu", None)
if bambu is not None:
    print("TOML_BACKEND=bambu")
    # single-quote: extra may contain spaces (e.g. "--pipelining=inference --speculative-sdc-scheduling")
    print("TOML_BAMBU_OPT='" + str(bambu.get("opt", "")) + "'")
    print("TOML_BAMBU_EXTRA='" + str(bambu.get("extra", "")) + "'")
else:
    print("TOML_BACKEND=")
    print("TOML_BAMBU_OPT=")
    print("TOML_BAMBU_EXTRA=")
PYEOF
) || {
    echo "[Warn] failed to read [codegen] from TOML (section may not exist)"
    TOML_CONFIG="" TOML_SPARSE="" TOML_FOLD="" TOML_BSHIFT="" TOML_QUANT="" TOML_QUANT_BITS=""
    TOML_UNROLL="" TOML_DATAFLOW="" TOML_CHECKPOINT="" TOML_PARALLEL_FACTOR="" TOML_STREAMING=""
    TOML_CONV_OC_FACTOR="" TOML_CONV_OC_MAX="" TOML_DATA_WIDTH="" TOML_PACK_SPIKES="" TOML_CONV_PARALLEL_MAX=""
    TOML_BACKEND="" TOML_BAMBU_OPT="" TOML_BAMBU_EXTRA=""
    TOML_CSIM_APFIXED=""
}

# Apply TOML defaults where CLI did not override
if [[ -n "$TOML_CONFIG" && "$_CLI_CONFIG_SET" != "1" ]]; then
    CONFIG="$TOML_CONFIG"
fi
if [[ "$TOML_SPARSE" == "true" && -z "$SPARSE_FLAG" && "$_CLI_SPARSE_SET" != "0" ]]; then
    SPARSE_FLAG="--sparse"
fi
if [[ "$TOML_FOLD" == "true" && -z "$FOLD_FLAG" ]]; then
    FOLD_FLAG="--fold-dequant"
fi
if [[ "$TOML_BSHIFT" == "true" && -z "$BSHIFT_FLAG" ]]; then
    BSHIFT_FLAG="--bit-shift-beta"
fi
if [[ -n "$TOML_SD_ENCODING" && -z "$SD_ENC_FLAG" ]]; then
    SD_ENC_FLAG="--sd-encoding $TOML_SD_ENCODING"
fi
if [[ -n "$TOML_QUANT" && "$_CLI_QUANT_SET" != "1" ]]; then
    QUANT_METHOD="$TOML_QUANT"
fi
if [[ -n "$TOML_QUANT_BITS" && -z "$QUANT_BITS" ]]; then
    QUANT_BITS="$TOML_QUANT_BITS"
fi
if [[ -n "$TOML_UNROLL" && -z "$UNROLL_FLAG" ]]; then
    UNROLL_FLAG="--unroll $TOML_UNROLL"
fi
if [[ "$TOML_DATAFLOW" == "true" && -z "$DATAFLOW_FLAG" ]]; then
    DATAFLOW_FLAG="--dataflow"
fi
if [[ "$TOML_PACK_SPIKES" == "true" && -z "$PACK_SPIKES_FLAG" ]]; then
    PACK_SPIKES_FLAG="--pack-spikes"
fi
if [[ -n "$TOML_CONV_PARALLEL_MAX" && -z "$CONV_PARALLEL_MAX_FLAG" ]]; then
    CONV_PARALLEL_MAX_FLAG="--conv-parallel-max $TOML_CONV_PARALLEL_MAX"
fi
if [[ -n "$TOML_CHECKPOINT" && -z "$USER_PRETRAINED" ]]; then
    TOML_CKPT_OVERRIDE="$TOML_CHECKPOINT"
fi
if [[ -n "$TOML_PARALLEL_FACTOR" && -z "$PARALLEL_FACTOR" ]]; then
    PARALLEL_FACTOR="--parallel-factor $TOML_PARALLEL_FACTOR"
fi
if [[ "$TOML_STREAMING" == "true" && -z "$STREAMING_FLAG" ]]; then
    STREAMING_FLAG="--streaming"
fi
if [[ -n "$TOML_CONV_OC_FACTOR" && -z "$CONV_OC_FACTOR_FLAG" ]]; then
    CONV_OC_FACTOR_FLAG="--conv-oc-factor $TOML_CONV_OC_FACTOR"
fi
if [[ -n "$TOML_CONV_OC_MAX" && -z "$CONV_OC_MAX_FLAG" ]]; then
    CONV_OC_MAX_FLAG="--conv-oc-max $TOML_CONV_OC_MAX"
fi
if [[ -n "$TOML_DATA_WIDTH" && -z "$DATA_WIDTH_FLAG" ]]; then
    DATA_WIDTH_FLAG="--data-width $TOML_DATA_WIDTH"
fi
# CLI wins over TOML for the data_t fixed-point type. Width and integer width are a
# pair: giving only the width leaves the converter's default of int = width // 2.
if [[ -n "$CLI_DATA_WIDTH" ]]; then
    DATA_WIDTH_FLAG="--data-width $CLI_DATA_WIDTH"
fi
if [[ -n "$CLI_DATA_INT" ]]; then
    DATA_INT_FLAG="--data-int $CLI_DATA_INT"
fi
if [[ -n "$DATA_INT_FLAG" && -z "$DATA_WIDTH_FLAG" ]]; then
    echo "[Error] --data-int-width needs --data-width as well (they set ap_fixed<W,I> together)." >&2
    exit 1
fi
# csim_apfixed: compile the GCC test with real ap_fixed types so it exercises the
# quantized widths and saturation, not float. Config route defaults on; TOML/CLI override.
if [[ "$_CLI_CSIM_SET" != "1" ]]; then
    if [[ "$TOML_CSIM_APFIXED" == "false" ]]; then
        CSIM_APFIXED=0
    else
        CSIM_APFIXED=1
    fi
fi

# QAT-FT: auto-generate the _qat_ft.toml and read its project name.
if [[ "$QUANT_METHOD" == "qat_ft" ]]; then
    echo "[Info] Auto-generating QAT-FT config from base: $CFG_FILE -> $QAT_FT_CFG"
    python3 -c "
import sys, toml
cfg = toml.load(sys.argv[1])
cfg['project']['name'] = cfg['project']['name'] + '_qat_ft'
ft_epochs = cfg.get('codegen', {}).get('qat_ft_epochs', 1)
cfg['training']['epochs'] = ft_epochs
lr_factor = cfg.get('optimizer', {}).get('qat_lr_factor', 0.1)
cfg.setdefault('optimizer', {})['qat_lr_factor'] = lr_factor
with open(sys.argv[2], 'w') as f:
    toml.dump(cfg, f)
" "$CFG_FILE" "$QAT_FT_CFG"

    read -r QAT_FT_PROJECT_NAME <<< $(python3 -c '
import sys, toml
cfg = toml.load(sys.argv[1])
print(cfg.get("project", {}).get("name", ""))
' "$QAT_FT_CFG" 2>/dev/null) || {
        QAT_FT_PROJECT_NAME=""
    }
    if [[ -z "$QAT_FT_PROJECT_NAME" ]]; then
        QAT_FT_PROJECT_NAME="${PROJECT_NAME}_qat_ft"
    fi
fi

# Determine quantization settings based on config
if [[ "$CONFIG" == *"Q"* ]]; then
    QUANT_MODE="int8_fixed"
    USE_QUANT=1
else
    QUANT_MODE="none"
    USE_QUANT=0
    QUANT_METHOD="none"
fi

# Set paths based on project name and quant method.
if [[ "$QUANT_METHOD" == "qat_ft" && "$USE_QUANT" -eq 1 ]]; then
    PRETRAIN_CKPT="checkpoints/${PROJECT_NAME}.pth"
    CKPT_PTH="checkpoints/${QAT_FT_PROJECT_NAME}.pth"
    IR_DIR="ir_output/${QAT_FT_PROJECT_NAME}"
    FINAL_PROJECT_NAME="$QAT_FT_PROJECT_NAME"
else
    # Direct QAT / PTQ / float: train_model saves under the base [project].name,
    # so all paths use PROJECT_NAME directly. Only qat_ft renames its project, to
    # keep the fine-tuned result separate from the base.
    CKPT_PTH="checkpoints/${PROJECT_NAME}.pth"
    IR_DIR="ir_output/${PROJECT_NAME}"
    FINAL_PROJECT_NAME="$PROJECT_NAME"
fi

# Handle user-specified project name override (for A/B experiments)
if [[ -n "$USER_PROJECT" ]]; then
    FINAL_PROJECT_NAME="$USER_PROJECT"
fi

# One folder per project: <base>/<name>/{cpp, xilinx|bambu, build_information.txt}
# SNN2B_OUTPUT_DIR overrides the generated-project directory.
PROJECT_DIR="${SNN2B_OUTPUT_DIR:-backend_projects}/${FINAL_PROJECT_NAME}"
OUT_C_DIR="${PROJECT_DIR}/cpp"

# Handle user-specified pretrained model
if [[ -n "$USER_PRETRAINED" ]]; then
    if [[ ! -f "$USER_PRETRAINED" ]]; then
        echo "[Error] Pretrained model not found at $USER_PRETRAINED"
        exit 1
    fi
    PRETRAIN_CKPT="$USER_PRETRAINED"
fi

echo "  SNN Config-Route Software Flow"
echo "Project: $FINAL_PROJECT_NAME"
echo "Config: $CONFIG"
echo "Quant Method: $QUANT_METHOD"
echo "TOML: $CFG_FILE"
if [[ "$QUANT_METHOD" == "qat_ft" && "$USE_QUANT" -eq 1 ]]; then
echo "QAT-FT TOML:  $QAT_FT_CFG"
echo "Pretrain: $PRETRAIN_CKPT"
fi
if [[ -n "$USER_PRETRAINED" ]]; then
echo "User Pretrained: $USER_PRETRAINED"
fi
if [[ -n "$PARALLEL_FACTOR" ]]; then echo "Parallel Factor: ${PARALLEL_FACTOR#--parallel-factor }"; fi
if [[ -n "$UNROLL_FLAG" ]]; then echo "Unroll: ${UNROLL_FLAG#--unroll }"; fi
if [[ -n "$PACK_SPIKES_FLAG" ]]; then echo "Pack spikes: enabled"; fi
if [[ -n "$DATAFLOW_FLAG" ]]; then echo "Dataflow: enabled"; fi
if [[ -n "$CONV_OC_FACTOR_FLAG" ]]; then echo "Conv OC: ${CONV_OC_FACTOR_FLAG#--conv-oc-factor }"; fi
if [[ -n "$CONV_OC_MAX_FLAG" ]]; then echo "Conv OC max: ${CONV_OC_MAX_FLAG#--conv-oc-max }"; fi
if [[ -n "$SPARSE_FLAG" ]]; then echo "Sparse: enabled"; fi
if [[ -n "$QUANT_BITS" ]]; then echo "Quant Bits: $QUANT_BITS"; fi
if [[ -n "$STREAMING_FLAG" ]]; then echo "Streaming: enabled"; fi
if [[ -n "$TOML_CKPT_OVERRIDE" ]]; then echo "TOML Ckpt: $TOML_CKPT_OVERRIDE (skip training)"; fi
echo "Checkpoint: $CKPT_PTH"
echo "IR Dir: $IR_DIR"
echo "Output: $OUT_C_DIR"
echo "Timesteps: $TIMESTEPS"
echo "Dataset: $DATASET_KIND"

mkdir -p "${OUT_C_DIR}"
start_log "${ROOT_DIR}/${OUT_C_DIR}/sw_flow.log"

# Step 1: Train model
echo -e "\n[1/5] Training model..."

if [[ -n "$TOML_CKPT_OVERRIDE" ]]; then
    echo "     Using checkpoint from TOML [codegen].checkpoint: $TOML_CKPT_OVERRIDE"
    if [[ ! -f "$TOML_CKPT_OVERRIDE" ]]; then
        echo "[Error] TOML checkpoint not found at $TOML_CKPT_OVERRIDE"
        exit 1
    fi
    REAL_SRC=$(realpath "$TOML_CKPT_OVERRIDE")
    REAL_DST=$(realpath "$CKPT_PTH" 2>/dev/null || echo "")
    if [[ "$REAL_SRC" != "$REAL_DST" ]]; then
        cp "$TOML_CKPT_OVERRIDE" "$CKPT_PTH"
    fi
    echo "     Skipping training..."

elif [[ "$QUANT_METHOD" == "qat_ft" && "$USE_QUANT" -eq 1 ]]; then
    if [[ -n "$USER_PRETRAINED" ]]; then
        echo "     Using user-specified pretrained model: $USER_PRETRAINED"
        echo "     Skipping pretraining step..."
    else
        echo "     Step 1a: Pretraining FP32 model..."
        python3 frontend/train_model.py "$CFG_FILE" 2>&1 | tee -a "$FLOW_LOG"
    fi
    echo "     Step 1b: QAT Fine-Tuning from pretrained model..."
    python3 frontend/train_model.py "$QAT_FT_CFG" --qat --pretrained "$PRETRAIN_CKPT" 2>&1 | tee -a "$FLOW_LOG"

elif [[ "$QUANT_METHOD" == "qat" && "$USE_QUANT" -eq 1 ]]; then
    if [[ -n "$USER_PRETRAINED" ]]; then
        echo "     QAT Fine-Tuning from user pretrained model: $USER_PRETRAINED"
        python3 frontend/train_model.py "$CFG_FILE" --qat --pretrained "$USER_PRETRAINED" 2>&1 | tee -a "$FLOW_LOG"
    else
        echo "     Using Quantization-Aware Training (QAT) from scratch"
        python3 frontend/train_model.py "$CFG_FILE" --qat 2>&1 | tee -a "$FLOW_LOG"
    fi

elif [[ "$QUANT_METHOD" == "ptq" && "$USE_QUANT" -eq 1 ]]; then
    if [[ -n "$USER_PRETRAINED" ]]; then
        echo "     Using user pretrained model for PTQ: $USER_PRETRAINED"
        REAL_SRC=$(realpath "$USER_PRETRAINED")
        REAL_DST=$(realpath "$CKPT_PTH" 2>/dev/null || echo "")
        if [[ "$REAL_SRC" != "$REAL_DST" ]]; then cp "$USER_PRETRAINED" "$CKPT_PTH"; fi
    else
        echo "     Using standard FP32 training (for PTQ)"
        python3 frontend/train_model.py "$CFG_FILE" 2>&1 | tee -a "$FLOW_LOG"
    fi

else
    # FP32 mode (S or SP)
    if [[ -n "$USER_PRETRAINED" ]]; then
        echo "     Using user pretrained model: $USER_PRETRAINED"
        REAL_SRC=$(realpath "$USER_PRETRAINED")
        REAL_DST=$(realpath "$CKPT_PTH" 2>/dev/null || echo "")
        if [[ "$REAL_SRC" != "$REAL_DST" ]]; then cp "$USER_PRETRAINED" "$CKPT_PTH"; fi
    else
        echo "     Using standard FP32 training"
        python3 frontend/train_model.py "$CFG_FILE" 2>&1 | tee -a "$FLOW_LOG"
    fi
fi

if [[ ! -f "$CKPT_PTH" ]]; then
    echo "[Error] checkpoint not found at $CKPT_PTH"
    exit 1
fi

# Step 2: Export IR
echo -e "\n[2/5] Exporting IR (quant_mode=$QUANT_MODE, method=$QUANT_METHOD)..."

QUANT_BITS_ARG=""
if [[ -n "$QUANT_BITS" ]]; then QUANT_BITS_ARG="--quant-bits $QUANT_BITS"; fi

QAT_FLAG=""
if [[ "$QUANT_METHOD" == "qat_ft" || "$QUANT_METHOD" == "qat" ]] && [[ "$USE_QUANT" -eq 1 ]]; then
    QAT_FLAG="--qat"
fi

# When the toml declares a spike/binarized input ($SD_BINARIZE), force input_is_binary=true into
# the IR so fabric is recognized even for old checkpoints predating the data-detected field.
# Otherwise leave it 'auto' (checkpoint field, rate-encoding fallback).
IS_BINARY_FLAG=""
[[ "$SD_BINARIZE" == "1" ]] && IS_BINARY_FLAG="--input-is-binary true"
ENCODING_FLAG=""
[[ -n "$RESOLVED_ENCODING" ]] && ENCODING_FLAG="--encoding $RESOLVED_ENCODING"

run_logged python3 frontend/export_ir.py "$CKPT_PTH" \
    --timesteps "$TIMESTEPS" --out-dir "$IR_DIR" \
    --quant-mode "$QUANT_MODE" $QAT_FLAG $QUANT_BITS_ARG \
    --dataset-kind "$DATASET_KIND" $IS_BINARY_FLAG $ENCODING_FLAG

if [[ ! -f "$IR_DIR/ir.json" ]]; then
    echo "[Error] IR not found at $IR_DIR/ir.json"
    exit 1
fi

# Step 3: Convert to C++
echo -e "\n[3/5] Converting IR to C++ (config=$CONFIG)..."
CONVERTER_PROJECT_ARG=""
if [[ -n "$USER_PROJECT" ]]; then CONVERTER_PROJECT_ARG="--project $USER_PROJECT"; fi

# Backend routing: --backend on the CLI, else a [codegen.bambu] section in the TOML.
# The TOML section is what carries the bambu tuning options, so --backend bambu on its
# own runs bambu at its defaults.
BACKEND_ARG=""
BAMBU_OPT_ARG=""
BAMBU_EXTRA_ARG=""
BACKEND="${CLI_BACKEND:-${TOML_BACKEND:-}}"
case "$BACKEND" in
    "") ;;
    vitis) BACKEND_ARG="--backend vitis" ;;
    bambu)
        BACKEND_ARG="--backend bambu"
        # '=' syntax required: -O2 / --pipelining=... look like flags to argparse
        if [[ -n "${TOML_BAMBU_OPT:-}" ]];   then BAMBU_OPT_ARG="--bambu-opt=${TOML_BAMBU_OPT}"; fi
        if [[ -n "${TOML_BAMBU_EXTRA:-}" ]]; then BAMBU_EXTRA_ARG="--bambu-extra=${TOML_BAMBU_EXTRA}"; fi
        echo "  Backend: bambu (opt='${TOML_BAMBU_OPT}' extra='${TOML_BAMBU_EXTRA}')"
        ;;
    *)
        echo "[Error] unknown --backend '$BACKEND' (expected: vitis | bambu)" >&2
        exit 1
        ;;
esac

run_logged python3 converter/converter.py "$IR_DIR/ir.json" --config "$CONFIG" \
    $PARALLEL_FACTOR $UNROLL_FLAG $PACK_SPIKES_FLAG $CONV_PARALLEL_MAX_FLAG $DATAFLOW_FLAG $SPARSE_FLAG $FOLD_FLAG $BSHIFT_FLAG \
    $MUL_FABRIC_FLAG $SD_ENC_FLAG $CONVERTER_PROJECT_ARG $STREAMING_FLAG \
    $CONV_OC_FACTOR_FLAG $CONV_OC_MAX_FLAG $DATA_WIDTH_FLAG $DATA_INT_FLAG \
    $BACKEND_ARG $BAMBU_OPT_ARG ${BAMBU_EXTRA_ARG:+"$BAMBU_EXTRA_ARG"}

# Step 4: Prepare test data (cached in test_data/<dataset>/<encoding>/t<timesteps>/)
echo -e "\n[4/5] Preparing test data..."

# Cache test data separately for each dataset, timestep count, and encoding.
TEST_DATA_DIR="${ROOT_DIR}/test_data/${DATASET_KIND}/${INPUT_ENC_LABEL}/t${TIMESTEPS}"

if [[ "$DATASET_KIND" == "mnist" ]]; then
    echo "  MNIST: main_mnist.c reads raw IDX files directly."
    python3 tools/download_data.py mnist >> "$FLOW_LOG" 2>&1 || true
else
    # Cache is keyed on timesteps AND encoding: .gen_meta records "T_encoding" so a T change
    # OR a count<->spike (binarize) change forces regeneration: same T with different encoding
    # is different data and must not silently reuse the cache.
    GEN_META="${TEST_DATA_DIR}/.gen_meta"
    CACHE_KEY="${TIMESTEPS}_${INPUT_ENC_LABEL}"
    CACHED_KEY=""
    [[ -f "$GEN_META" ]] && CACHED_KEY=$(cat "$GEN_META" 2>/dev/null)
    BIN_FLAG=""; [[ "$SD_BINARIZE" == "1" ]] && BIN_FLAG="--binarize"
    if [[ -f "${TEST_DATA_DIR}/fall.bin" && "$CACHED_KEY" == "$CACHE_KEY" ]]; then
        echo "  Test data already exists for T=${TIMESTEPS} enc=${INPUT_ENC_LABEL}: ${TEST_DATA_DIR}"
    else
        echo "  Generating test data for ${DATASET_KIND} (T=${TIMESTEPS} enc=${INPUT_ENC_LABEL}) -> ${TEST_DATA_DIR}..."
        mkdir -p "${TEST_DATA_DIR}"
        case "$DATASET_KIND" in
            nmnist)
                run_logged python3 tools/export_nmnist_bin.py data "${TEST_DATA_DIR}" --timesteps "$TIMESTEPS" $BIN_FLAG
                ;;
            cifar10dvs)
                run_logged python3 tools/export_cifar10dvs_bin.py "${TEST_DATA_DIR}" --data-path data --timesteps "$TIMESTEPS"
                ;;
            dvsgesture)
                run_logged python3 tools/export_dvsgesture_bin.py "${TEST_DATA_DIR}" --data-path data --max-frames "$TIMESTEPS"
                ;;
            *)
                echo "[Warn] no export script for dataset '${DATASET_KIND}', skipping test data."
                ;;
        esac
        # Record the (timesteps, encoding) the cache was built with (only after data was produced).
        [[ -f "${TEST_DATA_DIR}/fall.bin" ]] && echo "$CACHE_KEY" > "$GEN_META"
    fi
fi

# Step 5: Compile and test
echo -e "\n[5/5] Compiling and testing C++ code..."

if [[ "${SNN2B_SKIP_TEST:-0}" == "1" ]]; then
    echo "  [SNN2B_SKIP_TEST=1] Skipping csim compile+test; the generated C++ is ready for HLS."
    TEST_RESULT="(skipped: SNN2B_SKIP_TEST)"
else

mkdir -p "${OUT_C_DIR}"

# ap_fixed csim flags: with these the GCC test uses the real quantized types and
# saturation; without them it runs in float and cannot catch width issues. Falls
# back to float (with a warning) when XILINX_HLS is not available.
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

if [[ "$DATASET_KIND" == "mnist" ]]; then
    cp "tools/main_mnist.c" "$OUT_C_DIR/"
    pushd "$OUT_C_DIR" >/dev/null

    SRC_FILES="main_mnist.c model.cpp reset_node.cpp"
    for f in fc_layer*.cpp neuron_layer*.cpp conv_layer*.cpp dw_conv_layer*.cpp pool_layer*.cpp; do
        [ -f "$f" ] && SRC_FILES="$SRC_FILES $f"
    done

    run_logged g++ -std=c++11 -O2 -Wall -Wextra $GCC_WARN_FLAGS $CSIM_FLAGS -o test_mnist $SRC_FILES -lm
    ./test_mnist "${ROOT_DIR}/data/MNIST"
    TEST_RESULT=$(grep "\[RESULT\]" sw_test.log | tail -1)
    popd >/dev/null
else
    cp "tools/main_test.c" "$OUT_C_DIR/"
    pushd "$OUT_C_DIR" >/dev/null

    SRC_FILES="main_test.c model.cpp reset_node.cpp"
    for f in fc_layer*.cpp neuron_layer*.cpp conv_layer*.cpp dw_conv_layer*.cpp pool_layer*.cpp; do
        [ -f "$f" ] && SRC_FILES="$SRC_FILES $f"
    done

    run_logged g++ -std=c++11 -O2 -Wall -Wextra $GCC_WARN_FLAGS $CSIM_FLAGS -o test_full $SRC_FILES -lm
    if [[ -f "${TEST_DATA_DIR}/fall.bin" ]]; then
        if [[ -n "$MAX_SAMPLES" ]]; then
            echo "  Test capped at ${MAX_SAMPLES} samples"
        fi
        # argv[2] (MAX_SAMPLES) caps the sample count; empty = full set.
        ./test_full "${TEST_DATA_DIR}" ${MAX_SAMPLES}
        TEST_RESULT=$(grep "\[RESULT\]" sw_test.log | tail -1)
    else
        echo "[Info] No test data for '${DATASET_KIND}'; skipping the GCC accuracy test."
        echo "       Code generation finished. To run the test, put your data at"
        echo "       ${TEST_DATA_DIR#"${ROOT_DIR}/"}/{fall.bin,labels.bin} in the expected format"
        echo "       (fall.bin: N x ENCODED_SIZE float32, row-major; labels.bin: N int32)."
        TEST_RESULT="(skipped: no test data for ${DATASET_KIND})"
    fi
    popd >/dev/null
fi
fi  # end SNN2B_SKIP_TEST guard

echo ""
echo "  Config-Route SW Flow Completed!"
echo "Checkpoint: $CKPT_PTH"
echo "IR: $IR_DIR/ir.json"
echo "C++ Code:   $OUT_C_DIR"
if [[ "$DATASET_KIND" == "mnist" ]]; then
echo "Test: $OUT_C_DIR/test_mnist"
else
echo "Test: $OUT_C_DIR/test_full  (data: ${TEST_DATA_DIR})"
fi
echo "GCC Result: ${TEST_RESULT:-N/A}"

# Save SW report
SW_REPORT="${OUT_C_DIR}/sw_report.txt"
{
    echo "=== SW Flow Report ==="
    echo "Date: $(date)"
    echo "Project: ${FINAL_PROJECT_NAME}"
    echo "Config: ${CONFIG}"
    echo "Quant Method: ${QUANT_METHOD}"
    if [[ -n "$QUANT_BITS" ]]; then echo "Quant Bits: ${QUANT_BITS}"; fi
    if [[ -n "$UNROLL_FLAG" ]]; then echo "Unroll: ${UNROLL_FLAG#--unroll }"; fi
    if [[ -n "$PACK_SPIKES_FLAG" ]]; then echo "Pack spikes: enabled"; fi
if [[ -n "$DATAFLOW_FLAG" ]]; then echo "Dataflow: enabled"; fi
    if [[ -n "$CONV_OC_FACTOR_FLAG" ]]; then echo "Conv OC: ${CONV_OC_FACTOR_FLAG#--conv-oc-factor }"; fi
    if [[ -n "$CONV_OC_MAX_FLAG" ]]; then echo "Conv OC max: ${CONV_OC_MAX_FLAG#--conv-oc-max }"; fi
    if [[ -n "$SPARSE_FLAG" ]]; then echo "Sparse: enabled"; fi
    echo "Checkpoint: ${CKPT_PTH}"
    echo "IR: ${IR_DIR}/ir.json"
    echo "C++ Code:     ${OUT_C_DIR}"
    echo "Dataset: ${DATASET_KIND} (T=${TIMESTEPS})"
    echo "GCC Result: ${TEST_RESULT:-N/A}"
    echo "Test Log: ${OUT_C_DIR}/sw_test.log"
    echo "Flow Log: ${OUT_C_DIR}/sw_flow.log"
} > "${SW_REPORT}"
echo "SW Report: ${SW_REPORT}"
report_warnings

# Append SW-flow provenance to build_information.txt (converter wrote the codegen part).
BUILD_INFO="${PROJECT_DIR}/build_information.txt"
if [[ -f "$BUILD_INFO" ]]; then
{
    echo ""
    echo "sw flow"
    echo "date:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "toml:       ${CFG_FILE}"
    echo "checkpoint: ${CKPT_PTH}"
    echo "config:     ${CONFIG}  quant: ${QUANT_METHOD}${QUANT_BITS:+ ${QUANT_BITS}bit}"
    echo "command:    ${INVOCATION}"
    echo "gcc_result: ${TEST_RESULT:-N/A}"
} >> "$BUILD_INFO"
fi

echo "$FINAL_PROJECT_NAME" > "${ROOT_DIR}/.last_sw_project"
