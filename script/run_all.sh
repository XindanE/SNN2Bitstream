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

# Model source is --toml <config> or --custom
# <module.Class> (+ flags); stage is sw | full | hw (default: full)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Name to print in help text: the ./snn2bitstream forwarder passes its own name
PROG="${SNN2B_INVOKED_AS:-$(basename "${BASH_SOURCE[0]}")}"

usage() {
    cat <<EOF
Usage:
  $PROG [sw|full|hw] --toml <config.toml> [options]
  $PROG [sw|full|hw] --custom <module.Class> --weights <pt> --project <name> --timestep <T> [options]
  $PROG hw <project_name>

Stage (default: full):
  sw    train/standardize -> IR -> C++ -> GCC test
  full  sw + Xilinx (HLS + Vivado + Vitis)
  hw    Xilinx only, on an already-generated backend_projects/<name>/

Model source (pick one; not needed for hw):
  --toml   <config.toml>                              config route
  --custom <module.Class> --weights <pt> --project <name> --timestep <T> [--input-shape C,H,W]

Other options (--config, --quant, --project, --sparse, --unroll, --dataflow, --dataset, ...) are
forwarded to the route script.

Examples:
  $PROG --toml configs/mnist_fcn_rate.toml --config SPQ --quant qat_ft
  $PROG sw --toml configs/mnist_fcn_rate.toml
  $PROG --custom user_model.my.FCN --weights ckpt.pt --project myproj --timestep 10 --dataset nmnist
  $PROG hw myproj

See README.md for the full option list and their accepted values.
EOF
}

# Optional leading stage subcommand
STAGE="full"
case "${1:-}" in
    sw|full|hw) STAGE="$1"; shift ;;
    -h|--help)  usage; exit 0 ;;
esac

# hw: Xilinx only on an existing project
if [[ "$STAGE" == "hw" ]]; then
    if [[ $# -lt 1 ]]; then echo "[Error] hw needs a project name"; usage; exit 1; fi
    exec "${ROOT_DIR}/script/run_xilinx.sh" "$1"
fi

# Detect model source; with --custom, --toml only supplies [codegen]
SOURCE=""
for a in "$@"; do
    case "$a" in
        --toml)   [[ -z "$SOURCE" ]] && SOURCE="toml" ;;
        --custom) SOURCE="custom" ;;
    esac
done
if [[ -z "$SOURCE" ]]; then
    echo "[Error] provide a model source: --toml <config> or --custom <module.Class>"
    usage; exit 1
fi

# SW flow, dispatched by source
if [[ "$SOURCE" == "toml" ]]; then
    "${ROOT_DIR}/script/run_config_sw.sh" "$@"
else
    # Custom route: translate the flag-based source into run_custom_sw.sh's positional call
    MODEL="" WEIGHTS="" PROJECT="" TIMESTEP=""
    REST=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --custom)                MODEL="$2";    shift 2 ;;
            --weights)               WEIGHTS="$2";  shift 2 ;;
            --project)               PROJECT="$2";  shift 2 ;;
            --timestep|--timesteps)  TIMESTEP="$2"; shift 2 ;;
            *)                       REST+=("$1");  shift ;;
        esac
    done
    for pair in "MODEL:--custom" "WEIGHTS:--weights" "PROJECT:--project" "TIMESTEP:--timestep"; do
        name="${pair%%:*}"; flag="${pair##*:}"
        if [[ -z "${!name}" ]]; then echo "[Error] custom route needs ${flag}"; usage; exit 1; fi
    done
    "${ROOT_DIR}/script/run_custom_sw.sh" "$MODEL" "$WEIGHTS" "$PROJECT" "$TIMESTEP" ${REST[@]+"${REST[@]}"}
fi

# full: also run the Xilinx flow on the just-generated project
if [[ "$STAGE" == "full" ]]; then
    PROJ_NAME=$(cat "${ROOT_DIR}/.last_sw_project" 2>/dev/null || echo "")
    if [[ -z "$PROJ_NAME" ]]; then
        echo "[Error] .last_sw_project not found; the SW flow may have failed."
        exit 1
    fi
    "${ROOT_DIR}/script/run_xilinx.sh" "$PROJ_NAME"
fi
