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

# Logging helpers shared by the SW and HW flow scripts.

GCC_WARN_FLAGS="-Wno-unknown-pragmas -Wno-unused-label -Wno-unused-function -Wno-comment"
export PYTHONWARNINGS="ignore::FutureWarning"

start_log() {
    FLOW_LOG="$1"
    mkdir -p "$(dirname "$FLOW_LOG")"
    : > "$FLOW_LOG"
}

fail_with_log() {
    echo "  [Error] $1"
    echo "  ---- last 40 lines of ${FLOW_LOG#"${ROOT_DIR}/"} ----"
    tail -n 40 "$FLOW_LOG"
    exit 1
}

run_logged() {
    echo "\$ $*" >> "$FLOW_LOG"
    if ! "$@" >> "$FLOW_LOG" 2>&1; then
        fail_with_log "step failed: $*"
    fi
}

format_time() {
    printf "%dm%02ds" $(( $1 / 60 )) $(( $1 % 60 ))
}

run_timed() {
    local label="$1"; shift
    echo "\$ $*" >> "$FLOW_LOG"
    local start=$SECONDS status=0 pid
    # </dev/null: a background job that reads the terminal gets stopped.
    "$@" < /dev/null >> "$FLOW_LOG" 2>&1 &
    pid=$!
    trap "kill $pid 2>/dev/null" INT TERM
    while kill -0 "$pid" 2>/dev/null; do
        [[ -t 1 ]] && printf "\r  %s ... %s" "$label" "$(format_time $(( SECONDS - start )))"
        sleep 1
    done
    wait "$pid" || status=$?
    trap - INT TERM
    [[ -t 1 ]] && printf "\r"
    if [[ $status -ne 0 ]]; then
        echo "  ${label} ... failed after $(format_time $(( SECONDS - start )))"
        fail_with_log "step failed: $*"
    fi
    echo "  ${label} ... done ($(format_time $(( SECONDS - start ))))"
}

report_warnings() {
    local count
    count=$(grep -c "^ *\[Warn\]" "$FLOW_LOG" || true)
    if [[ "$count" -gt 0 ]]; then
        echo "Warnings: ${count} (see ${FLOW_LOG#"${ROOT_DIR}/"})"
    fi
}
