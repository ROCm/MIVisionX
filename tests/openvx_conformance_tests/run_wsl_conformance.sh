#!/usr/bin/env bash
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# run_wsl_conformance.sh — run OpenVX 1.3.2 conformance inside WSL
#
# Usage (from the repo root in WSL bash):
#   ./tests/openvx_conformance_tests/run_wsl_conformance.sh [HOST|HIP|ALL]
#
# The optional positional argument selects the backend (default: HOST).
# Logs land in ~/mivisionx-conformance/ (the runConformanceTests.py default).
#
# Requirements:
#   HOST backend: cmake, gcc/clang, python3, git
#   HIP  backend: above + ROCm installed at /opt/rocm (or $ROCM_PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BACKEND="${1:-HOST}"
BACKEND="${BACKEND^^}"

# Consume the backend positional so it is not forwarded again via "$@" below;
# otherwise runConformanceTests.py's argparse sees an unexpected positional and
# exits. Only shift when a positional was actually supplied.
if [[ $# -gt 0 ]]; then
    shift
fi

case "${BACKEND}" in
  HOST|HIP|ALL) ;;
  *)
    echo "ERROR: unknown backend '${BACKEND}'. Choose HOST, HIP, or ALL."
    exit 2
    ;;
esac

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' not found. Install it and re-run."
        exit 1
    fi
}

require_cmd cmake
require_cmd python3
require_cmd git

if [[ "${BACKEND}" == "HIP" || "${BACKEND}" == "ALL" ]]; then
    if [[ ! -d "${ROCM_PATH}" ]]; then
        echo "ERROR: ROCm not found at ${ROCM_PATH}."
        echo "  Install ROCm 7.13+ (https://rocm.docs.amd.com/en/latest/install/rocm.html)"
        echo "  or set ROCM_PATH to the correct prefix."
        exit 1
    fi
    if [[ ! -f "${ROCM_PATH}/lib/libamdhip64.so" ]]; then
        echo "WARNING: ${ROCM_PATH}/lib/libamdhip64.so not found — HIP tests may fail."
    fi
fi

# ---------------------------------------------------------------------------
# Build dependencies (Ubuntu / Debian WSL)
# ---------------------------------------------------------------------------

install_build_deps() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "--- Installing build dependencies via apt-get ---"
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends \
            build-essential cmake git python3
    fi
}

if ! command -v make >/dev/null 2>&1; then
    install_build_deps
fi

# ---------------------------------------------------------------------------
# Run conformance
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo " MIVisionX OpenVX 1.3.2 Conformance — WSL"
echo " Backend : ${BACKEND}"
echo " Repo    : ${REPO_ROOT}"
echo " ROCm    : ${ROCM_PATH}"
echo "========================================================"
echo ""

# Export ROCm into PATH / LD_LIBRARY_PATH so cmake and the CTS linker find
# HIP headers and libraries when building the HIP backend.
if [[ "${BACKEND}" == "HIP" || "${BACKEND}" == "ALL" ]]; then
    export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    # Disable lazy HIP initialisation messages that pollute test output.
    export AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}"
fi

python3 "${SCRIPT_DIR}/runConformanceTests.py" \
    --backend_type "${BACKEND}" \
    --jobs "$(nproc)" \
    "$@"

echo ""
echo "Conformance run complete. Logs: ~/mivisionx-conformance/"
