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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# robustness-nogpu (gpu_access: none), MIVisionX checks: CPU paths with no GPU at all.
#
# In CI the job container has no /dev/kfd and no render node. On a GPU host,
# ROCR_VISIBLE_DEVICES=-1 (set by run_suite.sh --gpu-access none) emulates that:
# it leaves ROCr with zero GPU agents but keeps /dev/kfd and the render nodes
# openable; env::no-gpu-agents records which situation this run is in.
set -uo pipefail
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init robustness-nogpu

if ! vp_tier_ge comprehensive; then
  vp_skip "tier::below-comprehensive" "robustness-nogpu runs in the comprehensive tier and above"
  vp_finish
  exit 0
fi

CR="${VP_REPO}/suites/robustness/checked_run.py"
PROBES="${VP_REPO}/suites/robustness/probes"
export TMPDIR="${VP_WORK}/tmp"
mkdir -p "${TMPDIR}"

fresh_dir() {
  local d="${VP_WORK}/cwd/$1"
  rm -rf "${d}"
  mkdir -p "${d}"
  printf '%s' "${d}"
}

cr() {
  local id="$1"; shift
  "${VP_PY}" "${CR}" --id "${id}" --timeout 300 --error-rc 70 "$@"
}

# The premise: nothing may see a GPU. If this fails, every result below is suspect.
cr "env::no-gpu-agents" --timeout 120 -- "${VP_PY}" "${PROBES}/gpu_agents.py" --expect 0

# MIVisionX: a CPU graph must work; asking for the GPU must fail cleanly.
cr "mivisionx::runvx-cpu-graph" --backend CPU --env AGO_DEFAULT_TARGET=CPU --cwd "$(fresh_dir mvx_cpu)" \
  -- "${VP_PY}" "${PROBES}/runvx_graph.py" --mode correct
cr "mivisionx::runvx-gpu-request-clean-error" --backend GPU --cwd "$(fresh_dir mvx_gpu)" \
  -- "${VP_PY}" "${PROBES}/runvx_graph.py" --mode error --affinity GPU

vp_finish
exit 0
