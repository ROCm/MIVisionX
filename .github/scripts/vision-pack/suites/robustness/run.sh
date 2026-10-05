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

# Robustness (gpu_access: all), MIVisionX checks: unsupported-GPU negatives and exit-code honesty.
#
# Device indices come from enumerating GPU agents with rocminfo in this
# environment (ROCR_VISIBLE_DEVICES unset), never from host indices: inside a
# container ROCr numbers only the GPUs whose render nodes were passed in.
# Every probe that uses a GPU pins exactly one with ROCR_VISIBLE_DEVICES, and
# GPU work runs serially.
set -uo pipefail
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init robustness

if ! vp_tier_ge comprehensive; then
  vp_skip "tier::below-comprehensive" "robustness runs in the comprehensive tier and above"
  vp_finish
  exit 0
fi

CR="${VP_REPO}/suites/robustness/checked_run.py"
PROBES="${VP_SUITE_DIR}/probes"
RVT="${ROCM_PATH}/share/mivisionx/test/vision_tests/runVisionTests.py"
export TMPDIR="${VP_WORK}/tmp"
mkdir -p "${TMPDIR}"

fresh_dir() {
  local d="${VP_WORK}/cwd/$1"
  rm -rf "${d}"
  mkdir -p "${d}"
  printf '%s' "${d}"
}

# cr <id> [checked_run options] -- cmd... ; probes report a crashed child with exit 70.
cr() {
  local id="$1"; shift
  "${VP_PY}" "${CR}" --id "${id}" --timeout 300 --error-rc 70 "$@"
}

# ---------------------------------------------------------------------------
# GPU agents in ROCr order, as this process sees them.
# ---------------------------------------------------------------------------
mapfile -t AGENTS < <(env -u ROCR_VISIBLE_DEVICES timeout -k 5 60 rocminfo 2>/dev/null \
  | awk '/^  Name:/ { n = $2 } /^ *Device Type:/ { if ($3 == "GPU") print n }')
vp__log "GPU agents (ROCr order): ${AGENTS[*]:-none}; VP_GFX=${VP_GFX:-none}; VP_UNSUPPORTED_GPUS=${VP_UNSUPPORTED_GPUS:-none}"

# agent_index <gfx> <occurrence>: index of the n-th (0-based) GPU agent with that gfx, or empty.
agent_index() {
  local gfx="$1" want="$2" i seen=0
  for i in "${!AGENTS[@]}"; do
    if [[ "${AGENTS[$i]}" == "${gfx}" ]]; then
      if [[ "${seen}" == "${want}" ]]; then echo "${i}"; return; fi
      seen=$((seen + 1))
    fi
  done
}

CHOSEN_IDX=""
[[ -n "${VP_GFX}" ]] && CHOSEN_IDX="$(agent_index "${VP_GFX}" 0)"

# ---------------------------------------------------------------------------
# Unsupported-GPU negatives, with the same probe on the chosen GPU as the control.
# ---------------------------------------------------------------------------
# gpu_probes <group> <agent index> <mode>
gpu_probes() {
  local g="$1" idx="$2" mode="$3" tag="${1//[^A-Za-z0-9]/_}"
  local -a env=(--backend GPU --env "ROCR_VISIBLE_DEVICES=${idx}")
  cr "${g}::mivisionx-gpu-graph" "${env[@]}" --cwd "$(fresh_dir "${tag}_mvx")" \
    -- "${VP_PY}" "${PROBES}/runvx_graph.py" --mode "${mode}" --affinity GPU
}

if [[ -z "${VP_UNSUPPORTED_GPUS:-}" ]]; then
  vp_skip "unsupported-gpu::none-present" "no unsupported GPU on this runner"
elif [[ ${#AGENTS[@]} -eq 0 ]]; then
  vp_result "unsupported-gpu::agent-enumeration" error "rocminfo lists no GPU agents although VP_UNSUPPORTED_GPUS=${VP_UNSUPPORTED_GPUS}"
else
  if [[ -n "${CHOSEN_IDX}" ]]; then
    gpu_probes "unsupported-gpu.control" "${CHOSEN_IDX}" correct
  else
    vp_result "unsupported-gpu.control::device-visible" error "${VP_GFX:-no chosen GPU} is not among the GPU agents (${AGENTS[*]})"
  fi
  declare -A occurrence=()
  IFS=, read -r -a unsupported <<<"${VP_UNSUPPORTED_GPUS}"
  for u in "${unsupported[@]}"; do
    gfx="${u%%:*}"
    [[ -n "${gfx}" ]] || continue
    n="${occurrence[${gfx}]:-0}"
    occurrence[${gfx}]=$((n + 1))
    group="unsupported-gpu.${gfx}"
    [[ "${n}" -gt 0 ]] && group="${group}.${n}"
    idx="$(agent_index "${gfx}" "${n}")"
    if [[ -z "${idx}" ]]; then
      vp_result "${group}::device-visible" error "${gfx} (${u}) is not among the GPU agents here (${AGENTS[*]})"
      continue
    fi
    vp__log "${group}: ROCR_VISIBLE_DEVICES=${idx}"
    gpu_probes "${group}" "${idx}" honest
  done
fi

# ---------------------------------------------------------------------------
# Exit-code honesty: controlled failing cases that must end with a non-zero status.
# ---------------------------------------------------------------------------
# H16: runVisionTests.py ignores runvx's exit status. A runvx that always fails must fail the run.
FAKE="${VP_WORK}/fake-runvx"
rm -rf "${FAKE}"
mkdir -p "${FAKE}"
printf '#!/bin/sh\necho "fake runvx: simulated failure" >&2\nexit 1\n' >"${FAKE}/runvx"
chmod +x "${FAKE}/runvx"
cr "exit-code::runvisiontests-ignores-runvx-failure" --expect-nonzero --require "fake runvx: simulated failure" \
  --cwd "$(fresh_dir runvisiontests)" -- "${VP_PY}" "${RVT}" --runvx_directory "${FAKE}" \
  --hardware_mode CPU --backend_type CPU --test_filter 1 --num_frames 1

vp_finish
exit 0
