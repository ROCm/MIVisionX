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

# Detect the AMD GPUs on this host from the KFD sysfs topology and pick the
# one the vision-pack build targets.
#
#   detect_gpu.sh [--manifest vision-pack-manifest.json] [--expected gfx1201]
#                 [--allow-none] [--github]
#
# Prints KEY=VALUE lines (and appends them to $GITHUB_OUTPUT and $GITHUB_ENV
# with --github):
#   VP_GPU_PRESENT      1 when a targeted GPU was chosen, 0 otherwise (--allow-none)
#   VP_NO_GPU_REASON    why VP_GPU_PRESENT is 0 (empty otherwise)
#   VP_GFX              chosen GPU architecture, e.g. gfx1201
#   VP_RENDER_MINOR     its DRM render minor (container gets /dev/dri/renderD<minor>)
#   VP_GPU_INDEX        its index among GPU agents (ROCR_VISIBLE_DEVICES on bare metal)
#   VP_SDK_FAMILY       TheRock -tests tarball family for this GPU
#   VP_GPU_LIST         every GPU as gfx:minor:index, comma-separated
#   VP_UNSUPPORTED_GPUS GPUs present but not in the manifest's gpu_targets
#
# Without a usable GPU (no KFD topology: exit 3, no GPU agents: exit 3, no GPU
# the build targets: exit 4) it fails, unless --allow-none is given: then it
# warns, prints VP_GPU_PRESENT=0 with the reason and empty GPU fields, and
# exits 0. A GPU that differs from --expected/EXPECTED_GFX always fails (exit
# 5): the runner hardware changed, which is not the same as having no GPU.
#
# The sysfs topology works before any ROCm is installed and is what
# offload-arch reads internally. A GPU is "supported" when its gfx appears in
# every gpu_targets list of the manifest (all four vision libraries).
# VP_KFD_TOPOLOGY overrides the topology directory (used by the unit tests).
#
# When ROCR_VISIBLE_DEVICES is set, non-empty and not -1 (a CI runner that
# assigns one GPU per job), only that GPU is considered: a numeric value, or
# the first entry of a comma-separated list, is its index among the GPU agents
# (VP_GPU_INDEX); GPU-<hex> is matched against the KFD node's unique_id.
# VP_GPU_LIST and VP_UNSUPPORTED_GPUS then name only that GPU. If no GPU
# agent matches, the script fails (exit 6) and never falls back to another
# GPU, even with --allow-none.
set -euo pipefail

manifest=""
expected="${EXPECTED_GFX:-}"
github=0
allow_none=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --expected) expected="$2"; shift 2 ;;
    --allow-none) allow_none=1; shift ;;
    --github) github=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

out=()
list=() unsupported=()

emit() {
  printf '%s\n' "${out[@]}"
  if [[ "${github}" == 1 ]]; then
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
      printf '%s\n' "${out[@]}" | sed 's/^VP_//; s/^\([A-Z_]*\)=/\L\1=/' >>"${GITHUB_OUTPUT}"
    fi
    if [[ -n "${GITHUB_ENV:-}" ]]; then
      printf '%s\n' "${out[@]}" >>"${GITHUB_ENV}"
    fi
  fi
}

# no_gpu <reason> <exit code without --allow-none>
no_gpu() {
  if [[ "${allow_none}" != 1 ]]; then
    echo "::error::$1" >&2
    exit "$2"
  fi
  echo "::warning::$1; continuing without a GPU (only the needs_gpu: false suites can run)" >&2
  out=(
    "VP_GPU_PRESENT=0"
    "VP_NO_GPU_REASON=$1"
    "VP_GFX="
    "VP_RENDER_MINOR="
    "VP_GPU_INDEX="
    "VP_SDK_FAMILY="
    "VP_GPU_LIST=$(IFS=,; echo "${list[*]:-}")"
    "VP_UNSUPPORTED_GPUS=$(IFS=,; echo "${unsupported[*]:-}")"
  )
  emit
  exit 0
}

topo="${VP_KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology/nodes}"
[[ -d "${topo}" ]] || no_gpu "no KFD topology at ${topo} (amdgpu driver not loaded?)" 3

family_for() {
  case "$1" in
    gfx942) echo gfx94X-dcgpu-tests ;;
    gfx950) echo gfx950-dcgpu-tests ;;
    gfx90a|gfx908|gfx906|gfx900|gfx90c) echo "$1-tests" ;;
    gfx1010|gfx1011|gfx1012) echo gfx101X-dgpu-tests ;;
    gfx103[0-6]) echo gfx103X-all-tests ;;
    gfx110[0-3]) echo gfx110X-all-tests ;;
    gfx1150|gfx1151|gfx1152|gfx1153) echo "$1-tests" ;;
    gfx1200|gfx1201) echo gfx120X-all-tests ;;
    gfx125*) echo gfx125X-dcgpu-tests ;;
    *) echo multiarch-tests ;;
  esac
}

supported() {
  local gfx="$1"
  [[ -z "${manifest}" || ! -f "${manifest}" ]] && return 0
  local ok
  ok="$(jq -r --arg g "${gfx}" '
      (.gpu_targets // {}) as $t
      | if ($t | length) == 0 then "unknown"
        elif ([$t[] | any(.[]; . == $g)] | all) then "yes" else "no" end' "${manifest}")"
  [[ "${ok}" == yes || "${ok}" == unknown ]]
}

# The GPU the runner assigned through ROCR_VISIBLE_DEVICES, if any.
want_idx="" want_uid=""
rvd="${ROCR_VISIBLE_DEVICES:-}"
if [[ -n "${rvd}" && "${rvd}" != -1 ]]; then
  sel="${rvd%%,*}"
  sel="${sel//[[:space:]]/}"
  if [[ "${sel}" =~ ^[0-9]+$ ]]; then
    want_idx="$((10#${sel}))"
  elif [[ "${sel}" =~ ^GPU-([0-9A-Fa-f]{1,16})$ ]]; then
    want_uid="$(printf '%u' "$((16#${BASH_REMATCH[1]}))")"
  else
    echo "::error::cannot select a GPU from ROCR_VISIBLE_DEVICES=${rvd} (expected an index or GPU-<hex>)" >&2
    exit 6
  fi
fi

chosen="" chosen_minor="" chosen_idx="" matched=0 seen=()
idx=0
while IFS= read -r node; do
  props="${node}/properties"
  [[ -r "${props}" ]] || continue
  v="$(awk '$1 == "gfx_target_version" {print $2}' "${props}")"
  [[ "${v:-0}" -gt 0 ]] || continue          # CPU agents report 0
  gfx="$(printf 'gfx%d%x%x' $((v / 10000)) $(((v / 100) % 100)) $((v % 100)))"
  minor="$(awk '$1 == "drm_render_minor" {print $2}' "${props}")"
  if [[ -n "${want_idx}${want_uid}" ]]; then
    uid="$(awk '$1 == "unique_id" {print $2}' "${props}")"
    seen+=("${gfx}:${minor}:${idx}:uid=${uid:-none}")
    # unique_id 0 means the GPU reports none, so it can never match a UUID.
    if [[ "${matched}" == 0 && ( "${idx}" == "${want_idx}" \
          || ( -n "${want_uid}" && "${uid:-0}" != 0 && "${uid}" == "${want_uid}" ) ) ]]; then
      matched=1
      list+=("${gfx}:${minor}:${idx}")
      if supported "${gfx}"; then
        chosen="${gfx}"; chosen_minor="${minor}"; chosen_idx="${idx}"
      else
        unsupported+=("${gfx}:${minor}:${idx}")
      fi
    fi
    idx=$((idx + 1))
    continue
  fi
  list+=("${gfx}:${minor}:${idx}")
  if supported "${gfx}"; then
    if [[ -z "${chosen}" ]]; then chosen="${gfx}"; chosen_minor="${minor}"; chosen_idx="${idx}"; fi
  else
    unsupported+=("${gfx}:${minor}:${idx}")
  fi
  idx=$((idx + 1))
done < <(ls -1d "${topo}"/* 2>/dev/null | sort -V)

if [[ -n "${want_idx}${want_uid}" ]]; then
  [[ ${#seen[@]} -gt 0 ]] || no_gpu "no GPU agents in ${topo}" 3
  if [[ "${matched}" == 0 ]]; then
    echo "::error::ROCR_VISIBLE_DEVICES=${rvd} matches no GPU agent on this host (gfx:minor:index:unique_id: ${seen[*]}); refusing to pick another GPU" >&2
    exit 6
  fi
  [[ -n "${chosen}" ]] || no_gpu "the GPU assigned by ROCR_VISIBLE_DEVICES=${rvd} (${list[*]}) is not targeted by the vision-pack build" 4
fi
[[ ${#list[@]} -gt 0 ]] || no_gpu "no GPU agents in ${topo}" 3
[[ -n "${chosen}" ]] || no_gpu "no GPU on this host is targeted by the vision-pack build (found: ${list[*]})" 4
if [[ -n "${expected}" && "${expected}" != "${chosen}" ]]; then
  echo "::error::detected ${chosen} but EXPECTED_GFX=${expected}; the runner hardware changed" >&2
  exit 5
fi

out=(
  "VP_GPU_PRESENT=1"
  "VP_NO_GPU_REASON="
  "VP_GFX=${chosen}"
  "VP_RENDER_MINOR=${chosen_minor}"
  "VP_GPU_INDEX=${chosen_idx}"
  "VP_SDK_FAMILY=$(family_for "${chosen}")"
  "VP_GPU_LIST=$(IFS=,; echo "${list[*]}")"
  "VP_UNSUPPORTED_GPUS=$(IFS=,; echo "${unsupported[*]:-}")"
)
emit
