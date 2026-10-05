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

# Run one suite directly in the CI job container (no Docker inside the job),
# against a ROCm + vision-pack prefix made by prepare_rocm.sh.
#
#   run_suite.sh --suite NAME --prefix DIR --out DIR [--tier T] [--gfx GFX]
#       [--data DIR] [--cache DIR] [--dist-tarball FILE]
#       [--gpu-access chosen|all|none]
#
# --tier defaults to $VP_TIER, else comprehensive; --gfx defaults to $VP_GFX.
# The entrypoint (default suites/NAME/run.sh), timeout_minutes and gpu_access
# come from suites/suites.yaml next to build_tools/; --gpu-access overrides
# gpu_access. The suite gets the environment contract of build_tools/lib/vp.sh
# with VP_NO_CONTAINER=1: core dumps are suppressed, Python uses the hostsafe
# excepthook, and deliberate crash reproducers run only with
# VP_ALLOW_CRASH_TESTS=1. --data must contain rocal_data/; --cache (default
# $RUNNER_TEMP/vp/cache/suites) holds downloads such as the OpenVX CTS clone.
#
# GPU visibility: gpu_env.sh is sourced first, so HIP_VISIBLE_DEVICES is gone.
# chosen and all keep ROCR_VISIBLE_DEVICES exactly as the runner set it (a CI
# job has no other GPU to offer; without ROCR_VISIBLE_DEVICES, chosen pins
# VP_GPU_INDEX from detect_gpu.sh when that is set). none sets
# ROCR_VISIBLE_DEVICES=-1, which leaves no GPU agent. VP_UNSUPPORTED_GPUS is
# always empty, so no other job's GPU is touched.
#
# The suite runs under timeout -k 60 with a budget of timeout_minutes * 60 -
# 240 seconds (at least 60). A suite's run.sh exits 0 once it has recorded its
# results, so a non-zero exit is an infrastructure problem (harness crash,
# time budget exhausted): <out>/runner-status.json records it, merge.py turns
# it into <suite>::infra::runner, and this script exits with the same code.
# Files under the prefix or the dataset that change during the run are
# reported as warnings: suites must treat both as read-only.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suite="" prefix="" out="" tier="${VP_TIER:-comprehensive}" gfx="${VP_GFX:-}" data="" cache="" dist="" access=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --tier) tier="$2"; shift 2 ;;
    --gfx) gfx="$2"; shift 2 ;;
    --data) data="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --dist-tarball) dist="$2"; shift 2 ;;
    --gpu-access) access="$2"; shift 2 ;;
    -h|--help) sed -n '/^# Run one suite directly/,/^set -euo pipefail/{/^set /!p}' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "${suite}" && -n "${prefix}" && -n "${out}" ]] || { echo "--suite, --prefix and --out are required" >&2; exit 2; }
case "${tier}" in quick|standard|comprehensive|full) ;; *) echo "--tier must be quick, standard, comprehensive or full" >&2; exit 2 ;; esac
case "${access}" in ""|chosen|all|none) ;; *) echo "--gpu-access must be chosen, all or none" >&2; exit 2 ;; esac
command -v jq >/dev/null || { echo "::error::run_suite: jq is required" >&2; exit 2; }

mkdir -p "${out}"
out="$(cd "${out}" && pwd)"
budget=0

# runner-status.json as merge.py reads it: {suite, rc, reason, budget_s}.
write_status() {
  jq -n --arg suite "${suite}" --argjson rc "$1" --arg reason "$2" --argjson budget "${budget}" \
    '{suite:$suite, rc:$rc, reason:$reason, budget_s:$budget}' >"${out}/runner-status.json"
}
config_error() {
  echo "::error title=${suite}::run_suite: $*" >&2
  write_status 2 "$*"
  exit 2
}

py="${VP_PY:-}"
if [[ -z "${py}" ]]; then
  if [[ -x /usr/bin/python3 ]]; then py=/usr/bin/python3; else py=python3; fi
fi
yaml="${KIT}/suites/suites.yaml"
[[ -f "${yaml}" ]] || config_error "${yaml} does not exist"
if ! fields="$("${py}" - "${yaml}" "${suite}" <<'EOF'
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is missing (apt-get install python3-yaml)")
    sys.exit(3)
path, name = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    suites = (yaml.safe_load(f) or {}).get("suites") or {}
s = suites.get(name)
if not isinstance(s, dict):
    print(f"suite {name!r} is not in {path} (known: {', '.join(sorted(suites)) or 'none'})")
    sys.exit(3)
print(s.get("entrypoint") or f"suites/{name}/run.sh")
print(int(s.get("timeout_minutes", 60)))
print(s.get("gpu_access") or "chosen")
EOF
)"; then
  config_error "${fields:-cannot read ${yaml}}"
fi
{ read -r entry; read -r timeout_min; read -r yaml_access; } <<<"${fields}"
access="${access:-${yaml_access}}"
case "${access}" in chosen|all|none) ;; *) config_error "suites.yaml gives ${suite} gpu_access ${access}; expected chosen, all or none" ;; esac
[[ -f "${KIT}/${entry}" ]] || config_error "entrypoint ${entry} of ${suite} does not exist under ${KIT}"

[[ -d "${prefix}" ]] || config_error "--prefix ${prefix} is not a directory"
prefix="$(cd "${prefix}" && pwd)"
manifest="${prefix}/share/vision-pack/vision-pack-manifest.json"
[[ -f "${manifest}" ]] || config_error "${prefix} has no share/vision-pack/vision-pack-manifest.json (not a prepared prefix)"
if [[ -n "${data}" ]]; then
  if [[ -d "${data}" ]]; then
    data="$(cd "${data}" && pwd)"
  else
    echo "::warning title=${suite}::--data ${data} is not a directory; the dataset checks will report blocked" >&2
    data=""
  fi
fi
cache="${cache:-${RUNNER_TEMP:-/tmp}/vp/cache/suites}"
mkdir -p "${cache}"
cache="$(cd "${cache}" && pwd)"
if [[ -n "${dist}" ]]; then
  [[ -f "${dist}" ]] || config_error "--dist-tarball ${dist} does not exist"
  dist="$(readlink -f "${dist}")"
fi
budget=$((timeout_min * 60 - 240))
((budget >= 60)) || budget=60

# GPU visibility (see the header).
# shellcheck source=/dev/null
source "${KIT}/build_tools/gpu_env.sh"
case "${access}" in
  none) export ROCR_VISIBLE_DEVICES=-1 ;;
  chosen)
    if [[ -z "${ROCR_VISIBLE_DEVICES:-}" && -n "${VP_GPU_INDEX:-}" ]]; then
      export ROCR_VISIBLE_DEVICES="${VP_GPU_INDEX}"
    fi ;;
esac

# What a fresh test container would see: no ROCm or Python paths inherited
# from the job, and the image's default PATH.
unset ROCM_HOME HIP_PATH LD_LIBRARY_PATH PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROCM_PATH="${prefix}"
export VP_OUT="${out}"
export VP_REPO="${KIT}"
export VP_TIER="${tier}"
export VP_GFX="${gfx}"
export VP_SUITE_NAME="${suite}"
export VP_MANIFEST="${manifest}"
export VP_VISION_PACK_SRC="${KIT}/vision-pack"
if [[ -n "${data}" ]]; then export VP_DATA="${data}"; else unset VP_DATA; fi
export VP_CACHE="${cache}"
export VP_GPU_ACCESS="${access}"
if [[ -n "${dist}" ]]; then export VP_DIST_TARBALL="${dist}"; else unset VP_DIST_TARBALL; fi
export VP_NO_CONTAINER=1
export VP_UNSUPPORTED_GPUS=""
export VP_EXTENDED="${VP_EXTENDED:-0}"
VP_BUILD_JOBS="${VP_BUILD_JOBS:-$(nproc)}"
export VP_BUILD_JOBS
export VP_ALLOW_CRASH_TESTS="${VP_ALLOW_CRASH_TESTS:-0}"
export VP_PY="${py}"

scratch="$(mktemp -d "${RUNNER_TEMP:-/tmp}/vp-run-suite.XXXXXX")"
trap 'rm -rf "${scratch}"' EXIT
marker="${scratch}/started"
touch "${marker}"

echo "run_suite: suite=${suite} tier=${tier} gfx=${gfx:-none} gpu_access=${access}" \
  "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} crash_tests=${VP_ALLOW_CRASH_TESTS}"
echo "run_suite: prefix=${prefix} data=${data:-none} out=${out}"
echo "run_suite: entrypoint=${entry} timeout=${budget}s (timeout_minutes ${timeout_min})"

cd "${out}"
echo "::group::${suite}"
rc=0
timeout -k 60 "${budget}" bash "${KIT}/${entry}" || rc=$?
echo "::endgroup::"

reason=ok
if ((rc == 124 || rc == 137)); then
  reason="time budget of ${budget}s exhausted"
elif ((rc != 0)); then
  reason="exited ${rc}"
fi
write_status "${rc}" "${reason}"

{
  echo "### ${suite}"
  if [[ -f "${out}/summary.json" ]]; then
    jq -r '"| status | count |\n|---|---|\n" + (.counts // {} | to_entries | map("| \(.key) | \(.value) |") | join("\n"))' \
      "${out}/summary.json"
  else
    echo "No summary.json: the suite did not finish (${reason})."
  fi
} >>"${GITHUB_STEP_SUMMARY:-/dev/null}" || true

# Stand-in for read-only mounts: list what changed under the prefix and the dataset.
for dir in "${prefix}" ${data:+"${data}"}; do
  changed=()
  mapfile -t changed < <(find "${dir}" -xdev -cnewer "${marker}" -print 2>/dev/null | head -n 21 || true)
  if [[ ${#changed[@]} -gt 0 ]]; then
    more=""
    [[ ${#changed[@]} -gt 20 ]] && more=" (first 20 shown)"
    echo "::warning title=${suite} wrote to a read-only tree::${dir} changed during the run${more}; suites must not modify it"
    for f in "${changed[@]:0:20}"; do echo "::warning::${suite} modified ${f}"; done
  fi
done || true

if ((rc != 0)); then
  echo "::error title=${suite}::suite ${reason}; results may be partial"
  cat <<EOF
Reproduce on a machine with a ROCm + vision-pack prefix (same kit):
  ${KIT}/build_tools/run_suite.sh --suite ${suite} --prefix <prefix> --out <dir> --tier ${tier} \\
      --gfx ${gfx:-<gfx>} --gpu-access ${access}${data:+ --data <MIVisionX-data>}
EOF
fi
exit "${rc}"
