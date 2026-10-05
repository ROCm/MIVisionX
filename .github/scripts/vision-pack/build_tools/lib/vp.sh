# shellcheck shell=bash
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

# Common helpers for the vision-pack QA suites.
#
# Usage, at the top of suites/<name>/run.sh:
#
#   #!/usr/bin/env bash
#   set -uo pipefail          # not -e: a failing check must not abort the suite
#   source "${VP_REPO}/build_tools/lib/vp.sh"
#   vp_init <suite-name>
#   vp_run  "group::name" --timeout 300 -- some command args
#   vp_finish
#
# Environment contract (set by build_tools/run_suite.sh):
#   ROCM_PATH     ROCm SDK + vision-pack overlay prefix (read-only)
#   VP_OUT        writable output directory for this suite
#   VP_REPO       repository root (read-only)
#   VP_DATA       dataset root that CONTAINS rocal_data/ (read-only; may be empty)
#   VP_TIER       quick | standard | comprehensive | full (default comprehensive)
#   VP_GFX        detected GPU architecture, e.g. gfx1201 (may be empty)
#   VP_MANIFEST   vision-pack manifest JSON (default $ROCM_PATH/share/vision-pack/...)
#   VP_VISION_PACK_SRC  vision-pack submodule checkout pinned to the tested commit
#   VP_EXTENDED   1 when the extended image (ROCm torch, tensorflow, jax) is in use
#   VP_UNSUPPORTED_GPUS  "gfx:renderminor:index,..." GPUs present but not targeted
#   VP_NO_CONTAINER 1 when running directly on a host or CI job (run_suite.sh)
#
# Results: see build_tools/results/emit.py for the record format. Suites write
# records only through vp_result/vp_run/vp_ctest/vp_pytest or the Python API.

VP_TIERS=(quick standard comprehensive full)

vp__log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

vp__slug() {
  local s="$1"
  s="${s//::/__}"
  s="$(printf '%s' "${s}" | tr -c 'A-Za-z0-9._+-' '_')"
  printf '%s' "${s:0:180}"
}

vp__tier_index() {
  local t="$1" i
  for i in "${!VP_TIERS[@]}"; do
    [[ "${VP_TIERS[$i]}" == "${t}" ]] && { echo "$i"; return; }
  done
  echo 2
}

# vp_tier_ge <tier>: true when the current tier is at least <tier>.
vp_tier_ge() {
  (( $(vp__tier_index "${VP_TIER}") >= $(vp__tier_index "$1") ))
}

vp__scale_timeout() {
  awk -v t="$1" -v s="${VP_TIMEOUT_SCALE:-1}" 'BEGIN { printf "%d", (t * s < 1 ? 1 : t * s) }'
}

# vp_scale_timeout <seconds>: the timeout scaled by VP_TIMEOUT_SCALE (for suites' own timeouts).
vp_scale_timeout() { vp__scale_timeout "$@"; }

vp_init() {
  VP_SUITE="${1:?vp_init <suite>}"
  : "${ROCM_PATH:?ROCM_PATH must point at the ROCm + vision-pack prefix}"
  : "${VP_OUT:?VP_OUT must be a writable output directory}"
  VP_REPO="${VP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  VP_TIER="${VP_TIER:-comprehensive}"
  VP_DATA="${VP_DATA:-}"
  VP_GFX="${VP_GFX:-}"
  VP_EXTENDED="${VP_EXTENDED:-0}"
  VP_MANIFEST="${VP_MANIFEST:-${ROCM_PATH}/share/vision-pack/vision-pack-manifest.json}"
  VP_VISION_PACK_SRC="${VP_VISION_PACK_SRC:-${VP_REPO}/vision-pack}"
  VP_PY="${VP_PY:-python3}"
  VP_EMIT="${VP_REPO}/build_tools/results/emit.py"
  # Suites whose entrypoint lives elsewhere (robustness-nogpu -> suites/robustness/nogpu.sh)
  # get the directory of the script that called vp_init.
  VP_SUITE_DIR="${VP_REPO}/suites/${VP_SUITE}"
  if [[ ! -d "${VP_SUITE_DIR}" && -n "${BASH_SOURCE[1]:-}" ]]; then
    VP_SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  fi

  mkdir -p "${VP_OUT}/logs" "${VP_OUT}/junit" "${VP_OUT}/raw" "${VP_OUT}/perf" "${VP_OUT}/work"
  VP_RESULTS="${VP_OUT}/results.jsonl"
  VP_WORK="${VP_OUT}/work"
  : >"${VP_RESULTS}"

  # Customer mode: libraries must resolve through their RUNPATH inside the
  # prefix. VP_CI_PARITY=1 reproduces upstream CI's LD_LIBRARY_PATH instead.
  if [[ "${VP_CI_PARITY:-0}" == 1 ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"
  else
    unset LD_LIBRARY_PATH
  fi
  # MIVisionX targets are chosen per test; a global value silently turns the
  # default-target tests into CPU tests (finding N4).
  unset AGO_DEFAULT_TARGET
  # One GPU-visibility variable only (ROCR_VISIBLE_DEVICES, set by the
  # launcher when needed); HIP_VISIBLE_DEVICES stacked on top re-indexes.
  unset HIP_VISIBLE_DEVICES
  unset HSA_OVERRIDE_GFX_VERSION

  # Bare-metal runs (run_suite.sh): never feed crashes to apport. A core-size
  # limit of exactly 1 byte makes the kernel abort core dumps to a pipe (its
  # recursion guard), so native crashes write no /var/crash report; the
  # hostsafe sitecustomize does the same for uncaught Python exceptions.
  if [[ "${VP_NO_CONTAINER:-0}" == 1 ]]; then
    prlimit --core=1:1 --pid $$ 2>/dev/null || ulimit -c 0
    VP_EXTRA_PYTHONPATH="${VP_EXTRA_PYTHONPATH:-${VP_REPO}/build_tools/lib/hostsafe}"
  fi

  export ROCM_PATH VP_OUT VP_REPO VP_DATA VP_TIER VP_GFX VP_EXTENDED VP_MANIFEST VP_VISION_PACK_SRC VP_PY
  export VP_SUITE VP_SUITE_DIR VP_RESULTS VP_WORK VP_EMIT
  export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
  export HIP_PLATFORM=amd
  export PYTHONPATH="${ROCM_PATH}/lib${VP_EXTRA_PYTHONPATH:+:${VP_EXTRA_PYTHONPATH}}"
  export PYTHONDONTWRITEBYTECODE=1
  export MPLBACKEND=Agg
  export HOME="${VP_WORK}/home"
  mkdir -p "${HOME}"
  if [[ -n "${VP_DATA}" ]]; then
    export ROCAL_DATA_PATH="${VP_DATA}"
  fi

  vp__log "suite=${VP_SUITE} tier=${VP_TIER} gfx=${VP_GFX:-none} ROCM_PATH=${ROCM_PATH}"
  vp__log "python=$(command -v "${VP_PY}") ($("${VP_PY}" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?'))"
  VP_T0="$(date +%s)"
}

# vp_result <id> <status> [message] [duration_s] [log] [backend] [attempts] [repro]
vp_result() {
  local id="$1" status="$2" msg="${3:-}" dur="${4:-0}" log="${5:-}" backend="${6:-}" attempts="${7:-1}" repro="${8:-}"
  [[ "${id}" == "${VP_SUITE}::"* ]] || id="${VP_SUITE}::${id}"
  log="${log#"${VP_OUT}"/}"
  [[ "${dur}" =~ ^[0-9]+([.][0-9]+)?$ ]] || dur=0
  jq -cn --arg suite "${VP_SUITE}" --arg id "${id}" --arg status "${status}" \
    --arg message "${msg:0:4000}" --argjson duration_s "${dur}" --arg log "${log}" \
    --arg backend "${backend}" --argjson attempts "${attempts}" --arg repro "${repro}" \
    '{suite:$suite, id:$id, status:$status, duration_s:$duration_s, message:$message,
      log:$log, backend:$backend, attempts:$attempts, repro:$repro}' >>"${VP_RESULTS}"
}

vp_skip()    { vp_result "$1" skip "${2:-not applicable}"; }
vp_blocked() { vp_result "$1" blocked "${2:-missing dependency}"; }

# vp_run <id> [--timeout S] [--retry] [--cwd DIR] [--expect-rc N] [--backend B]
#        [--env K=V]... -- cmd args...
# Runs a command with a timeout, logs it, records pass/fail/error/flaky.
# Returns 0 for pass/flaky, 1 otherwise.
vp_run() {
  local id="$1"; shift
  local timeout="${VP_DEFAULT_TIMEOUT:-900}" retry="${VP_RETRY_ALL:-0}" cwd="" expect=0 backend=""
  local -a envs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --timeout) timeout="$2"; shift 2 ;;
      --retry) retry=1; shift ;;
      --no-retry) retry=0; shift ;;
      --cwd) cwd="$2"; shift 2 ;;
      --expect-rc) expect="$2"; shift 2 ;;
      --backend) backend="$2"; shift 2 ;;
      --env) envs+=("$2"); shift 2 ;;
      --) shift; break ;;
      *) break ;;
    esac
  done
  local log
  log="${VP_OUT}/logs/$(vp__slug "${id}").log"
  local t
  t="$(vp__scale_timeout "${timeout}")"
  local attempt=1 rc status msg t0 t1 dur
  local repro
  repro="$(printf '%q ' ${envs[@]+"${envs[@]}"} "$@")"
  [[ -n "${cwd}" ]] && repro="(cd $(printf '%q' "${cwd}") && ${repro})"
  : >"${log}"
  while :; do
    {
      echo "### ${id} (attempt ${attempt}, timeout ${t}s)"
      echo "### cwd: ${cwd:-$(pwd)}"
      echo "### cmd: ${repro}"
    } >>"${log}"
    t0="$(date +%s.%N)"
    (
      [[ -n "${cwd}" ]] && { cd "${cwd}" || exit; }
      if [[ ${#envs[@]} -gt 0 ]]; then
        exec env "${envs[@]}" timeout -k 30 "${t}" "$@"
      else
        exec timeout -k 30 "${t}" "$@"
      fi
    ) >>"${log}" 2>&1 </dev/null
    rc=$?
    t1="$(date +%s.%N)"
    dur="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.3f", b - a }')"
    msg=""
    if [[ "${rc}" -eq "${expect}" ]]; then
      status=pass
    elif [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
      status=error; msg="timeout after ${t}s (rc ${rc})"
    elif [[ "${rc}" -gt 128 && "${rc}" -le 192 ]]; then
      # 129-192 = killed by signal 1-64; 193-255 are ordinary exit codes (e.g. exit(-1) = 255).
      status=error; msg="killed by signal $((rc - 128)) (rc ${rc})"
    else
      status=fail; msg="exit ${rc} (expected ${expect})"
    fi
    if [[ "${status}" != pass && "${retry}" == 1 && "${attempt}" == 1 ]]; then
      attempt=2
      echo "### first attempt ${status}: ${msg}; retrying once" >>"${log}"
      continue
    fi
    if [[ "${status}" == pass && "${attempt}" == 2 ]]; then
      status=flaky; msg="passed on retry"
    fi
    break
  done
  [[ "${status}" == fail || "${status}" == error ]] && msg="${msg}; $(tail -c 1500 "${log}" | tr '\n' ' ' | tr -s ' ')"
  vp_result "${id}" "${status}" "${msg}" "${dur}" "${log}" "${backend}" "${attempt}" "${repro}"
  [[ "${status}" == pass || "${status}" == flaky ]]
}

# vp_ingest_junit <junit.xml> <group> [rerun.xml] [backend] [log]
vp_ingest_junit() {
  local first="$1" group="$2" rerun="${3:-}" backend="${4:-}" log="${5:-}"
  log="${log#"${VP_OUT}"/}"
  local -a args=(ingest-junit --out "${VP_RESULTS}" --suite "${VP_SUITE}" --group "${group}" --backend "${backend}"
                 --log "${log}")
  [[ -n "${rerun}" && -f "${rerun}" ]] && args+=(--rerun "${rerun}")
  "${VP_PY}" "${VP_EMIT}" "${args[@]}" "${first}" >&2 \
    || vp_result "${group}::junit-ingest" error "could not ingest ${first}"
}

# vp_ctest <group> <build_dir> [extra ctest args...]
# Runs ctest serially with JUnit output, re-runs failures once, ingests both.
vp_ctest() {
  local group="$1" build="$2"; shift 2
  local junit
  junit="${VP_OUT}/raw/$(vp__slug "${group}")"
  "${VP_REPO}/build_tools/ctest_junit.sh" "${build}" "${junit}" "$@" \
    >"${VP_OUT}/logs/$(vp__slug "${group}").ctest.log" 2>&1
  if [[ -f "${junit}.xml" ]]; then
    vp_ingest_junit "${junit}.xml" "${group}" "${junit}.rerun.xml"
  else
    vp_result "${group}::ctest" error "ctest produced no JUnit (see logs/$(vp__slug "${group}").ctest.log)" \
      0 "${VP_OUT}/logs/$(vp__slug "${group}").ctest.log"
  fi
}

# vp_pytest <group> [pytest args...]
# Runs pytest with JUnit output (cache kept in the work dir, never next to the
# installed tests), re-runs the failures once, ingests both runs. Each pytest
# session is bounded by VP_PYTEST_TIMEOUT seconds (default 5400).
vp_pytest() {
  local group="$1"; shift
  local junit cache log
  junit="${VP_OUT}/raw/$(vp__slug "${group}")"
  cache="${VP_WORK}/pytest-cache-$(vp__slug "${group}")"
  log="${VP_OUT}/logs/$(vp__slug "${group}").pytest.log"
  local t
  t="$(vp__scale_timeout "${VP_PYTEST_TIMEOUT:-5400}")"
  timeout -k 60 "${t}" "${VP_PY}" -m pytest "$@" -o cache_dir="${cache}" -p no:randomly \
    --junitxml="${junit}.xml" -o junit_family=xunit2 -q >"${log}" 2>&1
  local rc=$?
  if [[ "${rc}" -eq 1 ]]; then
    echo "### re-running last failures once" >>"${log}"
    timeout -k 60 "${t}" "${VP_PY}" -m pytest "$@" -o cache_dir="${cache}" --lf \
      --junitxml="${junit}.rerun.xml" -o junit_family=xunit2 -q >>"${log}" 2>&1
  fi
  if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    vp_result "${group}::pytest-timeout" error "pytest exceeded ${t}s; results are partial" 0 "${log}"
  fi
  if [[ -f "${junit}.xml" ]]; then
    vp_ingest_junit "${junit}.xml" "${group}" "${junit}.rerun.xml"
  fi
  # rc 2..5 means pytest itself failed (usage error, internal error, nothing collected).
  if [[ "${rc}" -ge 2 && "${rc}" -le 5 ]]; then
    vp_result "${group}::pytest-session" error "pytest exited ${rc}" 0 "${log}"
  fi
}

# vp_require <id> <cmd|python-module>... : records blocked and returns 1 if missing.
vp_require_cmd() {
  local id="$1"; shift
  local c missing=()
  for c in "$@"; do command -v "${c}" >/dev/null 2>&1 || missing+=("${c}"); done
  if [[ ${#missing[@]} -gt 0 ]]; then
    vp_blocked "${id}" "missing command(s): ${missing[*]}"
    return 1
  fi
}

vp_require_py() {
  local id="$1"; shift
  local m missing=()
  for m in "$@"; do "${VP_PY}" -c "import ${m}" >/dev/null 2>&1 || missing+=("${m}"); done
  if [[ ${#missing[@]} -gt 0 ]]; then
    vp_blocked "${id}" "missing python module(s): ${missing[*]}"
    return 1
  fi
}

vp_require_data() {
  local id="$1" sub="${2:-rocal_data}"
  if [[ -z "${VP_DATA}" || ! -e "${VP_DATA}/${sub}" ]]; then
    vp_blocked "${id}" "dataset ${sub} not available (VP_DATA=${VP_DATA:-unset})"
    return 1
  fi
}

# vp_copy_lmdb <dest>: private copies of the LMDB datasets. rocAL's LMDB
# readers rewrite lock.mdb even when only reading (M14), and Caffe2 opens the
# environment read-write, so they must never point at the shared dataset.
vp_copy_lmdb() {
  local dest="$1"
  mkdir -p "${dest}"
  local d
  for d in caffe caffe2; do
    if [[ -d "${VP_DATA}/rocal_data/${d}" && ! -d "${dest}/${d}" ]]; then
      cp -a "${VP_DATA}/rocal_data/${d}" "${dest}/${d}"
      chmod -R u+w "${dest}/${d}"
    fi
  done
}

# vp_crash_tests_allowed: deliberate crash probes (segfault/UAF/GPU-fault
# reproducers) run in CI containers, but on a bare host only when
# VP_ALLOW_CRASH_TESTS=1, so a developer machine is never disturbed.
vp_crash_tests_allowed() {
  [[ "${VP_NO_CONTAINER:-0}" != 1 || "${VP_ALLOW_CRASH_TESTS:-0}" == 1 ]]
}

# vp_perf <name> <json-file>: register a perf result file (copied into perf/).
vp_perf() {
  local name="$1" file="$2"
  [[ -f "${file}" ]] && cp "${file}" "${VP_OUT}/perf/$(vp__slug "${name}").json"
}

# vp_cmake_build <id> <src> <build> [cmake args...]: configure + build, recorded.
vp_cmake_build() {
  local id="$1" src="$2" build="$3"; shift 3
  vp_run "${id}::configure" --timeout 600 --no-retry -- \
    cmake -S "${src}" -B "${build}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DROCM_PATH="${ROCM_PATH}" \
    "-DCMAKE_PREFIX_PATH=${ROCM_PATH};${ROCM_PATH}/lib/cmake;${ROCM_PATH}/lib/llvm" "$@" \
    && vp_run "${id}::build" --timeout 3600 --no-retry -- \
      cmake --build "${build}" --parallel "${VP_BUILD_JOBS:-$(nproc)}"
}

vp_finish() {
  local t1
  t1="$(date +%s)"
  "${VP_PY}" "${VP_EMIT}" to-junit --in "${VP_RESULTS}" --out "${VP_OUT}/junit/${VP_SUITE}.xml" --suite-name "${VP_SUITE}"
  "${VP_PY}" "${VP_EMIT}" summary --in "${VP_RESULTS}" --out "${VP_OUT}/summary.json" >&2
  jq -n --arg suite "${VP_SUITE}" --arg tier "${VP_TIER}" --arg gfx "${VP_GFX}" \
    --argjson seconds "$((t1 - VP_T0))" --slurpfile s "${VP_OUT}/summary.json" \
    '{suite:$suite, tier:$tier, gfx:$gfx, wall_seconds:$seconds} + $s[0]' >"${VP_OUT}/suite.json"
  vp__log "suite ${VP_SUITE} finished in $((t1 - VP_T0))s"
  return 0
}
