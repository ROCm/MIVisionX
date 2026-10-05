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

# Helpers for suites/mivisionx/run.sh (sourced after build_tools/lib/vp.sh).

MVX_CTS_URL="${MVX_CTS_URL:-https://github.com/KhronosGroup/OpenVX-cts.git}"
MVX_CTS_REF="${MVX_CTS_REF:-openvx_1.3.2}"
MVX_OPENVX_MARK_URL="${MVX_OPENVX_MARK_URL:-https://github.com/kiritigowda/openvx-mark.git}"

# mvx_retry <attempts> <cmd...>: re-run a flaky network command with a growing pause.
mvx_retry() {
  local n="$1" i=1
  shift
  until "$@"; do
    if (( i >= n )); then
      return 1
    fi
    echo "### attempt ${i}/${n} failed: $*; retrying in $((i * 20))s"
    sleep $((i * 20))
    i=$((i + 1))
  done
}

mvx__clone_once() { # <url> <ref> <dir>
  rm -rf "$3"
  timeout -k 30 900 git clone --depth 1 --branch "$2" "$1" "$3"
}

mvx__lfs_pull() { # <dir>
  git -C "$1" lfs install --local >/dev/null && timeout -k 30 900 git -C "$1" lfs pull
}

# mvx_git_cache <name> <url> <ref> <lfs 0|1>: shallow clone into $VP_CACHE/<name>/src under
# $VP_CACHE/<name>.lock. The clone goes to a temp dir and is renamed only once complete
# (with .vp-complete holding the SHA), so a half-finished clone is never reused.
# Prints the source dir; returns 2 when no cache is configured, 1 when the fetch failed.
mvx_git_cache() {
  local name="$1" url="$2" ref="$3" lfs="$4"
  local cache="${VP_CACHE:-}" log="${VP_OUT}/logs/fetch-${1}.log"
  [[ -n "${cache}" ]] || return 2
  mkdir -p "${cache}/${name}" || return 2
  local dest="${cache}/${name}/src"
  (
    if ! flock -w 5400 9; then
      echo "### could not take ${cache}/${name}.lock"
      exit 1
    fi
    if [[ -f "${dest}/.vp-complete" ]]; then
      echo "### reusing ${dest} ($(cat "${dest}/.vp-complete"))"
      exit 0
    fi
    rm -rf "${dest}"
    tmp="$(mktemp -d "${cache}/${name}/.clone.XXXXXX")" || exit 1
    trap 'rm -rf "${tmp}"' EXIT
    mvx_retry 3 mvx__clone_once "${url}" "${ref}" "${tmp}/src" || exit 1
    if [[ "${lfs}" == 1 ]]; then
      if git lfs version >/dev/null 2>&1; then
        mvx_retry 3 mvx__lfs_pull "${tmp}/src" || exit 1
      elif grep -qs 'filter=lfs' "${tmp}/src/.gitattributes"; then
        echo "### git-lfs is not installed but ${url} uses LFS"
        exit 1
      fi
    fi
    git -C "${tmp}/src" rev-parse HEAD >"${tmp}/src/.vp-complete" || exit 1
    mv "${tmp}/src" "${dest}"
  ) 9>"${cache}/${name}.lock" >>"${log}" 2>&1 || return 1
  printf '%s\n' "${dest}"
}

# mvx_fetch_file <cache-subdir> <url> <name>: download one file into $VP_CACHE under a lock.
mvx_fetch_file() {
  local sub="$1" url="$2" name="$3" cache="${VP_CACHE:-}"
  [[ -n "${cache}" ]] || return 2
  mkdir -p "${cache}/${sub}" || return 2
  local dest="${cache}/${sub}/${name}"
  (
    flock -w 600 9 || exit 1
    [[ -s "${dest}" ]] && exit 0
    mvx_retry 3 curl -fsSL --max-time 120 -o "${dest}.part" "${url}" || { rm -f "${dest}.part"; exit 1; }
    mv "${dest}.part" "${dest}"
  ) 9>"${cache}/${sub}.lock" >>"${VP_OUT}/logs/fetch-${sub}.log" 2>&1 || return 1
  printf '%s\n' "${dest}"
}

# ---------------------------------------------------------------------------
# Khronos OpenVX CTS
# ---------------------------------------------------------------------------

# name|timeout|filter: upstream MIVisionX conformance-hip.yml shards (openvx_1.3.2)
# shellcheck disable=SC2034  # used by run.sh
MVX_CTS_SHARDS=(
  "baseline|300|GraphBase.*:Logging.*:SmokeTestBase.*:SmokeTest.*:TargetBase.*:Target.*"
  "graph|600|Graph.*:GraphCallback.*:GraphDelay.*:GraphROI.*:UserNode.*"
  "data-objects|300|Scalar.*:Array.*:ObjectArray.*:Matrix.*:Convolution.*:Distribution.*:LUT.*:Histogram.*"
  "image-ops|600|Image.*:vxCopyImagePatch.*:vxMapImagePatch.*:vxCreateImageFromChannel.*:vxCopyRemapPatch.*:vxMapRemapPatch.*"
  "vision-color|300|ColorConvert.*:ChannelExtract.*:ChannelCombine.*:vxConvertDepth.*:vxuConvertDepth.*"
  "vision-filters|600|Box3x3.*:Gaussian3x3.*:Median3x3.*:Dilate3x3.*:Erode3x3.*:Sobel3x3.*:Magnitude.*:Phase.*:NonLinearFilter.*:Convolve.*:EqualizeHistogram.*"
  "vision-arithmetic|600|vxAddSub.*:vxuAddSub.*:vxMultiply.*:vxuMultiply.*:vxBinOp8u.*:vxuBinOp8u.*:vxBinOp16s.*:vxuBinOp16s.*:vxBinOp1u.*:vxuBinOp1u.*:vxNot.*:vxuNot.*:WeightedAverage.*:Threshold.*"
  "vision-geometric|600|Scale.*:WarpAffine.*:WarpPerspective.*:Remap.*:HalfScaleGaussian.*"
  "vision-features|600|HarrisCorners.*:FastCorners.*:vxCanny.*:vxuCanny.*"
  "vision-statistics|300|MeanStdDev.*:MinMaxLoc.*:Integral.*"
  "vision-pyramid|300|GaussianPyramid.*:LaplacianPyramid.*:LaplacianReconstruct.*:OptFlowPyrLK.*"
  "pipelining|1800|GraphPipeline.*"
  "streaming|900|GraphStreaming.*"
)
# shellcheck disable=SC2034  # used by run.sh
MVX_CTS_CRASHER="Image.DISABLED_testAccessCopyWriteUniformImage"

# mvx_cts_build <src> <build>: amdclang at -O2 (upstream: -O3 miscompiles the harness).
mvx_cts_build() {
  local rp="${ROCM_PATH}"
  vp_cmake_build "cts.build" "$1" "$2" \
    -DCMAKE_C_COMPILER="${rp}/lib/llvm/bin/amdclang" -DCMAKE_CXX_COMPILER="${rp}/lib/llvm/bin/amdclang++" \
    "-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG" "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOPENVX_INCLUDES="${rp}/include/mivisionx" \
    "-DOPENVX_LIBRARIES=${rp}/lib/libopenvx.so;${rp}/lib/libvxu.so;pthread;dl;m;rt" \
    -DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_PIPELINING=ON -DOPENVX_USE_STREAMING=ON \
    "-DCMAKE_C_STANDARD_LIBRARIES=-L${rp}/lib -lamdhip64" "-DCMAKE_CXX_STANDARD_LIBRARIES=-L${rp}/lib -lamdhip64" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,${rp}/lib -Wl,-rpath,${rp}/lib"
}

# mvx_cts_run <target> <group> <timeout> <filter|-> [extra args]: one vx_test_conformance
# invocation, parsed per test with build_tools/results/cts_to_junit.py and ingested as
# mivisionx::<group>::<test>.
# LD_LIBRARY_PATH only adds the CTS's own lib dir: libopenvx dlopen()s libtest-testmodule.so.
mvx_cts_run() {
  local target="$1" group="$2" to="$3" filter="$4"
  shift 4
  local slug log raw t rc
  slug="$(vp__slug "${group}")"
  log="${VP_OUT}/logs/${slug}.log"
  raw="${VP_OUT}/raw/${slug}"
  t="$(vp__scale_timeout "${to}")"
  local -a args=("$@")
  [[ "${filter}" != "-" ]] && args+=("--filter=${filter}")
  vp__log "cts ${group} (timeout ${t}s)"
  # shellcheck disable=SC2153  # MVX_CTS_SRC/BUILD are exported by run.sh part_cts
  {
    echo "### cwd: ${MVX_CTS_BUILD}"
    echo "### cmd: AGO_DEFAULT_TARGET=${target} VX_TEST_DATA_PATH=${MVX_CTS_SRC}/test_data/ LD_LIBRARY_PATH=${MVX_CTS_BUILD}/lib ./bin/vx_test_conformance ${args[*]}"
  } >"${log}"
  (
    cd "${MVX_CTS_BUILD}" || exit 2
    exec env AGO_DEFAULT_TARGET="${target}" VX_TEST_DATA_PATH="${MVX_CTS_SRC}/test_data/" AGO_LOG_STDERR=1 \
      LD_LIBRARY_PATH="${MVX_CTS_BUILD}/lib" timeout -k 30 "${t}" ./bin/vx_test_conformance ${args[@]+"${args[@]}"}
  ) >>"${log}" 2>&1 </dev/null
  rc=$?
  "${VP_PY}" "${VP_REPO}/build_tools/results/cts_to_junit.py" --suite "${group}" --log "${log}" --rc "${rc}" --out "${raw}.xml" \
    --summary "${raw}.summary.json" --names "${raw}.names" >>"${VP_OUT}/logs/cts-parse.log" 2>&1
  [[ "${MVX_CTS_NO_INGEST:-0}" == 1 ]] && return 0
  if [[ -f "${raw}.xml" ]]; then
    "${VP_PY}" "${VP_EMIT}" ingest-junit --out "${VP_RESULTS}" --suite "${VP_SUITE}" --group "${group}" \
      --backend "${target}" --log "logs/${slug}.log" "${raw}.xml" >&2 \
      || vp_result "${group}::junit-ingest" error "could not ingest ${raw}.xml" 0 "${log}" "${target}"
  else
    vp_result "${group}::#REPORT" error "cts_to_junit.py produced no JUnit (exit ${rc})" 0 "${log}" "${target}"
  fi
}
