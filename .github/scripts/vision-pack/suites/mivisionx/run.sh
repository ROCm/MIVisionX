#!/usr/bin/env bash
# shellcheck disable=SC2329  # the part_* functions run through want/timed
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

# mivisionx suite: MIVisionX (OpenVX) from the vision-pack overlay in $ROCM_PATH.
#
#   quick          ctest smoke subset, two runvx graphs on CPU and GPU
#   standard       + full ctest (32), the 16 API tests on both targets, shipped sample
#                    GDFs (M2), API probes (M18 and low items)
#   comprehensive  + Khronos CTS (13 upstream shards x CPU/GPU, one unfiltered GPU run),
#                    GDF sweep, vision nodes, parity, samples_raw, vx_rpp, crash sweeps
#                    (container only), cu_mask, ctest parallel race
#   full           + CTS optional tests (--run_disabled minus the H12 crasher), perf
#
# MVX_PARTS="cts gdf" runs just those parts (development aid), whatever the tier.
# AGO_DEFAULT_TARGET is set per test only (vision-pack's own CI exports it globally, N4).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../build_tools/lib/vp.sh
source "${VP_REPO:-${HERE}/../..}/build_tools/lib/vp.sh"
vp_init mivisionx
# shellcheck source=lib.sh
source "${VP_SUITE_DIR}/lib.sh"

TEST_ROOT="${ROCM_PATH}/share/mivisionx/test"
RUNVX="${ROCM_PATH}/bin/runvx"
PY="${VP_PY}"
export PYTHONPATH="${VP_SUITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
MVX_CTEST_SMOKE='^(vx_core_test|vx_rpp_test|runvx_test|openvx_canny|openvx_canny_CPU|openvx_canny_GPU)$'
API_TESTS=(
  canny:openvx_canny channel_extract:openvx_channel_extract color_convert:openvx_color_convert
  accumulate:openvx_accum graph:openvx_graph graph_api:openvx_graph_api tensor_api:openvx_tensor_api
  data_objects:openvx_data_objects_api user_kernel:openvx_user_kernel_api
  tensor_advanced:openvx_tensor_advanced_api threshold_query:openvx_threshold_query_api
  graph_import:openvx_graph_import_api vxu_api:openvx_vxu_api coverage_boost:openvx_coverage_boost
  vision_coverage:openvx_vision_coverage pipelining_api:openvx_pipelining_api
)
PARITY_KERNELS="And,Or,Xor,Not,AbsDiff,Add,Subtract,Multiply,Box3x3,Gaussian3x3,Median3x3,Erode3x3,Dilate3x3,Sobel3x3,CustomConvolution,NonLinearFilter,ColorConvert_RGB2IYUV,ChannelExtract,ChannelCombine,ConvertDepth,ScaleImage_Half,WarpAffine,WarpPerspective,Remap,Histogram,EqualizeHist,MeanStdDev,MinMaxLoc,IntegralImage,GaussianPyramid,HalfScaleGaussian,CannyEdgeDetector,HarrisCorners,FastCorners,OpticalFlowPyrLK,Magnitude,Phase,TableLookup,Threshold_Binary,WeightedAverage"

want() { # <part>: selected by MVX_PARTS, or by the tier
  if [[ -n "${MVX_PARTS:-}" ]]; then
    [[ " ${MVX_PARTS} " == *" $1 "* ]]
    return
  fi
  case "$1" in
    ctest|runvx-smoke) return 0 ;;
    api|samples-gdf|api-probe) vp_tier_ge standard ;;
    cts|gdf|vision|parity|samples-raw|vxrpp|crash|cumask|race) vp_tier_ge comprehensive ;;
    cts-optional|perf) vp_tier_ge full ;;
    *) return 1 ;;
  esac
}

py() { # <script> [args]: a Python harness; a harness crash is recorded, never hidden
  local script="$1"
  shift
  "${PY}" "${VP_SUITE_DIR}/${script}" "$@" >>"${VP_OUT}/logs/harness.log" 2>&1
  local rc=$?
  [[ "${rc}" -eq 0 ]] || vp_result "harness::${script%.py}${MVX_HARNESS_TAG:+.${MVX_HARNESS_TAG}}" error \
    "harness exited ${rc}; see logs/harness.log: $(tail -c 800 "${VP_OUT}/logs/harness.log" | tr '\n' ' ')" 0 \
    "${VP_OUT}/logs/harness.log"
}

# ---------------------------------------------------------------------------

if [[ ! -x "${RUNVX}" || ! -f "${TEST_ROOT}/CMakeLists.txt" ]]; then
  vp_blocked "prefix::mivisionx-installed" "runvx or share/mivisionx/test missing under ROCM_PATH=${ROCM_PATH}"
  vp_finish
  exit 0
fi
vp_require_cmd "prefix::build-tools" cmake ninja || true
HAVE_BUILD=0
command -v cmake >/dev/null && command -v ninja >/dev/null && HAVE_BUILD=1

part_ctest() {
  local b="${VP_WORK}/ctest"
  if [[ "${HAVE_BUILD}" != 1 ]]; then
    vp_blocked "ctest::configure" "cmake/ninja not installed"
    return
  fi
  vp_run "ctest.build::configure" --timeout 600 -- cmake -S "${TEST_ROOT}" -B "${b}" -G Ninja \
    -DROCM_PATH="${ROCM_PATH}" -DPython3_EXECUTABLE="$(command -v "${PY}")" || return
  if vp_tier_ge standard || [[ " ${MVX_PARTS:-} " == *" ctest-full "* ]]; then
    vp_ctest ctest "${b}"
  else
    vp_ctest ctest "${b}" -R "${MVX_CTEST_SMOKE}"
  fi
}

part_runvx_smoke() {
  local t
  for t in CPU GPU; do
    MVX_HARNESS_TAG="runvx-smoke.${t}" py gdf_sweep.py --target "${t}" --group "runvx-smoke.${t}" \
      --only vision_profile/43_feature_tracker.gdf logical/And_alt.gdf
  done
}

part_api() {
  local entry dir exe b t
  for entry in "${API_TESTS[@]}"; do
    dir="${entry%%:*}"
    exe="${entry##*:}"
    b="${VP_WORK}/api/${dir}"
    # shellcheck disable=SC2016  # the single-quoted script expands its own positional arguments
    if [[ "${HAVE_BUILD}" == 1 ]] && vp_run "api.build::${dir}" --timeout 600 -- bash -c \
      'cmake -S "$1" -B "$2" -G Ninja -DROCM_PATH="$3" "-DCMAKE_CXX_STANDARD_LIBRARIES=-L$3/lib -lamdhip64" && cmake --build "$2"' \
      _ "${TEST_ROOT}/openvx_api_tests/${dir}" "${b}" "${ROCM_PATH}"; then
      for t in CPU GPU; do
        vp_run "api.${t}::${exe}" --timeout 300 --backend "${t}" --cwd "${b}" \
          --env "AGO_DEFAULT_TARGET=${t}" --env "INSTALL_PATH=${ROCM_PATH}" -- "./${exe}"
      done
    else
      for t in CPU GPU; do
        vp_result "api.${t}::${exe}" error "not built (see api.build::${dir})" 0 "" "${t}"
      done
    fi
  done
}

part_samples_gdf() {
  # The documented sample graphs (M2: runvx has no OpenCV; read-gdf-sample.gdf is not runvx syntax).
  # Through gdf_sweep.py rather than vp_run: runvx exits 255 on errors, which vp_run reads as a signal.
  local g
  MVX_HARNESS_TAG=samples.gdf py gdf_sweep.py --target CPU --frames 1 --timeout 120 --group samples.gdf \
    --root "${ROCM_PATH}/share/mivisionx/samples/gdf" --only canny.gdf skintonedetect.gdf read-gdf-sample.gdf
  for g in canny-LIVE.gdf skintonedetect-LIVE.gdf; do
    vp_skip "samples.gdf::${g}" "needs a live camera"
  done
}

part_cts() {
  if [[ "${HAVE_BUILD}" != 1 ]] || ! command -v git >/dev/null; then
    vp_blocked "cts.build::fetch" "cmake, ninja or git not installed"
    return 1
  fi
  local rc=0
  # shellcheck disable=SC2153  # MVX_CTS_URL/REF come from lib.sh
  MVX_CTS_SRC="$(mvx_git_cache cts "${MVX_CTS_URL}" "${MVX_CTS_REF}" 1)" || rc=$?
  if [[ "${rc}" -eq 2 ]]; then
    vp_blocked "cts.build::fetch" "VP_CACHE is not set"
    return 1
  elif [[ "${rc}" -ne 0 || ! -d "${MVX_CTS_SRC}" ]]; then
    vp_blocked "cts.build::fetch" "could not clone ${MVX_CTS_URL} ${MVX_CTS_REF} (see logs/fetch-cts.log)"
    return 1
  fi
  vp_result "cts.build::fetch" pass "OpenVX-cts ${MVX_CTS_REF} at $(cat "${MVX_CTS_SRC}/.vp-complete")" 0 \
    "${VP_OUT}/logs/fetch-cts.log"
  cat "${MVX_CTS_SRC}/.vp-complete" >"${VP_OUT}/raw/cts-sha.txt"
  MVX_CTS_BUILD="${VP_WORK}/cts-build"
  mvx_cts_build "${MVX_CTS_SRC}" "${MVX_CTS_BUILD}" || return 1
  [[ -x "${MVX_CTS_BUILD}/bin/vx_test_conformance" ]] || {
    vp_result "cts.build::binary" error "vx_test_conformance not produced"
    return 1
  }
  export MVX_CTS_SRC MVX_CTS_BUILD
}

part_cts_required() {
  local t s name to filter
  for t in CPU GPU; do
    for s in "${MVX_CTS_SHARDS[@]}"; do
      IFS='|' read -r name to filter <<<"${s}"
      mvx_cts_run "${t}" "cts.${t}.${name}" "${to}" "${filter}"
    done
  done
  # One unfiltered run: proves the shards cover every required test. Recorded as two
  # aggregate checks (the per-test results come from the shards).
  local g="cts.GPU.unfiltered" raw="${VP_OUT}/raw/cts.GPU.unfiltered" log="${VP_OUT}/logs/cts.GPU.unfiltered.log"
  MVX_CTS_NO_INGEST=1 mvx_cts_run GPU "${g}" 3600 -
  if [[ -f "${raw}.summary.json" ]]; then
    local st
    st="$(jq -r 'if (.problems|length)==0 and .fail==0 and .crash==0 then "pass" else "fail" end' "${raw}.summary.json")"
    vp_result "${g}::run" "${st}" "$(jq -r '"\(.parsed) tests: \(.pass) pass, \(.fail) fail, \(.crash) crash; \(.problems|join("; "))"' \
      "${raw}.summary.json")" 0 "${log}" GPU
    sort -u "${VP_OUT}"/raw/cts.GPU.{baseline,graph,data-objects,image-ops,vision-*,pipelining,streaming}.names \
      >"${raw}.shard-union" 2>/dev/null
    local missing dups
    missing="$(comm -13 "${raw}.shard-union" <(sort -u "${raw}.names") | wc -l)"
    dups="$(cat "${VP_OUT}"/raw/cts.GPU.{baseline,graph,data-objects,image-ops,vision-*,pipelining,streaming}.names 2>/dev/null \
      | sort | uniq -d | wc -l)"
    vp_result "${g}::shard-coverage" "$([[ "${missing}" -eq 0 ]] && echo pass || echo fail)" \
      "$(sort -u "${raw}.names" | wc -l) tests unfiltered, ${missing} not in any shard, ${dups} in more than one shard: $(comm -13 \
      "${raw}.shard-union" <(sort -u "${raw}.names") | head -5 | tr '\n' ' ')" 0 "${log}" GPU
  else
    vp_result "${g}::run" error "no summary (see ${log})" 0 "${log}" GPU
  fi
}

part_cts_optional() {
  local t
  for t in CPU GPU; do
    mvx_cts_run "${t}" "cts.${t}.optional" 3600 "*DISABLED*:-${MVX_CTS_CRASHER}" --run_disabled
  done
}

part_gdf() {
  MVX_HARNESS_TAG=gdf.CPU py gdf_sweep.py --target CPU
  MVX_HARNESS_TAG=gdf.GPU py gdf_sweep.py --target GPU --fallback --metrics "${VP_OUT}/raw/gdf-fallback.json"
  vp_perf gdf-gpu-fallback "${VP_OUT}/raw/gdf-fallback.json"
}

part_vision() {
  local t s
  MVX_HARNESS_TAG=lint py vision_nodes.py --lint
  for t in CPU GPU; do
    for s in 1080p 5x3 10x10; do
      MVX_HARNESS_TAG="${t}.${s}" py vision_nodes.py --target "${t}" --size "${s}" --work "${VP_WORK}/vision" --python "${PY}"
    done
  done
}

part_parity() {
  MVX_HARNESS_TAG=1920x1080 py parity.py --width 1920 --height 1080 --work "${VP_WORK}"
  MVX_HARNESS_TAG=1282x722 py parity.py --width 1282 --height 722 --work "${VP_WORK}"
  rm -rf "${VP_WORK}"/parity-*/{cpu1,cpu2,gpu1,gpu2,inputs}
}

part_crash() {
  local kernels=(box_3x3 gaussian_3x3 median_3x3 dilate_3x3 erode_3x3)
  local sizes=(3x3 4x4 5x3 8x8 10x10 15x15 16x16 17x17 31x31 32x32 33x33 64x64 128x128)
  local ok=0 k sz t w h kind order
  local why="deliberate crash probe (H11/H12): runs in CI containers; set VP_ALLOW_CRASH_TESTS=1 to run on a bare host"
  local exe="${VP_WORK}/crash/roi_release"
  if vp_crash_tests_allowed; then
    ok=1
    mkdir -p "${VP_WORK}/crash"
    vp_run "crash.build::roi_release" --timeout 300 -- "${ROCM_PATH}/lib/llvm/bin/amdclang" -O1 \
      -I"${ROCM_PATH}/include/mivisionx" "${VP_SUITE_DIR}/roi_release.c" -o "${exe}" -L"${ROCM_PATH}/lib" -lopenvx \
      -Wl,-rpath,"${ROCM_PATH}/lib" || ok=2
  fi
  for t in CPU GPU; do
    for k in "${kernels[@]}"; do
      for sz in "${sizes[@]}"; do
        w="${sz%x*}"
        h="${sz#*x}"
        if [[ "${ok}" == 0 ]]; then
          vp_skip "crash.${t}::${k}/${sz}" "${why}"
        else
          vp_run "crash.${t}::${k}/${sz}" --timeout 120 --backend "${t}" --cwd "${VP_WORK}/crash" -- \
            "${RUNVX}" -frames:10 "-affinity:${t}" node "org.khronos.openvx.${k}" "uniform-image:${w},${h},U008,0xaa" \
            "image:${w},${h},U008"
        fi
      done
    done
    for kind in regular uniform; do
      for order in parent-first child-first; do
        if [[ "${ok}" == 0 ]]; then
          vp_skip "crash.${t}::roi-release/${kind}/${order}" "${why}"
        elif [[ "${ok}" == 2 ]]; then
          vp_result "crash.${t}::roi-release/${kind}/${order}" error "roi_release did not build" 0 "" "${t}"
        else
          vp_run "crash.${t}::roi-release/${kind}/${order}" --timeout 120 --backend "${t}" --cwd "${VP_WORK}/crash" \
            --env "AGO_DEFAULT_TARGET=${t}" -- "${exe}" "${kind}" "${order}"
        fi
      done
    done
    if [[ "${ok}" == 0 ]]; then
      vp_skip "crash.cts.${t}::${MVX_CTS_CRASHER}" "${why}"
    elif [[ -x "${MVX_CTS_BUILD:-/nonexistent}/bin/vx_test_conformance" ]]; then
      mvx_cts_run "${t}" "crash.cts.${t}" 300 "${MVX_CTS_CRASHER}" --run_disabled
    else
      vp_blocked "crash.cts.${t}::${MVX_CTS_CRASHER}" "CTS not built"
    fi
  done
}

part_cumask() {
  local commit url script gdf="${TEST_ROOT}/amd_openvx_gdfs/geometric/Remap_U8_U8_Bilinear_4K.gdf"
  commit="$(jq -r '.submodules[]? | select(.path=="mivisionx") | .commit' "${VP_MANIFEST}" 2>/dev/null)"
  url="$(jq -r '.submodules[]? | select(.path=="mivisionx") | .url' "${VP_MANIFEST}" 2>/dev/null)"
  if [[ -z "${commit}" || "${url}" != https://github.com/* ]]; then
    vp_blocked "cu-mask::remap_4K" "no MIVisionX commit/URL in ${VP_MANIFEST}"
    return
  fi
  url="${url%.git}"
  url="https://raw.githubusercontent.com/${url#https://github.com/}/${commit}/tests/hip_cu_mask_tests/test_hip_cu_mask_remap.py"
  # The installed ctest points at this script, which is not installed (M4); run the
  # upstream copy from the tested MIVisionX commit to check the feature itself.
  if ! script="$(mvx_fetch_file "mivisionx-${commit}" "${url}" test_hip_cu_mask_remap.py)"; then
    vp_blocked "cu-mask::remap_4K" "could not fetch ${url}"
    return
  fi
  mkdir -p "${VP_WORK}/cu_mask"
  cp "${script}" "${VP_WORK}/cu_mask/"
  vp_run "cu-mask::remap_4K" --timeout 900 --backend GPU --cwd "${VP_WORK}/cu_mask" --env AGO_DEFAULT_TARGET=GPU -- \
    "${PY}" test_hip_cu_mask_remap.py --runvx "${RUNVX}" --gdf "${gdf}" --cu-counts 2,4,8,16,32,all
}

part_race() {
  local b="${VP_WORK}/ctest-race"
  [[ "${HAVE_BUILD}" == 1 ]] || {
    vp_blocked "ctest-race::openvx_canny_parallel" "cmake/ninja not installed"
    return
  }
  # openvx_canny_CPU/_GPU run the binary openvx_canny builds, without a DEPENDS: in
  # parallel they start before it exists (why vp_ctest is serial).
  vp_run "ctest-race.build::configure" --timeout 600 -- cmake -S "${TEST_ROOT}" -B "${b}" -G Ninja \
    -DROCM_PATH="${ROCM_PATH}" -DPython3_EXECUTABLE="$(command -v "${PY}")" \
    && vp_run "ctest-race::openvx_canny_parallel" --timeout 600 --cwd "${b}" -- \
      ctest -j3 --timeout 300 -R '^openvx_canny(_CPU|_GPU)?$'
}

part_perf() {
  "${PY}" "${VP_SUITE_DIR}/perf.py" runvx --out "${VP_OUT}/raw/perf-runvx.json" >>"${VP_OUT}/logs/harness.log" 2>&1 \
    || vp_result "harness::perf.runvx" error "perf.py runvx failed; see logs/harness.log"
  vp_perf runvx-profile "${VP_OUT}/raw/perf-runvx.json"
  local src rc=0 b="${VP_WORK}/openvx-mark"
  src="$(mvx_git_cache openvx-mark "${MVX_OPENVX_MARK_URL}" "${MVX_OPENVX_MARK_REF:-main}" 0)" || rc=$?
  if [[ "${rc}" -ne 0 || "${HAVE_BUILD}" != 1 ]]; then
    vp_blocked "perf.openvx-mark::build" "could not fetch ${MVX_OPENVX_MARK_URL} or cmake/ninja missing"
    return
  fi
  cat "${src}/.vp-complete" >"${VP_OUT}/raw/openvx-mark-sha.txt"
  vp_cmake_build "perf.openvx-mark" "${src}" "${b}" \
    -DCMAKE_C_COMPILER="${ROCM_PATH}/lib/llvm/bin/amdclang" -DCMAKE_CXX_COMPILER="${ROCM_PATH}/lib/llvm/bin/amdclang++" \
    -DOPENVX_INCLUDES="${ROCM_PATH}/include/mivisionx" -DOPENVX_LIB_DIR="${ROCM_PATH}/lib" || return
  vp_run "perf.openvx-mark::validate-timing" --timeout 300 --backend GPU --cwd "${b}" --env AGO_DEFAULT_TARGET=GPU -- \
    ./openvx-mark --validate-timing
  # MIVisionX perf-gate-hip settings: 40 parity kernels, 4K, 100 iterations, 10 warmup, 1 thread
  vp_run "perf.openvx-mark::bench" --timeout 3600 --backend GPU --cwd "${b}" --env AGO_DEFAULT_TARGET=GPU -- \
    ./openvx-mark --kernel "${PARITY_KERNELS}" --skip-pipelines --resolution 4K --iterations 100 --warmup 10 \
    --threads 1 --output-dir "${b}/results"
  if [[ -f "${b}/results/benchmark_results.json" ]]; then
    cp "${b}/results/benchmark_results.json" "${VP_OUT}/raw/openvx-mark.json"
    "${PY}" "${VP_SUITE_DIR}/perf.py" openvx-mark --json "${b}/results/benchmark_results.json" \
      --out "${VP_OUT}/raw/perf-openvx-mark.json" >>"${VP_OUT}/logs/harness.log" 2>&1
    vp_perf openvx-mark "${VP_OUT}/raw/perf-openvx-mark.json"
  else
    vp_result "perf.openvx-mark::results" error "no benchmark_results.json" 0 "" GPU
  fi
}

timed() { # <part> <command...>: run a part when selected, and log its wall time
  local part="$1" t0=$SECONDS rc
  shift
  vp__log "part ${part} start"
  "$@"
  rc=$?
  echo "${part} $((SECONDS - t0))" >>"${VP_OUT}/raw/part-seconds.txt"
  vp__log "part ${part} done in $((SECONDS - t0))s"
  return "${rc}"
}

: >"${VP_OUT}/raw/part-seconds.txt"
want ctest && timed ctest part_ctest
want runvx-smoke && timed runvx-smoke part_runvx_smoke
want api && timed api part_api
want samples-gdf && timed samples-gdf part_samples_gdf
want api-probe && timed api-probe py api_probe.py --work "${VP_WORK}"
want samples-raw && timed samples-raw py samples_raw.py --work "${VP_WORK}"
want vxrpp && timed vxrpp py vxrpp.py --work "${VP_WORK}"
want parity && timed parity part_parity
want vision && timed vision part_vision
want gdf && timed gdf part_gdf
want cumask && timed cumask part_cumask
want race && timed race part_race
CTS_OK=0
if want cts || want cts-optional || want crash; then
  timed cts-build part_cts && CTS_OK=1
fi
if [[ "${CTS_OK}" == 1 ]]; then
  want cts && timed cts part_cts_required
  want cts-optional && timed cts-optional part_cts_optional
fi
want crash && timed crash part_crash
want perf && timed perf part_perf
vp_finish
exit 0
