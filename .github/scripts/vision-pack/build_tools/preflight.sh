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

# GPU pre-flight, run inside the test container before every GPU suite
# (TheRock's sanity pattern). Fails fast, within a few minutes, so a broken
# driver or runtime shows up as one clear infrastructure error instead of
# hundreds of test failures.
#
#   preflight.sh            (uses ROCM_PATH, VP_GFX, VP_OUT from the environment)
#
# Checks: rocminfo lists a GPU agent with the expected gfx; amd-smi static
# works (warning only); a tiny HIP kernel compiled for VP_GFX runs and returns
# the right answer. Writes ${VP_OUT}/preflight.log and preflight.json.
set -uo pipefail

: "${ROCM_PATH:?}"
out="${VP_OUT:-/tmp}"
gfx="${VP_GFX:-}"
log="${out}/preflight.log"
mkdir -p "${out}"
: >"${log}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
unset LD_LIBRARY_PATH HIP_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION

fail() {
  echo "::error title=GPU pre-flight::$*" | tee -a "${log}"
  jq -n --arg reason "$*" --arg gfx "${gfx}" '{ok:false, reason:$reason, gfx:$gfx}' >"${out}/preflight.json"
  exit 1
}

echo "== rocminfo" >>"${log}"
if ! timeout 120 rocminfo >>"${log}" 2>&1; then
  fail "rocminfo failed (driver/runtime problem); see preflight.log"
fi
agents="$(grep -E '^\s+Name:\s+gfx' "${log}" | awk '{print $2}' | sort -u | tr '\n' ' ')"
echo "GPU agents visible: ${agents}" | tee -a "${log}"
[[ -n "${agents}" ]] || fail "rocminfo shows no GPU agent"
if [[ -n "${gfx}" ]] && ! grep -qw "${gfx}" <<<"${agents}"; then
  fail "expected ${gfx} but rocminfo shows: ${agents}"
fi

echo "== amd-smi static" >>"${log}"
if command -v amd-smi >/dev/null 2>&1; then
  timeout 60 amd-smi static >>"${log}" 2>&1 || echo "::warning::amd-smi static failed (not fatal)" | tee -a "${log}"
fi

echo "== HIP sanity kernel" >>"${log}"
work="$(mktemp -d)"
cat >"${work}/sanity.hip" <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void axpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  const int n = 1 << 20;
  float *x, *y;
  if (hipMallocManaged(&x, n * sizeof(float)) != hipSuccess) { puts("alloc failed"); return 2; }
  if (hipMallocManaged(&y, n * sizeof(float)) != hipSuccess) { puts("alloc failed"); return 2; }
  for (int i = 0; i < n; ++i) { x[i] = 1.0f; y[i] = 2.0f; }
  hipLaunchKernelGGL(axpy, dim3((n + 255) / 256), dim3(256), 0, 0, n, 3.0f, x, y);
  hipError_t e = hipGetLastError();
  if (e == hipSuccess) e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("kernel failed: %s\n", hipGetErrorString(e)); return 3; }
  for (int i = 0; i < n; ++i) if (y[i] != 5.0f) { printf("wrong result at %d: %f\n", i, y[i]); return 4; }
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  printf("sanity ok on %s (%s)\n", p.name, p.gcnArchName);
  return 0;
}
EOF
arch_flag=()
[[ -n "${gfx}" ]] && arch_flag=(--offload-arch="${gfx}")
compiler="${ROCM_PATH}/bin/hipcc"
[[ -x "${compiler}" ]] || compiler="${ROCM_PATH}/lib/llvm/bin/amdclang++"
if ! timeout 300 "${compiler}" -x hip "${arch_flag[@]}" -O2 -o "${work}/sanity" "${work}/sanity.hip" >>"${log}" 2>&1; then
  fail "could not compile the HIP sanity kernel; see preflight.log"
fi
if ! timeout 120 "${work}/sanity" >>"${log}" 2>&1; then
  fail "HIP sanity kernel failed; see preflight.log"
fi
tail -1 "${log}"
jq -n --arg gfx "${gfx}" --arg agents "${agents}" '{ok:true, gfx:$gfx, agents:$agents}' >"${out}/preflight.json"
rm -rf "${work}"
echo "pre-flight ok"
