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

# Run a ctest suite serially with JUnit output and re-run the failures once.
#
#   ctest_junit.sh <build_dir> <junit_prefix> [extra ctest args, e.g. -R/-E filters]
#
# Writes <junit_prefix>.xml, <junit_prefix>.count (registered tests) and, when
# anything failed, <junit_prefix>.rerun.xml. A test that fails the first run
# and passes the re-run is reported as flaky by emit.py ingest-junit.
#
# Serial on purpose: MIVisionX and rocAL re-run binaries built by earlier
# --build-and-test entries without DEPENDS, and several tests share working
# directories. ROCM_PATH must be exported because nested --build-and-test
# configures read $ENV{ROCM_PATH} and otherwise fall back to /opt/rocm.
set -uo pipefail

build="${1:?build dir}"
prefix="${2:?junit prefix}"
shift 2
: "${ROCM_PATH:?ROCM_PATH must be set}"
export ROCM_PATH

cd "${build}" || { echo "no build dir ${build}"; exit 2; }

count="$(ctest -N "$@" 2>/dev/null | awk '/Total Tests:/ {print $NF}')"
echo "${count:-0}" >"${prefix}.count"
echo "registered tests: ${count:-0}"

common=(--output-on-failure --timeout "${VP_CTEST_TIMEOUT:-1500}"
        --test-output-size-passed 4096 --test-output-size-failed 262144)

ctest "${common[@]}" --no-tests=error --output-junit "${prefix}.xml" "$@"
rc=$?
if [[ "${rc}" -ne 0 && -f "${prefix}.xml" ]]; then
  echo "### re-running failed tests once"
  ctest "${common[@]}" --rerun-failed --output-junit "${prefix}.rerun.xml" "$@" || true
fi
exit 0
