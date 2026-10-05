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

# Source this before anything touches HIP:
#
#   source build_tools/gpu_env.sh
#
# A CI runner that assigns one GPU per job sets ROCR_VISIBLE_DEVICES and
# HIP_VISIBLE_DEVICES to the same host index. The two stack: ROCr hides every
# other GPU, then HIP re-indexes into what is left, so HIP_VISIBLE_DEVICES=1
# selects nothing (hipErrorNoDevice). Keep ROCR_VISIBLE_DEVICES as the single
# GPU-visibility variable: a HIP_VISIBLE_DEVICES without ROCR_VISIBLE_DEVICES
# moves into ROCR_VISIBLE_DEVICES, and HIP_VISIBLE_DEVICES is always unset.
# Neither variable is ever left set to an empty string.

if [[ -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  unset ROCR_VISIBLE_DEVICES
  if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    printf 'gpu_env: moved HIP_VISIBLE_DEVICES=%s to ROCR_VISIBLE_DEVICES\n' "${ROCR_VISIBLE_DEVICES}" >&2
  fi
fi
unset HIP_VISIBLE_DEVICES
