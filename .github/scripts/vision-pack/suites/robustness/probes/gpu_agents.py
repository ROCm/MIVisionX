#!/usr/bin/env python3
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

"""Count the GPU agents this process can see, through rocminfo and through HIP.

    gpu_agents.py --expect N    exit 0 when both views report exactly N GPUs
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys


def rocminfo_gpus() -> tuple[list[str], str]:
    exe = shutil.which("rocminfo")
    if exe is None:
        return [], "rocminfo not found"
    p = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    gpus, name = [], None
    for line in p.stdout.splitlines():
        m = re.match(r"^\s{2}Name:\s+(\S+)", line)
        if m:
            name = m.group(1)
        if re.match(r"^\s+Device Type:\s+GPU", line) and name:
            gpus.append(name)
    return gpus, f"rocminfo exit {p.returncode}{(': ' + p.stderr.strip()[-300:]) if p.returncode else ''}"


def hip_count() -> tuple[int, str]:
    try:
        lib = ctypes.CDLL(os.path.join(os.environ["ROCM_PATH"], "lib", "libamdhip64.so"))
    except OSError as e:
        return -1, f"cannot load libamdhip64: {e}"
    n = ctypes.c_int(-1)
    rc = lib.hipGetDeviceCount(ctypes.byref(n))
    lib.hipGetErrorName.restype = ctypes.c_char_p
    return (n.value if rc == 0 else 0), f"hipGetDeviceCount -> {lib.hipGetErrorName(rc).decode()} count {n.value}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", type=int, required=True)
    a = ap.parse_args()
    gpus, note = rocminfo_gpus()
    count, hnote = hip_count()
    print(f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '(unset)')} "
          f"/dev/kfd {'present' if os.path.exists('/dev/kfd') else 'absent'}; "
          f"render nodes: {sorted(n for n in os.listdir('/dev/dri') if n.startswith('renderD')) if os.path.isdir('/dev/dri') else []}")
    print(f"{note}; GPU agents: {gpus}")
    print(hnote)
    ok = len(gpus) == a.expect and count == a.expect
    print(f"{'PASS' if ok else 'FAIL'}: expected {a.expect} GPUs, rocminfo sees {len(gpus)}, HIP sees {count}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
