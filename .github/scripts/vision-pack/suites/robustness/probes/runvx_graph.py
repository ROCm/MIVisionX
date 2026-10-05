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

"""Run a tiny verifiable OpenVX graph with runvx and judge the outcome (see verdict.py).

    runvx_graph.py --mode correct|honest|error [--affinity GPU]

The graph is absdiff(uniform 125, uniform 132), so every output byte must be 7. The GDF and its output
are written to the current directory (runvx chdir()s into the GDF's directory). Without --affinity the
target comes from AGO_DEFAULT_TARGET, which the caller sets for this test only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from verdict import crashed, finish

W, H = 64, 48
GDF = f"""data in1 = uniform-image:{W},{H},U008,125
data in2 = uniform-image:{W},{H},U008,132
data out = image:{W},{H},U008:WRITE,out.raw
node org.khronos.openvx.absdiff in1 in2 out
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["correct", "honest", "error"])
    ap.add_argument("--affinity", choices=["CPU", "GPU"])
    a = ap.parse_args()
    runvx = shutil.which("runvx")
    if runvx is None:
        print("runvx not found on PATH")
        finish(a.mode, "clean_error", "runvx missing")
    cwd = Path.cwd()
    (cwd / "absdiff.gdf").write_text(GDF)
    out = cwd / "out.raw"
    out.unlink(missing_ok=True)
    cmd = [runvx, "-frames:2"] + ([f"-affinity:{a.affinity}"] if a.affinity else []) + ["absdiff.gdf"]
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=240)
    print(p.stdout[-3000:], p.stderr[-3000:], sep="\n", flush=True)
    if p.returncode < 0:
        crashed(f"runvx killed by signal {-p.returncode}")
    # WRITE appends one image per frame.
    data = np.fromfile(out, dtype=np.uint8) if out.exists() else np.zeros(0, np.uint8)
    frames = data.size // (W * H)
    bad = int((data != 7).sum()) if frames >= 1 and data.size % (W * H) == 0 else W * H
    detail = f"runvx exit {p.returncode}, {frames} output frame(s) of {W}x{H}, wrong bytes {bad}"
    if p.returncode != 0:
        finish(a.mode, "clean_error", detail)
    finish(a.mode, "correct" if bad == 0 else "wrong", detail)


if __name__ == "__main__":
    main()
