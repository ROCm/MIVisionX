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

"""vx_rpp harness: build vxrpp_test.cpp against the prefix, run every case on CPU and GPU.

  vxrpp.<T>::<case>          the graph verifies and executes twice (vx_rpp gaps:
                             ColorJitter GPU, Glitch, ColorToGreyscale NHWC output)
  vxrpp.<T>.repeat::<case>   second execution == first execution (H10)
  vxrpp.<T>.ref::<case>      output == numpy reference on the interior (H6: Blur,
                             Median, Gaussian, Dilate, Erode shifted by 12 bytes)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from mvx_common import ROCM_PATH, VP_OUT, record, run, tail, write_log

N, H, W, C = 2, 120, 160, 3
MARGIN = 3
SHIFT_PX = 4  # H6 signature: output offset by 12 bytes = 4 RGB pixels
REF_CASES = ("Copy", "Flip", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "ChannelPermute", "Dilate3", "Erode3",
             "MedianFilter3", "Blur", "GaussianFilter5")


def win(a, k):
    """(N, H, W, C, k*k) stack of the kxk neighbourhood, edge-padded."""
    r = k // 2
    p = np.pad(a, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    return np.stack([p[:, y:y + H, x:x + W, :] for y in range(k) for x in range(k)], axis=-1)


def gauss5(a, sigmas):
    out = np.empty(a.shape, np.float64)
    for n, s in enumerate(sigmas):
        x = np.arange(-2, 3, dtype=np.float64)
        g = np.exp(-(x * x) / (2 * s * s))
        k2 = np.outer(g, g)
        k2 /= k2.sum()
        w = win(a[n:n + 1].astype(np.float64), 5)
        out[n] = (w * k2.reshape(-1)).sum(axis=-1)[0]
    return np.clip(np.rint(out), 0, 255)


def references(a, b):
    ai = a.astype(np.int64)
    flip = ai.copy()
    flip[0] = ai[0, :, ::-1, :]
    flip[1] = ai[1, ::-1, :, :]
    perm = np.stack([ai[0][..., [2, 1, 0]], ai[1][..., [1, 2, 0]]])
    return {  # name: (reference, tolerance)
        "Copy": (ai, 0),
        "Flip": (flip, 0),
        "BitwiseAnd": (ai & b, 0),
        "BitwiseOr": (ai | b, 0),
        "BitwiseXor": (ai ^ b, 0),
        "ChannelPermute": (perm, 0),
        "Dilate3": (win(ai, 3).max(axis=-1), 0),
        "Erode3": (win(ai, 3).min(axis=-1), 0),
        "MedianFilter3": (np.median(win(ai, 3), axis=-1).astype(np.int64), 0),
        "Blur": (np.rint(win(ai, 3).mean(axis=-1)).astype(np.int64), 1),
        "GaussianFilter5": (gauss5(ai, (1.0, 2.0)).astype(np.int64), 2),
    }


def cmp_ref(out, ref, tol):
    o = out.reshape(N, H, W, C).astype(np.int64)
    d = np.abs(o - ref)[:, MARGIN:-MARGIN, MARGIN:-MARGIN, :]
    bad = float((d > tol).mean())
    # H6: output byte i holds reference byte i + 12 (4 RGB pixels to the left)
    shifted = np.roll(o.reshape(-1), SHIFT_PX * C).reshape(o.shape)
    ds = np.abs(shifted - ref)[:, MARGIN:-MARGIN, MARGIN + SHIFT_PX:-MARGIN - SHIFT_PX, :]
    sig = f"; the output is the reference shifted left by {SHIFT_PX} px (12 bytes)" if (ds > tol).mean() < 0.001 else ""
    return bad == 0.0, f"{100 * bad:.2f}% of interior values differ from numpy by more than {tol} (max {int(d.max())}){sig}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    root = Path(a.work) / "vxrpp"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    log = VP_OUT / "logs" / "vxrpp.log"
    exe = root / "vxrpp_test"
    src = Path(__file__).with_name("vxrpp_test.cpp")
    cc = ROCM_PATH / "lib" / "llvm" / "bin" / "amdclang++"
    r = run([cc, "-O2", "-std=c++17", f"-I{ROCM_PATH / 'include' / 'mivisionx'}", src, "-o", exe,
             f"-L{ROCM_PATH / 'lib'}", "-lopenvx", "-lvx_rpp", f"-Wl,-rpath,{ROCM_PATH / 'lib'}"], 600)
    write_log(log, f"### build: {r.repro()}", r.out)
    record("vxrpp.build::vxrpp_test", r.status(), "" if r.status() == "pass" else f"{r.why()}: {tail(r.out)}",
           r.dt, log, "", r.repro())
    if r.status() != "pass":
        return 0
    names = run([exe, "CPU", root, "list"], 60).out.split()
    for t in ("CPU", "GPU"):
        d = root / t.lower()
        d.mkdir()
        refs = None
        for name in names:
            r = run([exe, t, d, name], a.timeout)
            write_log(log, f"### {t} {name} rc={int(r.rc)}", r.out)
            res = [ln for ln in r.out.splitlines() if ln.startswith("RESULT ")]
            repro = r.repro()
            deterministic = not name.startswith("rng_") and name != "Nop"
            if r.status() != "pass" or not res or " PASS " not in res[0] + " ":
                status = "error" if r.status() == "error" else "fail"
                record(f"vxrpp.{t}::{name}", status, f"{r.why()}: {res[0] if res else tail(r.out, 300)}",
                       r.dt, log, t, repro)
                # keep the ID set stable: the follow-up checks are not applicable without an output
                for sub, applies in (("repeat", deterministic), ("ref", name in REF_CASES)):
                    if applies:
                        record(f"vxrpp.{t}.{sub}::{name}", "skip", f"the graph did not execute (vxrpp.{t}::{name})",
                               0, log, t, repro)
                continue
            record(f"vxrpp.{t}::{name}", "pass", res[0], r.dt, log, t, repro)
            out, run1 = (d / (name + ".bin")).read_bytes(), (d / (name + ".run1.bin")).read_bytes()
            if deterministic:
                o1, o2 = np.frombuffer(run1, np.uint8), np.frombuffer(out, np.uint8)
                if o1.size != o2.size:
                    msg = f"output size changed between executions ({o1.size} -> {o2.size} bytes)"
                else:
                    idx = np.flatnonzero(o1 != o2)
                    msg = "" if idx.size == 0 else (
                        f"second graph execution differs from the first on {idx.size} of {o1.size} bytes "
                        f"({100 * idx.size / o1.size:.2f}%), first at byte {idx[0]}")
                record(f"vxrpp.{t}.repeat::{name}", "fail" if msg else "pass", msg, 0, log, t, repro)
            if refs is None:
                ina = np.fromfile(d / "in.bin", np.uint8).reshape(N, H, W, C)
                inb = np.fromfile(d / "in2.bin", np.uint8).reshape(N, H, W, C).astype(np.int64)
                refs = references(ina, inb)
            if name in refs:
                ok, msg = cmp_ref(np.frombuffer(run1, np.uint8), *refs[name])
                record(f"vxrpp.{t}.ref::{name}", "pass" if ok else "fail", "first execution: " + msg, 0, log, t,
                       repro)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
