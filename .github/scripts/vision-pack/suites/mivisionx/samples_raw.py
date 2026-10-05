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

"""Shipped sample graphs (skintonedetect.gdf, canny) with raw input and numpy references.

The shipped GDFs read their input through OpenCV ("read input <jpg>"/camera),
which this runvx lacks (M2), so the input JPEGs are decoded with Pillow into
raw RGB and the graphs are re-emitted with raw read/write in a scratch dir.

  samples.<T>::skintone        output == numpy skin rule (C1: GPU U1 kernels)
  samples.<T>::canny_<k>_<L>   edge map vs numpy Canny on the graph's own luma,
                               at most CANNY_TOL of interior pixels may differ (M16)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from mvx_common import ROCM_PATH, RUNVX, VP_OUT, error_lines, record, run, tail, write_log

W, H = 480, 360
CANNY_TOL = 0.01

CANNY = """data input  = image:480,360,RGB2:read,{inp}
data output = image:480,360,U008:write,{out}
data yuv  = image-virtual:0,0,IYUV
data luma = image:480,360,U008:write,{luma}
node org.khronos.openvx.color_convert input yuv
node org.khronos.openvx.channel_extract yuv !CHANNEL_Y luma
data hyst = threshold:RANGE,U008,U008:INIT,80,100
data gradient_size = scalar:INT32,{gs}
node org.khronos.openvx.canny_edge_detector luma hyst gradient_size !{norm} output
"""


def skin_gdf(src: str, inp: Path, out: Path) -> str:
    lines = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("read input"):
            lines += [f"read input {inp}", f"write output {out}"]
        elif s.startswith(("view ", "read ", "write ")):
            continue
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def skin_ref(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.int64) for i in range(3))
    rmg, rmb = np.clip(r - g, 0, 255), np.clip(r - b, 0, 255)
    return (r > 95) & (g > 40) & (b > 20) & (rmg > 15) & (rmb > 0)


def sobel(a, k):
    d = {3: [-1, 0, 1], 5: [-1, -2, 0, 2, 1], 7: [-1, -4, -5, 0, 5, 4, 1]}[k]
    s = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [1, 6, 15, 20, 15, 6, 1]}[k]
    r = k // 2
    p = np.pad(a.astype(np.float64), r, mode="edge")

    def rows(img, ker):
        return sum(ker[i] * img[:, i:i + a.shape[1]] for i in range(k))

    def cols(img, ker):
        return sum(ker[i] * img[i:i + a.shape[0], :] for i in range(k))
    return cols(rows(p, d), s), cols(rows(p, s), d)


def canny_ref(luma, lo, hi, k, norm):
    gx, gy = sobel(luma, k)
    mag = np.abs(gx) + np.abs(gy) if norm == "NORM_L1" else np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    q = np.zeros(mag.shape, np.int32)
    q[(ang >= 22.5) & (ang < 67.5)] = 1
    q[(ang >= 67.5) & (ang < 112.5)] = 2
    q[(ang >= 112.5) & (ang < 157.5)] = 3
    m = np.pad(mag, 1)
    c = m[1:-1, 1:-1]
    nb = {0: (m[1:-1, :-2], m[1:-1, 2:]), 1: (m[:-2, :-2], m[2:, 2:]), 2: (m[:-2, 1:-1], m[2:, 1:-1]),
          3: (m[:-2, 2:], m[2:, :-2])}
    keep = np.zeros(mag.shape, bool)
    for d, (a1, a2) in nb.items():
        keep |= (q == d) & (c > a1) & (c >= a2)
    nms = np.where(keep, mag, 0)
    weak, edges = nms > lo, nms > hi
    while True:
        e = np.pad(edges, 1)
        grown = weak & (e[:-2, :-2] | e[:-2, 1:-1] | e[:-2, 2:] | e[1:-1, :-2] | e[1:-1, 2:] | e[2:, :-2] |
                        e[2:, 1:-1] | e[2:, 2:] | edges)
        if (grown == edges).all():
            return edges
        edges = grown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    try:
        from PIL import Image
    except ImportError:
        for t in ("CPU", "GPU"):
            record(f"samples.{t}::skintone", "blocked", "Pillow (python3-pil) is not installed", backend=t)
        return 0
    root = Path(a.work) / "samples_raw"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    share = ROCM_PATH / "share" / "mivisionx" / "samples"
    log = VP_OUT / "logs" / "samples_raw.log"
    face = np.asarray(Image.open(share / "images" / "face.jpg").convert("RGB").resize((W, H), Image.BILINEAR))
    face1 = np.asarray(Image.open(share / "images" / "face1.jpg").convert("RGB").resize((W, H), Image.BILINEAR))
    p_face, p_face1 = root / "face_480x360.rgb", root / "face1_480x360.rgb"
    face.tofile(p_face)
    face1.tofile(p_face1)
    skin_src = (share / "gdf" / "skintonedetect.gdf").read_text()
    ref_skin = skin_ref(face1)

    def go(name, text, t):
        gdf = root / (f"{name}_{t}.gdf")
        gdf.write_text(text)
        r = run([RUNVX, "-frames:1", f"-affinity:{t}", "-dump-profile", "file", gdf], a.timeout)
        write_log(log, f"### {name} {t} rc={int(r.rc)}", gdf.read_text(), r.out)
        return r

    for t in ("CPU", "GPU"):
        out = root / (f"skintone_{t}.u8")
        r = go("skintone", skin_gdf(skin_src, p_face1, out), t)
        tid = f"samples.{t}::skintone"
        if r.status() != "pass" or not out.exists():
            record(tid, r.status() if r.status() != "pass" else "fail", f"{r.why()}: {error_lines(r.out) or tail(r.out)}",
                   r.dt, log, t, r.repro())
        else:
            o = np.fromfile(out, np.uint8).reshape(H, W) > 0
            bad = o != ref_skin
            record(tid, "fail" if bad.any() else "pass",
                   f"skin pixels {100 * o.mean():.2f}% (numpy {100 * ref_skin.mean():.2f}%), {100 * bad.mean():.3f}% of pixels wrong", r.dt, log, t, r.repro())
        # 7x7 is left out: OpenVX scales 7x7 gradients before the thresholds, which this reference does not model
        for gs in (3, 5):
            for norm in ("NORM_L1", "NORM_L2"):
                name = f"canny_{int(gs)}_{norm}"
                out, luma = root / (f"{name}_{t}.u8"), root / (f"{name}_{t}_luma.u8")
                r = go(name, CANNY.format(inp=p_face, out=out, luma=luma, gs=gs, norm=norm), t)
                tid = f"samples.{t}::{name}"
                if r.status() != "pass" or not out.exists() or not luma.exists():
                    record(tid, r.status() if r.status() != "pass" else "fail",
                           f"{r.why()}: {error_lines(r.out) or tail(r.out)}", r.dt, log, t, r.repro())
                    continue
                o = np.fromfile(out, np.uint8).reshape(H, W) > 0
                lu = np.fromfile(luma, np.uint8).reshape(H, W)
                ref = canny_ref(lu, 80, 100, gs, norm)
                b = gs // 2 + 2
                frac = float((o != ref)[b:-b, b:-b].mean())
                record(tid, "pass" if frac <= CANNY_TOL else "fail",
                       f"edges {100 * o.mean():.2f}% (numpy {100 * ref.mean():.2f}%), {100 * frac:.3f}% of interior pixels differ from numpy (tolerance {100 * CANNY_TOL:.1f}%)", r.dt, log, t, r.repro())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
