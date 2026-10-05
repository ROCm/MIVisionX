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

"""CPU vs GPU parity and numpy references for MIVisionX kernels, via runvx.

Every case becomes a GDF with seeded random raw inputs (":read,") and raw
outputs (":write,") in a scratch directory. It runs twice per affinity; the
outputs are then judged per target:

  1. the run must exit 0 on both repetitions;
  2. both repetitions must be bit-identical (GPU races such as H15);
  3. with a numpy reference (thresholds, S16 multiply, AREA scaling, half-scale
     Gaussian, packed RGB/RGBX, U1 threshold->logic graphs, Box3x3 border
     modes) the output must match it within the case tolerance;
  4. without one, the GPU output must match the CPU output within the kernel
     tolerance on the interior (borders are implementation-defined), and the
     CPU result only has to run and be deterministic.

IDs: mivisionx::parity.<W>x<H>.<CPU|GPU>::<case>
"""
from __future__ import annotations

import argparse
import re
import shlex
import shutil
from pathlib import Path

import numpy as np
from mvx_common import RUNVX, VP_OUT, error_lines, load_vision_nodes, record, run, slug, tail, write_log

# border (pixels) excluded from CPU-vs-GPU comparisons, per kernel
BORDER = {"box_3x3": 1, "dilate_3x3": 1, "erode_3x3": 1, "median_3x3": 1, "gaussian_3x3": 1,
          "custom_convolution": 1, "sobel_3x3": 1, "canny_edge_detector": 3, "halfscale_gaussian": 2,
          "scale_image": 1, "warp_affine": 1, "warp_perspective": 1, "remap": 1, "non_linear_filter": 1,
          "harris_corners": 3, "fast_corners": 3, "gaussian_pyramid": 2}
# max |CPU-GPU| on the interior (rounding differences between the two code paths)
TOL = {"color_convert": 2, "gaussian_pyramid": 2}
EDGE_MAP_TOL = 0.005  # canny: fraction of interior pixels allowed to differ


def plane_layout(fmt, w, h):
    """[(rows, row_bytes, dtype, elements_per_pixel)] per plane."""
    one = {"U008": (np.uint8, 1, 1), "S016": (np.int16, 2, 1), "U016": (np.uint16, 2, 1), "U032": (np.uint32, 4, 1),
           "S032": (np.int32, 4, 1), "RGB2": (np.uint8, 3, 3), "RGBX": (np.uint8, 4, 4), "UYVY": (np.uint8, 2, 2),
           "YUYV": (np.uint8, 2, 2)}
    if fmt in one:
        dt, bpp, epp = one[fmt]
        return [(h, w * bpp, dt, epp)]
    if fmt == "IYUV":
        return [(h, w, np.uint8, 1), (h // 2, w // 2, np.uint8, 1), (h // 2, w // 2, np.uint8, 1)]
    if fmt in ("NV12", "NV21"):
        return [(h, w, np.uint8, 1), (h // 2, w, np.uint8, 2)]
    if fmt == "YUV4":
        return [(h, w, np.uint8, 1)] * 3
    raise ValueError(fmt)


def raw_size(fmt, w, h):
    return sum(r * b for r, b, _, _ in plane_layout(fmt, w, h))


def load_planes(path, fmt, w, h):
    buf = np.fromfile(path, dtype=np.uint8)
    planes, off = [], 0
    for rows, rb, dt, epp in plane_layout(fmt, w, h):
        n = rows * rb
        if off + n > buf.size:
            return None
        planes.append((buf[off:off + n].reshape(rows, rb).view(dt).astype(np.int64), epp))
        off += n
    return planes if off == buf.size else None


def as_image(path, fmt, w, h):
    """Single-plane image as (H, W) or (H, W, C) int64 array."""
    p = load_planes(path, fmt, w, h)
    if p is None:
        return None
    a, epp = p[0]
    return a.reshape(h, w, epp) if epp > 1 else a


class Case:
    def __init__(self, name, kernel, gdf, outputs, inputs=(), ref=None):
        self.name, self.kernel, self.gdf, self.outputs = name, kernel, gdf, outputs
        self.inputs = list(inputs)  # [(path, fmt, w, h)]
        self.ref = ref              # fn(case) -> {stem: (ref, tol, border, kind)}


class Inputs:
    def __init__(self, root: Path, seed: int):
        self.root, self.seed, self.n = root, seed, 0

    def make(self, fmt, w, h, tag=None):
        tag = tag or f"i{int(self.n)}"
        self.n += 1
        p = self.root / (f"{tag}_{fmt}_{int(w)}x{int(h)}.raw")
        if not p.exists():
            rng = np.random.default_rng([self.seed, w, h, sum(map(ord, fmt)), sum(map(ord, tag))])
            rng.integers(0, 256, size=raw_size(fmt, w, h), dtype=np.uint8).tofile(p)
        return p


def node_list_cases(W, H, inp: Inputs):
    cases, seen = [], {}
    for name, fmt in load_vision_nodes(W, H):
        seen[name] = seen.get(name, 0) + 1
        cname = name if seen[name] == 1 else f"{name}_{int(seen[name])}"
        toks = shlex.split(fmt)
        kernel, args = toks[0], toks[1:]
        lines, refs, attrs, outputs, ins = [], [], [], [], []
        seen_uniform = any(a.startswith("uniform-image:") for a in args)
        first_image_used = False
        for di, a in enumerate(args):
            if a.startswith("!"):
                refs.append(a)
                continue
            if a.startswith("attr:"):
                attrs.append(a)
                continue
            dn = f"d{int(di)}"
            a = a.replace(",RGBA", ",RGBX")  # the table's invalid 'RGBA' is covered by vision.static (M19)
            m = re.match(r"(uniform-image|image):(\d+),(\d+),(\w{4})", a)
            if m:
                kind, w, h, f = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                is_input = kind == "uniform-image" or (not seen_uniform and not first_image_used)
                if kind == "image" and not seen_uniform:
                    first_image_used = True
                if is_input:
                    p = inp.make(f, w, h, f"{slug(cname)}_{dn}")
                    ins.append((p, f, w, h))
                    lines.append(f"data {dn} = image:{int(w)},{int(h)},{f}:read,{p}")
                else:
                    lines.append(f"data {dn} = image:{w},{h},{f}:write,{{OUT}}/{dn}.raw")
                    outputs.append((dn, "image", f, w, h))
            elif a.startswith("lut:UINT8,256"):
                p = inp.root / "lut_u8_256.raw"
                if not p.exists():
                    np.random.default_rng(inp.seed).integers(0, 256, 256, dtype=np.uint8).tofile(p)
                lines.append(f"data {dn} = lut:UINT8,256:read,{p}")
            elif a.startswith("array:KEYPOINT"):
                lines.append(f"data {dn} = {a}:write,{{OUT}}/{dn}.txt")
                outputs.append((dn, "keypoints", None, int(a.split(",")[1]), 0))
            elif a.startswith("scalar:SIZE") and kernel.endswith("fast_corners"):
                lines.append(f"data {dn} = {a}:write,{{OUT}}/{dn}.txt")
                outputs.append((dn, "count", None, 0, 0))
            else:
                lines.append(f"data {dn} = {a}")
            refs.append(dn)
        lines.append(f"node {kernel} {' '.join(refs)} {' '.join(attrs)}")
        cases.append(Case(cname, kernel.split(".")[-1], "\n".join(lines) + "\n", outputs, ins, NODE_REFS.get(name)))
    return cases


# ---------------------------------------------------------------------------
# numpy references for node-table cases
# ---------------------------------------------------------------------------

def _in(c, i):
    p, f, w, h = c.inputs[i]
    return as_image(p, f, w, h)


def _out(c, i=0):
    return c.outputs[i]


def ref_threshold(kind, lo, hi=None):
    def f(c):
        a = _in(c, 0)
        t = (a > lo) if kind == "binary" else ((a >= lo) & (a <= hi))
        return {_out(c)[0]: (np.where(t, 255, 0), 0, 0, "exact")}
    return f


def ref_mul_s16(wrap):
    def f(c):
        a, b = _in(c, 0), _in(c, 1)
        p = a * b
        r = ((p + 32768) % 65536) - 32768 if wrap else np.clip(p, -32768, 32767)
        return {_out(c)[0]: (r, 0, 0, "exact")}
    return f


def ref_combine(c):
    planes = [_in(c, i) for i in range(len(c.inputs))]
    return {_out(c)[0]: (np.stack(planes, axis=-1), 0, 0, "exact")}


def ref_rgbx_to_rgb(c):
    return {_out(c)[0]: (_in(c, 0)[:, :, :3], 0, 0, "exact")}


def ref_rgb_to_rgbx(c):
    a = _in(c, 0)
    return {_out(c)[0]: (np.concatenate([a, np.full(a.shape[:2] + (1,), 255)], axis=-1), 0, 0, "exact")}


def block_mean(a, k, oh, ow):
    s = a[: oh * k, : ow * k].reshape(oh, k, ow, k).sum(axis=(1, 3))
    return (s + (k * k) // 2) // (k * k)


def ref_area(c):
    _, _, _, ow, oh = _out(c)
    a = _in(c, 0)
    k = a.shape[1] // ow
    return {_out(c)[0]: (block_mean(a, k, oh, ow), 1, 1, "tol")}


NODE_REFS = {
    "Threshold_U8_U8_Binary": ref_threshold("binary", 127),
    "Threshold_U8_U8_Range": ref_threshold("range", 100, 200),
    "Threshold_U8_S16_Binary": ref_threshold("binary", 127),
    "Threshold_U8_S16_Range": ref_threshold("range", 100, 200),
    "Mul_S16_S16S16_Wrap_Trunc": ref_mul_s16(True),
    "Mul_S16_S16S16_Wrap_Round": ref_mul_s16(True),
    "Mul_S16_S16S16_Sat_Trunc": ref_mul_s16(False),
    "Mul_S16_S16S16_Sat_Round": ref_mul_s16(False),
    "ChannelCombine_U32_U8U8U8U8_RGBX": ref_combine,
    "ColorConvert_RGB_RGBX": ref_rgbx_to_rgb,
    "ColorConvert_RGBX_RGB": ref_rgb_to_rgbx,
    "ScaleImage_U8_U8_Area": ref_area,
    # ScaleGaussianHalf has no reference on purpose: the spec (and the CTS, which accepts any of 9 source
    # positions) leaves the sampling phase open, so M16 is judged as CPU/GPU disagreement.
}


# ---------------------------------------------------------------------------
# cases beyond the runVisionTests node table
# ---------------------------------------------------------------------------

def extra_cases(W, H, inp: Inputs):
    u8a = inp.make("U008", W, H, "xa")
    u8b = inp.make("U008", W, H, "xb")
    u8c = inp.make("U008", W, H, "xc")
    s16a = inp.make("S016", W, H, "xs")
    rgb = inp.make("RGB2", W, H, "xrgb")
    ex = []

    def add(name, kernel, body, outputs, inputs=(), ref=None):
        ex.append(Case(name, kernel, body, outputs, inputs, ref))

    def img(p, f="U008", w=W, h=H):
        return f"image:{int(w)},{int(h)},{f}:read,{p}"

    def wr(stem, fmt="U008", w=W, h=H, ext="raw"):
        # {OUT} is filled in per run by run_case(), so it stays a literal here
        return f"image:{w},{h},{fmt}:write,{{OUT}}/{stem}.{ext}"

    def gdf(*lines):
        return "".join(ln + "\n" for ln in lines)

    rm = f"remap:{W},{H},{W},{H}"
    add("Remap_U8_Bilinear_hflip", "remap",
        gdf(f"data in = {img(u8a)}", f"data rm = {rm}:init,hflip", f"data out = {wr('out')}",
            "node org.khronos.openvx.remap in rm !BILINEAR out"),
        [("out", "image", "U008", W, H)])
    add("Remap_U8_Nearest_vflip", "remap",
        gdf(f"data in = {img(u8a)}", f"data rm = {rm}:init,vflip", f"data out = {wr('out')}",
            "node org.khronos.openvx.remap in rm !NEAREST_NEIGHBOR out"),
        [("out", "image", "U008", W, H)])
    add("Remap_RGB_Bilinear_hflip", "remap",
        gdf(f"data in = {img(rgb, 'RGB2')}", f"data rm = {rm}:init,hflip", f"data out = {wr('out', 'RGB2')}",
            "node org.khronos.openvx.remap in rm !BILINEAR out"),
        [("out", "image", "RGB2", W, H)])
    add("WarpPerspective_U8_Bilinear", "warp_perspective",
        gdf(f"data in = {img(u8a)}", "data m = matrix:FLOAT32,3,3:INIT,{0.9;0.05;0.0001;0.1;1.1;0.00005;10;20;1}",
            f"data out = {wr('out')}",
            "node org.khronos.openvx.warp_perspective in m !BILINEAR out attr:BORDER_MODE:CONSTANT,0"),
        [("out", "image", "U008", W, H)])
    add("EqualizeHist_U8", "equalize_histogram",
        gdf(f"data in = {img(u8a)}", f"data out = {wr('out')}", "node org.khronos.openvx.equalize_histogram in out"),
        [("out", "image", "U008", W, H)])
    add("IntegralImage_U32_U8", "integral_image",
        gdf(f"data in = {img(u8a)}", f"data out = {wr('out', 'U032')}", "node org.khronos.openvx.integral_image in out"),
        [("out", "image", "U032", W, H)])
    add("Accumulate_S16_U8", "accumulate",
        gdf(f"data in = {img(u8a)}", f"data acc = image:{W},{H},S016:read,{s16a}:write,{{OUT}}/out.raw",
            "node org.khronos.openvx.accumulate in acc"),
        [("out", "image", "S016", W, H)])
    add("AccumulateWeighted_U8", "accumulate_weighted",
        gdf(f"data in = {img(u8a)}", "data alpha = scalar:FLOAT32,0.3",
            f"data acc = image:{W},{H},U008:read,{u8b}:write,{{OUT}}/out.raw",
            "node org.khronos.openvx.accumulate_weighted in alpha acc"),
        [("out", "image", "U008", W, H)])
    add("AccumulateSquare_S16_U8", "accumulate_square",
        gdf(f"data in = {img(u8a)}", "data sh = scalar:UINT32,4",
            f"data acc = image:{W},{H},S016:read,{s16a}:write,{{OUT}}/out.raw",
            "node org.khronos.openvx.accumulate_square in sh acc"),
        [("out", "image", "S016", W, H)])
    add("NonLinearFilter_Median_Box3x3", "non_linear_filter",
        gdf(f"data in = {img(u8a)}", "data mask = matrix:UINT8,3,3:INIT,{255;255;255;255;255;255;255;255;255}",
            "data fn = scalar:ENUM,VX_NONLINEAR_FILTER_MEDIAN", f"data out = {wr('out')}",
            "node org.khronos.openvx.non_linear_filter fn in mask out"),
        [("out", "image", "U008", W, H)])
    add("Magnitude_Phase_via_Sobel", "sobel_3x3",
        gdf(f"data in = {img(u8a)}", f"data gx = {wr('gx', 'S016')}", f"data gy = {wr('gy', 'S016')}",
            f"data mag = {wr('mag', 'S016')}", f"data ph = {wr('ph')}",
            "node org.khronos.openvx.sobel_3x3 in gx gy", "node org.khronos.openvx.magnitude gx gy mag",
            "node org.khronos.openvx.phase gx gy ph"),
        [("gx", "image", "S016", W, H), ("gy", "image", "S016", W, H), ("mag", "image", "S016", W, H),
         ("ph", "image", "U008", W, H)])
    add("HarrisCorners_3x3", "harris_corners",
        gdf(f"data in = {img(u8a)}", "data corners = array:KEYPOINT,20000:write,{OUT}/kp.txt",
            "data num = scalar:SIZE,0:write,{OUT}/n.txt", "data st = scalar:FLOAT32,0.0005",
            "data md = scalar:FLOAT32,5.0", "data se = scalar:FLOAT32,0.04", "data gs = scalar:INT32,3",
            "data bs = scalar:INT32,3", "node org.khronos.openvx.harris_corners in st md se gs bs corners num"),
        [("kp", "keypoints", None, 20000, 0), ("n", "count", None, 0, 0)])
    add("GaussianPyramid_4L", "gaussian_pyramid",
        gdf(f"data in = {img(u8a)}", f"data pyr = pyramid:4,HALF,{W},{H},U008:write,{{OUT}}/L%d.raw",
            "node org.khronos.openvx.gaussian_pyramid in pyr"),
        [(f"L{i}", "image", "U008", -(-W // (1 << i)), -(-H // (1 << i))) for i in range(4)])
    add("Copy_U8", "copy",
        gdf(f"data in = {img(u8a)}", f"data out = {wr('out')}", "node org.khronos.openvx.copy in out"),
        [("out", "image", "U008", W, H)], [(u8a, "U008", W, H)],
        lambda c: {"out": (_in(c, 0), 0, 0, "exact")})
    add("Histogram_256", "histogram",
        gdf(f"data in = {img(u8a)}", "data hist = distribution:256,0,256:write,{OUT}/hist.raw",
            "node org.khronos.openvx.histogram in hist"),
        [("hist", "raw32", None, 256, 1)])
    add("MeanStdDev_U8", "mean_stddev",
        gdf(f"data in = {img(u8a)}", "data mean = scalar:FLOAT32,0.0:write,{OUT}/mean.txt",
            "data sd = scalar:FLOAT32,0.0:write,{OUT}/sd.txt", "node org.khronos.openvx.mean_stddev in mean sd"),
        [("mean", "scalar", None, 0, 0), ("sd", "scalar", None, 0, 0)])
    add("MinMaxLoc_U8", "minmaxloc",
        gdf(f"data in = {img(u8a)}", "data mn = scalar:UINT8,0:write,{OUT}/mn.txt",
            "data mx = scalar:UINT8,0:write,{OUT}/mx.txt", "data mnc = scalar:SIZE,0:write,{OUT}/mnc.txt",
            "data mxc = scalar:SIZE,0:write,{OUT}/mxc.txt",
            "node org.khronos.openvx.minmaxloc in mn mx NULL NULL mnc mxc"),
        [(s, "scalar", None, 0, 0) for s in ("mn", "mx", "mnc", "mxc")])
    add("ChannelCombine_RGB", "channel_combine",
        gdf(f"data r = {img(u8a)}", f"data g = {img(u8b)}", f"data b = {img(u8c)}", f"data out = {wr('out', 'RGB2')}",
            "node org.khronos.openvx.channel_combine r g b null out"),
        [("out", "image", "RGB2", W, H)], [(u8a, "U008", W, H), (u8b, "U008", W, H), (u8c, "U008", W, H)], ref_combine)
    add("ColorConvert_RGB_IYUV_RGB_roundtrip", "color_convert_roundtrip",
        gdf(f"data in = {img(rgb, 'RGB2')}", f"data yuv = {wr('yuv', 'IYUV')}", f"data back = {wr('back', 'RGB2')}",
            "node org.khronos.openvx.color_convert in yuv", "node org.khronos.openvx.color_convert yuv back"),
        [("yuv", "image", "IYUV", W, H), ("back", "image", "RGB2", W, H)])
    # H14: AREA scaling at 3x (the node table only has 2x); the input is an exact multiple of the output
    ow3, oh3 = W // 3, H // 3
    u8_3x = inp.make("U008", ow3 * 3, oh3 * 3, "x3")
    add("ScaleImage_U8_U8_Area_3x", "scale_image",
        gdf(f"data in = {img(u8_3x, 'U008', ow3 * 3, oh3 * 3)}", f"data out = {wr('out', 'U008', ow3, oh3)}",
            "node org.khronos.openvx.scale_image in out !AREA"),
        [("out", "image", "U008", ow3, oh3)], [(u8_3x, "U008", ow3 * 3, oh3 * 3)], ref_area)
    # C1: 1-bit threshold -> logical graphs (the optimizer lowers them to U1 kernels)
    thr = {"a": ("v1", 95), "b": ("v2", 40), "c": ("v3", 20)}
    u1 = {
        "U1_Threshold_And": ("ab", ["and v1 v2 out"], lambda a, b, c: (a > 95) & (b > 40)),
        "U1_Threshold_Or": ("ab", ["or v1 v2 out"], lambda a, b, c: (a > 95) | (b > 40)),
        "U1_Threshold_Xor": ("ab", ["xor v1 v2 out"], lambda a, b, c: (a > 95) ^ (b > 40)),
        "U1_Threshold_Not": ("a", ["not v1 out"], lambda a, b, c: ~(a > 95)),
        "U1_Threshold_And3": ("abc", ["and v1 v2 v12", "and v12 v3 out"],
                              lambda a, b, c: (a > 95) & (b > 40) & (c > 20)),
    }
    for name, (srcs, ops, fn) in u1.items():
        # every virtual image must be consumed, or vxVerifyGraph rejects the graph
        virt = [thr[s][0] for s in srcs] + (["v12"] if len(ops) > 1 else [])
        body = (f"data a = {img(u8a)}\ndata b = {img(u8b)}\ndata c = {img(u8c)}\n" +
                "".join(f"data t{s} = threshold:BINARY,U008,U008:INIT,{int(thr[s][1])}\n" for s in srcs) +
                "".join(f"data {v} = image-virtual:0,0,U008\n" for v in virt) +
                f"data out = {wr('out')}\n" +
                "".join(f"node org.khronos.openvx.threshold {s} t{s} {thr[s][0]}\n" for s in srcs) +
                "".join(f"node org.khronos.openvx.{op}\n" for op in ops))

        def ref(c, fn=fn):
            a, b, cc = (_in(c, i) for i in range(3))
            return {"out": (fn(a, b, cc), 0, 0, "bool")}
        add(name, "u1_logic", body, [("out", "image", "U008", W, H)],
            [(u8a, "U008", W, H), (u8b, "U008", W, H), (u8c, "U008", W, H)], ref)
    # M17: border modes must be honoured (or rejected), not ignored
    for mode, padmode in (("REPLICATE", "edge"), ("CONSTANT,0", "constant")):
        def ref_box(c, padmode=padmode):
            a = _in(c, 0)
            p = np.pad(a, 1, mode=padmode)
            s = sum(p[y: y + a.shape[0], x: x + a.shape[1]] for y in range(3) for x in range(3))
            return {"out": ([s // 9, (s + 4) // 9], 1, 0, "tol")}
        add(f"Box3x3_Border_{mode.split(',')[0].title()}", "box_3x3",
            gdf(f"data in = {img(u8a)}", f"data out = {wr('out')}",
                f"node org.khronos.openvx.box_3x3 in out attr:BORDER_MODE:{mode}"),
            [("out", "image", "U008", W, H)], [(u8a, "U008", W, H)], ref_box)
    return ex


# ---------------------------------------------------------------------------
# execution and comparison
# ---------------------------------------------------------------------------

def run_case(case, aff, outdir: Path, gdfdir: Path, timeout):
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.iterdir():
        f.unlink()
    gdf = gdfdir / (f"{slug(case.name)}_{outdir.name}.gdf")
    gdf.write_text(case.gdf.replace("{OUT}", str(outdir)))
    return run([RUNVX, "-frames:1", f"-affinity:{aff}", "-dump-profile", "file", gdf], timeout), gdf


def read_kp(path):
    if not path.exists():
        return None
    return {(int(f[0]), int(f[1])) for f in (ln.split() for ln in path.read_text().splitlines()) if len(f) >= 3}


def read_scalar(path):
    return path.read_text().strip() if path.exists() else None


def diff_images(a_path, b_path, o, border, phase=False):
    """(max interior |diff|, mismatch count, interior mismatch fraction) or a string on shape errors."""
    stem, _, fmt, w, h = o
    A, B = load_planes(a_path, fmt, w, h), load_planes(b_path, fmt, w, h)
    if A is None or B is None:
        return "output missing or of the wrong size"
    mx, mism, im, it = 0, 0, 0, 0
    for (a, epp), (b, _) in zip(A, B, strict=False):
        d = np.abs(a - b)
        if phase:
            d = np.minimum(d, 256 - d)
        mism += int((d != 0).sum())
        r, cc = d.shape
        bc = border * epp
        di = d[border:r - border, bc:cc - bc] if (r > 2 * border and cc > 2 * bc) else d
        if di.size:
            mx = max(mx, int(di.max()))
            im += int((di != 0).sum())
            it += di.size
    return mx, mism, (im / it if it else 0.0)


def same_output(o, da: Path, db: Path):
    stem, kind = o[0], o[1]
    if kind in ("image", "raw32"):
        pa, pb = da / (stem + ".raw"), db / (stem + ".raw")
        if not (pa.exists() and pb.exists()):
            return False
        return pa.read_bytes() == pb.read_bytes()
    if kind == "keypoints":
        a, b = read_kp(da / (stem + ".txt")), read_kp(db / (stem + ".txt"))
        if a is not None and b is not None and max(len(a), len(b)) >= o[3]:
            return True  # a full array keeps an arbitrary subset; the count output is checked instead
        return a == b
    return read_scalar(da / (stem + ".txt")) == read_scalar(db / (stem + ".txt"))


def compare_ref(case, o, outdir: Path, refs):
    stem, _, fmt, w, h = o
    ref, tol, border, mode = refs[stem]
    got = as_image(outdir / (stem + ".raw"), fmt, w, h)
    if got is None:
        return False, f"{stem}: output missing or wrong size"
    cands = ref if isinstance(ref, list) else [ref]
    best = None
    for r in cands:
        r = np.asarray(r).reshape(got.shape) if np.asarray(r).size == got.size else np.asarray(r)
        if mode == "bool":
            d = ((got > 0) != r.astype(bool)).astype(np.int64)
        else:
            d = np.abs(got - r)
        if border:
            d = d[border:-border, border:-border]
        res = (int(d.max()) if d.size else 0, int((d != 0).sum()), d.size)
        if best is None or res[:2] < best[:2]:
            best = res
    mx, mism, tot = best
    lim = 0 if mode in ("exact", "bool") else tol
    return mx <= lim, (f"{stem} vs numpy reference: {mism}/{tot} pixels differ ({100.0 * mism / max(1, tot):.3f}%), "
                       f"max diff {mx} (tolerance {lim})")


def compare_cpu(case, o, cpu: Path, gpu: Path):
    stem, kind = o[0], o[1]
    k = case.kernel
    if kind == "image":
        border = BORDER.get(k, 0)
        tol = TOL.get(k, 1)
        if k == "color_convert_roundtrip":
            tol = 2 if stem == "yuv" else 4
        r = diff_images(cpu / (stem + ".raw"), gpu / (stem + ".raw"), o, border, phase=(k == "phase" or stem == "ph"))
        if isinstance(r, str):
            return False, f"{stem}: {r}"
        mx, mism, frac = r
        if k == "canny_edge_detector":
            ok = frac <= EDGE_MAP_TOL
            return ok, f"{stem}: {100 * frac:.3f}% of interior edge pixels differ from CPU (tolerance {100 * EDGE_MAP_TOL:.1f}%)"
        return mx <= tol, f"{stem}: interior max |GPU-CPU| {int(mx)} (tolerance {int(tol)}), {int(mism)} pixels differ overall"
    if kind == "raw32":
        a, b = np.fromfile(cpu / (stem + ".raw"), np.uint32), np.fromfile(gpu / (stem + ".raw"), np.uint32)
        ok = a.shape == b.shape and bool((a == b).all())
        return ok, f"{stem}: {'identical' if ok else 'differs from CPU'}"
    if kind == "keypoints":
        a, b = read_kp(cpu / (stem + ".txt")), read_kp(gpu / (stem + ".txt"))
        if a is None or b is None:
            return False, f"{stem}: keypoint output missing"
        if len(a) >= o[3] or len(b) >= o[3]:
            return True, f"{stem}: keypoint array full (capacity {int(o[3])}); compared through the count"
        j = len(a & b) / max(1, len(a | b))
        return j >= 0.9, f"{stem}: keypoint Jaccard {j:.3f} vs CPU (need >= 0.9)"
    va, vb = read_scalar(cpu / (stem + ".txt")), read_scalar(gpu / (stem + ".txt"))
    if va is None or vb is None:
        return False, f"{stem}: scalar output missing"
    try:
        fa, fb = float(va), float(vb)
    except ValueError:
        return va == vb, f"{stem}: CPU {va[:30]} GPU {vb[:30]}"
    rel = abs(fa - fb) / max(1e-9, abs(fa))
    lim = 0.02 if kind == "count" else 1e-3
    return rel <= lim, f"{stem}: CPU {va} GPU {vb} (rel diff {rel:.4f}, tolerance {lim:.3f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    W, H = a.width, a.height
    size = f"{int(W)}x{int(H)}"
    root = Path(a.work) / (f"parity-{size}")
    shutil.rmtree(root, ignore_errors=True)
    indir, gdfdir = root / "inputs", root / "gdf"
    indir.mkdir(parents=True)
    gdfdir.mkdir()
    inp = Inputs(indir, a.seed)
    cases = node_list_cases(W, H, inp) + extra_cases(W, H, inp)
    if a.filter:
        cases = [c for c in cases if re.search(a.filter, c.name)]
    log = VP_OUT / "logs" / (f"parity.{size}.log")
    counts: dict[str, int] = {}
    for c in cases:
        runs, gdfs = {}, {}
        for aff in ("CPU", "GPU"):
            for rep in (1, 2):
                tag = f"{aff.lower()}{int(rep)}"
                runs[tag], gdfs[tag] = run_case(c, aff, root / tag, gdfdir, a.timeout)
        refs = None
        if c.ref and c.inputs:
            try:
                refs = c.ref(c)
            except Exception as e:  # noqa: BLE001 - reported, never hidden
                refs = None
                record(f"parity.{size}::{c.name}.reference", "error", f"reference failed: {e!r}")
        lines = [f"### {c.name}"]
        for aff in ("CPU", "GPU"):
            r1, r2 = runs[aff.lower() + "1"], runs[aff.lower() + "2"]
            d1 = root / (aff.lower() + "1")
            repro = r1.repro(cwd=None)
            msgs, status = [], "pass"
            bad = [r for r in (r1, r2) if r.status() != "pass"]
            if bad:
                status = bad[0].status()
                msgs.append(f"runvx {bad[0].why()}: {error_lines(bad[0].out, 2) or tail(bad[0].out, 300)}")
            else:
                nd = [o[0] for o in c.outputs if not same_output(o, d1, root / (aff.lower() + "2"))]
                if nd:
                    status = "fail"
                    msgs.append(f"nondeterministic: repeated {aff} runs differ in {', '.join(nd)}")
                for o in c.outputs:
                    if refs and o[0] in refs:
                        ok, m = compare_ref(c, o, d1, refs)
                    elif aff == "GPU":
                        if runs["cpu1"].status() != "pass":
                            status = "error" if status == "pass" else status
                            msgs.append(f"{o[0]}: no reference (CPU run failed)")
                            continue
                        ok, m = compare_cpu(c, o, root / "cpu1", d1)
                    else:
                        continue
                    if not ok:
                        status = "fail"
                    msgs.append(m)
            counts[status] = counts.get(status, 0) + 1
            msg = "; ".join(str(m) for m in msgs)
            lines.append(f"{aff} {status} rc={int(r1.rc)},{int(r2.rc)} {msg}")
            record(f"parity.{size}.{aff}::{c.name}", status, msg if status != "pass" else msg[:300],
                   r1.dt + r2.dt, log, aff, f"{repro}  # GDF: {gdfs[aff.lower() + '1']}")
        write_log(log, *lines, *(f"  {g}\n{g.read_text()}" for g in (gdfs["gpu1"],)))
    print(f"parity {size}: {len(cases)} cases, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
