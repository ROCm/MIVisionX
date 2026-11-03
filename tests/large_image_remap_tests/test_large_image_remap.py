#!/usr/bin/env python3
"""
CTest regression test for GitHub issue #1806: the CPU Remap kernels
(HafCpu_Remap_*) decode each entry of the fixed-point remap table assuming it
is always 3-fractional-bit and always fits in a *signed* 16-bit value.

Both assumptions break for large source images:

  * remap_fractional_bits is adaptive (ago_util.cpp): it is reduced from the
    base of 3 down to 2/1/0 bits as source width/height crosses
    8192/16384/32768 pixels, so that the packed fixed-point coordinate always
    fits in a uint16. A kernel that hardcodes ">> 3" / "& 7" (or an SIMD
    shift-by-3) silently mis-decodes every coordinate once the source is
    >= 8192px on either axis.

  * The remap table is stored as *unsigned* 16-bit fixed point
    (ago_coord2d_ushort_t); several kernels locally reinterpret the buffer as
    *signed* 16-bit (ago_coord2d_short_t) and right-shift with sign
    extension. Once a coordinate's fixed-point encoding reaches >= 32768
    (e.g. any x >= 4096 at the still-current 3 fractional bits), it reads as
    negative, fails the ">= 0" bounds check, and the pixel incorrectly gets
    the border/clamped value instead of the correct sample.

This test exercises every (format x interpolation) combination at one source
width per fractional-bit tier (3, 2, 1, 0 bits -- i.e. widths 6000, 10000,
20000, 40000), plus one large-height case to cover the y-axis decode path,
and checks that a distinctive pixel placed deep in the "large coordinate"
region of the source is sampled correctly rather than replaced by garbage or
border values. For Bilinear, a half-pixel-offset case additionally checks
that the blend weights use the image's actual fractional-bit scale instead
of a hardcoded 8/32/6.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Formats under test: (label, runvx fourcc, bytes per pixel).
FORMATS = [
    ("U8", "U008", 1),
    ("RGB", "RGB2", 3),
    ("RGBX", "RGBX", 4),
]

# One representative source width per remap_fractional_bits tier, matching
# the thresholds in ago_util.cpp (agoVerifyAndFinalizeDataFromDescription):
#   < 8192       -> 3 bits (base AGO_REMAP_FRACTIONAL_BITS)
#   8192..16383  -> 2 bits
#   16384..32767 -> 1 bit
#   >= 32768     -> 0 bits
# target_col is chosen deep enough into each tier that: (a) it is still well
# below the next threshold, and (b) for the 3-bit tier specifically, its
# fixed-point encoding (col * 8) is >= 32768, which is exactly the case that
# the signed-reinterpretation bug (Bug A in issue #1806) mis-handles.
WIDTH_CASES = [
    (3, 6000, 5000),
    (2, 10000, 9000),
    (1, 20000, 18000),
    (0, 40000, 35000),
]


def repeat_bytes(values, nbytes):
    """Pack nbytes/len(values) copies of `values` into a little-endian int,
    matching the GDF uniform-image / pixel literal convention of lowest
    channel first."""
    out = 0
    for i in range(nbytes):
        out |= (values[i % len(values)] & 0xFF) << (8 * i)
    return out


def fixed_point(coord, frac_bits):
    """Match vxSetRemapPoint: (vx_uint16)(coord * (1 << frac_bits) + 0.5f)."""
    return int(coord * (1 << frac_bits) + 0.5)


def bilinear_ref(sx, sy, src_w, src_h, frac_bits, get_pixel, nch):
    """Per-channel reference mirroring the fixed-point bilinear kernel math,
    generalized to an arbitrary fractional-bit count."""
    scale = 1 << frac_bits
    fxp, fyp = fixed_point(sx, frac_bits), fixed_point(sy, frac_bits)
    frac_mask = scale - 1
    mx, my = fxp >> frac_bits, fyp >> frac_bits
    fx, fy = fxp & frac_mask, fyp & frac_mask
    mx = min(mx, src_w - 2)
    my = min(my, src_h - 2)
    norm_shift = 2 * frac_bits
    norm_round = (1 << (norm_shift - 1)) if norm_shift > 0 else 0
    w00 = (scale - fx) * (scale - fy)
    w10 = fx * (scale - fy)
    w01 = (scale - fx) * fy
    w11 = fx * fy
    out = []
    for c in range(nch):
        v00 = get_pixel(mx, my, c)
        v10 = get_pixel(mx + 1, my, c)
        v01 = get_pixel(mx, my + 1, c)
        v11 = get_pixel(mx + 1, my + 1, c)
        out.append((v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11 + norm_round) >> norm_shift)
    return out


def build_image(src_w, src_h, nch, target_x, target_y, target_val, fill_val=0):
    buf = bytearray([fill_val] * (src_w * src_h * nch))
    idx = (target_y * src_w + target_x) * nch
    for c in range(nch):
        buf[idx + c] = target_val[c % len(target_val)]
    return bytes(buf)


def run_runvx(runvx_exe, gdf_path):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "CPU"
    cmd = [str(runvx_exe), "-frames:1", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=120)


def check_case(work_dir, runvx_exe, tag, fourcc, nch, src_w, src_h,
               src_bytes, sx, sy, expected):
    gdf_path = work_dir / f"{tag}.gdf"
    out_path = work_dir / f"{tag}_out.raw"
    remap_path = work_dir / f"{tag}_remap.txt"
    src_path = work_dir / f"{tag}_src.raw"
    src_path.write_bytes(src_bytes)
    remap_path.write_text(f"{sx} {sy}\n")
    interp = "BILINEAR" if "bilinear" in tag else "NEAREST_NEIGHBOR"
    content = (
        f"data input_1 = image:{src_w},{src_h},{fourcc}:read,{src_path}\n"
        f"data output_1 = image:1,1,{fourcc}:write,{out_path}\n"
        f"data remap_table = remap:{src_w},{src_h},1,1:read,{remap_path}\n"
        f"node org.khronos.openvx.remap input_1 remap_table !{interp} output_1 "
        f"attr:BORDER_MODE:CONSTANT,40\n"
    )
    gdf_path.write_text(content)

    result = run_runvx(runvx_exe, gdf_path)
    if result.returncode != 0:
        return False, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}"

    if not out_path.exists() or out_path.stat().st_size != nch:
        return False, f"unexpected output size (expected {nch} bytes)"

    got = list(out_path.read_bytes())
    if got != list(expected):
        return False, f"got {got}, expected {list(expected)}"
    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Large-image Remap regression test (issue #1806)")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # X-axis coverage: one width per remap_fractional_bits tier.
        for frac_bits, src_w, target_col in WIDTH_CASES:
            src_h = 2
            for label, fourcc, nch in FORMATS:
                target_val = tuple((10 + c * 11) % 256 for c in range(nch))
                src_bytes = build_image(src_w, src_h, nch, target_col, 0, target_val)

                # Nearest-neighbor: exact integer coordinate must round-trip.
                tag = f"nearest_{frac_bits}b_{label}"
                ok, msg = check_case(work_dir, runvx_exe, tag, fourcc, nch,
                                      src_w, src_h, src_bytes,
                                      float(target_col), 0.0, target_val)
                tagname = f"Nearest fracBits={frac_bits} w={src_w} {label}"
                print(f"{'PASS' if ok else 'FAIL'} [{tagname}]" + ("" if ok else f": {msg}"))
                if not ok:
                    failures.append((tagname, msg))

                # Bilinear: exact integer coordinate (pure corner sample).
                tag = f"bilinear_int_{frac_bits}b_{label}"
                ok, msg = check_case(work_dir, runvx_exe, tag, fourcc, nch,
                                      src_w, src_h, src_bytes,
                                      float(target_col), 0.0, target_val)
                tagname = f"Bilinear(int) fracBits={frac_bits} w={src_w} {label}"
                print(f"{'PASS' if ok else 'FAIL'} [{tagname}]" + ("" if ok else f": {msg}"))
                if not ok:
                    failures.append((tagname, msg))

                # Bilinear: half-pixel offset -- checks the weight/normalization
                # scale (scale, normShift, normRound) is derived from the
                # image's actual fractional-bit count, not hardcoded 8/32/6.
                fill_val = tuple((200 + c * 7) % 256 for c in range(nch))
                src_bytes2 = build_image(src_w, src_h, nch, target_col, 0,
                                          target_val, fill_val=0)
                # also populate the neighbor column so the blend is well-defined
                buf = bytearray(src_bytes2)
                for c in range(nch):
                    buf[((0 * src_w + (target_col + 1)) * nch) + c] = fill_val[c % nch]
                src_bytes2 = bytes(buf)

                def get_pixel(x, y, c, _buf=buf, _w=src_w, _nch=nch):
                    return _buf[(y * _w + x) * _nch + c]

                expected = bilinear_ref(target_col + 0.5, 0.0, src_w, src_h,
                                         frac_bits, get_pixel, nch)
                tag = f"bilinear_frac_{frac_bits}b_{label}"
                ok, msg = check_case(work_dir, runvx_exe, tag, fourcc, nch,
                                      src_w, src_h, src_bytes2,
                                      target_col + 0.5, 0.0, expected)
                tagname = f"Bilinear(frac) fracBits={frac_bits} w={src_w} {label}"
                print(f"{'PASS' if ok else 'FAIL'} [{tagname}]" + ("" if ok else f": {msg}"))
                if not ok:
                    failures.append((tagname, msg))

        # Y-axis coverage: the sign-reinterpretation bug (#1806 Bug A) is
        # symmetric between x and y in the shared decode helpers; cover the
        # 3-fractional-bit tier (the one affected while still at the base
        # fractional-bit count) with a tall, narrow image.
        src_w, src_h, target_row = 4, 6000, 5000
        for label, fourcc, nch in FORMATS:
            target_val = tuple((30 + c * 13) % 256 for c in range(nch))
            src_bytes = build_image(src_w, src_h, nch, 0, target_row, target_val)
            tag = f"nearest_y_{label}"
            ok, msg = check_case(work_dir, runvx_exe, tag, fourcc, nch,
                                  src_w, src_h, src_bytes,
                                  0.0, float(target_row), target_val)
            tagname = f"Nearest y-axis h={src_h} {label}"
            print(f"{'PASS' if ok else 'FAIL'} [{tagname}]" + ("" if ok else f": {msg}"))
            if not ok:
                failures.append((tagname, msg))

    if failures:
        print(f"\n{len(failures)} test(s) failed.", file=sys.stderr)
        return 1

    print("\nAll large-image Remap tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
