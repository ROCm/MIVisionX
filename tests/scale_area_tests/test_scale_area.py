#!/usr/bin/env python3
"""
CTest helper: exact-value regression for the CPU VX_INTERPOLATION_AREA paths.

AREA is a box average, so for a carefully chosen input the correct output is
known exactly and can be asserted per pixel. That matters here because the
Khronos CTS cannot catch either defect this guards: its AREA check accepts any
value between the minimum and maximum of a 5x5 source neighbourhood, which a
one-row shift sits comfortably inside, and the shipped AREA GDFs use a uniform
image, where a shift is invisible by construction.

Two scenarios, both on the CPU backend:

1. "ramp-2to1": a vertical ramp where pixel(x, y) = 10 * y, downscaled 2:1. Each
   output pixel averages a 2x2 block spanning source rows 2y and 2y+1, giving
   (2*20y + 2*(20y+10) + 2) >> 2 == 20*y + 5. A fast path that truncates the
   -0.5 scale-matrix offset instead of rounding it reads rows 2y-1 and 2y for
   every row but the first and returns 20*y - 5, so the shift is unambiguous
   rather than a similarity score.

   Only HafCpu_ScaleImage_U8_U8_Area is reachable from a graph. Its near-copy
   HafCpu_ScaleImage_U8_U8_Area_Constant carries the same coordinate mapping and
   is fixed alongside it, but it is dead code today: it has no declaration in
   ago_haf_cpu.h, no kernel id of its own, and ago_drama_divide.cpp routes
   UNDEFINED, REPLICATE and CONSTANT borders alike to the plain AREA kernel. A
   VX_BORDER_MODE_CONSTANT run would therefore re-test this same path rather
   than that one, so it is not attempted here.

2. "blocks-3to1": an input in which every 3x3 block holds a single value --
   pixel(x, y) = y // 3 -- downscaled 3:1, so output pixel (x, y) must equal
   exactly y. Any wrong pixel is therefore identifiable by position, which is
   what separates the two defects in the general path: a missing scalar tail in
   the vertical-sum loop corrupts the last column, and a bottom-row clamp
   computed from the width rather than the row stride corrupts the last rows.
   Three widths isolate them independently -- a source width must be a multiple
   of 8 to avoid the first and of 16 to avoid the second:

       1296  multiple of both   -> clean even before the fix
       1320  multiple of 8 only -> bottom rows wrong
       1281  neither            -> last column and bottom rows wrong

Only the CPU backend is exercised. These two defects are CPU-only, and the GPU
AREA path has separate defects of its own tracked elsewhere, so running it here
would report failures that belong to a different fix.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def write_rows(path, rows, width):
    """Write an image as a list of per-row byte values; no numpy."""
    with open(path, "wb") as f:
        for value in rows:
            f.write(bytes([value]) * width)


def run_runvx(runvx_exe, gdf_path):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "CPU"
    cmd = [str(runvx_exe), "-frames:1", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=300)


def build_gdf(path, src, src_w, src_h, dst, dst_w, dst_h):
    path.write_text(
        f"data in  = image:{src_w},{src_h},U008:read,{src}\n"
        f"data out = image:{dst_w},{dst_h},U008:write,{dst}\n"
        "node org.khronos.openvx.scale_image in out !AREA\n")


def check_ramp_2to1(work_dir, runvx_exe):
    """2:1 downscale of a vertical ramp; out[y] must be 20*y + 5 on every column."""
    src_w, src_h = 48, 24
    dst_w, dst_h = src_w // 2, src_h // 2
    src = work_dir / "ramp.u8"
    write_rows(src, [10 * y for y in range(src_h)], src_w)

    dst = work_dir / "ramp_out.u8"
    gdf = work_dir / "ramp.gdf"
    build_gdf(gdf, src, src_w, src_h, dst, dst_w, dst_h)

    result = run_runvx(runvx_exe, gdf)
    tag = "ramp-2to1"
    if result.returncode != 0:
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]
    if not dst.exists():
        return [(tag, f"output not created: {dst}")]
    data = dst.read_bytes()
    if len(data) != dst_w * dst_h:
        return [(tag, f"size mismatch: expected {dst_w * dst_h}, got {len(data)}")]

    for y in range(dst_h):
        expected = 20 * y + 5
        row = data[y * dst_w:(y + 1) * dst_w]
        for x, got in enumerate(row):
            if got != expected:
                return [(tag, f"row {y} column {x}: got {got}, expected {expected} "
                              f"(a value of {expected - 20} means source rows "
                              f"{2 * y - 1} and {2 * y} were averaged instead of "
                              f"{2 * y} and {2 * y + 1})")]
    print(f"PASS [{tag}]: {dst_h} rows exact, out[y] == 20*y + 5")
    return []


def check_blocks_3to1(work_dir, runvx_exe, src_w):
    """3:1 downscale where every 3x3 block holds one value; out(x, y) must equal y."""
    src_h = 720
    dst_w, dst_h = src_w // 3, src_h // 3
    src = work_dir / f"blocks_{src_w}.u8"
    write_rows(src, [y // 3 for y in range(src_h)], src_w)

    dst = work_dir / f"blocks_out_{src_w}.u8"
    gdf = work_dir / f"blocks_{src_w}.gdf"
    build_gdf(gdf, src, src_w, src_h, dst, dst_w, dst_h)

    result = run_runvx(runvx_exe, gdf)
    tag = f"blocks-3to1 {src_w}x{src_h}"
    if result.returncode != 0:
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]
    if not dst.exists():
        return [(tag, f"output not created: {dst}")]
    data = dst.read_bytes()
    if len(data) != dst_w * dst_h:
        return [(tag, f"size mismatch: expected {dst_w * dst_h}, got {len(data)}")]

    wrong = 0
    last_column = 0
    rows_hit = set()
    first = None
    for y in range(dst_h):
        row = data[y * dst_w:(y + 1) * dst_w]
        for x, got in enumerate(row):
            if got != y:
                wrong += 1
                rows_hit.add(y)
                if x == dst_w - 1:
                    last_column += 1
                if first is None:
                    first = (x, y, got)
    if wrong:
        x, y, got = first
        return [(tag, f"{wrong} of {dst_w * dst_h} outputs wrong; last column wrong "
                      f"in {last_column} of {dst_h} rows; {len(rows_hit)} rows "
                      f"affected; first at (x={x}, y={y}) got {got}, expected {y}")]
    print(f"PASS [{tag}]: {dst_w * dst_h} outputs exact, out(x, y) == y")
    return []


def main():
    parser = argparse.ArgumentParser(description="CPU AREA scale correctness test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        failures += check_ramp_2to1(work_dir, runvx_exe)
        for src_w in (1296, 1320, 1281):
            failures += check_blocks_3to1(work_dir, runvx_exe, src_w)

    if failures:
        print(f"\n{len(failures)} test(s) failed:", file=sys.stderr)
        for tag, msg in failures:
            print(f"  [{tag}] {msg}", file=sys.stderr)
        return 1

    print("\nAll AREA scale tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
