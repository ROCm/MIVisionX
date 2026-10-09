#!/usr/bin/env python3
"""
CTest helper: exact-value regression for the HIP VX_INTERPOLATION_AREA paths.

HipExec_ScaleImage_U8_U8_Area picks a kernel per ratio, and before #1776 that
choice was wrong for every integer ratio except 2:1 and 4:1: 3:1, 5:1 and 6:1
ran a kernel hard-coded for a 2x2 block and 8:1 one hard-coded for 4x4, each
scaled by the true 1/(Sx*Sy). A constant-90 image returned 40, 14, 10 and 22.

Nothing in the repository could notice. The five shipped AREA GDFs assert only
that runvx exits 0, and they use a uniform image besides, so no comparison is
made at all. That is how four wrong ratios survived.

Every case here is exact rather than approximate. AREA is a box average, so an
input whose Nx x Ny source blocks each hold a single value has a known integer
answer, independent of how the kernel rounds:

1. "blocks": pixel(x, y) = v(x / Nx, y / Ny) for a per-block value v, downscaled
   by the exact ratio, so output pixel (bx, by) must equal v(bx, by) exactly.
   Covers 2:1, 3:1, 4:1, 5:1, 6:1 and 8:1 - the four ratios #1776 broke plus the
   two that already worked - and the non-square blocks 4x2 and 8x5, which no
   ratio in the issue covered and which the old `need_align` test could not
   express.

2. "tail": the same check at a destination width that is not a multiple of 8.
   Each thread writes 8 pixels, so the last group of a row is partial; that
   store is bounded rather than unconditional, and a regression would clobber
   the row padding or, for a destination whose stride is not ALIGN16, the next
   row.

3. "tight": the same check into a destination imported with
   vxCreateImageFromHandle at a tight stride. vxCreateImageFromHandle stores the
   caller's stride_y verbatim (vx_api.cpp:1048) rather than padding it to
   ALIGN16, so for a width that is not a multiple of 8 an unbounded 8-pixel
   store crosses into the next row rather than landing in padding. This is what
   distinguishes a bounded store from one that merely stays inside the padding,
   and it covers all three kernels the selector can dispatch to: 2:1 to
   Hip_ScaleImage_U8_U8_Area, 3:1 to _Area_Int, 4:1 to _Area_Sad.

4. "accumulate": a constant-255 4105x4105 image scaled to 1x1. The block sums to
   4,297,011,375, which overflows a 32-bit accumulator and returns 0 instead of
   255. 4105 is the first square side that does so.

GPU only. The CPU AREA paths have separate defects, covered by
tests/scale_area_tests.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def block_value(bx, by):
    """Per-block value, varying in both axes and never 0 or 255."""
    return (bx * 7 + by * 13) % 251


def write_blocks(path, dst_w, dst_h, nx, ny):
    """Write a source whose every nx x ny block holds one value; no numpy."""
    with open(path, "wb") as f:
        for by in range(dst_h):
            row = bytes(block_value(bx, by) for bx in range(dst_w))
            expanded = bytes(b for b in row for _ in range(nx))
            for _ in range(ny):
                f.write(expanded)


def run_runvx(runvx_exe, gdf_path):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "GPU"
    cmd = [str(runvx_exe), "-frames:1", "-affinity:GPU", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=300)


def build_gdf(path, src, src_w, src_h, dst, dst_w, dst_h):
    path.write_text(
        f"data in  = image:{src_w},{src_h},U008:read,{src}\n"
        f"data out = image:{dst_w},{dst_h},U008:write,{dst}\n"
        "node org.khronos.openvx.scale_image in out !AREA\n")


def check_blocks(work_dir, runvx_exe, dst_w, dst_h, nx, ny, tag):
    """Downscale by exactly nx x ny; output (bx, by) must equal block_value."""
    src_w, src_h = dst_w * nx, dst_h * ny
    src = work_dir / f"blocks_{nx}x{ny}_{dst_w}x{dst_h}.u8"
    dst = work_dir / f"out_{nx}x{ny}_{dst_w}x{dst_h}.u8"
    gdf = work_dir / f"blocks_{nx}x{ny}_{dst_w}x{dst_h}.gdf"
    write_blocks(src, dst_w, dst_h, nx, ny)
    build_gdf(gdf, src, src_w, src_h, dst, dst_w, dst_h)

    result = run_runvx(runvx_exe, gdf)
    if result.returncode != 0:
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]
    if not dst.exists():
        return [(tag, f"output not created: {dst}")]
    data = dst.read_bytes()
    if len(data) != dst_w * dst_h:
        return [(tag, f"size mismatch: expected {dst_w * dst_h}, got {len(data)}")]

    wrong = 0
    first = None
    last_group = 0
    first_partial = (dst_w // 8) * 8
    for by in range(dst_h):
        row = data[by * dst_w:(by + 1) * dst_w]
        for bx, got in enumerate(row):
            want = block_value(bx, by)
            if got != want:
                wrong += 1
                if bx >= first_partial:
                    last_group += 1
                if first is None:
                    first = (bx, by, got, want)
    if wrong:
        bx, by, got, want = first
        detail = (f"{wrong} of {dst_w * dst_h} outputs wrong; first at "
                  f"(x={bx}, y={by}) got {got}, expected {want}")
        if last_group:
            detail += (f"; {last_group} of them in the partial last group "
                       f"(x >= {first_partial}), which points at the bounded store "
                       f"rather than the block arithmetic")
        return [(tag, detail)]
    print(f"PASS [{tag}]: {dst_w * dst_h} outputs exact for a {nx}x{ny} block")
    return []


def check_tight_stride(work_dir, runvx_exe, dst_w, dst_h, nx, ny):
    """Same block check, but into a handle whose stride is tight rather than ALIGN16."""
    tag = f"tight stride {nx}:1 {dst_w} wide"
    src_w, src_h = dst_w * nx, dst_h * ny
    src = work_dir / f"tight_src_{nx}_{dst_w}.u8"
    dst = work_dir / f"tight_out_{nx}_{dst_w}.u8"
    gdf = work_dir / f"tight_{nx}_{dst_w}.gdf"
    write_blocks(src, dst_w, dst_h, nx, ny)
    gdf.write_text(
        f"data in  = image:{src_w},{src_h},U008:read,{src}\n"
        f"data out = image-from-handle:U008,{{{dst_w};{dst_h};1;{dst_w}}},"
        f"VX_MEMORY_TYPE_HOST:write,{dst}\n"
        "node org.khronos.openvx.scale_image in out !AREA\n")

    result = run_runvx(runvx_exe, gdf)
    if result.returncode != 0:
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]
    data = dst.read_bytes()
    if len(data) != dst_w * dst_h:
        return [(tag, f"size mismatch: expected {dst_w * dst_h}, got {len(data)}")]

    wrong = 0
    first = None
    head_of_row = 0
    for by in range(dst_h):
        for bx in range(dst_w):
            want = block_value(bx, by)
            got = data[by * dst_w + bx]
            if got != want:
                wrong += 1
                if bx < 8:
                    head_of_row += 1
                if first is None:
                    first = (bx, by, got, want)
    if wrong:
        bx, by, got, want = first
        detail = (f"{wrong} of {dst_w * dst_h} outputs wrong; first at "
                  f"(x={bx}, y={by}) got {got}, expected {want}")
        if head_of_row:
            detail += (f"; {head_of_row} of them in the first 8 columns of a row, "
                       f"which is where the previous row's last thread overruns a "
                       f"stride of {dst_w}")
        return [(tag, detail)]
    print(f"PASS [{tag}]: {dst_w * dst_h} outputs exact at a {dst_w}-byte stride")
    return []


def check_accumulate(work_dir, runvx_exe):
    """4105x4105 of 255 to 1x1: sums to 4,297,011,375, past a 32-bit accumulator."""
    side = 4105
    tag = f"accumulate {side}x{side}->1x1"
    src = work_dir / "const255.u8"
    dst = work_dir / "const255_out.u8"
    gdf = work_dir / "const255.gdf"
    with open(src, "wb") as f:
        row = bytes([255]) * side
        for _ in range(side):
            f.write(row)
    build_gdf(gdf, src, side, side, dst, 1, 1)

    result = run_runvx(runvx_exe, gdf)
    if result.returncode != 0:
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]
    data = dst.read_bytes()
    if len(data) != 1:
        return [(tag, f"size mismatch: expected 1 byte, got {len(data)}")]
    if data[0] != 255:
        return [(tag, f"got {data[0]}, expected 255 "
                      f"(a 32-bit accumulator wraps {255 * side * side} to "
                      f"{(255 * side * side) % (1 << 32)} and returns 0)")]
    print(f"PASS [{tag}]: 255, so the accumulator held {255 * side * side}")
    return []


def main():
    parser = argparse.ArgumentParser(description="HIP AREA scale correctness test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # Square ratios: 2 and 4 had dedicated kernels already, the rest are the
        # ones #1776 sent to a kernel of the wrong block shape.
        for n in (2, 3, 4, 5, 6, 8):
            failures += check_blocks(work_dir, runvx_exe, 160, 120, n, n,
                                     f"blocks {n}:1")

        # Non-square blocks, which no ratio in the issue covered.
        failures += check_blocks(work_dir, runvx_exe, 160, 120, 4, 2, "blocks 4x2")
        failures += check_blocks(work_dir, runvx_exe, 160, 120, 8, 5, "blocks 8x5")

        # Destination widths that are not multiples of 8, so the last group of
        # every row is a partial store. 101 is also not a multiple of 4.
        for dst_w in (101, 150):
            failures += check_blocks(work_dir, runvx_exe, dst_w, 120, 3, 3,
                                     f"tail {dst_w} wide")

        # A tight imported stride, where a partial group genuinely crosses into
        # the next row instead of landing in ALIGN16 padding. One ratio per
        # kernel the selector can reach: 2:1 -> _Area, 3:1 -> _Area_Int,
        # 4:1 -> _Area_Sad.
        for nx in (2, 3, 4):
            failures += check_tight_stride(work_dir, runvx_exe, 101, 40, nx, nx)

        failures += check_accumulate(work_dir, runvx_exe)

    if failures:
        print(f"\n{len(failures)} test(s) failed:", file=sys.stderr)
        for tag, msg in failures:
            print(f"  [{tag}] {msg}", file=sys.stderr)
        return 1

    print("\nAll HIP AREA scale tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
