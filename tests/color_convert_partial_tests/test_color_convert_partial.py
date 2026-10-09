#!/usr/bin/env python3
"""
CTest helper: HIP RGB/RGBX stores must stay inside the destination row.

The colour kernels convert 8 pixels per thread and used to store all of them
with one wide write - 24 bytes for RGB, 32 for RGBX - guarded only by
`x >= dstWidth`. That stops a thread entirely past the edge but not one partly
past it. Rows are padded only to a multiple of 16 bytes, so an RGB row of width
1282 is 3846 bytes inside a 3856-byte stride and the last thread writes bytes
3840..3863 - eight of them belonging to the next row:

    row y:  [ 1282 pixels = 3846 bytes ][ 10 pad ]   stride 3856
            last thread writes 3840 .. 3863  ->  3856..3863 is row y+1

Row y's last thread and row y+1's first thread then write the same bytes and
whichever lands last wins, so the corruption is non-deterministic and moves
between runs. Each case therefore runs several times rather than once; a single
pass can come out clean on a kernel that is still wrong.

No existing coverage can see this. The colour GDFs and vision tests use 480,
1280 and 1920, all multiples of 8, so the partial group never exists.

Widths are even throughout: ovxKernel_ColorConvert rejects an odd width outright
(ago_kernel_api.cpp:357), so 1281-style widths cannot reach these kernels.

Two kinds of call site, which differ in how the valid-pixel count is derived and
so need separate cover:

1. "direct": RGBX -> RGB and RGB -> RGBX, where the kernel's x is a pixel index.
   Both are pure per-pixel repacks, so the expected bytes are computed here
   directly and every pixel is asserted.

2. "grouped": UYVY -> RGB, one of the ten paired YUV kernels where x indexes a
   group of 8 against a compressed width, so the first pixel is x * 8 and an
   off-by-one in the count is a different mistake from case 1. Rather than
   reimplement the YUV coefficients, the same source is converted at a width
   that is a multiple of 8 and at one that is not, and the overlapping columns
   must agree: each output pixel depends only on its own 2-pixel YUV group, so
   width cannot legitimately change any of them.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUNS = 3
HEIGHT = 64


def run_runvx(runvx_exe, gdf_path):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "GPU"
    cmd = [str(runvx_exe), "-frames:1", "-affinity:GPU", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=300)


def pixel(x, y):
    """A value per channel that varies along both axes."""
    return ((x * 5 + y * 3) % 251,
            (x * 11 + y * 7) % 251,
            (x * 3 + y * 13) % 251)


def write_rgb(path, width, height, bpp):
    with open(path, "wb") as f:
        for y in range(height):
            row = bytearray()
            for x in range(width):
                r, g, b = pixel(x, y)
                row += bytes([r, g, b, 255][:bpp])
            f.write(bytes(row))


def write_uyvy(path, width, height):
    with open(path, "wb") as f:
        for y in range(height):
            row = bytearray()
            for x in range(0, width, 2):
                r, g, b = pixel(x, y)
                row += bytes([(r + 40) % 251, (g + 16) % 251,
                              (b + 70) % 251, (g + 90) % 251])
            f.write(bytes(row))


def build_gdf(path, src, src_fmt, dst, dst_fmt, width, height):
    path.write_text(
        f"data in  = image:{width},{height},{src_fmt}:read,{src}\n"
        f"data out = image:{width},{height},{dst_fmt}:write,{dst}\n"
        "node org.khronos.openvx.color_convert in out\n")


def check_direct(work_dir, runvx_exe, width, src_fmt, src_bpp, dst_fmt, dst_bpp, label):
    """RGBX <-> RGB: a per-pixel repack, so every output byte is known here."""
    tag = f"direct {label} {width}x{HEIGHT}"
    src = work_dir / f"{label}_{width}.src"
    write_rgb(src, width, HEIGHT, src_bpp)

    failures = []
    for run in range(1, RUNS + 1):
        dst = work_dir / f"{label}_{width}_run{run}.dst"
        gdf = work_dir / f"{label}_{width}_run{run}.gdf"
        build_gdf(gdf, src, src_fmt, dst, dst_fmt, width, HEIGHT)
        result = run_runvx(runvx_exe, gdf)
        if result.returncode != 0:
            failures.append((tag, f"run {run}: runvx exited {result.returncode}:"
                                  f"\n{result.stdout[-800:]}"))
            continue
        data = dst.read_bytes()
        if len(data) != width * HEIGHT * dst_bpp:
            failures.append((tag, f"run {run}: size mismatch: expected "
                                  f"{width * HEIGHT * dst_bpp}, got {len(data)}"))
            continue

        wrong = 0
        first = None
        head_of_row = 0
        for y in range(HEIGHT):
            for x in range(width):
                r, g, b = pixel(x, y)
                want = bytes([r, g, b, 255][:dst_bpp])
                off = (y * width + x) * dst_bpp
                got = data[off:off + dst_bpp]
                if got != want:
                    wrong += 1
                    if x < 8:
                        head_of_row += 1
                    if first is None:
                        first = (x, y, got.hex(), want.hex())
        if wrong:
            x, y, got, want = first
            detail = (f"run {run}: {wrong} of {width * HEIGHT} pixels wrong; "
                      f"first at (x={x}, y={y}) got {got}, expected {want}")
            if head_of_row:
                detail += (f"; {head_of_row} of them in the first 8 columns of a "
                           f"row, which is where the previous row's last thread "
                           f"overruns")
            failures.append((tag, detail))
    if not failures:
        print(f"PASS [{tag}]: {width * HEIGHT} pixels exact in {RUNS} runs")
    return failures


def check_grouped(work_dir, runvx_exe, width, clean_width):
    """UYVY -> RGB at an unaligned width must match the same source at an aligned one."""
    tag = f"grouped UYVY->RGB {width} vs {clean_width}"
    ref_src = work_dir / f"uyvy_{clean_width}.src"
    write_uyvy(ref_src, clean_width, HEIGHT)

    # Same pixel values, cropped: pixel() does not depend on the width, so the
    # first `width` columns hold identical YUV groups in both sources.
    src = work_dir / f"uyvy_{width}.src"
    write_uyvy(src, width, HEIGHT)

    ref_dst = work_dir / f"uyvy_{clean_width}.rgb"
    ref_gdf = work_dir / f"uyvy_{clean_width}.gdf"
    build_gdf(ref_gdf, ref_src, "UYVY", ref_dst, "RGB2", clean_width, HEIGHT)
    result = run_runvx(runvx_exe, ref_gdf)
    if result.returncode != 0:
        return [(tag, f"reference run exited {result.returncode}:"
                      f"\n{result.stdout[-800:]}")]
    ref = ref_dst.read_bytes()

    failures = []
    for run in range(1, RUNS + 1):
        dst = work_dir / f"uyvy_{width}_run{run}.rgb"
        gdf = work_dir / f"uyvy_{width}_run{run}.gdf"
        build_gdf(gdf, src, "UYVY", dst, "RGB2", width, HEIGHT)
        result = run_runvx(runvx_exe, gdf)
        if result.returncode != 0:
            failures.append((tag, f"run {run}: runvx exited {result.returncode}:"
                                  f"\n{result.stdout[-800:]}"))
            continue
        data = dst.read_bytes()
        if len(data) != width * HEIGHT * 3:
            failures.append((tag, f"run {run}: size mismatch: expected "
                                  f"{width * HEIGHT * 3}, got {len(data)}"))
            continue

        wrong = 0
        first = None
        head_of_row = 0
        for y in range(HEIGHT):
            for x in range(width):
                got = data[(y * width + x) * 3:(y * width + x) * 3 + 3]
                want = ref[(y * clean_width + x) * 3:(y * clean_width + x) * 3 + 3]
                if got != want:
                    wrong += 1
                    if x < 8:
                        head_of_row += 1
                    if first is None:
                        first = (x, y, got.hex(), want.hex())
        if wrong:
            x, y, got, want = first
            detail = (f"run {run}: {wrong} of {width * HEIGHT} pixels differ from "
                      f"the {clean_width}-wide conversion of the same input; first "
                      f"at (x={x}, y={y}) got {got}, expected {want}")
            if head_of_row:
                detail += (f"; {head_of_row} of them in the first 8 columns of a row")
            failures.append((tag, detail))
    if not failures:
        print(f"PASS [{tag}]: {width * HEIGHT} pixels agree in {RUNS} runs")
    return failures


def main():
    parser = argparse.ArgumentParser(description="HIP colour-convert partial store test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        # Three kinds of width, because a partial group is not the same thing as
        # an overrun. With a group of 8 pixels ending at ceil(w/8)*8 and a row
        # padded to ALIGN16(w * bpp):
        #
        #   w     RGB row/stride/end      RGBX row/stride/end     overruns
        #   1282  3846 / 3856 / 3864      5128 / 5136 / 5152      both
        #   1284  3852 / 3856 / 3864      5136 / 5136 / 5152      both
        #   1286  3858 / 3872 / 3864      5144 / 5152 / 5152      neither
        #   1280  3840 / 3840 / 3840      5120 / 5120 / 5120      no partial
        #
        # 1282 and 1284 are the live cases. 1286 still takes the partial-store
        # path with 6 valid pixels but stays inside the padding, so it checks
        # that the bounded store writes the right pixels rather than that it
        # stops in time - a miscounted `valid` shows up there. 1280 has no
        # partial group at all, so a failure there would mean the defect is
        # something other than the partial store.
        for width in (1282, 1284, 1286, 1280):
            failures += check_direct(work_dir, runvx_exe, width,
                                     "RGBX", 4, "RGB2", 3, "RGBX-RGB")
            failures += check_direct(work_dir, runvx_exe, width,
                                     "RGB2", 3, "RGBX", 4, "RGB-RGBX")
        failures += check_grouped(work_dir, runvx_exe, 1282, 1288)

    if failures:
        print(f"\n{len(failures)} test(s) failed:", file=sys.stderr)
        for tag, msg in failures:
            print(f"  [{tag}] {msg}", file=sys.stderr)
        return 1

    print("\nAll colour-convert partial store tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
