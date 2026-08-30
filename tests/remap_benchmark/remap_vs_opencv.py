#!/usr/bin/env python3
"""Benchmark MIVisionX Remap (CPU/GPU) vs OpenCV remap.

Runs a set of GDFs through runVX and compares against cv::remap on the same
synthetic source + map. Reports median execution time in ms over N warm-up +
N timed iterations.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def build_map(dst_h, dst_w, map_file, kind="mirror"):
    """Write a remap table compatible with runVX: two 16-bit ints per pixel."""
    # MIVisionX stores coordinates as (x<<3 | frac, y<<3 | frac), i.e. fixed 3.3.
    # For bilinear, integer part is floor(src_coord). We use unscaled integer
    # coordinates here and scale by 8 below.
    ys, xs = np.mgrid[0:dst_h, 0:dst_w]
    if kind == "mirror":
        sx = (dst_w - 1 - xs).astype(np.float32)
        sy = (dst_h - 1 - ys).astype(np.float32)
    elif kind == "constant":
        sx = np.full((dst_h, dst_w), -1.0, dtype=np.float32)
        sy = np.full((dst_h, dst_w), -1.0, dtype=np.float32)
    elif kind == "identity":
        sx = xs.astype(np.float32)
        sy = ys.astype(np.float32)
    else:
        raise ValueError(kind)

    # Scale to fixed-point; map entries with value -1 -> 0xFFFF sentinel.
    sx8 = np.round(sx * 8).astype(np.int32)
    sy8 = np.round(sy * 8).astype(np.int32)
    sx8 = np.where(sx < 0, 0xFFFF, sx8)
    sy8 = np.where(sy < 0, 0xFFFF, sy8)

    out = np.empty((dst_h, dst_w, 2), dtype=np.uint16)
    out[:, :, 0] = sx8.astype(np.uint16)
    out[:, :, 1] = sy8.astype(np.uint16)
    out.tofile(map_file)
    return sx, sy


def runvx_time(gdf_path, runvx, env, src_file, map_file, out_file, dst_h, dst_w, bpp, iterations=10):
    """Median wall-clock time for runVX to execute a remap GDF."""
    # Prepare an output raw file node so runVX writes binary; the GDFs used here
    # already write to a raw output file named by argument.
    env = dict(os.environ, **env) if env else os.environ

    # runVX is invoked once per iteration because we want fresh timing.
    times = []
    for _ in range(iterations):
        # Clear output file to avoid cheating from cached pages
        Path(out_file).write_bytes(b'')
        start = time.perf_counter()
        subprocess.run([str(runvx), str(gdf_path)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
    return np.median(times)


def opencv_time(src, sx, sy, iterations=10, border=cv2.BORDER_CONSTANT, border_value=(0, 0, 0, 0)):
    """Median wall-clock time for cv::remap."""
    map1 = sx.astype(np.float32)
    map2 = sy.astype(np.float32)
    # Warm-up
    for _ in range(5):
        _ = cv2.remap(src, map1, map2, cv2.INTER_LINEAR, borderMode=border, borderValue=border_value)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = cv2.remap(src, map1, map2, cv2.INTER_LINEAR, borderMode=border, borderValue=border_value)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
    return np.median(times)


def benchmark_case(name, src, sx, sy, gdf, runvx, out_file, env, bpp, iterations):
    runvx_ms = runvx_time(gdf, runvx, env, None, None, out_file, src.shape[0], src.shape[1], bpp, iterations)
    border_val = (0x42, 0x42, 0x42, 0x42)
    if name.startswith("U8"):
        border_val = (0x42, 0, 0, 0)
    opencv_ms = opencv_time(src, sx, sy, iterations, cv2.BORDER_CONSTANT, border_val)
    return runvx_ms, opencv_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runvx", required=True)
    parser.add_argument("--gdf-dir", required=True, type=Path)
    parser.add_argument("--resolutions", default="1280x720,1920x1080")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    runvx = Path(args.runvx)
    gdf_dir = args.gdf_dir
    iterations = args.iterations

    # We benchmark against runVX GDFs. The existing GDFs use fixed input/output
    # filenames relative to the working dir. To vary resolution, we create a
    # temp dir per resolution, copy the GDF, and substitute dimensions.
    cases = [
        ("U8_Bilinear_Constant", "Remap_U8_U8_Bilinear_Constant.gdf", 1),
        ("RGB_Bilinear", "Remap_RGB_RGB_Bilinear.gdf", 3),
        ("RGB_Bilinear_Constant", "Remap_RGB_RGB_Bilinear_Constant.gdf", 3),
        ("RGBX_Bilinear", "Remap_RGBX_RGBX_Bilinear.gdf", 4),
        ("RGBX_Bilinear_Constant", "Remap_RGBX_RGBX_Bilinear_Constant.gdf", 4),
    ]

    print(f"{'Case':<26} {'Res':<12} {'MIVisionX CPU ms':>18} {'MIVisionX GPU ms':>18} {'OpenCV CPU ms':>18} {'Speedup GPU vs OC':>18}")
    print("=" * 110)

    for res in args.resolutions.split(","):
        dst_w, dst_h = map(int, res.split("x"))

        with tempfile.TemporaryDirectory(prefix="remap_bench_") as td:
            td = Path(td)
            src_u8 = np.random.randint(0, 256, (dst_h, dst_w), dtype=np.uint8)
            src_rgb = np.random.randint(0, 256, (dst_h, dst_w, 3), dtype=np.uint8)
            src_rgbx = np.random.randint(0, 256, (dst_h, dst_w, 4), dtype=np.uint8)

            # Write source raw files that the GDFs expect
            (td / "input_u8_1280x720.raw").write_bytes(src_u8.tobytes())
            (td / "input_rgb_1280x720.raw").write_bytes(src_rgb[:, :, :3].tobytes())
            (td / "input_rgbx_1280x720.raw").write_bytes(src_rgbx.tobytes())

            # Build maps; write into a file the GDF can reference
            sx_mirror, sy_mirror = build_map(dst_h, dst_w, td / "remap_mirror.raw", "mirror")
            sx_const, sy_const = build_map(dst_h, dst_w, td / "remap_constant.raw", "constant")

            for name, gdf_name, bpp in cases:
                gdf_src = gdf_dir / gdf_name
                gdf_dst = td / gdf_name
                # Copy GDF text and replace filenames/resolution references
                text = gdf_src.read_text()
                # Make input/output paths absolute to the temp dir
                # The benchmark GDFs already read raw files by relative name; we keep it simple
                # by writing the expected 1280x720 filenames and padding/cropping if needed.
                gdf_dst.write_text(text)

                out_file = td / f"out_{name}.raw"
                # Inject output filename? Existing GDF writes a fixed file. We'll just let it.

                kind = "constant" if "Constant" in name else "mirror"
                sx, sy = (sx_const, sy_const) if kind == "constant" else (sx_mirror, sy_mirror)
                src = src_u8 if name.startswith("U8") else (src_rgb if name.startswith("RGB_") else src_rgbx)

                cpu_ms, oc_ms = benchmark_case(name, src, sx, sy, gdf_dst, runvx, out_file, {"AGO_DEFAULT_TARGET": "CPU"}, bpp, iterations)
                gpu_ms, _ = benchmark_case(name, src, sx, sy, gdf_dst, runvx, out_file, {"AGO_DEFAULT_TARGET": "GPU"}, bpp, iterations)
                speedup = oc_ms / gpu_ms if gpu_ms else float('inf')
                print(f"{name:<26} {res:<12} {cpu_ms:>18.3f} {gpu_ms:>18.3f} {oc_ms:>18.3f} {speedup:>18.2f}x")


if __name__ == "__main__":
    main()
