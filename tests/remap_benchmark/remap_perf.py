#!/usr/bin/env python3
"""Compare MIVisionX Remap GDF execution vs OpenCV remap.

Uses the existing geometric GDFs at 1280x720. runVX is timed with a large
frame count; OpenCV is timed with an equivalent loop. Output is ms/frame
(median of several runs).
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


RES = (1280, 720)


def parse_remap_txt(path):
    """Parse the runVX remap_*.txt file: each line is 'sx sy'."""
    coords = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x_str, y_str = line.split()
            coords.append((int(x_str), int(y_str)))
    arr = np.array(coords, dtype=np.int32)
    h, w = RES[1], RES[0]
    sx = arr[:, 0].reshape(h, w).astype(np.float32)
    sy = arr[:, 1].reshape(h, w).astype(np.float32)
    return sx, sy


def runvx_frame_ms(gdf, runvx, env, frames):
    env = dict(os.environ, **env)
    # Warm-up
    subprocess.run([str(runvx), f'-frames:{max(1, frames // 10)}', str(gdf)], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([str(runvx), f'-frames:{frames}', str(gdf)], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed / frames)
    return np.median(times)


def opencv_frame_ms(src, sx, sy, border, border_val, iterations):
    # Warm-up
    for _ in range(max(1, iterations // 10)):
        cv2.remap(src, sx, sy, cv2.INTER_LINEAR, borderMode=border, borderValue=border_val)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            cv2.remap(src, sx, sy, cv2.INTER_LINEAR, borderMode=border, borderValue=border_val)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed / iterations)
    return np.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runvx', required=True, type=Path)
    parser.add_argument('--gdf-dir', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=100)
    args = parser.parse_args()

    w, h = RES
    cases = [
        ('U8_Bilinear', 'Remap_U8_U8_Bilinear.gdf', cv2.BORDER_REPLICATE, (0, 0, 0, 0)),
        ('U8_Bilinear_Constant', 'Remap_U8_U8_Bilinear_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
        ('U8_Nearest_Constant', 'Remap_U8_U8_Nearest_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
        ('RGB_Bilinear', 'Remap_RGB_RGB_Bilinear.gdf', cv2.BORDER_REPLICATE, (0xCC, 0xBB, 0xAA, 0)),
        ('RGB_Bilinear_Constant', 'Remap_RGB_RGB_Bilinear_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
        ('RGB_Nearest_Constant', 'Remap_RGB_RGB_Nearest_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
        ('RGBX_Bilinear', 'Remap_RGBX_RGBX_Bilinear.gdf', cv2.BORDER_REPLICATE, (0xCC, 0xBB, 0xAA, 0xAA)),
        ('RGBX_Bilinear_Constant', 'Remap_RGBX_RGBX_Bilinear_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
        ('RGBX_Nearest_Constant', 'Remap_RGBX_RGBX_Nearest_Constant.gdf', cv2.BORDER_CONSTANT, (0, 0, 0, 0)),
    ]

    sx, sy = parse_remap_txt(args.gdf_dir / 'inputs' / 'remap_1280x720_mirror.txt')

    # Sources matching the GDF uniform-image values
    u8_src = np.full((h, w), 0xAA, dtype=np.uint8)
    rgb_src = np.full((h, w, 3), (0xCC, 0xBB, 0xAA), dtype=np.uint8)
    rgbx_src = np.full((h, w, 4), (0xCC, 0xBB, 0xAA, 0xAA), dtype=np.uint8)

    print(f"{'Case':<26} {'MIVX CPU ms/frame':>18} {'MIVX GPU ms/frame':>18} {'OpenCV ms/frame':>18} {'GPU vs OpenCV':>15}")
    print("=" * 100)

    for name, gdf_name, border, border_val in cases:
        gdf = args.gdf_dir / gdf_name
        cpu_ms = runvx_frame_ms(gdf, args.runvx, {'AGO_DEFAULT_TARGET': 'CPU'}, args.frames)
        gpu_ms = runvx_frame_ms(gdf, args.runvx, {'AGO_DEFAULT_TARGET': 'GPU'}, args.frames)

        if name.startswith('U8'):
            src = u8_src
        elif name.startswith('RGBX'):
            src = rgbx_src
        else:
            src = rgb_src

        # For constant-border GDFs the border value is 0 in the GDF; OpenCV matches.
        oc_ms = opencv_frame_ms(src, sx, sy, border, border_val, args.frames)
        ratio = oc_ms / gpu_ms if gpu_ms else float('inf')
        print(f"{name:<26} {cpu_ms:>18.3f} {gpu_ms:>18.3f} {oc_ms:>18.3f} {ratio:>14.2f}x")


if __name__ == '__main__':
    main()
