#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from pathlib import Path


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) * 0.5


def runvx_median_ms(gdf, runvx, env, frames):
    env = dict(os.environ, **env)
    subprocess.run([str(runvx), f'-frames:{max(1, frames // 10)}', str(gdf)], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([str(runvx), f'-frames:{frames}', str(gdf)], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed / frames)
    return median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runvx', required=True, type=Path)
    parser.add_argument('--gdf-dir', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=100)
    args = parser.parse_args()

    cases = [
        ('U8_Bilinear', 'Remap_U8_U8_Bilinear.gdf'),
        ('U8_Bilinear_Constant', 'Remap_U8_U8_Bilinear_Constant.gdf'),
        ('U8_Nearest_Constant', 'Remap_U8_U8_Nearest_Constant.gdf'),
        ('RGB_Bilinear', 'Remap_RGB_RGB_Bilinear.gdf'),
        ('RGB_Bilinear_Constant', 'Remap_RGB_RGB_Bilinear_Constant.gdf'),
        ('RGB_Nearest_Constant', 'Remap_RGB_RGB_Nearest_Constant.gdf'),
        ('RGBX_Bilinear', 'Remap_RGBX_RGBX_Bilinear.gdf'),
        ('RGBX_Bilinear_Constant', 'Remap_RGBX_RGBX_Bilinear_Constant.gdf'),
        ('RGBX_Nearest_Constant', 'Remap_RGBX_RGBX_Nearest_Constant.gdf'),
    ]

    print(f"{'Case':<26} {'CPU ms/frame':>14} {'GPU ms/frame':>14}")
    print("=" * 60)
    for name, gdf_name in cases:
        gdf = args.gdf_dir / gdf_name
        cpu_ms = runvx_median_ms(gdf, args.runvx, {'AGO_DEFAULT_TARGET': 'CPU'}, args.frames)
        gpu_ms = runvx_median_ms(gdf, args.runvx, {'AGO_DEFAULT_TARGET': 'GPU'}, args.frames)
        print(f"{name:<26} {cpu_ms:>14.3f} {gpu_ms:>14.3f}")


if __name__ == '__main__':
    main()
