#!/usr/bin/env python3
"""
CTest helper: verify that VX_BORDER_MODE_CONSTANT is honored by U8/RGB/RGBX Remap.

Two independent scenarios are exercised on both CPU and GPU backends:

1. "full-border": every destination pixel maps fully out of bounds (source
   (-1, -1)). vxSetRemapPoint marks these as the out-of-bounds sentinel, so every
   output pixel must equal the constant border exactly.

2. "edge-blend": every destination pixel maps to the non-sentinel source-edge
   coordinate (src_w - 0.5, src_h - 0.5). Only the (src_w-1, src_h-1) neighbor is
   in bounds; the other three 2x2 neighbors are out of bounds and must contribute
   the constant border. This lands on interior destination pixels, which is the
   region a coordinate-heuristic SIMD fast path used to (incorrectly) treat as
   fully interior -- there it skipped per-sample border substitution and over-read
   the source buffer. The expected value is computed from the same fixed-point
   bilinear formula the kernels use, so a regression that samples the source edge
   (edge replication) or reads out-of-bounds memory instead of the border is
   caught.
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


def repeat_byte(value, nbytes):
    """Pack a single byte into every channel so byte order is irrelevant."""
    out = 0
    for _ in range(nbytes):
        out = (out << 8) | (value & 0xFF)
    return out


def fixed_point(coord):
    """Match vxSetRemapPoint: (vx_uint16)(coord * 8 + 0.5f), 3 fractional bits."""
    return int(coord * 8.0 + 0.5)


def bilinear_ref(sx, sy, src_w, src_h, src_val, border):
    """Per-channel reference mirroring the constant-border bilinear kernel."""
    if sx < 0.0 or sy < 0.0 or sx >= src_w or sy >= src_h:
        return border  # sentinel -> full border
    fxp, fyp = fixed_point(sx), fixed_point(sy)
    mx, my = fxp >> 3, fyp >> 3
    fx, fy = fxp & 7, fyp & 7

    def sample(xx, yy):
        return src_val if (0 <= xx < src_w and 0 <= yy < src_h) else border

    v00 = sample(mx, my)
    v10 = sample(mx + 1, my)
    v01 = sample(mx, my + 1)
    v11 = sample(mx + 1, my + 1)
    w00 = (8 - fx) * (8 - fy)
    w10 = fx * (8 - fy)
    w01 = (8 - fx) * fy
    w11 = fx * fy
    return (v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11 + 32) >> 6


def build_gdf(work_dir, scenario, fourcc, bpp, src_w, src_h, dst_w, dst_h,
              src_byte, border_byte):
    """Write a GDF plus its remap table for the given scenario."""
    gdf_path = work_dir / f"remap_{scenario}_{fourcc}.gdf"
    out_path = work_dir / f"out_{scenario}_{fourcc}.raw"
    remap_path = work_dir / f"remap_{scenario}_{fourcc}.txt"

    if scenario == "full-border":
        sx, sy = -1.0, -1.0
    elif scenario == "edge-blend":
        sx, sy = src_w - 0.5, src_h - 0.5
    else:
        raise ValueError(scenario)

    with open(remap_path, "w") as f:
        for _ in range(dst_w * dst_h):
            f.write(f"{sx} {sy}\n")

    src_val = repeat_byte(src_byte, bpp)
    border_val = repeat_byte(border_byte, bpp)
    content = (
        f"data input_1 = uniform-image:{src_w},{src_h},{fourcc},{src_val}\n"
        f"data output_1 = image:{dst_w},{dst_h},{fourcc}:write,{out_path}\n"
        f"data remap_table = remap:{src_w},{src_h},{dst_w},{dst_h}:read,{remap_path}\n"
        f"node org.khronos.openvx.remap input_1 remap_table !BILINEAR output_1 "
        f"attr:BORDER_MODE:CONSTANT,{border_val}\n"
    )
    gdf_path.write_text(content)

    expected_byte = bilinear_ref(sx, sy, src_w, src_h, src_byte, border_byte)
    return gdf_path, out_path, expected_byte


def run_runvx(runvx_exe, gdf_path, backend):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = backend
    cmd = [str(runvx_exe), "-frames:1", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=120)


def verify_output(out_path, dst_w, dst_h, bpp, expected_byte, tolerance):
    expected_size = dst_w * dst_h * bpp
    if not out_path.exists():
        return False, f"output file not created: {out_path}"
    actual_size = out_path.stat().st_size
    if actual_size != expected_size:
        return False, f"output size mismatch: expected {expected_size}, got {actual_size}"

    data = out_path.read_bytes()
    for i, b in enumerate(data):
        if abs(b - expected_byte) > tolerance:
            return False, (f"first mismatch at byte {i}: got {b}, "
                           f"expected {expected_byte} (+/-{tolerance})")
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Constant-border Remap correctness test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    parser.add_argument("--gdf-dir", required=False, help="unused; kept for CTest compatibility")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    src_w, src_h, dst_w, dst_h = 16, 16, 32, 32
    src_byte, border_byte = 200, 40
    # full-border is exact; edge-blend allows 1 LSB for any benign CPU/GPU
    # rounding difference. The bug this guards produces edge replication (~200)
    # or out-of-bounds garbage, both far outside a 1-LSB band around the
    # expected blend (80), so discrimination is preserved.
    scenarios = [("full-border", 0), ("edge-blend", 1)]

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for scenario, tol in scenarios:
            for label, fourcc, bpp in FORMATS:
                gdf_path, out_path, expected = build_gdf(
                    work_dir, scenario, fourcc, bpp, src_w, src_h, dst_w, dst_h,
                    src_byte, border_byte)
                for backend in ("CPU", "GPU"):
                    result = run_runvx(runvx_exe, gdf_path, backend)
                    tag = f"{scenario} {label} {backend}"
                    if result.returncode != 0:
                        failures.append((tag, f"runvx exited {result.returncode}:\n"
                                              f"{result.stdout[-1000:]}"))
                        print(f"FAIL [{tag}]: runvx exited with code {result.returncode}")
                        continue
                    ok, msg = verify_output(out_path, dst_w, dst_h, bpp, expected, tol)
                    if not ok:
                        failures.append((tag, msg))
                        print(f"FAIL [{tag}]: {msg}")
                    else:
                        print(f"PASS [{tag}]: output byte {expected} (+/-{tol}) honored")

    if failures:
        print(f"\n{len(failures)} test(s) failed.", file=sys.stderr)
        return 1

    print("\nAll constant-border Remap tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
