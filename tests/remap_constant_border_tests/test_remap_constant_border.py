#!/usr/bin/env python3
"""
CTest helper: verify that VX_BORDER_MODE_CONSTANT is honored by RGB/RGBX Remap.

Creates a temporary GDF with a uniform input image and a remap table that maps
some destination pixels to out-of-bounds source coordinates. With
attr:BORDER_MODE:CONSTANT,<border>, the out-of-bounds pixels in the output must
be exactly <border>. The test runs on both CPU and GPU backends.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def build_gdf(work_dir, src_w, src_h, dst_w, dst_h, fmt_fourcc, bpp, border_value):
    """Build a GDF that maps the whole destination outside the source image."""
    gdf_path = work_dir / f"remap_constant_border_{fmt_fourcc}.gdf"
    out_path = work_dir / f"out_{fmt_fourcc}.raw"
    # Uniform input image with a value different from the border, so that
    # sampling the source by mistake is detectable.
    input_value = 0xAABBCC if bpp == 3 else 0xAABBCCDD
    # Remap table: map every destination pixel to source (-1, -1), which is
    # out of bounds and should produce the constant border.
    remap_path = work_dir / f"remap_oob_{fmt_fourcc}.txt"
    with open(remap_path, "w") as f:
        for _ in range(dst_w * dst_h):
            f.write("-1 -1\n")

    content = (
        f"data input_1 = uniform-image:{src_w},{src_h},{fmt_fourcc},{input_value}\n"
        f"data output_1 = image:{dst_w},{dst_h},{fmt_fourcc}:write,{out_path}\n"
        f"data remap_table = remap:{src_w},{src_h},{dst_w},{dst_h}:read,{remap_path}\n"
        f"node org.khronos.openvx.remap input_1 remap_table !BILINEAR output_1 "
        f"attr:BORDER_MODE:CONSTANT,{border_value}\n"
    )
    gdf_path.write_text(content)
    return gdf_path, out_path


def run_runvx(runvx_exe, gdf_path, backend):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = backend
    cmd = [str(runvx_exe), "-frames:1", str(gdf_path)]
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=120)
    return result


def verify_output(out_path, dst_w, dst_h, bpp, border_value):
    expected_size = dst_w * dst_h * bpp
    if not out_path.exists():
        return False, f"output file not created: {out_path}"
    actual_size = out_path.stat().st_size
    if actual_size != expected_size:
        return False, f"output size mismatch: expected {expected_size}, got {actual_size}"

    data = out_path.read_bytes()
    # Use a border value with the same byte in every channel so that byte-order
    # inside the packed pixel value does not matter. Input value 0xAABBCC(DD)
    # is chosen to differ from any uniform border.
    expected_byte = border_value & 0xFF
    bad = [b for b in data if b != expected_byte]
    if bad:
        for i, b in enumerate(data):
            if b != expected_byte:
                return False, f"first mismatch at byte {i}: got {b}, expected {expected_byte}"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Constant-border Remap correctness test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    parser.add_argument("--gdf-dir", required=True, help="directory containing existing Remap GDFs")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    tests = [
        # (format name, runvx fourcc, bytes per pixel, src_w, src_h, dst_w, dst_h, border_value)
        # border_value is chosen so every packed byte is the same (0x424242 or 0x63636363),
        # making the verification independent of channel byte order.
        ("RGB", "RGB2", 3, 16, 16, 32, 32, 0x424242),
        ("RGBX", "RGBX", 4, 16, 16, 32, 32, 0x63636363),
    ]

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for label, fourcc, bpp, src_w, src_h, dst_w, dst_h, border in tests:
            gdf_path, out_path = build_gdf(work_dir, src_w, src_h, dst_w, dst_h,
                                            fourcc, bpp, border)
            for backend in ("CPU", "GPU"):
                result = run_runvx(runvx_exe, gdf_path, backend)
                if result.returncode != 0:
                    failures.append((label, backend,
                                     f"runvx failed (exit {result.returncode}):\n"
                                     f"{result.stdout[-1000:]}"))
                    print(f"FAIL [{label} {backend}]: runvx exited with code {result.returncode}")
                    continue
                ok, msg = verify_output(out_path, dst_w, dst_h, bpp, border)
                if not ok:
                    failures.append((label, backend, msg))
                    print(f"FAIL [{label} {backend}]: {msg}")
                else:
                    print(f"PASS [{label} {backend}]: constant border {border:#x} honored")

    if failures:
        print(f"\n{len(failures)} test(s) failed.", file=sys.stderr)
        return 1

    print("\nAll constant-border Remap tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
