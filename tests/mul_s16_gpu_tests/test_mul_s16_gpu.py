#!/usr/bin/env python3
"""
CTest helper: exact-value regression for the HIP S16 x S16 -> S16 multiply with round-to-nearest-even.

The kernels formed scale * a * b in float32, whose 24-bit mantissa cannot hold the product of two
16-bit values, so the product was rounded before it was rounded to an integer: with scale 1.0 and
WRAP, 70% of random full-range pixels were wrong. Every case here is exact: the reference is computed
here with fractions, from the same float32 scale the kernel receives.

For each policy combination the test multiplies two random full-range S16 images and compares the
GPU output with the exact result of round_half_even(scale * a * b), wrapped or saturated to 16 bits.
"""
import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

WIDTH, HEIGHT = 256, 64


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def reference(a, b, scale, saturate):
    p = round(Fraction(scale) * a * b)  # Fraction rounds half to even
    if saturate:
        return max(-32768, min(32767, p))
    return (p + 32768) % 65536 - 32768


def write_s16(path, values):
    path.write_bytes(struct.pack("<%dh" % len(values), *values))


def run_gpu(runvx, gdf):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "GPU"
    return subprocess.run([str(runvx), "-frames:1", "-affinity:GPU", str(gdf)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)


def check(work, runvx, a, b, scale, convert, tag):
    fa, fb, out, gdf = work / "a.raw", work / "b.raw", work / "out.raw", work / "mul.gdf"
    write_s16(fa, a)
    write_s16(fb, b)
    if out.exists():
        out.unlink()
    gdf.write_text(
        f"data a = image:{WIDTH},{HEIGHT},S016:read,{fa}\n"
        f"data b = image:{WIDTH},{HEIGHT},S016:read,{fb}\n"
        f"data o = image:{WIDTH},{HEIGHT},S016:write,{out}\n"
        f"data s = scalar:FLOAT32,{scale!r}\n"
        f"node org.khronos.openvx.multiply a b s !{convert} !ROUND_POLICY_TO_NEAREST_EVEN o\n")
    result = run_gpu(runvx, gdf)
    if result.returncode != 0 or not out.exists():
        return [(tag, f"runvx exited {result.returncode}:\n{result.stdout[-800:]}")]
    got = struct.unpack("<%dh" % (WIDTH * HEIGHT), out.read_bytes())
    saturate = convert == "SATURATE"
    scale32 = f32(scale)
    wrong = [i for i in range(WIDTH * HEIGHT) if got[i] != reference(a[i], b[i], scale32, saturate)]
    if wrong:
        i = wrong[0]
        return [(tag, f"{len(wrong)} of {WIDTH * HEIGHT} pixels wrong; first: a={a[i]} b={b[i]} "
                      f"expected {reference(a[i], b[i], scale32, saturate)} got {got[i]}")]
    print(f"PASS [{tag}]: {WIDTH * HEIGHT} pixels exact")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runvx", required=True)
    args = parser.parse_args()
    rng = random.Random(1)
    n = WIDTH * HEIGHT
    full = ([rng.randint(-32768, 32767) for _ in range(n)], [rng.randint(-32768, 32767) for _ in range(n)])
    # operands that keep scale * a * b inside the S16 range even though a * b is far beyond 2^24
    wide = ([rng.randint(-32768, 32767) for _ in range(n)], [rng.randint(-32768, 32767) for _ in range(n)])
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for convert in ("WRAP", "SATURATE"):
            failures += check(work, args.runvx, *full, 1.0, convert, f"{convert} scale 1")
        # 2^-16: the scaled result stays inside the S16 range for every pair, so the rounding of the
        # product matters for the saturating kernel too
        failures += check(work, args.runvx, *wide, 1.0 / 65536, "SATURATE", "SATURATE scale 2^-16")
        failures += check(work, args.runvx, *wide, 1.0 / 65536, "WRAP", "WRAP scale 2^-16")
    for tag, detail in failures:
        print(f"FAIL [{tag}]: {detail}")
    if failures:
        print(f"{len(failures)} test(s) failed:")
        return 1
    print("All HIP S16 multiply tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
