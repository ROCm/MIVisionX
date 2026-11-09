#!/usr/bin/env python3
"""
CTest helper: value regression for the CPU Phase kernel at widths that are not a multiple of 8.

The vector loops step 16 and then 8 pixels; the last (width mod 8) pixels of each row go through a
scalar tail. The tail wrote through a pointer the vector loops never advanced, so those pixels landed in
columns 0..(tail-1) and the tail columns were left unwritten.

Each width is checked column by column against atan2 computed here. The kernel's atan2 is an
approximation, so a value is accepted within one step of 256, modulo 256 (0 and 255 are neighbours).
"""
import argparse
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HEIGHT = 3
WIDTHS = (1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 130)


def expected(gx, gy):
    angle = math.atan2(gy, gx)
    if angle < 0:
        angle += 2 * math.pi
    return int(angle * 128 / math.pi + 0.5) & 0xFF


def check(work, runvx, width, rng):
    n = width * HEIGHT
    gx = [rng.randint(-3000, 3000) for _ in range(n)]
    gy = [rng.randint(-3000, 3000) for _ in range(n)]
    fx, fy, out, gdf = work / "gx.raw", work / "gy.raw", work / "out.raw", work / "phase.gdf"
    fx.write_bytes(struct.pack("<%dh" % n, *gx))
    fy.write_bytes(struct.pack("<%dh" % n, *gy))
    if out.exists():
        out.unlink()
    gdf.write_text(
        f"data gx = image:{width},{HEIGHT},S016:read,{fx}\n"
        f"data gy = image:{width},{HEIGHT},S016:read,{fy}\n"
        f"data o = image:{width},{HEIGHT},U008:write,{out}\n"
        "node org.khronos.openvx.phase gx gy o\n")
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "CPU"
    r = subprocess.run([str(runvx), "-frames:1", "-affinity:CPU", str(gdf)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    tag = f"width {width}"
    if r.returncode != 0 or not out.exists():
        return [(tag, f"runvx exited {r.returncode}:\n{r.stdout[-600:]}")]
    got = out.read_bytes()
    bad = [i for i in range(n) if abs((got[i] - expected(gx[i], gy[i]) + 128) % 256 - 128) > 1]
    if bad:
        cols = sorted({i % width for i in bad})
        return [(tag, f"{len(bad)} of {n} pixels wrong, columns {cols[:8]}{'...' if len(cols) > 8 else ''}")]
    print(f"PASS [{tag}]: {n} pixels within 1 of atan2")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runvx", required=True)
    args = parser.parse_args()
    rng = random.Random(3)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for width in WIDTHS:
            failures += check(Path(tmp), args.runvx, width, rng)
    for tag, detail in failures:
        print(f"FAIL [{tag}]: {detail}")
    if failures:
        print(f"{len(failures)} test(s) failed:")
        return 1
    print("All CPU Phase tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
