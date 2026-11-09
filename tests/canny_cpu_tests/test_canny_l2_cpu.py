#!/usr/bin/env python3
"""
CTest helper: Canny with gradient_size 3 and VX_NORM_L2 on the CPU must find the same edges as VX_NORM_L1.

The kernel squared the 16-bit Sobel gradients with a 16-bit multiply, which wraps at |G| = 256 (256^2 is
65536, so 0). On a vertical step the magnitude at the step column was lost and the edge moved to the
columns next to it.

The image is a step of height h at column 32: columns 0-31 are 0, column 32 is h/2, columns 33 and up are h.
The 3x3 Sobel gives |Gx| = 4h at column 32 and 2h at columns 31 and 33, so non-maximum suppression keeps
exactly column 32, for L1 and for L2 alike. The step is also checked transposed (a horizontal edge).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SIZE = 64
HEIGHTS = (32, 64, 100, 128, 200, 255)  # |Gx| = 4h: below and above the 256 where the square wrapped


def step_image(h, transpose):
    rows = []
    for y in range(SIZE):
        for x in range(SIZE):
            u, v = (y, x) if transpose else (x, y)
            rows.append(0 if u < 32 else (h // 2 if u == 32 else h))
    return bytes(rows)


def edge_positions(data, transpose):
    """The columns (or rows, if transposed) that hold an edge in the middle line of the image."""
    mid = SIZE // 2
    line = [data[mid * SIZE + x] for x in range(SIZE)] if not transpose else [data[y * SIZE + mid] for y in range(SIZE)]
    return [i for i, v in enumerate(line) if v]


def edges(work, runvx, img, norm):
    src, out, gdf = work / "in.u8", work / "edges.u8", work / "canny.gdf"
    src.write_bytes(img)
    if out.exists():
        out.unlink()
    gdf.write_text(
        f"data in = image:{SIZE},{SIZE},U008:read,{src}\n"
        f"data out = image:{SIZE},{SIZE},U008:write,{out}\n"
        "data t = threshold:RANGE,U008,U008:INIT,80,100\n"
        "data g = scalar:INT32,3\n"
        f"node org.khronos.openvx.canny_edge_detector in t g !{norm} out\n")
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = "CPU"
    r = subprocess.run([str(runvx), "-frames:1", "-affinity:CPU", str(gdf)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"runvx exited {r.returncode}:\n{r.stdout[-600:]}")
    return out.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runvx", required=True)
    args = parser.parse_args()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for transpose in (False, True):
            for h in HEIGHTS:
                tag = f"step {h}{' transposed' if transpose else ''}"
                img = step_image(h, transpose)
                l1 = edge_positions(edges(work, args.runvx, img, "NORM_L1"), transpose)
                l2 = edge_positions(edges(work, args.runvx, img, "NORM_L2"), transpose)
                if l2 != [32] or l1 != [32]:
                    failures.append((tag, f"edge at {l1} with L1 and at {l2} with L2, expected [32] for both"))
                else:
                    print(f"PASS [{tag}]: L1 and L2 both find the edge at 32")
    for tag, detail in failures:
        print(f"FAIL [{tag}]: {detail}")
    if failures:
        print(f"{len(failures)} test(s) failed:")
        return 1
    print("All CPU Canny L2 tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
