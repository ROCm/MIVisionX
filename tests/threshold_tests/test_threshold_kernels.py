#!/usr/bin/env python3
"""
CTest helper: exact-value regression for the OpenVX threshold kernels.

Two independent scenarios are exercised on both CPU and GPU backends. Both
assert every output pixel against a reference computed here, because threshold
is exact integer logic -- there is no rounding to tolerate.

1. "u1-logic": a threshold feeding and/or/xor through virtual images. The graph
   optimizer lowers this to the 1-bit kernels (Threshold_U1_U8_Binary /
   _Range plus And/Or/Xor_U8_U1U1), which is the only way those kernels are
   reached. Those kernels pack 8 pixels per byte and process them four at a
   time through float4 lanes, so a constant that fails to broadcast across the
   lanes leaves pixels 1-3 of every group of four permanently 0. The per-lane
   mismatch counts are reported alongside the total so that signature is
   immediately visible rather than showing up as an undifferentiated count.

2. "s16-threshold": a signed 16-bit source thresholded to a U8 mask, with
   values deliberately outside 0..255 as well as inside. S16 images hold Sobel
   gradients, which routinely exceed 255 and go negative, so thresholds of 256,
   300, -50 and the range -100..100 are the ordinary case rather than an edge
   case. A kernel that truncates the threshold to its low byte passes every
   in-range value and fails only these.

Image dimensions are chosen so that neither scenario depends on row padding:
every width here is a multiple of 16, so the U8 and S16 rows are both unpadded
and a stride-handling defect cannot mask or manufacture a failure in this test.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


WIDTH, HEIGHT = 256, 64

# BINARY thresholds and RANGE bounds for the S16 scenario. The first three are
# inside 0..255 and must already pass; the rest are the ones low-byte
# truncation gets wrong (256 -> 0, 300 -> 44, -50 -> 206, -100..100 ->
# 156..100 which selects nothing, 100..400 -> 100..144).
S16_BINARY = [0, 127, 255, 256, 300, -50]
S16_RANGE = [(10, 200), (-100, 100), (100, 400)]


def u8_pattern(i, mul, add):
    """Deterministic byte pattern; no numpy so the test runs wherever CTest does."""
    return (i * mul + add) % 256


def s16_pattern(i):
    """Deterministic signed values spanning [-1000, 1000)."""
    return ((i * 97) % 2000) - 1000


def run_runvx(runvx_exe, gdf_path, backend):
    env = os.environ.copy()
    env["AGO_DEFAULT_TARGET"] = backend
    cmd = [str(runvx_exe), "-frames:1", str(gdf_path)]
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=120)


def check_u1_logic(work_dir, runvx_exe, backend):
    """threshold -> and/or/xor through virtual images; exact, with per-lane detail."""
    count = WIDTH * HEIGHT
    a = bytes(u8_pattern(i, 37, 0) for i in range(count))
    b = bytes(u8_pattern(i, 53, 11) for i in range(count))
    (work_dir / "a.u8").write_bytes(a)
    (work_dir / "b.u8").write_bytes(b)

    gdf = work_dir / f"u1_logic_{backend}.gdf"
    outs = {op: work_dir / f"{op}_{backend}.u8" for op in ("and", "or", "xor")}
    gdf.write_text(
        f"data a   = image:{WIDTH},{HEIGHT},U008:read,{work_dir / 'a.u8'}\n"
        f"data b   = image:{WIDTH},{HEIGHT},U008:read,{work_dir / 'b.u8'}\n"
        "data ta  = threshold:BINARY,U008,U008:INIT,95\n"
        "data tb  = threshold:RANGE,U008,U008:INIT,40,200\n"
        "data va  = image-virtual:0,0,U008\n"
        "data vb  = image-virtual:0,0,U008\n"
        f"data and = image:{WIDTH},{HEIGHT},U008:write,{outs['and']}\n"
        f"data or  = image:{WIDTH},{HEIGHT},U008:write,{outs['or']}\n"
        f"data xor = image:{WIDTH},{HEIGHT},U008:write,{outs['xor']}\n"
        "node org.khronos.openvx.threshold a ta va\n"
        "node org.khronos.openvx.threshold b tb vb\n"
        "node org.khronos.openvx.and va vb and\n"
        "node org.khronos.openvx.or  va vb or\n"
        "node org.khronos.openvx.xor va vb xor\n")

    result = run_runvx(runvx_exe, gdf, backend)
    if result.returncode != 0:
        return [(f"u1-logic {backend}",
                 f"runvx exited {result.returncode}:\n{result.stdout[-1000:]}")]

    ta = [x > 95 for x in a]
    tb = [40 <= x <= 200 for x in b]
    refs = {
        "and": [p and q for p, q in zip(ta, tb)],
        "or": [p or q for p, q in zip(ta, tb)],
        "xor": [p != q for p, q in zip(ta, tb)],
    }

    failures = []
    for op, ref in refs.items():
        path = outs[op]
        if not path.exists():
            failures.append((f"u1-logic {op} {backend}", f"output not created: {path}"))
            continue
        data = path.read_bytes()
        if len(data) != count:
            failures.append((f"u1-logic {op} {backend}",
                             f"size mismatch: expected {count}, got {len(data)}"))
            continue
        # per-lane counts: the float4-broadcast defect only ever spares x % 4 == 0
        lanes = [0, 0, 0, 0]
        first = None
        for i, (got, want) in enumerate(zip(data, ref)):
            if (got > 0) != want:
                lanes[(i % WIDTH) % 4] += 1
                if first is None:
                    first = (i % WIDTH, i // WIDTH, got, 255 if want else 0)
        total = sum(lanes)
        if total:
            x, y, got, want = first
            failures.append((f"u1-logic {op} {backend}",
                             f"{total} of {count} pixels wrong "
                             f"(by x%4 lane: {lanes}); first at (x={x}, y={y}) "
                             f"got {got}, expected {want}"))
        else:
            print(f"PASS [u1-logic {op} {backend}]: {count} pixels exact, all four lanes")
    return failures


def check_s16(work_dir, runvx_exe, backend):
    """S16 -> U8 threshold, including values outside 0..255."""
    count = WIDTH * HEIGHT
    values = [s16_pattern(i) for i in range(count)]
    src = work_dir / "in.s16"
    src.write_bytes(struct.pack(f"<{count}h", *values))

    cases = ([("BINARY", (t,)) for t in S16_BINARY] +
             [("RANGE", lu) for lu in S16_RANGE])

    failures = []
    for kind, bounds in cases:
        label = f"{kind} {','.join(str(v) for v in bounds)}"
        tag = f"s16 {label} {backend}"
        safe = label.replace(" ", "_").replace(",", "_").replace("-", "m")
        out = work_dir / f"s16_{safe}_{backend}.u8"
        gdf = work_dir / f"s16_{safe}_{backend}.gdf"
        gdf.write_text(
            f"data in  = image:{WIDTH},{HEIGHT},S016:read,{src}\n"
            f"data thr = threshold:{kind},S016,U008:INIT,"
            f"{','.join(str(v) for v in bounds)}\n"
            f"data out = image:{WIDTH},{HEIGHT},U008:write,{out}\n"
            "node org.khronos.openvx.threshold in thr out\n")

        result = run_runvx(runvx_exe, gdf, backend)
        if result.returncode != 0:
            failures.append((tag, f"runvx exited {result.returncode}:\n"
                                  f"{result.stdout[-1000:]}"))
            print(f"FAIL [{tag}]: runvx exited {result.returncode}")
            continue
        if not out.exists():
            failures.append((tag, f"output not created: {out}"))
            continue
        data = out.read_bytes()
        if len(data) != count:
            failures.append((tag, f"size mismatch: expected {count}, got {len(data)}"))
            continue

        if kind == "BINARY":
            ref = [v > bounds[0] for v in values]
        else:
            ref = [bounds[0] <= v <= bounds[1] for v in values]

        wrong = 0
        first = None
        for i, (got, want) in enumerate(zip(data, ref)):
            if (got > 0) != want:
                wrong += 1
                if first is None:
                    first = (i % WIDTH, i // WIDTH, values[i], got, 255 if want else 0)
        if wrong:
            x, y, v, got, want = first
            failures.append((tag, f"{wrong} of {count} pixels wrong; first at "
                                  f"(x={x}, y={y}) source {v}, got {got}, expected {want}"))
            print(f"FAIL [{tag}]: {wrong} of {count} pixels wrong")
        else:
            print(f"PASS [{tag}]: {count} pixels exact")
    return failures


def main():
    parser = argparse.ArgumentParser(description="OpenVX threshold kernel correctness test")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for backend in ("CPU", "GPU"):
            failures += check_u1_logic(work_dir, runvx_exe, backend)
            failures += check_s16(work_dir, runvx_exe, backend)

    if failures:
        print(f"\n{len(failures)} test(s) failed:", file=sys.stderr)
        for tag, msg in failures:
            print(f"  [{tag}] {msg}", file=sys.stderr)
        return 1

    print("\nAll threshold kernel tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
