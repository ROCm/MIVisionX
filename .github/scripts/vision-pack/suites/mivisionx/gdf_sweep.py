#!/usr/bin/env python3
# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Per-GDF runvx sweep over the installed amd_openvx_gdfs tree.

For every GDF: runvx -dump-gdf -frames:10 -affinity:<T> -dump-profile file <gdf>
under a timeout. Pass = exit 0 plus the "total elapsed time" line and the
",GRAPH" profile line. runOpenVX.py is not used: it hides the elapsed-time line
and reports one result for the whole tree. runvx chdir()s into the GDF's
directory (the vision GDFs read inputs/ relative to it); nothing is written.

IDs: mivisionx::gdf.<T>.<category>::<file>.gdf (cpu/hidden -> cpu-hidden),
     mivisionx::gpu-fallback::<category>/<file>.gdf (--fallback, GPU only),
     or mivisionx::<--group>::<category>/<file>.gdf with --only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mvx_common import RUNVX, TEST_ROOT, VP_OUT, record, run, slug, tail, write_log

CATEGORIES = ["arithmetic", "color", "filter", "geometric", "logical", "statistical", "vision",
              "vision_profile", "cpu", "cpu/hidden"]
PROFILE = re.compile(r"^\s*\d+,\s*[\d.]+,\s*[\d.]+,\s*[\d.]+,\s*[\d.]+,(CPU|GPU),(\S+)\s*$", re.M)
FALLBACK_GDFS = re.compile(r"^vision/(Canny|Harris)_.*\.gdf$")


def run_one(gdf: Path, target: str, frames: int, timeout: float):
    cmd = [RUNVX, "-dump-gdf", f"-frames:{int(frames)}", f"-affinity:{target}", "-dump-profile", "file", gdf]
    r = run(cmd, timeout, cwd=gdf.parent)
    has_time = "total elapsed time" in r.out
    has_graph = re.search(r",(CPU|GPU),GRAPH\s*$", r.out, re.M) is not None
    status = r.status()
    msg = ""
    if status == "pass" and not (has_time and has_graph):
        status = "fail"
        msg = "exit 0 but missing {}".format(" and ".join(
            x for x, ok in (("'total elapsed time'", has_time), ("',GRAPH' profile line", has_graph)) if not ok))
    elif status != "pass":
        msg = r.why()
    if status != "pass":
        errs = [ln.strip() for ln in r.out.splitlines() if "ERROR" in ln][:3]
        msg += ": " + (" | ".join(errs) if errs else tail(r.out, 400))
    return r, status, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=("CPU", "GPU"))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--only", nargs="*", help="relative GDF paths to run instead of the full tree")
    ap.add_argument("--group", default="", help="group for --only runs")
    ap.add_argument("--root", help="directory the --only paths are relative to (default: amd_openvx_gdfs); "
                                   "with --root the IDs use the bare file name")
    ap.add_argument("--fallback", action="store_true", help="GPU: record CPU fallback for Canny/Harris GDFs")
    ap.add_argument("--metrics", help="write a vp_perf JSON with fallback counts here")
    a = ap.parse_args()
    root = Path(a.root) if a.root else TEST_ROOT / "amd_openvx_gdfs"
    if a.only:
        items = [(Path(p).parent.as_posix(), root / p) for p in a.only]
    else:
        items = [(c, g) for c in CATEGORIES for g in sorted((root / c).glob("*.gdf"))]
    if not items:
        record(f"gdf.{a.target}::discovery", "error", f"no GDFs under {root}")
        return 0
    fallback_kernels: set[str] = set()
    fallback_gdfs = 0
    counts: dict[str, int] = {}
    for cat, gdf in items:
        rel = f"{cat}/{gdf.name}"
        if a.only:
            tid = f"{a.group}::{gdf.name if a.root else rel}"
            log = VP_OUT / "logs" / (f"{slug(a.group)}.log")
        else:
            grp = f"gdf.{a.target}.{cat.replace('/', '-')}"
            tid = f"{grp}::{gdf.name}"
            log = VP_OUT / "logs" / (f"{grp}.log")
        if not gdf.is_file():
            record(tid, "error", f"GDF not installed: {gdf}", log=log, backend=a.target)
            continue
        r, status, msg = run_one(gdf, a.target, a.frames, a.timeout)
        write_log(log, f"### {tid} rc={int(r.rc)} {r.dt:.2f}s", f"### cmd: {r.repro(cwd=gdf.parent)}", r.out)
        record(tid, status, msg, r.dt, log, a.target, r.repro(cwd=gdf.parent))
        counts[status] = counts.get(status, 0) + 1
        if a.target == "GPU" and status == "pass":
            cpu_nodes = sorted({k.split(".")[-1] for d, k in PROFILE.findall(r.out) if d == "CPU" and k != "GRAPH"})
            fallback_kernels.update(cpu_nodes)
            fallback_gdfs += bool(cpu_nodes)
            if a.fallback and FALLBACK_GDFS.match(rel):
                record(f"gpu-fallback::{rel}", "fail" if cpu_nodes else "pass",
                       f"GPU-affinity graph ran {len(cpu_nodes)} kernel(s) on CPU: {', '.join(cpu_nodes)}"
                       if cpu_nodes else "", r.dt, log, "GPU", r.repro(cwd=gdf.parent))
    print(f"gdf {a.target}: {counts}")
    if a.metrics and a.target == "GPU":
        Path(a.metrics).write_text(json.dumps({"metrics": [
            {"name": "gdf.gpu_fallback_kernel_types", "value": len(fallback_kernels), "unit": "count",
             "lower_is_better": True, "backend": "GPU"},
            {"name": "gdf.gpu_fallback_graphs", "value": fallback_gdfs, "unit": "count",
             "lower_is_better": True, "backend": "GPU"}],
            "fallback_kernels": sorted(fallback_kernels)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
