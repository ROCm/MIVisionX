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

"""Run a small set of OpenVX GDF tests under rocprofv3 and produce a markdown summary.

This is intended as a lightweight CI showcase for the MIVisionX ROCTX/rocprof
instrumentation. It builds runvx with MIVISIONX_ENABLE_ROCPROF=ON, enables
markers at runtime via MIVISIONX_ROCPROF=1, and wraps the runvx GDF execution
with rocprofv3.

For CPU runs only marker traces are expected (no GPU kernels). For HIP runs
both marker and kernel dispatch traces are expected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

log = logging.getLogger("run_rocprof_gdf")

ROCPROF_TIMEOUT_SEC = 600
DEFAULT_FRAMES = 10


class RangeStat(NamedTuple):
    name: str
    count: int
    total_us: float
    max_us: float


class KernelStat(NamedTuple):
    name: str
    count: int
    total_us: float
    max_us: float


def run_rocprof(
    *,
    rocprof: Path,
    runvx: Path,
    gdf: Path,
    affinity: str,
    frames: int,
    output_dir: Path,
    backend: str,
) -> Tuple[bool, Optional[Path]]:
    """Run a single GDF under rocprofv3 and return (ok, output_csv_dir)."""
    gdf_name = gdf.stem
    per_gdf_dir = output_dir / gdf_name
    per_gdf_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MIVISIONX_ROCPROF"] = "1"
    env["LD_LIBRARY_PATH"] = f"{runvx.parent.parent / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    if backend == "HIP":
        # Pin to a single device for HIP runs. All three visibility vars must
        # agree: rocprofv3 aborts (SIGABRT) if HIP_VISIBLE_DEVICES and
        # GPU_DEVICE_ORDINAL disagree ("Conflicting visibility of agent-N").
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["ROCR_VISIBLE_DEVICES"] = "0"
        env["GPU_DEVICE_ORDINAL"] = "0"

    cmd = [
        str(rocprof),
        "--marker-trace",
        "--kernel-trace",
        "--summary",
        "--summary-units", "usec",
        "--output-format", "csv",
        "--output-directory", str(per_gdf_dir),
        "--",
        str(runvx),
        "-dump-gdf",
        f"-frames:{frames}",
        f"-affinity:{affinity}",
        "-dump-profile", "file",
        str(gdf),
    ]

    log.info("Running: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
            timeout=ROCPROF_TIMEOUT_SEC,
            # Run from the GDF's directory so GDFs with relative input paths
            # (e.g. Remap's inputs/remap_*.txt) resolve their data files.
            cwd=str(gdf.parent),
        )
    except subprocess.TimeoutExpired as exc:
        log.error("rocprof timeout for %s: %s", gdf_name, exc)
        (per_gdf_dir / "timeout.txt").write_text("rocprof timed out")
        return False, None

    (per_gdf_dir / "rocprof.log").write_text(completed.stdout)

    if completed.returncode != 0:
        log.error("rocprof failed for %s (exit %d)", gdf_name, completed.returncode)
        return False, None

    # Find the generated CSV directory. rocprofv3 writes to a subdir named
    # hostname/pid unless told otherwise; the --output-directory prefix may
    # contain additional nested dirs.
    csv_dirs = [p for p in per_gdf_dir.rglob("*_marker_api_trace.csv") if p.is_file()]
    if not csv_dirs:
        log.warning("No marker CSV found for %s", gdf_name)
        return True, per_gdf_dir
    csv_dir = csv_dirs[0].parent
    return True, csv_dir


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _marker_name(row: Dict[str, str]) -> str:
    """Extract the ROCTX message/range name from a marker CSV row."""
    # Prefer an explicit Name/Label/Arg0 column if present.
    for key in ("Name", "Label", "Arg0", "args"):
        value = row.get(key, "").strip()
        if value and value not in ("<unknown>", "N/A"):
            return value

    # Otherwise try to extract it from Function: roctxRangePush("...")
    func = row.get("Function", "").strip()
    m = re.search(r'"([^"]*)"', func)
    if m:
        return m.group(1)
    m = re.search(r"\(([^)]*)\)", func)
    if m:
        return m.group(1)
    return func


def parse_marker_ranges(csv_dir: Path) -> List[RangeStat]:
    """Parse rocprofv3 marker_api_trace.csv into range statistics.

    rocprofv3 can emit marker ranges in two shapes:
    * MARKER_CORE_RANGE_API rows where each row is a complete range with a
      Function name and Start/End timestamps (newer rocprofv3).
    * MARKER_CORE_API rows with separate roctxRangePush/roctxRangePop events
      that must be paired to compute durations.
    """
    marker_file = next(csv_dir.glob("*_marker_api_trace.csv"), None)
    if marker_file is None:
        return []

    durations: Dict[str, List[float]] = defaultdict(list)

    # First pass: complete-range rows (MARKER_CORE_RANGE_API / MARKER_RANGE_API).
    with marker_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row.get("Domain", "").strip()
            if "RANGE" not in domain:
                continue
            name = _marker_name(row)
            if not name:
                continue
            start = _safe_float(row.get("Start_Timestamp", "0"))
            end = _safe_float(row.get("End_Timestamp", "0"))
            if end <= start:
                continue
            durations[name].append(end - start)

    if durations:
        stats = []
        for name, values in sorted(durations.items()):
            if not name:
                name = "<unnamed>"
            stats.append(
                RangeStat(
                    name=name, count=len(values), total_us=sum(values), max_us=max(values)
                )
            )
        return stats

    # Second pass: legacy/event-based rows with push/pop pairs.
    open_ranges: Dict[Tuple[str, int], Tuple[str, float]] = {}
    with marker_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            func = row.get("Function", "").strip()
            thread = row.get("Thread_Id", "0")
            ts = _safe_float(row.get("Start_Timestamp", "0"))
            name = _marker_name(row)

            op = (row.get("Operation", "") or row.get("Kind", "")).strip()
            is_push = "roctxRangePush" in func or "RangePush" in op or op == "push"
            is_pop = "roctxRangePop" in func or "RangePop" in op or op == "pop"

            if not is_push and not is_pop:
                continue

            if is_push:
                depth = len([k for k in open_ranges if k[0] == thread])
                open_ranges[(thread, depth)] = (name, ts)
            elif is_pop:
                thread_pushes = {
                    k: v for k, v in open_ranges.items() if k[0] == thread
                }
                if not thread_pushes:
                    continue
                max_depth = max(k[1] for k in thread_pushes)
                range_name, start_ts = open_ranges.pop((thread, max_depth))
                end_ts = _safe_float(row.get("End_Timestamp", str(ts)))
                durations[range_name].append(abs(end_ts - start_ts))

    stats = []
    for name, values in sorted(durations.items()):
        if not name:
            name = "<unnamed>"
        stats.append(
            RangeStat(name=name, count=len(values), total_us=sum(values), max_us=max(values))
        )
    return stats


def parse_kernel_dispatches(csv_dir: Path) -> List[KernelStat]:
    """Parse rocprofv3 kernel_dispatch_trace.csv into kernel statistics."""
    kernel_file = next(csv_dir.glob("*_kernel_dispatch_trace.csv"), None)
    if kernel_file is None:
        return []

    durations: Dict[str, List[float]] = defaultdict(list)
    with kernel_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", row.get("Name", "")).strip()
            if not name:
                continue
            start = _safe_float(row.get("Start_Timestamp", "0"))
            end = _safe_float(row.get("End_Timestamp", "0"))
            durations[name].append(abs(end - start))

    stats = []
    for name, values in sorted(durations.items()):
        stats.append(
            KernelStat(
                name=name,
                count=len(values),
                total_us=sum(values),
                max_us=max(values),
            )
        )
    return stats


def _format_us(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.2f} us"


def build_summary(
    *,
    backend: str,
    gdf_results: List[Tuple[Path, bool, Optional[Path]]],
    frames: int,
) -> str:
    """Build a markdown summary of the rocprof runs."""
    lines: List[str] = []
    lines.append(f"# ROCprof GDF Profiling Summary — {backend} backend")
    lines.append("")
    lines.append(
        "This job exercises the MIVisionX ROCTX instrumentation by running a small "
        "set of GDF graphs under `rocprofv3`. The OpenVX library was built with "
        "`-DMIVISIONX_ENABLE_ROCPROF=ON` and `MIVISIONX_ROCPROF=1` is set at runtime."
    )
    lines.append("")
    lines.append("## How to reproduce locally")
    lines.append("")
    lines.append("```bash")
    lines.append("# Build MIVisionX with ROCTX support")
    lines.append("cmake -B build -S . -DMIVISIONX_ENABLE_ROCPROF=ON [ -DBACKEND=CPU|HIP ]")
    lines.append("cmake --build build --target runvx")
    lines.append("")
    lines.append("# Profile a GDF")
    lines.append("export MIVISIONX_ROCPROF=1")
    lines.append(
        "rocprofv3 --marker-trace --kernel-trace --summary --output-format csv \\"
    )
    lines.append("  --output-directory ./rocprof_results -- \\")
    lines.append("  ./build/bin/runvx -dump-gdf -frames:10 -affinity:GPU \\")
    lines.append("  tests/amd_openvx_gdfs/color/ColorConvert_RGB_RGBX.gdf")
    lines.append("```")
    lines.append("")
    lines.append(f"## Results per GDF ({frames} frames each)")
    lines.append("")
    lines.append("| GDF | Status | Top ROCTX range | Kernel dispatches |")
    lines.append("|-----|--------|-----------------|-------------------|")

    aggregate_ranges: Dict[str, List[float]] = defaultdict(list)
    aggregate_kernels: Dict[str, List[float]] = defaultdict(list)
    total_ok = 0

    for gdf, ok, csv_dir in gdf_results:
        status = "PASS" if ok else "FAIL"
        top_range = "—"
        kernel_count = 0
        if ok and csv_dir:
            ranges = parse_marker_ranges(csv_dir)
            kernels = parse_kernel_dispatches(csv_dir)
            if ranges:
                top = max(ranges, key=lambda s: s.total_us)
                top_range = f"`{top.name}` {_format_us(top.total_us)}"
                for r in ranges:
                    aggregate_ranges[r.name].append(r.total_us)
            if kernels:
                kernel_count = sum(k.count for k in kernels)
                for k in kernels:
                    aggregate_kernels[k.name].extend([k.total_us / max(k.count, 1)] * k.count)
        lines.append(f"| `{gdf.name}` | {status} | {top_range} | {kernel_count} |")
        if ok:
            total_ok += 1

    lines.append("")
    lines.append(f"**Passed:** {total_ok}/{len(gdf_results)}")
    lines.append("")

    if aggregate_ranges:
        lines.append("## Aggregate ROCTX ranges")
        lines.append("")
        lines.append("| Range name | Occurrences | Total time | Max time |")
        lines.append("|------------|------------:|-----------:|---------:|")
        for name, values in sorted(aggregate_ranges.items(), key=lambda kv: -sum(kv[1])):
            total = sum(values)
            lines.append(
                f"| `{name}` | {len(values)} | {_format_us(total)} | {_format_us(max(values))} |"
            )
        lines.append("")

    if aggregate_kernels:
        lines.append("## Aggregate kernel dispatches (HIP only)")
        lines.append("")
        lines.append("| Kernel name | Count | Total time | Max time |")
        lines.append("|-------------|------:|-----------:|---------:|")
        for name, values in sorted(aggregate_kernels.items(), key=lambda kv: -sum(kv[1])):
            total = sum(values)
            lines.append(
                f"| `{name}` | {len(values)} | {_format_us(total)} | {_format_us(max(values))} |"
            )
        lines.append("")
    elif backend == "HIP":
        lines.append(
            "_No kernel dispatch data captured. Verify that the HIP backend "
            "dispatched kernels and that rocprofv3 was able to collect traces._"
        )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        "Raw `rocprofv3` CSV traces are attached to this workflow run as "
        f"`rocprof-gdf-{backend.lower()}-results`."
    )
    lines.append("")
    return "\n".join(lines)


def discover_gdfs(gdf_root: Path, categories: Optional[List[str]]) -> List[Path]:
    if categories:
        gdfs: List[Path] = []
        for cat in categories:
            gdfs.extend(sorted((gdf_root / cat).glob("*.gdf")))
        return gdfs

    # Default showcase set: a representative, fast kernel from each category.
    default = [
        "color/ColorConvert_RGB_RGBX.gdf",
        "arithmetic/Add_U8_U8U8_Sat.gdf",
        "filter/Box_U8_U8_3x3.gdf",
        "geometric/ScaleImage_U8_U8_Bilinear.gdf",
        "geometric/Remap_U8_U8_Bilinear.gdf",
    ]
    return [gdf_root / p for p in default if (gdf_root / p).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenVX GDF tests under rocprofv3 and generate a markdown summary."
    )
    parser.add_argument(
        "--runvx",
        type=Path,
        required=True,
        help="Path to the runvx executable.",
    )
    parser.add_argument(
        "--backend",
        choices=["CPU", "HIP"],
        required=True,
        help="Backend affinity to use for runvx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store per-GDF rocprof traces.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        required=True,
        help="Path to write the markdown summary.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of frames per GDF (default: {DEFAULT_FRAMES}).",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="GDF category directories to run (default: a small representative set).",
    )
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=Path("/opt/rocm/bin/rocprofv3"),
        help="Path to rocprofv3 (default: /opt/rocm/bin/rocprofv3).",
    )
    parser.add_argument(
        "--gdf-root",
        type=Path,
        help="Override path to the amd_openvx_gdfs directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    rocprof = shutil.which(args.rocprof) or str(args.rocprof)
    if not Path(rocprof).is_file():
        log.error("rocprofv3 not found: %s", rocprof)
        return 1

    if not args.runvx.is_file():
        log.error("runvx not found: %s", args.runvx)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gdf_root:
        gdf_root = args.gdf_root
    else:
        gdf_root = args.runvx.parent.parent.parent / "tests" / "amd_openvx_gdfs"
    gdfs = discover_gdfs(gdf_root, args.category)
    if not gdfs:
        log.error("No GDF files discovered in %s", gdf_root)
        return 1

    affinity = "GPU" if args.backend == "HIP" else "CPU"
    results: List[Tuple[Path, bool, Optional[Path]]] = []
    failures = []

    for gdf in gdfs:
        log.info("Profiling %s on %s", gdf.name, args.backend)
        ok, csv_dir = run_rocprof(
            rocprof=Path(rocprof),
            runvx=args.runvx,
            gdf=gdf,
            affinity=affinity,
            frames=args.frames,
            output_dir=args.output_dir,
            backend=args.backend,
        )
        results.append((gdf, ok, csv_dir))
        if not ok:
            failures.append(gdf.name)

    summary = build_summary(
        backend=args.backend,
        gdf_results=results,
        frames=args.frames,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(summary)
    log.info("Summary written to %s", args.summary_out)

    # Also write a machine-readable JSON artifact for downstream jobs.
    json_out = args.summary_out.with_suffix(".json")
    json_data = {
        "backend": args.backend,
        "frames": args.frames,
        "results": [
            {
                "gdf": str(gdf),
                "ok": ok,
                "csv_dir": str(csv_dir) if csv_dir else None,
            }
            for gdf, ok, csv_dir in results
        ],
    }
    json_out.write_text(json.dumps(json_data, indent=2))

    if failures:
        log.error("Failures: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
