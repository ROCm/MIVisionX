# Copyright (c) 2020 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""Run OpenCV extension GDF tests using RunVX and generate a benchmark report.

This script:
  1. Runs OpenCV extension .gdf tests through the RunVX utility.
  2. Parses per-node profiling data (frame count, timing).
  3. Generates a Markdown benchmark report with system information.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import logging
import platform
import shutil
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2021 - 2024, AMD MIVisionX - AMD OpenCV EXT Vision Test Full Report"
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

log = logging.getLogger("runOpenCVTests")

GDF_TESTS = [
    "absdiff.gdf",
    "adaptiveThreshold.gdf",
    "add.gdf",
    "addWeighted.gdf",
    "bilateralFilter.gdf",
    "bitwise_and.gdf",
    "bitwise_not.gdf",
    "bitwise_or.gdf",
    "bitwise_xor.gdf",
    "blur.gdf",
    "boxFilter.gdf",
    "canny.gdf",
    "compare.gdf",
    "convertScaleAbs.gdf",
    "cornerMinEigenVal.gdf",
    "cornerharris.gdf",
    "cvBuildPyramid.gdf",
    "cvtColor.gdf",
    "dilate.gdf",
    "distanceTransform.gdf",
    "divide.gdf",
    "erode.gdf",
    "fastNlMeansDenoising.gdf",
    "fastNlMeansDenoisingColored.gdf",
    "filter2D.gdf",
    "flip.gdf",
    "gaussianBlur.gdf",
    "goodFeatures.gdf",
    "laplacian.gdf",
    "medianBlur.gdf",
    "morphologyEX.gdf",
    "multiply.gdf",
    "pyrDown.gdf",
    "pyrUp.gdf",
    "resize.gdf",
    "scharr.gdf",
    "sepFilter2D.gdf",
    "simple_blob.gdf",
    "sobel.gdf",
    "subtract.gdf",
    "threshold.gdf",
    "transpose.gdf",
    "warpAffine.gdf",
    "warpPerspective.gdf",
    "BRISK_Compute.gdf",
    "BRISK_Detect.gdf",
    "FAST.gdf",
    "MSER_Detect.gdf",
    "ORB_Compute.gdf",
    "ORB_Detect.gdf",
]  # type: List[str]

GDF_CONTRIB_TESTS = [
    "star_Detect.gdf",
    "SIFT_Compute.gdf",
    "SIFT_Detect.gdf",
    "SURF_Compute.gdf",
    "SURF_Detect.gdf",
]  # type: List[str]

DEFAULT_RUNVX_DIR = "/opt/rocm/bin"
DEFAULT_NUM_FRAMES = 1000
MAX_NUM_FRAMES = 10000

# type alias for a single benchmark row: (name, count, tmp, avg, min, max)
BenchmarkRow = Tuple[str, int, float, float, float, float]

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_darwin() -> bool:
    """Return True when running on macOS."""
    return platform.system() == "Darwin"


def get_platform_ip() -> str:
    """Best-effort local IP lookup; returns '' on failure."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


def tool_output(cmd: List[str]) -> str:
    """Run *cmd* and return its output, or a 'not found' message."""
    if shutil.which(cmd[0]) is None:
        return "%s: not found" % cmd[0]
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return (p.stdout or "").rstrip()


# ---------------------------------------------------------------------------
# Profiling parser
# ---------------------------------------------------------------------------


def parse_graph_profile(output: str) -> Optional[BenchmarkRow]:
    """Extract (name, count, tmp, avg, min, max) from the GRAPH profile line.

    Profiling lines look like::

        10,  0.293,  0.613,  0.293,  2.083,CPU,GRAPH
    """
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or "GRAPH" not in stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) >= 7 and parts[-1] == "GRAPH":
            try:
                return (
                    "",
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except (ValueError, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def write_formatted(output: str, f: object) -> None:
    """Write *output* inside a fenced code block."""
    f.write("````\n")  # type: ignore[union-attr]
    f.write("%s\n\n" % output)  # type: ignore[union-attr]
    f.write("````\n")  # type: ignore[union-attr]


BENCH_HEADER = (
    "| OpenVX Node                             "
    "|Frames Count| tmp (ms) | avg (ms) | min (ms) | max (ms) |"
)
BENCH_SEP = (
    "|-----------------------------------------"
    "|------------|----------|----------|----------|----------|"
)


def format_bench_row(row: BenchmarkRow) -> str:
    """Format a single benchmark row as a Markdown table line."""
    name, count, tmp, avg, min_ms, max_ms = row
    return (
        "| %-39s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |"
        % (name, count, tmp, avg, min_ms, max_ms)
    )


def write_system_report(
    *,
    report_path: Path,
    runvx_path: Path,
    benchmarks: List[BenchmarkRow],
) -> None:
    """Generate a Markdown system + benchmark report at *report_path*."""
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    fqdn = socket.getfqdn()
    ip = get_platform_ip()

    sys_info = tool_output(["inxi", "-c0", "-S"])
    cpu_info = tool_output(["inxi", "-c0", "-C"])
    gpu_info = tool_output(["inxi", "-c0", "-G"])
    memory_info = tool_output(["inxi", "-c0", "-m"])
    board_info = tool_output(["inxi", "-c0", "-M"])

    dep_cmd = ["otool", "-L"] if _is_darwin() else ["ldd"]
    lib_tree = tool_output(dep_cmd + [str(runvx_path)])

    with report_path.open("w", encoding="utf-8") as f:
        f.write("MIVisionX - OpenCV Extension Report\n")
        f.write("================================\n\n")
        f.write("Generated: %s\n\n" % report_dtstr)
        f.write("Platform: %s (%s)\n" % (fqdn, ip))
        f.write("--------\n\n")

        write_formatted(sys_info, f)
        write_formatted(cpu_info, f)
        write_formatted(gpu_info, f)
        write_formatted(board_info, f)
        write_formatted(memory_info, f)

        f.write("\n\nBenchmark Report\n")
        f.write("--------\n\n\n")
        f.write("%s\n" % BENCH_HEADER)
        f.write("%s\n" % BENCH_SEP)
        for row in benchmarks:
            f.write("%s\n" % format_bench_row(row))
        f.write("\n\n")

        f.write("Dynamic Libraries Report\n")
        f.write("-----------------\n\n")
        write_formatted(lib_tree, f)
        f.write("\n")
        f.write(
            "\n\n---\n**Copyright AMD ROCm MIVisionX 2021 - 2024"
            " -- runOpenCVTests.py V-%s**\n\n" % __version__
        )

    print("\nSTATUS: Output Report File - %s" % report_path.resolve())


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_tests(
    *,
    gdf_list: List[str],
    gdf_dir: Path,
    runvx: Path,
    num_frames: int,
    log_path: Path,
    show_runvx_raw_time: bool,
) -> Tuple[List[str], List[BenchmarkRow]]:
    """Run each GDF and return (failures, benchmark_rows)."""
    failures = []  # type: List[str]
    benchmarks = []  # type: List[BenchmarkRow]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        for i, gdf_name in enumerate(gdf_list, 1):
            gdf_path = gdf_dir / gdf_name
            if not gdf_path.is_file():
                log.warning("GDF not found, skipping: %s", gdf_path)
                continue

            header = "Running Test GDF - %s" % gdf_name
            print("%d/%d: %s" % (i, len(gdf_list), header))
            lf.write("%s\n" % header)

            cmd = [
                str(runvx),
                "-frames:%d" % num_frames,
                "-dump-profile",
                "file",
                str(gdf_path),
            ]
            log.debug("run: %s", " ".join(cmd))

            start = time.perf_counter()
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed_s = time.perf_counter() - start

            output = completed.stdout or ""
            for line in output.splitlines():
                if line.strip().startswith("> total elapsed time:"):
                    if show_runvx_raw_time:
                        print(line)
                    lf.write("%s\n" % line)
                    continue
                print(line)
                lf.write("%s\n" % line)

            print(
                "%s: total elapsed time - %.3f sec" % (gdf_name, elapsed_s)
            )
            lf.write(
                "%s: total elapsed time (harness) - %.3f sec\n"
                % (gdf_name, elapsed_s)
            )

            profile = parse_graph_profile(output)
            if profile:
                benchmarks.append(
                    (gdf_name, profile[1], profile[2], profile[3], profile[4], profile[5])
                )

            if completed.returncode != 0:
                desc = "%s (exit %d)" % (gdf_name, completed.returncode)
                log.error("FAILED: %s", desc)
                failures.append(desc)

            print()
            lf.write("\n")

    return failures, benchmarks


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and run OpenCV extension GDF tests."""
    parser = argparse.ArgumentParser(
        description="Run OpenCV extension GDF tests via RunVX and generate benchmark report.",
    )
    parser.add_argument(
        "--runvx_directory",
        type=str,
        default=DEFAULT_RUNVX_DIR,
        help="RunVX executable directory (default: %s)" % DEFAULT_RUNVX_DIR,
    )
    parser.add_argument(
        "--list_tests",
        type=str,
        default="no",
        help="List tests without running (default: no [options: no/yes])",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames per test (default: %d, range: 1-%d)"
        % (DEFAULT_NUM_FRAMES, MAX_NUM_FRAMES),
    )
    parser.add_argument(
        "--contrib",
        action="store_true",
        help="Include OpenCV contrib extension tests (SIFT, SURF, STAR)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--show-runvx-raw-time",
        action="store_true",
        help="Also show runvx rounded '> total elapsed time' lines (default: hidden)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # -- Validate arguments ---------------------------------------------------
    list_flag = args.list_tests.upper()
    if list_flag not in ("NO", "YES"):
        print("ERROR: --list_tests options supported - [no or yes]")
        return 2

    if not 1 <= args.num_frames <= MAX_NUM_FRAMES:
        print("ERROR: --num_frames not in range [1 - %d]" % MAX_NUM_FRAMES)
        return 2

    script_dir = Path(__file__).resolve().parent
    gdf_dir = script_dir / "gdfs"

    gdf_list = list(GDF_TESTS)
    if args.contrib:
        gdf_list.extend(GDF_CONTRIB_TESTS)

    # -- List mode ------------------------------------------------------------
    if list_flag == "YES":
        print("\nOpenCV Extension GDF Tests\n")
        print(" %-5s - %-30s" % ("Test ID", "Test Name"))
        for i, name in enumerate(gdf_list, 1):
            print("   %-5d - %-30s" % (i, name))
        return 0

    # -- Locate runvx ---------------------------------------------------------
    runvx = Path(args.runvx_directory) / "runvx"
    if not runvx.is_file():
        found = shutil.which("runvx")
        if found:
            runvx = Path(found)
            log.info("runvx not at %s; using PATH: %s", args.runvx_directory, runvx)
        else:
            print("ERROR: RunVX executable not found at %s or in PATH" % runvx)
            return 1

    print("\nMIVisionX runOpenCVTests V-%s\n" % __version__)
    print("STATUS: RunVX path - %s" % runvx)
    print("STATUS: Frames - %d  Tests - %d\n" % (args.num_frames, len(gdf_list)))

    # -- Prepare output directories -------------------------------------------
    results_dir = Path("gdf_test_results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    output_data_dir = gdf_dir / "amd_opencv_test_results"
    if output_data_dir.exists():
        shutil.rmtree(output_data_dir)
    output_data_dir.mkdir(parents=True)

    # -- Run tests ------------------------------------------------------------
    print("runOpenCVTests - OpenVX - OpenCV EXT Vision Tests\n")
    log_path = results_dir / "openCV_visionOutput.log"

    failures, benchmarks = run_tests(
        gdf_list=gdf_list,
        gdf_dir=gdf_dir,
        runvx=runvx,
        num_frames=args.num_frames,
        log_path=log_path,
        show_runvx_raw_time=args.show_runvx_raw_time,
    )

    # -- Benchmark table (to stdout + md file) --------------------------------
    print(BENCH_HEADER)
    print(BENCH_SEP)
    bench_md_path = results_dir / "openCV_visionOutput.md"
    with bench_md_path.open("w", encoding="utf-8") as f:
        f.write("%s\n" % BENCH_HEADER)
        f.write("%s\n" % BENCH_SEP)
        for row in benchmarks:
            line = format_bench_row(row)
            print(line)
            f.write("%s\n" % line)

    print("\nSTATUS: OpenCV EXT Vision Test Results - %s" % results_dir.resolve())

    # -- System + benchmark report --------------------------------------------
    platform_name = platform.platform()
    file_dtstr = datetime.now().strftime("%Y%m%d")
    report_path = Path(
        "opencv_vision_report_%s_%s.md" % (platform_name, file_dtstr)
    )
    write_system_report(
        report_path=report_path,
        runvx_path=runvx,
        benchmarks=benchmarks,
    )

    # -- Summary --------------------------------------------------------------
    if failures:
        log.error("%d test(s) failed:", len(failures))
        for f in failures:
            log.error("  %s", f)
        return 1

    print("\nrunOpenCVTests.py completed - V:%s\n" % __version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
