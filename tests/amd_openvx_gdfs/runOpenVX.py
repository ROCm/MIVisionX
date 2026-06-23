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

"""Run OpenVX GDF node tests using the RunVX utility.

This script discovers .gdf test files across multiple node categories
(arithmetic, color, filter, geometric, logical, statistical, vision,
vision_profile, cpu) and executes each through RunVX with configurable
frame count, backend affinity, and profiling output.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse
import logging
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

__copyright__ = "Copyright 2018 - 2025, AMD MIVisionX - OpenVX GDF Test"
__license__ = "MIT"
__version__ = "1.4.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

log = logging.getLogger("runOpenVX")

# (display label, subdirectory relative to script)
TEST_CATEGORIES = [
    ("Arithmetic", "arithmetic"),
    ("Color", "color"),
    ("Filter", "filter"),
    ("Geometric", "geometric"),
    ("Logical", "logical"),
    ("Statistical", "statistical"),
    ("Vision", "vision"),
    ("Vision Apps", "vision_profile"),
    ("CPU", "cpu"),
]  # type: List[Tuple[str, str]]

HIDDEN_CATEGORY = ("Hidden CPU", "cpu/hidden")  # type: Tuple[str, str]

VALID_BACKENDS = ("CPU", "HIP", "OCL", "GPU")
DEFAULT_RUNVX_DIR = "/opt/rocm/bin"
DEFAULT_NUM_FRAMES = 100
MAX_NUM_FRAMES = 10000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_gdf_files(directory: Path) -> List[str]:
    """Return sorted list of ``.gdf`` filenames in *directory*, or [] if missing.

    Only ``.gdf`` files are returned so stray artifacts that may appear in the
    directory at runtime (for example ``default.profraw`` emitted by a
    coverage-instrumented ``runvx``) are never mistaken for a GDF test.
    """
    if not directory.is_dir():
        log.warning("Directory not found: %s", directory)
        return []
    return sorted(f.name for f in directory.iterdir() if f.is_file() and f.suffix == ".gdf")


def print_test_list(
    script_dir: Path,
    *,
    include_hidden: bool,
) -> None:
    """Print all discovered GDF test names grouped by category."""
    categories = list(TEST_CATEGORIES)
    if include_hidden:
        categories.append(HIDDEN_CATEGORY)
    for label, subdir in categories:
        names = get_gdf_files(script_dir / subdir)
        if not names:
            continue
        print("\n%s Node Tests\n" % label)
        print(" %-5s - %-30s" % ("Test ID", "GDF Name"))
        for i, name in enumerate(names, 1):
            print("   %-5d - %-30s" % (i, name))


def run_gdf_category(
    *,
    label: str,
    gdf_dir: Path,
    runvx: Path,
    num_frames: int,
    affinity: str,
    error_check: bool,
    show_runvx_raw_time: bool,
) -> List[str]:
    """Run every GDF in *gdf_dir* and return a list of failure descriptions."""
    names = get_gdf_files(gdf_dir)
    failures = []  # type: List[str]
    for i, name in enumerate(names, 1):
        gdf_path = gdf_dir / name
        print("%s: Running GDF - %d:%s" % (label, i, name))
        cmd = [
            str(runvx),
            "-dump-gdf",
            "-frames:%d" % num_frames,
            "-affinity:%s" % affinity,
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

        # runvx prints "total elapsed time" with coarse rounding; show precise wall time.
        for line in (completed.stdout or "").splitlines():
            if line.strip().startswith("> total elapsed time:"):
                if show_runvx_raw_time:
                    print(line)
                continue
            print(line)
        print(
            "%s:%s: total elapsed time (harness) - %.3f sec"
            % (label, name, elapsed_s)
        )
        rc = completed.returncode
        if rc != 0:
            desc = "%s/%s (exit %d)" % (label, name, rc)
            if error_check:
                log.error("FAILED: %s", desc)
                failures.append(desc)
            else:
                log.warning("Expected failure: %s", desc)
        print()
    return failures


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and run OpenVX GDF tests."""
    parser = argparse.ArgumentParser(
        description="Run OpenVX GDF node tests via RunVX.",
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
        help="List GDF tests without running (default: no [options: no/yes])",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames per test (default: %d, range: 1-%d)"
        % (DEFAULT_NUM_FRAMES, MAX_NUM_FRAMES),
    )
    parser.add_argument(
        "--backend_type",
        type=str,
        default="CPU",
        help="Backend type (default: CPU [options: CPU/HIP/OCL/GPU])",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default="no",
        help="Include hidden (expected-failure) tests (default: no [options: no/yes])",
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

    backend_type = args.backend_type.upper()
    if backend_type not in VALID_BACKENDS:
        print("ERROR: OpenVX Backends supported - %s" % list(VALID_BACKENDS))
        return 2

    if not 1 <= args.num_frames <= MAX_NUM_FRAMES:
        print("ERROR: --num_frames not in range [1 - %d]" % MAX_NUM_FRAMES)
        return 2

    hidden_flag = args.hidden.upper()
    if hidden_flag not in ("NO", "YES"):
        print("ERROR: --hidden options supported - [no or yes]")
        return 2

    include_hidden = hidden_flag == "YES"
    script_dir = Path(__file__).resolve().parent

    # -- List mode ------------------------------------------------------------
    if list_flag == "YES":
        print_test_list(script_dir, include_hidden=include_hidden)
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

    affinity = "CPU" if backend_type == "CPU" else "GPU"

    print("\nMIVisionX runOpenVX V-%s\n" % __version__)
    print("STATUS: RunVX path - %s" % runvx)
    print(
        "STATUS: Backend - %s  Affinity - %s  Frames - %d\n"
        % (backend_type, affinity, args.num_frames)
    )

    # -- Run tests ------------------------------------------------------------
    print("runOpenVX - OpenVX Vision GDF Tests\n")
    all_failures = []  # type: List[str]

    for label, subdir in TEST_CATEGORIES:
        failures = run_gdf_category(
            label=label,
            gdf_dir=script_dir / subdir,
            runvx=runvx,
            num_frames=args.num_frames,
            affinity=affinity,
            error_check=True,
            show_runvx_raw_time=args.show_runvx_raw_time,
        )
        all_failures.extend(failures)

    if include_hidden:
        run_gdf_category(
            label=HIDDEN_CATEGORY[0],
            gdf_dir=script_dir / HIDDEN_CATEGORY[1],
            runvx=runvx,
            num_frames=args.num_frames,
            affinity=affinity,
            error_check=False,
            show_runvx_raw_time=args.show_runvx_raw_time,
        )

    # -- Summary --------------------------------------------------------------
    if all_failures:
        log.error("%d test(s) failed:", len(all_failures))
        for f in all_failures:
            log.error("  %s", f)
        return 1

    print("\nrunOpenVX.py completed - V:%s\n" % __version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
