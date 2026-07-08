#!/usr/bin/env python3
"""
CTest helper: verify MIVISIONX_HIP_CU_COUNT limits CU usage for a 4K Remap graph.

Runs the same 4K Remap GDF under different CU limits and checks that:
  - each run completes successfully,
  - the CU-limiting INFO log appears when the env var is set,
  - the default (unset) path does not show the limiting log.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run_runvx(runvx_exe, gdf_path, cu_count):
    """Run runvx with the given CU count. cu_count=None means default/all."""
    env = os.environ.copy()
    if cu_count is not None:
        env["MIVISIONX_HIP_CU_COUNT"] = str(cu_count)
    else:
        env.pop("MIVISIONX_HIP_CU_COUNT", None)

    cmd = [str(runvx_exe), "-frames:1", "-dump-profile", str(gdf_path)]
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result


def check_output(result, cu_count, expect_limit):
    """Return (ok, message) tuple."""
    if result.returncode != 0:
        return False, f"runvx exited with code {result.returncode}\n{result.stdout[-1500:]}"

    combined = result.stdout + result.stderr.decode() if result.stderr else result.stdout
    has_limit_log = re.search(r"INFO: limiting HIP graph stream to", combined) is not None

    if expect_limit and not has_limit_log:
        return False, f"expected CU-limit INFO log for count={cu_count}, but none found"
    if not expect_limit and has_limit_log:
        return False, f"unexpected CU-limit INFO log when running with default/all CUs"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="HIP CU-mask control test for Remap 4K")
    parser.add_argument("--runvx", required=True, help="path to runvx executable")
    parser.add_argument("--gdf", required=True, help="path to Remap 4K GDF file")
    parser.add_argument("--cu-counts", default="2,4,8,16,32",
                        help="comma-separated CU counts to test (default: 2,4,8,16,32)")
    args = parser.parse_args()

    runvx_exe = Path(args.runvx)
    gdf_path = Path(args.gdf)

    if not runvx_exe.exists():
        print(f"ERROR: runvx not found: {runvx_exe}", file=sys.stderr)
        return 1
    if not gdf_path.exists():
        print(f"ERROR: GDF not found: {gdf_path}", file=sys.stderr)
        return 1

    cu_counts = []
    has_all = False
    for c in args.cu_counts.split(","):
        c = c.strip()
        if c.lower() == "all":
            has_all = True
        else:
            cu_counts.append(int(c))

    # Always test the default/unset path once; if "all" was given, it represents this case.
    test_cases = [(None, False)]  # (cu_count, expect_limit_log)
    for cu in cu_counts:
        test_cases.append((cu, True))

    failures = []
    for cu, expect_limit in test_cases:
        label = "default" if cu is None else f"cu={cu}"
        result = run_runvx(runvx_exe, gdf_path, cu)
        ok, msg = check_output(result, cu, expect_limit)
        if not ok:
            failures.append((label, msg))
            print(f"FAIL [{label}]: {msg.splitlines()[0]}")
        else:
            # Show the relevant INFO line when present.
            m = re.search(r"INFO: limiting HIP graph stream to .*", result.stdout)
            info = f" -- {m.group(0)}" if m else ""
            print(f"PASS [{label}]{info}")

    if failures:
        print(f"\n{len(failures)} test(s) failed.", file=sys.stderr)
        return 1

    print("\nAll CU-mask control tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
