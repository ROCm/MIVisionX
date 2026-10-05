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

"""runVisionTests.py per target and size, with per-node results.

runVisionTests.py writes into its working directory (openvx_node_results/,
vision_report_*.md), so every invocation gets a fresh directory. It ignores
runvx exit codes (H16), so the verdict comes from nodePerformanceOutput.log:
each "Running OpenVX Node - N:Name" header starts a node section, and a node
passes when its section has the "<T>,GRAPH" profile line. A section with
ERROR lines is a fail; a section that just stops (the runvx child died, e.g.
the CPU Box3x3 segfault at small widths, H11) is an error.

IDs: mivisionx::vision.<T>.<size>::<node>
     mivisionx::vision.<T>.<size>::script-exit-honesty   (H16)
     mivisionx::vision.static::<check>                   (--lint, M19)
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from mvx_common import ROCM_PATH, TEST_ROOT, VP_OUT, load_vision_nodes, record, run, tail, write_log

SIZES = {"1080p": (1920, 1080), "5x3": (5, 3), "10x10": (10, 10)}
HEADER = re.compile(r"^Running OpenVX Node - (\d+):(\S+)")


def lint() -> None:
    """M19: static checks of the shipped case table."""
    nodes = load_vision_nodes(1920, 1080)
    rgba = [n for n, f in nodes if re.search(r",RGBA\b", f)]
    record("vision.static::rgba-format", "fail" if rgba else "pass",
           (f"{len(rgba)} cases use the image format 'RGBA', which runvx rejects (VX_ERROR_NO_RESOURCES): "
            f"{', '.join(rgba)}") if rgba else "")
    by_cmd: dict[str, list[str]] = {}
    for n, f in nodes:
        by_cmd.setdefault(f, []).append(n)
    dups = [v for v in by_cmd.values() if len(v) > 1]
    names = [n for n, c in Counter(n for n, _ in nodes).items() if c > 1]
    record("vision.static::duplicate-cases", "fail" if dups or names else "pass",
           "; ".join(["identical commands: " + " = ".join(v) for v in dups] +
                     (["duplicate names: " + ", ".join(names)] if names else [])))
    swapped = []
    for n, f in nodes:
        m = re.search(r"scalar:BOOL,(\d)", f)
        if "fast_corners" in f and m:
            suppress = m.group(1) == "1"
            if ("NoSupression" in n) == suppress:
                swapped.append(f"{n} uses nonmax_suppression={m.group(1)}")
    record("vision.static::case-names", "fail" if swapped else "pass", "; ".join(swapped))
    record("vision.static::node-count", "pass" if len(nodes) == 116 else "fail",
           f"{len(nodes)} cases (116 expected)")


def parse(log: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    cur = None
    for line in log.splitlines():
        m = HEADER.match(line)
        if m:
            cur = m.group(2)
            if cur in sections:
                cur = f"{cur}#{m.group(1)}"
            sections[cur] = []
        elif cur:
            sections[cur].append(line)
    return sections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("CPU", "GPU"))
    ap.add_argument("--size", choices=tuple(SIZES))
    ap.add_argument("--work", help="parent of the fresh scratch dir")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--python", default="python3")
    ap.add_argument("--lint", action="store_true")
    a = ap.parse_args()
    if a.lint:
        lint()
        return 0
    group = f"vision.{a.target}.{a.size}"
    w, h = SIZES[a.size]
    script = TEST_ROOT / "vision_tests" / "runVisionTests.py"
    scratch = Path(a.work) / group
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    cmd = [a.python, script, "--runvx_directory", ROCM_PATH / "bin", "--num_frames", "10"]
    if a.target == "GPU":
        cmd += ["--hardware_mode", "GPU", "--backend_type", "HIP"]
    if a.size != "1080p":
        cmd += ["--width", str(w), "--height", str(h)]
    r = run(cmd, a.timeout, cwd=scratch)
    log = VP_OUT / "logs" / (f"{group}.log")
    write_log(log, f"### cmd: {r.repro(cwd=scratch)}", r.out)
    node_log = scratch / "openvx_node_results" / "nodePerformanceOutput.log"
    expected = [n for n, _ in load_vision_nodes(w, h)]
    if not node_log.is_file():
        record(f"{group}::nodePerformanceOutput", "error",
               f"no nodePerformanceOutput.log ({r.why()}): {tail(r.out)}", r.dt, log, a.target, r.repro(cwd=scratch))
        return 0
    shutil.copy(node_log, VP_OUT / "logs" / (f"{group}.nodePerformanceOutput.log"))
    sections = parse(node_log.read_text(errors="replace"))
    bad = 0
    graph = re.compile(rf",{a.target},GRAPH\s*$", re.M)
    for name in dict.fromkeys(expected):
        body = sections.get(name)
        if body is None:
            status, msg = "error", "node never ran (no header in nodePerformanceOutput.log)"
        else:
            text = "\n".join(body)
            errs = [ln.strip() for ln in body if "ERROR" in ln]
            if graph.search(text):
                status, msg = "pass", ""
            elif errs:
                status, msg = "fail", " | ".join(errs[:3])
            else:
                status, msg = "error", "runvx output stops without a result (crash): " + tail(text, 300)
        bad += status != "pass"
        record(f"{group}::{name}", status, msg, 0, f"logs/{group}.nodePerformanceOutput.log", a.target,
               r.repro(cwd=scratch))
    extra = [n for n in sections if n not in expected]
    if extra:
        record(f"{group}::unexpected-nodes", "fail", f"headers not in the case table: {', '.join(extra)}",
               backend=a.target)
    if r.status() == "pass":
        record(f"{group}::script-exit-honesty", "fail" if bad else "pass",
               f"runVisionTests.py exited 0 although {int(bad)} of {len(expected)} nodes failed" if bad else "",
               r.dt, log, a.target, r.repro(cwd=scratch))
    else:
        record(f"{group}::script-exit-honesty", "pass" if bad else "fail",
               f"runVisionTests.py {r.why()} ({int(bad)} failing nodes)", r.dt, log, a.target, r.repro(cwd=scratch))
    print(f"{group}: {len(expected)} nodes, {int(bad)} not passing, script {r.why()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
