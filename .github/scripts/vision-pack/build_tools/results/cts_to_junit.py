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

"""Convert a Khronos OpenVX CTS log (vx_test_conformance) to JUnit XML.

The CTS (test_engine.c, openvx_1.3.2) prints::

    [ RUN 0001 ] GraphBase.vxCreateGraph ...          (--quiet: "[ RUN      ] ...")
    [     DONE ] GraphBase.vxCreateGraph (1.2 ms)      (no "(... ms)" with --show_test_duration=0)
    [ !FAILED! ] vxuCanny.DISABLED_BitExactL1/0/3x3 thresh=120 output=VX_DF_IMAGE_U8 (3.1 ms)
    [ !FAILED! ] Test setup                            (setup failure, followed by the real line)
    [ PASSED   ] 5824 test(s)                          (summary lines, not tests)
    [ FAILED   ] 3 test(s), listed below:  /  [ FAILED   ] <name>
    #REPORT: 20260925012033 ALL 15286 8195 5824 5824 5824 0 (version 1.3.2)
             <time> <ALL|FILTERED|testid> <total> <disabled> <started> <completed> <passed> <failed>

Test names can contain spaces, so names are taken up to " ..." / " (x ms)".
A test that has RUN but never DONE/FAILED crashed or timed out. The #REPORT
started/failed counts are cross-checked against the parsed results; a missing
or mismatched report, or an abnormal exit, adds a "#REPORT" error testcase, so
a truncated log can never pass.

    cts_to_junit.py --suite cts.GPU.baseline --log cts.log --rc <exit> --out cts.xml \
        [--summary s.json] [--names names.txt]
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET

RUN = re.compile(r"^\[ RUN(?: \d+)? +\] (.+?) \.\.\.\s*$")
END = re.compile(r"^\[ (    DONE|!FAILED!) \] (.+?)(?: \(([\d.]+) ms\))?\s*$")
REPORT = re.compile(r"^#REPORT: \S+ (\S+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")


def parse(lines: list[str]) -> tuple[dict[str, dict], dict | None]:
    state: dict[str, dict] = {}
    report = None
    cur = None
    for line in lines:
        if m := RUN.match(line):
            cur = m[1]
            state[cur] = {"status": "crash", "time": 0.0, "detail": []}
            continue
        if (m := END.match(line)) and not (m[1] == "!FAILED!" and m[2] == "Test setup"):
            st = state.setdefault(m[2], {"status": "crash", "time": 0.0, "detail": []})
            st["status"] = "pass" if m[1] == "    DONE" else "fail"
            if m[3]:
                st["time"] = float(m[3]) / 1000.0
            cur = None
            continue
        if m := REPORT.match(line):
            keys = ("total", "disabled", "started", "completed", "passed", "failed")
            report = {"id": m[1], **{k: int(v) for k, v in zip(keys, m.groups()[1:], strict=False)}}
            continue
        if cur is not None and len(state[cur]["detail"]) < 60:
            state[cur]["detail"].append(line)
    return state, report


def build(suite: str, state: dict[str, dict], report: dict | None, rc: int) -> tuple[ET.Element, dict]:
    ts = ET.Element("testsuite", name=suite)
    counts = {"pass": 0, "fail": 0, "crash": 0}
    for name, st in state.items():
        counts[st["status"]] += 1
        # classname == name so emit.py ingest-junit keeps the bare test name
        tc = ET.SubElement(ts, "testcase", classname=name, name=name, time=f"{st['time']:.3f}")
        detail = "\n".join(st["detail"]).strip()
        if st["status"] == "fail":
            ET.SubElement(tc, "failure", message="CTS failure").text = detail
        elif st["status"] == "crash":
            ET.SubElement(tc, "error", message="RUN without DONE (crash or timeout)").text = detail
    problems = []
    if report is None:
        problems.append("no #REPORT line (truncated log)")
    else:
        if report["started"] != len(state):
            problems.append(f"#REPORT started={report['started']} but parsed {len(state)} tests")
        if report["failed"] != counts["fail"]:
            problems.append(f"#REPORT failed={report['failed']} but parsed {counts['fail']} failures")
    if rc == 124 or rc == 137:
        problems.append(f"timeout (exit {rc})")
    elif rc > 128 or rc < 0:
        problems.append(f"killed by signal {rc - 128 if rc > 128 else -rc} (exit {rc})")
    elif rc not in (0, 1):
        problems.append(f"exit {rc}")
    elif rc == 1 and counts["fail"] == 0 and state:
        problems.append("exit 1 without any failed test")
    if not state and not problems:
        problems.append("no tests ran")
    if problems:
        tc = ET.SubElement(ts, "testcase", classname="#REPORT", name="#REPORT")
        ET.SubElement(tc, "error", message="; ".join(problems)[:500]).text = "; ".join(problems)
    summary = {"suite": suite, "parsed": len(state), **counts, "report": report, "rc": rc, "problems": problems}
    return ts, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--names", help="write the parsed test names (one per line) here")
    a = ap.parse_args()
    with open(a.log, errors="replace") as f:
        lines = f.read().splitlines()
    state, report = parse(lines)
    ts, summary = build(a.suite, state, report, a.rc)
    root = ET.Element("testsuites")
    root.append(ts)
    ET.indent(root)
    ET.ElementTree(root).write(a.out, encoding="utf-8", xml_declaration=True)
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump(summary, f, indent=1)
    if a.names:
        with open(a.names, "w") as f:
            f.write("".join(n + "\n" for n in state))
    print(f"{a.suite}: {len(state)} tests, pass={summary['pass']} fail={summary['fail']} crash={summary['crash']} "
          f"report={report} rc={a.rc} problems={summary['problems']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
