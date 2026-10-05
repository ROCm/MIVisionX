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

"""Result records for the vision-pack QA suites.

Canonical format: one JSON object per line in ``<suite out>/results.jsonl``::

    {"suite": "roccv", "id": "roccv::ctest::test_op_flip", "status": "pass",
     "duration_s": 1.2, "message": "", "log": "logs/x.log", "backend": "GPU",
     "attempts": 1, "repro": ""}

``id`` is ``<suite>::<group>::<name>``. ``status`` is one of
pass | fail | error | skip | blocked | flaky:

- fail: the check ran and produced a wrong result or a non-zero exit;
- error: crash, signal, timeout, or the check could not complete;
- skip: not applicable (e.g. a combination the test itself rejects);
- blocked: could not run because a dependency or data set is missing;
- flaky: failed on the first attempt and passed on the retry.

Suites never apply known-issue baselines; report/triage.py does that.

Subcommands::

    emit.py record       --out F --suite S --id ID --status ST [--message M] [...]
    emit.py ingest-junit --out F --suite S --group G first.xml [--rerun rerun.xml]
    emit.py to-junit     --in F --out junit.xml
    emit.py summary      --in F --out summary.json

The module is also importable (``append_record``, ``read_records``) so Python
harnesses can write records without spawning a process per result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = ("pass", "fail", "error", "skip", "blocked", "flaky")
MAX_MESSAGE = 4000


def normalize_id(suite: str, test_id: str) -> str:
    return test_id if test_id.startswith(f"{suite}::") else f"{suite}::{test_id}"


def make_record(suite: str, test_id: str, status: str, *, duration_s: float = 0.0, message: str = "",
                log: str = "", backend: str = "", attempts: int = 1, repro: str = "") -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {STATUSES}")
    return {
        "suite": suite,
        "id": normalize_id(suite, test_id),
        "status": status,
        "duration_s": round(float(duration_s or 0.0), 3),
        "message": (message or "")[:MAX_MESSAGE],
        "log": log or "",
        "backend": backend or "",
        "attempts": int(attempts or 1),
        "repro": repro or "",
    }


def append_record(path: str | os.PathLike, suite: str, test_id: str, status: str, **fields) -> dict:
    rec = make_record(suite, test_id, status, **fields)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_records(path: str | os.PathLike) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"warning: {path}:{n}: bad JSON ({e})", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# JUnit ingestion (ctest --output-junit, pytest --junitxml, cts_to_junit.py)
# ---------------------------------------------------------------------------

def _junit_cases(path: str) -> list[dict]:
    """Return [{name, status, message, time}] for every testcase in a JUnit file."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        return [{"name": f"junit-unreadable:{Path(path).name}", "status": "error",
                 "message": f"cannot parse {path}: {e}", "time": 0.0}]
    cases = []
    for tc in root.iter("testcase"):
        name = tc.get("name", "?")
        classname = tc.get("classname", "")
        # pytest: classname is the dotted module (+class); keep it for uniqueness.
        # ctest: classname repeats the test name, so drop it.
        if classname and classname != name:
            full = f"{classname}::{name}"
        else:
            full = name
        status, message = "pass", ""
        fail = tc.find("failure")
        err = tc.find("error")
        skipped = tc.find("skipped")
        if fail is not None:
            status = "fail"
            message = (fail.get("message") or "") + "\n" + (fail.text or "")
        elif err is not None:
            status = "error"
            message = (err.get("message") or "") + "\n" + (err.text or "")
        elif skipped is not None:
            status = "skip"
            message = skipped.get("message") or skipped.text or ""
        # ctest marks not-run tests with status="notrun" (and sometimes disabled).
        if tc.get("status") in ("notrun", "disabled") and status == "pass":
            status = "error" if tc.get("status") == "notrun" else "skip"
            message = message or f"ctest status {tc.get('status')}"
        try:
            t = float(tc.get("time") or 0.0)
        except ValueError:
            t = 0.0
        cases.append({"name": full, "status": status, "message": message.strip(), "time": t})
    return cases


def ingest_junit(out: str, suite: str, group: str, first: str, rerun: str | None = None,
                 backend: str = "", log: str = "") -> list[dict]:
    first_cases = _junit_cases(first)
    rerun_map = {c["name"]: c for c in _junit_cases(rerun)} if rerun and Path(rerun).exists() else {}
    records = []
    for c in first_cases:
        status, attempts, message = c["status"], 1, c["message"]
        if status in ("fail", "error") and c["name"] in rerun_map:
            attempts = 2
            second = rerun_map[c["name"]]
            if second["status"] == "pass":
                status = "flaky"
                message = "passed on retry; first attempt: " + message
            else:
                message = second["message"] or message
        records.append(append_record(out, suite, f"{group}::{c['name']}", status, duration_s=c["time"],
                                     message=message, backend=backend, attempts=attempts, log=log))
    return records


# ---------------------------------------------------------------------------
# JUnit emission
# ---------------------------------------------------------------------------

def to_junit(records: list[dict], out: str, suite_name: str | None = None) -> None:
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_suite[suite_name or r.get("suite", "suite")].append(r)
    root = ET.Element("testsuites")
    for sname, recs in by_suite.items():
        counts = Counter(r["status"] for r in recs)
        ts = ET.SubElement(root, "testsuite", name=sname, tests=str(len(recs)),
                           failures=str(counts["fail"]), errors=str(counts["error"]),
                           skipped=str(counts["skip"] + counts["blocked"]),
                           time=f"{sum(r.get('duration_s', 0) for r in recs):.3f}")
        for r in recs:
            rid = r["id"]
            parts = rid.split("::")
            classname = "::".join(parts[:-1]) if len(parts) > 1 else sname
            tc = ET.SubElement(ts, "testcase", classname=classname, name=parts[-1],
                               time=f"{r.get('duration_s', 0):.3f}")
            msg = r.get("message", "")
            if r["status"] == "fail":
                ET.SubElement(tc, "failure", message=msg.splitlines()[0][:500] if msg else "failed").text = msg
            elif r["status"] == "error":
                ET.SubElement(tc, "error", message=msg.splitlines()[0][:500] if msg else "error").text = msg
            elif r["status"] in ("skip", "blocked"):
                ET.SubElement(tc, "skipped", message=f"{r['status']}: {msg}"[:500])
            elif r["status"] == "flaky":
                ET.SubElement(tc, "system-out").text = f"FLAKY: {msg}"
            props = ET.SubElement(tc, "properties")
            for key in ("backend", "log", "repro", "attempts"):
                if r.get(key):
                    ET.SubElement(props, "property", name=key, value=str(r[key]))
    ET.indent(root)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)


def summary(records: list[dict]) -> dict:
    counts = Counter(r["status"] for r in records)
    return {"total": len(records), "counts": {s: counts.get(s, 0) for s in STATUSES}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record")
    p.add_argument("--out", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--status", required=True, choices=STATUSES)
    p.add_argument("--message", default="")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--log", default="")
    p.add_argument("--backend", default="")
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--repro", default="")

    p = sub.add_parser("ingest-junit")
    p.add_argument("--out", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--rerun")
    p.add_argument("--backend", default="")
    p.add_argument("--log", default="")
    p.add_argument("junit")

    p = sub.add_parser("to-junit")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suite-name")

    p = sub.add_parser("summary")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out")

    a = ap.parse_args(argv)
    if a.cmd == "record":
        append_record(a.out, a.suite, a.id, a.status, duration_s=a.duration, message=a.message, log=a.log,
                      backend=a.backend, attempts=a.attempts, repro=a.repro)
    elif a.cmd == "ingest-junit":
        recs = ingest_junit(a.out, a.suite, a.group, a.junit, a.rerun, a.backend, a.log)
        s = summary(recs)
        print(f"ingested {s['total']} testcases from {a.junit}: {s['counts']}")
    elif a.cmd == "to-junit":
        to_junit(read_records(a.inp), a.out, a.suite_name)
    elif a.cmd == "summary":
        s = summary(read_records(a.inp))
        if a.out:
            Path(a.out).write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        c = s["counts"]
        print(f"{s['total']} results: " + ", ".join(f"{k}={v}" for k, v in c.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
