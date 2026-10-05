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

"""Run one command under a timeout and record a single result with pass criteria vp_run cannot express.

    checked_run.py --id <suite>::<group>::<name> [--timeout S] [--cwd DIR] [--env K=V]... [--backend B]
                   [--expect-rc N | --expect-nonzero] [--skip-rc N] [--require REGEX]... [--forbid REGEX]...
                   [--blocked-if REGEX] [--why TEXT] -- cmd args...

Status, in order of precedence:
  error    timeout, the child was killed by a signal (a crash is never a clean result), it exited with an
           --error-rc code, or --error-if matched (the controlled case no longer behaves as designed);
  blocked  --blocked-if matched the output (e.g. a stand-in module was touched);
  skip     the exit code equals --skip-rc (upstream SKIP_RETURN_CODE convention, 77);
  fail     wrong exit code, a --require pattern is missing or a --forbid pattern is present;
  pass     otherwise.
--expect-nonzero accepts any clean exit 1..127 (used by exit-code honesty checks, where exit 0 on a
failure is the bug). The command's output goes to $VP_OUT/logs/<slug>.log; the record goes to
$VP_RESULTS through build_tools/results/emit.py. Always exits 0 unless its own arguments are wrong.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def slug(test_id: str) -> str:
    s = test_id.replace("::", "__")
    s = re.sub(r"[^A-Za-z0-9._+-]", "_", s)
    return s[:180]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--cwd")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--backend", default="")
    ap.add_argument("--expect-rc", type=int, default=0)
    ap.add_argument("--expect-nonzero", action="store_true")
    ap.add_argument("--skip-rc", type=int)
    ap.add_argument("--require", action="append", default=[])
    ap.add_argument("--forbid", action="append", default=[])
    ap.add_argument("--blocked-if")
    ap.add_argument("--error-if", help="output pattern meaning the check itself is invalid (status error)")
    ap.add_argument("--error-rc", type=int, action="append", default=[],
                    help="exit code a probe uses to report a crash of its child (status error)")
    ap.add_argument("--pass-if-file", help="with --expect-nonzero: exit 0 still passes if this glob (in --cwd) matches")
    ap.add_argument("--why", default="", help="context prefixed to the result message")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given")

    out_dir = Path(os.environ["VP_OUT"])
    suite = os.environ["VP_SUITE"]
    sys.path.insert(0, str(Path(os.environ["VP_REPO"]) / "build_tools" / "results"))
    from emit import append_record

    scale = float(os.environ.get("VP_TIMEOUT_SCALE", "1") or 1)
    timeout = max(1.0, a.timeout * scale)
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = out_dir / "logs" / f"{slug(a.id)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    repro = " ".join([shlex.quote(e) for e in a.env] + [shlex.quote(c) for c in cmd])
    if a.cwd:
        repro = f"(cd {shlex.quote(a.cwd)} && {repro})"

    t0 = time.monotonic()
    timed_out = False
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"### {a.id} (timeout {timeout:.0f}s)\n### cwd: {a.cwd or os.getcwd()}\n### cmd: {repro}\n")
        lf.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdin=subprocess.DEVNULL, stdout=lf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            lf.write(f"### cannot start: {e}\n")
            append_record(os.environ["VP_RESULTS"], suite, a.id, "error", message=f"cannot start: {e}",
                          log=str(log.relative_to(out_dir)), backend=a.backend, repro=repro)
            return 0
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
        # A child that leaves background processes behind must not outlive the check.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        lf.write(f"\n### exit {rc}{' (timeout)' if timed_out else ''}\n")
    dur = time.monotonic() - t0
    text = log.read_text(encoding="utf-8", errors="replace").split("\n", 3)[-1]

    missing = [r for r in a.require if not re.search(r, text, re.M)]
    present = [r for r in a.forbid if re.search(r, text, re.M)]
    if timed_out:
        status, msg = "error", f"timeout after {timeout:.0f}s"
    elif rc < 0:
        status, msg = "error", f"killed by signal {-rc} ({signal.Signals(-rc).name})"
    elif rc in a.error_rc:
        status, msg = "error", f"exit {rc}: the probed process crashed"
    elif a.blocked_if and re.search(a.blocked_if, text, re.M):
        status, msg = "blocked", f"output matched {a.blocked_if!r}"
    elif a.error_if and re.search(a.error_if, text, re.M):
        status, msg = "error", f"check invalid: output matched {a.error_if!r}"
    elif a.skip_rc is not None and rc == a.skip_rc:
        status, msg = "skip", f"exit {rc} (skip return code)"
    elif a.expect_nonzero and rc == 0 and a.pass_if_file and list(Path(a.cwd or ".").glob(a.pass_if_file)):
        status, msg = "pass", ""
    elif a.expect_nonzero and rc == 0:
        status, msg = "fail", "exit 0 although the command was expected to report a failure"
    elif a.expect_nonzero and rc > 128:
        status, msg = "error", f"exit {rc}: crashed instead of failing cleanly"
    elif not a.expect_nonzero and rc != a.expect_rc:
        status, msg = "fail", f"exit {rc} (expected {a.expect_rc})"
    elif missing:
        status, msg = "fail", "missing expected output: " + "; ".join(missing)
    elif present:
        status, msg = "fail", "unexpected output: " + "; ".join(present)
    else:
        status, msg = "pass", ""
    if status in ("fail", "error", "blocked"):
        tail = " ".join(text[-1500:].replace(f"### exit {rc}", "").split())
        msg = f"{msg}; {tail}"
    if a.why and status != "pass":
        msg = f"{a.why}: {msg}"
    append_record(os.environ["VP_RESULTS"], suite, a.id, status, duration_s=dur, message=msg,
                  log=str(log.relative_to(out_dir)), backend=a.backend, repro=repro)
    print(f"[checked_run] {a.id}: {status} ({dur:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
