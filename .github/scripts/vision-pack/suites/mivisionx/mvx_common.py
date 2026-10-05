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

"""Shared helpers for the mivisionx suite harnesses.

Records go straight to $VP_RESULTS through build_tools/results/emit.py, which
is much faster than one vp_result call per check for thousands of results.
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("VP_REPO", Path(__file__).resolve().parents[2])) / "build_tools" / "results"))
from emit import append_record  # noqa: E402

SUITE = "mivisionx"
ROCM_PATH = Path(os.environ.get("ROCM_PATH", ""))
VP_OUT = Path(os.environ.get("VP_OUT", "."))
RESULTS = os.environ.get("VP_RESULTS", str(VP_OUT / "results.jsonl"))
RUNVX = ROCM_PATH / "bin" / "runvx"
TEST_ROOT = ROCM_PATH / "share" / "mivisionx" / "test"


def record(test_id: str, status: str, message: str = "", duration: float = 0.0, log: str | Path = "",
           backend: str = "", repro: str = "", attempts: int = 1) -> None:
    log = str(log)
    out = str(VP_OUT) + "/"
    if log.startswith(out):
        log = log[len(out):]
    append_record(RESULTS, SUITE, test_id, status, duration_s=duration, message=message, log=log,
                  backend=backend, repro=repro, attempts=attempts)


class Run:
    __slots__ = ("rc", "out", "dt", "timed_out", "cmd")

    def __init__(self, rc: int, out: str, dt: float, timed_out: bool, cmd: list[str]):
        self.rc, self.out, self.dt, self.timed_out, self.cmd = rc, out, dt, timed_out, cmd

    @property
    def signal(self) -> int:
        return -self.rc if self.rc < 0 else 0

    def status(self) -> str:
        """pass/fail/error from the exit status alone (callers refine 'pass')."""
        if self.timed_out or self.rc < 0:
            return "error"
        return "pass" if self.rc == 0 else "fail"

    def why(self) -> str:
        if self.timed_out:
            return "timeout"
        if self.rc < 0:
            try:
                return f"killed by {signal.Signals(-self.rc).name}"
            except ValueError:
                return f"killed by signal {-self.rc}"
        return f"exit {self.rc}"

    def repro(self, env: dict | None = None, cwd: str | Path | None = None) -> str:
        pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
        cmd = " ".join(shlex.quote(c) for c in self.cmd)
        s = (pre + " " + cmd).strip()
        return f"(cd {shlex.quote(str(cwd))} && {s})" if cwd else s


def run(cmd: list, timeout: float, cwd: str | Path | None = None, env: dict | None = None) -> Run:
    """Run with a hard timeout; the child gets its own process group so a hang is fully killed."""
    cmd = [str(c) for c in cmd]
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        timed_out = True
    return Run(p.returncode, out.decode("utf-8", "replace"), time.perf_counter() - t0, timed_out, cmd)


def tail(text: str, n: int = 1200) -> str:
    return re.sub(r"\s+", " ", text[-n:]).strip()


def error_lines(text: str, n: int = 3) -> str:
    return " | ".join([ln.strip() for ln in text.splitlines() if "ERROR" in ln][:n])


def write_log(path: Path, *chunks: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(c if c.endswith("\n") else c + "\n")
    return path


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", s)[:160]


def load_vision_nodes(width: int, height: int, script: Path | None = None) -> list[tuple[str, str]]:
    """The openvxNodes table of the installed runVisionTests.py, evaluated for one size."""
    src = (script or TEST_ROOT / "vision_tests" / "runVisionTests.py").read_text()
    start = src.index("openvxNodes = [")
    i = src.index("[", start)
    depth = 0
    end = None
    for j in range(i, len(src)):
        if src[j] == "[":
            depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    ns = {"width": width, "height": height, "widthDiv2": width // 2, "heightDiv2": height // 2,
          "str": str, "int": int}
    exec(src[start:end], ns)  # noqa: S102 - evaluates the shipped test table, nothing else
    return ns["openvxNodes"]
