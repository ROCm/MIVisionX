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

"""Shared verdict logic for the robustness probes.

A probe observes one outcome and the caller chooses what is acceptable:

  outcome  correct       the operation completed and its output matches the reference
           wrong         the operation reported success but the output is wrong (silent failure)
           clean_error   the operation reported an error (exception or non-zero status) and exited normally
           exit0_error   the library reported an error but terminated the process with status 0
  mode     correct       only "correct" passes (supported device, CPU paths)
           honest        "correct" or "clean_error" passes (unsupported GPU: work or say so)
           error         only "clean_error" passes (a backend that cannot exist here must refuse)

Probe exit status: 0 pass, 1 fail, 70 a child process crashed (checked_run.py --error-rc 70 maps it to error).
"""
from __future__ import annotations

import sys

ACCEPT = {
    "correct": {"correct"},
    "honest": {"correct", "clean_error"},
    "error": {"clean_error"},
}
CRASH_RC = 70


def finish(mode: str, outcome: str, detail: str = "") -> None:
    ok = outcome in ACCEPT[mode]
    print(f"OUTCOME: {outcome}{(' - ' + detail) if detail else ''}", flush=True)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (mode={mode}, accepted={sorted(ACCEPT[mode])})", flush=True)
    sys.exit(0 if ok else 1)


def crashed(detail: str) -> None:
    print(f"OUTCOME: crash - {detail}", flush=True)
    sys.exit(CRASH_RC)
