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

"""Static and runtime API probes for the shipped MIVisionX headers and libraries.

  api-probe.exports::<function>       declared in include/mivisionx/VX/*.h and exported by
                                      libopenvx/libvxu (M18: code compiles, then fails to link)
  api-probe::vx_ext_amd.h-compiles-as-C
  api-probe.<T>::rgba-create          VX_DF_IMAGE_RGBA is declared; creating one must work or
                                      fail with a format error, not VX_ERROR_NO_RESOURCES
  api-probe.<T>::copy-node            vxCopyNode creates, verifies and executes
  api-probe.<T>::threshold-readonly-attribute   setting VX_THRESHOLD_TYPE/INPUT_FORMAT fails
  api-probe.<T>::convolution-9x7      a 9x7 custom convolution that verifies also executes
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from mvx_common import ROCM_PATH, VP_OUT, record, run, tail, write_log

DECL = re.compile(r"VX_API_CALL\s+(\w+)\s*\(")
# OpenCL interop entry points do not apply to the HIP backend
NOT_APPLICABLE_HEADERS = {"vx_khr_opencl_interop.h", "vx_khr_icd.h"}
OK_FORMAT_ERRORS = {"VX_SUCCESS", "VX_ERROR_INVALID_FORMAT", "VX_ERROR_NOT_SUPPORTED"}


def exported_symbols(libs: list[Path]) -> set[str] | None:
    nm = shutil.which("nm") or str(ROCM_PATH / "lib" / "llvm" / "bin" / "llvm-nm")
    syms: set[str] = set()
    for lib in libs:
        r = run([nm, "-D", "--defined-only", lib.resolve()], 120)
        if r.rc != 0:
            return None
        syms.update(ln.split()[-1] for ln in r.out.splitlines() if len(ln.split()) >= 3 and ln.split()[1] in "TtWiV")
    return syms


def exports(log: Path) -> None:
    inc = ROCM_PATH / "include" / "mivisionx" / "VX"
    declared: dict[str, str] = {}
    for h in sorted(inc.glob("*.h")):
        for fn in DECL.findall(h.read_text(errors="replace")):
            declared.setdefault(fn, h.name)
    libs = [ROCM_PATH / "lib" / "libopenvx.so", ROCM_PATH / "lib" / "libvxu.so"]
    syms = exported_symbols(libs)
    if syms is None or not declared:
        record("api-probe.exports::scan", "error", f"could not read headers or symbol tables (declared={len(declared)})")
        return
    missing = []
    for fn, h in sorted(declared.items()):
        tid = f"api-probe.exports::{fn}"
        if fn in syms:
            record(tid, "pass", f"declared in {h}", log=log)
        elif h in NOT_APPLICABLE_HEADERS:
            record(tid, "skip", f"declared in {h} (not applicable to the HIP backend); not exported", log=log)
        else:
            missing.append(fn)
            record(tid, "fail", f"declared in VX/{h} but exported by neither libopenvx nor libvxu", log=log)
    write_log(log, f"### {len(declared)} declared, {len(missing)} not exported: {' '.join(missing)}")
    print(f"exports: {len(declared)} declared, {len(missing)} missing")


def c_header(work: Path, log: Path) -> None:
    src = work / "vx_ext_amd_c.c"
    src.write_text("#include <VX/vx.h>\n#include <vx_ext_amd.h>\nint main(void) { return 0; }\n")
    cc = ROCM_PATH / "lib" / "llvm" / "bin" / "amdclang"
    r = run([cc, "-x", "c", "-std=c99", "-fsyntax-only", f"-I{ROCM_PATH / 'include' / 'mivisionx'}", src], 120)
    write_log(log, f"### C compile of vx_ext_amd.h rc={int(r.rc)}", r.out)
    errs = [ln for ln in r.out.splitlines() if "error:" in ln][:3]
    record("api-probe::vx_ext_amd.h-compiles-as-C", r.status(),
           "" if r.status() == "pass" else f"{r.why()}: {' | '.join(errs) or tail(r.out)}", r.dt, log, "", r.repro())


def probes(work: Path, log: Path, timeout: float) -> None:
    exe = work / "api_probe"
    cxx = ROCM_PATH / "lib" / "llvm" / "bin" / "amdclang++"
    r = run([cxx, "-O1", "-std=c++17", f"-I{ROCM_PATH / 'include' / 'mivisionx'}", Path(__file__).with_name("api_probe.cpp"),
             "-o", exe, f"-L{ROCM_PATH / 'lib'}", "-lopenvx", f"-Wl,-rpath,{ROCM_PATH / 'lib'}"], 300)
    write_log(log, f"### build api_probe rc={int(r.rc)}", r.out)
    names = ("rgba-create", "copy-node", "threshold-readonly-attribute", "convolution-9x7")
    if r.status() != "pass":
        for t in ("CPU", "GPU"):
            for n in names:
                record(f"api-probe.{t}::{n}", "error", f"api_probe did not build: {tail(r.out, 400)}", backend=t)
        return
    for t in ("CPU", "GPU"):
        env = {"AGO_DEFAULT_TARGET": t}
        r = run([exe], timeout, cwd=work, env=env)
        write_log(log, f"### AGO_DEFAULT_TARGET={t} rc={int(r.rc)}", r.out)
        got = {m[0]: m[1].split() for m in re.findall(r"^PROBE (\S+) (.*)$", r.out, re.M)}
        for n in names:
            tid = f"api-probe.{t}::{n}"
            v = got.get(n)
            if v is None:
                record(tid, "error", f"probe produced no result ({r.why()}): {tail(r.out, 300)}", r.dt, log, t,
                       r.repro(env))
                continue
            if n == "rgba-create":
                ok = v[0] in OK_FORMAT_ERRORS
                msg = f"vxCreateImage(VX_DF_IMAGE_RGBA) -> {v[0]}"
            elif n == "copy-node":
                ok = v == ["VX_SUCCESS"] * 3
                msg = "create {}, verify {}, process {}".format(*tuple(v))
            elif n == "threshold-readonly-attribute":
                ok = "VX_SUCCESS" not in v
                msg = "set VX_THRESHOLD_TYPE -> {}, set VX_THRESHOLD_INPUT_FORMAT -> {} (read-only: must fail)".format(*tuple(v))
            else:
                ok = v[2] != "VX_SUCCESS" or v[3] == "VX_SUCCESS"
                msg = "coefficients {}, node {}, verify {}, process {}".format(*tuple(v))
            record(tid, "pass" if ok else "fail", msg, r.dt, log, t, r.repro(env))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    work = Path(a.work) / "api_probe"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    log = VP_OUT / "logs" / "api-probe.log"
    exports(log)
    c_header(work, log)
    probes(work, log, a.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
