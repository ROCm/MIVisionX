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

"""Turn suites/suites.yaml into the job matrices of the QA workflow.

    plan_matrix.py --tier comprehensive [--suites rocal,roccv] [--has-cpu-runner true]
                   [--extended false] [--github]

Outputs JSON objects ``{"include": [...]}`` for the self-hosted jobs, split by
runner label, plus the list of hosted suites and every expected suite (for the
report's missing-suite check). ``nogpu_matrix`` holds the planned suites with
``needs_gpu: false`` (with ``gpu_access: none``), which run instead of both
self-hosted matrices when the runner has no usable GPU;
``gpu_required`` lists the planned suites that cannot. With --github they are
written to $GITHUB_OUTPUT as gpu_matrix, cpu_matrix, nogpu_matrix, hosted,
expected, gpu_required, has_gpu, has_cpu and has_nogpu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

TIERS = ["quick", "standard", "comprehensive", "full"]


def plan(cfg: dict, tier: str, only: set[str], has_cpu_runner: bool, extended: bool) -> dict:
    if tier not in TIERS:
        raise SystemExit(f"unknown tier {tier!r}; expected one of {TIERS}")
    gpu, cpu, nogpu, hosted, expected, gpu_required = [], [], [], [], [], []
    for name, s in cfg["suites"].items():
        if only and name not in only:
            continue
        if tier not in s.get("tiers", []):
            continue
        expected.append(name)
        runner = s.get("runner", "gpu")
        if runner == "hosted":
            hosted.append(name)
            continue
        image = s.get("image", "test")
        if extended and tier == "full":
            image = "extended"
        access = s.get("gpu_access", "chosen")
        if access not in ("chosen", "all", "none"):
            raise SystemExit(f"suite {name}: gpu_access must be chosen, all or none (got {access!r})")
        needs_gpu = s.get("needs_gpu", True)
        if not isinstance(needs_gpu, bool):
            raise SystemExit(f"suite {name}: needs_gpu must be true or false (got {needs_gpu!r})")
        entry = {
            "suite": name,
            "timeout": int(s.get("timeout_minutes", 60)),
            "image": image,
            "entrypoint": s.get("entrypoint", f"suites/{name}/run.sh"),
            "gpu_access": access,
        }
        if runner == "cpu" and has_cpu_runner:
            cpu.append(entry)
        else:
            gpu.append(entry)
        if needs_gpu:
            gpu_required.append(name)
        else:
            nogpu.append({**entry, "gpu_access": "none"})
    return {"gpu_matrix": {"include": gpu}, "cpu_matrix": {"include": cpu},
            "nogpu_matrix": {"include": nogpu},
            "hosted": hosted, "expected": expected, "gpu_required": gpu_required,
            "has_gpu": bool(gpu), "has_cpu": bool(cpu), "has_nogpu": bool(nogpu)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "suites" / "suites.yaml"))
    ap.add_argument("--tier", default="comprehensive")
    ap.add_argument("--suites", default="")
    ap.add_argument("--has-cpu-runner", default="false")
    ap.add_argument("--extended", default="false")
    ap.add_argument("--github", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    only = {s.strip() for s in a.suites.split(",") if s.strip()}
    unknown = only - set(cfg["suites"])
    if unknown:
        print(f"::error::unknown suite(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    p = plan(cfg, a.tier, only, a.has_cpu_runner.lower() == "true", a.extended.lower() == "true")
    print(json.dumps(p, indent=2))
    if a.github and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"gpu_matrix={json.dumps(p['gpu_matrix'])}\n")
            f.write(f"cpu_matrix={json.dumps(p['cpu_matrix'])}\n")
            f.write(f"nogpu_matrix={json.dumps(p['nogpu_matrix'])}\n")
            f.write(f"hosted={','.join(p['hosted'])}\n")
            f.write(f"expected={','.join(p['expected'])}\n")
            f.write(f"gpu_required={','.join(p['gpu_required'])}\n")
            f.write(f"has_gpu={'true' if p['has_gpu'] else 'false'}\n")
            f.write(f"has_cpu={'true' if p['has_cpu'] else 'false'}\n")
            f.write(f"has_nogpu={'true' if p['has_nogpu'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
