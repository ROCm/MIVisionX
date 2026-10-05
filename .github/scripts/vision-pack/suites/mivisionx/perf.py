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

"""Performance data for the mivisionx suite (full tier; run last, nothing else on the GPU).

  perf.py runvx --out perf.json         runvx -dump-profile medians for shipped GDFs, CPU and GPU
      -> metrics runvx.<T>.<gdf>.median_ms; checks perf.runvx.<T>::<gdf>
  perf.py openvx-mark --json F --out perf.json
      -> metrics openvx-mark.<T>.<kernel>.median_ms (+ cv_percent for the gate's --max-cv 10,
         --min-abs-ms 1 rule); checks perf.openvx-mark.<T>::<kernel> (verified output)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mvx_common import RUNVX, TEST_ROOT, VP_OUT, record, run, tail, write_log

GDFS = [
    "geometric/Remap_U8_U8_Bilinear_4K.gdf", "geometric/WarpAffine_U8_U8_Bilinear.gdf",
    "geometric/ScaleImage_U8_U8_Bilinear.gdf", "filter/Gaussian_U8_U8_3x3.gdf", "filter/Convolve_U8_U8_7x7.gdf",
    "color/ColorConvert_RGB_IYUV.gdf", "vision/Canny_3x3_L1NORM.gdf", "vision/Harris_3x3.gdf",
    "vision_profile/29_opticalFlowLK.gdf", "vision_profile/43_feature_tracker.gdf", "logical/And_alt.gdf",
]
OVERALL = re.compile(r"csv,OVERALL,\s*(\w+),\s*(\d+),[^,]*,\s*([\d.]+),\s*([\d.]+),.*\(median ([\d.]+)\)")


def metric(name, value, unit, backend, **extra):
    m = {"name": name, "value": value, "unit": unit, "lower_is_better": True, "backend": backend}
    m.update(extra)
    return m


def runvx_perf(out: Path, frames: int) -> None:
    log = VP_OUT / "logs" / "perf.runvx.log"
    metrics = []
    for rel in GDFS:
        gdf = TEST_ROOT / "amd_openvx_gdfs" / rel
        stem = Path(rel).stem
        for t in ("CPU", "GPU"):
            r = run([RUNVX, f"-frames:{int(frames)}", f"-affinity:{t}", "-dump-profile", "file", gdf], 600, cwd=gdf.parent)
            write_log(log, f"### {rel} {t} rc={int(r.rc)}", r.out)
            m = OVERALL.search(r.out)
            tid = f"perf.runvx.{t}::{rel}"
            if r.status() != "pass" or not m:
                record(tid, r.status() if r.status() != "pass" else "fail", f"{r.why()}: {tail(r.out, 300)}",
                       r.dt, log, t, r.repro(cwd=gdf.parent))
                continue
            cpu_nodes = len(re.findall(r",CPU,com\.amd", r.out))
            record(tid, "pass", f"median {m.group(5)} ms/frame over {m.group(2)} frames", r.dt, log, t,
                   r.repro(cwd=gdf.parent))
            metrics.append(metric(f"runvx.{t}.{stem}.median_ms", float(m.group(5)), "ms", t,
                                  min_ms=float(m.group(4)), cpu_nodes=cpu_nodes))
    out.write_text(json.dumps({"metrics": metrics}, indent=1))


def openvx_mark(js: Path, out: Path, target: str) -> None:
    rep = json.loads(js.read_text())
    metrics = []
    for res in rep.get("results", []):
        name = res.get("name", "?")
        wc = res.get("wall_clock") or {}
        tid = f"perf.openvx-mark.{target}::{name}"
        if not res.get("verified"):
            record(tid, "fail", "output not verified", backend=target)
        elif "median_ms" not in wc:
            record(tid, "error", "verified, but openvx-mark reported no wall-clock timing", backend=target)
        else:
            record(tid, "pass", f"median {wc['median_ms']:.3f} ms, CV {wc.get('cv_percent', 0):.2f}%", backend=target)
        if "median_ms" in wc:
            metrics.append(metric(f"openvx-mark.{target}.{name}.median_ms", wc["median_ms"], "ms", target,
                                  cv_percent=wc.get("cv_percent"), resolution=res.get("resolution"),
                                  megapixels_per_sec=res.get("megapixels_per_sec")))
    if not metrics:
        record(f"perf.openvx-mark.{target}::results", "error", "benchmark_results.json has no results", backend=target)
    out.write_text(json.dumps({"metrics": metrics}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("runvx", "openvx-mark"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--target", default="GPU")
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "runvx":
        runvx_perf(Path(a.out), a.frames)
    else:
        openvx_mark(Path(a.json), Path(a.out), a.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
