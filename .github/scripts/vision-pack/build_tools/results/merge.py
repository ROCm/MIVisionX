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

"""Merge the per-suite result directories of one QA run.

Each job uploads an artifact (``results-gpu``, ``results-cpu``) with one
directory per suite, holding ``results.jsonl``, ``suite.json``, ``junit/``,
``logs/`` and ``perf/``. After ``actions/download-artifact`` they sit side by
side under one directory. This script collects them into a single ``merged/``
view used by report/triage.py:

    merged/results.jsonl     every record, de-duplicated by id (last one wins)
    merged/suites.json       per-suite metadata; several result directories of
                             one suite (the hosted packaging jobs) are combined
    merged/environment.json  runner, SDK, images, pre-flight (results-environment)
    merged/perf/<suite>__<name>.json

It adds synthetic ``error`` records so infrastructure problems always show:
``<suite>::infra::no-results`` for an expected suite without results,
``<suite>::infra::runner`` when the CI step reported a non-zero exit
(runner-status.json: container failure, time budget), and
``preflight::infra::gpu-preflight`` when the GPU pre-flight failed.

When the runner had no usable GPU (environment.json ``gpu_present: false`` or
prepared.json ``mode: no-gpu``) only the ``needs_gpu: false`` suites ran:
``preflight::infra::no-gpu`` carries the reason, and every expected suite
that needs a GPU (suites/suites.yaml) is ``<suite>::infra::no-gpu`` instead
of ``no-results``.

    merge.py --results downloaded/ --out merged/ [--expect packaging,rocal,...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emit import make_record, read_records  # noqa: E402

ENV_FILES = ("environment.json", "prepared.json", "manifest.json", "preflight.json")
SUITES_YAML = Path(__file__).resolve().parents[2] / "suites" / "suites.yaml"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def no_gpu_reason(environment: dict) -> str | None:
    """Why the runner had no usable GPU, or None if it had one (or nobody recorded it)."""
    env, prepared = environment.get("environment") or {}, environment.get("prepared") or {}
    if env.get("gpu_present") is False or prepared.get("mode") == "no-gpu":
        return env.get("no_gpu_reason") or "no usable GPU detected"
    return None


def gpu_suites(config: Path = SUITES_YAML) -> set[str] | None:
    """Self-hosted suites without ``needs_gpu: false``; None if suites.yaml is unreadable."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        cfg = yaml.safe_load(config.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    return {name for name, s in (cfg.get("suites") or {}).items()
            if s.get("runner", "gpu") != "hosted" and s.get("needs_gpu", True) is not False}


def _combine(old: dict, new: dict) -> dict:
    """Combine two suite.json of the same suite (e.g. packaging-deb + -rpm)."""
    out = dict(old)
    out["result_dirs"] = sorted(set(old.get("result_dirs", [])) | set(new.get("result_dirs", [])))
    out["wall_seconds"] = old.get("wall_seconds", 0) + new.get("wall_seconds", 0)
    out["total"] = old.get("total", 0) + new.get("total", 0)
    counts = dict(old.get("counts", {}))
    for k, v in new.get("counts", {}).items():
        counts[k] = counts.get(k, 0) + v
    out["counts"] = counts
    return out


def merge(root: Path, out: Path, expect: list[str], need_gpu: set[str] | None = None) -> dict:
    """``need_gpu``: suites that cannot run without a GPU (default: from suites.yaml)."""
    (out / "perf").mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    suites: dict[str, dict] = {}
    environment: dict = {}

    for jsonl in sorted(root.rglob("results.jsonl")):
        if out in jsonl.parents:
            continue
        sdir = jsonl.parent
        meta = _load(sdir / "suite.json")
        recs = read_records(jsonl)
        suite = meta.get("suite") or (recs[0]["suite"] if recs else sdir.name)
        rel = str(sdir.relative_to(root))
        meta.update({"suite": suite, "result_dirs": [rel]})
        suites[suite] = _combine(suites[suite], meta) if suite in suites else meta
        for r in recs:
            r["result_dir"] = rel
            records[r["id"]] = r
        if (sdir / "perf").is_dir():
            for pf in sorted((sdir / "perf").glob("*.json")):
                shutil.copy(pf, out / "perf" / f"{suite}__{pf.name}")

    for status in sorted(root.rglob("runner-status.json")):
        st = _load(status)
        if st.get("rc", 0) != 0:
            suite = st.get("suite") or status.parent.name
            rec = make_record(suite, f"{suite}::infra::runner", "error",
                              message=f"the CI step {st.get('reason', 'failed')}; results may be partial")
            records[rec["id"]] = rec
            suites.setdefault(suite, {"suite": suite, "result_dirs": []})["runner"] = st

    # The GPU job's results hold environment/ and preflight/; files of the same
    # name inside suite result directories are not the run's.
    for name in ENV_FILES:
        for f in sorted(root.rglob(name)):
            if out in f.parents or f.parent.name not in ("environment", "preflight"):
                continue
            environment[name.removesuffix(".json")] = _load(f)
    pre = environment.get("preflight")
    if pre is not None and not pre.get("ok", False):
        rec = make_record("preflight", "preflight::infra::gpu-preflight", "error",
                          message=f"GPU pre-flight failed: {pre.get('reason', 'unknown')}")
        records[rec["id"]] = rec
    why = no_gpu_reason(environment)
    if why is not None:
        rec = make_record("preflight", "preflight::infra::no-gpu", "error",
                          message=f"the runner had no usable GPU ({why}); only the needs_gpu: false suites ran")
        records[rec["id"]] = rec
        if need_gpu is None:
            need_gpu = gpu_suites()

    for suite in expect:
        if suite in suites:
            continue
        if why is not None and (need_gpu is None or suite in need_gpu):
            rec = make_record(suite, f"{suite}::infra::no-gpu", "error",
                              message=f"not run: the runner had no usable GPU ({why})")
            suites[suite] = {"suite": suite, "missing": True, "no_gpu": True, "result_dirs": []}
        else:
            rec = make_record(suite, f"{suite}::infra::no-results", "error",
                              message="the suite job produced no results (job failed, timed out or was cancelled)")
            suites[suite] = {"suite": suite, "missing": True, "result_dirs": []}
        records[rec["id"]] = rec

    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in records.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "suites.json").write_text(json.dumps(suites, indent=2) + "\n", encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    return {"records": len(records), "suites": len(suites)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory containing the downloaded suite outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", default="", help="comma-separated suites that must have produced results")
    a = ap.parse_args()
    expect = [s.strip() for s in a.expect.split(",") if s.strip()]
    stats = merge(Path(a.results), Path(a.out), expect)
    print(f"merged {stats['records']} records from {stats['suites']} suites into {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
