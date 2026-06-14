#!/usr/bin/env python3
"""
Perf-regression gate for AMD OpenVX (MIVisionX) pull requests.

Compares two openvx-mark `benchmark_results.json` reports captured on
the SAME runner VM (so hardware variance is zero) -- one from the PR's
build, one from the merge target's (main's) build -- and decides
whether the PR regresses performance against main.

Exits 0 on pass / acceptable change, exits 1 on regression. Always
writes a markdown verdict block to stdout, suitable for piping into
`$GITHUB_STEP_SUMMARY`.

Defaults:
    --geomean-floor 0.95   (no more than 5% aggregate slowdown)
    --kernel-floor  0.90   (no kernel may regress more than 10%)
    --warn-floor    0.95   (soft-warn band for individual kernels in
                            [0.90, 0.95); 5-10% slower -> advisory)
    --max-cv        5.0    (skip kernels above this run-to-run noise)

Each filter is applied independently; a kernel that doesn't pass the
filters (unverified, noisy, missing on either side) is reported in a
"skipped" section but does not contribute to the gate decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    mode: str
    resolution: str
    mps: float
    sustained_ms: float
    cv_percent: float
    verified: bool
    stability_warning: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.name, self.mode, self.resolution)


def _row_from(d: dict) -> Row:
    wc = d.get("wall_clock", {}) or {}
    return Row(
        name=d.get("name", "<unknown>"),
        mode=d.get("mode", ""),
        resolution=d.get("resolution", ""),
        mps=float(d.get("megapixels_per_sec") or 0.0),
        sustained_ms=float(d.get("sustained_ms") or 0.0),
        cv_percent=float(wc.get("cv_percent") or 0.0),
        verified=bool(d.get("verified", True)),
        stability_warning=bool(d.get("stability_warning", False)),
    )


def _load(path: str) -> dict[tuple[str, str, str], Row]:
    with open(path) as f:
        report = json.load(f)
    out: dict[tuple[str, str, str], Row] = {}
    for r in report.get("results", []):
        row = _row_from(r)
        out[row.key] = row
    return out


def _load_system(path: str) -> dict:
    """Return the `system` block from a benchmark_results.json, or {}."""
    with open(path) as f:
        report = json.load(f)
    return report.get("system", {}) or {}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class KernelVerdict:
    key: tuple[str, str, str]
    main: Row
    pr: Row
    ratio: float  # pr_mps / main_mps
    status: str  # "ok" | "warn" | "fail"
    reason: str = ""


@dataclass
class SkipRecord:
    key: tuple[str, str, str]
    reason: str
    main: "Row | None" = None
    pr: "Row | None" = None


def _classify(
    main: Row,
    pr: Row,
    *,
    kernel_floor: float,
    warn_floor: float,
    gate_eligible: bool = True,
    min_abs_ms: float = 0.0,
) -> KernelVerdict:
    if main.mps <= 0 or pr.mps <= 0:
        return KernelVerdict(
            key=main.key,
            main=main,
            pr=pr,
            ratio=0.0,
            status="fail" if gate_eligible else "warn",
            reason="zero throughput" + (
                "" if gate_eligible
                else f" (warn-only: under {min_abs_ms:.1f}ms, too fast to gate)"
            ),
        )
    ratio = pr.mps / main.mps
    if ratio < kernel_floor:
        if not gate_eligible:
            return KernelVerdict(
                key=main.key,
                main=main,
                pr=pr,
                ratio=ratio,
                status="warn",
                reason=(f"PR/main = {ratio:.3f}x < kernel floor {kernel_floor:.3f}x, "
                        f"but warn-only: runtime under {min_abs_ms:.1f}ms "
                        f"(main={main.sustained_ms:.3f}ms pr={pr.sustained_ms:.3f}ms) "
                        f"-- too fast to gate reliably (layout/alignment noise)"),
            )
        return KernelVerdict(
            key=main.key,
            main=main,
            pr=pr,
            ratio=ratio,
            status="fail",
            reason=f"PR/main = {ratio:.3f}x < kernel floor {kernel_floor:.3f}x",
        )
    if ratio < warn_floor:
        return KernelVerdict(
            key=main.key,
            main=main,
            pr=pr,
            ratio=ratio,
            status="warn",
            reason=f"PR/main = {ratio:.3f}x < warn floor {warn_floor:.3f}x",
        )
    return KernelVerdict(
        key=main.key,
        main=main,
        pr=pr,
        ratio=ratio,
        status="ok",
        reason="",
    )


def _geomean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 1.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_hardware(main_system: dict | None, pr_system: dict | None) -> str:
    main_system = main_system or {}
    pr_system = pr_system or {}

    def cell(d: dict, key: str, default: str = "\u2014") -> str:
        v = d.get(key)
        if v is None or v == "":
            return default
        return str(v)

    out: list[str] = []
    out.append("### Hardware")
    out.append("")
    out.append("| Field | main run | PR run |")
    out.append("|---|---|---|")
    fields = [
        ("CPU model",  "cpu_model"),
        ("CPU cores",  "cpu_cores"),
        ("RAM (GB)",   "ram_gb"),
        ("Hostname",   "hostname"),
        ("OS version", "os_version"),
        ("Timestamp",  "timestamp"),
    ]
    for label, key in fields:
        m = cell(main_system, key)
        p = cell(pr_system, key)
        out.append(f"| {label} | `{m}` | `{p}` |")

    main_cpu = cell(main_system, "cpu_model", "")
    pr_cpu = cell(pr_system, "cpu_model", "")
    main_host = cell(main_system, "hostname", "")
    pr_host = cell(pr_system, "hostname", "")
    drifted = (main_cpu and pr_cpu and main_cpu != pr_cpu) or (
        main_host and pr_host and main_host != pr_host
    )
    if drifted:
        out.append("")
        out.append(
            "> **Warning:** the two runs reported different runner "
            "hardware (CPU model or hostname). The perf comparison may be "
            "biased by the hardware delta -- interpret regressions cautiously."
        )

    return "\n".join(out)


def _emoji(status: str) -> str:
    return {
        "ok": "[ok]",
        "warn": "[warn]",
        "fail": "[fail]",
        "skip": "[skip]",
    }[status]


def _render(
    *,
    verdicts: list[KernelVerdict],
    skipped: list[SkipRecord],
    geomean_ratio: float,
    geomean_floor: float,
    kernel_floor: float,
    warn_floor: float,
    max_cv: float,
    min_abs_ms: float,
    overall_pass: bool,
    main_system: dict | None = None,
    pr_system: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append("## Perf gate (PR vs main)")
    lines.append("")
    lines.append(
        "Both AMD OpenVX builds were benchmarked on the **same runner VM** "
        "with the same workload, so hardware variance is zero -- the "
        "ratios below are pure software-side deltas attributable to "
        "this PR."
    )
    lines.append("")
    lines.append(_render_hardware(main_system, pr_system))
    lines.append("")

    if overall_pass:
        lines.append(
            "### **Verdict: PASS** "
            f"(geomean PR/main = {geomean_ratio:.3f}x, "
            f"{_count_status(verdicts, 'fail')} hard failures, "
            f"{_count_status(verdicts, 'warn')} warnings)"
        )
    else:
        lines.append(
            "### **Verdict: FAIL** "
            f"(geomean PR/main = {geomean_ratio:.3f}x, "
            f"floor = {geomean_floor:.3f}x; "
            f"{_count_status(verdicts, 'fail')} kernel(s) below "
            f"per-kernel floor of {kernel_floor:.3f}x)"
        )
    lines.append("")
    lines.append("### Thresholds")
    lines.append("")
    lines.append("| Knob | Value | Meaning |")
    lines.append("|---|---:|---|")
    lines.append(f"| Geomean floor   | {geomean_floor:.3f}x | "
                 f"PR may not be more than {(1 - geomean_floor) * 100:.1f}% slower in aggregate. |")
    lines.append(f"| Per-kernel floor | {kernel_floor:.3f}x | "
                 f"No single kernel may regress more than {(1 - kernel_floor) * 100:.1f}%. |")
    lines.append(f"| Warn floor      | {warn_floor:.3f}x | "
                 f"Soft warn for any kernel slower than {(1 - warn_floor) * 100:.1f}%. |")
    lines.append(f"| Max CV%         | {max_cv:.1f}% | "
                 f"Kernels with run-to-run CV above this are skipped. |")
    if min_abs_ms > 0:
        lines.append(f"| Min abs time    | {min_abs_ms:.1f}ms | "
                     f"Kernels faster than this on both sides are **warn-only** (still "
                     f"shown and in the geomean, but cannot hard-fail) -- too short to "
                     f"gate reliably (code-layout/alignment noise dominates). |")
    lines.append("")

    fails = [v for v in verdicts if v.status == "fail"]
    warns = [v for v in verdicts if v.status == "warn"]

    if fails:
        lines.append("### Hard regressions (block merge)")
        lines.append("")
        lines.append(_table([sorted(fails, key=lambda v: v.ratio)]))
        lines.append("")

    if warns:
        lines.append("### Soft regressions (warn only)")
        lines.append("")
        lines.append(_table([sorted(warns, key=lambda v: v.ratio)]))
        lines.append("")

    all_rows: list[KernelVerdict] = list(verdicts)
    for s in skipped:
        if s.main is not None and s.pr is not None and s.main.mps > 0 and s.pr.mps > 0:
            ratio = s.pr.mps / s.main.mps
        else:
            ratio = 0.0
        all_rows.append(KernelVerdict(
            key=s.key,
            main=s.main if s.main is not None else _empty_row(s.key),
            pr=s.pr if s.pr is not None else _empty_row(s.key),
            ratio=ratio,
            status="skip",
            reason=f"skipped: {s.reason}",
        ))

    if all_rows:
        def sort_key(v: KernelVerdict) -> tuple[float, tuple[str, str, str]]:
            r = v.ratio if v.ratio > 0 else float("inf")
            return (r, v.key)

        all_sorted = sorted(all_rows, key=sort_key)
        n_fail = _count_status(verdicts, "fail")
        n_warn = _count_status(verdicts, "warn")
        n_ok = _count_status(verdicts, "ok")
        n_skip = len(skipped)
        lines.append(
            f"### All kernels ({len(all_rows)} total -- "
            f"{n_fail} fail, {n_warn} warn, {n_ok} ok, {n_skip} skipped; "
            f"sorted worst -> best)"
        )
        lines.append("")
        lines.append(_table([all_sorted]))
        lines.append("")

    return "\n".join(lines) + "\n"


def _empty_row(key: tuple[str, str, str]) -> Row:
    name, mode, res = key
    return Row(
        name=name,
        mode=mode,
        resolution=res,
        mps=0.0,
        sustained_ms=0.0,
        cv_percent=0.0,
        verified=False,
        stability_warning=False,
    )


def _count_status(verdicts: list[KernelVerdict], status: str) -> int:
    return sum(1 for v in verdicts if v.status == status)


def _table(groups: list[list[KernelVerdict]]) -> str:
    def _mps(v: float) -> str:
        return f"{v:.2f}" if v > 0 else "\u2014"
    def _ms(v: float) -> str:
        return f"{v:.3f}" if v > 0 else "\u2014"
    def _ratio(v: float) -> str:
        return f"**{v:.3f}x**" if v > 0 else "\u2014"

    rows: list[str] = []
    rows.append("| Status | Kernel | Mode | Res | main MP/s | PR MP/s | PR/main | main ms | PR ms | Notes |")
    rows.append("|:---|---|---|---|---:|---:|---:|---:|---:|---|")
    for group in groups:
        for v in group:
            n, m, r = v.key
            rows.append(
                f"| {_emoji(v.status)} | `{n}` | {m} | {r} | "
                f"{_mps(v.main.mps)} | {_mps(v.pr.mps)} | "
                f"{_ratio(v.ratio)} | "
                f"{_ms(v.main.sustained_ms)} | {_ms(v.pr.sustained_ms)} | "
                f"{v.reason} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("main_json", help="benchmark_results.json from main's build")
    p.add_argument("pr_json", help="benchmark_results.json from PR's build")
    p.add_argument("--geomean-floor", type=float, default=0.95,
                   help="Aggregate geomean floor (default 0.95 = up to 5%% regression)")
    p.add_argument("--kernel-floor", type=float, default=0.90,
                   help="Per-kernel floor (default 0.90 = up to 10%% regression)")
    p.add_argument("--warn-floor", type=float, default=0.95,
                   help="Soft warn floor (default 0.95 = warn for individual "
                        "kernels in [-10%%, -5%%); below 5%% is treated as noise)")
    p.add_argument("--max-cv", type=float, default=5.0,
                   help="Skip kernels whose CV%% exceeds this threshold (default 5.0)")
    p.add_argument("--min-abs-ms", type=float, default=0.0,
                   help="Skip (do not hard-gate) kernels whose runtime on both sides is "
                        "below this many milliseconds. Sub-millisecond kernels cannot be "
                        "compared reliably across two separately-linked binaries on a "
                        "shared runner: an unrelated code-size change shifts a hot loop's "
                        "instruction alignment and produces deterministic >10%% swings that "
                        "have nothing to do with the algorithm. Default 0 = disabled.")
    p.add_argument("--summary-out", default=None,
                   help="Append the markdown verdict to this file (e.g. $GITHUB_STEP_SUMMARY)")
    p.add_argument("--skip-name", action="append", default=[],
                   help="Skip a kernel by name (case-sensitive). May be repeated.")
    args = p.parse_args(argv)

    main_rows = _load(args.main_json)
    pr_rows = _load(args.pr_json)
    main_system = _load_system(args.main_json)
    pr_system = _load_system(args.pr_json)

    skipped: list[SkipRecord] = []
    verdicts: list[KernelVerdict] = []

    skip_names = set(args.skip_name)

    for key in sorted(set(main_rows) & set(pr_rows)):
        m, r = main_rows[key], pr_rows[key]
        if m.name in skip_names:
            skipped.append(SkipRecord(key=key, reason="explicitly skipped by --skip-name", main=m, pr=r))
            continue
        if not (m.verified and r.verified):
            skipped.append(SkipRecord(key=key, reason="unverified on at least one side", main=m, pr=r))
            continue
        if m.stability_warning or r.stability_warning:
            skipped.append(SkipRecord(key=key, reason="stability_warning on at least one side", main=m, pr=r))
            continue
        if m.cv_percent > args.max_cv or r.cv_percent > args.max_cv:
            skipped.append(SkipRecord(
                key=key,
                reason=f"CV% over {args.max_cv}% (main={m.cv_percent:.2f}% pr={r.cv_percent:.2f}%)",
                main=m,
                pr=r,
            ))
            continue

        # Kernels too fast to measure reliably across two separately-linked
        # binaries (sub-ms hot loops are dominated by code-layout/alignment
        # noise) stay visible and counted in the geomean, but can only ever
        # warn -- never hard-fail the gate.
        gate_eligible = not (
            args.min_abs_ms > 0
            and max(m.sustained_ms, r.sustained_ms) < args.min_abs_ms
        )

        verdicts.append(_classify(
            m, r,
            kernel_floor=args.kernel_floor,
            warn_floor=args.warn_floor,
            gate_eligible=gate_eligible,
            min_abs_ms=args.min_abs_ms,
        ))

    for key in sorted(set(main_rows) - set(pr_rows)):
        skipped.append(SkipRecord(
            key=key,
            reason="missing in PR run (new on main?)",
            main=main_rows[key],
        ))
    for key in sorted(set(pr_rows) - set(main_rows)):
        skipped.append(SkipRecord(
            key=key,
            reason="missing in main run (new in PR -- not gated)",
            pr=pr_rows[key],
        ))

    geomean_ratio = _geomean(v.ratio for v in verdicts if v.ratio > 0)

    has_hard_fail = any(v.status == "fail" for v in verdicts)
    geomean_fail = geomean_ratio < args.geomean_floor and len(verdicts) > 0
    overall_pass = not (has_hard_fail or geomean_fail)

    md = _render(
        verdicts=verdicts,
        skipped=skipped,
        geomean_ratio=geomean_ratio,
        geomean_floor=args.geomean_floor,
        kernel_floor=args.kernel_floor,
        warn_floor=args.warn_floor,
        main_system=main_system,
        pr_system=pr_system,
        max_cv=args.max_cv,
        min_abs_ms=args.min_abs_ms,
        overall_pass=overall_pass,
    )

    sys.stdout.write(md)
    if args.summary_out:
        with open(args.summary_out, "a") as f:
            f.write(md)

    if not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
