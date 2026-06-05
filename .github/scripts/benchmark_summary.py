#!/usr/bin/env python3
"""
Render an intuitive GitHub Actions step summary for the AMD OpenVX vs OpenCV
vision-kernel benchmark.

The upstream `ci_pairwise_summary.py` is built for an N-implementation matrix
(MIVisionX / Khronos / rustVX / OpenCV) and, for the single MIVisionX-vs-OpenCV
pair this CI actually runs, it renders a confusing 2x2 "geomean speedup matrix"
and a dense 9-column table with no plain-English verdict. This helper answers
the one question a reader of this PR actually has, up front:

    "How much faster is AMD OpenVX (MIVisionX) than writing the same vision
     kernel directly in OpenCV?"

It loads the two `benchmark_results.json` reports, joins them per kernel on
(name, mode, resolution), and emits — in reading order:

  1. TL;DR — one bold sentence + a small at-a-glance facts table.
  2. Speedup by category — where the win is biggest / smallest.
  3. Top wins for AMD OpenVX.
  4. Where OpenCV is faster (only shown when there are any).
  5. Full per-kernel table, collapsed in a <details> block.

Speedup orientation matches scripts/compare_reports.py:
    speedup = candidate MP/s / baseline MP/s   (>1.00x => candidate faster)

Output goes to stdout; the caller redirects it into "$GITHUB_STEP_SUMMARY".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass


# Canonical category display order (anything unknown is appended, sorted).
_CATEGORY_ORDER = [
    "pixelwise", "filters", "color", "geometric",
    "statistical", "multiscale", "feature", "misc",
]

_CATEGORY_LABEL = {
    "pixelwise": "Pixel-wise",
    "filters": "Filters",
    "color": "Color",
    "geometric": "Geometric",
    "statistical": "Statistical",
    "multiscale": "Multi-scale",
    "feature": "Feature",
    "misc": "Misc",
}


@dataclass
class Pair:
    name: str
    category: str
    mode: str
    resolution: str
    cand_mps: float
    base_mps: float
    cand_ms: float
    base_ms: float

    @property
    def speedup(self) -> float:
        return self.cand_mps / self.base_mps


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _index(report: dict) -> dict:
    out = {}
    for r in report.get("results", []):
        key = (r.get("name", ""), r.get("mode", ""), r.get("resolution", ""))
        out[key] = r
    return out


def _mps(r: dict) -> float:
    return float(r.get("megapixels_per_sec") or 0.0)


def _median_ms(r: dict) -> float:
    wc = r.get("wall_clock", {}) or {}
    v = wc.get("median_ms")
    if v is None:
        # Older reports stored sustained_ms only.
        v = r.get("sustained_ms")
    return float(v or 0.0)


def _verified(r: dict) -> bool:
    return bool(r.get("verified", True))


def _geomean(values: list[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def build_pairs(cand: dict, base: dict) -> list[Pair]:
    ci, bi = _index(cand), _index(base)
    pairs: list[Pair] = []
    for key in sorted(set(ci) & set(bi)):
        rc, rb = ci[key], bi[key]
        if not (_verified(rc) and _verified(rb)):
            continue
        mc, mb = _mps(rc), _mps(rb)
        if mc <= 0 or mb <= 0:
            continue
        pairs.append(Pair(
            name=key[0],
            category=rc.get("category", "") or rb.get("category", ""),
            mode=key[1],
            resolution=key[2],
            cand_mps=mc,
            base_mps=mb,
            cand_ms=_median_ms(rc),
            base_ms=_median_ms(rb),
        ))
    return pairs


def _fmt_speedup(s: float) -> str:
    return f"{s:.2f}x"


def _runner_line(report: dict) -> str:
    sysinfo = report.get("system", {}) or {}
    cpu = sysinfo.get("cpu_model") or "unknown CPU"
    cores = sysinfo.get("cpu_cores")
    ram = sysinfo.get("ram_gb")
    bits = [str(cpu)]
    if cores:
        bits.append(f"{cores} cores")
    if ram:
        bits.append(f"{ram} GB RAM")
    return f"{bits[0]} ({', '.join(bits[1:])})" if len(bits) > 1 else bits[0]


def render(
    pairs: list[Pair],
    *,
    cand_label: str,
    base_label: str,
    cand_report: dict,
    expected_kernels: int | None,
    out,
) -> None:
    resolutions = sorted({p.resolution for p in pairs})
    res_str = ", ".join(resolutions) if resolutions else "—"

    out.write(
        f"# {cand_label} vs {base_label} — vision kernel speedup (CPU / host backend)\n\n"
    )
    out.write(
        f"> **This is a CPU-only (host backend) benchmark — no GPU is involved.** "
        f"{cand_label} is built with `GPU_SUPPORT=OFF` (the AGO host/CPU code "
        f"path) and {base_label} runs on the CPU, so every number below is a "
        f"pure CPU-vs-CPU comparison.\n>\n"
        f"> **How much faster is {cand_label} than doing the same work directly "
        f"in {base_label}?** Both binaries are built from the same "
        f"[openvx-mark](https://github.com/kiritigowda/openvx-mark) source tree "
        f"(identical timer, stats, and JSON schema) and run on the **same CPU "
        f"runner** at **{res_str}**, **single-threaded** (`--threads 1`), "
        f"20 measured iterations + 5 warmup.\n>\n"
        f"> **Speedup = {cand_label} MP/s ÷ {base_label} MP/s.** "
        f"A value of **> 1.00x means {cand_label} is faster.**\n\n"
    )

    if not pairs:
        out.write(
            "_No comparable verified kernels found in both reports — nothing to "
            "summarize. Check the bench steps above for build/run failures._\n\n"
        )
        return

    speedups = [p.speedup for p in pairs]
    geomean = _geomean(speedups)
    median = sorted(speedups)[len(speedups) // 2]
    wins = [p for p in pairs if p.speedup > 1.0]
    losses = [p for p in pairs if p.speedup < 1.0]
    best = max(pairs, key=lambda p: p.speedup)
    worst = min(pairs, key=lambda p: p.speedup)
    total = len(pairs)

    coverage = f"{total}"
    if expected_kernels:
        coverage = f"{total} / {expected_kernels}"

    # --- 1. TL;DR -----------------------------------------------------------
    out.write("## TL;DR\n\n")
    if geomean >= 1.0:
        out.write(
            f"**{cand_label} is on average {_fmt_speedup(geomean)} faster than "
            f"{base_label}** (geometric mean) across **{coverage} vision "
            f"kernels**, winning on **{len(wins)} of {total}**.\n\n"
        )
    else:
        out.write(
            f"**{cand_label} is on average {_fmt_speedup(1.0 / geomean)} *slower* "
            f"than {base_label}** (geometric mean) across **{coverage} vision "
            f"kernels**, winning on only **{len(wins)} of {total}**.\n\n"
        )

    out.write("| Metric | Value |\n")
    out.write("|:---|:---|\n")
    out.write("| Backend | **CPU / host only** (no GPU) |\n")
    out.write(f"| Average speedup (geomean) | **{_fmt_speedup(geomean)}** |\n")
    out.write(f"| Median speedup | {_fmt_speedup(median)} |\n")
    out.write(f"| Kernels where {cand_label} wins | {len(wins)} / {total} |\n")
    out.write(f"| Kernels where {base_label} wins | {len(losses)} / {total} |\n")
    out.write(
        f"| Biggest win | `{best.name}` — {_fmt_speedup(best.speedup)} |\n"
    )
    out.write(
        f"| Biggest gap | `{worst.name}` — {_fmt_speedup(worst.speedup)} |\n"
    )
    out.write(f"| Runner | {_runner_line(cand_report)} |\n\n")

    # --- 2. Speedup by category --------------------------------------------
    by_cat: dict[str, list[Pair]] = {}
    for p in pairs:
        by_cat.setdefault(p.category, []).append(p)

    ordered_cats = [c for c in _CATEGORY_ORDER if c in by_cat]
    ordered_cats += sorted(c for c in by_cat if c not in _CATEGORY_ORDER)

    out.write("## Speedup by category\n\n")
    out.write("| Category | Kernels | Avg speedup (geomean) | Best | Worst |\n")
    out.write("|:---|---:|---:|:---|:---|\n")
    for cat in ordered_cats:
        group = by_cat[cat]
        g = _geomean([p.speedup for p in group])
        b = max(group, key=lambda p: p.speedup)
        w = min(group, key=lambda p: p.speedup)
        label = _CATEGORY_LABEL.get(cat, cat or "—")
        out.write(
            f"| {label} | {len(group)} | **{_fmt_speedup(g)}** | "
            f"`{b.name}` ({_fmt_speedup(b.speedup)}) | "
            f"`{w.name}` ({_fmt_speedup(w.speedup)}) |\n"
        )
    out.write("\n")

    # --- 3. Top wins --------------------------------------------------------
    top = sorted(wins, key=lambda p: p.speedup, reverse=True)[:5]
    if top:
        out.write(f"## Top {len(top)} wins for {cand_label}\n\n")
        out.write(
            f"| Kernel | {cand_label} MP/s | {base_label} MP/s | Speedup |\n"
        )
        out.write("|:---|---:|---:|---:|\n")
        for p in top:
            out.write(
                f"| `{p.name}` | {p.cand_mps:,.1f} | {p.base_mps:,.1f} | "
                f"**{_fmt_speedup(p.speedup)}** |\n"
            )
        out.write("\n")

    # --- 4. Where the baseline wins (only if any) --------------------------
    if losses:
        worst_first = sorted(losses, key=lambda p: p.speedup)
        out.write(f"## Where {base_label} is faster ({len(losses)})\n\n")
        out.write(
            f"> {base_label} edges ahead on these kernels. The `{base_label} "
            f"advantage` column shows how much faster {base_label} is "
            f"(= 1 ÷ speedup).\n\n"
        )
        out.write(
            f"| Kernel | {cand_label} MP/s | {base_label} MP/s | Speedup | "
            f"{base_label} advantage |\n"
        )
        out.write("|:---|---:|---:|---:|---:|\n")
        for p in worst_first:
            out.write(
                f"| `{p.name}` | {p.cand_mps:,.1f} | {p.base_mps:,.1f} | "
                f"{_fmt_speedup(p.speedup)} | {_fmt_speedup(1.0 / p.speedup)} |\n"
            )
        out.write("\n")

    # --- 5. Full per-kernel table (collapsed) ------------------------------
    full = sorted(pairs, key=lambda p: p.speedup, reverse=True)
    out.write(
        f"<details><summary><b>All {total} kernels</b> — full per-kernel table "
        f"(sorted by speedup, biggest {cand_label} win first)</summary>\n\n"
    )
    out.write(
        f"| # | Kernel | Category | Speedup | {cand_label} MP/s | "
        f"{base_label} MP/s | {cand_label} ms | {base_label} ms |\n"
    )
    out.write("|---:|:---|:---|---:|---:|---:|---:|---:|\n")
    for i, p in enumerate(full, start=1):
        label = _CATEGORY_LABEL.get(p.category, p.category or "—")
        speed = (f"**{_fmt_speedup(p.speedup)}**" if p.speedup >= 1.0
                 else f"_{_fmt_speedup(p.speedup)}_")
        out.write(
            f"| {i} | `{p.name}` | {label} | {speed} | "
            f"{p.cand_mps:,.1f} | {p.base_mps:,.1f} | "
            f"{p.cand_ms:.3f} | {p.base_ms:.3f} |\n"
        )
    out.write("\n</details>\n\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("candidate_json",
                    help="benchmark_results.json from the OpenVX impl (AMD OpenVX)")
    ap.add_argument("baseline_json",
                    help="benchmark_results.json from the OpenCV baseline")
    ap.add_argument("--candidate-label", default="AMD OpenVX (MIVisionX)")
    ap.add_argument("--baseline-label", default="OpenCV")
    ap.add_argument("--expected-kernels", type=int, default=None,
                    help="Expected kernel count, shown as coverage (e.g. 42).")
    ap.add_argument("--summary-out", default=None,
                    help="Append output to this file instead of stdout "
                         "(e.g. $GITHUB_STEP_SUMMARY).")
    args = ap.parse_args(argv)

    for path in (args.candidate_json, args.baseline_json):
        if not os.path.isfile(path):
            sys.stderr.write(f"ERROR: report not found: {path}\n")
            return 2

    cand = _load(args.candidate_json)
    base = _load(args.baseline_json)
    pairs = build_pairs(cand, base)

    if args.summary_out:
        out = open(args.summary_out, "a")
    else:
        out = sys.stdout
    try:
        render(
            pairs,
            cand_label=args.candidate_label,
            base_label=args.baseline_label,
            cand_report=cand,
            expected_kernels=args.expected_kernels,
            out=out,
        )
    finally:
        if args.summary_out:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
