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

"""Classify one night's results and decide the verdict.

    triage.py --merged merged/ --known baselines/known_issues.yaml
              [--expected baselines/expected_counts.yaml] [--perf-policy baselines/perf_policy.yaml]
              [--history DIR] [--plan run-plan.json] [--tier comprehensive] [--gfx gfx1201]
              [--night YYYY-MM-DD] [--run-url URL] [--report-url URL]
              [--out triage.json] [--summary summary.md] [--status-out status.json.gz]
              [--perf-out perf.jsonl] [--issues-state-out issues-state.json]

    triage.py --results out/ --known baselines/known_issues.yaml --summary summary.md   (local use)

Everything but --merged or --results is optional; --history is a directory of
earlier nights' status files, and without it no previous night is compared.

Every result ID gets one class:

    pass, new_test            fine
    new_failure, still_failing, infra_error         red
    known_fail, known_flaky, flaky, fixed, blocked  yellow
    known_blocked, skip, quarantined                neutral

against baselines/known_issues.yaml (glob match on the ID, scoped by gfx and
tier) and, with --history, the previous night of the same tier and GPU.
Expected-count floors, zero-result suites and hard performance
regressions also turn the night red; removed tests and soft performance drops
make it yellow. Suites never apply baselines themselves: this is the only place
a failure becomes "known".

A night on a runner without a usable GPU (merge.py's ``*::infra::no-gpu``
records) is red with one "runner had no usable GPU" reason and one issue
(``runner::no-gpu``); the degraded suites' "no GPU" blocked checks are
known_blocked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import gzip
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "build_tools" / "results"))
from merge import no_gpu_reason  # noqa: E402

FAILING = {"fail", "error"}
RED = {"new_failure", "still_failing", "infra_error"}
YELLOW = {"known_fail", "known_flaky", "flaky", "fixed", "blocked"}
TIERS = ["quick", "standard", "comprehensive", "full"]
NO_GPU_KEY = "runner::no-gpu"  # every <suite>::infra::no-gpu of a night: one issue, not one per suite


# --------------------------------------------------------------------------- inputs

def load_yaml(path: str | None, default):
    if not path or not Path(path).exists():
        return default
    return yaml.safe_load(Path(path).read_text()) or default


def load_records(merged: Path) -> list[dict]:
    recs = []
    with open(merged / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def split_id(rid: str) -> tuple[str, str, str]:
    parts = rid.split("::", 2)
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def load_match_file(base: Path, name: str) -> tuple[set[str], list[str]]:
    """Exact IDs and globs of a match_file (one per line, '#' comments)."""
    exact, globs = set(), []
    for line in (base / name).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (globs.append if any(c in line for c in "*?[") else exact.add)(line)
    return exact, globs


class Known:
    """baselines/known_issues.yaml entries, matched by glob or exact ID list, gfx and tier.

    `match_file` (relative to the YAML file) holds long exact lists, e.g. the
    thousands of parameterised CTS results a finding covers."""

    def __init__(self, doc: dict, gfx: str, tier: str):
        base = Path(doc.get("_base", "."))
        self.entries = []
        self._exact: dict[int, set[str]] = {}
        self._globs: dict[int, list[str]] = {}
        for e in doc.get("issues", []):
            gfx_scope = e.get("gfx", ["*"])
            tiers = e.get("tiers", TIERS)
            if not any(fnmatch.fnmatchcase(gfx or "", g) for g in gfx_scope):
                continue
            if tier and tier not in tiers:
                continue
            globs, exact = list(e.get("match", [])), set()
            if e.get("match_file"):
                exact, more = load_match_file(base, e["match_file"])
                globs += more
            self._exact[id(e)], self._globs[id(e)] = exact, globs
            self.entries.append(e)
        self._cache: dict[str, dict | None] = {}

    def match(self, rid: str) -> dict | None:
        if rid not in self._cache:
            self._cache[rid] = next(
                (e for e in self.entries
                 if rid in self._exact[id(e)] or any(fnmatch.fnmatchcase(rid, m) for m in self._globs[id(e)])), None)
        return self._cache[rid]


# --------------------------------------------------------------------------- history

def load_history(hist: Path | None, tier: str, gfx: str, night: str) -> dict:
    """Previous night (same tier and gfx), 30-night trend, perf and issue state."""
    out = {"prev_night": None, "prev": {}, "trend": [], "perf": [], "issues_state": {}}
    if not hist or not hist.is_dir():
        return out
    idx = hist / "index.json"
    nights = json.loads(idx.read_text()).get("nights", []) if idx.exists() else []
    nights = sorted(nights, key=lambda n: n.get("night", ""))
    out["trend"] = [n for n in nights if n.get("night") != night][-30:]
    for n in reversed(nights):
        if n.get("night") == night or n.get("tier") != tier or (gfx and n.get("gfx") != gfx):
            continue
        status = hist / n.get("dir", f"nightly/{n['night']}") / "status.json.gz"
        if status.exists():
            with gzip.open(status, "rt", encoding="utf-8") as f:
                out["prev"] = json.load(f)
            out["prev_night"] = n["night"]
            break
    perf = hist / "perf" / f"{gfx or 'unknown'}.jsonl"
    if perf.exists():
        out["perf"] = [json.loads(line) for line in perf.read_text().splitlines() if line.strip()]
    state = hist / "issues-state.json"
    if state.exists():
        out["issues_state"] = json.loads(state.read_text())
    return out


# --------------------------------------------------------------------------- classification

def classify(rec: dict, known: dict | None, prev: dict | None, have_prev: bool, no_gpu: bool = False) -> str:
    rid, s = rec["id"], rec["status"]
    ps = prev[0] if prev else None
    if known and known.get("kind") == "quarantine":
        return "quarantined"
    if "::infra::" in rid:
        return "infra_error" if s in FAILING else "pass"
    if s == "pass":
        if known and known.get("kind") == "xfail":
            return "fixed"
        return "new_test" if have_prev and ps is None else "pass"
    if s == "flaky":
        return "known_flaky" if known else "flaky"
    if s in FAILING:
        if known and known.get("kind") in ("xfail", "flaky"):
            return "known_fail"
        if ps in FAILING:
            return "still_failing"
        return "new_failure"
    if s == "blocked":
        # Without a GPU the GPU checks of the needs_gpu: false suites are blocked by
        # design; infra::no-gpu already makes the night red.
        if no_gpu and "no gpu" in (rec.get("message") or "").lower():
            return "known_blocked"
        # A known finding already explains the test; being blocked tonight adds nothing.
        return "known_blocked" if known and known.get("kind") in ("skip", "xfail", "flaky") else "blocked"
    return "skip"


def expected_count_items(cfg: dict, records: list[dict], suites: dict, tier: str) -> list[dict]:
    items = []
    by_suite: dict[str, list[str]] = defaultdict(list)
    for r in records:
        s, g, _ = split_id(r["id"])
        by_suite[s].append(g)
    for rule in cfg.get("counts", []):
        suite = rule["suite"]
        if tier not in rule.get("tiers", TIERS):
            continue
        meta = suites.get(suite)
        if not meta or meta.get("missing"):
            continue  # already an infra error
        actual = sum(1 for g in by_suite.get(suite, []) if fnmatch.fnmatchcase(g, rule["group"]))
        ok = actual >= rule["min"]
        items.append({"suite": suite, "group": rule["group"], "min": rule["min"], "actual": actual, "ok": ok,
                      "note": rule.get("note", "")})
    for suite, meta in suites.items():
        if not meta.get("missing") and not by_suite.get(suite) and suite != "preflight":
            items.append({"suite": suite, "group": "*", "min": 1, "actual": 0, "ok": False,
                          "note": "the suite ran but recorded no results"})
    return items


def perf_eval(merged: Path, policy: dict, history: list[dict], night: str) -> dict:
    window = int(policy.get("window_nights", 7))
    min_hist = int(policy.get("min_history", 3))
    floor = float(policy.get("metric_floor", 0.90))
    hard_floor = float(policy.get("hard_metric_floor", 0.75))
    geo_floor = float(policy.get("geomean_floor", 0.95))
    cv_cap = float(policy.get("cv_cap", 0.10))
    min_ms = float(policy.get("min_abs_ms", 1.0))

    past: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for h in history:
        if h.get("night") != night:
            past[h["key"]].append((h["night"], float(h["value"])))
    metrics, append = [], []
    for pf in sorted((merged / "perf").glob("*.json")) if (merged / "perf").is_dir() else []:
        suite = pf.name.split("__", 1)[0]
        try:
            doc = json.loads(pf.read_text())
        except ValueError:
            continue
        for m in doc.get("metrics", []):
            try:
                value = float(m["value"])
            except (KeyError, TypeError, ValueError):
                continue
            key = "/".join(x for x in (suite, m.get("backend", ""), m["name"]) if x)
            unit = str(m.get("unit", ""))
            lower = bool(m.get("lower_is_better", unit in ("s", "ms", "us", "ns", "sec", "msec")))
            append.append({"night": night, "key": key, "value": value, "unit": m.get("unit", "")})
            vals = [v for _, v in sorted(past.get(key, []))][-window:]
            entry = {"key": key, "value": value, "unit": m.get("unit", ""), "lower_is_better": lower,
                     "history": [v for _, v in sorted(past.get(key, []))][-30:]}
            if len(vals) < min_hist or value <= 0:
                entry["status"] = "new"
                metrics.append(entry)
                continue
            med = statistics.median(vals)
            cv = statistics.pstdev(vals) / med if med else 0.0
            ratio = (med / value) if lower else (value / med)
            entry.update({"median": med, "ratio": ratio, "cv": cv})
            small = entry["unit"] in ("ms", "msec") and max(med, value) < min_ms
            if cv > cv_cap or small:
                entry["status"] = "warn-only" if ratio < floor else "ok"
            elif ratio < hard_floor:
                entry["status"] = "hard"
            elif ratio < floor:
                entry["status"] = "soft"
            else:
                entry["status"] = "ok"
            metrics.append(entry)
    gated = [m["ratio"] for m in metrics if m.get("status") in ("ok", "soft", "hard") and m.get("ratio", 0) > 0]
    geomean = math.exp(sum(math.log(r) for r in gated) / len(gated)) if gated else None
    hard = any(m["status"] == "hard" for m in metrics) or (geomean is not None and geomean < geo_floor)
    soft = any(m["status"] in ("soft", "warn-only") and m.get("ratio", 1) < floor for m in metrics)
    return {"metrics": metrics, "geomean": geomean, "hard": hard, "soft": soft, "append": append,
            "policy": {"window_nights": window, "metric_floor": floor, "hard_metric_floor": hard_floor,
                       "geomean_floor": geo_floor, "cv_cap": cv_cap, "min_abs_ms": min_ms}}


def no_gpu_issue_body(its: list[dict], why: str | None, nights: int, report_url: str) -> str:
    suites = sorted({it["suite"] for it in its} - {"preflight"})
    return "\n".join([
        f"The self-hosted runner had no usable GPU: {why or 'see the environment in the report'}.",
        f"{len(suites)} GPU suite(s) did not run: {', '.join(suites) or '(none planned)'}. "
        "Only the suites with `needs_gpu: false` ran, and the night was not recorded as tested, "
        "so the next poll tests the release again.",
        f"No GPU for {nights} night(s). Report: {report_url or '(see the workflow run)'}", "",
        "Check the runner host: is the amdgpu driver loaded (`/dev/kfd`, `/dev/dri/renderD*`, "
        "`/sys/class/kfd/kfd/topology/nodes`)? Does the build target its GPU (`build_tools/detect_gpu.sh "
        "--manifest <manifest>`)?"])


def regression_issues(items: list[dict], state: dict, suites_ran: set[str], report_url: str,
                      no_gpu: str | None = None) -> tuple[list, dict]:
    """One issue per (suite, group) with red results; close after 3 green nights.

    The ``<suite>::infra::no-gpu`` items share one issue (NO_GPU_KEY); it counts
    a green night whenever ``suites_ran`` contains "runner" (a night with a GPU)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["class"] in RED:
            s, g, _ = split_id(it["id"])
            groups[NO_GPU_KEY if it["id"].endswith("::infra::no-gpu") else f"{s}::{g}"].append(it)
    actions, new_state = [], {}
    for key, its in sorted(groups.items()):
        fp = hashlib.sha1(key.encode()).hexdigest()[:10]
        nights = max(it.get("nights_failing", 1) for it in its)
        if key == NO_GPU_KEY:
            actions.append({"action": "open_or_comment", "fingerprint": fp, "key": key, "nights": nights,
                            "title": f"vision-pack QA infrastructure: runner had no usable GPU [{fp}]",
                            "body": no_gpu_issue_body(its, no_gpu, nights, report_url)})
            new_state[fp] = {"key": key, "green_streak": 0, "last_red": dt.date.today().isoformat()}
            continue
        lines = [f"- `{it['id']}` ({it['status']}): {it.get('message', '')[:200]}" for it in its[:50]]
        if len(its) > 50:
            lines.append(f"- ... and {len(its) - 50} more")
        repro = next((it.get("repro") for it in its if it.get("repro")), "")
        body = "\n".join([
            f"{len(its)} result(s) in `{key}` are failing and are not covered by `baselines/known_issues.yaml`.",
            f"Failing for {nights} night(s). Report: {report_url or '(see the workflow run)'}", "", *lines])
        if repro:
            body += f"\n\nReproduce (first failure):\n```\n{repro}\n```"
        body += ("\n\nEither fix the regression, or add a known-issue entry (with an upstream issue link) "
                 "if it is an accepted upstream bug.")
        actions.append({"action": "open_or_comment", "fingerprint": fp, "key": key, "nights": nights,
                        "title": f"vision-pack QA regression: {key} [{fp}]", "body": body})
        new_state[fp] = {"key": key, "green_streak": 0, "last_red": dt.date.today().isoformat()}
    for fp, st in state.items():
        if fp in new_state:
            continue
        suite = st.get("key", "").split("::", 1)[0]
        streak = st.get("green_streak", 0) + (1 if suite in suites_ran else 0)
        if streak >= 3:
            actions.append({"action": "close", "fingerprint": fp, "key": st.get("key", ""),
                            "title": f"vision-pack QA regression: {st.get('key', '')} [{fp}]",
                            "body": f"Green for {streak} consecutive nights; closing. Report: {report_url}"})
        else:
            new_state[fp] = {**st, "green_streak": streak}
    return actions, new_state


# --------------------------------------------------------------------------- main triage

def triage(merged: Path, known_doc: dict, expected_cfg: dict, perf_policy: dict, hist: dict,
           plan: dict, tier: str, gfx: str, night: str, run_url: str, report_url: str) -> dict:
    records = load_records(merged)
    suites = json.loads((merged / "suites.json").read_text()) if (merged / "suites.json").exists() else {}
    env = json.loads((merged / "environment.json").read_text()) if (merged / "environment.json").exists() else {}
    no_gpu = no_gpu_reason(env)
    # A night without a usable GPU matches baseline entries scoped to gfx: [none].
    known = Known(known_doc, gfx or ("none" if no_gpu is not None else ""), tier)
    prev = hist["prev"]
    have_prev = bool(prev)

    classes: Counter = Counter()
    items: list[dict] = []
    status_out: dict[str, list] = {}
    per_suite: dict[str, Counter] = defaultdict(Counter)
    per_group: dict[tuple[str, str], Counter] = defaultdict(Counter)
    known_hits: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        rid = r["id"]
        k = known.match(rid)
        p = prev.get(rid)
        cls = classify(r, k, p, have_prev, no_gpu is not None)
        failing_now = r["status"] in FAILING
        nights = (p[1] + 1 if p and p[0] in FAILING else 1) if failing_now else 0
        status_out[rid] = [r["status"], nights]
        classes[cls] += 1
        s, g, _ = split_id(rid)
        per_suite[s][cls] += 1
        per_group[(s, g)][r["status"]] += 1
        if k:
            known_hits[k["id"]][r["status"]] += 1
        if cls not in ("pass", "new_test") or failing_now:
            items.append({"id": rid, "suite": s, "group": g, "status": r["status"], "class": cls,
                          "known": k["id"] if k else None, "message": (r.get("message") or "")[:600],
                          "repro": r.get("repro", ""), "log": r.get("log", ""), "backend": r.get("backend", ""),
                          "result_dir": r.get("result_dir", ""), "prev": p[0] if p else None,
                          "nights_failing": nights, "duration_s": r.get("duration_s", 0)})

    # Removed tests: in the previous comparable night, gone now, in a suite that ran.
    ran = {s for s, m in suites.items() if not m.get("missing")}
    current = {r["id"] for r in records}
    removed = sorted(rid for rid in prev if rid not in current and split_id(rid)[0] in ran)

    counts = expected_count_items(expected_cfg, records, suites, tier)
    count_fail = [c for c in counts if not c["ok"]]
    perf = perf_eval(merged, perf_policy, hist["perf"], night)

    known_rows = []
    for e in known_doc.get("issues", []):
        hits = known_hits.get(e["id"], Counter())
        failing = hits["fail"] + hits["error"] + hits["flaky"]
        if failing:
            state = "reproduced"
        elif hits["pass"]:
            # A race that did not trigger tonight is not a fix.
            state = "not-reproduced" if e.get("kind") == "flaky" else "fixed"
        else:
            state = "not-run"
        in_scope = any(x is e for x in known.entries)
        known_rows.append({"id": e["id"], "title": e.get("title", ""), "severity": e.get("severity", ""),
                           "owner": e.get("owner", ""), "kind": e.get("kind", "xfail"),
                           "issue": e.get("issue", ""), "review_by": str(e.get("review_by", "")),
                           "strict": bool(e.get("strict", False)), "state": state if in_scope else "out-of-scope",
                           "failing": failing, "passing": hits["pass"], "not_run": hits["skip"] + hits["blocked"]})
    fixed_strict = [k for k in known_rows if k["state"] == "fixed" and k["strict"]]

    reasons, verdict = [], "green"
    red_counts = {c: classes[c] for c in RED if classes[c]}
    no_gpu_items = [i for i in items if i["class"] == "infra_error" and i["id"].endswith("::infra::no-gpu")]
    if no_gpu_items:
        skipped = len({i["suite"] for i in no_gpu_items} - {"preflight"})
        reasons.append(f"runner had no usable GPU ({no_gpu or 'see the environment'})"
                       + (f": {skipped} GPU suite{'' if skipped == 1 else 's'} did not run" if skipped else ""))
        red_counts["infra_error"] -= len(no_gpu_items)
        red_counts = {c: n for c, n in red_counts.items() if n}
    if red_counts:
        reasons += [f"{n} {c.replace('_', ' ')}" for c, n in sorted(red_counts.items())]
    if count_fail:
        reasons += [f"{c['suite']}::{c['group']}: {c['actual']} results, expected at least {c['min']}"
                    for c in count_fail]
    if perf["hard"]:
        reasons.append("hard performance regression")
    if reasons:
        verdict = "red"
    else:
        yellow = {c: classes[c] for c in YELLOW if classes[c]}
        if yellow or removed or perf["soft"]:
            verdict = "yellow"
            reasons += [f"{n} {c.replace('_', ' ')}" for c, n in sorted(yellow.items())]
            if removed:
                reasons.append(f"{len(removed)} removed tests")
            if perf["soft"]:
                reasons.append("soft performance drop")
    if fixed_strict:
        reasons.append("fixed known issues to remove from the baseline: " + ", ".join(k["id"] for k in fixed_strict))

    # A night with a GPU counts towards closing the runner::no-gpu issue.
    had_gpu = {"runner"} if gfx and no_gpu is None else set()
    actions, issues_state = regression_issues(items, hist["issues_state"], ran | had_gpu, report_url, no_gpu)

    suite_rows = []
    for s in sorted(set(per_suite) | set(suites)):
        meta = suites.get(s, {})
        c = per_suite.get(s, Counter())
        suite_rows.append({"suite": s, "missing": bool(meta.get("missing")), "no_gpu": bool(meta.get("no_gpu")),
                           "wall_seconds": meta.get("wall_seconds"), "total": sum(c.values()),
                           "classes": dict(c), "status_counts": meta.get("counts", {})})
    group_rows = [{"suite": s, "group": g, "total": sum(c.values()), "counts": dict(c)}
                  for (s, g), c in sorted(per_group.items())]

    return {
        "schema": 1,
        "night": night,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run": {"tag": plan.get("tag", ""), "version": plan.get("version", ""), "sha": plan.get("sha", ""),
                "mode": plan.get("mode", ""), "fingerprint": plan.get("fingerprint", ""), "tier": tier, "gfx": gfx,
                "run_url": run_url, "report_url": report_url, "upstream": plan.get("upstream_nightly", {}),
                "release_url": plan.get("release_url", ""), "problems": plan.get("problems", [])},
        "environment": env,
        "no_gpu": no_gpu,
        "verdict": verdict,
        "reasons": reasons,
        "totals": {"records": len(records), "by_status": dict(Counter(r["status"] for r in records)),
                   "by_class": dict(classes)},
        "suites": suite_rows,
        "groups": group_rows,
        "items": sorted(items, key=lambda i: (i["class"] not in RED, i["suite"], i["id"])),
        "known": known_rows,
        "expected_counts": counts,
        "delta": {"previous_night": hist["prev_night"], "removed": removed[:500], "removed_count": len(removed),
                  **{k: classes[k] for k in ("new_failure", "still_failing", "fixed", "new_test")}},
        "perf": {k: v for k, v in perf.items() if k != "append"},
        "trend": hist["trend"],
        "issue_actions": actions,
        "_status": status_out,
        "_perf_append": perf["append"],
        "_issues_state": issues_state,
    }


# --------------------------------------------------------------------------- outputs

def summary_md(t: dict, top: int = 20) -> str:
    run = t["run"]
    env = t.get("environment", {})
    sdk = env.get("prepared", {}).get("sdk", {})
    no_gpu = t.get("no_gpu")
    gpu = f"`{run['gfx']}`" if run["gfx"] else "no GPU" if no_gpu else "`?`"
    lines = [
        f"## vision-pack QA: {t['verdict'].upper()}",
        "",
        f"vision-pack `{run['version'] or '?'}` ({run['tag'] or run['mode']}, `{(run['sha'] or '')[:12]}`) "
        f"on {gpu}, tier **{run['tier']}**, {t['night']}.",
    ]
    if t["reasons"]:
        lines += ["", "**Why:** " + "; ".join(t["reasons"]) + "."]
    if run.get("report_url"):
        lines += ["", f"Report: {run['report_url']}"]
    lines += ["", "| Suite | results | new failures | still failing | known | flaky | fixed | blocked | skipped |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in t["suites"]:
        c = s["classes"]
        if s["missing"]:
            lines.append(f"| {s['suite']} | {'not run: no GPU' if s.get('no_gpu') else 'no results'} | | | | | | | |")
            continue
        lines.append(f"| {s['suite']} | {s['total']} | {c.get('new_failure', 0)} | {c.get('still_failing', 0)} | "
                     f"{c.get('known_fail', 0) + c.get('known_flaky', 0)} | {c.get('flaky', 0)} | {c.get('fixed', 0)} | "
                     f"{c.get('blocked', 0) + c.get('known_blocked', 0)} | {c.get('skip', 0)} |")
    red = [i for i in t["items"] if i["class"] in RED]
    if red:
        lines += ["", f"### New and unbaselined failures ({len(red)})", ""]
        for i in red[:top]:
            nights = f", failing {i['nights_failing']} nights" if i["nights_failing"] > 1 else ""
            lines.append(f"- `{i['id']}` ({i['status']}{nights}): {i['message'][:160]}")
            if i.get("repro"):
                lines.append(f"  - repro: `{i['repro'][:300]}`")
        if len(red) > top:
            lines.append(f"- ... and {len(red) - top} more (see the report)")
    bad_counts = [c for c in t["expected_counts"] if not c["ok"]]
    if bad_counts:
        lines += ["", "### Missing tests", ""]
        lines += [f"- `{c['suite']}::{c['group']}`: {c['actual']} of at least {c['min']} {c['note']}" for c in bad_counts]
    fixed = [k for k in t["known"] if k["state"] == "fixed"]
    if fixed:
        lines += ["", "### Known issues that passed tonight", ""]
        lines += [f"- {k['id']}: {k['title']}" for k in fixed]
    up = run.get("upstream") or {}
    lines += ["", "### Environment", "",
              f"- GPU: none ({no_gpu})" if no_gpu else f"- GPU: {gpu}",
              f"- SDK: `{sdk.get('name', '?')}` (fallback: {sdk.get('fallback', '?')})",
              f"- Runner: {env.get('environment', {}).get('runner', '?')}, driver "
              f"{env.get('environment', {}).get('amdgpu_driver', '?')}, kernel {env.get('environment', {}).get('kernel', '?')}",
              f"- Upstream nightly: {up.get('conclusion') or up.get('status', '?')} {up.get('url', '')}"]
    if t["delta"]["previous_night"]:
        d = t["delta"]
        lines.append(f"- Compared with {d['previous_night']}: {d['new_failure']} new failures, {d['fixed']} fixed, "
                     f"{d['new_test']} new tests, {d['removed_count']} removed")
    return "\n".join(lines) + "\n"


def write_outputs(t: dict, a: argparse.Namespace) -> None:
    public = {k: v for k, v in t.items() if not k.startswith("_")}
    if a.out:
        Path(a.out).write_text(json.dumps(public, indent=1) + "\n", encoding="utf-8")
    if a.summary:
        Path(a.summary).write_text(summary_md(t), encoding="utf-8")
    if a.status_out:
        with gzip.open(a.status_out, "wt", encoding="utf-8") as f:
            json.dump(t["_status"], f, separators=(",", ":"))
    if a.perf_out:
        with open(a.perf_out, "w", encoding="utf-8") as f:
            for row in t["_perf_append"]:
                f.write(json.dumps(row) + "\n")
    if a.issues_state_out:
        Path(a.issues_state_out).write_text(json.dumps(t["_issues_state"], indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--merged", help="output of build_tools/results/merge.py")
    src.add_argument("--results", help="raw suite result directories (merged on the fly)")
    ap.add_argument("--expect", default="", help="with --results: suites that must be present")
    ap.add_argument("--known", default=str(HERE.parent / "baselines" / "known_issues.yaml"))
    ap.add_argument("--expected", default=str(HERE.parent / "baselines" / "expected_counts.yaml"))
    ap.add_argument("--perf-policy", default=str(HERE.parent / "baselines" / "perf_policy.yaml"))
    ap.add_argument("--history", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--tier", default="")
    ap.add_argument("--gfx", default="")
    ap.add_argument("--night", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    ap.add_argument("--run-url", default="")
    ap.add_argument("--report-url", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--status-out", default="")
    ap.add_argument("--perf-out", default="")
    ap.add_argument("--issues-state-out", default="")
    a = ap.parse_args()

    if a.results:
        from merge import merge
        tmp = Path(tempfile.mkdtemp(prefix="vp-merged-"))
        merge(Path(a.results), tmp, [s for s in a.expect.split(",") if s])
        merged = tmp
    else:
        merged = Path(a.merged)
    plan = json.loads(Path(a.plan).read_text()) if a.plan and Path(a.plan).exists() else {}
    env = json.loads((merged / "environment.json").read_text()) if (merged / "environment.json").exists() else {}
    gfx = a.gfx or env.get("prepared", {}).get("gfx", "") or env.get("preflight", {}).get("gfx", "")
    if not gfx:
        suites = json.loads((merged / "suites.json").read_text()) if (merged / "suites.json").exists() else {}
        gfx = next((m.get("gfx") for m in suites.values() if m.get("gfx")), "")
    tier = a.tier or plan.get("tier", "") or "comprehensive"

    hist = load_history(Path(a.history) if a.history else None, tier, gfx, a.night)
    known_doc = load_yaml(a.known, {})
    known_doc["_base"] = str(Path(a.known).resolve().parent)
    t = triage(merged, known_doc, load_yaml(a.expected, {}), load_yaml(a.perf_policy, {}),
               hist, plan, tier, gfx, a.night, a.run_url, a.report_url)
    write_outputs(t, a)
    print(f"verdict: {t['verdict']} ({'; '.join(t['reasons']) or 'all clear'})")
    print(f"classes: {dict(sorted(t['totals']['by_class'].items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
