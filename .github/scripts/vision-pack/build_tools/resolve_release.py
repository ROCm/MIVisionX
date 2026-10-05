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

"""Decide what tonight's QA run tests.

Release mode (default): pick the newest ``nightly-YYYYMMDD`` prerelease of
vision-pack (``/releases/latest`` skips prereleases, so the list is scanned),
check that the release's ``target_commitish``, the tag's commit and the
version suffix ``+g<sha>`` of the assets agree, count the assets, and compute
the run fingerprint. Build mode: resolve ``--ref`` to a commit instead.

    resolve_release.py [--repo kiritigowda/vision-pack] [--tag nightly-20260926]
        [--mode release|build] [--ref main] [--tier comprehensive]
        [--sdk-family F] [--sdk-date YYYYMMDD] [--sdk-url URL]
        [--tested-file FILE] [--force] [--max-age-days 3]
        [--out plan.json] [--github]

The fingerprint covers the mode, tag or commit, tier, SDK override and every
asset digest, so re-published assets or a different tier are tested again.
--tested-file optionally names a file of already-tested fingerprints (one per
line, '#' comments); without it nothing counts as tested. If the fingerprint
is listed there and --force is off, the plan says skip. Problems (SHA
disagreement, missing assets) do not stop the run; they are reported so the
report can mark the night as an infrastructure failure. Uses
GH_TOKEN/GITHUB_TOKEN when set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
TAG_RE = re.compile(r"^nightly-(\d{8})$")
KINDS = {
    "deb": re.compile(r"^amdrocm-.*\.deb$"),
    "rpm": re.compile(r"^amdrocm-.*\.rpm$"),
    "tarball": re.compile(r"^vision-pack-dist-linux-multiarch-.*\.tar\.gz$"),
}
VERSION_RE = re.compile(r"^vision-pack-dist-linux-multiarch-(.+)\.tar\.gz$")


def api(path: str, retries: int = 4):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "vision-pack-qa"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = path if path.startswith("http") else f"{API}{path}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def tag_commit(repo: str, tag: str) -> str:
    ref = api(f"/repos/{repo}/git/ref/tags/{tag}")["object"]
    while ref["type"] == "tag":  # annotated tag: follow to the commit
        ref = api(f"/repos/{repo}/git/tags/{ref['sha']}")["object"]
    return ref["sha"]


def newest_nightly(repo: str) -> dict:
    releases = api(f"/repos/{repo}/releases?per_page=50")
    nightlies = [r for r in releases if TAG_RE.match(r["tag_name"]) and not r.get("draft")]
    if not nightlies:
        raise SystemExit(f"::error::no nightly-YYYYMMDD release found in {repo}")
    return max(nightlies, key=lambda r: r["tag_name"])


def upstream_nightly(repo: str, sha: str, published_at: str | None) -> dict:
    """Upstream's nightly run that published this release (its validation
    continues after publishing). Later nightlies on the same commit that
    published nothing are ignored; with no timestamp, the newest match wins."""
    try:
        runs = api(f"/repos/{repo}/actions/workflows/nightly.yml/runs?per_page=30")["workflow_runs"]
    except Exception as e:  # informational only
        return {"status": "unknown", "error": str(e)}
    matching = [r for r in runs if r.get("head_sha") == sha]
    before = [r for r in matching if published_at and r.get("created_at", "") <= published_at]
    pick = max(before or matching, key=lambda r: r.get("created_at", ""), default=None)
    if pick is None:
        return {"status": "not-found"}
    return {"status": pick.get("status"), "conclusion": pick.get("conclusion"), "url": pick.get("html_url"),
            "run_id": pick.get("id"), "created_at": pick.get("created_at")}


def fingerprint(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def read_tested(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.split()[0] for line in f if line.strip() and not line.startswith("#")}


def plan_release(a: argparse.Namespace) -> dict:
    rel = api(f"/repos/{a.repo}/releases/tags/{a.tag}") if a.tag else newest_nightly(a.repo)
    tag = rel["tag_name"]
    m = TAG_RE.match(tag)
    date = m.group(1) if m else ""
    problems: list[str] = []

    assets = {k: [] for k in KINDS}
    for asset in rel.get("assets", []):
        for kind, rx in KINDS.items():
            if rx.match(asset["name"]):
                assets[kind].append({"name": asset["name"], "size": asset["size"],
                                     "digest": asset.get("digest") or "", "state": asset.get("state"),
                                     "url": asset["browser_download_url"]})
    for kind, want in (("deb", a.expect_deb), ("rpm", a.expect_rpm), ("tarball", a.expect_tarball)):
        if len(assets[kind]) != want:
            problems.append(f"expected {want} {kind} assets, found {len(assets[kind])}")
    for kind in KINDS:
        for asset in assets[kind]:
            if not asset["digest"].startswith("sha256:"):
                problems.append(f"asset {asset['name']} has no sha256 digest")
            if asset["state"] != "uploaded":
                problems.append(f"asset {asset['name']} is {asset['state']}")

    version = ""
    if assets["tarball"]:
        vm = VERSION_RE.match(assets["tarball"][0]["name"])
        version = vm.group(1) if vm else ""

    # SHA triple: release target, tag commit, and the +g<sha> version suffix
    # (the manifest sha inside the tarball is checked by fetch_release.sh).
    target = rel.get("target_commitish", "")
    try:
        commit = tag_commit(a.repo, tag)
    except Exception as e:
        commit = ""
        problems.append(f"could not resolve tag {tag}: {e}")
    if re.fullmatch(r"[0-9a-f]{40}", target) and commit and target != commit:
        problems.append(f"release target_commitish {target} != tag commit {commit}")
    gm = re.search(r"\+g([0-9a-f]{7,40})$", version)
    if commit and gm and not commit.startswith(gm.group(1)):
        problems.append(f"asset version {version} does not match tag commit {commit}")
    sha = commit or (target if re.fullmatch(r"[0-9a-f]{40}", target) else "")

    digests = sorted(f"{x['name']}={x['digest']}" for k in KINDS for x in assets[k])
    fp = fingerprint(["mode=release", f"tag={tag}", f"tier={a.tier}",
                      f"sdk={a.sdk_family}|{a.sdk_date}|{a.sdk_url}", *digests])

    age_days = None
    if date:
        published = dt.datetime.strptime(date, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        age_days = (dt.datetime.now(dt.timezone.utc) - published).days
    return {
        "mode": "release", "tag": tag, "date": date, "version": version, "sha": sha,
        "target_commitish": target, "tag_commit": commit, "release_url": rel.get("html_url"),
        "published_at": rel.get("published_at"), "prerelease": rel.get("prerelease"),
        "assets": {k: len(v) for k, v in assets.items()}, "asset_list": assets,
        "fingerprint": fp, "problems": problems, "age_days": age_days,
        "stale": age_days is not None and age_days > a.max_age_days,
        "upstream_nightly": upstream_nightly(a.repo, sha, rel.get("published_at")) if sha else {"status": "unknown"},
    }


def plan_build(a: argparse.Namespace) -> dict:
    commit = api(f"/repos/{a.repo}/commits/{a.ref}")["sha"]
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    fp = fingerprint(["mode=build", f"sha={commit}", f"tier={a.tier}", f"sdk={a.sdk_family}|{a.sdk_date}|{a.sdk_url}"])
    return {"mode": "build", "tag": "", "date": today, "version": "", "sha": commit, "ref": a.ref,
            "fingerprint": fp, "problems": [], "age_days": 0, "stale": False,
            "assets": {}, "asset_list": {}, "upstream_nightly": {"status": "not-applicable"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="kiritigowda/vision-pack")
    ap.add_argument("--mode", choices=["release", "build"], default="release")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--tier", default="comprehensive")
    ap.add_argument("--sdk-family", default="")
    ap.add_argument("--sdk-date", default="")
    ap.add_argument("--sdk-url", default="")
    ap.add_argument("--tested-file", default="", help="optional file of already-tested fingerprints (default: none)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-age-days", type=int, default=3)
    ap.add_argument("--expect-deb", type=int, default=16)
    ap.add_argument("--expect-rpm", type=int, default=16)
    ap.add_argument("--expect-tarball", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--github", action="store_true")
    a = ap.parse_args()

    plan = plan_build(a) if a.mode == "build" else plan_release(a)
    plan["tier"] = a.tier
    plan["sdk_override"] = {"family": a.sdk_family, "date": a.sdk_date, "url": a.sdk_url}
    tested = plan["fingerprint"] in read_tested(a.tested_file)
    plan["already_tested"] = tested
    plan["skip"] = tested and not a.force
    plan["skip_reason"] = "already tested (fingerprint listed in --tested-file)" if plan["skip"] else ""

    text = json.dumps(plan, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    summary = {k: v for k, v in plan.items() if k != "asset_list"}
    print(json.dumps(summary, indent=2))
    for p in plan["problems"]:
        print(f"::error::{p}", file=sys.stderr)
    if plan["stale"]:
        print(f"::warning::the newest vision-pack nightly ({plan['tag']}) is {plan['age_days']} days old",
              file=sys.stderr)

    if a.github:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for key in ("mode", "tag", "date", "version", "sha", "fingerprint"):
                f.write(f"{key}={plan[key]}\n")
            f.write(f"skip={'true' if plan['skip'] else 'false'}\n")
            f.write(f"stale={'true' if plan['stale'] else 'false'}\n")
            f.write(f"problems={len(plan['problems'])}\n")
            f.write(f"upstream_conclusion={plan['upstream_nightly'].get('conclusion') or plan['upstream_nightly'].get('status')}\n")
        with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as f:
            f.write("### What tonight tests\n\n| | |\n|---|---|\n")
            if plan["mode"] == "release":
                f.write(f"| Release | [{plan['tag']}]({plan['release_url']}) ({plan['version']}) |\n")
                f.write(f"| Assets | {plan['assets']} |\n")
            else:
                f.write(f"| Build | `{plan['ref']}` |\n")
            f.write(f"| Commit | `{plan['sha']}` |\n| Tier | {plan['tier']} |\n")
            f.write(f"| Fingerprint | `{plan['fingerprint']}` |\n")
            up = plan["upstream_nightly"]
            f.write(f"| Upstream nightly | {up.get('conclusion') or up.get('status')} {up.get('url') or ''} |\n")
            f.write(f"| Decision | {'skip: ' + plan['skip_reason'] if plan['skip'] else 'test'} |\n")
            for p in plan["problems"]:
                f.write(f"\n> **Problem:** {p}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
