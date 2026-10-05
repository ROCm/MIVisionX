#!/usr/bin/env bash
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

# Build the per-run ROCm prefix on the GPU runner: TheRock "-tests" SDK for
# the detected GPU family, with the vision-pack dist tarball overlaid on top.
#
#   prepare_rocm.sh --tarball <vision-pack-dist-linux-multiarch-*.tar.gz> \
#       --run-dir "$RUNNER_TEMP/vp/run" (--gfx gfx1201 | --no-gpu) \
#       [--cache "${RUNNER_TEMP:-/tmp}/vp/cache/sdk"] [--family gfx120X-all-tests] \
#       [--date YYYYMMDD] [--url <sdk tarball url>] [--resolve-only] [--github]
#
# Result: <run-dir>/rocm (the prefix; suites must treat it as read-only) and
# <run-dir>/prepared.json (what was used, including the fallback taken).
#
# The SDK is looked up with <repo>/vision-pack/build_tools/fetch_rocm_sdk.py,
# where <repo> is the directory above this script's build_tools/: a checkout of
# kiritigowda/vision-pack (build_tools/ is enough) at the tested release commit.
#
# The SDK must match the GPU: vision-pack is built against gfx94X-dcgpu-tests,
# but that SDK carries no RDNA code objects for the ROCm libraries, so the
# runtime SDK is chosen per family, from the same nightly date as the build
# SDK recorded in the manifest (rocm_sdk). Fallbacks, in order: the latest SDK
# of the family, multiarch-tests from that date, the latest multiarch-tests.
# Both trees must share one prefix because every RUNPATH is $ORIGIN-relative.
#
# --no-gpu is for a runner without a usable GPU: the family is the build SDK's
# own (from rocm_sdk, same fallbacks), the dist_amdgpu_targets check is
# skipped, and prepared.json records "mode": "no-gpu" with an empty gfx. Only
# the needs_gpu: false suites run against such a prefix.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH="${REPO}/vision-pack/build_tools/fetch_rocm_sdk.py"
tarball="" run_dir="" gfx="${VP_GFX:-}" cache="${VP_SDK_CACHE:-${RUNNER_TEMP:-/tmp}/vp/cache/sdk}"
family="" date="" url="" resolve_only=0 github=0 no_gpu=0 gfx_arg=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tarball) tarball="$2"; shift 2 ;;
    --run-dir) run_dir="$2"; shift 2 ;;
    --gfx) gfx="$2"; gfx_arg=1; shift 2 ;;
    --no-gpu) no_gpu=1; shift ;;
    --cache) cache="$2"; shift 2 ;;
    --family) family="$2"; shift 2 ;;
    --date) date="$2"; shift 2 ;;
    --url) url="$2"; shift 2 ;;
    --resolve-only) resolve_only=1; shift ;;
    --github) github=1; shift ;;
    -h|--help) sed -n '/^# Build the per-run ROCm prefix/,/^set -euo pipefail/{/^set /!p}' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

log() { printf '[prepare-rocm] %s\n' "$*" >&2; }
die() { printf '::error::prepare-rocm: %s\n' "$*" >&2; exit 1; }

[[ -n "${tarball}" && -f "${tarball}" ]] || die "--tarball must name the vision-pack dist tarball"
mode=gpu
if [[ "${no_gpu}" == 1 ]]; then
  [[ "${gfx_arg}" == 0 ]] || die "--no-gpu and --gfx are mutually exclusive"
  mode=no-gpu gfx=""
else
  [[ -n "${gfx}" ]] || die "--gfx (or VP_GFX) is required; run detect_gpu.sh first, or pass --no-gpu"
fi
[[ "${resolve_only}" == 1 || -n "${run_dir}" ]] || die "--run-dir is required"
[[ -f "${FETCH}" ]] || die "${FETCH} missing: check out kiritigowda/vision-pack (build_tools/) at the release commit into ${REPO}/vision-pack"
command -v jq >/dev/null || die "jq is required"

# gfx -> TheRock "-tests" family. The -tests variants bundle rpp, rocDecode,
# rocJPEG and the rocdecode utils that the vision libraries need.
family_for_gfx() {
  case "$1" in
    gfx1200|gfx1201) echo gfx120X-all-tests ;;
    gfx1100|gfx1101|gfx1102|gfx1103) echo gfx110X-all-tests ;;
    gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036) echo gfx103X-all-tests ;;
    gfx1150|gfx1151|gfx1152|gfx1153) echo "$1-tests" ;;
    gfx1250|gfx1251) echo gfx125X-dcgpu-tests ;;
    gfx940|gfx941|gfx942) echo gfx94X-dcgpu-tests ;;
    gfx950) echo gfx950-dcgpu-tests ;;
    gfx900|gfx906|gfx908|gfx90a|gfx90c) echo "$1-tests" ;;
    gfx1010|gfx1011|gfx1012) echo gfx101X-dgpu-tests ;;
    *) echo multiarch-tests ;;
  esac
}

# therock-dist-linux-<family>-<version>.tar.gz (URL or name) -> family
family_of_sdk() {
  sed -nE 's/.*therock-dist-linux-(.+)-[0-9]+\.[0-9]+\.[0-9]+[a-z0-9.]*\.tar\.gz$/\1/p' <<<"${1##*/}"
}

# The manifest travels inside the dist tarball.
manifest_json="$(tar -xzOf "${tarball}" ./share/vision-pack/vision-pack-manifest.json 2>/dev/null \
  || tar -xzOf "${tarball}" share/vision-pack/vision-pack-manifest.json 2>/dev/null)" \
  || die "no share/vision-pack/vision-pack-manifest.json in ${tarball}"
vp_sha="$(jq -r '.sha // empty' <<<"${manifest_json}")"
vp_version="$(jq -r '.version // empty' <<<"${manifest_json}")"
build_sdk="$(jq -r '.rocm_sdk // empty' <<<"${manifest_json}")"
if [[ -z "${date}" && "${build_sdk}" =~ a(20[0-9]{6})\.tar\.gz$ ]]; then
  date="${BASH_REMATCH[1]}"
fi
if [[ -n "${family}" ]]; then
  requested_family="${family}"
elif [[ "${mode}" == no-gpu ]]; then
  requested_family="$(family_of_sdk "${build_sdk}")"
  if [[ -z "${requested_family}" ]]; then
    echo "::warning::prepare-rocm: no build SDK family in the manifest's rocm_sdk '${build_sdk}'; using multiarch-tests" >&2
    requested_family=multiarch-tests
  fi
else
  requested_family="$(family_for_gfx "${gfx}")"
fi
log "vision-pack ${vp_version} (${vp_sha:0:12}); build SDK ${build_sdk##*/}; ${gfx:-no GPU} -> ${requested_family}, date ${date:-latest}"

resolve() { # family [date] -> URL on stdout
  local args=(--gpu-family "$1" --print-url) out
  [[ -n "${2:-}" ]] && args+=(--date "$2")
  out="$(python3 "${FETCH}" "${args[@]}" 2>/dev/null | tail -n 1)" || return 1
  [[ "${out}" == https://* ]] || return 1
  printf '%s\n' "${out}"
}

fallback="none"
sdk_family="${requested_family}"
if [[ -n "${url}" ]]; then
  fallback="pinned-url"
  sdk_family="$(family_of_sdk "${url}")"
elif url="$(resolve "${requested_family}" "${date}")"; then
  :
elif url="$(resolve "${requested_family}")"; then
  fallback="family-latest"
elif url="$(resolve multiarch-tests "${date}")"; then
  fallback="multiarch-date"; sdk_family="multiarch-tests"
elif url="$(resolve multiarch-tests)"; then
  fallback="multiarch-latest"; sdk_family="multiarch-tests"
else
  die "could not resolve any TheRock SDK for ${requested_family} (date ${date:-latest}) or multiarch-tests"
fi
sdk_name="${url##*/}"
[[ "${fallback}" == none || "${fallback}" == pinned-url ]] || echo "::warning::prepare-rocm: SDK fallback ${fallback}: ${sdk_name}"
log "SDK: ${sdk_name} (fallback: ${fallback})"

if [[ "${resolve_only}" == 1 ]]; then
  jq -n --arg url "${url}" --arg name "${sdk_name}" --arg family "${sdk_family}" \
    --arg requested "${requested_family}" --arg date "${date}" --arg fallback "${fallback}" \
    --arg gfx "${gfx}" --arg mode "${mode}" --arg vp_sha "${vp_sha}" --arg vp_version "${vp_version}" \
    '{gfx:$gfx, mode:$mode, sdk:{url:$url, name:$name, family:$family, requested_family:$requested,
      date:$date, fallback:$fallback}, vision_pack:{sha:$vp_sha, version:$vp_version}}'
  exit 0
fi

# Content-addressed cache: one flock per tarball, download to .part, record the
# size/ETag the server reported and our own sha256 (TheRock publishes no
# checksums), atomic rename. A cached copy is reused only while the server
# still reports the same ETag and size; mtime is refreshed for the LRU prune.
mkdir -p "${cache}"
sdk_path="${cache}/${sdk_name}"
meta="${sdk_path}.meta.json"
exec {lockfd}>"${sdk_path}.lock"
flock -w 7200 "${lockfd}" || die "timed out waiting for the cache lock on ${sdk_name}"

head="$(curl -fsSIL --retry 5 --retry-delay 5 --max-time 60 "${url}" | tr -d '\r')" || die "HEAD ${url} failed"
size="$(awk 'tolower($1)=="content-length:" {v=$2} END {print v}' <<<"${head}")"
etag="$(awk 'tolower($1)=="etag:" {v=$2} END {print v}' <<<"${head}")"
[[ "${size}" =~ ^[0-9]+$ ]] || die "server reported no Content-Length for ${url}"

cached=0
if [[ -f "${sdk_path}" && -f "${meta}" && "$(stat -c %s "${sdk_path}")" == "${size}" ]] \
   && jq -e --arg e "${etag}" --argjson s "${size}" '.etag == $e and .size == $s' "${meta}" >/dev/null; then
  log "verifying cached ${sdk_name}"
  if [[ "$(sha256sum "${sdk_path}" | cut -d' ' -f1)" == "$(jq -r .sha256 "${meta}")" ]]; then
    cached=1
  else
    log "cached copy is corrupt; downloading again"
  fi
fi
if [[ "${cached}" == 0 ]]; then
  log "downloading ${sdk_name} ($((size / 1048576)) MiB)"
  rm -f "${sdk_path}" "${meta}"
  curl -fL --retry 5 --retry-delay 15 --retry-all-errors --continue-at - --no-progress-meter \
    -o "${sdk_path}.part" "${url}" || die "download of ${url} failed"
  got="$(stat -c %s "${sdk_path}.part")"
  [[ "${got}" == "${size}" ]] || { rm -f "${sdk_path}.part"; die "size mismatch for ${sdk_name}: ${got} != ${size}"; }
  sha="$(sha256sum "${sdk_path}.part" | cut -d' ' -f1)"
  mv "${sdk_path}.part" "${sdk_path}"
  jq -n --arg url "${url}" --arg name "${sdk_name}" --argjson size "${size}" --arg etag "${etag}" \
    --arg sha256 "${sha}" --arg at "$(date -u +%FT%TZ)" \
    '{url:$url, name:$name, size:$size, etag:$etag, sha256:$sha256, downloaded_at:$at}' >"${meta}"
fi
touch "${sdk_path}" "${meta}"
sdk_sha="$(jq -r .sha256 "${meta}")"

prefix="${run_dir}/rocm"
if [[ -e "${prefix}" ]]; then
  log "removing the previous prefix of this run"
  rm -rf "${prefix}"
fi
mkdir -p "${prefix}"
unz=(-z)
command -v pigz >/dev/null && unz=(-I pigz)
log "extracting the SDK into ${prefix}"
tar "${unz[@]}" -xf "${sdk_path}" -C "${prefix}" --strip-components=1 --no-same-owner
flock -u "${lockfd}"

dist_info="${prefix}/share/therock/dist_info.json"
[[ -f "${dist_info}" ]] || die "${sdk_name} has no share/therock/dist_info.json"
targets="$(jq -r '.dist_amdgpu_targets // ""' "${dist_info}")"
if [[ "${mode}" == gpu ]]; then
  [[ ";${targets};" == *";${gfx};"* ]] \
    || die "${sdk_name} was not built for ${gfx} (dist_amdgpu_targets: ${targets}); pass --family multiarch-tests"
fi
missing=()
for f in include/rpp/rpp.h include/rocjpeg/rocjpeg.h share/rocdecode/utils; do
  [[ -e "${prefix}/${f}" ]] || missing+=("${f}")
done
[[ ${#missing[@]} -eq 0 ]] || die "${sdk_name} lacks ${missing[*]}; it is not a -tests SDK"

log "overlaying $(basename "${tarball}")"
tar -xzf "${tarball}" -C "${prefix}" --no-same-owner
[[ -x "${prefix}/bin/runvx" ]] || die "bin/runvx missing or not executable after the overlay"
installed_sha="$(jq -r '.sha // empty' "${prefix}/share/vision-pack/vision-pack-manifest.json")"
[[ "${installed_sha}" == "${vp_sha}" ]] || die "installed manifest sha ${installed_sha} != tarball manifest sha ${vp_sha}"

dist_sha="$(sha256sum "${tarball}" | cut -d' ' -f1)"
jq -n --arg prefix "$(cd "${prefix}" && pwd)" --arg gfx "${gfx}" --arg mode "${mode}" \
  --arg url "${url}" --arg name "${sdk_name}" --arg family "${sdk_family}" --arg requested "${requested_family}" \
  --arg date "${date}" --arg fallback "${fallback}" --arg sha256 "${sdk_sha}" --argjson size "${size}" \
  --arg etag "${etag}" --arg targets "${targets}" --arg cached "${cached}" \
  --arg dist "$(basename "${tarball}")" --arg dist_sha "${dist_sha}" \
  --arg vp_sha "${vp_sha}" --arg vp_version "${vp_version}" --arg build_sdk "${build_sdk}" \
  --arg at "$(date -u +%FT%TZ)" \
  '{prefix:$prefix, gfx:$gfx, mode:$mode, prepared_at:$at,
    sdk:{url:$url, name:$name, family:$family, requested_family:$requested, date:$date, fallback:$fallback,
         sha256:$sha256, size:$size, etag:$etag, dist_amdgpu_targets:$targets, from_cache:($cached=="1")},
    vision_pack:{tarball:$dist, sha256:$dist_sha, sha:$vp_sha, version:$vp_version, build_sdk:$build_sdk}}' \
  >"${run_dir}/prepared.json"
log "prefix ready: ${prefix}"

if [[ "${github}" == 1 && -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "prefix=$(cd "${prefix}" && pwd)"
    echo "sdk_name=${sdk_name}"
    echo "sdk_url=${url}"
    echo "sdk_family=${sdk_family}"
    echo "sdk_fallback=${fallback}"
    echo "sdk_sha256=${sdk_sha}"
    echo "vp_sha=${vp_sha}"
    echo "vp_version=${vp_version}"
    echo "mode=${mode}"
  } >>"${GITHUB_OUTPUT}"
  {
    echo "### ROCm prefix"
    echo "| | |"
    echo "|---|---|"
    if [[ "${mode}" == no-gpu ]]; then
      echo "| GPU | none (no-gpu mode: only the suites that need no GPU run) |"
    else
      echo "| GPU | \`${gfx}\` |"
    fi
    echo "| SDK | \`${sdk_name}\` (fallback: ${fallback}, cached: ${cached}) |"
    echo "| SDK sha256 | \`${sdk_sha}\` |"
    echo "| vision-pack | \`${vp_version}\` (\`${vp_sha:0:12}\`) |"
  } >>"${GITHUB_STEP_SUMMARY:-/dev/null}"
fi
