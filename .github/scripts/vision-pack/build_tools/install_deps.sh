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

# Install the suites' dependencies into the CI job container (Ubuntu 24.04,
# running as root). This mirrors the vision-pack QA test image:
#
#   install_deps.sh [--media]
#
#   - exports the image's environment (DEBIAN_FRONTEND, LANG, PIP_*,
#     PYTHONDONTWRITEBYTECODE) and appends it to $GITHUB_ENV, so later steps
#     of the job see it too;
#   - apt-get installs build_tools/apt-packages.txt (no recommends);
#   - pip installs build_tools/requirements-test.txt (the hash-pinned CPU
#     torch wheel; --break-system-packages because Ubuntu's Python is
#     externally managed, PEP 668) and checks that cv2, numpy and torch import;
#   - lets git accept repositories owned by another user (the checkout is not
#     owned by root).
#
# --media also installs ffmpeg (the rocpydecode suite's media image).
# Safe to run more than once.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
media=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --media) media=1; shift ;;
    -h|--help) sed -n '/^# Install the suites/,/^set -euo pipefail/{/^set /!p}' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

die() { printf '::error::install_deps: %s\n' "$*" >&2; exit 1; }
[[ "$(id -u)" == 0 ]] || die "must run as root (as the CI job container does)"

for kv in DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_DISABLE_PIP_VERSION_CHECK=1 \
          PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1; do
  export "${kv?}"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "${kv}" >>"${GITHUB_ENV}"
  fi
done

mapfile -t packages < <(sed -e 's/#.*//' "${HERE}/apt-packages.txt" | awk 'NF { print $1 }')
[[ ${#packages[@]} -gt 0 ]] || die "no packages listed in ${HERE}/apt-packages.txt"
apt_opts=(-o Acquire::Retries=3)

echo "::group::apt-get install (${#packages[@]} packages)"
apt-get "${apt_opts[@]}" update
apt-get "${apt_opts[@]}" install -y --no-install-recommends "${packages[@]}"
rm -rf /var/lib/apt/lists/*
echo "::endgroup::"

echo "::group::pip install -r requirements-test.txt"
python3 -m pip install --break-system-packages --require-hashes --no-deps \
  -r "${HERE}/requirements-test.txt"
python3 -c 'import cv2, numpy, torch; print("cv2", cv2.__version__, "numpy", numpy.__version__, "torch", torch.__version__)'
echo "::endgroup::"

safe="$(git config --system --get-all safe.directory 2>/dev/null || true)"
if ! grep -qxF '*' <<<"${safe}"; then
  git config --system --add safe.directory '*'
fi
mkdir -p /opt/vp

if [[ "${media}" == 1 ]]; then
  echo "::group::apt-get install ffmpeg"
  apt-get "${apt_opts[@]}" update
  apt-get "${apt_opts[@]}" install -y --no-install-recommends ffmpeg
  rm -rf /var/lib/apt/lists/*
  echo "::endgroup::"
  ffmpeg -hide_banner -version | sed -n 1p
fi
echo "install_deps: done ($(python3 --version 2>&1), media=${media})"
