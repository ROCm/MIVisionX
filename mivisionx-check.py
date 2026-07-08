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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
MIVisionX 4.0 pre-install environment check.

Detects and reports the system environment relevant to building MIVisionX:
OS, compiler, ROCm, HIP, OpenCL, RPP, OpenMP, OpenCV, OpenSSL, and GPU devices.
No arguments required -- run it and read the output.

Usage:
    python3 mivisionx-check.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colours (disabled automatically when stdout is not a tty)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def bold(t):   return _c("1",  t)
def cyan(t):   return _c("36", t)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, ""


def find_file(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def header(title: str) -> None:
    print()
    print(bold(f"-- {title} " + "-" * max(0, 60 - len(title))))


def found(label: str, detail: str = "") -> None:
    tag = green("  [FOUND]  ")
    print(f"{tag}{label}" + (f"  ({detail})" if detail else ""))


def info(label: str) -> None:
    tag = cyan("  [INFO]   ")
    print(f"{tag}{label}")


def notfound(label: str) -> None:
    tag = yellow("  [--]     ")
    print(f"{tag}{label}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def detect_rocm_path() -> Path:
    return Path(os.environ.get("ROCM_PATH", "/opt/rocm"))


def check_os() -> None:
    header("Operating System")
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/etc/os-release") as f:
                os_info = dict(line.strip().split("=", 1) for line in f if "=" in line)
            name = os_info.get("NAME", "Linux").strip('"')
            version = os_info.get("VERSION_ID", "").strip('"')
            found(f"{name} {version}")
        except OSError:
            found("Linux")
    elif sys_name == "Darwin":
        rc, out = run(["sw_vers", "-productVersion"])
        found(f"macOS {out}" if rc == 0 else "macOS")
        info("macOS supports the CPU backend only")
    elif sys_name == "Windows":
        found(f"Windows {platform.version()}")
        info("Windows supports the OpenCL backend only (HIP not available)")
    else:
        info(f"OS: {sys_name}")


def check_cmake() -> None:
    header("CMake")
    cmake = shutil.which("cmake")
    if not cmake:
        notfound("cmake not found")
        return
    rc, out = run(["cmake", "--version"])
    version_line = out.splitlines()[0] if out else ""
    version_str = version_line.replace("cmake version", "").strip()
    found(f"cmake {version_str}", cmake)


def check_compiler(rocm_path: Path) -> None:
    header("C++17 Compiler")
    amdclang = rocm_path / "lib" / "llvm" / "bin" / "amdclang++"
    if amdclang.exists():
        rc, out = run([str(amdclang), "--version"])
        version_line = out.splitlines()[0] if out else ""
        found("amdclang++", str(amdclang))
        if version_line:
            info(version_line)
        return
    for compiler in ("clang++", "g++", "c++"):
        path = shutil.which(compiler)
        if path:
            found(compiler, path)
            info("amdclang++ not found -- GPU builds require the ROCm Core SDK")
            return
    notfound("No C++17 compiler found")


def check_rocm(rocm_path: Path) -> None:
    header("ROCm Core SDK")
    if not rocm_path.exists():
        notfound(f"ROCm not found at {rocm_path}")
        return
    ver_file = rocm_path / ".info" / "version"
    if ver_file.exists():
        version = ver_file.read_text().strip()
        found(f"ROCm {version}", str(rocm_path))
    else:
        found(f"ROCm found at {rocm_path}", "version file not present")


def check_hip(rocm_path: Path) -> None:
    header("HIP Runtime")
    hipcc = rocm_path / "bin" / "hipcc"
    libamdhip = find_file([
        rocm_path / "lib" / "libamdhip64.so",
        rocm_path / "lib64" / "libamdhip64.so",
    ])
    if hipcc.exists():
        rc, out = run([str(hipcc), "--version"])
        version_line = next((l for l in out.splitlines() if "HIP version" in l), "")
        found("hipcc", str(hipcc))
        if version_line:
            info(version_line)
    else:
        notfound(f"hipcc not found at {hipcc}")
    if libamdhip:
        found("libamdhip64.so", str(libamdhip))
    else:
        notfound("libamdhip64.so not found")


def check_opencl(rocm_path: Path) -> None:
    header("OpenCL")
    libocl = find_file([
        rocm_path / "lib" / "libOpenCL.so",
        rocm_path / "lib" / "libOpenCL.so.1",
        Path("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"),
        Path("/usr/lib/libOpenCL.so"),
    ])
    cl_header = find_file([
        rocm_path / "include" / "CL" / "cl.h",
        Path("/usr/include/CL/cl.h"),
    ])
    if libocl:
        found("libOpenCL.so", str(libocl))
    else:
        notfound("libOpenCL.so not found")
    if cl_header:
        found("CL/cl.h", str(cl_header))
    else:
        notfound("CL/cl.h not found")


def check_half(rocm_path: Path) -> None:
    header("half (float16 library)")
    half_header = find_file([
        rocm_path / "include" / "half" / "half.hpp",
        rocm_path / "include" / "half.hpp",
        Path("/usr/include/half/half.hpp"),
        Path("/usr/local/include/half/half.hpp"),
    ])
    if half_header:
        found("half.hpp", str(half_header))
    else:
        notfound("half.hpp not found")


def check_rpp(rocm_path: Path) -> None:
    header("RPP -- ROCm Performance Primitives")
    rpp_cmake = find_file([
        rocm_path / "lib" / "cmake" / "rpp" / "rppConfig.cmake",
        rocm_path / "lib" / "cmake" / "rpp-targets.cmake",
        Path("/usr/lib/cmake/rpp/rppConfig.cmake"),
    ])
    rpp_header = find_file([
        rocm_path / "include" / "rpp" / "rpp.h",
        Path("/usr/include/rpp/rpp.h"),
    ])
    rpp_lib = find_file([
        rocm_path / "lib" / "librpp.so",
        Path("/usr/lib/librpp.so"),
        Path("/usr/lib/x86_64-linux-gnu/librpp.so"),
    ])

    if not (rpp_cmake or rpp_header):
        notfound("RPP not found")
        return

    version_str = ""
    if rpp_cmake:
        try:
            import re
            cmake_text = rpp_cmake.read_text()
            for line in cmake_text.splitlines():
                m = re.search(r"(\d+\.\d+\.\d+)", line)
                if m:
                    version_str = m.group(1)
                    break
        except OSError:
            pass

    found(f"RPP{' ' + version_str if version_str else ''}", str(rpp_lib or rpp_header))

    rpp_backend_h = find_file([
        rocm_path / "include" / "rpp" / "rpp_backend.h",
        Path("/usr/include/rpp/rpp_backend.h"),
    ])
    if rpp_backend_h:
        content = rpp_backend_h.read_text()
        has_hip = "RPP_BACKEND_HIP 1" in content
        info(f"HIP backend: {'enabled' if has_hip else 'not enabled'}")


def check_openmp(rocm_path: Path) -> None:
    header("OpenMP")
    libomp = find_file([
        rocm_path / "lib" / "libomp.so",
        rocm_path / "llvm" / "lib" / "libomp.so",
        Path("/usr/lib/llvm-18/lib/libomp.so"),
        Path("/usr/lib/x86_64-linux-gnu/libomp.so"),
        Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1"),
    ])
    if libomp:
        found("OpenMP library", str(libomp))
    else:
        notfound("OpenMP library not found in standard locations")
        info("May be provided transitively by the ROCm LLVM toolchain")


def check_openssl() -> None:
    header("OpenSSL / libcrypto (optional -- RunVX checksums)")
    if platform.system() == "Windows":
        info("Windows uses wincrypt.h -- OpenSSL not required")
        return
    ssl_header = find_file([
        Path("/usr/include/openssl/hmac.h"),
        Path("/usr/local/include/openssl/hmac.h"),
        Path("/usr/local/opt/openssl/include/openssl/hmac.h"),
    ])
    libcrypto = find_file([
        Path("/usr/lib/x86_64-linux-gnu/libcrypto.so"),
        Path("/usr/lib/libcrypto.so"),
        Path("/usr/local/lib/libcrypto.so"),
    ])
    if ssl_header and libcrypto:
        found("OpenSSL", str(ssl_header.parent))
    elif ssl_header:
        found("OpenSSL headers", str(ssl_header))
    else:
        notfound("OpenSSL not found -- RunVX will use built-in checksum fallback")


def check_opencv() -> None:
    header("OpenCV (optional -- RunVX display)")
    rc, out = run(["pkg-config", "--modversion", "opencv4"])
    if rc == 0:
        found(f"OpenCV {out.strip()}", "via pkg-config (opencv4)")
        return
    rc, out = run(["pkg-config", "--modversion", "opencv"])
    if rc == 0:
        found(f"OpenCV {out.strip()}", "via pkg-config")
        return
    cv_header = find_file([
        Path("/usr/include/opencv4/opencv2/opencv.hpp"),
        Path("/usr/local/include/opencv4/opencv2/opencv.hpp"),
        Path("/usr/include/opencv2/opencv.hpp"),
    ])
    if cv_header:
        found("OpenCV headers", str(cv_header))
        return
    notfound("OpenCV not found -- RunVX will be built without display support")


def check_gpu_devices() -> None:
    header("GPU Devices")
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        notfound("rocminfo not found -- cannot enumerate GPU devices")
        return
    rc, out = run([rocminfo])
    if rc != 0:
        info("rocminfo returned an error")
        return
    agents = [l.strip() for l in out.splitlines() if "Name:" in l and "gfx" in l.lower()]
    if agents:
        for agent in agents:
            found(f"GPU: {agent.replace('Name:', '').strip()}")
    else:
        notfound("No AMD GPU agents detected by rocminfo")


def check_python() -> None:
    header("Python")
    version = sys.version.split()[0]
    found(f"Python {version}", sys.executable)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rocm_path = detect_rocm_path()

    print()
    print(bold("MIVisionX 4.0 -- Pre-Install Environment Check"))
    print(f"  ROCm path : {rocm_path}")
    print(f"  OS        : {platform.system()} {platform.release()}")

    check_os()
    check_cmake()
    check_compiler(rocm_path)
    check_rocm(rocm_path)
    check_hip(rocm_path)
    check_opencl(rocm_path)
    check_half(rocm_path)
    check_rpp(rocm_path)
    check_openmp(rocm_path)
    check_openssl()
    check_opencv()
    check_gpu_devices()
    check_python()

    print()
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
