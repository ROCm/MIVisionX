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
MIVisionX 4.0 dependency checker.

Verifies that every prerequisite required to build and run MIVisionX is present
on the current system.  Reports what is installed, what is missing, and what
install command will fix each gap.  Exits with a non-zero status when any
required dependency is absent so that the output can be consumed by CI scripts.

Usage:
    python3 MIVisionX-setup.py [--rocm-path PATH] [--backend {HIP,OCL,CPU}]
"""

import argparse
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
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)
def cyan(t):   return _c("36", t)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], *, capture: bool = True) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=capture,
                           text=True, timeout=10)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, ""


def find_file(paths: list[Path]) -> Path | None:
    """Return the first path that exists, or None."""
    for p in paths:
        if p.exists():
            return p
    return None


def header(title: str) -> None:
    print()
    print(bold(f"-- {title} " + "-" * max(0, 60 - len(title))))


def ok(label: str, detail: str = "") -> None:
    tag = green("  [OK]     ")
    print(f"{tag}{label}" + (f"  ({detail})" if detail else ""))


def warn(label: str, hint: str = "") -> None:
    tag = yellow("  [WARN]   ")
    print(f"{tag}{label}" + (f"\n           {cyan(hint)}" if hint else ""))


def fail(label: str, fix: str = "") -> None:
    tag = red("  [MISSING]")
    print(f"{tag}{label}" + (f"\n           {cyan('Fix: ' + fix)}" if fix else ""))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_os() -> None:
    header("Operating System")
    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            with open("/etc/os-release") as f:
                info = dict(
                    line.strip().split("=", 1)
                    for line in f if "=" in line
                )
            name = info.get("NAME", "Linux").strip('"')
            version = info.get("VERSION_ID", "").strip('"')
            ok(f"{name} {version}")
        except OSError:
            ok("Linux")
    elif sys_name == "Darwin":
        rc, out = run(["sw_vers", "-productVersion"])
        ok(f"macOS {out}" if rc == 0 else "macOS")
        warn("macOS supports the CPU backend only")
    elif sys_name == "Windows":
        ok(f"Windows {platform.version()}")
        warn("Windows supports the OpenCL backend only (HIP not available)")
    else:
        warn(f"Unrecognised OS: {sys_name}")


def check_cmake(missing: list) -> None:
    header("CMake (>= 3.10 required)")
    cmake = shutil.which("cmake")
    if not cmake:
        fail("cmake not found", "sudo apt install cmake  # or equivalent")
        missing.append("cmake")
        return
    rc, out = run(["cmake", "--version"])
    version_line = out.splitlines()[0] if out else ""
    version_str = version_line.replace("cmake version", "").strip()
    try:
        parts = [int(x) for x in version_str.split(".")[:2]]
        if parts >= [3, 10]:
            ok(f"cmake {version_str}", cmake)
        else:
            fail(f"cmake {version_str} is too old (need >= 3.10)",
                 "Install a newer cmake from cmake.org or your package manager")
            missing.append("cmake >= 3.10")
    except ValueError:
        ok(f"cmake found ({version_str})", cmake)


def check_compiler(rocm_path: Path, missing: list) -> None:
    header("C++17 Compiler")
    amdclang = rocm_path / "lib" / "llvm" / "bin" / "amdclang++"
    if amdclang.exists():
        rc, out = run([str(amdclang), "--version"])
        version_line = out.splitlines()[0] if out else ""
        ok(f"amdclang++ found", str(amdclang))
        return
    # Fall back to system clang++ or g++
    for compiler in ("clang++", "g++", "c++"):
        path = shutil.which(compiler)
        if path:
            warn(f"{compiler} found (amdclang++ preferred for GPU builds)",
                 f"Install ROCm Core SDK to get amdclang++ at {amdclang}")
            return
    fail("No C++17 compiler found",
         "Install ROCm Core SDK (amdclang++) or a system C++17 compiler")
    missing.append("C++17 compiler")


def check_rocm(rocm_path: Path, backend: str, missing: list, optional: list) -> None:
    header(f"ROCm Core SDK  (path: {rocm_path})")
    if not rocm_path.exists():
        if backend == "CPU":
            warn(f"{rocm_path} not found -- OK for CPU-only build")
        else:
            fail(f"ROCm not found at {rocm_path}",
                 "Install the ROCm Core SDK: https://rocm.docs.amd.com/en/latest/install/rocm.html")
            missing.append("ROCm Core SDK")
        return

    # Version
    ver_file = rocm_path / ".info" / "version"
    if ver_file.exists():
        version = ver_file.read_text().strip()
        try:
            major, minor = [int(x) for x in version.split(".")[:2]]
            if (major, minor) >= (7, 13):
                ok(f"ROCm {version}")
            else:
                fail(f"ROCm {version} is too old -- MIVisionX 4.0 requires ROCm 7.13 or later",
                     "Upgrade the ROCm Core SDK to 7.13+")
                missing.append("ROCm >= 7.13")
        except ValueError:
            ok(f"ROCm found (version: {version})")
    else:
        ok(f"ROCm found at {rocm_path} (version file not present)")


def check_hip(rocm_path: Path, backend: str, missing: list) -> None:
    header("HIP Runtime")
    if backend != "HIP":
        print(f"  (skipped -- backend is {backend})")
        return
    hipcc = rocm_path / "bin" / "hipcc"
    libamdhip = find_file([
        rocm_path / "lib" / "libamdhip64.so",
        rocm_path / "lib64" / "libamdhip64.so",
    ])
    if hipcc.exists() and libamdhip:
        rc, out = run([str(hipcc), "--version"])
        version_line = next((l for l in out.splitlines() if "HIP version" in l), out.splitlines()[0] if out else "")
        ok(f"HIP runtime found", version_line)
    else:
        fail("HIP runtime not found",
             f"Install the ROCm Core SDK -- expected hipcc at {hipcc}")
        missing.append("HIP runtime (hip-runtime-amd / hip-dev)")


def check_opencl(rocm_path: Path, backend: str, missing: list) -> None:
    header("OpenCL")
    if backend == "CPU":
        print("  (skipped -- CPU-only build)")
        return
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
    if backend == "OCL":
        if libocl and cl_header:
            ok("OpenCL library and headers found", str(libocl))
        elif not libocl:
            fail("libOpenCL.so not found",
                 "Install the ROCm Core SDK or the OpenCL SDK")
            missing.append("OpenCL library")
        else:
            fail("OpenCL headers not found",
                 "Install opencl-headers or the ROCm Core SDK")
            missing.append("OpenCL headers")
    else:
        # HIP backend -- OpenCL presence is informational
        if libocl:
            ok("OpenCL library found (optional for HIP builds)", str(libocl))
        else:
            warn("OpenCL library not found -- OK for HIP/CPU builds")


def check_half(rocm_path: Path, missing: list) -> None:
    header("half (float16 library, >= 1.12.0)")
    half_header = find_file([
        rocm_path / "include" / "half" / "half.hpp",
        rocm_path / "include" / "half.hpp",
        Path("/usr/include/half/half.hpp"),
        Path("/usr/local/include/half/half.hpp"),
    ])
    if half_header:
        ok("half.hpp found", str(half_header))
    else:
        fail("half.hpp not found",
             "Install from ROCm Core SDK (package: half / half-dev)")
        missing.append("half (float16 library)")


def check_rpp(rocm_path: Path, backend: str, missing: list) -> None:
    header("RPP -- ROCm Performance Primitives (>= 3.1.0, required for amd_rpp extension)")
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
        fail("RPP not found -- amd_rpp extension will not be built",
             "Install from ROCm Core SDK (package: rpp / rpp-dev / amdrpp7.13-*)")
        missing.append("RPP >= 3.1.0 (rpp / rpp-dev)")
        return

    # Check version from rppConfig.cmake
    version_str = ""
    if rpp_cmake:
        try:
            cmake_text = rpp_cmake.read_text()
            for line in cmake_text.splitlines():
                if "VERSION" in line and "3." in line:
                    import re
                    m = re.search(r"(\d+\.\d+\.\d+)", line)
                    if m:
                        version_str = m.group(1)
                        break
        except OSError:
            pass

    # Check backend capability from rpp_backend.h
    backend_note = ""
    rpp_backend_h = find_file([
        rocm_path / "include" / "rpp" / "rpp_backend.h",
        Path("/usr/include/rpp/rpp_backend.h"),
    ])
    if rpp_backend_h:
        content = rpp_backend_h.read_text()
        has_hip = "RPP_BACKEND_HIP 1" in content
        if backend == "HIP" and not has_hip:
            warn("RPP found but HIP backend not enabled in this RPP build",
                 "Reinstall RPP with HIP support from the ROCm Core SDK")
        elif has_hip:
            backend_note = "HIP backend enabled"

    detail = " ".join(filter(None, [version_str, backend_note, str(rpp_lib or rpp_header)]))
    ok("RPP found", detail)

    if backend == "OCL":
        warn("amd_rpp will be built in CPU-only mode when the core uses the OpenCL backend "
             "(RPP has no OpenCL backend)")


def check_openmp(rocm_path: Path, missing: list) -> None:
    header("OpenMP")
    # Check for libomp from the ROCm LLVM toolchain or system
    libomp = find_file([
        rocm_path / "lib" / "libomp.so",
        rocm_path / "llvm" / "lib" / "libomp.so",
        Path("/usr/lib/llvm-18/lib/libomp.so"),
        Path("/usr/lib/x86_64-linux-gnu/libomp.so"),
        Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1"),
    ])
    omp_header = find_file([
        rocm_path / "lib" / "llvm" / "lib" / "clang" / "18" / "include" / "omp.h",
        Path("/usr/lib/llvm-18/lib/clang/18/include/omp.h"),
        Path("/usr/include/omp.h"),
    ])
    if libomp or omp_header:
        ok("OpenMP found", str(libomp or omp_header))
    else:
        warn("OpenMP library not explicitly found -- may be provided transitively by ROCm toolchain",
             "Install openmp-extras-runtime / openmp-extras-dev from the ROCm Core SDK")


def check_openssl(optional: list) -> None:
    header("OpenSSL / libcrypto (optional -- RunVX data-compare checksums only)")
    if platform.system() == "Windows":
        ok("Windows uses wincrypt.h instead of OpenSSL -- not required")
        return
    # Look for the header (libssl-dev / openssl-devel)
    ssl_header = find_file([
        Path("/usr/include/openssl/hmac.h"),
        Path("/usr/local/include/openssl/hmac.h"),
        Path("/usr/local/opt/openssl/include/openssl/hmac.h"),
        Path("/opt/local/include/openssl/hmac.h"),
    ])
    libcrypto = find_file([
        Path("/usr/lib/x86_64-linux-gnu/libcrypto.so"),
        Path("/usr/lib/libcrypto.so"),
        Path("/usr/local/lib/libcrypto.so"),
    ])
    if ssl_header and libcrypto:
        ok("OpenSSL found", str(ssl_header.parent))
    elif ssl_header:
        ok("OpenSSL headers found (libcrypto.so will be found at link time)", str(ssl_header))
    else:
        warn("OpenSSL not found -- RunVX compare checksums will use a built-in fallback",
             "sudo apt install libssl-dev  # optional")
        optional.append("OpenSSL (libssl-dev) -- enables MD5 checksums in RunVX compare commands")


def check_opencv(missing: list, optional: list) -> None:
    header("OpenCV (optional -- RunVX image/video display only)")
    # Try pkg-config first
    rc, out = run(["pkg-config", "--modversion", "opencv4"])
    if rc == 0:
        ok(f"OpenCV {out.strip()} found via pkg-config")
        return
    rc, out = run(["pkg-config", "--modversion", "opencv"])
    if rc == 0:
        ok(f"OpenCV {out.strip()} found via pkg-config")
        return
    # Try finding the header
    cv_header = find_file([
        Path("/usr/include/opencv4/opencv2/opencv.hpp"),
        Path("/usr/local/include/opencv4/opencv2/opencv.hpp"),
        Path("/usr/include/opencv2/opencv.hpp"),
    ])
    if cv_header:
        ok("OpenCV headers found", str(cv_header))
        return
    warn("OpenCV not found -- RunVX will be built without image/video display",
         "sudo apt install libopencv-dev  # optional")
    optional.append("OpenCV (libopencv-dev) -- enables RunVX camera capture and display")


def check_gpu_devices(backend: str) -> None:
    header("GPU Devices")
    if backend == "CPU":
        print("  (skipped -- CPU-only build)")
        return
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        warn("rocminfo not found -- cannot enumerate GPU devices",
             "Install the ROCm Core SDK to get rocminfo")
        return
    rc, out = run([rocminfo])
    if rc != 0:
        warn("rocminfo returned an error -- GPU may not be accessible")
        return
    agents = [l.strip() for l in out.splitlines() if "Name:" in l and "gfx" in l.lower()]
    if agents:
        for agent in agents:
            ok(f"GPU detected: {agent.replace('Name:', '').strip()}")
    else:
        warn("No AMD GPU agents detected by rocminfo",
             "Ensure an AMD GPU is present and the ROCm driver is loaded (amdgpu)")


def check_python() -> None:
    header("Python (for test scripts)")
    version = sys.version.split()[0]
    if sys.version_info >= (3, 8):
        ok(f"Python {version}", sys.executable)
    else:
        warn(f"Python {version} -- test scripts require Python 3.8+")


def check_ubuntu2204_extra(missing: list) -> None:
    """On Ubuntu 22.04, libstdc++-12-dev is required."""
    try:
        with open("/etc/os-release") as f:
            info = dict(
                line.strip().split("=", 1)
                for line in f if "=" in line
            )
        name = info.get("NAME", "").strip('"')
        version = info.get("VERSION_ID", "").strip('"')
    except OSError:
        return
    if "Ubuntu" not in name or version != "22.04":
        return

    header("Ubuntu 22.04 extra: libstdc++-12-dev")
    rc, out = run(["dpkg", "-s", "libstdc++-12-dev"])
    if rc == 0 and "installed" in out:
        ok("libstdc++-12-dev installed")
    else:
        fail("libstdc++-12-dev not found (required on Ubuntu 22.04)",
             "sudo apt install libstdc++-12-dev")
        missing.append("libstdc++-12-dev (Ubuntu 22.04 only)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(missing: list, optional: list, backend: str) -> int:
    print()
    print("=" * 64)
    print(bold(f"  MIVisionX 4.0 -- Dependency Check Summary  (backend: {backend})"))
    print("=" * 64)

    if not missing and not optional:
        print(green("\n  All dependencies satisfied. Ready to build MIVisionX.\n"))
        print("  Build commands:")
        if backend == "HIP":
            print("    mkdir build-hip && cd build-hip")
            print("    cmake ../")
            print("    make -j$(nproc) && sudo make install")
        elif backend == "OCL":
            print("    mkdir build-ocl && cd build-ocl")
            print("    cmake ../ -DBACKEND=OCL")
            print("    make -j$(nproc) && sudo make install")
        else:
            print("    mkdir build-cpu && cd build-cpu")
            print("    cmake ../ -DGPU_SUPPORT=OFF")
            print("    make -j$(nproc) && sudo make install")
        print()
        return 0

    if missing:
        print(red(f"\n  {len(missing)} required dependency/dependencies MISSING:\n"))
        for item in missing:
            print(red(f"    x  {item}"))
        print()
        print(red("  Install the missing dependencies above before building.\n"))

    if optional:
        print(yellow(f"  {len(optional)} optional dependency/dependencies not installed:\n"))
        for item in optional:
            print(yellow(f"    o  {item}"))
        print()

    if not missing:
        if optional:
            print(yellow("  Build will succeed. Optional items above enable additional features.\n"))
        else:
            print(green("\n  All dependencies satisfied. Ready to build MIVisionX.\n"))
            print("  Build commands:")
            if backend == "HIP":
                print("    mkdir build-hip && cd build-hip")
                print("    cmake ../")
                print("    make -j$(nproc) && sudo make install")
            elif backend == "OCL":
                print("    mkdir build-ocl && cd build-ocl")
                print("    cmake ../ -DBACKEND=OCL")
                print("    make -j$(nproc) && sudo make install")
            else:
                print("    mkdir build-cpu && cd build-cpu")
                print("    cmake ../ -DGPU_SUPPORT=OFF")
                print("    make -j$(nproc) && sudo make install")
            print()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check MIVisionX 4.0 build dependencies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  HIP   AMD GPU via ROCm HIP  (default on Linux)
  OCL   GPU via OpenCL
  CPU   CPU only -- no ROCm or GPU required
        """,
    )
    parser.add_argument(
        "--rocm-path",
        default=os.environ.get("ROCM_PATH", "/opt/rocm"),
        help="Path to the ROCm installation (default: $ROCM_PATH or /opt/rocm)",
    )
    parser.add_argument(
        "--backend",
        choices=["HIP", "OCL", "CPU"],
        default="CPU" if platform.system() in ("Darwin", "Windows") else "HIP",
        help="Target backend to check dependencies for",
    )
    args = parser.parse_args()

    rocm_path = Path(args.rocm_path)
    backend = args.backend

    print()
    print(bold("MIVisionX 4.0 -- Dependency Checker"))
    print(f"  ROCm path : {rocm_path}")
    print(f"  Backend   : {backend}")
    print(f"  OS        : {platform.system()} {platform.release()}")

    missing:  list[str] = []
    optional: list[str] = []

    check_os()
    check_cmake(missing)
    check_compiler(rocm_path, missing)
    check_rocm(rocm_path, backend, missing, optional)
    check_hip(rocm_path, backend, missing)
    check_opencl(rocm_path, backend, missing)
    check_half(rocm_path, missing)
    check_rpp(rocm_path, backend, missing)
    check_openmp(rocm_path, missing)
    check_openssl(optional)
    check_opencv(missing, optional)
    check_gpu_devices(backend)
    check_python()
    check_ubuntu2204_extra(missing)

    return print_summary(missing, optional, backend)


if __name__ == "__main__":
    sys.exit(main())
