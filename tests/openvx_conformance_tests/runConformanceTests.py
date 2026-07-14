# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

"""Build and run OpenVX 1.3 Conformance Tests (CTS) for AMD MIVisionX.

This script automates:
  1. Building MIVisionX for one or more backends (HOST / OCL / HIP).
  2. Generating a system information report.
  3. Cloning, configuring, and building the Khronos OpenVX-cts suite.
  4. Running the CTS and collecting per-backend conformance logs.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, IO, List, TextIO, Tuple
import argparse
import logging
import os
import platform
import re
import signal
import shutil
import shlex
import socket
import subprocess
import sys

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Conformance System Report"
__license__ = "MIT"
__version__ = "1.4.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Command = List[str]
Env = Dict[str, str]

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

log = logging.getLogger("runConformanceTests")


@dataclasses.dataclass(frozen=True)
class Defaults:
    """Compile-time defaults that are referenced in multiple places."""

    CTS_REPO: str = "https://github.com/KhronosGroup/OpenVX-cts.git"
    CTS_BRANCH: str = "openvx_1.3.2"
    HIP_LIB: str = "/opt/rocm/lib/libamdhip64.so"
    LINUX_LINK_LIBS: Tuple[str, ...] = ("pthread", "dl", "m", "rt")
    PARALLEL_JOBS: int = 8
    VALID_BACKENDS: Tuple[str, ...] = ("ALL", "HOST", "HIP", "OCL")
    ENV_KEYS_TO_LOG: Tuple[str, ...] = (
        "VX_TEST_DATA_PATH",
        "AGO_DEFAULT_TARGET",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    )


DEFAULTS = Defaults()

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_darwin() -> bool:
    """Return True when running on macOS."""
    return platform.system() == "Darwin"


def get_platform_ip() -> str:
    """Best-effort local IP lookup; returns '' on failure."""
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return ""


def sanitize_filename(s: str, *, max_len: int = 180) -> str:
    """Replace non-portable characters so *s* is safe as a filename."""
    s = s.strip()
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.ASCII)
    s = re.sub(r"_+", "_", s)
    if not s:
        return "output"
    return s[:max_len]


def add_library_path(env: Env, path: Path) -> None:
    """Prepend *path* to the platform-appropriate shared-library search env vars."""
    p = str(path)
    keys = ["DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"] if _is_darwin() else ["LD_LIBRARY_PATH"]
    for key in keys:
        existing = env.get(key, "")
        parts = [x for x in existing.split(":") if x] if existing else []
        if p in parts:
            continue
        env[key] = ":".join([p, *parts]) if parts else p


def resolve_cts_dir(directory_arg: str) -> Path:
    """Resolve the conformance working directory from the user-supplied path."""
    base = Path(os.path.expanduser(directory_arg)).resolve()
    if base.name == "mivisionx-conformance":
        return base
    return (base / "mivisionx-conformance").resolve()


def path_is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when *path* is inside *parent* (or equal to it)."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def move_cwd_out_of_cts_dir(cts_dir: Path, source_root: Path) -> None:
    """Avoid deleting the process current directory during conformance cleanup."""
    try:
        cwd = Path.cwd().resolve()
    except FileNotFoundError:
        os.chdir(source_root)
        log.info("Current working directory no longer exists; switched to %s", source_root)
        return

    if path_is_relative_to(cwd, cts_dir):
        os.chdir(source_root)
        log.info("Current working directory is inside %s; switched to %s", cts_dir, source_root)


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------


def format_cmd(cmd: Command) -> str:
    """Return a shell-safe, human-readable representation of *cmd*."""
    try:
        return shlex.join(cmd)
    except Exception:
        return " ".join(cmd)


def run_checked(cmd: Command, *, cwd: Path | None = None, env: Env | None = None) -> None:
    """Run *cmd* and raise ``CalledProcessError`` on non-zero exit."""
    log.debug("run: %s", format_cmd(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_capture(cmd: Command, *, cwd: Path | None = None, env: Env | None = None) -> str:
    """Run *cmd* and return its combined stdout+stderr as a stripped string."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return (p.stdout or "").rstrip()


def run_and_tee(cmd: Command, *, cwd: Path, env: Env, output_path: Path) -> int:
    """Run *cmd*, stream output to stdout **and** write it to *output_path*."""
    log.debug("run: %s (tee -> %s)", format_cmd(cmd), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


# ---------------------------------------------------------------------------
# Markdown report helpers
# ---------------------------------------------------------------------------


def write_formatted(output: str, f: TextIO) -> None:
    """Write *output* inside a fenced code block."""
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def write_lines_as_table(header: list[str], lines: list[str], f: TextIO) -> None:
    """Write whitespace-delimited *lines* as a Markdown table."""
    for h in header:
        f.write("|%s" % h)
    f.write("|\n")

    for h in header:
        f.write("|:---")
    f.write("|\n")

    for line in lines:
        fields = line.split()
        for field in fields:
            f.write("|%s" % field)
        f.write("|\n")


# ---------------------------------------------------------------------------
# System report
# ---------------------------------------------------------------------------


def write_system_report(
    report_path: Path,
    *,
    backend_type: str,
    openvx_lib_dirs: dict[str, Path],
    lib_type: str,
) -> None:
    """Generate a Markdown system-info report at *report_path*."""
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    fqdn = socket.getfqdn()
    ip = get_platform_ip()

    def tool_output(cmd: Command) -> str:
        if shutil.which(cmd[0]) is None:
            return f"{cmd[0]}: not found"
        return run_capture(cmd)

    sys_info = tool_output(["inxi", "-c0", "-S"])
    cpu_info = tool_output(["inxi", "-c0", "-C"]).rstrip()
    gpu_info = tool_output(["inxi", "-c0", "-G"]).rstrip()
    memory_info = tool_output(["inxi", "-c0", "-m"])
    board_info = tool_output(["inxi", "-c0", "-M"])

    ldd_links: Command = ["otool", "-L"] if _is_darwin() else ["ldd"]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("MIVisionX - System Report\n")
        f.write("================================\n\n")
        f.write(f"Generated: {report_dtstr}\n\n")
        f.write(f"Platform: {fqdn} ({ip})\n")
        f.write("--------\n\n")

        write_formatted(sys_info, f)
        write_formatted(cpu_info, f)
        write_formatted(gpu_info, f)
        write_formatted(board_info, f)
        write_formatted(memory_info, f)

        f.write("\n\n")
        f.write("OpenVX Shared Library Dependencies\n")
        f.write("-------------\n\n")
        for key, lib_dir in openvx_lib_dirs.items():
            if backend_type not in ("ALL", key):
                continue
            lib_path = lib_dir / f"libopenvx{lib_type}"
            out = tool_output(ldd_links + [str(lib_path)])
            f.write(f"* OpenVX {key} Library\n")
            write_formatted(out, f)
        f.write("\n")
        f.write(f"\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- runConformanceTests.py V-{__version__}**\n")


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def build_mivisionx(source_root: Path, build_dir: Path, cmake_args: Command, jobs: int) -> None:
    """Configure and build MIVisionX into *build_dir*."""
    build_dir.mkdir(parents=True, exist_ok=True)
    run_checked(["cmake", "-S", str(source_root), "-B", str(build_dir), *cmake_args])
    run_checked(["cmake", "--build", str(build_dir), "--parallel", str(jobs)])


def ensure_openvx_cts_repo(
    conformance_tests_dir: Path,
    *,
    repo_url: str,
    branch: str,
    update: bool,
) -> Path:
    """Clone or update the OpenVX-cts repo and return its path."""
    cts_src = conformance_tests_dir / "OpenVX-cts"
    conformance_tests_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("git") is None:
        raise RuntimeError("git not found in PATH; required to fetch OpenVX-cts")
    if cts_src.exists():
        if update:
            run_checked(["git", "fetch", "--all", "--tags"], cwd=cts_src)
            run_checked(["git", "checkout", branch], cwd=cts_src)
            run_checked(["git", "pull", "--ff-only"], cwd=cts_src)
        return cts_src
    run_checked(
        ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(cts_src)],
        cwd=conformance_tests_dir,
    )
    return cts_src


def configure_and_build_cts(
    *,
    build_dir: Path,
    cts_src: Path,
    openvx_include: Path,
    openvx_libraries: Command,
    jobs: int,
) -> None:
    """Configure and build the OpenVX CTS into *build_dir*."""
    build_dir.mkdir(parents=True, exist_ok=True)
    openvx_libs_arg = "-DOPENVX_LIBRARIES=" + ";".join(openvx_libraries)
    run_checked(
        [
            "cmake",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DOPENVX_INCLUDES={openvx_include}",
            openvx_libs_arg,
            # The OpenVX 1.3 CTS hardcodes -O3 for C sources. GCC 13 on WSL can
            # miscompile test_array.c's small-type verifier at that level even
            # when the implementation returns the correct array bytes.
            "-DOPENVX_CFLAGS=-O0",
            "-DOPENVX_CONFORMANCE_VISION=ON",
            str(cts_src),
        ],
        cwd=build_dir,
    )
    run_checked(["cmake", "--build", ".", "--parallel", str(jobs)], cwd=build_dir)

    exe = build_dir / "bin" / "vx_test_conformance"
    if not exe.exists():
        raise RuntimeError(f"CTS build succeeded but {exe} is missing; check the build output above.")


def build_openvx_lib_list(
    *,
    openvx_lib_dir: Path,
    lib_type: str,
    extra_libs: list[str] | None = None,
) -> list[str]:
    """Return the semicolon-joined library list needed by -DOPENVX_LIBRARIES."""
    libs: list[str] = [
        str(openvx_lib_dir / f"libopenvx{lib_type}"),
        str(openvx_lib_dir / f"libvxu{lib_type}"),
    ]
    if extra_libs:
        libs.extend(extra_libs)
    return libs


# ---------------------------------------------------------------------------
# CTS runtime helpers
# ---------------------------------------------------------------------------


def make_runtime_env(
    *,
    build_dir: Path,
    vx_test_data_path: Path,
    ago_default_target: str | None = None,
) -> Env:
    """Build an environment dict suitable for running vx_test_conformance."""
    env = dict(os.environ)
    env["VX_TEST_DATA_PATH"] = str(vx_test_data_path)
    if ago_default_target:
        env["AGO_DEFAULT_TARGET"] = ago_default_target
    add_library_path(env, build_dir / "lib")
    add_library_path(env, build_dir / "bin")
    add_library_path(env, build_dir)
    add_library_path(env, Path("."))
    return env


def log_runtime_env(env: Env) -> None:
    """Log the key environment variables that affect the CTS run."""
    for k in DEFAULTS.ENV_KEYS_TO_LOG:
        if k in env:
            log.debug("env %s=%s", k, env[k])


def warn_if_missing_cts_module(build_dir: Path, module_base: str) -> None:
    """Emit a warning if the CTS shared module cannot be found under *build_dir*."""
    patterns = [
        build_dir / f"lib{module_base}.*",
        build_dir / "lib" / f"lib{module_base}.*",
        build_dir / "bin" / f"lib{module_base}.*",
    ]
    for p in patterns:
        if any(p.parent.glob(p.name)):
            return
    log.warning("CTS module lib%s.* not found under %s; vxLoadKernels may fail", module_base, build_dir)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def tail_file(path: Path, *, max_lines: int = 80) -> str:
    """Return the last *max_lines* lines of *path*, or '' if missing."""
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = deque(f, maxlen=max_lines)
    return "".join(lines)


def describe_returncode(rc: int) -> str:
    """Translate a process return code into a human-readable description."""
    if rc == 0:
        return "success"
    if rc < 0:
        signum = -rc
        try:
            name = signal.Signals(signum).name
            return f"terminated by {name} ({signum})"
        except Exception:
            return f"terminated by signal {signum}"
    return f"exited with code {rc}"


def normalize_returncode(rc: int) -> int:
    """Convert a Python negative-signal rc to the shell convention (128 + signum)."""
    return (128 + (-rc)) if rc < 0 else rc


def write_cts_failure_summary(
    *,
    out_md: Path,
    rc: int,
    env: Env,
    build_dir: Path,
) -> Path:
    """Write a concise failure report next to the CTS log and return its path."""
    summary_path = out_md.parent / f"{out_md.stem}_FAILURE.txt"
    tail = tail_file(out_md, max_lines=120)
    last_run = ""
    for line in reversed(tail.splitlines()):
        if "[ RUN" in line:
            last_run = line.strip()
            break

    deps = ""
    exe = build_dir / "bin" / "vx_test_conformance"
    if exe.exists():
        if _is_darwin() and shutil.which("otool"):
            deps = run_capture(["otool", "-L", str(exe)])
        elif (not _is_darwin()) and shutil.which("ldd"):
            deps = run_capture(["ldd", str(exe)])

    env_lines = "\n".join(f"{k}={env.get(k, '')}" for k in DEFAULTS.ENV_KEYS_TO_LOG if k in env)

    summary: list[str] = [
        f"CTS run failed: {describe_returncode(rc)}",
        f"log: {out_md}",
    ]
    if last_run:
        summary.append(f"last test: {last_run}")
    summary.append("")
    if env_lines:
        summary.append("Environment (selected):")
        summary.append(env_lines)
        summary.append("")
    if deps:
        summary.append("Binary dependencies:")
        summary.append(deps)
        summary.append("")
    if tail:
        summary.append("Last output (tail):")
        summary.append(tail)
    summary_path.write_text("\n".join(summary).rstrip() + "\n", encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and orchestrate the full conformance workflow."""
    parser = argparse.ArgumentParser(description="Build and run OpenVX CTS for MIVisionX.")
    parser.add_argument(
        "--directory",
        type=str,
        default="~/mivisionx-conformance",
        help="Conformance build directory (default: ~/mivisionx-conformance).",
    )
    parser.add_argument(
        "--backend_type",
        type=str,
        default="ALL",
        help="Backend type (default: ALL [options: ALL/HOST/HIP/OCL])",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--skip-mivisionx-build", action="store_true", help="Skip building MIVisionX")
    parser.add_argument("--skip-cts-build", action="store_true", help="Skip configuring/building OpenVX-cts")
    parser.add_argument("--skip-cts-run", action="store_true", help="Skip running vx_test_conformance")
    parser.add_argument("--jobs", type=int, default=DEFAULTS.PARALLEL_JOBS, help="Parallel build jobs (default: 8)")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing conformance directory")
    parser.add_argument(
        "--sudo-clean",
        action="store_true",
        help="If deletion fails, try `sudo rm -rf` for the conformance directory (may prompt for password).",
    )
    parser.add_argument("--cts-repo", type=str, default=DEFAULTS.CTS_REPO, help="OpenVX-cts repo URL")
    parser.add_argument("--cts-branch", type=str, default=DEFAULTS.CTS_BRANCH, help="OpenVX-cts branch/tag")
    parser.add_argument("--update-cts", action="store_true", help="If OpenVX-cts exists, fetch and fast-forward it")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    backend_type = args.backend_type.upper()
    if backend_type not in DEFAULTS.VALID_BACKENDS:
        print(f"ERROR: OpenVX Backends supported - {list(DEFAULTS.VALID_BACKENDS)}")
        return 2

    cts_dir = resolve_cts_dir(args.directory)
    source_root = (Path(__file__).resolve().parent / ".." / "..").resolve()
    move_cwd_out_of_cts_dir(cts_dir, source_root)

    # -- Clean previous run ---------------------------------------------------
    if cts_dir.exists() and not args.no_clean:
        try:
            shutil.rmtree(cts_dir)
            log.info("Removing previous logs: %s", cts_dir)
        except Exception as e:
            if args.sudo_clean:
                run_checked(["sudo", "rm", "-rf", str(cts_dir)])
            else:
                raise RuntimeError(
                    f"Failed to delete {cts_dir}: {e}\nRe-run with --sudo-clean if the directory is root-owned."
                ) from e

    cts_dir.mkdir(parents=True, exist_ok=True)

    lib_type = ".dylib" if _is_darwin() else ".so"
    extra_link_libs: list[str] = list(DEFAULTS.LINUX_LINK_LIBS) if not _is_darwin() else []
    hip_deps: list[str] = [DEFAULTS.HIP_LIB] if not _is_darwin() else []

    # -- Build MIVisionX ------------------------------------------------------
    build_host_dir = cts_dir / "build-host"
    build_ocl_dir = cts_dir / "build-opencl"
    build_hip_dir = cts_dir / "build-hip"

    if not args.skip_mivisionx_build:
        if backend_type in ("ALL", "HOST"):
            build_mivisionx(source_root, build_host_dir, ["-DGPU_SUPPORT=OFF"], args.jobs)
        if backend_type in ("ALL", "OCL"):
            build_mivisionx(source_root, build_ocl_dir, ["-DBACKEND=OCL"], args.jobs)
        if backend_type in ("ALL", "HIP"):
            build_mivisionx(source_root, build_hip_dir, ["-DBACKEND=HIP"], args.jobs)

    openvx_include = source_root / "amd_openvx" / "openvx" / "include"
    openvx_lib_dirs = {
        "HOST": build_host_dir / "lib",
        "OCL": build_ocl_dir / "lib",
        "HIP": build_hip_dir / "lib",
    }

    # -- System report --------------------------------------------------------
    platform_name = sanitize_filename(platform.platform())
    file_dtstr = datetime.now().strftime("%Y%m%d")
    report_filename = f"system_info_report_{platform_name}_{file_dtstr}.md"
    report_path = cts_dir / report_filename
    write_system_report(
        report_path,
        backend_type=backend_type,
        openvx_lib_dirs=openvx_lib_dirs,
        lib_type=lib_type,
    )
    print(f"\nSTATUS: Output Report File - {report_path}")

    # -- CTS repo + build dirs ------------------------------------------------
    conformance_tests_dir = cts_dir / "conformance_tests"
    cts_src = ensure_openvx_cts_repo(
        conformance_tests_dir,
        repo_url=args.cts_repo,
        branch=args.cts_branch,
        update=args.update_cts,
    )
    vx_test_data_path = cts_src / "test_data"

    cts_host_build = conformance_tests_dir / "build-cts-host"
    cts_ocl_build = conformance_tests_dir / "build-cts-opencl"
    cts_hip_build = conformance_tests_dir / "build-cts-hip"

    # -- Build CTS ------------------------------------------------------------
    if not args.skip_cts_build:
        if backend_type in ("ALL", "HOST"):
            libs = build_openvx_lib_list(openvx_lib_dir=openvx_lib_dirs["HOST"], lib_type=lib_type, extra_libs=extra_link_libs)
            configure_and_build_cts(
                build_dir=cts_host_build, cts_src=cts_src, openvx_include=openvx_include,
                openvx_libraries=libs, jobs=args.jobs,
            )

        if backend_type in ("ALL", "OCL"):
            libs = build_openvx_lib_list(openvx_lib_dir=openvx_lib_dirs["OCL"], lib_type=lib_type, extra_libs=extra_link_libs)
            configure_and_build_cts(
                build_dir=cts_ocl_build, cts_src=cts_src, openvx_include=openvx_include,
                openvx_libraries=libs, jobs=args.jobs,
            )

        if backend_type in ("ALL", "HIP"):
            libs = build_openvx_lib_list(
                openvx_lib_dir=openvx_lib_dirs["HIP"], lib_type=lib_type,
                extra_libs=[*hip_deps, *extra_link_libs],
            )
            configure_and_build_cts(
                build_dir=cts_hip_build, cts_src=cts_src, openvx_include=openvx_include,
                openvx_libraries=libs, jobs=args.jobs,
            )

    # -- Run CTS --------------------------------------------------------------
    # Run all requested backends to completion; collect failures and report at the end.
    failures: List[str] = []

    if not args.skip_cts_run:
        if backend_type in ("ALL", "HOST"):
            warn_if_missing_cts_module(cts_host_build, "test-testmodule")
            env = make_runtime_env(build_dir=cts_host_build, vx_test_data_path=vx_test_data_path)
            log_runtime_env(env)
            out_md = cts_host_build / "OpenVX-HOST-CTS.md"
            rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_host_build, env=env, output_path=out_md)
            if rc != 0:
                summary = write_cts_failure_summary(out_md=out_md, rc=rc, env=env, build_dir=cts_host_build)
                log.error("CTS run failed (%s). See log: %s", describe_returncode(rc), out_md)
                log.error("Failure summary: %s", summary)
                failures.append("HOST (%s)" % describe_returncode(rc))
            host_data = report_path.read_text(encoding="utf-8")
            cts_log_data = out_md.read_text(encoding="utf-8")
            (cts_dir / "HOST_Conformance_Logs.md").write_text(host_data + "\n\n" + cts_log_data, encoding="utf-8")

        if backend_type in ("ALL", "OCL"):
            warn_if_missing_cts_module(cts_ocl_build, "test-testmodule")
            md_paths = []  # type: List[Path]
            for target in ("CPU", "GPU"):
                env = make_runtime_env(build_dir=cts_ocl_build, vx_test_data_path=vx_test_data_path, ago_default_target=target)
                log_runtime_env(env)
                out_md = cts_ocl_build / f"OpenVX-{target}-CTS-OCL.md"
                md_paths.append(out_md)
                rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_ocl_build, env=env, output_path=out_md)
                if rc != 0:
                    summary = write_cts_failure_summary(out_md=out_md, rc=rc, env=env, build_dir=cts_ocl_build)
                    log.error("CTS run failed (target=%s, %s). See log: %s", target, describe_returncode(rc), out_md)
                    log.error("Failure summary: %s", summary)
                    failures.append("OCL-%s (%s)" % (target, describe_returncode(rc)))
            ocl_data = report_path.read_text(encoding="utf-8")
            combined = ocl_data + "\n\n" + "\n\n".join(
                p.read_text(encoding="utf-8") for p in md_paths if p.exists()
            )
            (cts_dir / "OCL_Conformance_Logs.md").write_text(combined, encoding="utf-8")

        if backend_type in ("ALL", "HIP"):
            warn_if_missing_cts_module(cts_hip_build, "test-testmodule")
            md_paths = []  # type: List[Path]
            for target in ("CPU", "GPU"):
                env = make_runtime_env(build_dir=cts_hip_build, vx_test_data_path=vx_test_data_path, ago_default_target=target)
                log_runtime_env(env)
                out_md = cts_hip_build / f"OpenVX-{target}-CTS-HIP.md"
                md_paths.append(out_md)
                rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_hip_build, env=env, output_path=out_md)
                if rc != 0:
                    summary = write_cts_failure_summary(out_md=out_md, rc=rc, env=env, build_dir=cts_hip_build)
                    log.error("CTS run failed (target=%s, %s). See log: %s", target, describe_returncode(rc), out_md)
                    log.error("Failure summary: %s", summary)
                    failures.append("HIP-%s (%s)" % (target, describe_returncode(rc)))
            hip_data = report_path.read_text(encoding="utf-8")
            combined = hip_data + "\n\n" + "\n\n".join(
                p.read_text(encoding="utf-8") for p in md_paths if p.exists()
            )
            (cts_dir / "HIP_Conformance_Logs.md").write_text(combined, encoding="utf-8")

    if failures:
        log.error("CTS failures: %s", ", ".join(failures))
        return 1

    log.info("runConformanceTests.py - V:%s", __version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
