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

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Conformance System Report"
__license__ = "MIT"
__version__ = "1.2.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"


def _is_darwin() -> bool:
    return platform.system() == "Darwin"


def run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
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


def run_and_tee(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_path: Path,
) -> int:
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


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def write_lines_as_table(header, lines, f):
    for h in header:
        f.write("|%s" % h)
    f.write("|\n")

    for h in header:
        f.write("|:---")
    f.write("|\n")

    for l in lines:
        fields = l.split()
        for field in fields:
            f.write("|%s" % field)
        f.write("|\n")


def resolve_cts_dir(directory_arg: str) -> Path:
    base = Path(os.path.expanduser(directory_arg)).resolve()
    # Preserve prior behavior: treat arg as a "parent" dir unless it already is mivisionx-conformance.
    if base.name == "mivisionx-conformance":
        return base
    return (base / "mivisionx-conformance").resolve()


def add_library_path(env: dict[str, str], path: Path) -> None:
    key = "DYLD_LIBRARY_PATH" if _is_darwin() else "LD_LIBRARY_PATH"
    existing = env.get(key, "")
    parts = [str(path)]
    if existing:
        parts.append(existing)
    env[key] = ":".join(parts)


def get_platform_ip() -> str:
    # Best-effort and portable; we don't want the report to fail the test run.
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return ""


def write_system_report(
    report_path: Path,
    *,
    backend_type: str,
    openvx_lib_dirs: dict[str, Path],
    lib_type: str,
) -> None:
    platform_name = platform.platform()
    file_dtstr = datetime.now().strftime("%Y%m%d")
    report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    fqdn = socket.getfqdn()
    ip = get_platform_ip()

    def tool_output(cmd: list[str]) -> str:
        if shutil.which(cmd[0]) is None:
            return f"{cmd[0]}: not found"
        return run_capture(cmd)

    sys_info = tool_output(["inxi", "-c0", "-S"])
    cpu_info = tool_output(["inxi", "-c0", "-C"]).rstrip()
    gpu_info = tool_output(["inxi", "-c0", "-G"]).rstrip()
    memory_info = tool_output(["inxi", "-c0", "-m"])
    board_info = tool_output(["inxi", "-c0", "-M"])

    ldd_links = ["otool", "-L"] if _is_darwin() else ["ldd"]

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


def build_mivisionx(source_root: Path, build_dir: Path, cmake_args: list[str], jobs: int) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    run_checked(["cmake", "-S", str(source_root), "-B", str(build_dir), *cmake_args])
    run_checked(["cmake", "--build", str(build_dir), "--parallel", str(jobs)])


def ensure_openvx_cts_repo(conformance_tests_dir: Path) -> Path:
    cts_src = conformance_tests_dir / "OpenVX-cts"
    if cts_src.exists():
        return cts_src
    conformance_tests_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            "openvx_1.3",
            "https://github.com/KhronosGroup/OpenVX-cts.git",
            str(cts_src),
        ],
        cwd=conformance_tests_dir,
    )
    return cts_src


def configure_and_build_cts(
    *,
    build_dir: Path,
    cts_src: Path,
    openvx_include: Path,
    openvx_libraries: list[str],
    jobs: int,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    openvx_libs_arg = "-DOPENVX_LIBRARIES=" + ";".join(openvx_libraries)
    run_checked(
        [
            "cmake",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DOPENVX_INCLUDES={openvx_include}",
            openvx_libs_arg,
            "-DOPENVX_CONFORMANCE_VISION=ON",
            str(cts_src),
        ],
        cwd=build_dir,
    )
    run_checked(["cmake", "--build", ".", "--parallel", str(jobs)], cwd=build_dir)


def build_openvx_lib_list(
    *,
    openvx_lib_dir: Path,
    lib_type: str,
    extra_libs: list[str] | None = None,
) -> list[str]:
    libs: list[str] = [
        str(openvx_lib_dir / f"libopenvx{lib_type}"),
        str(openvx_lib_dir / f"libvxu{lib_type}"),
    ]
    if extra_libs:
        libs.extend(extra_libs)
    return libs


def make_runtime_env(*, build_dir: Path, vx_test_data_path: Path, ago_default_target: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["VX_TEST_DATA_PATH"] = str(vx_test_data_path)
    if ago_default_target:
        env["AGO_DEFAULT_TARGET"] = ago_default_target
    # Ensure CTS can dlopen() modules like libtest-testmodule.* from its working dir.
    add_library_path(env, build_dir / "lib")
    add_library_path(env, build_dir / "bin")
    add_library_path(env, build_dir)
    add_library_path(env, Path("."))
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run OpenVX CTS for MIVisionX.")
    parser.add_argument(
        "--directory",
        type=str,
        default="~/mivisionx-conformance",
        help="Conformance build directory (default: ~/mivisionx-conformance). If not named mivisionx-conformance, it will be created under this directory.",
    )
    parser.add_argument(
        "--backend_type",
        type=str,
        default="ALL",
        help="Backend type (default: ALL [options: ALL/HOST/HIP/OCL])",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Parallel build jobs (default: 8)")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing conformance directory")
    parser.add_argument(
        "--sudo-clean",
        action="store_true",
        help="If deletion fails, try `sudo rm -rf` for the conformance directory (may prompt for password).",
    )
    args = parser.parse_args()

    backend_type = args.backend_type.upper()
    if backend_type not in ("ALL", "HOST", "HIP", "OCL"):
        print("ERROR: OpenVX Backends supported - [ALL or HOST or HIP or OCL]")
        return 2

    cts_dir = resolve_cts_dir(args.directory)
    source_root = (Path(__file__).resolve().parent / ".." / "..").resolve()

    if cts_dir.exists() and not args.no_clean:
        try:
            shutil.rmtree(cts_dir)
            print(f"\nMIVisionX Conformance Tests: Removing Previous Logs -- {cts_dir}\n")
        except Exception as e:
            if args.sudo_clean:
                run_checked(["sudo", "rm", "-rf", str(cts_dir)])
            else:
                raise RuntimeError(
                    f"Failed to delete {cts_dir}: {e}\nRe-run with --sudo-clean if the directory is root-owned."
                ) from e

    cts_dir.mkdir(parents=True, exist_ok=True)

    lib_type = ".dylib" if _is_darwin() else ".so"
    extra_link_libs: list[str] = []
    hip_deps: list[str] = []
    if not _is_darwin():
        extra_link_libs = ["pthread", "dl", "m", "rt"]
        hip_deps = ["/opt/rocm/lib/libamdhip64.so"]

    # Build MIVisionX for selected backends
    build_host_dir = cts_dir / "build-host"
    build_ocl_dir = cts_dir / "build-opencl"
    build_hip_dir = cts_dir / "build-hip"

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

    # System report
    platform_name = platform.platform()
    file_dtstr = datetime.now().strftime("%Y%m%d")
    report_filename = f"system_info_report_{platform_name}_{file_dtstr}.md"
    report_path = Path(report_filename).resolve()
    write_system_report(
        report_path,
        backend_type=backend_type,
        openvx_lib_dirs=openvx_lib_dirs,
        lib_type=lib_type,
    )
    shutil.copy2(report_path, cts_dir / report_path.name)
    print(f"\nSTATUS: Output Report File - {report_path}")

    # CTS repo + build dirs
    conformance_tests_dir = cts_dir / "conformance_tests"
    cts_src = ensure_openvx_cts_repo(conformance_tests_dir)
    vx_test_data_path = cts_src / "test_data"

    cts_host_build = conformance_tests_dir / "build-cts-host"
    cts_ocl_build = conformance_tests_dir / "build-cts-opencl"
    cts_hip_build = conformance_tests_dir / "build-cts-hip"

    if backend_type in ("ALL", "HOST"):
        libs = build_openvx_lib_list(openvx_lib_dir=openvx_lib_dirs["HOST"], lib_type=lib_type, extra_libs=extra_link_libs)
        configure_and_build_cts(
            build_dir=cts_host_build,
            cts_src=cts_src,
            openvx_include=openvx_include,
            openvx_libraries=libs,
            jobs=args.jobs,
        )

    if backend_type in ("ALL", "OCL"):
        libs = build_openvx_lib_list(openvx_lib_dir=openvx_lib_dirs["OCL"], lib_type=lib_type, extra_libs=extra_link_libs)
        configure_and_build_cts(
            build_dir=cts_ocl_build,
            cts_src=cts_src,
            openvx_include=openvx_include,
            openvx_libraries=libs,
            jobs=args.jobs,
        )

    if backend_type in ("ALL", "HIP"):
        libs = build_openvx_lib_list(
            openvx_lib_dir=openvx_lib_dirs["HIP"],
            lib_type=lib_type,
            extra_libs=[*hip_deps, *extra_link_libs],
        )
        configure_and_build_cts(
            build_dir=cts_hip_build,
            cts_src=cts_src,
            openvx_include=openvx_include,
            openvx_libraries=libs,
            jobs=args.jobs,
        )

    # Run CTS
    if backend_type in ("ALL", "HOST"):
        env = make_runtime_env(build_dir=cts_host_build, vx_test_data_path=vx_test_data_path)
        out_md = cts_host_build / "OpenVX-HOST-CTS.md"
        rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_host_build, env=env, output_path=out_md)
        if rc != 0:
            return rc
        host_data = report_path.read_text(encoding="utf-8")
        cts_log_data = out_md.read_text(encoding="utf-8")
        Path("HOST_Conformance_Logs.md").write_text(host_data + "\n\n" + cts_log_data, encoding="utf-8")

    if backend_type in ("ALL", "OCL"):
        for target in ("CPU", "GPU"):
            env = make_runtime_env(build_dir=cts_ocl_build, vx_test_data_path=vx_test_data_path, ago_default_target=target)
            out_md = cts_ocl_build / f"OpenVX-{target}-CTS-OCL.md"
            rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_ocl_build, env=env, output_path=out_md)
            if rc != 0:
                return rc
        ocl_data = report_path.read_text(encoding="utf-8")
        gpu_md = (cts_ocl_build / "OpenVX-GPU-CTS-OCL.md").read_text(encoding="utf-8")
        Path("OCL_Conformance_Logs.md").write_text(ocl_data + "\n\n" + gpu_md, encoding="utf-8")

    if backend_type in ("ALL", "HIP"):
        for target in ("CPU", "GPU"):
            env = make_runtime_env(build_dir=cts_hip_build, vx_test_data_path=vx_test_data_path, ago_default_target=target)
            out_md = cts_hip_build / f"OpenVX-{target}-CTS-HIP.md"
            rc = run_and_tee(["./bin/vx_test_conformance"], cwd=cts_hip_build, env=env, output_path=out_md)
            if rc != 0:
                return rc
        hip_data = report_path.read_text(encoding="utf-8")
        gpu_md = (cts_hip_build / "OpenVX-GPU-CTS-HIP.md").read_text(encoding="utf-8")
        Path("HIP_Conformance_Logs.md").write_text(hip_data + "\n\n" + gpu_md, encoding="utf-8")

    print(f"\nrunConformanceTests.py - V:{__version__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
