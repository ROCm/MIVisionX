"""Manage build and installation of MIVisionX.

Copyright 2020-2021 Advanced Micro Devices, Inc.
"""

import os
import platform
import subprocess
import argparse
import pathlib
from typing import Tuple

ARGS = {}
param = {}
OS_info = {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    parser.add_argument(
        "-g",
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Generate Debug build (optional, default: False)",
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        required=False,
        default="build",
        help="Build directory path (optional, default: build)",
    )
    # parser.add_argument(
    #     "-i",
    #     "--install",
    #     required=False,
    #     default=False,
    #     dest="install",
    #     action="store_true",
    #     help="Install after build (optional, default: False)",
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     required=False,
    #     default=False,
    #     action="store_true",
    #     help="Verbose build (optional, default: False)",
    # )
    parser.add_argument(
        "--cmake-darg",
        required=False,
        dest="cmake_dargs",
        action="append",
        default=[],
        help=(
            "List of additional cmake defines for builds "
            "(optional, e.g. CMAKE_PREFIX_PATH=/foo/bar)"
        ),
    )
    # parser.add_argument(
    #     "-d",
    #     "--dependencies",
    #     required=False,
    #     action="store_true",
    #     help=(
    #         "Check dependencies, and download if possible "
    #         "(optional, default: False)"
    #     ),
    # )
    # parser.add_argument(
    #     "--build-utils",
    #     required=False,
    #     action="store_true",
    #     help="Build the utilities (optional, default: False)",
    # )
    # parser.add_argument(
    #     "-p",
    #     "--parallel",
    #     required=False,
    #     type=int,
    #     default=1,
    #     help=(
    #         "How many CPUs to use for compilation "
    #         "(set to 0 for all, default: 1)"
    #     ),
    # )
    parser.add_argument(
        "--opencl",
        required=False,
        default=os.environ.get(
            "OCL_ROOT", "C:\\Program Files (x86)\\OCL_SDK_Light"
        ),
        help="The installation path of OpenCL.",
    )
    parser.add_argument(
        "--opencv",
        required=False,
        default=os.environ.get("OpenCV_DIR", "C:\\OpenCV\\build"),
        help="The installation path of OpenCV.",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Do not build the targets, only configure the build files.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="The generator to use (default: CMake default)",
    )

    return parser.parse_args()


def os_detect():
    """Detect the OS version and number of processors."""
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file, encoding="utf-8") as info:
                for line in info:
                    if "=" in line:
                        key, val = line.strip().split("=")
                        OS_info[key] = val.replace('"', "")
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)


def create_dir(dir_path):
    """Create a (possibly existing) directory, and parents if required."""
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join(os.getcwd(), dir_path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


def delete_dir(dir_path):
    """Recursiely delete a directory if it exists."""
    dir_ = pathlib.Path(dir_path)
    if dir_.is_dir():
        for child in dir_.iterdir():
            if child.is_dir():
                delete_dir(child)
            else:
                child.unlink()
        dir_.rmdir()


def cmake_path(os_path):
    """Convert a path to be CMake compatible."""
    return os_path.replace("\\", "/") if os.name == "nt" else os_path


def config_cmd():
    """Create the CMake command to configure the project."""
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []

    src_path = cmake_path(cwd_path)
    build_path = os.path.abspath(ARGS.build_dir)
    build_path = cmake_path(
        os.path.join(build_path, "debug" if ARGS.debug else "release")
    )
    cmake_options.append(f"-S {src_path}")
    cmake_options.append("-B .")

    if ARGS.generator is not None:
        cmake_options.append(f'-G"{ARGS.generator}"')
    cmake_options.append(f"-DOpenCV_DIR={cmake_path(ARGS.opencv)}")
    cmake_options.append(f"-DCMAKE_PREFIX_PATH={cmake_path(ARGS.opencv)}")
    cmake_options.append(
        f"-DCMAKE_BUILD_TYPE={'Debug' if ARGS.debug else 'Release'}"
    )
    cmake_options.append("-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE")

    delete_dir(build_path)
    create_dir(build_path)
    os.chdir(build_path)

    if ARGS.cmake_dargs:
        for arg in ARGS.cmake_dargs:
            cmake_options.append(f"-D{arg}")

    return cmake_executable, " ".join(cmake_options)


def build_cmd() -> Tuple[str, str]:
    """Create the command to build the project."""
    build_options = []
    if os.name == "nt":
        build_exe = "cmake.exe"
        build_options.append("--build .")
        build_options.append("--target all_build")
    else:
        # TODO
        pass
    return build_exe, " ".join(build_options)


def run_cmd(exe, opts):
    """Run a given command with the given options."""
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(
        program, check=True, stderr=subprocess.STDOUT, shell=True
    )
    return proc.returncode


def main():
    """Configure and build MIVisionX."""
    global ARGS
    os_detect()
    ARGS = parse_args()

    # configure
    cmd, opts = config_cmd()
    run_cmd(cmd, opts)

    if not ARGS.configure_only:
        # build
        cmd, opts = build_cmd()
        run_cmd(cmd, opts)


if __name__ == "__main__":
    main()
