"""Copyright 2020-2021 Advanced Micro Devices, Inc.
Manage build and installation"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from typing import Tuple
import winreg
from fnmatch import fnmatchcase
from pathlib import Path

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Generate Debug build (optional, default: False)')
    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Build directory path (optional, default: build)')
    # parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
    #                     help='Install after build (optional, default: False)')
    # parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
    #                     help='Verbose build (optional, default: False)')
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')
    # parser.add_argument('-d', '--dependencies', required=False, action='store_true',
    #                     help='Check dependencies, and download if possible (optional, default: False)')
    # parser.add_argument(      '--build-utils', required=False, action='store_true',
    #                     help='Build the utilities (optional, default: False)')
    # parser.add_argument('-p', '--parallel', required=False, type=int, default=1,
    #                     help="How many CPUs to use for compilation (set to 0 for all, default: 1)")
    parser.add_argument(      '--opencl', required=False, default=os.environ.get("OCL_ROOT","C:\\Program Files (x86)\\OCL_SDK_Light"), 
                        help="The installation path of OpenCL.")
    parser.add_argument(      '--opencv', required=False, default=os.environ.get("OpenCV_DIR", "C:\\OpenCV\\build"), 
                        help="The installation path of OpenCV.")
    parser.add_argument(      '--configure-only', action="store_true", help="Do not build the targets, only configure the build files.")
    parser.add_argument(      '--generator', type=str, default=None, help='The generator to use (default: CMake default)')


    return parser.parse_args()

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        run_cmd( "rm" , f"-rf {dir_path}")

def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os_path

def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    
    src_path = cmake_path(cwd_path)
    build_path = os.path.abspath(args.build_dir)
    build_path = cmake_path(os.path.join(build_path, "debug" if args.debug else "release"))
    cmake_options.append(f"-S {src_path}")
    cmake_options.append(f"-B .")

    if args.generator is not None:
        cmake_options.append(f"-G\"{args.generator}\"")
    cmake_options.append(f"-DOpenCV_DIR={cmake_path(args.opencv)}")
    cmake_options.append(f"-DCMAKE_PREFIX_PATH={cmake_path(args.opencv)}")
    cmake_options.append(f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}")
    cmake_options.append(f"-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE")

    delete_dir(build_path)
    create_dir(build_path)
    os.chdir(build_path)

    if args.cmake_dargs:
        for arg in args.cmake_dargs:
            cmake_options.append(f"-D{arg}")

    return cmake_executable, " ".join(cmake_options)


def build_cmd() -> Tuple[str, str]:
    global args
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
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode


def main():
    global args
    os_detect()
    args = parse_args()

    # configure
    cmd, opts = config_cmd()
    run_cmd(cmd, opts)

    if not args.configure_only:
        # build
        cmd, opts = build_cmd()
        run_cmd(cmd, opts)

if __name__ == '__main__':
    main()