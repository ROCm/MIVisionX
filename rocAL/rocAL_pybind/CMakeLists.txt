# Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.0)
project(rali_pybind)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)
set(ROCM_PATH /opt/rocm)

set(MIN_CMAKE_VER_REQUIRED 3.10)
set(MIN_CXX_VER_REQUIRED 7.4)
set(BUILD_RALI_PYBIND true)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    find_package(AMDRPP QUIET)
    find_package(TurboJpeg QUIET)
endif()

if(NOT OpenCL_FOUND)
    set(BUILD_RALI_PYBIND false)
    message("-- ${Yellow}rocAL pybind requires OpenCL, Not Found${ColourReset}")
endif()
if(NOT AMDRPP_FOUND)
    set(BUILD_RALI_PYBIND false)
    message("-- ${Yellow}rocAL pybind requires AMDRPP, Not Found${ColourReset}")
endif()
if(NOT TurboJpeg_FOUND)
    set(BUILD_RALI_PYBIND false)
    message("-- ${Yellow}rocAL pybind requires TurboJpeg, Not Found${ColourReset}")
endif()
if(CMAKE_VERSION VERSION_LESS ${MIN_CMAKE_VER_REQUIRED})
    set(BUILD_RALI_PYBIND false)
    message("-- ${Yellow}rocAL pybind required CMake Version Not Found${ColourReset}")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MIN_CXX_VER_REQUIRED})
    set(BUILD_RALI_PYBIND false)
    message("-- ${Yellow}rocAL pybind required C++ Version Not Found${ColourReset}")
endif()

if(${BUILD_RALI_PYBIND})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

    link_directories(${ROCM_PATH}/rpp/lib ${ROCM_PATH}/mivisionx/lib)

    include_directories(../rocAL/include/
                        third_party_lib/
                        ../../amd_openvx/openvx/include/)

    add_subdirectory(third_party_lib/pybind11)

    pybind11_add_module(rali_pybind rali_pybind.cpp)
    target_link_libraries(rali_pybind PRIVATE rali vx_rpp amd_rpp)

    install(TARGETS rali_pybind DESTINATION lib)
    message("-- ${Green}Radeon Augmentation Library Python Binding - rali_pybind module added ${ColourReset}")
else()
    message("-- ${Red}WARNING: rocAL Pybind module excluded ${ColourReset}")
endif()
