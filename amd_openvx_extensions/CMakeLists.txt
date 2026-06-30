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

cmake_minimum_required(VERSION 3.10)
# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- amd_openvx_extensions:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(amd_openvx_extensions)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(rpp 3.1.0 QUIET)
if(GPU_SUPPORT)
    if("${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            SET(BUILD_OPENCL True)
        else()
            set(GPU_SUPPORT OFF)
            set(BUILD_OPENCL False)
        endif()
    elseif("${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
        find_package(HIP QUIET)
        if(HIP_FOUND)
            list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm)
        else()
            set(GPU_SUPPORT OFF)
        endif()
    endif()
endif()

if(rpp_FOUND)
    # RPP supports the CPU and HIP backends only (OpenCL backend support has been dropped in RPP)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${rpp_INCLUDE_DIR}/rpp/rpp_backend.h)
        file(READ ${rpp_INCLUDE_DIR}/rpp/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message("-- ${Red}WARNING: ${rpp_INCLUDE_DIR}/rpp/rpp_backend.h file Not Found. please install the latest RPP package ${ColourReset}")
    endif()

    if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND AND RPP_BACKEND_HIP_FOUND)
        add_subdirectory(amd_rpp)
        message("-- ${Green}AMD ROCm Performance Primitives -- amd_rpp module added with the HIP backend support${ColourReset}")
    elseif("${BACKEND}" STREQUAL "OPENCL")
        # RPP has no OpenCL backend; build the extension in CPU-only mode alongside the OpenCL core
        add_subdirectory(amd_rpp)
        message("-- ${Yellow}AMD ROCm Performance Primitives -- RPP has no OpenCL backend; amd_rpp module added with CPU-only support${ColourReset}")
    else()
        add_subdirectory(amd_rpp)
        message("-- ${Green}AMD ROCm Performance Primitives -- amd_rpp module added with CPU only backend support${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING: AMD RPP Not Found -- OpenVX Extension amd_rpp module excluded${ColourReset}")
endif()
