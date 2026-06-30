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
project(utilities)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
#set(ROCM_PATH /opt/rocm CACHE PATH " default ROCm installation path")

if(GPU_SUPPORT)
    if("${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            set(BUILD_OPENCL true)
        else()
            set(BUILD_OPENCL false)
            set(GPU_SUPPORT false)
        endif()
    elseif ("${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
        find_package(HIP QUIET)
        SET(BUILD_OPENCL false)
    endif()
endif()

add_subdirectory(runvx)

if(OpenCL_FOUND AND BUILD_OPENCL)
    message("-- ${Green}Utilities -- runvx module added with OPENCL support${ColourReset}")
elseif (HIP_FOUND)
    message("-- ${Green}Utilities -- runvx module added with HIP support${ColourReset}")
else()
    message("-- ${Green}Utilities -- runvx module added with CPU support${ColourReset}")
endif()
