# Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5)
project(openvx_hip CXX)

include_directories(${PROJECT_SOURCE_DIR}/../include ${PROJECT_SOURCE_DIR}/../ago ${PROJECT_SOURCE_DIR}/../api)

list(APPEND HIPVX_SOURCES
    arithmetic_kernels.cpp
    logical_kernels.cpp
    color_kernels.cpp
    filter_kernels.cpp
    statistical_kernels.cpp
    geometric_kernels.cpp
    vision_kernels.cpp
)

set_source_files_properties(${HIPVX_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")

add_library(openvx_hip OBJECT ${HIPVX_SOURCES})
set_target_properties(openvx_hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(openvx_hip PRIVATE __HIP_PLATFORM_AMD__)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /DVX_API_ENTRY=__declspec(dllexport)")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /DVX_API_ENTRY=__declspec(dllexport)")
else()
    set(HIP_CXX_FLAGS -std=gnu++14)
endif()
