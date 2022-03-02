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
project(nn_hip CXX)

set(NN_HIP_SOURCES
    nn_hip_kernels.cpp
    )

    set(hip_library_name amdhip64)
    set_source_files_properties(${NN_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    set(HIP_CXX_FLAGS -std=gnu++14)

    set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
    add_library(nn_hip OBJECT ${NN_HIP_SOURCES})
    set_target_properties(nn_hip PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(nn_hip PRIVATE __HIP_PLATFORM_AMD__)
