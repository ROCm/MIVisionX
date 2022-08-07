# Copyright (c) 2017 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
project(custom_lib)

set(CMAKE_CXX_STANDARD 14)

#following is needed for HIP support
list (APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    find_package(HIP QUIET)
    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)
    include_directories(${ROCM_PATH}/include)
    list(APPEND SOURCES
            custom_api.cpp
            custom_copy_impl.cpp
        )

    if (HIP_FOUND)
        message("-- ${Green}${PROJECT_NAME} -- Building with HIP backend${ColourReset}")
        set(ENABLE_HIP 1)
        add_definitions(-DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
        link_directories(${HIP_PATH}/lib)
        add_library(${PROJECT_NAME} SHARED ${SOURCES})
        target_link_libraries(${PROJECT_NAME} hip::device)
    else()
        message("-- ${Red}WARNING: HIP Not Found -- ${PROJECT_NAME} not built{ColourReset}")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-std=c++14")
endif()
