################################################################################
#
# MIT License
#
# Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
cmake_minimum_required (VERSION 3.0)

project (classification)

set (CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED)

set(ROCM_PATH /opt/rocm CACHE PATH "ROCm Installation Path")
#find the OPENVX backend type
set(OPENVX_BACKEND_OPENCL_FOUND 0)
set(OPENVX_BACKEND_HIP_FOUND 0)
if(EXISTS ${ROCM_PATH}/mivisionx/include/openvx_backend.h)
    file(READ ${ROCM_PATH}/mivisionx/include/openvx_backend.h OPENVX_BACKEND_FILE)
    string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${OPENVX_BACKEND_FILE})
    set(OPENVX_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
    string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${OPENVX_BACKEND_FILE})
    set(OPENVX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
else()
    message("-- ${Red}WARNING: ${ROCM_PATH}/mivisionx/include/openvx_backend.h file Not Found. please install the latest mivisionx! ${ColourReset}")
endif()

if (OPENVX_BACKEND_OPENCL_FOUND)
    find_package(OpenCL REQUIRED)
    include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/rocm/mivisionx/include)
include_directories(include)

link_directories(/opt/rocm/mivisionx/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c -std=c++11")

add_library(caffeModels SHARED source/caffeModels.cpp)
target_link_libraries(caffeModels openvx vx_nn pthread)

add_executable(classifier source/classifier.cpp)
target_compile_definitions(classifier PUBLIC ENABLE_OPENCV=1)
target_link_libraries(classifier openvx vx_nn caffeModels ${OpenCV_LIBRARIES}  pthread)


