################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc.
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

project (image_augmentation)

set (CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCV QUIET)
find_package(AMDRPP QUIET)

#if("${BACKEND}" STREQUAL "HIP")
#    set(ROCM_PATH /opt/rocm CACHE PATH "mivisionx default ROCm installation path")
#    set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to which HIP has been installed")
#    list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/cmake)
#    find_package(HIP QUIET REQUIRED)
#endif()    

include_directories ( ${OpenCV_INCLUDE_DIRS}  /opt/rocm/mivisionx/include/)

link_directories    (/opt/rocm/mivisionx/lib/)

add_executable(${PROJECT_NAME} ./image_augmentation.cpp)
target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mf16c -Wall ")
target_link_libraries(${PROJECT_NAME} rali ${OpenCV_LIBRARIES} )

install (TARGETS ${PROJECT_NAME} DESTINATION bin)
