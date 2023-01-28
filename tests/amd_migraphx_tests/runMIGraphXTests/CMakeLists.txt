################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc.
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
project(runMIGraphXTests)
set(CMAKE_CXX_STANDARD 14)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/mivisionx)
link_directories(${ROCM_PATH}/lib)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
find_package(migraphx QUIET)
if(migraphx_FOUND)
    add_executable(${PROJECT_NAME} runMIGraphXTests.cpp)
    target_link_libraries(${PROJECT_NAME} migraphx::c openvx vx_amd_migraphx)
else()
    message("-- ERROR!: MIGraphX not Found, please install MIGrpahX!-- ${PROJECT_NAME} excluded${ColourReset}")
endif()