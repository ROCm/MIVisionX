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

cmake_minimum_required(VERSION 3.10)
project(rali_pybind)

set(ROCM_PATH /opt/rocm)
set(MIN_CXX_VER_REQUIRED 7.4)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

link_directories(${ROCM_PATH}/mivisionx/lib/
                 ${ROCM_PATH}/rpp/lib
                )

include_directories(../include
                    ../../amd_openvx/openvx/include/
                    ${ROCM_PATH}/mivisionx/include/
                   )

pybind11_add_module(rali_pybind rali_pybind.cpp)
target_link_libraries(rali_pybind PRIVATE rali vx_rpp amd_rpp )
install(TARGETS rali_pybind DESTINATION lib)
