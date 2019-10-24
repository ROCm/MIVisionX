# Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required (VERSION 2.8)
project (applications)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCV QUIET)
find_package(OpenCL QUIET)
find_package(miopengemm PATHS /opt/rocm QUIET)
find_package(miopen     PATHS /opt/rocm QUIET)

set(ENABLE_CLOUD_INFERENCE 0)
if(OpenCL_FOUND AND miopengemm_FOUND AND miopen_FOUND)
    if(OpenCV_FOUND AND ENABLE_CLOUD_INFERENCE)
        add_subdirectory (cloud_inference/server_app)
        message("-- ${Green}Cloud Inference -- cloud_inference application added${ColourReset}")
    else()
        message("-- ${Red}WARNING:OpenCV Not Found/APP Disabled -- cloud_inference application excluded${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING:OpenCL/MIOpenGEMM/MIOpen Not Found -- cloud_inference application excluded${ColourReset}")
endif()
