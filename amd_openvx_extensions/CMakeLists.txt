# Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.0)
project(amd_openvx_extensions)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

if(OPENCL_SUPPORT)
    find_package(OpenCL QUIET)
else()
	message("-- ${Cyan}OpenCL Modules turned OFF by user option -D OPENCL_SUPPORT=OFF ${ColourReset}")
endif()
find_package(OpenCV QUIET)
find_package(FFmpeg QUIET)
find_package(AMDRPP QUIET)
find_package(miopengemm PATHS ${ROCM_PATH} QUIET)
find_package(miopen     PATHS ${ROCM_PATH} QUIET)

if(OpenCL_FOUND)
    if(AMDRPP_FOUND)
        add_subdirectory(amd_rpp)
    else()
        message("-- ${Red}WARNING:AMDRPP Not Found -- amd_rpp module excluded${ColourReset}")
    endif()
    if(LOOM)
        add_subdirectory(amd_loomsl)
        message("-- ${Green}AMD OpenVX Loom Stich Library Extension -- amd_loomsl module added${ColourReset}")
    else()
        message("-- ${Cyan}LOOM Module turned OFF by user option -D LOOM=OFF ${ColourReset}")
    endif()
    if(FFMPEG_FOUND)
        add_subdirectory(amd_media)
        message("-- ${Green}AMD OpenVX Media Extension -- amd_media module added${ColourReset}")
    else()
        message("-- ${Red}WARNING:FFMPEG Not Found -- amd_media module excluded${ColourReset}")
    endif(FFMPEG_FOUND)
else()
    message("-- ${Red}WARNING:OpenCL/FFMPEG Not Found -- amd_loomsl, & amd_media modules excluded${ColourReset}")
endif(OpenCL_FOUND)

if(NEURAL_NET)
    if(OpenCL_FOUND AND miopengemm_FOUND AND miopen_FOUND)
        add_subdirectory(amd_nn)
        message("-- ${Green}AMD OpenVX Neural Network Extension -- amd_nn module added${ColourReset}")
    else()
        message("-- ${Red}WARNING:OpenCL/MIOpen/MIOpenGEMM Not Found -- amd_nn module excluded${ColourReset}")
    endif()
else()
    message("-- ${Cyan}Neural Net Modules turned OFF by user option -D NEURAL_NET=OFF ${ColourReset}")
endif()

if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3)
        add_subdirectory(amd_opencv)
        message("-- ${Green}AMD OpenVX OpenCV Extension -- amd_opencv module added${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING:OpenCV Not Found -- amd_opencv module excluded${ColourReset}")
endif(OpenCV_FOUND)
