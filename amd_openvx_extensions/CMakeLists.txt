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
find_package(OpenCV QUIET)
find_package(FFmpeg QUIET)
find_package(AMDRPP QUIET)

if(GPU_SUPPORT)
    find_package(miopengemm PATHS ${ROCM_PATH} QUIET)
    find_package(miopen     PATHS ${ROCM_PATH} QUIET)
    if("${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            SET(BUILD_OPENCL True)
        else()
            set(GPU_SUPPORT OFF)
            set(BUILD_OPENCL False)
        endif()
    elseif("${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/cmake)
        # HSA_PATH
        IF(NOT DEFINED ENV{HSA_PATH})
            SET(HSA_PATH ${ROCM_PATH}/hsa)
        ELSE()
            SET(HSA_PATH $ENV{HSA_PATH})
        ENDIF()
        find_package(HIP QUIET REQUIRED)
    endif()
endif()

if(GPU_SUPPORT)
    if (OpenCL_FOUND AND BUILD_OPENCL)
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
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if(HIP_COMPILER STREQUAL clang)
              set(hip_library_name amdhip64)
        else()
              message(FATAL_ERROR "Unsupported HIP compiler")
        endif()
        if(FFMPEG_FOUND)
            message("-- ${Red}WARNING:FFMPEG Found -- amd_media module excluded with HIP Backend${ColourReset}")
        endif(FFMPEG_FOUND)
    else()
      message("-- ${Red}WARNING:GPU Support OpenCL/HIP Not Found -- amd_openvx_extensions modules for GPU excluded${ColourReset}")
    endif()
endif(GPU_SUPPORT)

if (NEURAL_NET)
    if(GPU_SUPPORT AND miopengemm_FOUND AND miopen_FOUND)
        if (OpenCL_FOUND AND BUILD_OPENCL)
            add_subdirectory(amd_nn)
            message("-- ${Green}AMD OpenVX Neural Network Extension -- amd_nn module added with OpenCL backend${ColourReset}")
        elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            add_subdirectory(amd_nn)
            message("-- ${Green}AMD OpenVX Neural Network Extension -- amd_nn module added with HIP backend${ColourReset}")
        else()
            message("-- ${Red}WARNING:GPU support with OpenCL/HIP Not Found -- amd_nn module excluded${ColourReset}")
        endif()
    else()
        message("-- ${Red}WARNING:GPU_SUPPORT/MIOpen/MIOpenGEMM Not Found -- amd_nn module excluded${ColourReset}")
    endif()
else()
    message("-- ${Cyan}Neural Net Modules turned OFF by user option -D NEURAL_NET=OFF ${ColourReset}")
endif()

if(AMDRPP_FOUND AND GPU_SUPPORT)
    add_subdirectory(amd_rpp)
elseif(AMDRPP_FOUND AND "${BACKEND}" STREQUAL "CPU")
    message("-- ${Red}WARNING:AMDRPP Found -- amd_rpp module excluded with CPU Backend${ColourReset}")
else()
    message("-- ${Red}WARNING:AMDRPP Not Found -- amd_rpp module excluded${ColourReset}")
endif()

if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3)
        add_subdirectory(amd_opencv)
        message("-- ${Green}AMD OpenVX OpenCV Extension -- amd_opencv module added${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING:OpenCV Not Found -- amd_opencv module excluded${ColourReset}")
endif()
