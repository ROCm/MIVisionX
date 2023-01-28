# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
project(rocal_pybind)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "Deafult ROCm Installation Path")

set(CMAKE_CXX_STANDARD 14)
set(BUILD_ROCAL_PYBIND true)

if(GPU_SUPPORT)
    find_package(OpenCL QUIET)
    find_package(AMDRPP QUIET)
    find_package(TurboJpeg QUIET)
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}rocAL PyBind -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        message("-- ${Yellow}rocAL pybind requires HIP, Not Found${ColourReset}")
    endif()
endif()

if(NOT OpenCL_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires OpenCL, Not Found${ColourReset}")
endif()
if(NOT HIP_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}rocAL pybind requires HIP, Not Found${ColourReset}")
endif()
if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires AMDRPP, Not Found${ColourReset}")
else()
    #find the RPP backend type
    set(RPP_BACKEND_OPENCL_FOUND 0)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h)
        file(READ ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message("-- ${Red}WARNING: ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h file Not Found. please run the setup script to install latest RPP package ${ColourReset}")
    endif()
    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT RPP_BACKEND_OPENCL_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend first!) -- rocAL pybind module excluded${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT RPP_BACKEND_HIP_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend first!) -- rocAL pybind module excluded${ColourReset}")
        endif()
    else()
        set(BUILD_ROCAL_PYBIND false)
        message("-- ${Red}WARNING: GPU support with OpenCL or HIP Not Found -- rocAL Pybind module excluded${ColourReset}")
    endif()
endif()

if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires TurboJpeg, Not Found${ColourReset}")
endif()

if(${BUILD_ROCAL_PYBIND})

    link_directories(${ROCM_PATH}/lib)

    include_directories(../rocAL/include/api/
                        third_party_lib/
                        ../../amd_openvx/openvx/include/)

    add_subdirectory(third_party_lib/pybind11)

    pybind11_add_module(rocal_pybind rocal_pybind.cpp)
    target_link_libraries(rocal_pybind PRIVATE rocal  vx_rpp amd_rpp)
    message("-- ${White}rocal_pybind -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
    install(TARGETS rocal_pybind DESTINATION lib)
    message("-- ${Green}Radeon Augmentation Library Python Binding - rocal_pybind module added ${ColourReset}")
else()
    message("-- ${Red}WARNING: rocAL Pybind module excluded ${ColourReset}")
endif()
