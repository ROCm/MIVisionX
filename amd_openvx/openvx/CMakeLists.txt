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
project(openvx)

set(CMAKE_CXX_STANDARD 14)

include_directories(include ago api)

list(APPEND SOURCES
    ago/ago_drama.cpp
    ago/ago_drama_alloc.cpp
    ago/ago_drama_analyze.cpp
    ago/ago_drama_divide.cpp
    ago/ago_drama_merge.cpp
    ago/ago_drama_remove.cpp
    ago/ago_haf_cpu.cpp
    ago/ago_haf_cpu_arithmetic.cpp
    ago/ago_haf_cpu_canny.cpp
    ago/ago_haf_cpu_ch_extract_combine.cpp
    ago/ago_haf_cpu_color_convert.cpp
    ago/ago_haf_cpu_fast_corners.cpp
    ago/ago_haf_cpu_filter.cpp
    ago/ago_haf_cpu_geometric.cpp
    ago/ago_haf_cpu_generic_functions.cpp
    ago/ago_haf_cpu_harris.cpp
    ago/ago_haf_cpu_histogram.cpp
    ago/ago_haf_cpu_logical.cpp
    ago/ago_haf_cpu_opticalflow.cpp
    ago/ago_haf_cpu_pyramid.cpp
    ago/ago_haf_gpu_common.cpp
    ago/ago_haf_gpu_conversion.cpp
    ago/ago_haf_gpu_corners.cpp
    ago/ago_haf_gpu_linear_filter.cpp
    ago/ago_haf_gpu_special_filters.cpp
    ago/ago_interface.cpp
    ago/ago_kernel_api.cpp
    ago/ago_kernel_list.cpp
    ago/ago_platform.cpp
    ago/ago_util.cpp
    ago/ago_util_opencl.cpp
    ago/ago_util_hip.cpp
    api/vxu.cpp
    api/vx_api.cpp
    api/vx_nodes.cpp
)

# Backend Specific Settings
if (GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_library(openvx SHARED ${SOURCES})
        add_library(vxu SHARED api/vxu.cpp)
        target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=1)
        target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
        target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=1)
        target_compile_definitions(vxu PUBLIC ENABLE_HIP=0)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(openvx ${OpenCL_LIBRARIES})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
        message("-- ${Green}AMD OpenVX -- OpenVX built with OpenCL Support${ColourReset}")
        message("-- ${Blue}Using OpenCL Library -- ${OpenCL_LIBRARIES}${ColourReset}")
    else()
        set(GPU_SUPPORT OFF)
        add_library(openvx SHARED ${SOURCES})
        add_library(vxu SHARED api/vxu.cpp)
        target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
        target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(vxu PUBLIC ENABLE_HIP=0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
        message("-- ${Red}WARNING:OpenCL Not Found -- OpenVX built for CPU only${ColourReset}")
    endif()
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
    # HSA_PATH
    if(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    else()
        SET(HSA_PATH $ENV{HSA_PATH})
    endif()
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_PATH} " version: " ${HIP_VERSION})
        if(HIP_COMPILER STREQUAL clang)
            set(hip_library_name amdhip64)
            add_subdirectory(hipvx)
            add_library(vxu SHARED api/vxu.cpp)
            target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=0)
            target_compile_definitions(vxu PUBLIC ENABLE_HIP=1)
            add_library(openvx SHARED ${SOURCES})
            set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
            set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
            include_directories(${HIP_PATH}/include ${HSA_PATH}/include)
            include_directories(hipvx)
            target_compile_definitions(openvx PUBLIC ENABLE_HIP=1 )
            target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
            link_directories(${HIP_PATH}/lib)
            message("-- ${Green}AMD OpenVX -- OpenVX built with HIP Support${ColourReset}")
        else()
            message(FATAL_ERROR "unsupported hip compiler")
        endif()
    else()
        set(GPU_SUPPORT OFF)
        add_library(openvx SHARED ${SOURCES})
        add_library(vxu SHARED api/vxu.cpp)
        target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
        target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(vxu PUBLIC ENABLE_HIP=0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
        message("-- ${Red}WARNING:HIP Not Found -- OpenVX built for CPU only${ColourReset}")
    endif()
else()
    add_library(openvx SHARED ${SOURCES})
    add_library(vxu SHARED api/vxu.cpp)
    target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
    target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
    target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=0)
    target_compile_definitions(vxu PUBLIC ENABLE_HIP=0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
    message("-- ${Red}WARNING:OpenCL/HIP Not Found or GPU Support Turned OFF -- OpenVX built for CPU only${ColourReset}")
endif()

install(TARGETS openvx DESTINATION lib)
install(TARGETS vxu DESTINATION lib)
install(FILES include/vx_ext_amd.h DESTINATION include)
install (FILES
            include/VX/vx.h
            include/VX/vx_api.h
            include/VX/vx_compatibility.h
            include/VX/vx_import.h
            include/VX/vx_kernels.h
            include/VX/vx_khr_icd.h
            include/VX/vx_khr_ix.h
            include/VX/vx_khr_nn.h
            include/VX/vx_khr_tiling.h
            include/VX/vx_khr_xml.h
            include/VX/vx_nodes.h
            include/VX/vx_types.h
            include/VX/vx_vendors.h
            include/VX/vxu.h
         DESTINATION include/VX)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /DVX_API_ENTRY=__declspec(dllexport)")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /DVX_API_ENTRY=__declspec(dllexport)")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11")
    target_link_libraries(openvx dl m)
    target_link_libraries(vxu openvx)
endif()
