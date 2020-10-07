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

cmake_minimum_required( VERSION 3.5 )
project(openvx CXX)

set(CMAKE_CXX_STANDARD 14)

if( "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
elseif( "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
endif()

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


#add_library(openvx SHARED ${SOURCES})

# Backend Specific Settings
if ("${BACKEND}" STREQUAL "OPENCL"  AND OpenCL_FOUND)
    add_library(openvx SHARED ${SOURCES})
    target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=1)
    target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    target_link_libraries(openvx ${OpenCL_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
    message("-- ${Green}AMD OpenVX -- OpenVX built with OpenCL Support${ColourReset}")

elseif ("${BACKEND}" STREQUAL "HIP")
    # Setup HIPCC compiler
    if(NOT DEFINED HIP_PATH)
      if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path to which HIP has been installed")
      else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
      endif()
    endif()
    if(NOT DEFINED ENV{HCC_PATH})
        set(HCC_HOME "${HIP_PATH}/../hcc")
        set(HCC_PATH "${HIP_PATH}/../hcc")
    else()
        set(HCC_PATH $ENV{HCC_PATH})
        set(HCC_HOME $ENV{HCC_PATH})
    endif()
    # HSA_PATH
    IF(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    ELSE()
        SET(HSA_PATH $ENV{HSA_PATH})
    ENDIF()

    find_package(HIP QUIET REQUIRED)
    if(HIP_FOUND)
      message(STATUS "Found HIP: " ${HIP_VERSION} ${HIP_PATH} ${HCC_HOME})
    else()
      message(FATAL_ERROR "Could'nt find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set.")
    endif()

    # build hip_sources if any
    list (APPEND HIP_SOURCES hipvx/arithmetic_kernels.cpp)
    #set (CMAKE_LINKER "${HIP_PATH}/bin/hipcc_cmake_linker_helper")

    #set (CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_LINKER> ${HIP_PATH}/../hcc <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    #if(HIP_PLATFORM STREQUAL "hcc")
    if(0)
        include(CheckCXXCompilerFlag)
        # A hack to make this work without the device enumerator
        link_libraries(-amdgpu-target=gfx803 -amdgpu-target=gfx900 -Wno-unused-command-line-argument)
        CHECK_CXX_COMPILER_FLAG("-amdgpu-target=gfx906 -Werror" GFX_906_FLAG_VALID)
        if(GFX_906_FLAG_VALID)
            link_libraries(-amdgpu-target=gfx906)
        endif()

        CHECK_CXX_COMPILER_FLAG(-amdgpu-target=gfx908 -Werror, GFX_908_FLAG_VALID)
        if(GFX_908_FLAG_VALID)
            link_libraries(-amdgpu-target=gfx908)
        endif()
        # end hack
    else()
        # Hcc's clang always defines __HCC__ even when not using hcc driver
	#add_definitions(-U__HCC__)
    endif()

    if(HIP_COMPILER STREQUAL clang)
        set(hip_library_name amdhip64)
    else()
        set(hip_library_name hip_hcc)
    endif()
    message("HIP library name: ${hip_library_name}")
    #FIND_LIBRARY(MIVISIONX_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${HIP_PATH}/lib)

    set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    set(HIP_HCC_FLAGS -std=c++14)
    hip_add_library(openvx_hip ${HIP_SOURCES} HIPCC_OPTIONS "${HIP_HCC_FLAGS}" HCC_OPTIONS "${HIP_HCC_FLAGS}" SHARED)
    target_compile_definitions(openvx_hip PRIVATE __HIP_PLATFORM_HCC__)
    set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    list(APPEND HIP_CXX_FLAGS "-msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -std=c++14")
    set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
    message("Cmake CXX compiler " ${CMAKE_CXX_COMPILER})

    add_library(openvx SHARED ${SOURCES})
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    include_directories(${HIP_PATH}/include ${HCC_PATH}/include ${HSA_PATH}/include)
    include_directories(hipvx)
    target_compile_definitions(openvx PUBLIC ENABLE_HIP=1 )
    target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
    target_link_libraries (openvx openvx_hip ${${hip_library_name}})
    message("-- ${Green}AMD OpenVX -- OpenVX built with HIP Support${ColourReset}")
    install(TARGETS openvx_hip DESTINATION lib)
else()
    add_library(openvx SHARED ${SOURCES} hip/hip_kernels.h)
    target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=0)
    target_compile_definitions(openvx PUBLIC ENABLE_HIP=0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
    message("-- ${Red}WARNING:OpenCL/HIP Not Found -- OpenVX built for CPU only${ColourReset}")
endif()

install(TARGETS openvx DESTINATION lib)
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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++14")
	target_link_libraries(openvx dl m)
endif()

