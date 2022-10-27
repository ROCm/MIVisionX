# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5)
project(vx_amd_migraphx)

set(CMAKE_CXX_STANDARD 14)

list (APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH})
        else()
            set(HIP_PATH $ENV{HIP_PATH})
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    find_package(HIP QUIET)

    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)

    find_package(migraphx QUIET)

    include_directories(../../amd_openvx/openvx/include
    ${ROCM_PATH}/include
    include)

    list(APPEND SOURCES
            src/kernels.cpp
            src/migraphx_node.cpp
        )

    if (HIP_FOUND AND migraphx_FOUND)
        message("-- ${Green}${PROJECT_NAME} -- Building with HIP backend${ColourReset}")
        set(ENABLE_HIP 1)
        add_definitions(-DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
        link_directories(${HIP_PATH}/lib)
        add_library(${PROJECT_NAME} SHARED ${SOURCES})
        target_link_libraries(${PROJECT_NAME} openvx migraphx::c hip::host)
        # install MIVisionX libs -- {ROCM_PATH}/lib
        install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})
        # install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
	install(FILES include/vx_amd_migraphx.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)
    else()
        message("-- ${Red}WARNING: MIGraphX or HIP Not Found -- ${PROJECT_NAME} module excluded${ColourReset}")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=gnu++14")
    endif()
endif()

message("-- ${White}vx_migraphx -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
