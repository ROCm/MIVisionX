# Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
project(vx_amd_media)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP})
    message("-- ${Green}amd_media -- Building with OpenCL${ColourReset}")
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    set(OpenCL_FOUND FALSE)
    if(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH})
    else()
        SET(HSA_PATH $ENV{HSA_PATH})
    endif()
    find_package(HIP QUIET REQUIRED)
    if(HIP_COMPILER STREQUAL clang)
        set(HIP_LIBRARY amdhip64)
    else()
        message(FATAL_ERROR "Unsupported HIP compiler")
    endif()
    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)

    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    message("-- ${Green}amd_media -- Building with HIP${ColourReset}")

endif()

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale)

include_directories(${AVUTIL_INCLUDE_DIR}
                    ${AVCODEC_INCLUDE_DIR}
                    ${SWSCALE_INCLUDE_DIR}
                    ${AVFORMAT_INCLUDE_DIR}
                    ../../amd_openvx/openvx/include
                    include
                    )

list(APPEND SOURCES decoder.cpp encoder.cpp kernels.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${AVUTIL_LIBRARY} ${AVCODEC_LIBRARY} ${SWSCALE_LIBRARY} ${AVFORMAT_LIBRARY} openvx)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    target_link_libraries(${PROJECT_NAME} ${HIP_LIBRARY})
endif()

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_amd_media.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute. 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -msse4.2 -std=gnu++14")
endif()

message("-- ${White}amd_media -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")
