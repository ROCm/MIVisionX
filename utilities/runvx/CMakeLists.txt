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
project(runvx)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV QUIET)

include_directories(../../amd_openvx/openvx/include)

list(APPEND SOURCES
    runvx.cpp
    vxArray.cpp
    vxConvolution.cpp
    vxDistribution.cpp
    vxEngine.cpp
    vxEngineUtil.cpp
    vxImage.cpp
    vxLUT.cpp
    vxMatrix.cpp
    vxParameter.cpp
    vxParamHelper.cpp
    vxPyramid.cpp
    vxRemap.cpp
    vxScalar.cpp
    vxThreshold.cpp
    vxTensor.cpp
    vxUtils.cpp
    )

#install(TARGETS runvx DESTINATION bin)
#install(DIRECTORY ../../samples DESTINATION .)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_executable(runvx ${SOURCES})
        target_link_libraries(runvx openvx pthread)
        target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=1)
        target_compile_definitions(runvx PUBLIC ENABLE_HIP=0)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(runvx ${OpenCL_LIBRARIES})
    else()
        add_executable(runvx ${SOURCES})
        target_link_libraries(runvx openvx pthread)
        target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(runvx PUBLIC ENABLE_HIP=0)
        message("-- ${Yellow}NOTE:OpenCL Not Found -- runVX built for CPU only${ColourReset}")
    endif()
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    # HSA_PATH
    if(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    else()
        SET(HSA_PATH $ENV{HSA_PATH})
    endif()

    list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/cmake)

    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_PATH} " version: " ${HIP_VERSION})
        if(HIP_COMPILER STREQUAL clang)
            set(hip_library_name amdhip64)
            list(APPEND HIP_CXX_FLAGS -std=c++14)
            include_directories (${ROCM_PATH}/include ${ROCM_PATH}/mivisionx/include ${HIP_PATH}/include ${HSA_PATH}/include)
            link_directories(${ROCM_PATH}/mivisionx/lib ${HIP_PATH}/lib)
            hip_add_executable(runvx ${SOURCES})
            target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=0)
            target_compile_definitions(runvx PUBLIC ENABLE_HIP=1)
            target_compile_options(runvx PUBLIC ${HIP_CXX_FLAGS})
            target_link_libraries (runvx openvx openvx_hip pthread)
            message("-- ${Green}runvx built with HIP Support${ColourReset}")
        else()
            message(FATAL_ERROR "unsupported hip compiler")
        endif()
    else()
        add_executable(runvx ${SOURCES})
        target_link_libraries(runvx openvx pthread)
        target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(runvx PUBLIC ENABLE_HIP=0)
        message("-- ${Yellow}NOTE:HIP Not Found -- runVX built for CPU only${ColourReset}")
    endif()
else()
    add_executable(runvx ${SOURCES})
    target_link_libraries(runvx openvx pthread)
    target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=0)
    target_compile_definitions(runvx PUBLIC ENABLE_HIP=0)
    message("-- ${Yellow}NOTE:GPU Support Not Found -- runVX built for CPU only${ColourReset}")
endif()

if(OpenCV_FOUND)
    target_compile_definitions(runvx PUBLIC ENABLE_OPENCV=1)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(runvx ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
    target_compile_definitions(runvx PUBLIC ENABLE_OPENCV=0)
    message("-- ${Yellow}NOTE:OpenCV Not Found -- runVX built without display support${ColourReset}")
endif(OpenCV_FOUND)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    find_path(SSL_INCLUDE
              NAMES openssl/hmac.h openssl/md5.h
              HINTS /usr/local/opt/openssl/include /usr/include /usr/local/include /opt/local/include
             )
    if(NOT "${SSL_INCLUDE}" STREQUAL "SSL_INCLUDE-NOTFOUND")
        include_directories(${SSL_INCLUDE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OpenSSL=1")
        target_link_libraries(runvx crypto)
    endif()
endif()
if (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    target_link_libraries(runvx ${hip_library_name})
endif()
install(TARGETS runvx DESTINATION bin)
install(DIRECTORY ../../samples DESTINATION .)
