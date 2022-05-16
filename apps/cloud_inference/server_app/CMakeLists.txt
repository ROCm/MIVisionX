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
project(inference_server_app)
set(CMAKE_CXX_STANDARD 14)

#set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm Installation Path")
# need to do this to avoid setting the default path to /usr/local
set(ROCM_PATH /opt/rocm CACHE PATH "mivisionx default ROCm installation path")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "mivisionx default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../amd_openvx/cmake)


list(APPEND SOURCES
		common.cpp
		arguments.cpp
		netutil.cpp
		configure.cpp
		compiler.cpp
		inference.cpp
    inference_hip.cpp
		server.cpp
		main.cpp
		profiler.cpp
		region.cpp
		shadow.cpp
	)

find_package(OpenCV REQUIRED)
  
if("${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND)
      message(FATAL_ERROR "inference_server_app - needs OpenCL Support" )
    endif()
    include_directories(../../../amd_openvx/openvx/include
        ../../../amd_openvx_extensions/amd_nn
        )
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    link_directories(${ROCM_PATH}/lib)
    add_executable(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} vx_nn openvx pthread dl)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
    message("-- ${Green}inference server app built with OPENCL Support${ColourReset}")
elseif("${BACKEND}" STREQUAL "HIP")
    # HSA_PATH
    if(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    else()
        SET(HSA_PATH $ENV{HSA_PATH})
    endif()

    list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${HIP_PATH}/cmake)
    find_package(HIP QUIET)
    if(NOT HIP_FOUND)
      message(FATAL_ERROR "inference_server_app - needs HIP support" )
    endif()
    if(HIP_COMPILER STREQUAL clang)
        set(hip_library_name amdhip64)
        list(APPEND HIP_CXX_FLAGS -std=gnu++14)
        include_directories(${ROCM_PATH}/include ${HIP_PATH}/include ${HSA_PATH}/include)
        include_directories(../../../amd_openvx/openvx/include ../../../amd_openvx_extensions/amd_nn)
        link_directories( ${ROCM_PATH}/lib ${HIP_PATH}/lib)
        add_executable(${PROJECT_NAME} ${SOURCES})
        target_link_libraries(${PROJECT_NAME} vx_nn openvx pthread dl ${hip_library_name})
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
        message("-- ${Green}inference server app built with HIP Support${ColourReset}")
    else()
        message(FATAL_ERROR "unsupported hip compiler")
    endif()
endif()

if(OpenCV_FOUND)
	target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
	target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
endif(OpenCV_FOUND)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse4.1 -mf16c")
endif()
