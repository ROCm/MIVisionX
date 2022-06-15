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

#including GNUInstall Dir definitions.
include( GNUInstallDirs )

set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm Installation Path")
# need to do this to avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "mivisionx default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../amd_openvx/cmake)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(../../../amd_openvx/openvx/include
  					${OpenCL_INCLUDE_DIRS}
					../../../amd_openvx_extensions/amd_nn
				   )
link_directories(${ROCM_PATH}/lib)

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

add_executable(inference_server_app ${SOURCES})
target_link_libraries(inference_server_app vx_nn openvx pthread dl)
install(TARGETS inference_server_app DESTINATION ${CMAKE_INSTALL_BINDIR})

if("${BACKEND}" STREQUAL "OPENCL")
	find_package(OpenCL QUIET)
	if(OpenCL_FOUND)
		set(ENABLE_OPENCL 1)
		set(ENABLE_HIP 0)
		include_directories(${OpenCL_INCLUDE_DIRS}/CL ${OpenCL_INCLUDE_DIRS}/Headers)
		target_link_libraries(inference_server_app ${OpenCL_LIBRARIES})
		message("-- ${Green}Inference Server App -- App built with OpenCL Support${ColourReset}")
        message("-- ${Blue}Using OpenCL Library -- ${OpenCL_LIBRARIES}${ColourReset}")
	else()
		set(ENABLE_OPENCL 0)
		set(ENABLE_HIP 0)
		message( FATAL_ERROR "-- ${Red}FATAL ERROR: inference_server_app - OpenCL Not Found${ColourReset}")
	endif()
	target_compile_definitions(inference_server_app PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
	target_compile_definitions(inference_server_app PUBLIC ENABLE_HIP=${ENABLE_HIP})
elseif("${BACKEND}" STREQUAL "HIP")
	set(OpenCL_FOUND FALSE)
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
			set(ENABLE_OPENCL 0)
			set(ENABLE_HIP 1)
		else()
			message( FATAL_ERROR "-- ${Red}FATAL_ERROR: Unsupported HIP compiler${ColourReset}")
		endif()
	else()
		set(ENABLE_OPENCL 0)
		set(ENABLE_HIP 0)
		message( FATAL_ERROR "-- ${Red}FATAL_ERROR: HIP Not Found${ColourReset}")
	endif()
	target_compile_definitions(inference_server_app PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
	target_compile_definitions(inference_server_app PUBLIC ENABLE_HIP=${ENABLE_HIP})
else()
	set(ENABLE_OPENCL 0)
	set(ENABLE_HIP 0)
	target_compile_definitions(inference_server_app PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
	target_compile_definitions(inference_server_app PUBLIC ENABLE_HIP=${ENABLE_HIP})
	message( FATAL_ERROR "-- ${Red}FATAL_ERROR: OpenCL/HIP Not Found${ColourReset}")
endif()

if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
	message("-- OpenCV Version Supported -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X")
	if(${OpenCV_VERSION_MAJOR} EQUAL 4)
	    target_compile_definitions(inference_server_app PUBLIC USE_OPENCV_4=1)
    else()
	    target_compile_definitions(inference_server_app PUBLIC USE_OPENCV_4=0)
    endif()
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(inference_server_app ${OpenCV_LIBRARIES})
else()
	message(FATAL_ERROR "OpenCV -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse4.1 -mf16c -fopenmp")
endif()
