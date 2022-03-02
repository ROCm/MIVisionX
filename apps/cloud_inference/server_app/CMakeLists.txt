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

set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm Installation Path")
# need to do this to avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH}/mivisionx CACHE PATH "mivisionx default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../amd_openvx/cmake)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL QUIET)
find_package(OpenCV REQUIRED)

if(NOT OpenCL_FOUND)
	message(FATAL_ERROR "inference_server_app - Only supported with OpenCL Backend" )
endif(NOT OpenCL_FOUND)

include_directories(../../../amd_openvx/openvx/include
  					${OpenCL_INCLUDE_DIRS}
					../../../amd_openvx_extensions/amd_nn
				   )
link_directories(${ROCM_PATH}/mivisionx/lib)

list(APPEND SOURCES
		common.cpp
		arguments.cpp
		netutil.cpp
		configure.cpp
		compiler.cpp
		inference.cpp
		server.cpp
		main.cpp
		profiler.cpp
		region.cpp
		shadow.cpp
	)

add_executable(inference_server_app ${SOURCES})
target_link_libraries(inference_server_app vx_nn openvx pthread dl)
install(TARGETS inference_server_app DESTINATION bin)

if(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS}/CL ${OpenCL_INCLUDE_DIRS}/Headers)
	target_link_libraries(inference_server_app ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)

if(OpenCV_FOUND)
	target_compile_definitions(inference_server_app PUBLIC ENABLE_OPENCV=1)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(inference_server_app ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
	target_compile_definitions(inference_server_app PUBLIC ENABLE_OPENCV=0)
endif(OpenCV_FOUND)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse4.1 -mf16c")
endif()
