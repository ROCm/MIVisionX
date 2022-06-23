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
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "MIVisionX Apps default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#Added Default libdir
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(miopen PATHS ${ROCM_PATH} REQUIRED)

if(miopen_FOUND)
    #find the MIOPEN backend type
    set(MIOPEN_BACKEND_OPENCL_FOUND 0)
    set(MIOPEN_BACKEND_HIP_FOUND 0)

    if(EXISTS ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/miopen/config.h)
        file(READ ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/miopen/config.h MIOPEN_CONFIG_FILE)
        string(REGEX MATCH "MIOPEN_BACKEND_OPENCL ([0-9]*)" _ ${MIOPEN_CONFIG_FILE})
        set(MIOPEN_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "MIOPEN_BACKEND_HIP ([0-9]*)" _ ${MIOPEN_CONFIG_FILE})
        set(MIOPEN_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${ROCM_PATH}/miopen/include/miopen/config.h file Not Found. please run the MIVisionX setup script to install the MIOpen package")
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)
link_directories(${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR})

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/source/*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

target_link_libraries(${PROJECT_NAME} vx_nn openvx pthread dl)

set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

if(MIOPEN_BACKEND_OPENCL_FOUND)
	find_package(OpenCL QUIET)
	if(OpenCL_FOUND)
		set(ENABLE_OPENCL 1)
		include_directories(${OpenCL_INCLUDE_DIRS}/CL ${OpenCL_INCLUDE_DIRS}/Headers)
		target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
		message("-- Inference Server App -- App built with OpenCL Support")
        message("-- Using OpenCL Library -- ${OpenCL_LIBRARIES}")
	else()
		message( FATAL_ERROR "-- FATAL ERROR: inference_server_app - OpenCL Not Found")
	endif()
	target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
	target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=${ENABLE_HIP})
elseif(MIOPEN_BACKEND_HIP_FOUND)
    if(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    else()
        SET(HSA_PATH $ENV{HSA_PATH})
    endif()
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_PATH} " version: " ${HIP_VERSION})
        if(HIP_COMPILER STREQUAL clang)
			set(ENABLE_HIP 1)
		else()
			message( FATAL_ERROR "Unsupported HIP compiler")
		endif()
	else()
		message( FATAL_ERROR "HIP Not Found")
	endif()
else()
	message(FATAL_ERROR "OpenCL/HIP Not Found")
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=${ENABLE_HIP})

if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
	message("-- OpenCV Version Supported -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X")
	if(${OpenCV_VERSION_MAJOR} EQUAL 4)
	    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
    else()
	    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
    endif()
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else()
	message(FATAL_ERROR "OpenCV -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION ${ROCM_PATH}/${CMAKE_INSTALL_BINDIR})