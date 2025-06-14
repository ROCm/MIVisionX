# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

project(inference_server_app)

# ROCm Path
set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "MIVisionX default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add Default libdir
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

# Find Modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV -- Display Component
find_package(OpenCV REQUIRED)

# RPP
find_package(AMDRPP QUIET)

# Choose Backend - TBD: ADD FindMIVisionX.cmake
set(MIVISIONX_BACKEND_OPENCL_FOUND 0)
set(MIVISIONX_BACKEND_HIP_FOUND 0)
if(EXISTS ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/openvx_backend.h)
    file(READ ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/openvx_backend.h MIVISIONX_CONFIG_FILE)
    string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${MIVISIONX_CONFIG_FILE})
    set(MIVISIONX_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
    string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${MIVISIONX_CONFIG_FILE})
    set(MIVISIONX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
else()
    message(FATAL_ERROR "${PROJECT_NAME} -- ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/openvx_backend.h File Not Found. Install MIVisionX")
endif()

# Application Includes & Libraries
include_directories(${PROJECT_SOURCE_DIR}/include ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)
link_directories(${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR})

# Application Source Files
file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/source/*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

# Link Libraries & Set CXX Flags
target_link_libraries(${PROJECT_NAME} vx_nn openvx pthread dl)
# -msse4.2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 built-in functions and code generation.
# -mf16c   -- Support F16C built-in functions and code generation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c")

# Link OpenCL/HIP
set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

if(MIVISIONX_BACKEND_OPENCL_FOUND)
	find_package(OpenCL QUIET)
	if(OpenCL_FOUND)
		set(ENABLE_OPENCL 1)
		include_directories(${OpenCL_INCLUDE_DIRS}/CL ${OpenCL_INCLUDE_DIRS}/Headers)
		target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
		message("-- ${PROJECT_NAME} -- App built with OpenCL Support")
		message("-- ${PROJECT_NAME} -- Using OpenCL Library -- ${OpenCL_LIBRARIES}")
	else()
		message( FATAL_ERROR "${PROJECT_NAME} -- OpenCL Not Found")
	endif()
elseif(MIVISIONX_BACKEND_HIP_FOUND)
	if(NOT DEFINED HIP_PATH)
		if(NOT DEFINED ENV{HIP_PATH})
			set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
		else()
			set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
		endif()
	endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
	if(HIP_FOUND)
		message(STATUS "${PROJECT_NAME} -- Using HIP - Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION})
		set(ENABLE_HIP 1)
		set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
		set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
		include_directories(${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR})
		link_directories(${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})
		target_link_libraries(${PROJECT_NAME} hip::host)
		message("-- ${PROJECT_NAME} -- Built with HIP Support")
	else()
		message(FATAL_ERROR "${PROJECT_NAME} -- HIP Not Found")
	endif()
else()
	message(FATAL_ERROR "${PROJECT_NAME} -- OpenCL/HIP Not Found")
endif()

# Set Backend
target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=${ENABLE_HIP})

# OpenCV 3/4 Support
if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
	message("-- ${PROJECT_NAME} -- OpenCV Version Supported -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X")
	if(${OpenCV_VERSION_MAJOR} EQUAL 4)
		target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
    else()
		target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
    endif()
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else()
	message(FATAL_ERROR "${PROJECT_NAME} -- OpenCV -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
endif()

# rocAL Support
include_directories(${ROCM_PATH}/include/rocal)
link_directories(${ROCM_PATH}/lib/)
target_link_libraries(${PROJECT_NAME} rocal)

# Install Application in ROCm Bin Directory
install(TARGETS ${PROJECT_NAME} DESTINATION ${ROCM_PATH}/${CMAKE_INSTALL_BINDIR})
