################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
cmake_minimum_required(VERSION 3.5)

project (rocAL_dataloader_mt)

set(CMAKE_CXX_STANDARD 14)

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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCV QUIET)
find_package(AMDRPP QUIET)

if(NOT DEFINED BACKEND)
  set(BACKEND "HIP")   # set default backend to HIP
endif()

if("${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(ENABLE_OPENCL 1)
        set(ENABLE_HIP 0)
        include_directories($${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
    else()
        message("-- ${Yellow}NOTE: OpenCL Not Found -- runVX built for CPU only${ColourReset}")
    endif()
elseif ("${BACKEND}" STREQUAL "HIP")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}rocAL_dataloader_mt -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        set(ENABLE_HIP 1)
        set(ENABLE_OPENCL 0)
        include_directories(${HIP_PATH}/include)
        link_directories(${HIP_PATH}/lib)
        message("-- ${Green}rali_dataloader_mt built with HIP Support${ColourReset}")
    else()
        message("-- ${Yellow}NOTE: HIP Not Found -- rali_dataloader_mt built for CPU only${ColourReset}")
    endif()
else()
    message("-- ${Yellow}NOTE: GPU Support Not Found or Turned OFF -- rali_dataloader_mt app built for CPU only${ColourReset}")
endif()


include_directories (${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/rocal)
link_directories    (${ROCM_PATH}/lib/)

add_executable(${PROJECT_NAME} ./rocAL_dataloader_mt.cpp)
if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} GREATER_EQUAL 3 )
        message("-- OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Supported")
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} GREATER_EQUAL 4)
	        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
	        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        message("-- WARNING: OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
    endif()
else()
    message("-- WARNING: OpenCV Not Found -- No Display Support")
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
endif()

target_link_libraries(${PROJECT_NAME} rocal)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mf16c -Wall -pthread")

# Binary files not required for ASAN package
if(NOT ENABLE_ASAN_PACKAGING)
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()























