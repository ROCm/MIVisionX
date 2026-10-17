# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

project(runvx_image_opencv)

find_package(OpenCV REQUIRED)

# Detect whether this CMake is being run from the MIVisionX source tree or
# from an installed plugin source directory on a target machine.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../amd_openvx/openvx/include/VX/vx.h)
    # Build-tree mode: OpenVX headers are available.
    include_directories(../../../../amd_openvx/openvx/include)
else()
    # Standalone mode: find OpenVX headers on the target system. The plugin
    # only uses VX_DF_IMAGE_* enum definitions, so it does not need to link
    # against libopenvx.so.
    find_path(OPENVX_INCLUDE_DIR
        NAMES VX/vx.h
        PATHS
            ${CMAKE_PREFIX_PATH}/include/mivisionx
            ${CMAKE_INSTALL_PREFIX}/include/mivisionx
            /opt/rocm/include/mivisionx
        NO_DEFAULT_PATH)

    find_path(OPENVX_EXT_INCLUDE_DIR
        NAMES vx_ext_amd.h
        PATHS
            ${CMAKE_PREFIX_PATH}/include/mivisionx
            ${CMAKE_INSTALL_PREFIX}/include/mivisionx
            /opt/rocm/include/mivisionx
        NO_DEFAULT_PATH)

    if(NOT OPENVX_INCLUDE_DIR)
        message(FATAL_ERROR "OpenVX header VX/vx.h not found. Set CMAKE_PREFIX_PATH to the MIVisionX install prefix (e.g., /opt/rocm).")
    endif()
    if(NOT OPENVX_EXT_INCLUDE_DIR)
        message(FATAL_ERROR "OpenVX extension header vx_ext_amd.h not found. Set CMAKE_PREFIX_PATH to the MIVisionX install prefix (e.g., /opt/rocm).")
    endif()

    include_directories(${OPENVX_INCLUDE_DIR})
    include_directories(${OPENVX_EXT_INCLUDE_DIR})
endif()

add_library(runvx_image_opencv SHARED
    plugin_image_opencv.cpp
)

target_link_libraries(runvx_image_opencv ${OpenCV_LIBRARIES})

if(NOT DEFINED RUNVX_PLUGIN_DIR)
    set(RUNVX_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set_target_properties(runvx_image_opencv PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${RUNVX_PLUGIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${RUNVX_PLUGIN_DIR}
)

install(TARGETS runvx_image_opencv
    DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/runvx_plugins
    COMPONENT runtime)
