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

set(RUNVX_PLUGIN_DIR ${CMAKE_BINARY_DIR}/runvx_plugins)
file(MAKE_DIRECTORY ${RUNVX_PLUGIN_DIR})

option(RUNVX_BUILD_IMAGE_OPENCV_PLUGIN "Build runvx OpenCV image/display plugin" ON)
option(RUNVX_BUILD_HASH_OPENSSL_PLUGIN "Build runvx OpenSSL hash plugin" ON)

if(RUNVX_BUILD_IMAGE_OPENCV_PLUGIN)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        add_subdirectory(image_opencv)
    else()
        message("-- ${Yellow}NOTE: OpenCV not found -- runvx image plugin will not be built (install OpenCV to build it)${ColourReset}")
    endif()
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/image_opencv/plugin_image_opencv.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image_opencv/plugin_image_opencv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_opencv/CMakeLists.txt
        DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/runvx_plugins/src/image_opencv
        COMPONENT runtime)
endif()

if(RUNVX_BUILD_HASH_OPENSSL_PLUGIN)
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND)
        add_subdirectory(hash_openssl)
    else()
        message("-- ${Yellow}NOTE: OpenSSL not found -- runvx hash plugin will not be built (install OpenSSL to build it)${ColourReset}")
    endif()
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/hash_openssl/plugin_hash_openssl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hash_openssl/CMakeLists.txt
        DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/runvx_plugins/src/hash_openssl
        COMPONENT runtime)
endif()

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/runvx_plugin_image.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/runvx_plugin_hash.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/runvx_plugin_loader.h
    DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/runvx_plugins/include
    COMPONENT runtime)
