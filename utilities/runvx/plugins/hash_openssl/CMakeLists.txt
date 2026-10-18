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

project(runvx_hash_openssl)

find_package(OpenSSL REQUIRED)

add_library(runvx_hash_openssl SHARED
    plugin_hash_openssl.cpp
)

target_link_libraries(runvx_hash_openssl OpenSSL::Crypto)

if(NOT DEFINED RUNVX_PLUGIN_DIR)
    set(RUNVX_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set_target_properties(runvx_hash_openssl PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${RUNVX_PLUGIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${RUNVX_PLUGIN_DIR}
)

install(TARGETS runvx_hash_openssl
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT runtime)
