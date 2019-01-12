# Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required (VERSION 2.8)
project (annmodule)
set (CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCL REQUIRED)
find_package(OpenCV QUIET)
include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )
include_directories (/opt/rocm/include)
link_directories    (/opt/rocm/lib)
list(APPEND SOURCES annmodule.cpp)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} openvx vx_nn)

set(TARGET_CPP Argmax.cpp VXtensor.cpp DGtest.cpp)
add_executable(DGTest main.cpp ${TARGET_CPP})
if (OpenCV_FOUND)
  target_compile_definitions(DGTest PUBLIC ENABLE_OPENCV=1)
  include_directories(${OpenCV_INCLUDE_DIRS})
  target_link_libraries(DGTest ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
  target_compile_definitions(DGTest PUBLIC ENABLE_OPENCV=0)
endif(OpenCV_FOUND)
target_link_libraries(DGTest openvx vx_nn ${PROJECT_NAME})
