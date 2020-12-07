################################################################################
#
# MIT License
#
# Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc.
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
cmake_minimum_required (VERSION 3.0)

project (classification)

set (CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/rocm/mivisionx/include)
include_directories(include)


link_directories(/opt/rocm/mivisionx/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c -std=c++11")
list(APPEND SOURCES
        source/detection.cpp
        source/common.cpp
        source/classification.cpp
        source/classifier.cpp
        source/segmentation.cpp
    )

include_directories(module_files)
add_library(neuralNetModels SHARED module_files/annmodule.cpp)
target_link_libraries(neuralNetModels openvx vx_nn pthread)

add_executable(classifier  ${SOURCES})
target_compile_definitions(classifier PUBLIC ENABLE_OPENCV=1)
target_link_libraries(classifier openvx vx_nn neuralNetModels ${OpenCV_LIBRARIES}  pthread)


