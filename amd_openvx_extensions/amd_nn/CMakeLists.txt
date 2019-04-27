# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
project (amd_nn)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)
set (CMAKE_CXX_STANDARD 11)

# Get rid of warning
if(POLICY CMP0054)
	cmake_policy(SET CMP0054 OLD)
endif()

find_package(OpenCL     REQUIRED)
find_package(miopengemm PATHS /opt/rocm)
find_package(miopen     PATHS /opt/rocm)
find_package(Protobuf)

if(NOT miopen_FOUND OR NOT miopengemm_FOUND)
    message(FATAL_ERROR "ERROR: couldn't find MIOpen and MIOpenGEMM -- make sure to install them")
endif()

list(APPEND PACKAGE_DEPENDS PACKAGE OpenCL)

if(Protobuf_FOUND)
	message("-- Found Protobuf -- installing inference generator.")
	add_subdirectory(../../utilities/inference_generator ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

include_directories(
    ../../amd_openvx/openvx/include
    src
    include
	../../utilities/inference_generator
    )
include_directories(${OpenCL_INCLUDE_DIRS})
list(APPEND SOURCES
    src/kernels.cpp
    src/activation_layer.cpp
    src/convolution_layer.cpp
    src/deconvolution_layer.cpp
    src/fully_connected_layer.cpp
    src/normalization_layer.cpp
    src/pooling_layer.cpp
    src/roi_pooling_layer.cpp
    src/softmax_layer.cpp
    src/batch_normalization_layer.cpp
    src/concat_layer.cpp
    src/slice_layer.cpp
    src/image_tensor_converter.cpp
    src/tensor_image_converter.cpp
    src/tensor_add.cpp
    src/tensor_subtract.cpp
    src/tensor_multiply.cpp
    src/tensor_matrix_multiply.cpp
    src/argmax_layer.cpp
    src/scale_layer.cpp
    src/upsample_nearest_layer.cpp
    src/tensor_table_lookup.cpp
    src/reshape_layer.cpp
    src/permute_layer.cpp
    src/prior_box_layer.cpp
    src/crop_layer.cpp
    src/crop_and_resize_layer.cpp
    )

add_library(vx_nn SHARED ${SOURCES})
target_link_libraries(vx_nn openvx MIOpen)

install (TARGETS vx_nn DESTINATION lib)
install (FILES include/vx_amd_nn.h DESTINATION include)
install (DIRECTORY ../../model_compiler DESTINATION .)
install (DIRECTORY ../../toolkit DESTINATION .)
install (DIRECTORY ../../apps DESTINATION .)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11")
endif()
