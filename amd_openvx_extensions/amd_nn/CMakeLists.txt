# Copyright (c) 2017 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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
project(amd_nn)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL     REQUIRED)
find_package(miopengemm PATHS ${ROCM_PATH} REQUIRED)
find_package(miopen     PATHS ${ROCM_PATH} REQUIRED)
find_package(Protobuf QUIET)

list(APPEND PACKAGE_DEPENDS PACKAGE OpenCL)

if(Protobuf_FOUND)
	add_subdirectory(../../utilities/inference_generator ${CMAKE_CURRENT_BINARY_DIR}/bin)
	message("-- ${Green}Inference Generator -- utilities/inference_generator module added${ColourReset}")
else(Protobuf_FOUND)
	message("-- ${Red}WARNING:Protobuf Not Found -- utilities/inference_generator module excluded${ColourReset}")	
endif(Protobuf_FOUND)

include_directories(../../amd_openvx/openvx/include
                    ${OpenCL_INCLUDE_DIRS}
                    src
                    include
	                ../../utilities/inference_generator
                    )

list(APPEND SOURCES
    src/kernels.cpp
    src/activation_layer.cpp
    src/convolution_layer.cpp
    src/deconvolution_layer.cpp
    src/fully_connected_layer.cpp
    src/normalization_layer.cpp
    src/local_response_normalization_layer.cpp
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
    src/tensor_min.cpp
    src/tensor_max.cpp
    src/tensor_exp.cpp
    src/tensor_log.cpp
    src/argmax_layer.cpp
    src/scale_layer.cpp
    src/upsample_nearest_layer.cpp
    src/tensor_table_lookup.cpp
    src/reshape_layer.cpp
    src/permute_layer.cpp
    src/prior_box_layer.cpp
    src/crop_layer.cpp
    src/crop_and_resize_layer.cpp
    src/detection_output.cpp
    src/cast_layer.cpp
    src/nms_layer.cpp
    src/gather_layer.cpp
    src/topk_layer.cpp
    src/reduce_min.cpp
    src/tile_layer.cpp
    src/profiler.cpp
    )

add_library(vx_nn SHARED ${SOURCES})
target_link_libraries(vx_nn openvx MIOpen miopengemm)

install(TARGETS vx_nn DESTINATION lib)
install(FILES include/vx_amd_nn.h DESTINATION include)
install(DIRECTORY ../../model_compiler DESTINATION .)
install(DIRECTORY ../../toolkit DESTINATION .)
install(DIRECTORY ../../apps DESTINATION .)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11")
endif()
