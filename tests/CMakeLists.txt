################################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- ROCM_PATH Set -- ${ROCM_PATH}${ColourReset}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(mivisionx-test)

# make test with CTest
enable_testing()
include(CTest)

if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(White       "${Esc}[37m")
endif()

# add find modules
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/mivisionx/test/cmake)

# find required libraries
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 QUIET)
find_package(OpenCV QUIET)
if(BUILT_WITH_SOURCE)
  set(MIVISIONX_BACKEND ${BACKEND})
  find_package(MIVISIONX QUIET)
  if(NOT MIVISIONX_FOUND)
    message("-- ${Yellow}${PROJECT_NAME}: Install MIVisionX before running make test / ctest${ColourReset}")
  endif()
else()
  find_package(MIVISIONX REQUIRED)
endif()

# 1 - vx_core test
add_test(NAME vx_core_test
          COMMAND ${RUNVX_EXECUTABLE}
          -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/vision_profile/43_feature_tracker.gdf
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 2 - vx_rpp Tests
if(VX_RPP_LIBRARY)
  add_test(NAME vx_rpp_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/vx_rpp_tests/gdf/test_vx_rpp.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(VX_RPP_LIBRARY)

# 7 - runvx Tests
if(RUNVX_EXECUTABLE)
  add_test(NAME runvx_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/logical/And_alt.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(RUNVX_EXECUTABLE)

# 8 - canny - vision graph
add_test(
  NAME
    openvx_canny
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_canny"
)

# 9 - channel extract - vision graph
add_test(
  NAME
    openvx_channel_extract
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/channel_extract"
                              "${CMAKE_CURRENT_BINARY_DIR}/channel_extract"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_channel_extract"
)

# 10 - color convert - vision graph
add_test(
  NAME
    openvx_color_convert
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/color_convert"
                              "${CMAKE_CURRENT_BINARY_DIR}/color_convert"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_color_convert"
)

# 11 - accumulate - vision graph
add_test(
  NAME
    openvx_accumulate
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/accumulate"
                              "${CMAKE_CURRENT_BINARY_DIR}/accumulate"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_accum"
)
set_property(TEST openvx_accumulate PROPERTY ENVIRONMENT "AGO_DUMP_GRAPH=1")

# 11_a - graph - graph test
add_test(
  NAME
    openvx_graph
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/graph"
                              "${CMAKE_CURRENT_BINARY_DIR}/graph"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_graph"
)
set_property(TEST openvx_graph PROPERTY ENVIRONMENT "AGO_DUMP_GRAPH=1;ROCM_PATH=${ROCM_PATH}")

# 11_b - graph API - graph API test
add_test(
  NAME
    openvx_graph_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/graph_api"
                              "${CMAKE_CURRENT_BINARY_DIR}/graph_api"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_graph_api"
)
set_property(TEST openvx_graph_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_c - tensor_api - tensor API test
add_test(
  NAME
    openvx_tensor_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/tensor_api"
                              "${CMAKE_CURRENT_BINARY_DIR}/tensor_api"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_tensor_api"
)
set_property(TEST openvx_tensor_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_d - data_objects - data objects API test
add_test(
  NAME
    openvx_data_objects_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/data_objects"
                              "${CMAKE_CURRENT_BINARY_DIR}/data_objects"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_data_objects_api"
)
set_property(TEST openvx_data_objects_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_e - user_kernel - user kernel API test
add_test(
  NAME
    openvx_user_kernel_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/user_kernel"
                              "${CMAKE_CURRENT_BINARY_DIR}/user_kernel"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_user_kernel_api"
)
set_property(TEST openvx_user_kernel_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_f - tensor_advanced - tensor advanced API test
add_test(
  NAME
    openvx_tensor_advanced_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/tensor_advanced"
                              "${CMAKE_CURRENT_BINARY_DIR}/tensor_advanced"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_tensor_advanced_api"
)
set_property(TEST openvx_tensor_advanced_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_g - threshold_query - threshold query API test
add_test(
  NAME
    openvx_threshold_query_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/threshold_query"
                              "${CMAKE_CURRENT_BINARY_DIR}/threshold_query"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_threshold_query_api"
)
set_property(TEST openvx_threshold_query_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_h - graph_import - graph import API test
add_test(
  NAME
    openvx_graph_import_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/graph_import"
                              "${CMAKE_CURRENT_BINARY_DIR}/graph_import"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_graph_import_api"
)
set_property(TEST openvx_graph_import_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_i - vxu_api - VXU API test
add_test(
  NAME
    openvx_vxu_api
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/vxu_api"
                              "${CMAKE_CURRENT_BINARY_DIR}/vxu_api"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_vxu_api"
)
set_property(TEST openvx_vxu_api PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_j - coverage_boost - coverage boost API test
add_test(
  NAME
    openvx_coverage_boost
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/coverage_boost"
                              "${CMAKE_CURRENT_BINARY_DIR}/coverage_boost"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_coverage_boost"
)
set_property(TEST openvx_coverage_boost PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 11_k - vision_coverage - vision feature coverage API test
add_test(
  NAME
    openvx_vision_coverage
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/openvx_api_tests/vision_coverage"
                              "${CMAKE_CURRENT_BINARY_DIR}/vision_coverage"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_vision_coverage"
)
set_property(TEST openvx_vision_coverage PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 12 - canny - vision graph force to CPU
add_test(NAME openvx_canny_CPU 
              COMMAND openvx_canny 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/canny
)
set_property(TEST openvx_canny_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 13 - channel extract - vision graph force to CPU
add_test(NAME openvx_channel_extract_CPU 
              COMMAND openvx_channel_extract 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/channel_extract
)
set_property(TEST openvx_channel_extract_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# 14 - color convert - vision graph force to CPU
add_test(NAME openvx_color_convert_CPU 
              COMMAND openvx_color_convert 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/color_convert
)
set_property(TEST openvx_color_convert_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

# OpenVX Tests
if(Python3_FOUND)
  # 15 - vision node group tests on CPU
  add_test(NAME openvx_tests_runVisionPython_CPU 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vision_tests/runVisionTests.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST openvx_tests_runVisionPython_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

  # 15_a - vision node group tests on CPU - test failures
  add_test(NAME openvx_tests_runVisionPython_CPU_5X3 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vision_tests/runVisionTests.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --width 5 --height 3
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST openvx_tests_runVisionPython_CPU_5X3 PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

  # 15_b - vision node group tests on CPU - test failures
  add_test(NAME openvx_tests_runVisionPython_CPU_10X10 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vision_tests/runVisionTests.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --width 10 --height 10
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST openvx_tests_runVisionPython_CPU_10X10 PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

  # 16 - openvx gdfs tests on CPU
  add_test(NAME openvx_gdf_tests_CPU 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/runOpenVX.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --backend_type CPU --hidden no 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST openvx_gdf_tests_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
endif(Python3_FOUND)

# HIP Tests
if(("${MIVISIONX_BACKEND}" STREQUAL "HIP") OR ("${MIVISIONX_BACKEND}" STREQUAL "OPENCL"))
  # 17 - canny - vision graph force to GPU
  add_test(NAME openvx_canny_GPU 
            COMMAND openvx_canny 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/canny
  )
  set_property(TEST openvx_canny_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")

  # 18 - channel extract - vision graph force to GPU
  add_test(NAME openvx_channel_extract_GPU 
              COMMAND openvx_channel_extract 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/channel_extract
  )
  set_property(TEST openvx_channel_extract_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")

  # 19 - color convert - vision graph force to GPU
  add_test(NAME openvx_color_convert_GPU 
              COMMAND openvx_color_convert 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/color_convert
  )
  set_property(TEST openvx_color_convert_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")

  # OpenVX Tests
  if(Python3_FOUND)
    # 20 - group vision nodes on GPU
    add_test(NAME openvx_tests_runVisionPython_GPU 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vision_tests/runVisionTests.py
          --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --hardware_mode GPU --backend_type ${MIVISIONX_BACKEND}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 21 - openvx gdfs tests on GPU
    add_test(NAME openvx_gdf_tests_GPU 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/runOpenVX.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --backend_type GPU --hidden no 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_property(TEST openvx_gdf_tests_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")

    # 22 - HIP CU-mask control verification using a 4K Remap graph
    if(("${MIVISIONX_BACKEND}" STREQUAL "HIP") AND RUNVX_EXECUTABLE)
      add_test(NAME openvx_hip_cu_mask_remap_4K
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/hip_cu_mask_tests/test_hip_cu_mask_remap.py
            --runvx ${RUNVX_EXECUTABLE}
            --gdf ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/geometric/Remap_U8_U8_Bilinear_4K.gdf
            --cu-counts "2,4,8,16,32,all"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
      set_property(TEST openvx_hip_cu_mask_remap_4K PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
    endif()
  endif(Python3_FOUND)
endif(("${MIVISIONX_BACKEND}" STREQUAL "HIP") OR ("${MIVISIONX_BACKEND}" STREQUAL "OPENCL"))