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
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
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

# 2 - vx_amd_media Tests
if(VX_AMD_MEDIA_LIBRARY)
  add_test(NAME vx_amd_media_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/amd_media_tests/gdf/decoder_test.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(VX_AMD_MEDIA_LIBRARY)

# 3 - vx_amd_migraphx Tests -- TBD: Disabled (ref:github issue 1410)
if(VX_AMD_MIGRAPHX_LIBRARY AND 0)
  add_test(NAME vx_amd_migraphx_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/amd_migraphx_tests/gdf/vx_migraphx_test.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(VX_AMD_MIGRAPHX_LIBRARY AND 0)

# 4 - VX_NN_LIBRARY Tests
if(VX_NN_LIBRARY)
  add_test(NAME vx_nn_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/gdf/test_vx_nn.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(VX_NN_LIBRARY)

# 5 - vx_opencv Tests
if(VX_OPENCV_LIBRARY)
  add_test(NAME vx_opencv_test
            COMMAND ${RUNVX_EXECUTABLE}
            -dump-profile file ${CMAKE_CURRENT_SOURCE_DIR}/amd_opencv_tests/gdfs/absdiff_test.gdf
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(VX_OPENCV_LIBRARY)

# 6 - vx_rpp Tests
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
set_property(TEST openvx_graph PROPERTY ENVIRONMENT "AGO_DUMP_GRAPH=1")

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
  set_property(TEST openvx_tests_runVisionPython_CPU_5X3 PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

  # 16 - openvx gdfs tests on CPU
  add_test(NAME openvx_gdf_tests_CPU 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/amd_openvx_gdfs/runOpenVX.py
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --backend_type CPU --hidden yes 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_property(TEST openvx_gdf_tests_CPU PROPERTY ENVIRONMENT "PATH=$PATH:${ROCM_PATH}/bin")
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
            --runvx_directory ${ROCM_PATH}/bin --num_frames 10 --backend_type GPU --hidden yes 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_property(TEST openvx_gdf_tests_GPU PROPERTY ENVIRONMENT "PATH=$PATH:${ROCM_PATH}/bin")
  endif(Python3_FOUND)
endif(("${MIVISIONX_BACKEND}" STREQUAL "HIP") OR ("${MIVISIONX_BACKEND}" STREQUAL "OPENCL"))

if("${MIVISIONX_BACKEND}" STREQUAL "HIP")
  # Neural Network tests
  if(VX_NN_LIBRARY AND Python3_FOUND)
    # 22 - caffe2nnir2openvx No Fuse flow
    add_test(NAME caffe2nnir2openvx_no_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 1
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 23 - caffe2nnir2openvx Fuse flow
    add_test(NAME caffe2nnir2openvx_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 2 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 24 - caffe2nnir2openvx FP16 flow
    add_test(NAME caffe2nnir2openvx_fp16
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 3 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 25 - onnx2nnir2openvx No Fuse flow
    add_test(NAME onnx2nnir2openvx_no_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 4 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )    

    # 26 - onnx2nnir2openvx Fuse flow
    add_test(NAME onnx2nnir2openvx_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 5 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 27 - onnx2nnir2openvx FP16 flow
    add_test(NAME onnx2nnir2openvx_fp16
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 6 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 28 - nnef2nnir2openvx No Fuse flow
    add_test(NAME nnef2nnir2openvx_no_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 7 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )    

    # 29 - nnef2nnir2openvx Fuse flow
    add_test(NAME nnef2nnir2openvx_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 8 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # 30 - nnef2nnir2openvx FP16 flow
    add_test(NAME nnef2nnir2openvx_fp16
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 9 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif(VX_NN_LIBRARY AND Python3_FOUND)
endif("${MIVISIONX_BACKEND}" STREQUAL "HIP")

if("${MIVISIONX_BACKEND}" STREQUAL "OPENCL" AND VX_NN_LIBRARY)
  message( "-- ${Yellow}${PROJECT_NAME}:NOTE: NN OpenCL Tests not supported${ColourReset}" )
endif("${MIVISIONX_BACKEND}" STREQUAL "OPENCL" AND VX_NN_LIBRARY)