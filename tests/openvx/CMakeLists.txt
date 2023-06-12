################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

add_test(
  NAME
    openvx_canny_CPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_canny_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
add_test(
  NAME
    openvx_channel_extract_CPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/channel_extract"
                              "${CMAKE_CURRENT_BINARY_DIR}/channel_extract"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_channel_extract_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
add_test(
  NAME
    openvx_color_convert_CPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/color_convert"
                              "${CMAKE_CURRENT_BINARY_DIR}/color_convert"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_color_convert_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
add_test(
  NAME
    openvx_canny_GPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_canny_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
add_test(
  NAME
    openvx_channel_extract_GPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/channel_extract"
                              "${CMAKE_CURRENT_BINARY_DIR}/channel_extract"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_channel_extract_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
add_test(
  NAME
    openvx_color_convert_GPU
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/color_convert"
                              "${CMAKE_CURRENT_BINARY_DIR}/color_convert"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
set_property(TEST openvx_color_convert_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
add_test(
  NAME
    openvx_canny
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
add_test(
  NAME
    openvx_channel_extract
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/channel_extract"
                              "${CMAKE_CURRENT_BINARY_DIR}/channel_extract"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
add_test(
  NAME
    openvx_color_convert
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/color_convert"
                              "${CMAKE_CURRENT_BINARY_DIR}/color_convert"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
add_test(
  NAME
    openvx_graph
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)