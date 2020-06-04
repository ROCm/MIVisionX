# Copyright (c) 2012-2014 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

project(test_data_generator)

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
endif()
#TODO: enable C99 features for non-gnu platforms (MSVC - ???)

if(MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode)
  option(ENABLE_SOLUTION_FOLDERS "Solution folder in Visual Studio or in other IDEs" ON)
endif()

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "Build type is not specified, use: Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "" FORCE)
endif()

# DEBUG/NDEBUG preprocessor definition
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DDEBUG)
else()
  add_definitions(-DNDEBUG)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


option(BUILD_HC_TEST_DATA_GENERATOR "Build generator for test data for Harris Corners tests" OFF)

if(BUILD_HC_TEST_DATA_GENERATOR)
  # Harris Corners test data generator needs OpenCV 2.4.7 with patch
  add_subdirectory(gen_harris_corners)
else()
  # these test data generators needs clean OpenCV 2.4.7
  add_subdirectory(gen_optflow_pyrlk)
  add_subdirectory(gen_canny)
endif()

