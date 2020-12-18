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
project (mvobjdetect)
set (CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCL REQUIRED)
find_package(OpenCV QUIET)
include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )
include_directories (/opt/rocm/mivisionx/include ${PROJECT_SOURCE_DIR} )
link_directories    (/opt/rocm/mivisionx/lib ${PROJECT_SOURCE_DIR}/lib)
option (USE_POSTPROC  "Use postprocessing module implementation" ON)
set(SOURCES mvobjdetect.cpp mvdeploy_api.cpp visualize.cpp)
add_executable(mvobjdetect ${SOURCES})
if (OpenCV_FOUND)
  target_compile_definitions(mvobjdetect PUBLIC ENABLE_OPENCV=1)
  include_directories(${OpenCV_INCLUDE_DIRS})
  target_link_libraries(mvobjdetect ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
  target_compile_definitions(mvobjdetect PUBLIC ENABLE_OPENCV=0)
endif(OpenCV_FOUND)
#add optional postprocess module
if (USE_POSTPROC)
  include_directories (mv_extras)
  add_subdirectory (mv_extras)
  set(EXTRA_LIBS ${EXTRA_LIBS} mv_extras)
endif (USE_POSTPROC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse4.1 -mf16c")
target_compile_definitions (mvobjdetect PRIVATE ENABLE_MVDEPLOY=1)
target_link_libraries(mvobjdetect openvx vx_nn vx_amd_media pthread mv_deploy ${EXTRA_LIBS} ${CMAKE_DL_LIBS})
