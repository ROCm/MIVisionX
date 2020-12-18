# Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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
project(amd_loomsl)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS} 
		    ${OpenCL_INCLUDE_DIRS}/Headers 
		    ../../amd_openvx/openvx/include
		   )

list(APPEND SOURCES
	kernels/alpha_blend.cpp
	kernels/chroma_key.cpp
	kernels/color_convert.cpp
	kernels/exp_comp.cpp
	kernels/exposure_compensation.cpp
	kernels/kernels.cpp
	kernels/lens_distortion_remap.cpp
	kernels/merge.cpp
	kernels/multiband_blender.cpp
	kernels/noise_filter.cpp
	kernels/pyramid_scale.cpp
	kernels/seam_find.cpp
	kernels/warp.cpp
	kernels/warp_eqr_to_aze.cpp
	kernels/initialize_setup_tables.cpp
	live_stitch_api.cpp
	profiler.cpp
	)

include_directories(. kernels)
add_library(vx_loomsl SHARED ${SOURCES})
target_link_libraries(vx_loomsl ${OpenCL_LIBRARIES} openvx)

install(TARGETS vx_loomsl DESTINATION lib)
install(FILES live_stitch_api.h DESTINATION include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11")
endif()
