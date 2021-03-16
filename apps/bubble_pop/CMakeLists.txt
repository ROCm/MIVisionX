################################################################################
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc.
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
###############################################################################
cmake_minimum_required(VERSION 3.0)
project(vx_pop)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(include)

if(DEFINED OPENVX_INCLUDES)
    message(STATUS "OPENVX_INCLUDES: \t${OPENVX_INCLUDES}")
    include_directories(${OPENVX_INCLUDES})
else()
    include_directories (/opt/rocm/mivisionx/include)
endif()

if(DEFINED OPENVX_LIBRARIES)
    message(STATUS "OPENVX_LIBRARIES: \t${OPENVX_LIBRARIES}")
else()
    set(OPENVX_LIBRARIES /opt/rocm/mivisionx/lib/libopenvx.so)
endif()

list    (APPEND SOURCES
		source/AMD_VX_Pop_Bubble.cpp
        source/AMD_VX_Pop_Donut.cpp
		source/internal_dataTranslator.cpp
		source/internal_publishKernels.cpp
		source/internal_vxNodes.cpp
	    )

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(vxPop source/AMD_app.cpp)
target_link_libraries(vxPop ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES} ${PROJECT_NAME})

file(COPY image DESTINATION ${CMAKE_BINARY_DIR})
file(COPY gdf DESTINATION ${CMAKE_BINARY_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries(vx_opencv OpenVX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
