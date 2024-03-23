# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5)
project(vx_loomio_media)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVUTIL_LIBRARY avutil)
find_library(AVCODEC_LIBRARY avcodec)
find_library(SWSCALE_LIBRARY swscale)
find_library(AVFORMAT_LIBRARY avformat)

if(NOT BUILD_SHARED_LIBS)
    find_program(hipcc_executable hipcc)
    message(STATUS "hipcc executable: ${hipcc_executable}")
    set(CMAKE_AR "${hipcc_executable}")
endif()


include_directories(
                    ${OpenCL_INCLUDE_DIRS} 
                    ${OpenCL_INCLUDE_DIRS}/Headers
                    ${AVUTIL_INCLUDE_DIR} 
                    ${AVCODEC_INCLUDE_DIR} 
                    ${SWSCALE_INCLUDE_DIR} 
                    ${AVFORMAT_INCLUDE_DIR}
                    ../../amd_openvx/openvx/include
                    )

list(APPEND SOURCES decoder.cpp encoder.cpp vx_loomio_media.cpp)

if(NOT BUILD_SHARED_LIBS)
    add_library(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE --emit-static-lib)
    get_property(link_libraries TARGET ${PROJECT_NAME} PROPERTY LINK_LIBRARIES)
    string (REPLACE ";" " " LINK_PROPS "${link_libraries}")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -o <TARGET> ${LINK_PROPS} <LINK_FLAGS> <OBJECTS>")
    target_link_libraries(${PROJECT_NAME} 
	INTERFACE ${OpenCL_LIBRARIES} ${AVUTIL_LIBRARY} ${AVCODEC_LIBRARY} ${SWSCALE_LIBRARY} ${AVFORMAT_LIBRARY} openvx)
    # install MIVisionX libs -- {ROCM_PATH}/lib
    install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT static)
else()
    add_library(${PROJECT_NAME} SHARED ${SOURCES})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} ${AVUTIL_LIBRARY} ${AVCODEC_LIBRARY} ${SWSCALE_LIBRARY} ${AVFORMAT_LIBRARY} openvx)
    # install MIVisionX libs -- {ROCM_PATH}/lib
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
endif()


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -msse4.2 -std=gnu++14")
endif()
