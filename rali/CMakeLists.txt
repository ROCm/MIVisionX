################################################################################
#
# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc.
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

cmake_minimum_required (VERSION 2.8)

project (rali)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCL QUIET)
find_package(AMDRPP QUIET)
find_package(TurboJpeg QUIET)

set(MIN_CMAKE_VER_REQUIRED 3.10)
set(MIN_CXX_VER_REQUIRED 7.4)
set(BUILD_RALI true)

if(NOT OpenCL_FOUND)
	set(BUILD_RALI false)
	message("-- ${Yellow}rali library requires OpenCL, Not Found ${ColourReset}")
endif()
if(NOT AMDRPP_FOUND)
	set(BUILD_RALI false)
	message("-- ${Yellow}rali library requires AMD's rpp library, Not Found ${ColourReset}")
endif()
if(NOT TurboJpeg_FOUND)
	set(BUILD_RALI false)
	message("-- ${Yellow}rali library requires TurboJpeg, Not Found ${ColourReset}")
endif()
if(CMAKE_VERSION VERSION_LESS ${MIN_CMAKE_VER_REQUIRED})
	set(BUILD_RALI false)
	message("-- ${Yellow}rali library requires cmake version greater than or equal to ${MIN_CMAKE_VER_REQUIRED}, currently ${CMAKE_VERSION} is installed${ColourReset}")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${MIN_CXX_VER_REQUIRED})
	set(BUILD_RALI false)
	message("-- ${Yellow}rali library requires g++ version greater than or equal to ${MIN_CXX_VER_REQUIRED}, currently ${CMAKE_CXX_COMPILER_VERSION} is installed${ColourReset}")
endif()


if(${BUILD_RALI})
	# Rali uses C++ 17 features
	set (CMAKE_CXX_STANDARD 17)
	message("-- ${Green}Radeon Augmentation Library -- rali module added${ColourReset}")
	install (FILES ./python/rali.py python/rali_lib.py ./python/rali_common.py ./python/rali_image.py ./python/rali_parameter.py ./python/rali_torch.py  ./python/rali_image_iterator.py DESTINATION ./rali/python )
	install (FILES include/rali_api.h include/rali_api_info.h include/rali_api_augmentation.h include/rali_api_data_loaders.h include/rali_api_types.h include/rali_api_data_transfer.h  include/rali_api_parameters.h DESTINATION ./include/)
	include_directories (
			${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers
			${TurboJpeg_INCLUDE_DIRS}
			../amd_openvx/openvx/include/
			../amd_openvx_extensions/amd_rpp/include/
			../amd_openvx_extensions/amd_media/include/
			./include)

	link_directories   ( ${AMDRPP_LIBRARIES_DIR} ${TurboJpeg_LIBRARIES_DIR})

	file(GLOB SOURCES "./source/*.cpp")

	add_library(${PROJECT_NAME} SHARED ${SOURCES})

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ")
	target_link_libraries(${PROJECT_NAME} -fPIC turbojpeg openvx vx_rpp )

	install (TARGETS rali DESTINATION lib)
else()
	message("-- ${Red}rali library is excluded ${ColourReset}")
endif()



















