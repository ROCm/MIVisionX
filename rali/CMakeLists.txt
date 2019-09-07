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


project (rali)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../amd_openvx/cmake)

find_package(OpenCL QUIET)
find_package(AMDRPP QUIET)
find_package(TurboJpeg QUIET)


if(OpenCL_FOUND)
	if(AMDRPP_FOUND)
		if(TurboJpeg_FOUND)
			if(CMAKE_VERSION VERSION_GREATER "3.10")
				if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.1)
					# Rali uses C++ 17 features
					set (CMAKE_CXX_STANDARD 17)
					message("-- rali library is included (All dependencies found)")
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
					message("-- ${Red} rali library requires g++ version greater than or equal to 7.1, rali build is excluded${ColourReset}")
				endif()
			else()
				message("-- ${Red} rali library requires cmake version greater than or equal to 3.10, rali build is excluded${ColourReset}")
			endif()
		else()
			message("-- ${Red}WARNING:TurboJpeg Not Found -- rali library is excluded${ColourReset}")
		endif()
	else()
		message("-- ${Red}WARNING:AMDRPP Not Found -- rali library is excluded${ColourReset}")
	endif()
else()
	message("-- ${Red}WARNING:OpenCL Not Found -- rali library is excluded${ColourReset}")
endif()





















