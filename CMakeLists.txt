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

cmake_minimum_required(VERSION 2.8)
project(MIVisionX)
set(VERSION "1.9.4")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ROCM_PATH /opt/rocm CACHE PATH "mivisionx default ROCm installation path")
# need to do this to avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH}/mivisionx CACHE PATH "mivisionx default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(BoldBlue    "${Esc}[1;34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
endif()
message("-- ${BoldBlue}MIVisionX Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}MIVisionX install path set to -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

option(NEURAL_NET "Build MIVisionX with Neural Net Support"   ON)
option(RALI       "Build MIVisionX with RALI Support"         ON)
option(LOOM       "Build MIVisionX with LOOM Support"         ON)
option(OPENCL_SUPPORT "Build MIVisionX with OpenCL Support"   ON)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(OPENCL_SUPPORT OFF)
  message("-- ${Magenta}Apple macOS Detected -- OpenCL Support turned OFF${ColourReset}")
endif()

add_subdirectory(amd_openvx)
add_subdirectory(amd_openvx_extensions)
add_subdirectory(apps)
add_subdirectory(utilities)
if(RALI)
  add_subdirectory(rali)
else()
  message("-- ${Cyan}RALI Module turned OFF by user option -DRALI=OFF ${ColourReset}")
endif()

install(DIRECTORY docs DESTINATION .)

# set package information
set(CPACK_PACKAGE_VERSION       ${VERSION})
set(CPACK_PACKAGE_NAME          "mivisionx")
set(CPACK_PACKAGE_RELEASE       1)
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "Kiriti Gowda <Kiriti.NageshGowda@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD Radeon")
set(CPACK_PACKAGE_ICON          "${CMAKE_SOURCE_DIR}/docs/images/MIVisionX.bmp")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY   "AMD MIVisionX toolkit is a comprehensive computer vision and machine intelligence libraries, utilities and applications bundled into one.")

# find linux distribution
find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
                OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                )

# generate .deb or .rpm package
if(LSB_RELEASE_ID_SHORT STREQUAL "Ubuntu" AND NOT APPLE)
    message("-- Ubuntu detected   -- .deb package will be created")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
    set(CPACK_GENERATOR                 "DEB")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS    "miopen-opencl")
elseif(APPLE)
    message("-- Apple detected    -- OSX package will be created")
    set(CPACK_BUNDLE_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
    set(CPACK_GENERATOR                 "Bundle")
else()
    message("-- CentOS detected   -- .rpm package will be created")
    set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
    set(CPACK_GENERATOR                 "RPM")
    set(CPACK_RPM_PACKAGE_REQUIRES      "miopen-opencl")
    set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
include(CPack)
