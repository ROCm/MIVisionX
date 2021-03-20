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

project(MIVisionX)
set(VERSION "2.0.0")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

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
message("-- ${BoldBlue}MIVisionX Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message("-- ${BoldBlue}MIVisionX Build Type -- Debug${ColourReset}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -ffunction-sections")
else()
  message("-- ${BoldBlue}MIVisionX Build Type -- Release${ColourReset}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
  endif()
ENDIF()

option(NEURAL_NET "Build MIVisionX with Neural Net Support"   ON)
option(RALI       "Build MIVisionX with RALI Support"         ON)
option(LOOM       "Build MIVisionX with LOOM Support"         ON)
option(OPENCL_SUPPORT "Build MIVisionX with OpenCL Support"   ON)

message("-- ${Cyan}MIVisionX Developer Options${ColourReset}")
message("-- ${Cyan}     -D NEURAL_NET=OFF [Turn OFF Neural Net Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D RALI=OFF [Turn OFF RALI Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D LOOM=OFF [Turn OFF LOOM Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D OPENCL_SUPPORT=OFF [Turn OFF OpenCL Modules (default:ON)]${ColourReset}")

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(OPENCL_SUPPORT OFF)
  message("-- ${Magenta}Apple macOS Detected -- OpenCL Support turned OFF${ColourReset}")
endif()

add_subdirectory(amd_openvx)
add_subdirectory(amd_openvx_extensions)
add_subdirectory(utilities)
if(RALI)
  add_subdirectory(rali)
else()
  message("-- ${Cyan}RALI Module turned OFF by user option -D RALI=OFF ${ColourReset}")
endif()

install(DIRECTORY docs DESTINATION .)

# set package information
set(CPACK_PACKAGE_VERSION       ${VERSION})
set(CPACK_PACKAGE_NAME          "mivisionx")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "MIVisionX Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD Radeon")
set(CPACK_PACKAGE_ICON          "${CMAKE_SOURCE_DIR}/docs/images/MIVisionX.bmp")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION   "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit
  MIVisionX toolkit is a set of comprehensive computer vision and machine intelligence libraries, \
utilities, and applications bundled into a single toolkit. AMD MIVisionX delivers highly \
optimized open-source implementation of the Khronos OpenVX and OpenVX Extensions along with \
Convolution Neural Net Model Compiler & Optimizer supporting ONNX, and Khronos NNEF exchange formats. \
The toolkit allows for rapid prototyping and deployment of optimized computer vision and \
machine learning inference workloads on a wide range of computer hardware, including \
small embedded x86 CPUs, APUs, discrete GPUs, and heterogeneous servers.")
set(CPACK_RPM_PACKAGE_SUMMARY      "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

set(CPACK_DEBIAN_FILE_NAME       "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME          "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE    "local")

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                RESULT_VARIABLE PROC_RESULT
                OUTPUT_VARIABLE EVAL_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string (APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
#set(CPACK_DEBIAN_PACKAGE_DEPENDS    "miopen-opencl") # Remove Explict requirements
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
#set(CPACK_RPM_PACKAGE_REQUIRES      "miopen-opencl") # Remove Explict requirements
set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")

if(APPLE)
  set(CPACK_GENERATOR "Bundle")
  set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
  set(CPACK_BUNDLE_ICON ${CPACK_PACKAGE_ICON})
  set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/.github/info.plist")
else()
  set(CPACK_GENERATOR "TGZ;ZIP")
  if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
  endif()
  if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
  endif()
endif()

include(CPack)
