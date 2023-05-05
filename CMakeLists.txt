# Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
set(VERSION "2.4.0")

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "MIVisionX default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# MIVisionX Default Options
option(ENHANCED_MESSAGE "MIVisionX Enhanced Message Option"        ON)
option(NEURAL_NET       "Build MIVisionX with Neural Net Support"  ON)
option(ROCAL            "Build MIVisionX with ROCAL Support"       ON)
option(ROCAL_PYTHON     "Build rocAL Python Package"               ON)
option(LOOM             "Build MIVisionX with LOOM Support"        ON)
option(GPU_SUPPORT      "Build MIVisionX with GPU Support"         ON)
option(MIGRAPHX         "Build MIVisionX with MIGraphX Support"    ON)
option(BUILD_DEV        "Build MIVisionX Development Package"      ON)
option(AMD_FP16_SUPPORT "Build MIVisionX with float16 Support"    OFF)

if(WIN32)
  set(BACKEND "OpenCL")
  set(ENHANCED_MESSAGE OFF)
  message("-- Windows Detected -- Backend Set to OpenCL")
endif()
set(DEFAULT_BUILD_TYPE "Release")

if(ENHANCED_MESSAGE)
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

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(GPU_SUPPORT OFF)
  set(ROCAL OFF)
  set(BACKEND "CPU")
  message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
endif()

if(NOT DEFINED BACKEND)
  set(BACKEND "HIP")   # set default backend to HIP
else()
  if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
    set(BACKEND "OPENCL")
  elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
    set(BACKEND "HIP")
  elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
    set(BACKEND "CPU")
    set(GPU_SUPPORT OFF)
  else()
    message("-- ${Red}Warning: MIVisionX backend option: ${BACKEND} -- unknown${ColourReset}")
    message("-- ${Red}Warning: MIVisionX default backend will be enforced${ColourReset}")
    set(BACKEND "HIP")
  endif()
endif()
message("-- ${BoldBlue}MIVisionX Backend set to -- ${BACKEND}${ColourReset}")

# MIVisionX Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "MIVisionX Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
  # -O0  -- Don't Optimize output file
  # -g -- generate debugging information
  # -Og -- Optimize for debugging experience rather than speed or size
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Og")
else()
  # -O3 -- Optimize output file
  # -DNDEBUG -- turn off asserts
  # -fPIC -- Generate position-independent code if possible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}MIVisionX Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

message("-- ${Cyan}MIVisionX Developer Options${ColourReset}")
message("-- ${Cyan}     -D NEURAL_NET=${NEURAL_NET} [Turn ON/OFF Neural Net Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D ROCAL=${ROCAL} [Turn ON/OFF ROCAL Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D ROCAL_PYTHON=${ROCAL_PYTHON} [Turn ON/OFF ROCAL Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D LOOM=${LOOM} [Turn ON/OFF LOOM Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D GPU_SUPPORT=${GPU_SUPPORT} [Turn ON/OFF GPU support (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D MIGRAPHX=${MIGRAPHX} [Turn ON/OFF MIGraphX Module (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select MIVisionX Backend [options:CPU/OPENCL/HIP](default:HIP)]${ColourReset}")
message("-- ${Cyan}     -D BUILD_DEV=${BUILD_DEV} [MIVisionX Developement/Runtime Build(default:ON)]${ColourReset}")
message("-- ${Cyan}     -D AMD_FP16_SUPPORT=${AMD_FP16_SUPPORT} [Turn ON/OFF OpenVX FP16 Support (default:OFF)]${ColourReset}")

if(AMD_FP16_SUPPORT)
  add_definitions(-DAMD_FP16_SUPPORT)
  message("-- ${Blue}MIVisionX -- -DAMD_FP16_SUPPORT definition added${ColourReset}")
endif(AMD_FP16_SUPPORT)

add_subdirectory(amd_openvx)
add_subdirectory(amd_openvx_extensions)
add_subdirectory(utilities)

if(ROCAL)
  add_subdirectory(rocAL)
  if(ROCAL_PYTHON)
    add_subdirectory(rocAL/rocAL_pybind)
  else()
    message("-- ${Cyan}ROCAL Python Module turned OFF by user option -D ROCAL_PYTHON=OFF ${ColourReset}")
  endif()
else()
  message("-- ${Cyan}ROCAL Module turned OFF by user option -D ROCAL=OFF ${ColourReset}")
endif()

# install MIVisionX docs -- {ROCM_PATH}/${CMAKE_INSTALL_DATADIR}/doc/mivisionx/
install(FILES docs/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/mivisionx)

# set package information
set(CPACK_PACKAGE_VERSION       ${VERSION})
set(CPACK_PACKAGE_NAME          "mivisionx")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "MIVisionX Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_ICON          "${CMAKE_SOURCE_DIR}/docs/data/MIVisionX.bmp")
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

# set license information
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/mivisionx)

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

# set dependency to ROCm if set to TRUE, default to OFF
set(ROCM_DEP_ROCMCORE OFF CACHE BOOL "Set rocm-core dependency")
if(ROCM_DEP_ROCMCORE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
  set(CPACK_RPM_PACKAGE_REQUIRES   "rocm-core")
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
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
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
