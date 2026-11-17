# Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- amd_openvx_extensions:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(runvx)

include_directories(../../amd_openvx/openvx/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Include plugin interface headers from each self-contained plugin directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugins/image_opencv)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugins/hash_openssl)

list(APPEND SOURCES
    src/runvx.cpp
    src/runvx_plugin_loader.cpp
    src/vxArray.cpp
    src/vxConvolution.cpp
    src/vxDistribution.cpp
    src/vxEngine.cpp
    src/vxEngineUtil.cpp
    src/vxImage.cpp
    src/vxLUT.cpp
    src/vxMatrix.cpp
    src/vxParameter.cpp
    src/vxParamHelper.cpp
    src/vxPyramid.cpp
    src/vxRemap.cpp
    src/vxScalar.cpp
    src/vxThreshold.cpp
    src/vxTensor.cpp
    src/vxUtils.cpp
    )

add_executable(runvx ${SOURCES})
target_link_libraries(runvx openvx)
set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(ENABLE_OPENCL 1)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(runvx ${OpenCL_LIBRARIES})
    else()
        message("-- ${Yellow}NOTE: OpenCL Not Found -- runVX built for CPU only${ColourReset}")
    endif()
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}runVX -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        set(ENABLE_HIP 1)
        set_target_properties(runvx PROPERTIES LINKER_LANGUAGE CXX)
        set_target_properties(runvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
        include_directories(${ROCM_PATH}/include)
        link_directories(${HIP_PATH}/lib)
        target_link_libraries(runvx hip::host)
        message("-- ${Green}runvx built with HIP Support${ColourReset}")
    else()
        message("-- ${Yellow}NOTE: HIP Not Found -- runVX built for CPU only${ColourReset}")
    endif()
else()
    message("-- ${Yellow}NOTE: GPU Support Not Found or Turned OFF -- runVX built for CPU only${ColourReset}")
endif()

target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(runvx PUBLIC ENABLE_HIP=${ENABLE_HIP})

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    target_link_libraries(runvx pthread ${CMAKE_DL_LIBS})
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute.
    target_compile_options(runvx PRIVATE -Wno-deprecated-declarations)
endif()

# Build optional runtime plugins. If OpenCV/OpenSSL are not found, the plugin
# is skipped but the core runvx executable still builds and runs. Plugin sources
# are always installed so they can be built on the target machine if the
# dependency becomes available later.
add_subdirectory(plugins)

# install MIVisionX executables -- {ROCM_PATH}/bin
install(TARGETS runvx DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)

if(BUILD_TESTING)
    add_test(NAME runvx_help COMMAND runvx -h)
    set_tests_properties(runvx_help PROPERTIES PASS_REGULAR_EXPRESSION "RunVX|Usage|runvx")
endif()

# install MIVisionX apps & samples -- {ROCM_PATH}/share/mivisionx/
install(DIRECTORY ${MIVISIONX_ROOT_DIR}/samples/gdf DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)

message("-- ${White}runVX -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
