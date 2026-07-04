# Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

# VX_RPP Version
# * must match with include/vx_ext_rpp_version.h
set(VERSION "3.3.2")

project(vx_rpp VERSION ${VERSION} LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

find_package(rpp 3.1.0 REQUIRED)

message("-- ${White}${PROJECT_NAME}: rpp found with find_package(rpp REQUIRED)${ColourReset}")
message("-- \t${White}rpp_INCLUDE_DIR -- ${rpp_INCLUDE_DIR}${ColourReset}")
message("-- \t${White}rpp_LIB_DIR -- ${rpp_LIB_DIR}${ColourReset}")
message("-- \t${White}rpp_LIBRARY -- ${rpp_LIBRARY}${ColourReset}")
message("-- \t${White}rpp_FOUND -- ${rpp_FOUND}${ColourReset}")
message("-- \t${White}rpp_VERSION -- ${rpp_VERSION}${ColourReset}")
message("-- \t${White}rpp_VERSION_MAJOR -- ${rpp_VERSION_MAJOR}${ColourReset}")
message("-- \t${White}rpp_VERSION_MINOR -- ${rpp_VERSION_MINOR}${ColourReset}")
message("-- \t${White}rpp_VERSION_PATCH -- ${rpp_VERSION_PATCH}${ColourReset}")

# RPP supports the CPU and HIP backends only (OpenCL backend support has been dropped in RPP).
# When the MIVisionX core is built with the OpenCL backend, vx_rpp is built in CPU-only mode.
set(OpenCL_FOUND FALSE)
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP REQUIRED)
endif()

include_directories(../../amd_openvx/openvx/include/ include)
# AMD RPP
include_directories(${rpp_INCLUDE_DIR})
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rpp::rpp)

list(APPEND SOURCES
        source/tensor/Blend.cpp
        source/tensor/BitwiseOps.cpp
        source/tensor/Blur.cpp
        source/tensor/Brightness.cpp
        source/tensor/ChannelPermute.cpp
        source/tensor/ColorJitter.cpp
        source/tensor/ColorTemperature.cpp
        source/tensor/ColorCast.cpp
        source/tensor/ColorToGreyscale.cpp
        source/tensor/ColorTwist.cpp
        source/tensor/Contrast.cpp
        source/tensor/Copy.cpp
        source/tensor/Crop.cpp
        source/tensor/CropAndPatch.cpp
        source/tensor/CropMirrorNormalize.cpp
        source/tensor/Dilate.cpp
        source/tensor/Downmix.cpp
        source/tensor/Erase.cpp
        source/tensor/Erode.cpp
        source/tensor/Exposure.cpp
        source/tensor/FishEye.cpp
        source/tensor/Flip.cpp
        source/tensor/Fog.cpp
        source/tensor/GammaCorrection.cpp
        source/tensor/GaussianFilter.cpp
        source/tensor/GaussianNoise.cpp
        source/tensor/Glitch.cpp
        source/tensor/GridMask.cpp
        source/tensor/Hue.cpp
        source/tensor/Jitter.cpp
        source/tensor/JpegCompressionDistortion.cpp
        source/tensor/LensCorrection.cpp
        source/tensor/Log.cpp
        source/tensor/Log1p.cpp
        source/tensor/Magnitude.cpp
        source/tensor/MedianFilter.cpp
        source/tensor/Lut.cpp
        source/tensor/MelFilterBank.cpp
        source/tensor/Noise.cpp
        source/tensor/NonSilentRegionDetection.cpp
        source/tensor/NonLinearBlend.cpp
        source/tensor/Nop.cpp
        source/tensor/Normalize.cpp
        source/tensor/Rotate.cpp
        source/tensor/Phase.cpp
        source/tensor/Pixelate.cpp
        source/tensor/Posterize.cpp
        source/tensor/PreemphasisFilter.cpp
        source/tensor/Rain.cpp
        source/tensor/Remap.cpp
        source/tensor/Resample.cpp
        source/tensor/Resize.cpp
        source/tensor/ResizeCrop.cpp
        source/tensor/ResizeCropMirror.cpp
        source/tensor/ResizeMirrorNormalize.cpp
        source/tensor/Ricap.cpp
        source/tensor/Saturation.cpp
        source/tensor/SequenceRearrange.cpp
        source/tensor/ShotNoise.cpp
        source/tensor/Snow.cpp
        source/tensor/Solarize.cpp
        source/tensor/Spatter.cpp
        source/tensor/Spectrogram.cpp
        source/tensor/TensorAddTensor.cpp
        source/tensor/TensorMax.cpp
        source/tensor/TensorMean.cpp
        source/tensor/TensorMin.cpp
        source/tensor/TensorMulScalar.cpp
        source/tensor/TensorStdDev.cpp
        source/tensor/TensorSum.cpp
        source/tensor/Threshold.cpp
        source/tensor/ToDecibels.cpp
        source/tensor/Vignette.cpp
        source/tensor/WarpAffine.cpp
        source/tensor/WarpPerspective.cpp
        source/tensor/Water.cpp
        source/tensor/Slice.cpp
        source/tensor/Transpose.cpp
        source/tensor/PythonFunction.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

if (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}vx_rpp -- Building with HIP${ColourReset}")
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_HIP=1 -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    add_library(vx_rpp SHARED ${SOURCES})
    set_target_properties(vx_rpp PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(vx_rpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_HIP=1") # Currently used by rpp. To be removed.
else()
    message("-- ${Green}vx_rpp -- Building with CPU backend${ColourReset}")
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_HIP=${ENABLE_HIP})
    add_library(vx_rpp SHARED ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=0" ) # Currently used by rpp. To be removed.
endif()

set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND 0)
if(EXISTS ${rpp_INCLUDE_DIR}/rpp/rpp_audio_augmentations_support.h)
    file(READ ${rpp_INCLUDE_DIR}/rpp/rpp_audio_augmentations_support.h RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE)
    string(REGEX MATCH "RPP_AUDIO_AUGMENTATIONS_SUPPORT ([0-9]*)" _ ${RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE})
    set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND ${CMAKE_MATCH_1})
endif()

# Audio features for VX_RPP
if((${rpp_VERSION_MAJOR} VERSION_GREATER "3") OR (${rpp_VERSION_MAJOR} VERSION_EQUAL "3" AND ${rpp_VERSION_MINOR} VERSION_GREATER_EQUAL "1" AND ${rpp_VERSION_PATCH} VERSION_GREATER_EQUAL "1")
AND RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUDIO_SUPPORT=1")
    message("-- ${White}vx_rpp Audio Features Included${ColourReset}")
    target_compile_definitions(vx_rpp PUBLIC RPP_AUDIO=1)
else()
    target_compile_definitions(vx_rpp PUBLIC RPP_AUDIO=0)
    message("-- ${Yellow}vx_rpp Audio Features Excluded${ColourReset}")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} openvx)
    # -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

target_link_libraries(vx_rpp ${LINK_LIBRARY_LIST})
message("-- ${White}vx_rpp -- Link Libraries:${LINK_LIBRARY_LIST}${ColourReset}")
message("-- ${White}vx_rpp -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_ext_rpp.h include/vx_ext_rpp_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
install(DIRECTORY ${MIVISIONX_ROOT_DIR}/tests/vx_rpp_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
