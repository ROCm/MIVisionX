# Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
project(vx_rpp)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)

find_package(AMDRPP    REQUIRED)
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL    QUIET)
elseif(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
    IF(NOT DEFINED ENV{HSA_PATH})
        SET(HSA_PATH ${ROCM_PATH}/hsa)
    ELSE()
        SET(HSA_PATH $ENV{HSA_PATH})
    ENDIF()
    find_package(HIP QUIET REQUIRED)
    if(HIP_COMPILER STREQUAL clang)
        set(HIP_LIBRARY amdhip64)
    else()
        message(FATAL_ERROR "Unsupported HIP compiler")
    endif()
endif()

include_directories(../../amd_openvx/openvx/include/
                    ${AMDRPP_INCLUDE_DIRS}
                    include
                   )

list(APPEND SOURCES
        source/image/AbsoluteDifferencebatchPD.cpp
        source/image/AccumulatebatchPD.cpp
        source/image/AccumulateSquaredbatchPD.cpp
        source/image/AccumulateWeightedbatchPD.cpp
        source/image/AddbatchPD.cpp
        source/image/BitwiseANDbatchPD.cpp
        source/image/BitwiseNOTbatchPD.cpp
        source/image/BlendbatchPD.cpp
        source/image/BlurbatchPD.cpp
        source/image/BoxFilterbatchPD.cpp
        source/image/BrightnessbatchPD.cpp
        source/image/CannyEdgeDetector.cpp
        source/image/ChannelCombinebatchPD.cpp
        source/image/ChannelExtractbatchPD.cpp
        source/image/ColorTemperaturebatchPD.cpp
        source/image/ColorTwistbatchPD.cpp
        source/image/ContrastbatchPD.cpp
        source/image/copy.cpp
        source/image/CropMirrorNormalizePD.cpp
        source/image/CropPD.cpp
        source/image/CustomConvolutionbatchPD.cpp
        source/image/DataObjectCopybatchPD.cpp
        source/image/DilatebatchPD.cpp
        source/image/ErodebatchPD.cpp
        source/image/ExclusiveORbatchPD.cpp
        source/image/ExposurebatchPD.cpp
        source/image/FastCornerDetector.cpp
        source/image/FisheyebatchPD.cpp
        source/image/FlipbatchPD.cpp
        source/image/FogbatchPD.cpp
        source/image/GammaCorrectionbatchPD.cpp
        source/image/GaussianFilterbatchPD.cpp
        source/image/GaussianImagePyramidbatchPD.cpp
        source/image/HarrisCornerDetector.cpp
        source/image/Histogram.cpp
        source/image/HistogramBalancebatchPD.cpp
        source/image/HistogramEqualizebatchPD.cpp
        source/image/HuebatchPD.cpp
        source/image/InclusiveORbatchPD.cpp
        source/image/JitterbatchPD.cpp
        source/image/LaplacianImagePyramid.cpp
        source/image/LensCorrectionbatchPD.cpp
        source/image/LocalBinaryPatternbatchPD.cpp
        source/image/LookUpTablebatchPD.cpp
        source/image/MagnitudebatchPD.cpp
        source/image/MaxbatchPD.cpp
        source/image/MeanStddev.cpp
        source/image/MedianFilterbatchPD.cpp
        source/image/MinbatchPD.cpp
        source/image/MinMaxLoc.cpp
        source/image/MultiplybatchPD.cpp
        source/image/NoisebatchPD.cpp
        source/image/NonLinearFilterbatchPD.cpp
        source/image/NonMaxSupressionbatchPD.cpp
        source/image/nop.cpp
        source/image/PhasebatchPD.cpp
        source/image/PixelatebatchPD.cpp
        source/image/RainbatchPD.cpp
        source/image/RandomCropLetterBoxbatchPD.cpp
        source/image/RandomShadowbatchPD.cpp
        source/image/Remap.cpp
        source/image/ResizebatchPD.cpp
        source/image/ResizeCropbatchPD.cpp
        source/image/ResizeCropMirrorPD.cpp
        source/image/RotatebatchPD.cpp
        source/image/SaturationbatchPD.cpp
        source/image/ScalebatchPD.cpp
        source/image/SequenceRearrange.cpp
        source/image/SnowbatchPD.cpp
        source/image/SobelbatchPD.cpp
        source/image/SubtractbatchPD.cpp
        source/image/TensorAdd.cpp
        source/image/TensorLookup.cpp
        source/image/TensorMatrixMultiply.cpp
        source/image/TensorMultiply.cpp
        source/image/TensorSubtract.cpp
        source/image/ThresholdingbatchPD.cpp
        source/image/VignettebatchPD.cpp
        source/image/WarpAffinebatchPD.cpp
        source/image/WarpPerspectivebatchPD.cpp
        source/image/Resizetensor.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

link_directories(${AMDRPP_LIBRARIES_DIR})

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL"  AND OpenCL_FOUND)
    message("-- ${Green}amd_rpp -- Building with OpenCL${ColourReset}")
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_OPENCL=1)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    add_library(vx_rpp SHARED ${SOURCES})
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}amd_rpp -- Building with HIP${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_HIP=1 -D__HIP_PLATFORM_AMD__)
    include_directories(${HIP_PATH}/include  ${HSA_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    add_library(vx_rpp SHARED ${SOURCES})
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${HIP_LIBRARY})
else()
    message("-- ${Red}WARNING: OpenCL/HIP Not Found -- amd_rpp module excluded${ColourReset}")
endif()

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS vx_rpp DESTINATION lib)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_ext_rpp.h DESTINATION include/mivisionx)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(OpenVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    target_link_libraries(openvx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
