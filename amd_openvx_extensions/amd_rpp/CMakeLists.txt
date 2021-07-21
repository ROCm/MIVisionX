# Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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
        source/AbsoluteDifferencebatchPD.cpp
        source/AccumulatebatchPD.cpp
        source/AccumulateSquaredbatchPD.cpp
        source/AccumulateWeightedbatchPD.cpp
        source/AddbatchPD.cpp
        source/BilateralFilterbatchPD.cpp
        source/BitwiseANDbatchPD.cpp
        source/BitwiseNOTbatchPD.cpp
        source/BlendbatchPD.cpp
        source/BlurbatchPD.cpp
        source/BoxFilterbatchPD.cpp
        source/BrightnessbatchPD.cpp
        source/CannyEdgeDetector.cpp
        source/ChannelCombinebatchPD.cpp
        source/ChannelExtractbatchPD.cpp
        source/ColorTemperaturebatchPD.cpp
        source/ColorTwistbatchPD.cpp
        source/ContrastbatchPD.cpp
        source/copy.cpp
        source/CropMirrorNormalizePD.cpp
        source/CropPD.cpp
        source/CustomConvolutionbatchPD.cpp
        source/DataObjectCopybatchPD.cpp
        source/DilatebatchPD.cpp
        source/ErodebatchPD.cpp
        source/ExclusiveORbatchPD.cpp
        source/ExposurebatchPD.cpp
        source/FastCornerDetector.cpp
        source/FisheyebatchPD.cpp
        source/FlipbatchPD.cpp
        source/FogbatchPD.cpp
        source/GammaCorrectionbatchPD.cpp
        source/GaussianFilterbatchPD.cpp
        source/GaussianImagePyramidbatchPD.cpp
        source/HarrisCornerDetector.cpp
        source/Histogram.cpp
        source/HistogramBalancebatchPD.cpp
        source/HistogramEqualizebatchPD.cpp
        source/HuebatchPD.cpp
        source/InclusiveORbatchPD.cpp
        source/JitterbatchPD.cpp
        source/LaplacianImagePyramid.cpp
        source/LensCorrectionbatchPD.cpp
        source/LocalBinaryPatternbatchPD.cpp
        source/LookUpTablebatchPD.cpp
        source/MagnitudebatchPD.cpp
        source/MaxbatchPD.cpp
        source/MeanStddev.cpp
        source/MedianFilterbatchPD.cpp
        source/MinbatchPD.cpp
        source/MinMaxLoc.cpp
        source/MultiplybatchPD.cpp
        source/NoisebatchPD.cpp
        source/NonLinearFilterbatchPD.cpp
        source/NonMaxSupressionbatchPD.cpp
        source/nop.cpp
        source/PhasebatchPD.cpp
        source/PixelatebatchPD.cpp
        source/RainbatchPD.cpp
        source/RandomCropLetterBoxbatchPD.cpp
        source/RandomShadowbatchPD.cpp
        source/Remap.cpp
        source/ResizebatchPD.cpp
        source/ResizeCropbatchPD.cpp
        source/ResizeCropMirrorPD.cpp
        source/RotatebatchPD.cpp
        source/SaturationbatchPD.cpp
        source/ScalebatchPD.cpp
        source/SnowbatchPD.cpp
        source/SobelbatchPD.cpp
        source/SubtractbatchPD.cpp
        source/TensorAdd.cpp
        source/TensorLookup.cpp
        source/TensorMatrixMultiply.cpp
        source/TensorMultiply.cpp
        source/TensorSubtract.cpp
        source/ThresholdingbatchPD.cpp
        source/VignettebatchPD.cpp
        source/WarpAffinebatchPD.cpp
        source/WarpPerspectivebatchPD.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

link_directories(${AMDRPP_LIBRARIES_DIR})

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL"  AND OpenCL_FOUND)
    message("-- ${Green}AMD VX RPP -- BUILDING WITH OPENCL BACKEND${ColourReset}")
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_OPENCL=1)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    add_library(vx_rpp SHARED ${SOURCES})
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}AMD VX RPP -- BUILDING WITH HIP BACKEND${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_HIP=1 -D__HIP_PLATFORM_HCC__)
    include_directories(${HIP_PATH}/include  ${HSA_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    add_library(vx_rpp SHARED ${SOURCES})
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${HIP_LIBRARY})
else()
    message("-- ${Red}AMD VX RPP -- OPENCL/HIP NOT FOUND${ColourReset}")
    message("-- ${Red}AMD VX RPP module excluded${ColourReset}")
endif()

install(TARGETS vx_rpp DESTINATION lib)
install(FILES include/vx_ext_rpp.h DESTINATION include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(OpenVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    target_link_libraries(openvx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
