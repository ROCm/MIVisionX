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
    find_package(OpenCL REQUIRED)
elseif(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    find_package(HIP REQUIRED)
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
        source/Resizetensor.cpp
        source/ResizeCropbatchPD.cpp
        source/ResizeCropMirrorPD.cpp
        source/RotatebatchPD.cpp
        source/SaturationbatchPD.cpp
        source/ScalebatchPD.cpp
        source/SequenceRearrange.cpp
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
    message("-- ${Green}amd_rpp -- Building with OpenCL${ColourReset}")
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_OPENCL=1)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    add_library(vx_rpp SHARED ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_OPENCL=1") # Currently used by rpp. To be removed.
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}amd_rpp -- Building with HIP${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_HIP=1 -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    add_library(vx_rpp SHARED ${SOURCES})
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # TBD -- CHECK to FIX
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_HIP=1") # Currently used by rpp. To be removed.
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system hip::host)
else()
    message("-- ${Green}amd_rpp -- Building with CPU backend${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP})
    add_library(vx_rpp SHARED ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" ) # Currently used by rpp. To be removed.
    target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system )
endif()

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS vx_rpp DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(BUILD_DEV)
    # install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
    install(FILES include/vx_ext_rpp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)
endif(BUILD_DEV)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(OpenVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    target_link_libraries(openvx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -std=gnu++14")
endif()

message("-- ${White}vx_rpp -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
