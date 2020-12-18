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

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../amd_openvx/cmake)

find_package(AMDRPP    REQUIRED)
find_package(OpenCL    QUIET)

include_directories(../../amd_openvx/openvx/include/
                    ${AMDRPP_INCLUDE_DIRS}
                    include
                   )

list(APPEND SOURCES
        source/AbsoluteDifference.cpp
        source/AbsoluteDifferencebatchPD.cpp
        source/AbsoluteDifferencebatchPDROID.cpp
        source/AbsoluteDifferencebatchPS.cpp
        source/Accumulate.cpp
        source/AccumulatebatchPD.cpp
        source/AccumulatebatchPDROID.cpp
        source/AccumulatebatchPS.cpp   
        source/AccumulateSquared.cpp
        source/AccumulateSquaredbatchPD.cpp
        source/AccumulateSquaredbatchPDROID.cpp
        source/AccumulateSquaredbatchPS.cpp   
        source/AccumulateWeighted.cpp
        source/AccumulateWeightedbatchPD.cpp
        source/AccumulateWeightedbatchPDROID.cpp
        source/AccumulateWeightedbatchPS.cpp   
        source/Add.cpp
        source/AddbatchPD.cpp
        source/AddbatchPDROID.cpp
        source/AddbatchPS.cpp
        source/BilateralFilter.cpp
        source/BilateralFilterbatchPD.cpp
        source/BilateralFilterbatchPDROID.cpp
        source/BilateralFilterbatchPS.cpp   
        source/BitwiseAND.cpp
        source/BitwiseANDbatchPD.cpp
        source/BitwiseANDbatchPDROID.cpp
        source/BitwiseANDbatchPS.cpp
        source/BitwiseNOT.cpp
        source/BitwiseNOTbatchPD.cpp
        source/BitwiseNOTbatchPDROID.cpp
        source/BitwiseNOTbatchPS.cpp
        source/Blend.cpp
        source/BlendbatchPD.cpp
        source/BlendbatchPS.cpp
        source/BlendbatchPDROID.cpp
        source/Blur.cpp
        source/BlurbatchPD.cpp
        source/BlurbatchPDROID.cpp
        source/BlurbatchPS.cpp
        source/BoxFilter.cpp
        source/BoxFilterbatchPD.cpp
        source/BoxFilterbatchPDROID.cpp
        source/BoxFilterbatchPS.cpp   
        source/Brightness.cpp
        source/BrightnessbatchPD.cpp
        source/BrightnessbatchPDROID.cpp
        source/BrightnessbatchPS.cpp
        source/CannyEdgeDetector.cpp
        source/CannyEdgeDetector.cpp   
        source/ChannelCombine.cpp
        source/ChannelCombinebatchPD.cpp
        source/ChannelCombinebatchPS.cpp
        source/ChannelExtract.cpp
        source/ChannelExtractbatchPD.cpp
        source/ChannelExtractbatchPS.cpp   
        source/ColorTemperature.cpp
        source/ColorTemperaturebatchPD.cpp
        source/ColorTemperaturebatchPDROID.cpp    
        source/ColorTemperaturebatchPS.cpp   
        source/ColorTwist.cpp
        source/ColorTwistbatchPD.cpp
        source/Contrast.cpp
        source/ContrastbatchPD.cpp
        source/ContrastbatchPDROID.cpp
        source/ContrastbatchPS.cpp
        source/ControlFlow.cpp
        source/ControlFlowbatchPD.cpp
        source/ControlFlowbatchPDROID.cpp
        source/ControlFlowbatchPS.cpp   
        source/copy.cpp
        source/CropMirrorNormalizePD.cpp
        source/CropPD.cpp
        source/CustomConvolution.cpp
        source/CustomConvolutionbatchPD.cpp
        source/CustomConvolutionbatchPDROID.cpp
        source/CustomConvolutionbatchPS.cpp   
        source/DataObjectCopy.cpp   
        source/DataObjectCopybatchPD.cpp
        source/DataObjectCopybatchPDROID.cpp
        source/DataObjectCopybatchPS.cpp
        source/Dilate.cpp
        source/DilatebatchPD.cpp
        source/DilatebatchPDROID.cpp   
        source/DilatebatchPS.cpp   
        source/Erode.cpp
        source/ErodebatchPD.cpp
        source/ErodebatchPDROID.cpp
        source/ErodebatchPS.cpp   
        source/ExclusiveOR.cpp
        source/ExclusiveORbatchPD.cpp
        source/ExclusiveORbatchPDROID.cpp
        source/ExclusiveORbatchPS.cpp
        source/Exposure.cpp
        source/ExposurebatchPD.cpp
        source/ExposurebatchPDROID.cpp
        source/ExposurebatchPS.cpp
        source/FastCornerDetector.cpp   
        source/Fisheye.cpp   
        source/FisheyebatchPD.cpp
        source/FisheyebatchPDROID.cpp
        source/FisheyebatchPS.cpp   
        source/Flip.cpp
        source/FlipbatchPD.cpp
        source/FlipbatchPDROID.cpp
        source/FlipbatchPS.cpp   
        source/Fog.cpp
        source/FogbatchPD.cpp
        source/FogbatchPDROID.cpp
        source/FogbatchPS.cpp
        source/GammaCorrection.cpp
        source/GammaCorrectionbatchPD.cpp
        source/GammaCorrectionbatchPDROID.cpp
        source/GammaCorrectionbatchPS.cpp
        source/GaussianFilter.cpp   
        source/GaussianFilterbatchPD.cpp
        source/GaussianFilterbatchPDROID.cpp
        source/GaussianFilterbatchPS.cpp   
        source/GaussianImagePyramid.cpp
        source/GaussianImagePyramidbatchPD.cpp
        source/GaussianImagePyramidbatchPS.cpp   
        source/HarrisCornerDetector.cpp   
        source/Histogram.cpp
        source/HistogramBalance.cpp
        source/HistogramBalancebatchPD.cpp
        source/HistogramBalancebatchPDROID.cpp
        source/HistogramBalancebatchPS.cpp   
        source/HistogramEqualize.cpp
        source/HistogramEqualizebatchPD.cpp
        source/HistogramEqualizebatchPDROID.cpp
        source/HistogramEqualizebatchPS.cpp   
        source/Hue.cpp
        source/HuebatchPD.cpp
        source/HuebatchPDROID.cpp
        source/HuebatchPS.cpp   
        source/InclusiveOR.cpp
        source/InclusiveORbatchPD.cpp
        source/InclusiveORbatchPDROID.cpp
        source/InclusiveORbatchPS.cpp
        source/Jitter.cpp
        source/JitterbatchPD.cpp
        source/JitterbatchPDROID.cpp
        source/JitterbatchPS.cpp
        source/LaplacianImagePyramid.cpp   
        source/LensCorrection.cpp   
        source/LensCorrectionbatchPD.cpp
        source/LensCorrectionbatchPDROID.cpp
        source/LensCorrectionbatchPS.cpp   
        source/LocalBinaryPattern.cpp
        source/LocalBinaryPatternbatchPD.cpp
        source/LocalBinaryPatternbatchPDROID.cpp
        source/LocalBinaryPatternbatchPS.cpp   
        source/LookUpTable.cpp
        source/LookUpTablebatchPD.cpp
        source/LookUpTablebatchPDROID.cpp
        source/LookUpTablebatchPS.cpp   
        source/Magnitude.cpp
        source/MagnitudebatchPD.cpp
        source/MagnitudebatchPDROID.cpp
        source/MagnitudebatchPS.cpp
        source/Max.cpp
        source/MaxbatchPD.cpp
        source/MaxbatchPDROID.cpp
        source/MaxbatchPS.cpp
        source/MeanStddev.cpp
        source/MedianFilter.cpp
        source/MedianFilterbatchPD.cpp
        source/MedianFilterbatchPDROID.cpp
        source/MedianFilterbatchPS.cpp   
        source/Min.cpp
        source/MinbatchPD.cpp
        source/MinbatchPDROID.cpp   
        source/MinbatchPS.cpp
        source/MinMaxLoc.cpp
        source/Multiply.cpp
        source/MultiplybatchPD.cpp
        source/MultiplybatchPDROID.cpp
        source/MultiplybatchPS.cpp
        source/Noise.cpp
        source/NoisebatchPD.cpp
        source/NoisebatchPDROID.cpp
        source/NoisebatchPS.cpp
        source/NonLinearFilter.cpp   
        source/NonLinearFilterbatchPD.cpp
        source/NonLinearFilterbatchPDROID.cpp
        source/NonLinearFilterbatchPS.cpp   
        source/NonMaxSupression.cpp
        source/NonMaxSupressionbatchPD.cpp
        source/NonMaxSupressionbatchPDROID.cpp
        source/NonMaxSupressionbatchPS.cpp   
        source/nop.cpp
        source/Occlusion.cpp
        source/OcclusionbatchPD.cpp
        source/OcclusionbatchPDROID.cpp
        source/OcclusionbatchPS.cpp   
        source/Phase.cpp
        source/PhasebatchPD.cpp
        source/PhasebatchPDROID.cpp
        source/PhasebatchPS.cpp
        source/Pixelate.cpp
        source/PixelatebatchPD.cpp
        source/PixelatebatchPDROID.cpp
        source/PixelatebatchPS.cpp
        source/Rain.cpp
        source/RainbatchPD.cpp
        source/RainbatchPDROID.cpp
        source/RainbatchPS.cpp
        source/RandomCropLetterBox.cpp
        source/RandomCropLetterBoxbatchPD.cpp
        source/RandomCropLetterBoxbatchPDROID.cpp
        source/RandomCropLetterBoxbatchPS.cpp   
        source/RandomShadow.cpp
        source/RandomShadowbatchPD.cpp
        source/RandomShadowbatchPDROID.cpp
        source/RandomShadowbatchPS.cpp   
        source/Remap.cpp
        source/Resize.cpp
        source/ResizebatchPD.cpp
        source/ResizebatchPDROID.cpp
        source/ResizebatchPS.cpp   
        source/ResizeCrop.cpp
        source/ResizeCropbatchPD.cpp
        source/ResizeCropbatchPDROID.cpp
        source/ResizeCropbatchPS.cpp   
        source/ResizeCropMirrorPD.cpp
        source/Rotate.cpp   
        source/RotatebatchPD.cpp
        source/RotatebatchPDROID.cpp
        source/RotatebatchPS.cpp   
        source/Saturation.cpp
        source/SaturationbatchPD.cpp
        source/SaturationbatchPDROID.cpp
        source/SaturationbatchPS.cpp   
        source/Scale.cpp    
        source/ScalebatchPD.cpp
        source/ScalebatchPDROID.cpp
        source/ScalebatchPS.cpp   
        source/Snow.cpp
        source/SnowbatchPD.cpp
        source/SnowbatchPDROID.cpp
        source/SnowbatchPS.cpp
        source/Sobel.cpp
        source/SobelbatchPD.cpp
        source/SobelbatchPDROID.cpp   
        source/SobelbatchPS.cpp   
        source/Subtract.cpp
        source/SubtractbatchPD.cpp
        source/SubtractbatchPDROID.cpp
        source/SubtractbatchPS.cpp
        source/TensorAdd.cpp
        source/TensorLookup.cpp   
        source/TensorMatrixMultiply.cpp   
        source/TensorMultiply.cpp
        source/TensorSubtract.cpp   
        source/Thresholding.cpp
        source/ThresholdingbatchPD.cpp
        source/ThresholdingbatchPDROID.cpp
        source/ThresholdingbatchPS.cpp
        source/Vignette.cpp
        source/VignettebatchPD.cpp
        source/VignettebatchPDROID.cpp
        source/VignettebatchPS.cpp   
        source/WarpAffine.cpp
        source/WarpAffinebatchPD.cpp
        source/WarpAffinebatchPDROID.cpp
        source/WarpAffinebatchPS.cpp
        source/WarpPerspective.cpp
        source/WarpPerspectivebatchPD.cpp
        source/WarpPerspectivebatchPDROID.cpp
        source/WarpPerspectivebatchPS.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

link_directories(${AMDRPP_LIBRARIES_DIR})

if(OpenCL_FOUND)
	message("-- ${Green}AMD VX RPP -- BUILDING WITH OPENCL BACKEND${ColourReset}")
	set(ENABLE_OPENCL 1)
	add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DRPP_BACKEND_OPENCL=1)
	include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system ${OpenCL_LIBRARIES})
else(OpenCL_FOUND)
	message("-- ${Red}AMD VX RPP -- OPENCL NOT FOUND${ColourReset}")
	set(ENABLE_OPENCL 0)
	add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL})
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp boost_filesystem boost_system)
endif(OpenCL_FOUND)

install(TARGETS vx_rpp DESTINATION lib)
install(FILES include/vx_ext_rpp.h DESTINATION include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries(OpenVX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	target_link_libraries( openvx)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
