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

cmake_minimum_required (VERSION 2.8)
project (rpp)

set (CMAKE_CXX_STANDARD 11)
cmake_policy(SET CMP0054 NEW)

find_package(OpenCL     QUIET)

include_directories (include)
include_directories (../../amd_openvx/openvx/include/)
include_directories (/opt/rocm/rpp/include/)

list (APPEND SOURCES
	source/kernel_rpp.cpp
	source/internal_publishKernels.cpp
    source/Brightness.cpp
    source/BrightnessbatchPS.cpp 
    source/BrightnessbatchPD.cpp 
    source/BrightnessbatchPDROID.cpp
    source/GammaCorrection.cpp
    source/GammaCorrectionbatchPD.cpp
    source/GammaCorrectionbatchPS.cpp 
    source/GammaCorrectionbatchPDROID.cpp
    source/Blend.cpp
    source/BlendbatchPD.cpp
    source/BlendbatchPS.cpp 
    source/BlendbatchPDROID.cpp
    source/Blur.cpp
    source/BlurbatchPS.cpp 
    source/BlurbatchPD.cpp 
    source/BlurbatchPDROID.cpp 
    source/Contrast.cpp
    source/ContrastbatchPD.cpp
    source/ContrastbatchPS.cpp 
    source/ContrastbatchPDROID.cpp 
    source/Pixelate.cpp
    source/PixelatebatchPD.cpp
    source/PixelatebatchPS.cpp 
    source/PixelatebatchPDROID.cpp
    source/Jitter.cpp
    source/JitterbatchPD.cpp
    source/JitterbatchPS.cpp 
    source/JitterbatchPDROID.cpp
    source/Snow.cpp
    source/SnowbatchPD.cpp
    source/SnowbatchPS.cpp
    source/SnowbatchPDROID.cpp
    source/Noise.cpp
    source/NoisebatchPD.cpp
    source/NoisebatchPS.cpp 
    source/NoisebatchPDROID.cpp
    source/Fog.cpp
    source/FogbatchPD.cpp
    source/FogbatchPS.cpp 
    source/FogbatchPDROID.cpp
    source/Rain.cpp
    source/RainbatchPD.cpp
    source/RainbatchPS.cpp 
    source/RainbatchPDROID.cpp
    source/Exposure.cpp
    source/ExposurebatchPD.cpp
    source/ExposurebatchPS.cpp 
    source/ExposurebatchPDROID.cpp
    source/AbsoluteDifference.cpp
    source/AbsoluteDifferencebatchPD.cpp
    source/AbsoluteDifferencebatchPS.cpp 
    source/AbsoluteDifferencebatchPDROID.cpp
    source/Add.cpp
    source/AddbatchPD.cpp
    source/AddbatchPS.cpp 
    source/AddbatchPDROID.cpp
    source/Subtract.cpp
    source/SubtractbatchPD.cpp
    source/SubtractbatchPS.cpp 
    source/SubtractbatchPDROID.cpp
    source/BitwiseAND.cpp 
    source/BitwiseANDbatchPD.cpp
    source/BitwiseANDbatchPS.cpp 
    source/BitwiseANDbatchPDROID.cpp
    source/BitwiseNOT.cpp 
    source/BitwiseNOTbatchPD.cpp
    source/BitwiseNOTbatchPS.cpp 
    source/BitwiseNOTbatchPDROID.cpp
    source/ExclusiveOR.cpp
    source/ExclusiveORbatchPD.cpp
    source/ExclusiveORbatchPS.cpp 
    source/ExclusiveORbatchPDROID.cpp  
    source/InclusiveOR.cpp
    source/InclusiveORbatchPD.cpp
    source/InclusiveORbatchPS.cpp 
    source/InclusiveORbatchPDROID.cpp 
    source/Magnitude.cpp
    source/MagnitudebatchPD.cpp
    source/MagnitudebatchPS.cpp 
    source/MagnitudebatchPDROID.cpp
    source/Multiply.cpp
    source/MultiplybatchPD.cpp
    source/MultiplybatchPS.cpp 
    source/MultiplybatchPDROID.cpp  
    source/Phase.cpp
    source/PhasebatchPD.cpp
    source/PhasebatchPS.cpp 
    source/PhasebatchPDROID.cpp
    source/Thresholding.cpp
    source/ThresholdingbatchPD.cpp
    source/ThresholdingbatchPS.cpp 
    source/ThresholdingbatchPDROID.cpp
    source/Max.cpp
    source/MaxbatchPD.cpp
    source/MaxbatchPS.cpp 
    source/MaxbatchPDROID.cpp 
    source/Min.cpp
    source/MinbatchPD.cpp
    source/MinbatchPS.cpp 
    source/MinbatchPDROID.cpp 
    source/RandomShadow.cpp
    source/RandomShadowbatchPD.cpp
    source/RandomShadowbatchPS.cpp 
    source/RandomShadowbatchPDROID.cpp
    source/HistogramBalance.cpp
    source/HistogramBalancebatchPD.cpp
    source/HistogramBalancebatchPS.cpp 
    source/HistogramBalancebatchPDROID.cpp
    source/HistogramEqualize.cpp
    source/HistogramEqualizebatchPD.cpp
    source/HistogramEqualizebatchPS.cpp 
    source/HistogramEqualizebatchPDROID.cpp
    source/ColorTemperature.cpp
    source/ColorTemperaturebatchPD.cpp
    source/ColorTemperaturebatchPS.cpp 
    source/ColorTemperaturebatchPDROID.cpp  
    source/Sobel.cpp
    source/SobelbatchPD.cpp
    source/SobelbatchPS.cpp 
    source/SobelbatchPDROID.cpp 
    source/Vignette.cpp
    source/VignettebatchPD.cpp
    source/VignettebatchPS.cpp 
    source/VignettebatchPDROID.cpp
    source/Dilate.cpp
    source/DilatebatchPD.cpp
    source/DilatebatchPS.cpp 
    source/DilatebatchPDROID.cpp 
    source/Erode.cpp
    source/ErodebatchPD.cpp
    source/ErodebatchPS.cpp 
    source/ErodebatchPDROID.cpp
    source/BoxFilter.cpp
    source/BoxFilterbatchPD.cpp
    source/BoxFilterbatchPS.cpp 
    source/BoxFilterbatchPDROID.cpp
    source/NonMaxSupression.cpp
    source/NonMaxSupressionbatchPD.cpp
    source/NonMaxSupressionbatchPS.cpp 
    source/NonMaxSupressionbatchPDROID.cpp
    source/MedianFilter.cpp
    source/MedianFilterbatchPD.cpp
    source/MedianFilterbatchPS.cpp 
    source/MedianFilterbatchPDROID.cpp
    source/Flip.cpp
    source/FlipbatchPD.cpp
    source/FlipbatchPS.cpp 
    source/FlipbatchPDROID.cpp
    source/Resize.cpp
    source/ResizebatchPD.cpp
    source/ResizebatchPS.cpp 
    source/ResizebatchPDROID.cpp
    source/ResizeCrop.cpp
    source/ResizeCropbatchPD.cpp
    source/ResizeCropbatchPS.cpp 
    source/ResizeCropbatchPDROID.cpp
    source/Rotate.cpp 
    source/RotatebatchPD.cpp
    source/RotatebatchPS.cpp 
    source/RotatebatchPDROID.cpp
    source/Fisheye.cpp 
    source/FisheyebatchPD.cpp
    source/FisheyebatchPS.cpp 
    source/FisheyebatchPDROID.cpp
    source/LensCorrection.cpp 
    source/LensCorrectionbatchPD.cpp
    source/LensCorrectionbatchPS.cpp 
    source/LensCorrectionbatchPDROID.cpp
    source/Scale.cpp  
    source/ScalebatchPD.cpp
    source/ScalebatchPS.cpp 
    source/ScalebatchPDROID.cpp
    source/ChannelExtract.cpp
    source/ChannelExtractbatchPD.cpp
    source/ChannelExtractbatchPS.cpp 
    source/ChannelCombine.cpp
    source/ChannelCombinebatchPD.cpp
    source/ChannelCombinebatchPS.cpp
    source/BilateralFilter.cpp
    source/BilateralFilterbatchPD.cpp
    source/BilateralFilterbatchPS.cpp 
    source/BilateralFilterbatchPDROID.cpp
    source/GaussianFilter.cpp 
    source/GaussianFilterbatchPD.cpp
    source/GaussianFilterbatchPS.cpp 
    source/GaussianFilterbatchPDROID.cpp
    source/NonLinearFilter.cpp 
    source/NonLinearFilterbatchPD.cpp
    source/NonLinearFilterbatchPS.cpp 
    source/NonLinearFilterbatchPDROID.cpp
    source/LocalBinaryPattern.cpp
    source/LocalBinaryPatternbatchPD.cpp
    source/LocalBinaryPatternbatchPS.cpp 
    source/LocalBinaryPatternbatchPDROID.cpp
    source/DataObjectCopy.cpp 
    source/DataObjectCopybatchPD.cpp
    source/DataObjectCopybatchPS.cpp
    source/DataObjectCopybatchPDROID.cpp
    source/GaussianImagePyramid.cpp
    source/GaussianImagePyramidbatchPD.cpp
    source/GaussianImagePyramidbatchPS.cpp 
    source/LaplacianImagePyramid.cpp 
    source/CannyEdgeDetector.cpp 
    source/HarrisCornerDetector.cpp 
    source/FastCornerDetector.cpp 
    source/Accumulate.cpp
    source/AccumulatebatchPD.cpp
    source/AccumulatebatchPS.cpp 
    source/AccumulatebatchPDROID.cpp
    source/AccumulateSquared.cpp
    source/AccumulateSquaredbatchPD.cpp
    source/AccumulateSquaredbatchPS.cpp 
    source/AccumulateSquaredbatchPDROID.cpp
    source/AccumulateWeighted.cpp
    source/AccumulateWeightedbatchPD.cpp
    source/AccumulateWeightedbatchPS.cpp 
    source/AccumulateWeightedbatchPDROID.cpp
    source/CannyEdgeDetector.cpp
    source/ControlFlow.cpp
    source/ControlFlowbatchPD.cpp
    source/ControlFlowbatchPS.cpp 
    source/ControlFlowbatchPDROID.cpp
    source/CustomConvolution.cpp
    source/CustomConvolutionbatchPD.cpp
    source/CustomConvolutionbatchPS.cpp 
    source/CustomConvolutionbatchPDROID.cpp
    source/Histogram.cpp
    source/Hue.cpp
    source/HuebatchPD.cpp
    source/HuebatchPS.cpp 
    source/HuebatchPDROID.cpp
    source/LookUpTable.cpp
    source/LookUpTablebatchPD.cpp
    source/LookUpTablebatchPS.cpp 
    source/LookUpTablebatchPDROID.cpp
    source/MeanStddev.cpp
    source/MinMaxLoc.cpp
    source/Occlusion.cpp
    source/OcclusionbatchPD.cpp
    source/OcclusionbatchPS.cpp 
    source/OcclusionbatchPDROID.cpp
    source/Saturation.cpp
    source/SaturationbatchPD.cpp
    source/SaturationbatchPS.cpp 
    source/SaturationbatchPDROID.cpp
    source/RandomCropLetterBox.cpp
    source/RandomCropLetterBoxbatchPD.cpp
    source/RandomCropLetterBoxbatchPS.cpp 
    source/RandomCropLetterBoxbatchPDROID.cpp
    source/Remap.cpp
    source/WarpAffine.cpp
    source/WarpAffinebatchPD.cpp
    source/WarpAffinebatchPS.cpp 
    source/WarpAffinebatchPDROID.cpp
    source/WarpPerspective.cpp
    source/WarpPerspectivebatchPD.cpp
    source/WarpPerspectivebatchPS.cpp 
    source/WarpPerspectivebatchPDROID.cpp
    source/TensorAdd.cpp
    source/TensorLookup.cpp 
    source/TensorMatrixMultiply.cpp 
    source/TensorMultiply.cpp
    source/TensorSubtract.cpp 
    source/ColorTwist.cpp
    source/ColorTwistbatchPD.cpp
    source/CropMirrorNormalizePD.cpp
    source/CropPD.cpp
    source/ResizeCropMirrorPD.cpp
    source/nop.cpp
    source/copy.cpp

	)

link_directories(/opt/rocm/rpp/lib/)

if (OpenCL_FOUND)
	message( "BUILDING WITH OPENCL BACKEND")
	set(ENABLE_OPENCL 1)
	add_definitions( -DENABLE_OPENCL=${ENABLE_OPENCL} -DRPP_BACKEND_OPENCL=1)
	include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp  boost_filesystem  boost_system ${OpenCL_LIBRARIES})
else(OpenCL_FOUND)
	message( "OPENCL NOT FOUND")
	set(ENABLE_OPENCL 0)
	add_definitions( -DENABLE_OPENCL=${ENABLE_OPENCL} )
	add_library(vx_rpp SHARED ${SOURCES})
	target_link_libraries(vx_rpp amd_rpp boost_filesystem  boost_system)
endif(OpenCL_FOUND)

install (TARGETS vx_rpp DESTINATION lib)
install (FILES include/vx_ext_rpp.h include/kernels_rpp.h DESTINATION include)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries( OpenVX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	target_link_libraries( openvx)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
