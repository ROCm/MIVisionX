# Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
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
project (amd_opencv)

set (CMAKE_CXX_STANDARD 11)

find_package (OpenCV REQUIRED)
find_package (OpenCL)

include_directories (../../amd_openvx/openvx/include)

list (APPEND SOURCES 
	OpenCV_absdiff.cpp
	OpenCV_adaptiveThreshold.cpp
	OpenCV_add.cpp
	OpenCV_AddWeighted.cpp
	OpenCV_bilateralFilter.cpp
	OpenCV_bitwise_and.cpp
	OpenCV_bitwise_not.cpp
	OpenCV_bitwise_or.cpp
	OpenCV_bitwise_xor.cpp
	OpenCV_Blur.cpp
	OpenCV_Boxfilter.cpp
	OpenCV_brisk_compute.cpp
	OpenCV_BRISK_detector.cpp
	OpenCV_buildOpticalFlowPyramid.cpp
	OpenCV_buildPyramid.cpp
	OpenCV_Canny.cpp
	OpenCV_compare.cpp
	OpenCV_convertScaleAbs.cpp
	OpenCV_cornerHarris.cpp
	OpenCV_cornerMinEigenVal.cpp
	OpenCV_countNonZero.cpp
	OpenCV_cvtColor.cpp
	OpenCV_data_translator.cpp
	OpenCV_dilate.cpp
	OpenCV_distanceTransform.cpp
	OpenCV_divide.cpp
	OpenCV_erode.cpp
	OpenCV_FAST_feature_detector.cpp
	OpenCV_fastNlMeansDenoising.cpp
	OpenCV_fastNlMeansDenoisingColored.cpp
	OpenCV_filter2D.cpp
	OpenCV_flip.cpp
	OpenCV_Gaussianblur.cpp
	OpenCV_Good_Features_to_Track_detector.cpp
	OpenCV_integral.cpp
	OpenCV_Laplacian.cpp
	OpenCV_Medianblur.cpp
	OpenCV_morphologyEx.cpp
	OpenCV_MSER_feature_detector.cpp
	OpenCV_multiply.cpp
	OpenCV_norm.cpp
	OpenCV_orb_compute.cpp
	OpenCV_ORB_detector.cpp
	OpenCV_pyrdown.cpp
	OpenCV_pyrup.cpp
	OpenCV_resize.cpp
	OpenCV_Scharr.cpp
	OpenCV_sepFilter2D.cpp
	OpenCV_SIFT_Compute.cpp
	OpenCV_SIFT_Detect.cpp
	OpenCV_simple_blob_detector.cpp
	OpenCV_Sobel.cpp
	OpenCV_Star_feature_detector.cpp
	OpenCV_subtract.cpp
	OpenCV_SURF_Compute.cpp
	OpenCV_SURF_Detect.cpp
	OpenCV_threshold.cpp
	OpenCV_transpose.cpp
	OpenCV_VX_Nodes.cpp
	OpenCV_warpAffine.cpp
	OpenCV_warpPerspective.cpp
	publishkernels.cpp
	)

add_library(vx_opencv SHARED ${SOURCES})
target_link_libraries(vx_opencv openvx)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(vx_opencv ${OpenCV_LIBRARIES})
target_compile_definitions(vx_opencv PUBLIC USE_OPENCV_CONTRIB=0)
install (TARGETS vx_opencv DESTINATION lib)

if (OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
	target_link_libraries(vx_opencv ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
