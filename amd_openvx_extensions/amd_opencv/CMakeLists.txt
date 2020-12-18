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
project(amd_opencv)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(../../amd_openvx/openvx/include
					include
				   )

list(APPEND SOURCES
	source/internal_dataTranslator.cpp
	source/internal_publishKernels.cpp
	source/internal_vxNodes.cpp
	source/OpenCV_absDiff.cpp
	source/OpenCV_adaptiveThreshold.cpp
	source/OpenCV_add.cpp
	source/OpenCV_addWeighted.cpp
	source/OpenCV_bilateralFilter.cpp
	source/OpenCV_bitwiseAnd.cpp
	source/OpenCV_bitwiseNot.cpp
	source/OpenCV_bitwiseOr.cpp
	source/OpenCV_bitwiseXor.cpp
	source/OpenCV_blur.cpp
	source/OpenCV_boxFilter.cpp
	source/OpenCV_briskCompute.cpp
	source/OpenCV_briskDetector.cpp
	source/OpenCV_buildOpticalFlowPyramid.cpp
	source/OpenCV_buildPyramid.cpp
	source/OpenCV_canny.cpp
	source/OpenCV_compare.cpp
	source/OpenCV_convertScaleAbs.cpp
	source/OpenCV_cornerHarris.cpp
	source/OpenCV_cornerMinEigenVal.cpp
	source/OpenCV_countNonZero.cpp
	source/OpenCV_cvtColor.cpp
	source/OpenCV_dilate.cpp
	source/OpenCV_distanceTransform.cpp
	source/OpenCV_divide.cpp
	source/OpenCV_erode.cpp
	source/OpenCV_fastFeatureDetector.cpp
	source/OpenCV_fastNlMeansDenoising.cpp
	source/OpenCV_fastNlMeansDenoisingColored.cpp
	source/OpenCV_filter2D.cpp
	source/OpenCV_flip.cpp
	source/OpenCV_gaussianBlur.cpp
	source/OpenCV_goodFeaturesToTrackdetector.cpp
	source/OpenCV_integral.cpp
	source/OpenCV_laplacian.cpp
	source/OpenCV_medianBlur.cpp
	source/OpenCV_morphologyEx.cpp
	source/OpenCV_mserFeatureDetector.cpp
	source/OpenCV_multiply.cpp
	source/OpenCV_norm.cpp
	source/OpenCV_orbCompute.cpp
	source/OpenCV_orbDetector.cpp
	source/OpenCV_pyrDown.cpp
	source/OpenCV_pyrUp.cpp
	source/OpenCV_resize.cpp
	source/OpenCV_scharr.cpp
	source/OpenCV_sepFilter2D.cpp
	source/OpenCV_siftCompute.cpp
	source/OpenCV_siftDetect.cpp
	source/OpenCV_simpleBlobDetector.cpp
	source/OpenCV_sobel.cpp
	source/OpenCV_starFeatureDetector.cpp
	source/OpenCV_subtract.cpp
	source/OpenCV_surfCompute.cpp
	source/OpenCV_surfDetect.cpp
	source/OpenCV_threshold.cpp
	source/OpenCV_transpose.cpp
	source/OpenCV_warpAffine.cpp
	source/OpenCV_warpPerspective.cpp
	)

add_library(vx_opencv SHARED ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(vx_opencv ${OpenCV_LIBRARIES})
target_compile_definitions(vx_opencv PUBLIC USE_OPENCV_CONTRIB=0)

install(TARGETS vx_opencv DESTINATION lib)
install(FILES include/vx_ext_opencv.h DESTINATION include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	target_link_libraries(vx_opencv OpenVX)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	target_link_libraries(vx_opencv openvx)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
endif()
