/*
 * Copyright (c) 2012-2026 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPENVX_DOXYGEN_DEFS_H
#define OPENVX_DOXYGEN_DEFS_H

/*!
 * \file vx_doxygen_defs.h
 * \brief Doxygen-only definitions for OpenVX documentation generation.
 *
 * This file is consumed directly by Doxygen (listed in the Doxyfile INPUT)
 * and is \b not included by any OpenVX header. It provides stub definitions,
 * conditional-type aliases, and documentation group declarations that are
 * needed solely to produce correct Doxygen output.
 */

/*==============================================================================
 * Calling-convention stubs
 *
 * Platform-specific calling conventions like __stdcall are replaced with
 * empty definitions so that Doxygen can parse prototypes cleanly.
 *============================================================================*/

/*! \def VX_API_ENTRY
 * \brief Tag for exported, public API functions (empty for Doxygen).
 * \ingroup group_basic_features
 */
#define VX_API_ENTRY

/*! \def VX_API_CALL
 * \brief Defines calling convention for OpenVX API (empty for Doxygen).
 * \ingroup group_basic_features
 */
#define VX_API_CALL

/*! \def VX_CALLBACK
 * \brief Defines calling convention for user callbacks (empty for Doxygen).
 * \ingroup group_basic_features
 */
#define VX_CALLBACK

/*==============================================================================
 * Conditional types
 *
 * Types that are behind platform or experimental guards in the real headers
 * are given portable stubs here so Doxygen always documents them.
 *============================================================================*/

/*! \brief A 16-bit float value.
 * \ingroup group_basic_features
 */
typedef uint16_t vx_float16;

/*==============================================================================
 * Doxygen module/group hierarchy
 *
 * These \defgroup commands define the top-level documentation structure.
 *============================================================================*/

/*!
 * \defgroup group_basic_features Basic Features
 * \brief The basic features of OpenVX, including types, macros, and base definitions.
 *
 * \defgroup group_context Context
 * \brief The OpenVX context is the object domain for all OpenVX objects.
 *
 * \defgroup group_graph Graph
 * \brief An OpenVX graph is a container for a set of nodes.
 *
 * \defgroup group_node Node
 * \brief A node is an instance of a kernel that is paired with a set of references (the parameters).
 *
 * \defgroup group_kernel Kernel
 * \brief A kernel is a computer vision function that can be used in a graph.
 *
 * \defgroup group_parameter Parameter
 * \brief A parameter object that describes the data type expectations of a kernel.
 *
 * \defgroup group_reference Reference
 * \brief The base object type for all OpenVX objects.
 *
 * \defgroup group_scalar Scalar
 * \brief The scalar object provides a way to hold a single value.
 *
 * \defgroup group_image Image
 * \brief The image object is the primary data object for computer vision.
 *
 * \defgroup group_array Array
 * \brief The array object provides a way to hold a collection of items of a single type.
 *
 * \defgroup group_object_array Object Array
 * \brief The object array provides a way to hold a collection of opaque OpenVX objects.
 *
 * \defgroup group_tensor Tensor
 * \brief The tensor data object provides multi-dimensional data storage.
 *
 * \defgroup group_lut LUT
 * \brief The look-up table object provides a way to do pixel transformations.
 *
 * \defgroup group_distribution Distribution
 * \brief The distribution object provides a frequency distribution of values.
 *
 * \defgroup group_threshold Threshold
 * \brief The threshold object provides a way to define thresholding parameters.
 *
 * \defgroup group_matrix Matrix
 * \brief The matrix object provides a way to hold a 2D array of values.
 *
 * \defgroup group_convolution Convolution
 * \brief The convolution object provides a user-defined convolution kernel.
 *
 * \defgroup group_pyramid Pyramid
 * \brief The pyramid object provides a multi-scale representation of an image.
 *
 * \defgroup group_remap Remap
 * \brief The remap object provides a per-pixel mapping of output pixels to input pixels.
 *
 * \defgroup group_delay Delay
 * \brief The delay object provides temporal storage for OpenVX objects.
 *
 * \defgroup group_log Logging
 * \brief The logging interface for OpenVX.
 *
 * \defgroup group_borders Border Modes
 * \brief The border mode configurations for nodes.
 *
 * \defgroup group_hint Hints
 * \brief Hints are optional indications of preferred behavior.
 *
 * \defgroup group_directive Directives
 * \brief Directives are mandatory requirements of behavior.
 *
 * \defgroup group_node_callback Node Callbacks
 * \brief Callback functions for node completion notification.
 *
 * \defgroup group_user_kernels User Kernels
 * \brief User-defined kernels allow extension of OpenVX.
 *
 * \defgroup group_performance Performance
 * \brief Performance measurement objects.
 *
 * \defgroup group_graph_parameters Graph Parameters
 * \brief Graph parameter interfaces.
 *
 * \defgroup group_adv_array Advanced Array Operations
 * \brief Advanced array manipulation functions.
 *
 * \defgroup group_adv_node Advanced Node Operations
 * \brief Advanced node manipulation functions.
 *
 * \defgroup group_import_kernel Import Kernel
 * \brief Kernel import and module loading functions.
 *
 * \defgroup group_control_flow Control Flow
 * \brief Control flow operations for conditional and iterative graph execution.
 *
 * \defgroup group_vision_function_colorconvert Color Convert
 * \brief Converts the format of an image from one to another.
 *
 * \defgroup group_vision_function_channelextract Channel Extract
 * \brief Extracts a channel from a multi-channel image.
 *
 * \defgroup group_vision_function_channelcombine Channel Combine
 * \brief Combines multiple single-channel images into one multi-channel image.
 *
 * \defgroup group_vision_function_sobel3x3 Sobel 3x3
 * \brief Computes a Sobel filter on an image.
 *
 * \defgroup group_vision_function_magnitude Magnitude
 * \brief Computes the magnitude of pixel-wise gradients.
 *
 * \defgroup group_vision_function_phase Phase
 * \brief Computes the phase of pixel-wise gradients.
 *
 * \defgroup group_vision_function_scale_image Scale Image
 * \brief Scales an image to a new resolution.
 *
 * \defgroup group_vision_function_lut LUT
 * \brief Maps input pixel values to output pixel values via a look-up table.
 *
 * \defgroup group_vision_function_histogram Histogram
 * \brief Computes an image histogram.
 *
 * \defgroup group_vision_function_equalize_hist Equalize Histogram
 * \brief Performs histogram equalization on an image.
 *
 * \defgroup group_vision_function_absdiff Absolute Difference
 * \brief Computes the absolute difference between two images.
 *
 * \defgroup group_vision_function_meanstddev Mean and Standard Deviation
 * \brief Computes the mean and standard deviation of an image.
 *
 * \defgroup group_vision_function_threshold Threshold
 * \brief Applies a threshold to an image.
 *
 * \defgroup group_vision_function_integral_image Integral Image
 * \brief Computes the integral image.
 *
 * \defgroup group_vision_function_erode_image Erode
 * \brief Performs morphological erosion on an image.
 *
 * \defgroup group_vision_function_dilate_image Dilate
 * \brief Performs morphological dilation on an image.
 *
 * \defgroup group_vision_function_median_image Median Filter
 * \brief Computes a median filter on an image.
 *
 * \defgroup group_vision_function_box_image Box Filter
 * \brief Computes a box filter on an image.
 *
 * \defgroup group_vision_function_gaussian_image Gaussian Filter
 * \brief Computes a Gaussian filter on an image.
 *
 * \defgroup group_vision_function_nonlinear_filter Non-Linear Filter
 * \brief Computes a non-linear filter on an image.
 *
 * \defgroup group_vision_function_custom_convolution Custom Convolution
 * \brief Convolves an image with a user-supplied kernel.
 *
 * \defgroup group_vision_function_gaussian_pyramid Gaussian Pyramid
 * \brief Creates a Gaussian image pyramid.
 *
 * \defgroup group_vision_function_laplacian_pyramid Laplacian Pyramid
 * \brief Creates a Laplacian image pyramid.
 *
 * \defgroup group_vision_function_laplacian_reconstruct Laplacian Reconstruct
 * \brief Reconstructs an image from a Laplacian pyramid.
 *
 * \defgroup group_vision_function_weighted_average Weighted Average
 * \brief Computes a weighted average of two images.
 *
 * \defgroup group_vision_function_minmaxloc Min, Max Location
 * \brief Finds the minimum and maximum values and their locations in an image.
 *
 * \defgroup group_vision_function_min Min
 * \brief Computes the pixel-wise minimum of two images.
 *
 * \defgroup group_vision_function_max Max
 * \brief Computes the pixel-wise maximum of two images.
 *
 * \defgroup group_vision_function_and Bitwise AND
 * \brief Computes a bitwise AND between two images.
 *
 * \defgroup group_vision_function_or Bitwise OR
 * \brief Computes a bitwise OR between two images.
 *
 * \defgroup group_vision_function_xor Bitwise XOR
 * \brief Computes a bitwise XOR between two images.
 *
 * \defgroup group_vision_function_not Bitwise NOT
 * \brief Computes a bitwise NOT of an image.
 *
 * \defgroup group_vision_function_mult Pixel-wise Multiplication
 * \brief Computes a pixel-wise multiplication of two images.
 *
 * \defgroup group_vision_function_add Addition
 * \brief Computes a pixel-wise addition of two images.
 *
 * \defgroup group_vision_function_sub Subtraction
 * \brief Computes a pixel-wise subtraction of two images.
 *
 * \defgroup group_vision_function_convertdepth Convert Bit Depth
 * \brief Converts between image bit depths.
 *
 * \defgroup group_vision_function_canny Canny Edge Detector
 * \brief Implements a Canny edge detector.
 *
 * \defgroup group_vision_function_warp_affine Warp Affine
 * \brief Applies an affine transformation to an image.
 *
 * \defgroup group_vision_function_warp_perspective Warp Perspective
 * \brief Applies a perspective transformation to an image.
 *
 * \defgroup group_vision_function_harris Harris Corners
 * \brief Detects corners in an image using the Harris method.
 *
 * \defgroup group_vision_function_fast FAST Corners
 * \brief Detects corners in an image using the FAST method.
 *
 * \defgroup group_vision_function_opticalflowpyrlk Optical Flow Pyramid (LK)
 * \brief Computes the optical flow using the Lucas-Kanade method on a pyramid.
 *
 * \defgroup group_vision_function_remap Remap
 * \brief Maps output image pixels to input image positions.
 *
 * \defgroup group_vision_function_match_template Match Template
 * \brief Compares an image to a template using normalized cross-correlation.
 *
 * \defgroup group_vision_function_lbp LBP
 * \brief Computes the Local Binary Pattern of an image.
 *
 * \defgroup group_vision_function_hog HOG
 * \brief Computes the Histogram of Oriented Gradients of an image.
 *
 * \defgroup group_vision_function_hough_lines_p Hough Lines P
 * \brief Finds lines in a binary image using the probabilistic Hough transform.
 *
 * \defgroup group_vision_function_bilateral_filter Bilateral Filter
 * \brief Applies a bilateral filter to an image.
 *
 * \defgroup group_vision_function_tensor_multiply Tensor Multiply
 * \brief Computes element-wise tensor multiplication.
 *
 * \defgroup group_vision_function_tensor_add Tensor Add
 * \brief Computes element-wise tensor addition.
 *
 * \defgroup group_vision_function_tensor_subtract Tensor Subtract
 * \brief Computes element-wise tensor subtraction.
 *
 * \defgroup group_vision_function_tensor_tablelookup Tensor Table Lookup
 * \brief Performs a table lookup on tensor elements.
 *
 * \defgroup group_vision_function_tensor_transpose Tensor Transpose
 * \brief Transposes the dimensions of a tensor.
 *
 * \defgroup group_vision_function_tensor_convert_depth Tensor Convert Depth
 * \brief Converts between tensor element bit depths.
 *
 * \defgroup group_vision_function_tensor_matrix_multiply Tensor Matrix Multiply
 * \brief Performs a matrix multiplication on tensors.
 *
 * \defgroup group_vision_function_copy Copy
 * \brief Copies data between data objects.
 *
 * \defgroup group_vision_function_nms Non-Maximum Suppression
 * \brief Suppresses non-maximum elements.
 */

/*! \anchor sub_node_parameters
 * \par Node Parameters
 * Scalar objects can be used as node parameters for kernels that require
 * single-value arguments. See the OpenVX specification for details.
 */

/*! \anchor sub_image_access
 * \par Image Access Example
 * See the OpenVX specification for an example of pixel addressing on
 * various image formats including <tt>\ref VX_DF_IMAGE_U1</tt>.
 */

#endif /* OPENVX_DOXYGEN_DEFS_H */
