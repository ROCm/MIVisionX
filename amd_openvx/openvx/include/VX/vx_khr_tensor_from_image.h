/*
 * Copyright (c) 2023-2026 The Khronos Group Inc.
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

#ifndef OPENVX_TENSOR_FROM_IMAGE_H
#define OPENVX_TENSOR_FROM_IMAGE_H

/*!
 * \file
 * \brief The OpenVX Tensor From Image extension.
 */

#define OPENVX_KHR_TENSOR_FROM_IMAGE  "vx_khr_tensor_from_image"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*!
 * \brief Creates tensor or virtual tensor from an image, given a rectangle.
 *
 * The original image may be virtual or non-virtual.
 *
 * \param [in] image                the reference to the parent image
 * \param [in] rect                 the region of interest rectangle.
 *                                  Must contain points within the parent image pixel space.
 * \param [in] fixed_point_position Specifies the fixed point position. 
 *                                  If 0, calculations are performed using integer arithmetic.
 *
 * The input image must be a non-virtual or virtual image with width, height and format all defined.
 * The rectangle rect must be defined within the pixel space of the parent image, or the pointer can
 * be NULL, in which case the entire image is assumed as the ROI.
 * For VX_DF_IMAGE_U1-type images there are some restrictions for the rectangle that can be used to
 * create a tensor using vxCreateTensorFromROI. Namely, the rectangle needs to have its left edge
 * aligned to a byte boundary in the parent image, i.e., _start_x in the vx_rectangle_t must be a multiple
 * of 8 (including 0). This is because images of type VX_DF_IMAGE_U1 must start on a byte boundary and
 * a tensor created by vxCreateTensorFromROI points to data in the original image.
 *
 * \return Returns a reference to the tensor; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  -   Invalid input reference
 *  -   Input reference is not to an image
 *  -   Input image does not have its width, height or format specified
 *  -   Input image does not contain the region described by rect or are not supported for the image format
 *  -   Out of resources
 * The new reference refers to data in the original image, so that updates to the tensor update the
 * parent image, and updates to the parent image in the region of interest update the tensor.
 * If the input image is virtual, the new reference returned is to a virtual tensor.
 * If the input image is non-virtual, the new reference returned is to a non-virtual tensor.
  */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromROI(vx_image image, const vx_rectangle_t* rect, vx_int8 fixed_point_position);

/*!
 * \brief Creates a tensor or virtual tensor from a single plane channel of an image.
 *
 * The original image may be virtual or non-virtual.
 *
 * \param [in] image                the reference to the parent image
 * \param [in] channel              the vx_channel to use.
 * \param [in] fixed_point_position Specifies the fixed point position. 
 *                                  If 0, calculations are performed using integer arithmetic.
 *
 * The input *image* must be a non-virtual or virtual image with
 * defined format that is one of the multi-planar formats YUV4, IYUV, NV12, NV21 or any other vendor
 * supported multi-planar format
 * The function supports only channels that occupy an entire plane of the multi-planar image.
 * Other cases are not supported. The following are legal:
 *  VX_CHANNEL_Y from YUV4, IYUV, NV12 or NV21
 *  VX_CHANNEL_U from YUV4 or IYUV
 *  VX_CHANNEL_V from YUV4, IYUV
 *  Any valid channel comprising the entire plane of any other vendor supported multi-planar format:
 *    we add RB_VX_CHANNEL_UV or VX_CHANNEL_1 from NV21 or NV12
 *
 * \return Returns a reference to the tensor; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  Invalid input reference
 *  Input reference is not to an image
 *  Input image does not have their format specified
 *  Input image is not in a supported multi-planar format
 *  channel is not a valid channel comprising the entire plane of the input format
 *  Out of resources
 * 
 * The new reference refers to data in the original image, so that updates to the tensor update
 * the parent image, and updates to the specified channel of the parent image update the tensor.
 *
 * If the input image is virtual, the new reference returned is to a virtual tensor.
 * If the input image is non-virtual, the new reference returned is to a non-virtual tensor.
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromChannel(vx_image image, vx_enum channel, vx_int8 fixed_point_position);

/*!
 * \brief Creates an object array or virtual object array of tensors from an object array of images, given a rectangle.
 *
 * The original object array may be virtual or non-virtual.
 *
 * \param [in] image_array          the reference to the parent object array
 * \param [in] rect                 the region of interest rectangle.
 *                                  Must contain points within the parent image pixel space.
 * \param [in] fixed_point_position Specifies the fixed point position. 
 *                                  If 0, calculations are performed using integer arithmetic.
 *
 * The input object array image_array must be a non-virtual or virtual object array of images with width,
 * height and format all defined.
 * The rectangle rect must be defined within the pixel space of the parent images, or the pointer can
 * be NULL, in which case the entire image is assumed as the ROI.
 * For VX_DF_IMAGE_U1-type images there are some restrictions for the rectangle that can be used to
 * create a images using vxCreateTensorObjectArrayFromROI. Namely, the rectangle needs to have its left edge
 * aligned to a byte boundary in the parent image, i.e., _start_x in the vx_rectangle_t must be a multiple
 * of 8 (including 0). This is because images of type VX_DF_IMAGE_U1 must start on a byte boundary and
 * sub-images created by vxCreateObjectArrayFromROI points to data in the original images.
 *
 * \return Returns a reference to the array of tensors; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  -   Invalid input reference
 *  -   Input reference is not to an object array
 *  -   Input object array does not contain images
 *  -   Input images do not have their width, height or format specified
 *  -   Input object array images bounds are not defined, do not contain the region described by rect
 *      or are not supported for the image format
 *  -   Out of resources
 * The new reference refers to data in the original array, so that updates to the tensors update the
 * parent images, and updates to the parent images in the region of interest update the tensors.
 * If the input object array is virtual, the new reference returned is to a virtual object array of
 * tensors.
 * If the input object array is non-virtual, the new reference returned is to a non-virtual object
 * array of tensors.
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateTensorObjectArrayFromROI(vx_object_array image_array, const vx_rectangle_t* rect, vx_int8 fixed_point_position);

/*!
 * \brief Creates an object array or virtual object array of tensors from a single plane channel of an object array of images.
 *
 * The original object array may be virtual or non-virtual.
 *
 * \param [in] image_array          the reference to the parent object array
 * \param [in] channel              the vx_channel to use.
 * \param [in] fixed_point_position Specifies the fixed point position. 
 *                                  If 0, calculations are performed using integer arithmetic.
 *
 * The input object array *image_array* must be a non-virtual or virtual object array of images with
 * defined format that is one of the multi-planar formats YUV4, IYUV, NV12, NV21 or any other vendor
 * supported multi-planar format
 * The function supports only channels that occupy an entire plane of the multi-planar images in image_array.
 * Other cases are not supported. The following are legal:
 *  VX_CHANNEL_Y from YUV4, IYUV, NV12 or NV21
 *  VX_CHANNEL_U from YUV4 or IYUV
 *  VX_CHANNEL_V from YUV4 or IYUV
 *  Any valid channel comprising the entire plane of any other vendor supported multi-planar format:
 *    we add RB_VX_CHANNEL_UV or VX_CHANNEL_1 from NV21 or NV12
 *
 * \return Returns a reference to the array of tensors; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  Invalid input reference
 *  Input reference is not to an object array
 *  Input object array does not contain images
 *  Input images do not have their format specified
 *  Input images are not in a supported multi-planar format
 *  channel is not a valid channel comprising the entire plane of the input format
 *  Out of resources
 * 
 * The new reference refers to data in the original array, so that updates to the tensors update
 * the parent images, and updates to the specified channel of the parent images update the tensors.
 *
 * If the input object array is virtual, the new reference returned is to a virtual object array of
 * tensors.
 * If the input object array is non-virtual, the new reference returned is to a non-virtual object
 * array of tensors.
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateTensorObjectArrayFromChannel(vx_object_array image_array, vx_enum channel, vx_int8 fixed_point_position);

/*!
 * \brief Creates an object array or virtual object array of tensors from an object array of tensors, given arrays of bounds.
 *
 * The original object array may be virtual or non-virtual.
 *
 * \param [in] tensor_array          the reference to the parent object array
 * \param [in] number_of_dims       Number of dimensions in the view. Error return if 0 or greater than number of tensor dimensions.
 *                                  If smaller than number of tensor dimensions, the lower dimensions are assumed.
 * \param [in] view_start           View start coordinates.
 * \param [in] view_end             View end coordinates.

 * The input object array image_array must be a non-virtual or virtual object array of images with type
 * and dimensions all defined.
 * The input pointers view_start and view_end must either be both NULL or point to arrays of length number_of_dims defining start
 * and end points within the parent tensor; if the pointers are NULL then the bounds will be set equal to zero and the size of each
 * dimension. The end point must never be less than the start point; the size of each dimension is given by (end point) - (start point).
 *
 * \return Returns a reference to the array of tensors; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  -   Invalid input reference
 *  -   Input reference is not to an object array
 *  -   Input object array does not contain tensors
 *  -   Input tensors do not have their type or dimensions specified
 *  -   Input tensors do not contain the region described by view_start and view_end or end points are less then start points
 *  -   Out of resources
 * The new reference refers to data in the original array, so that updates to the tensors update the
 * parent tensors, and updates to the parent tensors in the region of interest update the new tensors.
 * If the input object array is virtual, the new reference returned is to a virtual object array of
 * tensors.
 * If the input object array is non-virtual, the new reference returned is to a non-virtual object
 * array of tensors.
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateTensorObjectArrayFromView(vx_object_array tensor_array, vx_size number_of_dims, const vx_size* view_start, const vx_size* view_end);

#ifdef  __cplusplus
}
#endif

#endif
