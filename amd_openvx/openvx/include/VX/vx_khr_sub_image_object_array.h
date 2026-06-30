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

#ifndef OPENVX_SUB_IMAGE_OBJECT_ARRAY_H
#define OPENVX_SUB_IMAGE_OBJECT_ARRAY_H

/*!
 * \file
 * \brief The OpenVX Sub-image object arrays extension.
 */

#define OPENVX_KHR_SUB_IMAGE_OBJECT_ARRAY  "vx_sub_image_object_array"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*!
 * \brief Creates an object array or virtual object array of images from another, given a rectangle.
 *
 * The original object array may be virtual or non-virtual.
 *
 * \param [in] image_array     the reference to the parent object array
 * \param [in] rect            the region of interest rectangle.
 *                      Must contain points within the parent image pixel space.
 *
 * The input object array image_array must be a non-virtual or virtual object array of images with width,
 * height and format all defined.
 * The rectangle rect must be defined within the pixel space of the parent images.
 * For VX_DF_IMAGE_U1-type images there are some restrictions for the rectangle that can be used to
 * create a images using vxCreateObjectArrayFromROI. Namely, the rectangle needs to have its left edge
 * aligned to a byte boundary in the parent image, i.e., _start_x in the vx_rectangle_t must be a multiple
 * of 8 (including 0). This is because images of type VX_DF_IMAGE_U1 must start on a byte boundary and
 * sub-images created by vxCreateObjectArrayFromROI points to data in the original images.
 *
 * \return Returns a reference to the array of sub-images; any possible errors preventing a successful
 *         creation may be checked using vxGetStatus().
 * Possible causes of errors are:
 *  -   Invalid input reference
 *  -   Input reference is not to an object array
 *  -   Input object array does not contain images
 *  -   Input images do not have their width, height or format specified
 *  -   Input object array images bounds are not defined, do not contain the region described by rect
 *      or are not supported for the image format
 *  -   rect is NULL
 *  -   Out of resources
 * The new reference refers to data in the original array, so that updates to the new images update the
 * parent images, and updates to the parent images in the region of interest update the sub-images.
 * If the input object array is virtual, the new reference returned is to a virtual object array of
 * sub-images.
 * If the input object array is non-virtual, the new reference returned is to a non-virtual object
 * array of sub-images.
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateObjectArrayFromROI(vx_object_array image_array, const vx_rectangle_t* rect);

/*!
 * \brief Creates an object array or virtual object array of images from a single plane channel of another.
 *
 * The original object array may be virtual or non-virtual.
 *
 * \param [in] image_array     the reference to the parent object array
 * \param [in] channel         the vx_channel to use.
 *
 * The input object array *image_array* must be a non-virtual or virtual object array of images with
 * defined format that is one of the multi-planar formats YUV4, IYUV, NV12, NV21 or any other vendor
 * supported multi-planar format
 * The function supports only channels that occupy an entire plane of the multi-planar images in image_array.
 * Other cases are not supported. The following are legal:
 *  VX_CHANNEL_Y from YUV4, IYUV, NV12 or NV21
 *  VX_CHANNEL_U from YUV4 or IYUV
 *  VX_CHANNEL_V from YUV4, IYUV
 *  Any valid channel comprising the entire plane of any other vendor supported multi-planar format
 *
 * \return Returns a reference to the array of sub-images; any possible errors preventing a successful
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
 * The new reference refers to data in the original array, so that updates to the new images update
 * the parent images, and updates to the specified channel of the parent images update the sub-images.
 *
 * If the input object array is virtual, the new reference returned is to a virtual object array of
 * sub-images.
 * If the input object array is non-virtual, the new reference returned is to a non-virtual object
 * array of sub-images.
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateObjectArrayFromChannel(vx_object_array image_array, vx_enum channel);

#ifdef  __cplusplus
}
#endif

#endif
