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

#ifndef OPENVX_IMAGE_RAW_H
#define OPENVX_IMAGE_RAW_H

/*!
 * \file
 * \brief The Khronos Raw Image extension.
 */

/*!
 * \defgroup group_raw_image Raw Image Extension APIs
 * \brief APIs, types, and enums required for supporting RAW images
 * \ingroup group_vx_ext_host
 */

#define OPENVX_KHR_RAW_IMAGE  "vx_khr_raw_image"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif


/*! \brief Raw image attribute enumeration extensions to the <tt>\ref vx_image_attribute_e</tt> enumeration type.
 * \ingroup group_raw_image
 */
/*! \brief Image Attribute which queries if an image was created using <tt>\ref vxCreateRawImage</tt> API. Read-only. Use a <tt>\ref vx_bool</tt> parameter */
#define VX_IMAGE_IS_RAW                     (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0xA)
/*! \brief Image Attribute which queries a raw image for if its exposures are interleaved in memory. Read-only. Use a <tt>\ref vx_image_raw_exposure_interleaving_e</tt> parameter. */
#define VX_IMAGE_RAW_EXPOSURE_INTERLEAVING  (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0xB)
/*! \brief Image Attribute which queries a raw image for its format (see <tt>\ref vx_image_raw_format_t</tt>). Read-only. Use a pointer to a <tt>\ref vx_image_raw_format_t</tt> array. */
#define VX_IMAGE_RAW_FORMAT                 (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0xC)
/*! \brief Image Attribute which queries a raw image for its meta height at top of readout. Read-only. Use a <tt>\ref vx_uint32</tt> parameter. */
#define VX_IMAGE_RAW_META_HEIGHT_BEFORE     (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0xD)
/*! \brief Image Attribute which queries a raw image for its meta height at bottom of readout. Read-only. Use a <tt>\ref vx_uint32</tt> parameter. */
#define VX_IMAGE_RAW_META_HEIGHT_AFTER      (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) + 0xE)

/*! \brief Raw image format enumeration extensions to the <tt>\ref vx_df_image_e</tt> enumeration type.
 * \ingroup group_raw_image
 */
/*! \brief Returned for a raw image when the <tt>\ref vxQueryImage</tt> API is called on the <tt>\ref VX_IMAGE_FORMAT</tt> attribute. */
#define VX_DF_IMAGE_RAW                     (VX_DF_IMAGE('R','A','W','0'))

/*! \brief The enum type enumeration for raw images.
 * \ingroup group_raw_image
 */
#define VX_ENUM_IMAGE_RAW_BUFFER_ACCESS         (vx_enum)0x24   /*!< \brief A <tt>\ref vx_image_raw_buffer_access_e</tt>. */
#define VX_ENUM_IMAGE_RAW_PIXEL_CONTAINER       (vx_enum)0x25   /*!< \brief A <tt>\ref vx_image_raw_pixel_container_e</tt>. */
#define VX_ENUM_IMAGE_RAW_EXPOSURE_INTERLEAVING (vx_enum)0x26   /*!< \brief A <tt>\ref vx_image_raw_exposure_interleaving_e</tt>. */

/*! \brief Maximum number of RAW image exposures that can be contained in an image object formed from calling <tt>\ref vxCreateRawImage</tt>.
 * \ingroup group_raw_image
 */
#define VX_IMAGE_RAW_MAX_EXPOSURES  (3)

/*! \brief The raw image format structure that is given to the <tt>\ref vxCreateRawImage</tt> function.
 * \ingroup group_raw_image
 */
typedef struct _vx_image_raw_format_t {
    vx_uint32 pixel_container;      /*!< \brief Pixel Container, see \ref vx_image_raw_pixel_container_e */
    vx_uint32 msb;                  /*!< \brief Most significant bit in pixel container */
} vx_image_raw_format_t;

/*! \brief The raw image create params structure that is given to the <tt>\ref vxCreateRawImage</tt> function.
 * \ingroup group_raw_image
 */
typedef struct _vx_image_raw_create_params_t {
    vx_uint32 width;                 /*!< \brief The image width in pixels */
    vx_uint32 height;                /*!< \brief The image height in lines (not including meta rows). */
    vx_uint32 num_exposures;         /*!< \brief The number of exposures contained in the sensor readout for a given timestamp.
                                                 Max supported is \ref VX_IMAGE_RAW_MAX_EXPOSURES. */
    vx_uint32 exposure_interleaving; /*!< \brief Indicates the type of exposure interleaving, if any, in memory. see \ref vx_image_raw_exposure_interleaving_e. */
    vx_image_raw_format_t format[VX_IMAGE_RAW_MAX_EXPOSURES]; /*!< \brief Array of vx_image_raw_format_t structures indicating the pixel packing and
        bit alignment format of each exposure.  If line_interleaved == vx_false_e, then the number of
        valid structures in this array should be equal to the value of num_exposures.  If line_interleaved ==
        vx_true_e, then the format should be the same for each exposure in a single buffer, so the
        number of valid structures in this array should equal 1. */
    vx_uint32 meta_height_before;    /*!< \brief Number of lines of meta data at top of sensor readout  (before pixel data)
                                                 (uses the same width as original sensor readout width) */
    vx_uint32 meta_height_after;     /*!< \brief Number of lines of meta data at bottom of sensor readout  (after pixel data)
                                                 (uses the same width as original sensor readout width) */
} vx_image_raw_create_params_t;


/*! \brief The raw image buffer access enum.
 * \ingroup group_raw_image
 */
enum vx_image_raw_buffer_access_e {
    /*! \brief For accessing pointer to full allocated buffer (pixel buffer + meta buffer). */
    VX_IMAGE_RAW_ALLOC_BUFFER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_BUFFER_ACCESS) + 0x0,
    /*! \brief For accessing pointer to pixel buffer only */
    VX_IMAGE_RAW_PIXEL_BUFFER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_BUFFER_ACCESS) + 0x1,
    /*! \brief For accessing pointer to meta buffer only */
    VX_IMAGE_RAW_META_BEFORE_BUFFER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_BUFFER_ACCESS) + 0x2,
    /*! \brief For accessing pointer to meta buffer only */
    VX_IMAGE_RAW_META_AFTER_BUFFER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_BUFFER_ACCESS) + 0x3
};

/*! \brief The raw image pixel container enum.
 * \ingroup group_raw_image
 */
enum vx_image_raw_pixel_container_e {
    /*! \brief Two bytes per pixel in memory. */
    VX_IMAGE_RAW_16_BIT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_PIXEL_CONTAINER) + 0x0,
    /*! \brief One byte per pixel in memory. */
    VX_IMAGE_RAW_8_BIT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_PIXEL_CONTAINER) + 0x1,
    /*! \brief Packed 12 bit mode; Three bytes per two pixels in memory. */
    VX_IMAGE_RAW_P12_BIT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_PIXEL_CONTAINER) + 0x2
};

/*! \brief The raw image exposure interleaving enum.
 * \ingroup group_raw_image
 */
enum vx_image_raw_exposure_interleaving_e {
    /*! \brief Each exposure is readout and stored in separate planes; single exposure per CSI virtual channel. */
    VX_IMAGE_RAW_PLANAR = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_EXPOSURE_INTERLEAVING) + 0x0,
    /*! \brief Each exposure is readout and stored in line interleaved fashion; multiple exposures share same CSI virtual channel. */
    VX_IMAGE_RAW_LINE_INTERLEAVED = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_EXPOSURE_INTERLEAVING) + 0x1,
    /*! \brief Each exposure is readout and stored in pixel interleaved fashion; multiple exposures share same CSI virtual channel. */
    VX_IMAGE_RAW_PIXEL_INTERLEAVED = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_IMAGE_RAW_EXPOSURE_INTERLEAVING) + 0x2
};


/*! \brief Creates an opaque reference to a raw sensor image (including multi-exposure and metadata).
 * \details Not guaranteed to exist until the <tt>\ref vx_graph</tt> containing it has been verified.
 *
 * \param [in] context         The reference to the implementation context.
 * \param [in] params          The pointer to a \ref vx_image_raw_create_params_t structure
 * \param [in] size            The size of the structure pointed to by the params pointer, in bytes.
 *
 * \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \see vxMapImagePatch to obtain direct memory access to the image data.
 *
 * \ingroup group_raw_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateRawImage(vx_context context,
                                                   const vx_image_raw_create_params_t *params,
                                                   vx_size size);

/*! \brief Creates an opaque reference to a virtual raw sensor image no direct user access (including multi-exposure and metadata).
 * \details Not guaranteed to exist until the <tt>\ref vx_graph</tt> containing it has been verified.
 *
 * Virtual Raw Image Objects are useful when the Raw Image is used as internal graph edge.
 * Virtual Raw Image Objects are scoped within the parent graph only.
 *
 * \param [in] graph           The reference to the parent graph.
 * \param [in] params          The pointer to a \ref vx_image_raw_create_params_t structure
 * \param [in] size            The size of the structure pointed to by the params pointer, in bytes.
 *
 * \returns A raw image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \see vxMapImagePatch to obtain direct memory access to the image data.
 *
 * \ingroup group_raw_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateVirtualRawImage(vx_graph graph,
                                                          const vx_image_raw_create_params_t *params,
                                                          vx_size size);

/*! \brief Allows the application to copy a rectangular patch from/into an image object plane.
 * \param [in] image The reference to the image object that is the source or the
 * destination of the copy.
 * \param [in] image_rect The coordinates of the image patch. The patch must be within
 * the bounds of the image. (start_x, start_y) gives the coordinates of the topleft
 * pixel inside the patch, while (end_x, end_y) gives the coordinates of the bottomright
 * element out of the patch. Must be 0 <= start < end <= number of pixels in the image dimension.
 * \param [in] image_plane_index The plane index of the image object that is the source or the
 * destination of the patch copy.
 * \param [in] user_addr The address of a structure describing the layout of the
 * user memory location pointed by user_ptr. In the structure, only dim_x, dim_y,
 * stride_x and stride_y fields must be provided, other fields are ignored by the function.
 * The layout of the user memory must follow a row major order:
 * stride_x >= pixel size in bytes, and stride_y >= stride_x * dim_x.
 * \param [in] user_ptr The address of the memory location where to store the requested data
 * if the copy was requested in read mode, or from where to get the data to store into the image
 * object if the copy was requested in write mode. The accessible memory must be large enough
 * to contain the specified patch with the specified layout:
 * accessible memory in bytes >= (end_y - start_y) * stride_y.
 * \param [in] usage This declares the effect of the copy with regard to the image object
 * using the <tt>\ref vx_accessor_e</tt> enumeration. For uniform images, only VX_READ_ONLY
 * is supported. For other images, Only <tt>\ref VX_READ_ONLY</tt> and <tt>\ref VX_WRITE_ONLY</tt> are supported:
 * \param [in] flags An integer that allows passing options to the copy operation.
 * \arg <tt>\ref VX_READ_ONLY</tt> means that data is copied from the image object into the application memory
 * \arg <tt>\ref VX_WRITE_ONLY</tt> means that data is copied into the image object from the application memory
 * \param [in] user_mem_type A <tt>\ref vx_memory_type_e</tt> enumeration that specifies
 * the memory type of the memory referenced by the user_addr.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_OPTIMIZED_AWAY This is a reference to a virtual image that cannot be
 * accessed by the application.
 * \retval VX_ERROR_INVALID_REFERENCE image is not a valid <tt>\ref vx_image</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS An other parameter is incorrect.
 * \note The application may ask for data outside the bounds of the valid region, but
 * such data has an undefined value.
 * \note Some special restrictions apply to <tt>\ref VX_DF_IMAGE_U1</tt> images.
 * \ingroup group_image
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyImagePatchWithFlags(vx_image image,
                                                             const vx_rectangle_t *image_rect,
                                                             vx_uint32 image_plane_index,
                                                             const vx_imagepatch_addressing_t *user_addr,
                                                             void * user_ptr,
                                                             vx_enum usage,
                                                             vx_enum user_mem_type,
                                                             vx_uint32 flags);


#ifdef  __cplusplus
}
#endif

#endif
