/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include "test.h"
#include "test_bmp.h"

#include <VX/vx.h>

// #define DEBUG_CT_IMAGE

uint32_t ct_image_bits_per_pixel(vx_df_image format)
{
    switch(format)
    {
        case VX_DF_IMAGE_U8:
            return 8 * 1;
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
            return 8 * 2;
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case VX_DF_IMAGE_RGBX:
            return 8 * 4;
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_YUV4:
            return 8 * 3;
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            return 8 * 3 / 2;
        default:
            CT_RecordFailure();
            return 0;
    };
}

int ct_channels(vx_df_image format)
{
    switch(format)
    {
        case VX_DF_IMAGE_U8:
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
            return 1;
        case VX_DF_IMAGE_RGBX:
            return 4;
        case VX_DF_IMAGE_RGB:
            return 3;
        default:
            break;
    };
    return 1;
}

uint32_t ct_stride_bytes(CT_Image image)
{
    uint32_t factor = 0;
    switch(image->format)
    {
        case VX_DF_IMAGE_U8:
        case VX_DF_IMAGE_NV21:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_YUV4:
        case VX_DF_IMAGE_IYUV:
            factor = 1;
            break;
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            factor = 2;
            break;
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case VX_DF_IMAGE_RGBX:
            factor = 4;
            break;
        case VX_DF_IMAGE_RGB:
            factor = 3;
            break;
        default:
            // error should be already reported by image allocation function
            break;
    };
    return image->stride*factor;
}


static size_t ct_image_data_size(uint32_t width, uint32_t height, vx_df_image format)
{
    return (size_t)width * height * ct_image_bits_per_pixel(format) / 8;
}

#ifdef DEBUG_CT_IMAGE
static void ct_print_image(CT_Image img)
{
    printf("CT_Image 0x%p:\n", img);
    if (img)
    {
        printf("\t width       = %d\n", img->width);
        printf("\t height      = %d\n", img->height);
        printf("\t stride      = %d\n", img->stride);
        printf("\t format      = DF_IMAGE(%.*s)\n", sizeof(img->format), &img->format);
        printf("\t roi.x       = %d\n", img->roi.x);
        printf("\t roi.y       = %d\n", img->roi.y);
        printf("\t roi.width   = %d\n", img->roi.width);
        printf("\t roi.height  = %d\n", img->roi.height);
        printf("\t data        = 0x%p\n", img->data.y);
        printf("\t data_begin_ = 0x%p\n", img->data_begin_);
        printf("\t refcount_   = 0x%p (%u)\n", img->refcount_, img->refcount_ ? *img->refcount_ : 0);
    }
    printf("\n");
    fflush(stdout);
}
#endif

static void ct_image_addref(CT_Image img)
{
    if (img->refcount_)
        *img->refcount_ += 1;
}

static uint32_t ct_image_removeref(CT_Image img)
{
    if (img->refcount_)
        return --*img->refcount_;
    else
        return (uint32_t)(-1);
}

static void ct_release_image(CT_Image* img)
{
    if (!img || !*img) return;

#ifdef DEBUG_CT_IMAGE
    printf("RELEASING: "); ct_print_image(*img);
#endif

    if ((*img)->data.u32 && (*img)->refcount_) // if refcount_ is NULL then data is not ours
    {
        if (ct_image_removeref(*img) == 0)
        {
            free((*img)->data_begin_);
        }
        (*img)->data.u32 = 0;
        (*img)->data_begin_ = 0;
        (*img)->refcount_ = 0;
    }
    free(*img);
    *img = 0;
}

static CT_Image ct_allocate_image_hdr_impl(uint32_t width, uint32_t height, uint32_t step, vx_df_image format, int allocate_data)
{
    CT_Image image;

    CT_ASSERT_(return 0, step >= width);

    image = (CT_Image)malloc(sizeof(*image));
    CT_ASSERT_(return 0, image); // out of memory

    image->data.u32    = 0;
    image->data_begin_ = 0;
    image->refcount_   = 0;
    image->width  = width;
    image->height = height;
    image->stride = step;
    image->format = format;
    image->roi.x = 0;
    image->roi.y = 0;
    image->roi.width = width;
    image->roi.height = height;

    CT_RegisterForGarbageCollection(image, (CT_ObjectDestructor)ct_release_image, CT_GC_IMAGE);

    if (allocate_data)
    {
        uint8_t* bytes;
        size_t memory_size = ct_image_data_size(width, height, format) + sizeof(uint32_t) * 2;

        bytes = (uint8_t*)malloc(memory_size);
        CT_ASSERT_(return 0, bytes); // out of memory
        memset(bytes, 153, memory_size); // fill with some "magic" value
        image->data_begin_ = bytes;
        image->refcount_ = (uint32_t*)( (((size_t)bytes) + sizeof(uint32_t) - 1) & ~((size_t)(sizeof(uint32_t) - 1)) ); // align ptr to 4 bytes
        *image->refcount_ = 1;
        image->data.u32 = image->refcount_ + 1;
    }

    return image;
}

CT_Image ct_allocate_image(uint32_t width, uint32_t height, vx_df_image format)
{
    CT_Image image;

    switch(format)
    {
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            CT_ASSERT_(return 0, height % 2 == 0); // height must be even
            break;
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
            CT_ASSERT_(return 0, width % 2 == 0); // width must be even
            break;
    };

    image = ct_allocate_image_hdr_impl(width, height, width, format, 1);

#ifdef DEBUG_CT_IMAGE
    printf("ALLOCATED: "); ct_print_image(image);
#endif

    return image;
}

CT_Image ct_get_image_roi(CT_Image img, CT_Rect roi)
{
    uint32_t bpp;
    CT_Image image;

    CT_ASSERT_(return 0, img);
    CT_ASSERT_(return 0, img->data.u32);

    CT_ASSERT_(return 0, roi.x + roi.width  <= img->width);
    CT_ASSERT_(return 0, roi.y + roi.height <= img->height);

    switch(img->format) // not sure if roi for multi-plane formats is needed at all
    {
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
            CT_ASSERT_(return 0, roi.width % 2 == 0 && roi.x % 2 == 0); // width must be even
            break;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
        case VX_DF_IMAGE_YUV4:
        case VX_DF_IMAGE_IYUV:
            // do not support roi on multi-plane formats
            CT_ASSERT_(return 0, 0);
            break;
    };

    bpp = ct_image_bits_per_pixel(img->format);
    image = ct_allocate_image_hdr_impl(roi.width, roi.height, img->stride, img->format, 0);

    if (image)
    {
        image->roi.x = img->roi.x + roi.x;
        image->roi.y = img->roi.y + roi.y;
        image->roi.width  = img->roi.width;
        image->roi.height = img->roi.height;

        ct_image_addref(img);
        image->data_begin_ = img->data_begin_;
        image->refcount_   = img->refcount_;
        image->data.y      = img->data.y + (img->stride * roi.y + roi.x) * bpp / 8;
    }

#ifdef DEBUG_CT_IMAGE
    printf("ALLOCATED ROI: "); ct_print_image(image);
#endif

    return image;
}

CT_Image ct_allocate_image_hdr(uint32_t width, uint32_t height,  uint32_t stride, vx_df_image format, void* data)
{
    CT_Image image = ct_allocate_image_hdr_impl(width, height, stride, format, 0);

    if (image) image->data.y = (uint8_t*)data;

#ifdef DEBUG_CT_IMAGE
    printf("ALLOCATED HDR: "); ct_print_image(image);
#endif

    return image;
}

CT_Image ct_get_image_roi_(CT_Image img, uint32_t x_start, uint32_t y_start, uint32_t width, uint32_t height)
{
    CT_Rect roi = { x_start, y_start, width, height };
    return ct_get_image_roi(img, roi);
}

void ct_adjust_roi(CT_Image img, int left, int top, int right, int bottom)
{
    int new_x, new_y, new_width, new_height;
    ASSERT(img);

    // not very accurate with overflows, but should work for normal images
    new_x = img->roi.x + left;
    new_y = img->roi.y + top;
    new_width  = img->width  - left - right;
    new_height = img->height - top  - bottom;

    if (new_x < 0 || new_y < 0 || (uint32_t)(new_x + new_width) > img->roi.width || (uint32_t)(new_y + new_height) > img->roi.height)
    {
        FAIL("Invalid ROI adjustment");
    }
    else
    {
        img->data.y = img->data.y + (img->stride * top + left) * ct_image_bits_per_pixel(img->format) / 8;
        img->roi.x  = (uint32_t)new_x;
        img->roi.y  = (uint32_t)new_y;
        img->width  = (uint32_t)new_width;
        img->height = (uint32_t)new_height;
    }
}

CT_Image ct_read_image(const char* fileName, int dcn)
{
    FILE* f = 0;
    size_t sz;
    char* buf = 0;
    CT_Image image = 0;

    if (!fileName)
    {
        CT_ADD_FAILURE("Image file name not specified\n");
        return 0;
    }

    f = fopen(fileName, "rb");
    if (!f)
    {
        CT_ADD_FAILURE("Can't open image file: %s\n", fileName);
        return 0;
    }

    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    if( sz > 0 )
    {
        buf = (char*)malloc(sz);
        fseek(f, 0, SEEK_SET);
        if( fread(buf, 1, sz, f) == sz )
        {
            image = ct_read_bmp((unsigned char*)buf, (int)sz, dcn);
        }
    }

    free(buf);
    fclose(f);

    if(!image)
        CT_ADD_FAILURE("Can not read image from \"%s\"", fileName);

    return image;
}

void ct_write_image(const char* fileName, CT_Image image)
{
    char* dotpos;
    int result = -1;
    if (fileName)
    {
        dotpos = strrchr(fileName, '.');
        if(dotpos &&
           (strcmp(dotpos, ".bmp") == 0 ||
            strcmp(dotpos, ".BMP") == 0))
            result = ct_write_bmp(fileName, image);
        if( result < 0 )
            CT_ADD_FAILURE("Can not write image to \"%s\"", fileName);
    }
    else
    {
        CT_ADD_FAILURE("Image name is not specified (NULL)");
    }
}

static void ct_copy_from_vx_plane(uint8_t* vxsrc, uint8_t* ctdst, int element_sz, int stride, vx_imagepatch_addressing_t addr, int32_t x0, int32_t y0)
{
    int32_t x, y, i;
    uint32_t vx, vy;

    for (vy = 0, y = y0; vy < addr.dim_y; vy += addr.step_y, y++)
    {
        int j = addr.stride_y * vy * addr.scale_y / VX_SCALE_UNITY;
        for (vx = 0, x = x0; vx < addr.dim_x; vx += addr.step_x, x++)
        {
            uint8_t* vxdata = vxsrc + j + addr.stride_x * vx * addr.scale_x / VX_SCALE_UNITY;
            uint8_t* ctdata = ctdst + y * stride + x * element_sz;

            for (i = 0; i < element_sz; ++i)
                ctdata[i] = vxdata[i];
        }
    }

}

static void ct_copy_to_vx_plane(uint8_t* ct_ptr, uint8_t* vx_ptr, int element_sz, int stride, vx_imagepatch_addressing_t addr)
{
    int32_t x, y, i;
    uint32_t vx, vy;

    for (vy = 0, y = 0; vy < addr.dim_y; vy += addr.step_y, y++)
    {
        int j = addr.stride_y * vy * addr.scale_y / VX_SCALE_UNITY;
        for (vx = 0, x = 0; vx < addr.dim_x; vx += addr.step_x, x++)
        {
            uint8_t* vxdata = vx_ptr + j + addr.stride_x * vx * addr.scale_x / VX_SCALE_UNITY;
            uint8_t* ctdata = ct_ptr + y * stride + x * element_sz;

            for (i = 0; i < element_sz; ++i)
                vxdata[i] = ctdata[i];
        }
    }

}

CT_Image ct_image_from_vx_image_impl(vx_image vximg, const char* func, const char* file, int line)
{
    CT_Image ctimg;
    vx_uint32 width, height;
    vx_df_image format;

    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_WIDTH,   &width,   sizeof(width)),  func, file, line);
    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_HEIGHT,  &height,  sizeof(height)), func, file, line);
    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_FORMAT,  &format,  sizeof(format)), func, file, line);

    ctimg = ct_allocate_image(width, height, format);

    if (ctimg)
    {
        vx_rectangle_t rect = {0, 0, width, height};
        uint32_t stride_bytes = ct_stride_bytes(ctimg);
        if (!stride_bytes) return 0; // something goes wrong

        // BUG 13647
        // ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxGetValidRegionImage(vximg, &rect), func, file, line);

        { // copy plane #0
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 0, &addr, &pdata, VX_READ_ONLY), func, file, line);
            ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y, stride_bytes/ctimg->stride, stride_bytes, addr, rect.start_x, rect.start_y);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 0, &addr, pdata), func, file, line);
        }

        if (format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21)
        {
            // copy plane #1
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_READ_ONLY), func, file, line);
            ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y + ctimg->stride * height, 2, stride_bytes, addr, rect.start_x/2, rect.start_y/2);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 1, &addr, pdata), func, file, line);
        }
        else if (format == VX_DF_IMAGE_YUV4)
        {
            // there are 2 more full-size planes

            { // copy plane #1
                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
                void *pdata = 0;
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_READ_ONLY), func, file, line);
                ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y + ctimg->stride * height, 1, stride_bytes, addr, rect.start_x, rect.start_y);
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 1, &addr, pdata), func, file, line);
            }

            {// copy plane #2
                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
                void *pdata = 0;
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 2, &addr, &pdata, VX_READ_ONLY), func, file, line);
                ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y + ctimg->stride * height * 2, 1, stride_bytes, addr, rect.start_x, rect.start_y);
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 2, &addr, pdata), func, file, line);
            }
        }
        else if (format == VX_DF_IMAGE_IYUV)
        {
            // there are 2 more subsampled planes

            { // copy plane #1
                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
                void *pdata = 0;
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_READ_ONLY), func, file, line);
                ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y + ctimg->stride * height, 1, stride_bytes/2, addr, rect.start_x/2, rect.start_y/2);
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 1, &addr, pdata), func, file, line);
            }

            { // copy plane #2
                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
                void *pdata = 0;
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 2, &addr, &pdata, VX_READ_ONLY), func, file, line);
                ct_copy_from_vx_plane((uint8_t*)pdata, ctimg->data.y + ctimg->stride * height + ctimg->stride * height/2/2, 1, stride_bytes/2, addr, rect.start_x/2, rect.start_y/2);
                ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, 0, 2, &addr, pdata), func, file, line);
            }
        }
    }

    return ctimg;
}

vx_image ct_image_to_vx_image_impl(CT_Image ctimg, vx_context context, const char* func, const char* file, int line)
{
    int width = ctimg->width, height = ctimg->height;
    int format = ctimg->format;
    vx_image vximg = NULL;

    vximg = vxCreateImage(context, width, height, format);
    ASSERT_VX_OBJECT_AT_(return 0, vximg, VX_TYPE_IMAGE, func, file, line);

    return ct_image_copyto_vx_image_impl(vximg, ctimg, func, file, line);
}


vx_image ct_image_copyto_vx_image_impl(vx_image vximg, CT_Image ctimg, const char* func, const char* file, int line)
{
    vx_context context = vxGetContext((vx_reference)vximg);
    vx_uint32 ct_width = ctimg->width, ct_height = ctimg->height;
    vx_df_image ct_format = ctimg->format;
    vx_uint32 width, height;
    vx_df_image format;

    vx_rectangle_t rect = {0, 0, ct_width, ct_height};
    uint32_t stride_bytes = ct_stride_bytes(ctimg);
    ASSERT_AT_(return 0, stride_bytes > 0, func, file, line); // something goes wrong

    ASSERT_VX_OBJECT_AT_(return 0, vximg, VX_TYPE_IMAGE, func, file, line);
    ASSERT_VX_OBJECT_AT_(return 0, context, VX_TYPE_CONTEXT, func, file, line);

    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_WIDTH,   &width,   sizeof(width)),  func, file, line);
    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_HEIGHT,  &height,  sizeof(height)), func, file, line);
    ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxQueryImage(vximg, VX_IMAGE_ATTRIBUTE_FORMAT,  &format,  sizeof(format)), func, file, line);

    ASSERT_AT_(return 0, ct_width == width, func, file, line);
    ASSERT_AT_(return 0, ct_height == height, func, file, line);
    ASSERT_AT_(return 0, ct_format == format, func, file, line);

    { // copy plane #0
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        void *pdata = 0;
        ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 0, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
        ct_copy_to_vx_plane(ctimg->data.y, (uint8_t*)pdata, stride_bytes/ctimg->stride, stride_bytes, addr);
        ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 0, &addr, pdata), func, file, line);
    }

    if (format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21)
    {
        // copy plane #1
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        void *pdata = 0;
        ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
        ct_copy_to_vx_plane(ctimg->data.y + ctimg->stride * height, (uint8_t*)pdata, 2, stride_bytes, addr);
        ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 1, &addr, pdata), func, file, line);
    }
    else if (format == VX_DF_IMAGE_YUV4)
    {
        // there are 2 more full-size planes

        { // copy plane #1
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
            ct_copy_to_vx_plane(ctimg->data.y + ctimg->stride * height, (uint8_t*)pdata, 1, stride_bytes, addr);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 1, &addr, pdata), func, file, line);
        }

        {// copy plane #2
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 2, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
            ct_copy_to_vx_plane(ctimg->data.y + ctimg->stride * height * 2, (uint8_t*)pdata, 1, stride_bytes, addr);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 2, &addr, pdata), func, file, line);
        }
    }
    else if (format == VX_DF_IMAGE_IYUV)
    {
        // there are 2 more subsampled planes

        { // copy plane #1
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 1, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
            ct_copy_to_vx_plane(ctimg->data.y + ctimg->stride * height, (uint8_t*)pdata, 1, stride_bytes/2, addr);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 1, &addr, pdata), func, file, line);
        }

        { // copy plane #2
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void *pdata = 0;
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxAccessImagePatch(vximg, &rect, 2, &addr, &pdata, VX_WRITE_ONLY), func, file, line);
            ct_copy_to_vx_plane(ctimg->data.y + ctimg->stride * height + ctimg->stride * height/2/2, (uint8_t*)pdata, 1, stride_bytes/2, addr);
            ASSERT_EQ_VX_STATUS_AT_(return 0, VX_SUCCESS, vxCommitImagePatch(vximg, &rect, 2, &addr, pdata), func, file, line);
        }
    }

    return vximg;
}


/*
    wrap_half_modulo: specifies if the smallest value follows the biggest (e.g. 255 + 1 == 0 or not)
    0            = default - half of data type range
    (uint32_t)-1 = no wrap
    otherwise    = half of actual data range
*/
int ct_assert_eq_ctimage_impl(CT_Image expected, CT_Image actual, uint32_t threshold, uint32_t wrap_modulo,
                                const char* expected_str, const char* actual_str,
                                const char* func, const char* file, int line)

{
    uint32_t wrap_half_modulo = wrap_modulo/2;
    uint32_t i,j;
    uint32_t max_diff, max_x, max_y, vale, vala, diff_pixels;
    max_diff = max_x = max_y = vale = vala = diff_pixels = 0;
    if (!expected && !actual) return 1; // NULL == NULL, lets pass

    if (!expected || !actual)
    {
        if (actual)
            CT_RecordFailureAtFormat("Expected: %s == %s\n\tActual: NULL != %p", func, file, line, expected_str, actual_str, actual);
        else
            CT_RecordFailureAtFormat("Expected: %s == %s\n\tActual: %p != NULL", func, file, line, expected_str, actual_str, expected);
        return 0; // fail
    }

    if (expected->format != actual->format || expected->width  != actual->width || expected->height != actual->height)
    {
        CT_RecordFailureAtFormat("Expected: %s\n\t\twhich is %ux%u %.4s image\n\t"
                                 "Actual:   %s\n\t\tWhich is %ux%u %.4s image",
                                    func, file, line,
                                    expected_str, (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                    actual_str, (unsigned)actual->width, (unsigned)actual->height, &actual->format);
        return 0; // fail
    }

#define FIND_MAX_DIFF_SIMPLE(eptr, aptr)                                                        \
    {                                                                                           \
        for (i = 0; i < expected->height; ++i)                                                  \
            for (j = 0; j < expected->width; ++j)                                               \
            {                                                                                   \
                uint32_t diff = eptr[i * expected->stride + j] > aptr[i * actual->stride + j]   \
                              ? eptr[i * expected->stride + j] - aptr[i * actual->stride + j]   \
                              : aptr[i * actual->stride + j] - eptr[i * expected->stride + j];  \
                if (diff > wrap_half_modulo) diff = (uint32_t)(2 * wrap_half_modulo - diff);    \
                if (diff > max_diff)                                                            \
                {                                                                               \
                    max_diff = diff;                                                            \
                    max_y = i;                                                                  \
                    max_x = j;                                                                  \
                    vale = eptr[i * expected->stride + j];                                      \
                    vala = aptr[i * actual->stride + j];                                        \
                }                                                                               \
                if (diff > threshold)                                                           \
                    ++diff_pixels;                                                              \
            }                                                                                   \
    }

    if (expected->format == VX_DF_IMAGE_U8)
    {
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 7;
        FIND_MAX_DIFF_SIMPLE(expected->data.y, actual->data.y)
    }
    else if (expected->format == VX_DF_IMAGE_U16)
    {
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 15;
        FIND_MAX_DIFF_SIMPLE(expected->data.u16, actual->data.u16)
    }
    else if (expected->format == VX_DF_IMAGE_S16)
    {
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 15;
        FIND_MAX_DIFF_SIMPLE(expected->data.s16, actual->data.s16)
    }
    else if (expected->format == VX_DF_IMAGE_U32)
    {
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 31;
        FIND_MAX_DIFF_SIMPLE(expected->data.u32, actual->data.u32)
    }
    else if (expected->format == VX_DF_IMAGE_S32)
    {
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 31;
        FIND_MAX_DIFF_SIMPLE(expected->data.s32, actual->data.s32)
    }
    else
    {
        // skip check
    }

    if (expected->format == VX_DF_IMAGE_U8 || expected->format == VX_DF_IMAGE_U16 || expected->format == VX_DF_IMAGE_U32)
    {
        if (max_diff > threshold)
        {
            if (threshold == 0)
                CT_RecordFailureAtFormat("Testing (%s) image to be exactly equal to (%s)\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at point [x = %u, y = %u]:\n\t"
                                         "\tExpected: %u\n\t"
                                         "\tActual:   %u\n\t"
                                         "Totally %u pixels with different values",
                                         func, file, line, actual_str, expected_str,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, max_y, vale, vala, diff_pixels);
            else
                CT_RecordFailureAtFormat("Testing the difference between (%s) and (%s) images to be not greater than %u\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at point [x = %u, y = %u]:\n\t"
                                         "\tExpected: %u\n\t"
                                         "\tActual:   %u\n\t"
                                         "Totally %u pixels exceeding the given error",
                                         func, file, line, actual_str, expected_str, threshold,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, max_y, vale, vala, diff_pixels);

            return 0; // fail
        }
        return 1; // pass
    }

    if (expected->format == VX_DF_IMAGE_S16 || expected->format == VX_DF_IMAGE_S32)
    {
        if (max_diff > threshold)
        {
            if (threshold == 0)
                CT_RecordFailureAtFormat("Testing (%s) image to be exactly equal to (%s)\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at point [x = %u, y = %u]:\n\t"
                                         "\tExpected: %d\n\t"
                                         "\tActual:   %d\n\t"
                                         "Totally %u pixels with different values",
                                         func, file, line, actual_str, expected_str,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, max_y, (int32_t)vale, (int32_t)vala, diff_pixels);
            else
                CT_RecordFailureAtFormat("Testing the difference between (%s) and (%s) images to be not greater than %u\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at point [x = %u, y = %u]:\n\t"
                                         "\tExpected: %d\n\t"
                                         "\tActual:   %d\n\t"
                                         "Totally %u pixels exceeding the given error",
                                         func, file, line, actual_str, expected_str, threshold,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, max_y, (int32_t)vale, (int32_t)vala, diff_pixels);

            return 0; // fail
        }
        return 1;// pass
    }

    // for now check other formats as binary blobs
    if (expected->stride != expected->width || actual->stride != actual->width)
    {
        // no ROI support for now
        CT_RecordFailureAtFormat("Can't compare (%s) with (%s) because ROI comparison is not implemented for %.4s image format",
            func, file, line, expected_str, actual_str, &expected->format);
        return 0; // fail
    }

    {
        uint32_t size = (uint32_t)ct_image_data_size(expected->width, expected->height, expected->format);
        max_diff = max_x = max_y = vale = vala = diff_pixels = 0;
        if (!wrap_half_modulo) wrap_half_modulo = 1u << 7;
        for (i = 0; i < size; ++i)
        {
            int val = expected->data.y[i] - actual->data.y[i];
            if (val < 0) val = -val;

            if ((uint32_t)val > wrap_half_modulo) val = (int)(2 * wrap_half_modulo - val);

            if ((uint32_t)val > max_diff)
            {
                max_diff = (uint32_t)val;
                max_x = i;
                vale = expected->data.y[i];
                vala = actual->data.y[i];
            }
            if ((uint32_t)val > threshold)
                ++diff_pixels;
        }

        if (max_diff > threshold)
        {
            if (threshold == 0)
                CT_RecordFailureAtFormat("Testing (%s) image to be exactly equal to (%s)\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at offset %u:\n\t"
                                         "\tExpected: %u\n\t"
                                         "\tActual:   %u\n\t"
                                         "Totally %u bytes with different values",
                                         func, file, line, actual_str, expected_str,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, vale, vala, diff_pixels);
            else
                CT_RecordFailureAtFormat("Testing the difference between (%s) and (%s) images to be not greater than %u\n\t"
                                         "\t%ux%u %.4s images\n\t"
                                         "Max difference (%u) is found at offset %u:\n\t"
                                         "\tExpected: %u\n\t"
                                         "\tActual:   %u\n\t"
                                         "Totally %u bytes exceeding the given error",
                                         func, file, line, actual_str, expected_str, threshold,
                                         (unsigned)expected->width, (unsigned)expected->height, &expected->format,
                                         max_diff, max_x, vale, vala, diff_pixels);

            return 0; // fail
        }
    }

    return 1;
}


uint8_t* ct_image_data_ptr_8u(CT_Image image, uint32_t x, uint32_t y)
{
    uint8_t* ptr = &image->data.y[y * image->stride + x];
    return ptr;
}

uint8_t ct_image_data_replicate_8u(CT_Image image, int32_t x, int32_t y)
{
    uint8_t* ptr = NULL;

    EXPECT(image->width > 0 && image->height > 0);
    if (x < 0) x = 0;
    if (x >= (int)image->width) x = image->width - 1;
    if (y < 0) y = 0;
    if (y >= (int)image->height) y = image->height - 1;
    ptr = &image->data.y[y * image->stride + x];
    return *ptr;
}

uint8_t ct_image_data_constant_8u(CT_Image image, int32_t x, int32_t y, vx_uint32 constant_value)
{
    uint8_t* ptr = NULL;

    if (x < 0 || x >= (int)image->width || y < 0 || y >= (int)image->height)
        return (uint8_t)constant_value;
    ptr = &image->data.y[y * image->stride + x];
    return *ptr;
}

#ifndef MAX_DUMP_SIZE
#define MAX_DUMP_SIZE 5
#endif

void ct_dump_image_info_ex(CT_Image image, int dump_width, int dump_height)
{
    int x, y;
    const char* strend = "\n";

    int max_y = (int)image->height, max_x = (int)image->width;
    if (dump_height > 0 && max_y > dump_height) max_y = dump_height;
    if (dump_width > 0 && max_x > dump_width) max_x = dump_width;

    if(dump_width < 0)
        max_x = CT_MIN(max_x, MAX_DUMP_SIZE);
    if(dump_height < 0)
        max_y = CT_MIN(max_y, MAX_DUMP_SIZE);

    if(max_x < (int)image->width)
        strend = " ...\n";

    printf("CT_Image %p DF_IMAGE(%.*s): %dx%d\n",
            image, (int)sizeof(image->format), (char*)&image->format,
            image->width, image->height);

    if (image->format == VX_DF_IMAGE_U8)
    {
        for (y = 0; y < max_y; ++y)
        {
            for (x = 0; x < max_x; ++x)
            {
                uint8_t* ptr = ct_image_data_ptr_8u(image, x, y);
                printf("%3d ", (int)*ptr);
            }
            printf("%s", strend);
        }
    }
    else if (image->format == VX_DF_IMAGE_S16)
    {
        for (y = 0; y < max_y; ++y)
        {
            for (x = 0; x < max_x; ++x)
            {
                int16_t* ptr = CT_IMAGE_DATA_PTR_16S(image, x, y);
                printf("%6d ", (int)*ptr); // -32768..32767
            }
            printf("%s", strend);
        }
    }
    else if (image->format == VX_DF_IMAGE_U32)
    {
        for (y = 0; y < max_y; ++y)
        {
            for (x = 0; x < max_x; ++x)
            {
                uint32_t* ptr = CT_IMAGE_DATA_PTR_32U(image, x, y);
                printf("%16d ", (int)*ptr);
            }
            printf("%s", strend);
        }
    }
    else if (image->format == VX_DF_IMAGE_RGB)
    {
        for (y = 0; y < max_y; ++y)
        {
            for (x = 0; x < max_x; ++x)
            {
                uint8_t* ptr = (uint8_t*)CT_IMAGE_DATA_PTR_RGB(image, x, y);
                printf("(%3d,%3d,%3d) ", (int)ptr[0], (int)ptr[1], (int)ptr[2]);
            }
            printf("%s", strend);
        }
    }
    else if (image->format == VX_DF_IMAGE_RGBX)
    {
        for (y = 0; y < max_y; ++y)
        {
            for (x = 0; x < max_x; ++x)
            {
                uint8_t* ptr = (uint8_t*)CT_IMAGE_DATA_PTR_RGBX(image, x, y);
                printf("(%3d,%3d,%3d,%3d) ", (int)ptr[0], (int)ptr[1], (int)ptr[2], (int)ptr[3]);
            }
            printf("%s", strend);
        }
    }
    else
    {
        size_t sz = ct_image_data_size(image->width, image->height, image->format);
        size_t i = 0, j = 0;
        sz = CT_MIN(sz, (size_t)(max_x*max_y));
        for (i = 0; i < sz; i += 16)
        {
            for (j = 0; j < 16 && i + j < sz; j++)
            {
                printf("%3d ", image->data.y[i + j]);
            }
            printf("\n");
        }
    }
}

void ct_fill_ct_image_random(CT_Image image, uint64_t* seed, int a, int b)
{
    vx_df_image format = image->format;
    uint32_t p, x, y, nplanes=1, width[3] = {image->width, 0, 0}, height[3] = {image->height, 0, 0};
    uint32_t stride[3] = {ct_stride_bytes(image), 0, 0};

    if( format == VX_DF_IMAGE_RGB || format == VX_DF_IMAGE_RGBX || format == VX_DF_IMAGE_YUYV || format == VX_DF_IMAGE_UYVY )
    {
        width[0] *= format == VX_DF_IMAGE_RGB ? 3 : format == VX_DF_IMAGE_RGBX ? 4 : 2;
        format = VX_DF_IMAGE_U8;
    }
    else if( format == VX_DF_IMAGE_YUV4 || format == VX_DF_IMAGE_IYUV )
    {
        uint32_t scale = format == VX_DF_IMAGE_IYUV ? 2 : 1;
        nplanes = 3;
        width[1] = width[2] = width[0]/scale;
        height[1] = height[2] = height[0]/scale;
        stride[1] = stride[2] = stride[0]/scale;
        format = VX_DF_IMAGE_U8;
    }
    else if( format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21 )
    {
        nplanes = 2;
        width[1] = width[0];
        height[1] = height[0]/2;
        stride[1] = stride[0];
        format = VX_DF_IMAGE_U8;
    }

    ASSERT( format == VX_DF_IMAGE_U8 ||
            format == VX_DF_IMAGE_U16 || format == VX_DF_IMAGE_S16 ||
            format == VX_DF_IMAGE_U32 || format == VX_DF_IMAGE_S32);

#undef CASE_FILL_RNG
#define CASE_FILL_RNG(format, type, cast_macro) \
    case format: \
    { \
    uint8_t* ptr = image->data.y; \
    for( p = 0; p < nplanes; p++ ) \
    for( y = 0; y < height[p]; y++, ptr += stride[p] ) \
    { \
        type* tptr = (type*)ptr; \
        for( x = 0; x < width[p]; x++ ) \
        { \
            int val = CT_RNG_NEXT_INT(*seed, a, b); \
            tptr[x] = cast_macro(val); \
        } \
    } \
    } \
    break

    switch( format )
    {
        CASE_FILL_RNG(VX_DF_IMAGE_U8, uint8_t, CT_CAST_U8);
        CASE_FILL_RNG(VX_DF_IMAGE_U16, uint16_t, CT_CAST_U16);
        CASE_FILL_RNG(VX_DF_IMAGE_S16, int16_t, CT_CAST_S16);
        CASE_FILL_RNG(VX_DF_IMAGE_U32, uint32_t, CT_CAST_U32);
        CASE_FILL_RNG(VX_DF_IMAGE_S32, int32_t, CT_CAST_S32);
    default:
        break;
    }
}

CT_Image ct_allocate_ct_image_random(uint32_t width, uint32_t height,
                                     vx_df_image format, uint64_t* seed, int a, int b)
{
    CT_Image image = ct_allocate_image(width, height, format);
    if(image)
        ct_fill_ct_image_random(image, seed, a, b);
    return image;
}


uint32_t ct_get_num_planes(vx_df_image format)
{
    uint32_t nplanes = 0;

    switch (format)
    {
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGB:
    case VX_DF_IMAGE_RGBX:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY: nplanes = 1; break;

    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21: nplanes = 2; break;

    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV: nplanes = 3; break;

    default:
        CT_RecordFailure();
        break;
    }

    return nplanes;
}


int ct_get_num_channels(vx_df_image format)
{
    switch (format)
    {
        case VX_DF_IMAGE_U8:
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
            return 1;
        case VX_DF_IMAGE_RGBX:
            return 4;
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            return 3;
    };
    CT_RecordFailure();
    return 0;
}


int ct_image_get_channel_step_x(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    switch (format)
    {
        case VX_DF_IMAGE_U8:
            return 1;
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
            return 2;
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case VX_DF_IMAGE_RGBX:
            return 4;
        case VX_DF_IMAGE_RGB:
            return 3;
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            if (channel == VX_CHANNEL_Y)
                return 2;
            return 4;
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4:
            return 1;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            if (channel == VX_CHANNEL_Y)
                return 1;
            return 2;
    };
    CT_RecordFailure();
    return 0;
}


int ct_image_get_channel_step_y(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    switch (format)
    {
        case VX_DF_IMAGE_U8:
            return image->stride;
        case VX_DF_IMAGE_U16:
        case VX_DF_IMAGE_S16:
            return image->stride * 2;
        case VX_DF_IMAGE_U32:
        case VX_DF_IMAGE_S32:
        case VX_DF_IMAGE_RGBX:
            return image->stride * 4;
        case VX_DF_IMAGE_RGB:
            return image->stride * 3;
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            return image->stride * 2;
        case VX_DF_IMAGE_IYUV:
            if (channel == VX_CHANNEL_Y)
                return image->stride;
            return image->stride / 2;
        case VX_DF_IMAGE_YUV4:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            return image->stride;
    };
    CT_RecordFailure();
    return 0;
}


int ct_image_get_channel_subsampling_x(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    if (channel == VX_CHANNEL_Y)
        return 1;

    switch (format)
    {
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            return 2;
    }

    return 1;
}


int ct_image_get_channel_subsampling_y(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    if (channel == VX_CHANNEL_Y)
        return 1;

    switch (format)
    {
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            return 2;
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            return 1;
    }

    return 1;
}


int ct_image_get_channel_plane(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    switch (format)
    {
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
            return 0;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            if (channel == VX_CHANNEL_Y)
                return 0;
            return 1;
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4:
      switch (channel)
        {
        case VX_CHANNEL_Y:
          return 0;
        case VX_CHANNEL_U:
          return 1;
        case VX_CHANNEL_V:
              return 2;
        default:
          CT_FAIL_(return 0, "Invalid channel");
        }
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            return 0;
    }

    CT_FAIL_(return 0, "Not implemented");
}


int ct_image_get_channel_component(CT_Image image, vx_enum channel)
{
    vx_df_image format = image->format;

    switch (format)
    {
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
        switch (channel)
        {
        case VX_CHANNEL_R:
          return 0;
        case VX_CHANNEL_G:
          return 1;
        case VX_CHANNEL_B:
          return 2;
        case VX_CHANNEL_A:
          return 3;
        default:
          CT_FAIL_(return 0, "Invalid channel");
        }
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
            if (channel == VX_CHANNEL_Y)
                return 0;
            else if ((channel == VX_CHANNEL_U && format == VX_DF_IMAGE_NV12) ||
                     (channel == VX_CHANNEL_V && format == VX_DF_IMAGE_NV21))
                return 0;
            else
                return 1;
            break;
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4:
            return 0;
            break;
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_UYVY:
            if (channel == VX_CHANNEL_Y)
                return (format == VX_DF_IMAGE_YUYV ? 0 : 1);
            else
                return (format == VX_DF_IMAGE_YUYV ? 1 : 0) + (channel == VX_CHANNEL_U ? 0 : 2);
    }

    CT_FAIL_(return 0, "Not implemented");
}


uint8_t* ct_image_get_plane_base(CT_Image ctimg, int plane)
{
    int format = ctimg->format;

    if (plane == 0)
        return ctimg->data.y;

    if (format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21)
    {
        if (plane == 1)
            return ctimg->data.y + ctimg->stride * ctimg->height;
    }
    else if (format == VX_DF_IMAGE_YUV4)
    {
        if (plane == 1)
            return ctimg->data.y + ctimg->stride * ctimg->height;
        if (plane == 2)
            return ctimg->data.y + ctimg->stride * ctimg->height * 2;
    }
    else if (format == VX_DF_IMAGE_IYUV)
    {
        if (plane == 1)
            return ctimg->data.y + ctimg->stride * ctimg->height;
        if (plane == 2)
            return ctimg->data.y + ctimg->stride * ctimg->height + ctimg->stride * ctimg->height / 2 / 2;
    }

    return NULL;
}


CT_Image ct_image_create_clone(CT_Image image)
{
    CT_Image clone = NULL;
    size_t sz = 0;
    ASSERT_NO_FAILURE_(return NULL, clone = ct_allocate_image(image->width, image->height, image->format));

    sz = ct_image_data_size(image->width, image->height, image->format);

    memcpy(clone->data.y, image->data.y, sz);

    return clone;
}

int ct_image_read_rect_S32(CT_Image img, int32_t *dst, int32_t sx, int32_t sy, int32_t ex, int32_t ey, vx_border_mode_t border)
{
    int32_t width = (vx_int32)img->width, height = (vx_int32)img->height;
    int32_t stride_y = ct_image_get_channel_step_y(img, VX_CHANNEL_0);
    int32_t stride_x = ct_image_get_channel_step_x(img, VX_CHANNEL_0);
    vx_df_image format = img->format;
    const vx_uint8 *ptr = (const vx_uint8 *)img->data.y;
    vx_int32 x, y;
    vx_uint32 dest_index = 0;
    int res = 1;

    ASSERT_(return 0, format == VX_DF_IMAGE_U8); // Only 8U is supported

    if (border.mode == VX_BORDER_MODE_REPLICATE)
    {
        for (y = sy; y <= ey; ++y)
        {
            int32_t y_ = y < 0 ? 0 : y >= height ? height - 1 : y;

            for (x = sx; x <= ex; ++x, ++dest_index)
            {
                int32_t x_ = x < 0 ? 0 : x >= width ? width - 1 : x;

                ((int32_t*)dst)[dest_index] = *(vx_uint8*)(ptr + y_*stride_y + x_*stride_x);
            }
        }
    }
    else if (border.mode == VX_BORDER_MODE_CONSTANT)
    {
        vx_uint32 cval = border.constant_value;
        for (y = sy; y <= ey; ++y)
        {
            int ccase_y = y < 0 || y >= height;

            for (x = sx; x <= ex; ++x, ++dest_index)
            {
                int ccase = ccase_y || x < 0 || x >= width;

                if( !ccase )
                    ((int32_t*)dst)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                else
                    ((int32_t*)dst)[dest_index] = (int32_t)cval;
            }
        }
    }
    else if (border.mode == VX_BORDER_MODE_UNDEFINED)
    {
        for (y = sy; y <= ey; ++y)
        {
            int ccase_y = y < 0 || y >= height;

            for (x = sx; x <= ex; ++x, ++dest_index)
            {
                int ccase = ccase_y || x < 0 || x >= width;

                if (!ccase)
                {
                    ((int32_t*)dst)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                }
                else
                {
                    ((int32_t*)dst)[dest_index] = (int32_t)INT32_MIN;
                    res = 0;
                }
            }
        }
    }
    return res;
}
