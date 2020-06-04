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

#ifndef __VX_CT_IMAGE_H__
#define __VX_CT_IMAGE_H__

#include <VX/vx.h>

typedef struct CT_Rect_ {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} CT_Rect;

typedef struct CT_ImageHdr {
    uint32_t   width;  // in pixels
    uint32_t   height;
    uint32_t   stride; // in pixels
    vx_df_image  format;
    CT_Rect    roi;    // stores top left corner offset and full size of parent image

    union
    {
      uint8_t*  y;
      uint16_t* u16;
      int16_t*  s16;
      uint32_t* u32;
      int32_t*  s32;
      struct { uint8_t r,g,b; }*     rgb;
      struct { uint8_t y,u,v; }*     yuv;
      struct { uint8_t r,g,b,x; }*   rgbx;
      struct { uint8_t y0,u,y1,v; }* yuyv;
      struct { uint8_t u,y0,v,y1; }* uyvy;
    } data;

    // private area
    void* data_begin_;
    uint32_t* refcount_;
} *CT_Image;

int ct_channels(vx_df_image format);
uint32_t ct_stride_bytes(CT_Image image);
uint32_t ct_image_bits_per_pixel(vx_df_image format);

uint8_t* ct_image_get_plane_base(CT_Image img, int plane);
int ct_image_get_channel_step_x(CT_Image image, vx_enum channel);
int ct_image_get_channel_step_y(CT_Image image, vx_enum channel);
int ct_image_get_channel_subsampling_x(CT_Image image, vx_enum channel);
int ct_image_get_channel_subsampling_y(CT_Image image, vx_enum channel);
int ct_image_get_channel_plane(CT_Image image, vx_enum channel);
int ct_image_get_channel_component(CT_Image image, vx_enum channel);

uint32_t ct_get_num_planes(vx_df_image format);
int ct_get_num_channels(vx_df_image format);

CT_Image ct_allocate_image(uint32_t width, uint32_t height, vx_df_image format);
CT_Image ct_allocate_image_hdr(uint32_t width, uint32_t height, uint32_t stride, vx_df_image format, void* data);
CT_Image ct_get_image_roi(CT_Image img, CT_Rect roi);
CT_Image ct_get_image_roi_(CT_Image img, uint32_t x_start, uint32_t y_start, uint32_t width, uint32_t height);
void ct_adjust_roi(CT_Image img, int left, int top, int right, int bottom);

#if 1
#define CT_IMAGE_DATA_PTR_8U(image, x, y_) &(image)->data.y[(y_) * (image)->stride + (x)]
#else
uint8_t* ct_image_data_ptr_8u(CT_Image image, uint32_t x, uint32_t y);
#define CT_IMAGE_DATA_PTR_8U(image, x, y) ct_image_data_ptr_8u(image, x, y)
#endif

uint8_t ct_image_data_replicate_8u(CT_Image image, int32_t x, int32_t y);
#define CT_IMAGE_DATA_REPLICATE_8U(image, x, y) ct_image_data_replicate_8u(image, x, y)

uint8_t ct_image_data_constant_8u(CT_Image image, int32_t x, int32_t y, vx_uint32 constant_value);
#define CT_IMAGE_DATA_CONSTANT_8U(image, x, y, constant_value) ct_image_data_constant_8u(image, x, y, constant_value)

#define CT_IMAGE_DATA_PTR_16S(image, x, y) &(image)->data.s16[(y) * (image)->stride + (x)]

#define CT_IMAGE_DATA_PTR_32U(image, x, y) &(image)->data.u32[(y) * (image)->stride + (x)]

#define CT_IMAGE_DATA_PTR_RGB(image, x, y) &(image)->data.rgb[(y) * (image)->stride + (x)]
#define CT_IMAGE_DATA_PTR_RGBX(image, x, y) &(image)->data.rgbx[(y) * (image)->stride + (x)]


#define CT_FILL_IMAGE_8U(ret_error, image, op) \
    ASSERT_(ret_error, image != NULL); \
    ASSERT_(ret_error, image->format == VX_DF_IMAGE_U8); \
    ASSERT_(ret_error, image->width > 0); \
    ASSERT_(ret_error, image->height > 0); \
    { \
        uint32_t x, y; \
        for (y = 0; y < image->height; y++) { \
            for (x = 0; x < image->width; x++) { \
                uint8_t* dst_data = CT_IMAGE_DATA_PTR_8U(image, x, y); (void)dst_data; \
                op; \
            } \
        } \
    }


#define CT_FILL_IMAGE_16S(ret_error, image, op) \
    ASSERT_(ret_error, image != NULL); \
    ASSERT_(ret_error, image->format == VX_DF_IMAGE_S16); \
    ASSERT_(ret_error, image->width > 0); \
    ASSERT_(ret_error, image->height > 0); \
    { \
        uint32_t x, y; \
        for (y = 0; y < image->height; y++) { \
            for (x = 0; x < image->width; x++) { \
                int16_t* dst_data = CT_IMAGE_DATA_PTR_16S(image, x, y); \
                op; \
            } \
        } \
    }

#define CT_FILL_IMAGE_32U(ret_error, image, op) \
    ASSERT_(ret_error, image != NULL); \
    ASSERT_(ret_error, image->format == VX_DF_IMAGE_U32); \
    ASSERT_(ret_error, image->width > 0); \
    ASSERT_(ret_error, image->height > 0); \
    { \
        uint32_t x, y; \
        for (y = 0; y < image->height; y++) { \
            for (x = 0; x < image->width; x++) { \
                uint32_t* dst_data = CT_IMAGE_DATA_PTR_32U(image, x, y); \
                op; \
            } \
        } \
    }


CT_Image ct_read_image(const char* fileName, int dcn);
void ct_write_image(const char* fileName, CT_Image image);

#define ct_image_from_vx_image(vximg) ct_image_from_vx_image_impl(vximg, __FUNCTION__, __FILE__, __LINE__)
CT_Image ct_image_from_vx_image_impl(vx_image vximg, const char* func, const char* file, int line);

#define ct_image_to_vx_image(ctimg, context) ct_image_to_vx_image_impl(ctimg, context, __FUNCTION__, __FILE__, __LINE__)
vx_image ct_image_to_vx_image_impl(CT_Image ctimg, vx_context context, const char* func, const char* file, int line);

#define ct_image_copyto_vx_image(vximg, ctimg) ct_image_copyto_vx_image_impl(vximg, ctimg, __FUNCTION__, __FILE__, __LINE__)
vx_image ct_image_copyto_vx_image_impl(vx_image vximg, CT_Image ctimg, const char* func, const char* file, int line);

#define EXPECT_EQ_CTIMAGE(expected, actual) ct_assert_eq_ctimage_impl(expected, actual, 0, (uint32_t)-1, #expected, #actual, __FUNCTION__, __FILE__, __LINE__)
#define ASSERT_EQ_CTIMAGE(expected, actual)                                                                                     \
    do { if (ct_assert_eq_ctimage_impl(expected, actual, 0, (uint32_t)-1, #expected, #actual, __FUNCTION__, __FILE__, __LINE__))\
        {} else { CT_DO_FAIL; }} while(0)

#define EXPECT_CTIMAGE_NEAR(expected, actual, threshold) ct_assert_eq_ctimage_impl(expected, actual, threshold, (uint32_t)-1, #expected, #actual, __FUNCTION__, __FILE__, __LINE__)
#define ASSERT_CTIMAGE_NEAR(expected, actual, threshold)                                                                                \
    do { if (ct_assert_eq_ctimage_impl(expected, actual, threshold, (uint32_t)-1, #expected, #actual, __FUNCTION__, __FILE__, __LINE__))\
        {} else { CT_DO_FAIL; }} while(0)

#define CTIMAGE_ALLOW_WRAP 0

#define EXPECT_CTIMAGE_NEARWRAP(expected, actual, threshold, modulo) ct_assert_eq_ctimage_impl(expected, actual, threshold, modulo, #expected, #actual, __FUNCTION__, __FILE__, __LINE__)
#define ASSERT_CTIMAGE_NEARWRAP(expected, actual, threshold, modulo)                                                                \
    do { if (ct_assert_eq_ctimage_impl(expected, actual, threshold, modulo, #expected, #actual, __FUNCTION__, __FILE__, __LINE__))  \
        {} else { CT_DO_FAIL; }} while(0)

int ct_assert_eq_ctimage_impl(CT_Image expected, CT_Image actual, uint32_t threshold, uint32_t wrap_modulo,
                                const char* expected_str, const char* actual_str,
                                const char* func, const char* file, int line);

void ct_dump_image_info_ex(CT_Image image, int dump_width, int dump_height);
#define ct_dump_image_info(image) ct_dump_image_info_ex(image, -1, -1);

void ct_fill_ct_image_random(CT_Image image, uint64_t* seed, int a, int b);
CT_Image ct_allocate_ct_image_random(uint32_t width, uint32_t height, vx_df_image format, uint64_t* rng, int a, int b);

CT_Image ct_image_create_clone(CT_Image image);

int ct_image_read_rect_S32(CT_Image img, int32_t *dst, int32_t sx, int32_t sy, int32_t ex, int32_t ey, vx_border_mode_t border);

#endif // __VX_CT_IMAGE_H__
