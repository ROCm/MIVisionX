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

#include "test_engine/test.h"

#include <VX/vx.h>

TESTCASE(Image, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    vx_df_image format;
} format_arg;

TEST_WITH_ARG(Image, testRngImageCreation, format_arg,
    ARG_ENUM(VX_DF_IMAGE_U8),
    ARG_ENUM(VX_DF_IMAGE_U16),
    ARG_ENUM(VX_DF_IMAGE_S16),
    ARG_ENUM(VX_DF_IMAGE_U32),
    ARG_ENUM(VX_DF_IMAGE_S32),
    ARG_ENUM(VX_DF_IMAGE_RGB),
    ARG_ENUM(VX_DF_IMAGE_RGBX),
    ARG_ENUM(VX_DF_IMAGE_NV12),
    ARG_ENUM(VX_DF_IMAGE_NV21),
    ARG_ENUM(VX_DF_IMAGE_UYVY),
    ARG_ENUM(VX_DF_IMAGE_YUYV),
    ARG_ENUM(VX_DF_IMAGE_IYUV),
    ARG_ENUM(VX_DF_IMAGE_YUV4),
    ARG_ENUM(VX_DF_IMAGE_VIRT),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_image   clone   = 0;
    vx_df_image  format  = arg_->format;

    image = vxCreateImage(context, 4, 4, format);

    if (format == VX_DF_IMAGE_VIRT)
    {
        EXPECT_VX_ERROR(image, VX_ERROR_INVALID_PARAMETERS);
        PASS();
    }

    // VX_CALL(ct_dump_vx_image_info(image));

    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    ct_fill_image_random(image, &CT()->seed_);

    clone = ct_clone_image(image, 0);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    vxReleaseImage(&image);
    vxReleaseImage(&clone);

    ASSERT(image == 0);
    ASSERT(clone == 0);
}

TEST_WITH_ARG(Image, testVirtualImageCreation, format_arg,
    ARG_ENUM(VX_DF_IMAGE_U8),
    ARG_ENUM(VX_DF_IMAGE_U16),
    ARG_ENUM(VX_DF_IMAGE_S16),
    ARG_ENUM(VX_DF_IMAGE_U32),
    ARG_ENUM(VX_DF_IMAGE_S32),
    ARG_ENUM(VX_DF_IMAGE_RGB),
    ARG_ENUM(VX_DF_IMAGE_RGBX),
    ARG_ENUM(VX_DF_IMAGE_NV12),
    ARG_ENUM(VX_DF_IMAGE_NV21),
    ARG_ENUM(VX_DF_IMAGE_UYVY),
    ARG_ENUM(VX_DF_IMAGE_YUYV),
    ARG_ENUM(VX_DF_IMAGE_IYUV),
    ARG_ENUM(VX_DF_IMAGE_YUV4),
    ARG_ENUM(VX_DF_IMAGE_VIRT),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_image   clone   = 0;
    vx_df_image  format  = arg_->format;

    vx_graph graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    image = vxCreateVirtualImage(graph, 4, 4, format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    clone = ct_clone_image(image, graph);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    vxReleaseImage(&image);
    vxReleaseImage(&clone);
    vxReleaseGraph(&graph);

    ASSERT(image == 0);
    ASSERT(clone == 0);
    ASSERT(graph == 0);
}

typedef struct {
    const char* name;
    int width;
    int height;
    vx_df_image format;
} dims_arg;

TEST_WITH_ARG(Image, testVirtualImageCreationDims, dims_arg,
    ARG("0_0_REAL", 0, 0, VX_DF_IMAGE_U8),
    ARG("DISABLED_0_4_REAL", 0, 4, VX_DF_IMAGE_U8),
    ARG("DISABLED_4_0_REAL", 4, 0, VX_DF_IMAGE_U8),
    ARG("4_4_REAL", 4, 4, VX_DF_IMAGE_U8),
    ARG("0_0_VIRT", 0, 0, VX_DF_IMAGE_VIRT),
    ARG("DISABLED_0_4_VIRT", 0, 4, VX_DF_IMAGE_VIRT),
    ARG("DISABLED_4_0_VIRT", 4, 0, VX_DF_IMAGE_VIRT),
    ARG("4_4_VIRT", 4, 4, VX_DF_IMAGE_VIRT),
    )
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_image   clone   = 0;

    vx_graph graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    image = vxCreateVirtualImage(graph, arg_->width, arg_->height, arg_->format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    clone = ct_clone_image(image, graph);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    vxReleaseImage(&image);
    vxReleaseImage(&clone);
    vxReleaseGraph(&graph);

    ASSERT(image == 0);
    ASSERT(clone == 0);
    ASSERT(graph == 0);
}


TEST_WITH_ARG(Image, testConvert_CT_Image, format_arg,
    ARG_ENUM(VX_DF_IMAGE_U8),
    ARG_ENUM(VX_DF_IMAGE_U16),
    ARG_ENUM(VX_DF_IMAGE_S16),
    ARG_ENUM(VX_DF_IMAGE_U32),
    ARG_ENUM(VX_DF_IMAGE_S32),
    ARG_ENUM(VX_DF_IMAGE_RGB),
    ARG_ENUM(VX_DF_IMAGE_RGBX),
    ARG_ENUM(VX_DF_IMAGE_NV12),
    ARG_ENUM(VX_DF_IMAGE_NV21),
    ARG_ENUM(VX_DF_IMAGE_UYVY),
    ARG_ENUM(VX_DF_IMAGE_YUYV),
    ARG_ENUM(VX_DF_IMAGE_IYUV),
    ARG_ENUM(VX_DF_IMAGE_YUV4),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0,
               image2  = 0;
    CT_Image   ctimg   = 0,
               ctimg2  = 0;

    image = vxCreateImage(context, 16, 16, arg_->format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(image, &CT()->seed_));

    ASSERT_NO_FAILURE(ctimg = ct_image_from_vx_image(image));

    ASSERT_NO_FAILURE(image2 = ct_image_to_vx_image(ctimg, context));

    ASSERT_NO_FAILURE(ctimg2 = ct_image_from_vx_image(image2));

    ASSERT_EQ_CTIMAGE(ctimg, ctimg2);

    vxReleaseImage(&image);
    vxReleaseImage(&image2);

    ASSERT(image == 0);
    ASSERT(image2 == 0);
}


/*
// Generate input to cover these requirements:
// There should be a 8u image with randomly generated pixel intensities.
*/
static CT_Image own_generate_rand_image(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
        image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
} /* own_generate_rand_image() */


typedef struct
{
    const char*      testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char*      fileName;
    int              width;
    int              height;
    vx_df_image      format;

} CreateImageFromHandle_Arg;


#define VX_PLANE_MAX (4)

#define ADD_IMAGE_FORMAT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U8", __VA_ARGS__, VX_DF_IMAGE_U8)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U16", __VA_ARGS__, VX_DF_IMAGE_U16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S16", __VA_ARGS__, VX_DF_IMAGE_S16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U32", __VA_ARGS__, VX_DF_IMAGE_U32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S32", __VA_ARGS__, VX_DF_IMAGE_S32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGB", __VA_ARGS__, VX_DF_IMAGE_RGB)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGBX", __VA_ARGS__, VX_DF_IMAGE_RGBX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_UYVY", __VA_ARGS__, VX_DF_IMAGE_UYVY)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUYV", __VA_ARGS__, VX_DF_IMAGE_YUYV)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV12", __VA_ARGS__, VX_DF_IMAGE_NV12)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV21", __VA_ARGS__, VX_DF_IMAGE_NV21)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUV4", __VA_ARGS__, VX_DF_IMAGE_YUV4)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_IYUV", __VA_ARGS__, VX_DF_IMAGE_IYUV))

#define CREATE_IMAGE_FROM_HANDLE_PARAMETERS \
    CT_GENERATE_PARAMETERS("rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT, ARG, own_generate_rand_image, NULL)

TEST_WITH_ARG(Image, testCreateImageFromHandle, CreateImageFromHandle_Arg, CREATE_IMAGE_FROM_HANDLE_PARAMETERS)
{
    vx_uint32 n;
    vx_uint32 nplanes;
    vx_context context = context_->vx_context_;
    vx_image image = 0;
    vx_imagepatch_addressing_t addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };

    int channel[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    CT_Image src = NULL;
    CT_Image tst = NULL;

    void* ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    switch (arg_->format) {
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
        channel[0] = VX_CHANNEL_0;
        break;
    case VX_DF_IMAGE_RGB:
    case VX_DF_IMAGE_RGBX:
        channel[0] = VX_CHANNEL_R;
        channel[1] = VX_CHANNEL_G;
        channel[2] = VX_CHANNEL_B;
        channel[3] = VX_CHANNEL_A;
        break;
    case VX_DF_IMAGE_UYVY:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV:
        channel[0] = VX_CHANNEL_Y;
        channel[1] = VX_CHANNEL_U;
        channel[2] = VX_CHANNEL_V;
        break;
    default:
        ASSERT(0);
    }

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_NO_FAILURE(nplanes = ct_get_num_planes(arg_->format));

    for (n = 0; n < nplanes; n++)
    {
        addr[n].dim_x = src->width / ct_image_get_channel_subsampling_x(src, channel[n]);
        addr[n].dim_y = src->height / ct_image_get_channel_subsampling_y(src, channel[n]);
        addr[n].scale_x = ct_image_get_channel_subsampling_x(src, channel[n]);
        addr[n].scale_y = ct_image_get_channel_subsampling_y(src, channel[n]);
        addr[n].step_x = ct_image_get_channel_step_x(src, channel[n]);
        addr[n].step_y = ct_image_get_channel_step_y(src, channel[n]);
        addr[n].stride_x = ct_image_get_channel_step_x(src, channel[n]);
        addr[n].stride_y = ct_image_get_channel_step_y(src, channel[n]);

        ptrs[n] = ct_image_get_plane_base(src, n);
    }

    image = vxCreateImageFromHandle(context, arg_->format, addr, ptrs, VX_IMPORT_TYPE_HOST);

    ASSERT_NO_FAILURE(tst = ct_image_from_vx_image(image));

    EXPECT_EQ_CTIMAGE(src, tst);

    vxReleaseImage(&image);

    ASSERT(image == 0);
}


typedef struct
{
    const char*      testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char*      fileName;
    int              width;
    int              height;
    vx_df_image      format;

} FormatImagePatchAddress1d_Arg;


#define VX_PLANE_MAX (4)

#define ADD_IMAGE_FORMAT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U8", __VA_ARGS__, VX_DF_IMAGE_U8)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U16", __VA_ARGS__, VX_DF_IMAGE_U16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S16", __VA_ARGS__, VX_DF_IMAGE_S16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U32", __VA_ARGS__, VX_DF_IMAGE_U32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S32", __VA_ARGS__, VX_DF_IMAGE_S32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGB", __VA_ARGS__, VX_DF_IMAGE_RGB)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGBX", __VA_ARGS__, VX_DF_IMAGE_RGBX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_UYVY", __VA_ARGS__, VX_DF_IMAGE_UYVY)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUYV", __VA_ARGS__, VX_DF_IMAGE_YUYV)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV12", __VA_ARGS__, VX_DF_IMAGE_NV12)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV21", __VA_ARGS__, VX_DF_IMAGE_NV21)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUV4", __VA_ARGS__, VX_DF_IMAGE_YUV4)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_IYUV", __VA_ARGS__, VX_DF_IMAGE_IYUV))

#define FORMAT_IMAGE_PATCH_ADDRESS_1D_PARAMETERS \
    CT_GENERATE_PARAMETERS("rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT, ARG, own_generate_rand_image, NULL)

TEST_WITH_ARG(Image, testFormatImagePatchAddress1d, FormatImagePatchAddress1d_Arg, FORMAT_IMAGE_PATCH_ADDRESS_1D_PARAMETERS)
{
    vx_uint8* p1;
    vx_uint8* p2;
    vx_uint32 i;
    vx_int32  j;
    vx_uint32 n;
    vx_uint32 nplanes;
    vx_context context = context_->vx_context_;
    vx_image image1 = 0;
    vx_image image2 = 0;
    vx_rectangle_t rect = { 0, 0, 0, 0 };
    vx_imagepatch_addressing_t addr1[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    vx_imagepatch_addressing_t addr2[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };

    CT_Image src = NULL;
    CT_Image tst = NULL;

    void* ptrs1[VX_PLANE_MAX] = { 0, 0, 0, 0 };
    void* ptrs2[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    image1 = ct_image_to_vx_image(src, context);
    ASSERT_VX_OBJECT(image1, VX_TYPE_IMAGE);

    image2 = vxCreateImage(context, arg_->width, arg_->height, arg_->format);
    ASSERT_VX_OBJECT(image2, VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(nplanes = ct_get_num_planes(arg_->format));

    for (n = 0; n < nplanes; n++)
    {
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = src->width;
        rect.end_y = src->height;

        VX_CALL(vxAccessImagePatch(image1, &rect, n, &addr1[n], &ptrs1[n], VX_READ_ONLY));
        VX_CALL(vxAccessImagePatch(image2, &rect, n, &addr2[n], &ptrs2[n], VX_WRITE_ONLY));

        /* use linear addressing function */
        for (i = 0; i < addr1[n].dim_x*addr1[n].dim_y; i += addr1[n].step_x)
        {
            p1 = vxFormatImagePatchAddress1d(ptrs1[n], i, &addr1[n]);
            p2 = vxFormatImagePatchAddress1d(ptrs2[n], i, &addr2[n]);
            for (j = 0; j < addr1[n].stride_x; j++)
              p2[j] = p1[j];
        }

        VX_CALL(vxCommitImagePatch(image1, 0, n, &addr1[n], ptrs1[n]));
        VX_CALL(vxCommitImagePatch(image2, 0, n, &addr2[n], ptrs2[n]));
    }

    ASSERT_NO_FAILURE(tst = ct_image_from_vx_image(image2));

    ASSERT_EQ_CTIMAGE(tst, src);

    vxReleaseImage(&image1);
    vxReleaseImage(&image2);

    ASSERT(image1 == 0);
    ASSERT(image2 == 0);
}

TEST_WITH_ARG(Image, testUniformImage, format_arg,
    ARG_ENUM(VX_DF_IMAGE_U8),
    ARG_ENUM(VX_DF_IMAGE_U16),
    ARG_ENUM(VX_DF_IMAGE_S16),
    ARG_ENUM(VX_DF_IMAGE_U32),
    ARG_ENUM(VX_DF_IMAGE_S32),
    ARG_ENUM(VX_DF_IMAGE_RGB),
    ARG_ENUM(VX_DF_IMAGE_RGBX),
    ARG_ENUM(VX_DF_IMAGE_NV12),
    ARG_ENUM(VX_DF_IMAGE_NV21),
    ARG_ENUM(VX_DF_IMAGE_UYVY),
    ARG_ENUM(VX_DF_IMAGE_YUYV),
    ARG_ENUM(VX_DF_IMAGE_IYUV),
    ARG_ENUM(VX_DF_IMAGE_YUV4),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    CT_Image   ctimg   = 0;
    CT_Image   refimg  = 0;
    int i;

    union
    {
        vx_uint8 c[4];
        vx_int16 s16;
        vx_uint16 u16;
        vx_int32 s32;
        vx_uint32 u32;
    } vals;

    switch (arg_->format)
    {
        case VX_DF_IMAGE_S32: vals.s32 = 0x11223344; break;
        case VX_DF_IMAGE_U32: vals.u32 = 0x11223344; break;
        case VX_DF_IMAGE_S16: vals.s16 = 0x1122; break;
        case VX_DF_IMAGE_U16: vals.u16 = 0x1122; break;
        default:
            vals.c[0] = 11;
            vals.c[1] = 22;
            vals.c[2] = 33;
            vals.c[3] = 44;
            break;
    }

    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, 640, 480, arg_->format, &vals), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(ctimg = ct_image_from_vx_image(image));

    ASSERT_NO_FAILURE(refimg = ct_allocate_image(640, 480, arg_->format));

    switch (arg_->format)
    {
        case VX_DF_IMAGE_U8:
            memset(refimg->data.y, vals.c[0], 640*480);
            break;
        case VX_DF_IMAGE_U16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u16[i] = vals.u16;
            break;
        case VX_DF_IMAGE_S16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s16[i] = vals.s16;
            break;
        case VX_DF_IMAGE_U32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u32[i] = vals.u32;
            break;
        case VX_DF_IMAGE_S32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s32[i] = vals.s32;
            break;
        case VX_DF_IMAGE_RGB:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgb[i].r = vals.c[0];
                refimg->data.rgb[i].g = vals.c[1];
                refimg->data.rgb[i].b = vals.c[2];
            }
            break;
        case VX_DF_IMAGE_RGBX:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgbx[i].r = vals.c[0];
                refimg->data.rgbx[i].g = vals.c[1];
                refimg->data.rgbx[i].b = vals.c[2];
                refimg->data.rgbx[i].x = vals.c[3];
            }
            break;
        case VX_DF_IMAGE_YUV4:
            memset(refimg->data.y + 640*480*0, vals.c[0], 640*480);
            memset(refimg->data.y + 640*480*1, vals.c[1], 640*480);
            memset(refimg->data.y + 640*480*2, vals.c[2], 640*480);
            break;
        case VX_DF_IMAGE_IYUV:
            memset(refimg->data.y, vals.c[0], 640*480);
            memset(refimg->data.y + 640*480, vals.c[1], 640/2*480/2);
            memset(refimg->data.y + 640*480 + 640/2*480/2, vals.c[2], 640/2*480/2);
            break;
        case VX_DF_IMAGE_NV12:
            memset(refimg->data.y, vals.c[0], 640*480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.c[1];
                refimg->data.y[640*480 + 2 * i + 1] = vals.c[2];
            }
            break;
        case VX_DF_IMAGE_NV21:
            memset(refimg->data.y, vals.c[0], 640*480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.c[2];
                refimg->data.y[640*480 + 2 * i + 1] = vals.c[1];
            }
            break;
        case VX_DF_IMAGE_YUYV:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.yuyv[i].y0 = vals.c[0];
                refimg->data.yuyv[i].y1 = vals.c[0];
                refimg->data.yuyv[i].u = vals.c[1];
                refimg->data.yuyv[i].v = vals.c[2];
            }
            break;
        case VX_DF_IMAGE_UYVY:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.uyvy[i].y0 = vals.c[0];
                refimg->data.uyvy[i].y1 = vals.c[0];
                refimg->data.uyvy[i].u = vals.c[1];
                refimg->data.uyvy[i].v = vals.c[2];
            }
            break;
    };

    EXPECT_EQ_CTIMAGE(refimg, ctimg);

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);
}

static void mem_free(void**ptr)
{
    free(*ptr);
    *ptr = 0;
}

TEST(Image, testComputeImagePatchSize)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_uint8 val = 0xAB;
    vx_size memsz;
    vx_size count_pixels = 0;
    int i, j;
    vx_uint8* buffer;
    vx_uint8* buffer0;
    vx_rectangle_t rect = {0,0,640,480};
    vx_imagepatch_addressing_t addr = {640, 480, 1, 640, VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1};

    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, 640, 480, VX_DF_IMAGE_U8, &val), VX_TYPE_IMAGE);

    memsz = vxComputeImagePatchSize(image, &rect, 0);
    ASSERT(memsz >= 640*480);

    ASSERT(buffer = malloc(memsz));
    CT_RegisterForGarbageCollection(buffer, mem_free, CT_GC_OBJECT);
    buffer0 = buffer;

    // copy image data to our buffer
    VX_CALL(vxAccessImagePatch(image, &rect, 0, &addr, (void**)&buffer, VX_READ_ONLY));
    ASSERT_EQ_PTR(buffer0, buffer);
    // release reader
    VX_CALL(vxCommitImagePatch(image, 0, 0, &addr, buffer));
    ASSERT_EQ_PTR(buffer0, buffer);

    for (i = 0; i < 640; ++i)
    {
        for (j = 0; j < 480; ++j)
        {
            vx_uint8* ptr = (vx_uint8*)vxFormatImagePatchAddress2d(buffer, i, j, &addr);

            // no out-of-bound access
            ASSERT(ptr >= buffer && (vx_size)(ptr - buffer) < memsz);

            count_pixels += *ptr == val;
        }
    }

    ASSERT_EQ_INT(640*480, count_pixels);

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);
}

#define IMAGE_SIZE_X 320
#define IMAGE_SIZE_Y 200
#define PATCH_SIZE_X 33
#define PATCH_SIZE_Y 12
#define PATCH_ORIGIN_X 51
#define PATCH_ORIGIN_Y 15

TEST(Image, testAccessCopyWrite)
{
    vx_context context = context_->vx_context_;
    vx_uint8 *localPatchDense = malloc(PATCH_SIZE_X*PATCH_SIZE_Y*sizeof(vx_uint8));
    vx_uint8 *localPatchSparse = malloc(PATCH_SIZE_X*PATCH_SIZE_Y*3*3*sizeof(vx_uint8));
    vx_image image;
    int x, y;

    ASSERT_VX_OBJECT( image = vxCreateImage(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    /* Image Initialization */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxAccessImagePatch(image, &rectFull, 0, &addrFull, (void **)&p, VX_WRITE_ONLY) );
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                *pPixel = 0;
            }
        }
        VX_CALL( vxCommitImagePatch(image, &rectFull, 0, &addrFull, p) );

        /* Buffer Initialization */
        for (y = 0; y < PATCH_SIZE_Y; y++) {
            for (x = 0; x < PATCH_SIZE_X; x++) {
                localPatchDense[x + y*PATCH_SIZE_X] = x + y;

                localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] = 2*(x + y);
                localPatchSparse[(3*x+1) + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+1) + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+1) + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + (3*y+2)*3*PATCH_SIZE_X] = 0;
            }
        }
    }

    /* Write, COPY, No spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                sizeof(vx_uint8), PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchDense[0];
        VX_CALL( vxAccessImagePatch(image, &rectPatch, 0, &addrPatch, (void **)&p, VX_WRITE_ONLY) );
        ASSERT(p == &localPatchDense[0]);
        VX_CALL( vxCommitImagePatch(image, &rectPatch, 0, &addrPatch, p) );
    }
    /* Check (MAP) */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxAccessImagePatch(image, &rectFull, 0, &addrFull, (void **)&p, VX_WRITE_ONLY) );
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                if ( (x<PATCH_ORIGIN_X) || (x>=PATCH_ORIGIN_X+PATCH_SIZE_X) ||
                     (y<PATCH_ORIGIN_Y) || (y>=PATCH_ORIGIN_Y+PATCH_SIZE_Y) ) {
                    ASSERT( *pPixel == 0);
                }
                else {
                    ASSERT( *pPixel == (x + y - PATCH_ORIGIN_X - PATCH_ORIGIN_Y));
                }
            }
        }
        VX_CALL( vxCommitImagePatch(image, &rectFull, 0, &addrFull, p) );
    }


    /* Write, COPY, Spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                3*sizeof(vx_uint8), 3*3*PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchSparse[0];
        VX_CALL( vxAccessImagePatch(image, &rectPatch, 0, &addrPatch, (void **)&p, VX_WRITE_ONLY) );
        ASSERT(p == &localPatchSparse[0]);
        VX_CALL( vxCommitImagePatch(image, &rectPatch, 0, &addrPatch, p) );
    }
    /* Check (MAP) */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxAccessImagePatch(image, &rectFull, 0, &addrFull, (void **)&p, VX_WRITE_ONLY) );
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                if ( (x<PATCH_ORIGIN_X) || (x>=PATCH_ORIGIN_X+PATCH_SIZE_X) ||
                     (y<PATCH_ORIGIN_Y) || (y>=PATCH_ORIGIN_Y+PATCH_SIZE_Y) ) {
                    ASSERT( *pPixel == 0);
                }
                else {
                    ASSERT( *pPixel == (2*(x + y - PATCH_ORIGIN_X - PATCH_ORIGIN_Y)));
                }
            }
        }
        VX_CALL( vxCommitImagePatch(image, &rectFull, 0, &addrFull, p) );
    }



    VX_CALL( vxReleaseImage(&image) );
    ASSERT( image == 0);

    free(localPatchDense);
    free(localPatchSparse);
}

TEST(Image, testAccessCopyRead)
{
    vx_context context = context_->vx_context_;
    vx_uint8 *localPatchDense = malloc(PATCH_SIZE_X*PATCH_SIZE_Y*sizeof(vx_uint8));
    vx_uint8 *localPatchSparse = malloc(PATCH_SIZE_X*PATCH_SIZE_Y*3*3*sizeof(vx_uint8));
    vx_image image;
    int x, y;

    ASSERT_VX_OBJECT( image = vxCreateImage(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    /* Image Initialization */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxAccessImagePatch(image, &rectFull, 0, &addrFull, (void **)&p, VX_WRITE_ONLY) );
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                *pPixel = x + y;
            }
        }
        VX_CALL( vxCommitImagePatch(image, &rectFull, 0, &addrFull, p) );

        /* Buffer Initialization */
        for (y = 0; y < PATCH_SIZE_Y; y++) {
            for (x = 0; x < PATCH_SIZE_X; x++) {
                localPatchDense[x + y*PATCH_SIZE_X] = 0;

                localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + (3*y+2)*3*PATCH_SIZE_X] = 0;
            }
        }
    }

    /* READ, COPY, No spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                sizeof(vx_uint8), PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchDense[0];
        VX_CALL( vxAccessImagePatch(image, &rectPatch, 0, &addrPatch, (void **)&p, VX_READ_ONLY) );
        ASSERT(p == &localPatchDense[0]);
        ASSERT(addrPatch.stride_x == sizeof(vx_uint8));
        ASSERT(addrPatch.stride_y == PATCH_SIZE_X*sizeof(vx_uint8));
        VX_CALL( vxCommitImagePatch(image, &rectPatch, 0, &addrPatch, p) );
    }
    /* Check */
    for (y = 0; y < PATCH_SIZE_Y; y++) {
        for (x = 0; x < PATCH_SIZE_X; x++) {
            ASSERT(localPatchDense[x + y*PATCH_SIZE_X] == x + y + PATCH_ORIGIN_X + PATCH_ORIGIN_Y);
        }
    }

    /* READ, COPY, Spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                3*sizeof(vx_uint8), 3*3*PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchSparse[0];
        VX_CALL( vxAccessImagePatch(image, &rectPatch, 0, &addrPatch, (void **)&p, VX_READ_ONLY) );
        ASSERT(p == &localPatchSparse[0]);
        ASSERT(addrPatch.stride_x == 3*sizeof(vx_uint8));
        ASSERT(addrPatch.stride_y == 3*3*PATCH_SIZE_X*sizeof(vx_uint8));
        VX_CALL( vxCommitImagePatch(image, &rectPatch, 0, &addrPatch, p) );
    }
    /* Check */
    for (y = 0; y < PATCH_SIZE_Y; y++) {
        for (x = 0; x < PATCH_SIZE_X; x++) {
            ASSERT(localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] == x + y + PATCH_ORIGIN_X + PATCH_ORIGIN_Y);
            ASSERT(localPatchSparse[3*x+1 + 3*y*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + 3*y*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+1 + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+1 + (3*y+2)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + (3*y+2)*3*PATCH_SIZE_X] == 0);
        }
    }

    VX_CALL( vxReleaseImage(&image) );
    ASSERT( image == 0);

    free(localPatchDense);
    free(localPatchSparse);
}


TESTCASE_TESTS(Image,
    testRngImageCreation,
    testVirtualImageCreation,
    testVirtualImageCreationDims,
    testCreateImageFromHandle,
    testFormatImagePatchAddress1d,
    testConvert_CT_Image,
    testUniformImage,
    testComputeImagePatchSize,
    testAccessCopyWrite,
    testAccessCopyRead
    )
