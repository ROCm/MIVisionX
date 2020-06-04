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
#include <VX/vxu.h>


TESTCASE(ChannelCombine, CT_VXContext, ct_setup_vx_context, 0)


TEST(ChannelCombine, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src1_image = 0, src2_image = 0, src3_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src1_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src3_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_RGB), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxChannelCombineNode(graph, src1_image, src2_image, src3_image, NULL, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src1_image);
    vxReleaseImage(&src2_image);
    vxReleaseImage(&src3_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0);
    ASSERT(src1_image == 0);
    ASSERT(src2_image == 0);
    ASSERT(src3_image == 0);
}


static CT_Image channel_combine_image_generate_random(int width, int height, vx_df_image format)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
}


static void channel_combine_fill_chanel(CT_Image src, vx_enum channel, CT_Image dst)
{
    uint8_t *dst_base = NULL;
    int x, y;

    int plane, component;

    int x_subsampling = ct_image_get_channel_subsampling_x(dst, channel);
    int y_subsampling = ct_image_get_channel_subsampling_y(dst, channel);

    int xstep = ct_image_get_channel_step_x(dst, channel);
    int ystep = ct_image_get_channel_step_y(dst, channel);

    int src_width = dst->width / x_subsampling;
    int src_height = dst->height / y_subsampling;

    // Check that src was subsampled (by spec)
    ASSERT_EQ_INT(src_width, src->width);
    ASSERT_EQ_INT(src_height, src->height);

    ASSERT_NO_FAILURE(plane = ct_image_get_channel_plane(dst, channel));
    ASSERT_NO_FAILURE(component = ct_image_get_channel_component(dst, channel));

    ASSERT(dst_base = ct_image_get_plane_base(dst, plane));

    for (y = 0; y < src_height; y++)
    {
        for (x = 0; x < src_width; x++)
        {
            uint8_t *src_data = CT_IMAGE_DATA_PTR_8U(src, x, y);
            uint8_t *dst_data = dst_base + (x * xstep) + (y * ystep);
            dst_data[component] = *src_data;
        }
    }

    return;
}


static CT_Image channel_combine_create_reference_image(CT_Image src1, CT_Image src2, CT_Image src3, CT_Image src4, vx_df_image format)
{
    CT_Image dst = NULL;

    ASSERT_(return NULL, src1);
    ASSERT_NO_FAILURE_(return NULL, dst = ct_allocate_image(src1->width, src1->height, format));

    switch (format)
    {
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
            ASSERT_(return NULL, src1);
            ASSERT_(return NULL, src2);
            ASSERT_(return NULL, src3);
            if (format == VX_DF_IMAGE_RGB)
                ASSERT_(return NULL, src4 == NULL);
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src1, VX_CHANNEL_R, dst));
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src2, VX_CHANNEL_G, dst));
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src3, VX_CHANNEL_B, dst));
            if (format == VX_DF_IMAGE_RGBX)
                ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src4, VX_CHANNEL_A, dst));
            return dst;
        case VX_DF_IMAGE_NV12:
        case VX_DF_IMAGE_NV21:
        case VX_DF_IMAGE_UYVY:
        case VX_DF_IMAGE_YUYV:
        case VX_DF_IMAGE_IYUV:
        case VX_DF_IMAGE_YUV4:
            ASSERT_(return NULL, src1);
            ASSERT_(return NULL, src2);
            ASSERT_(return NULL, src3);
            ASSERT_(return NULL, src4 == NULL);
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src1, VX_CHANNEL_Y, dst));
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src2, VX_CHANNEL_U, dst));
            ASSERT_NO_FAILURE_(return NULL, channel_combine_fill_chanel(src3, VX_CHANNEL_V, dst));
            return dst;
    }

    CT_FAIL_(return NULL, "Not supported");
}

static void channel_combine_check(CT_Image src1, CT_Image src2, CT_Image src3, CT_Image src4, CT_Image dst)
{
    CT_Image dst_ref = NULL;

    ASSERT_NO_FAILURE(dst_ref = channel_combine_create_reference_image(src1, src2, src3, src4, dst->format));

    EXPECT_EQ_CTIMAGE(dst_ref, dst);
#if 0
    if (CT_HasFailure())
    {
#define DUMP_SRC(i) \
        if (src##i) \
        { \
            printf("=== SRC" #i " ===\n"); \
            ct_dump_image_info(src##i); \
        }
        DUMP_SRC(1) DUMP_SRC(2) DUMP_SRC(3) DUMP_SRC(4)
#undef DUMP_SRC
        printf("=== DST ===\n");
        ct_dump_image_info(dst);
        printf("=== EXPECTED ===\n");
        ct_dump_image_info(dst_ref);
    }
#endif
}

typedef struct {
    const char* testName;
    vx_df_image dst_format;
    int width, height;
} Arg;

#define ADD_CASE(testArgName, nextmacro, format, ...) \
    CT_EXPAND(nextmacro(testArgName "/" #format, __VA_ARGS__, format))

#define ADD_CASES(testArgName, nextmacro, ...) \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGB, __VA_ARGS__),  \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGBX, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV12, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV21, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_UYVY, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUYV, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_IYUV, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUV4, __VA_ARGS__), \


#define ADD_SIZE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=16x16", __VA_ARGS__, 16, 16))

#define ChannelCombine_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_CASES, ADD_SIZE, ARG)

TEST_WITH_ARG(ChannelCombine, testGraphProcessing, Arg,
    ChannelCombine_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image[4] = {0, 0, 0, 0};
    vx_image dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    int channels = 0, i;
    CT_Image src[4] = {NULL, NULL, NULL, NULL};
    CT_Image dst = NULL, dst_dummy = NULL;
    vx_enum channel_ref;

    ASSERT_NO_FAILURE(dst_dummy = ct_allocate_image(4, 4, arg_->dst_format));

    ASSERT_NO_FAILURE(channels = ct_get_num_channels(arg_->dst_format));
    channel_ref = (arg_->dst_format==VX_DF_IMAGE_RGB)||(arg_->dst_format==VX_DF_IMAGE_RGBX)?VX_CHANNEL_R:VX_CHANNEL_Y;
    for (i = 0; i < channels; i++)
    {
        int w = arg_->width / ct_image_get_channel_subsampling_x(dst_dummy, channel_ref + i);
        int h = arg_->height / ct_image_get_channel_subsampling_y(dst_dummy, channel_ref + i);
        ASSERT_NO_FAILURE(src[i] = channel_combine_image_generate_random(w, h, VX_DF_IMAGE_U8));
        ASSERT_VX_OBJECT(src_image[i] = ct_image_to_vx_image(src[i], context), VX_TYPE_IMAGE);
    }

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, arg_->width, arg_->height, arg_->dst_format), VX_TYPE_IMAGE);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    node = vxChannelCombineNode(graph, src_image[0], src_image[1], src_image[2], src_image[3], dst_image);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(channel_combine_check(src[0], src[1], src[2], src[3], dst));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&dst_image);
    ASSERT(dst_image == 0);

    for (i = 0; i < channels; i++)
    {
        vxReleaseImage(&src_image[i]);
        ASSERT(src_image[i] == 0);
    }
}

TEST_WITH_ARG(ChannelCombine, testImmediateProcessing, Arg,
    ChannelCombine_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image[4] = {0, 0, 0, 0};
    vx_image dst_image = 0;

    int channels = 0, i;
    CT_Image src[4] = {NULL, NULL, NULL, NULL};
    CT_Image dst = NULL, dst_dummy = NULL;
    vx_enum channel_ref;

    ASSERT_NO_FAILURE(dst_dummy = ct_allocate_image(4, 4, arg_->dst_format));

    ASSERT_NO_FAILURE(channels = ct_get_num_channels(arg_->dst_format));
    channel_ref = (arg_->dst_format==VX_DF_IMAGE_RGB)||(arg_->dst_format==VX_DF_IMAGE_RGBX)?VX_CHANNEL_R:VX_CHANNEL_Y;
    for (i = 0; i < channels; i++)
    {
        int w = arg_->width / ct_image_get_channel_subsampling_x(dst_dummy, channel_ref + i);
        int h = arg_->height / ct_image_get_channel_subsampling_y(dst_dummy, channel_ref + i);
        ASSERT_NO_FAILURE(src[i] = channel_combine_image_generate_random(w, h, VX_DF_IMAGE_U8));
        ASSERT_VX_OBJECT(src_image[i] = ct_image_to_vx_image(src[i], context), VX_TYPE_IMAGE);
    }

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, arg_->width, arg_->height, arg_->dst_format), VX_TYPE_IMAGE);

    VX_CALL(vxuChannelCombine(context, src_image[0], src_image[1], src_image[2], src_image[3], dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(channel_combine_check(src[0], src[1], src[2], src[3], dst));

    vxReleaseImage(&dst_image);
    ASSERT(dst_image == 0);

    for (i = 0; i < channels; i++)
    {
        vxReleaseImage(&src_image[i]);
        ASSERT(src_image[i] == 0);
    }
}

TESTCASE_TESTS(ChannelCombine,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
