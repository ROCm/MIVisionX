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


TESTCASE(ChannelExtract, CT_VXContext, ct_setup_vx_context, 0)


TEST(ChannelExtract, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_RGB), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxChannelExtractNode(graph, src_image, VX_CHANNEL_0, dst_image), VX_TYPE_NODE);

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}


static CT_Image channel_extract_image_generate_random(int width, int height, vx_df_image format)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
}


static void channel_extract_plane(CT_Image src, vx_enum channel, CT_Image* dst)
{
    uint8_t *src_base = NULL;
    int x, y;

    int plane, component;

    int x_subsampling = ct_image_get_channel_subsampling_x(src, channel);
    int y_subsampling = ct_image_get_channel_subsampling_y(src, channel);

    int xstep = ct_image_get_channel_step_x(src, channel);
    int ystep = ct_image_get_channel_step_y(src, channel);

    int dst_width = src->width / x_subsampling;
    int dst_height = src->height / y_subsampling;

    ASSERT_NO_FAILURE(plane = ct_image_get_channel_plane(src, channel));
    ASSERT_NO_FAILURE(component = ct_image_get_channel_component(src, channel));

    ASSERT(src_base = ct_image_get_plane_base(src, plane));

    ASSERT_NO_FAILURE(*dst = ct_allocate_image(dst_width, dst_height, VX_DF_IMAGE_U8));

    for (y = 0; y < dst_height; y++)
    {
        for (x = 0; x < dst_width; x++)
        {
            uint8_t* dst_data = CT_IMAGE_DATA_PTR_8U(*dst, x, y);
            uint8_t *src_data = src_base + (x * xstep) + (y * ystep);
            *dst_data = src_data[component];
        }
    }

    return;
}


static CT_Image channel_extract_create_reference_image(CT_Image src, vx_enum channelNum)
{
    CT_Image dst = NULL;

    ASSERT_NO_FAILURE_(return NULL, channel_extract_plane(src, channelNum, &dst));

    ASSERT_(return NULL, dst);
    return dst;
}


typedef struct {
    const char* testName;
    vx_df_image format;
    vx_enum channel;
    int width, height;
} Arg;

#define ADD_CASE(testArgName, nextmacro, format, channel, ...) \
    CT_EXPAND(nextmacro(testArgName "/" #format "/" #channel, __VA_ARGS__, format, channel))

#define ADD_CASES(testArgName, nextmacro, ...) \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGB, VX_CHANNEL_R, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGB, VX_CHANNEL_G, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGB, VX_CHANNEL_B, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGBX, VX_CHANNEL_R, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGBX, VX_CHANNEL_G, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGBX, VX_CHANNEL_B, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_RGBX, VX_CHANNEL_A, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV12, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV12, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV12, VX_CHANNEL_V, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV21, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV21, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_NV21, VX_CHANNEL_V, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_UYVY, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_UYVY, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_UYVY, VX_CHANNEL_V, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUYV, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUYV, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUYV, VX_CHANNEL_V, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_IYUV, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_IYUV, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_IYUV, VX_CHANNEL_V, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUV4, VX_CHANNEL_Y, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUV4, VX_CHANNEL_U, __VA_ARGS__), \
    ADD_CASE(testArgName, nextmacro, VX_DF_IMAGE_YUV4, VX_CHANNEL_V, __VA_ARGS__)


#define ADD_SIZE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/sz=16x16", __VA_ARGS__, 16, 16))

#define ChannelExtract_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_CASES, ADD_SIZE, ARG)

TEST_WITH_ARG(ChannelExtract, testGraphProcessing, Arg,
    ChannelExtract_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image src = NULL, dst = NULL;
    CT_Image dst_ref = NULL;

    ASSERT_NO_FAILURE(src = channel_extract_image_generate_random(arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(dst_ref = channel_extract_create_reference_image(src, arg_->channel));

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, dst_ref->width, dst_ref->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    node = vxChannelExtractNode(graph, src_image, arg_->channel, dst_image);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    EXPECT_EQ_CTIMAGE(dst_ref, dst);
#if 0
    if (CT_HasFailure())
    {
        printf("=== SRC ===\n");
        ct_dump_image_info(src);
        printf("=== DST ===\n");
        ct_dump_image_info(dst);
        printf("=== EXPECTED ===\n");
        ct_dump_image_info(dst_ref);
    }
#endif

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(ChannelExtract, testImmediateProcessing, Arg,
    ChannelExtract_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;

    CT_Image src = NULL, dst = NULL;
    CT_Image dst_ref = NULL;

    ASSERT_NO_FAILURE(src = channel_extract_image_generate_random(arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(dst_ref = channel_extract_create_reference_image(src, arg_->channel));

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, dst_ref->width, dst_ref->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    VX_CALL(vxuChannelExtract(context, src_image, arg_->channel, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    EXPECT_EQ_CTIMAGE(dst_ref, dst);
#if 0
    if (CT_HasFailure())
    {
        printf("=== SRC ===\n");
        ct_dump_image_info(src);
        printf("=== DST ===\n");
        ct_dump_image_info(dst);
        printf("=== EXPECTED ===\n");
        ct_dump_image_info(dst_ref);
    }
#endif

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TESTCASE_TESTS(ChannelExtract,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
