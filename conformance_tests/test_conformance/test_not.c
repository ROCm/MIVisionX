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

//#define CT_EXECUTE_ASYNC

static void referenceNot(CT_Image src, CT_Image dst)
{
    uint32_t i, j;

    ASSERT(src && dst);
    ASSERT(src->width == dst->width);
    ASSERT(src->height == dst->height);
    ASSERT(src->format == dst->format && src->format == VX_DF_IMAGE_U8);

    for (i = 0; i < dst->height; ++i)
        for (j = 0; j < dst->width; ++j)
            dst->data.y[i * dst->stride + j] = ~src->data.y[i * src->stride + j];
}

static void fillSquence(CT_Image dst, uint32_t seq_init)
{
    uint32_t i, j;
    uint32_t val = seq_init;

    ASSERT(dst);
    ASSERT(dst->format == VX_DF_IMAGE_U8);

    for (i = 0; i < dst->height; ++i)
        for (j = 0; j < dst->width; ++j)
            dst->data.y[i * dst->stride + j] = ++val;
}

TESTCASE(vxuNot, CT_VXContext, ct_setup_vx_context, 0)
TESTCASE(vxNot,  CT_VXContext, ct_setup_vx_context, 0)


TEST(vxuNot, testNegativeSizes)
{
    vx_image src16x88, dst88x16;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src16x88 = vxCreateImage(context, 16, 88, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst88x16 = vxCreateImage(context, 88, 16, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(src16x88, &CT()->seed_));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuNot(context, src16x88, dst88x16));

    vxReleaseImage(&src16x88);
    vxReleaseImage(&dst88x16);
}

TEST(vxNot, testNegativeSizes)
{
    vx_image src16x88, dst88x16;
    vx_graph graph;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src16x88 = vxCreateImage(context, 16, 88, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst88x16 = vxCreateImage(context, 88, 16, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(vxNotNode(graph, src16x88, dst88x16), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));

    vxReleaseImage(&src16x88);
    vxReleaseImage(&dst88x16);
    vxReleaseGraph(&graph);
}

static vx_image inference_image;
static vx_action VX_CALLBACK inference_image_test(vx_node node)
{
    vx_uint32 width  = 0;
    vx_uint32 height = 0;
    vx_df_image format = 0;

    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(inference_image, VX_IMAGE_ATTRIBUTE_WIDTH,   &width,   sizeof(width)));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(inference_image, VX_IMAGE_ATTRIBUTE_HEIGHT,  &height,  sizeof(height)));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(inference_image, VX_IMAGE_ATTRIBUTE_FORMAT,  &format,  sizeof(format)));

    EXPECT_EQ_INT(640, width);
    EXPECT_EQ_INT(480, height);
    EXPECT_EQ_INT(VX_DF_IMAGE_U8, format);

    return VX_ACTION_CONTINUE;
}

TEST(vxNot, testInference)
{
    vx_image src, dst, gr;
    vx_graph graph;
    vx_node n, tmp;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(src   = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst   = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(n     = vxNotNode(graph, src, dst), VX_TYPE_NODE);

    // grounding
    ASSERT_VX_OBJECT(gr    = vxCreateImage(context, 640, 480, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(tmp   = vxAddNode(graph, dst, src, VX_CONVERT_POLICY_WRAP, gr), VX_TYPE_NODE);

    // test
    inference_image = dst;
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxAssignNodeCallback(n, inference_image_test));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));

    vxReleaseNode(&n);
    vxReleaseNode(&tmp);
    vxReleaseImage(&src);
    vxReleaseImage(&dst);
    vxReleaseImage(&gr);
    vxReleaseGraph(&graph);
}

typedef struct {
    const char* name;
    uint32_t width;
    uint32_t height;
} size_arg;

#define SIZE_ARG(w,h) ARG(#w "x" #h, w, h)

#define NOT_SIZE_ARGS       \
    SIZE_ARG(640, 480),     \
    ARG_EXTENDED_BEGIN(),   \
    SIZE_ARG(1, 1),         \
    SIZE_ARG(15, 17),       \
    SIZE_ARG(32, 32),       \
    SIZE_ARG(1231, 1234),   \
    SIZE_ARG(1280, 720),    \
    SIZE_ARG(1920, 1080),   \
    ARG_EXTENDED_END()

TEST_WITH_ARG(vxuNot, testSizes, size_arg, NOT_SIZE_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, refdst, vxdst;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8);
        fillSquence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    ASSERT_VX_OBJECT(dst = vxCreateImage(context, arg_->width, arg_->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuNot(context, src, dst));

    ASSERT_NO_FAILURE({
        vxdst = ct_image_from_vx_image(dst);
        refdst = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8);
        referenceNot(ref_src, refdst);
    });

    ASSERT_EQ_CTIMAGE(refdst, vxdst);

    // checked release vx images
    vxReleaseImage(&dst);
    vxReleaseImage(&src);
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src);
}

TEST_WITH_ARG(vxNot, testSizes, size_arg, NOT_SIZE_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, refdst, vxdst;
    vx_graph graph;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8);
        fillSquence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    // build one-node graph
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(dst   = vxCreateImage(context, arg_->width, arg_->height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(vxNotNode(graph, src, dst), VX_TYPE_NODE);

    // run graph
#ifdef CT_EXECUTE_ASYNC
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

    ASSERT_NO_FAILURE({
        vxdst = ct_image_from_vx_image(dst);
        refdst = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8);
        referenceNot(ref_src, refdst);
    });

    ASSERT_EQ_CTIMAGE(refdst, vxdst);

    vxReleaseImage(&src);
    vxReleaseImage(&dst);
    vxReleaseGraph(&graph);
}


TESTCASE_TESTS(vxuNot, DISABLED_testNegativeSizes,                testSizes)
TESTCASE_TESTS(vxNot,  DISABLED_testNegativeSizes, testInference, testSizes)

