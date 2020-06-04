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

#define VALID_SHIFT_MIN 0
// #define VALID_SHIFT_MIN -64
#define VALID_SHIFT_MAX 7

#define CT_EXECUTE_ASYNC

static void referenceConvertDepth(CT_Image src, CT_Image dst, int shift, vx_enum policy)
{
    uint32_t i, j;

    ASSERT(src && dst);
    ASSERT(src->width == dst->width);
    ASSERT(src->height == dst->height);
    ASSERT((src->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16) || (src->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_U8));
    ASSERT(policy == VX_CONVERT_POLICY_WRAP || policy == VX_CONVERT_POLICY_SATURATE);

    if (shift > 16) shift = 16;
    if (shift < -16) shift = -16;

    if (src->format == VX_DF_IMAGE_U8)
    {
        // according to spec the policy is ignored
        // if (policy == VX_CONVERT_POLICY_WRAP)
        {
            // up-conversion + wrap
            if (shift < 0)
            {
                for (i = 0; i < dst->height; ++i)
                    for (j = 0; j < dst->width; ++j)
                        dst->data.s16[i * dst->stride + j] = ((unsigned)src->data.y[i * src->stride + j]) >> (-shift);
            }
            else
            {
                for (i = 0; i < dst->height; ++i)
                    for (j = 0; j < dst->width; ++j)
                        dst->data.s16[i * dst->stride + j] = ((unsigned)src->data.y[i * src->stride + j]) << shift;
            }
        }
        // else if (VX_CONVERT_POLICY_SATURATE)
        // {
        //     // up-conversion + saturate
        //     if (shift < 0)
        //     {
        //         for (i = 0; i < dst->height; ++i)
        //             for (j = 0; j < dst->width; ++j)
        //                 dst->data.s16[i * dst->stride + j] = ((unsigned)src->data.y[i * src->stride + j]) >> (-shift);
        //     }
        //     else
        //     {
        //         for (i = 0; i < dst->height; ++i)
        //             for (j = 0; j < dst->width; ++j)
        //             {
        //                 unsigned v = ((unsigned)src->data.y[i * src->stride + j]) << shift;
        //                 if (v > 32767) v = 32767;
        //                 dst->data.s16[i * dst->stride + j] = v;
        //             }
        //     }
        // }
    }
    else if (policy == VX_CONVERT_POLICY_WRAP)
    {
        // down-conversion + wrap
        if (shift < 0)
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    dst->data.y[i * dst->stride + j] = src->data.s16[i * src->stride + j] << (-shift);
        }
        else
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                    dst->data.y[i * dst->stride + j] = src->data.s16[i * src->stride + j] >> shift;
        }
    }
    else if (policy == VX_CONVERT_POLICY_SATURATE)
    {
        // down-conversion + saturate
        if (shift < 0)
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                {
                    int32_t v = src->data.s16[i * src->stride + j] << (-shift);
                    if (v > 255) v = 255;
                    if (v < 0) v = 0;
                    dst->data.y[i * dst->stride + j] = v;
                }
        }
        else
        {
            for (i = 0; i < dst->height; ++i)
                for (j = 0; j < dst->width; ++j)
                {
                    int32_t v = src->data.s16[i * src->stride + j] >> shift;
                    if (v > 255) v = 255;
                    if (v < 0) v = 0;
                    dst->data.y[i * dst->stride + j] = v;
                }
        }
    }
}

static void fillSquence(CT_Image dst, uint32_t seq_init)
{
    uint32_t i, j;
    uint32_t val = seq_init;

    ASSERT(dst);
    ASSERT(dst->format == VX_DF_IMAGE_U8 || dst->format == VX_DF_IMAGE_S16);

    if (dst->format == VX_DF_IMAGE_U8)
    {
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
                dst->data.y[i * dst->stride + j] = ++val;
    }
    else
    {
        for (i = 0; i < dst->height; ++i)
            for (j = 0; j < dst->width; ++j)
                dst->data.s16[i * dst->stride + j] = ++val;
    }
}

TESTCASE(vxuConvertDepth, CT_VXContext, ct_setup_vx_context, 0)
TESTCASE(vxConvertDepth,  CT_VXContext, ct_setup_vx_context, 0)


TEST(vxuConvertDepth, NegativeSizes)
{
    vx_image img16x88, img88x16, img16x16;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(img16x88 = vxCreateImage(context, 16, 88, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x16 = vxCreateImage(context, 88, 16, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x16 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(img16x88, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img88x16, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(img16x16, &CT()->seed_));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x88, img88x16, VX_CONVERT_POLICY_SATURATE, 0));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x88, img88x16, VX_CONVERT_POLICY_WRAP, 0));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x88, img88x16, VX_CONVERT_POLICY_SATURATE, 1));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img16x88, img88x16, VX_CONVERT_POLICY_WRAP, 1));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x16, img16x16, VX_CONVERT_POLICY_SATURATE, 0));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x16, img16x16, VX_CONVERT_POLICY_WRAP, 0));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x16, img16x16, VX_CONVERT_POLICY_SATURATE, 1));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, img88x16, img16x16, VX_CONVERT_POLICY_WRAP, 1));

    vxReleaseImage(&img16x88);
    vxReleaseImage(&img88x16);
    vxReleaseImage(&img16x16);
}

TEST(vxConvertDepth, NegativeSizes)
{
    vx_image img16x88, img88x16, img16x16;
    vx_graph graph;
    vx_node node;
    vx_scalar shift;
    vx_int32 sh = 1;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(shift = vxCreateScalar(context, VX_TYPE_INT32, &sh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(img16x88 = vxCreateImage(context, 16, 88, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img88x16 = vxCreateImage(context, 88, 16, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(img16x16 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8),  VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img16x88, img88x16, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img16x88, img88x16, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x16, img16x16, VX_CONVERT_POLICY_SATURATE, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, img88x16, img16x16, VX_CONVERT_POLICY_WRAP, shift), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    vxReleaseImage(&img16x88);
    vxReleaseImage(&img88x16);
    vxReleaseImage(&img16x16);
    vxReleaseScalar(&shift);
}

typedef struct {
    const char* name;
    uint32_t width;
    uint32_t height;
    vx_df_image format_from;
    vx_df_image format_to;
    vx_enum policy;
} cvt_depth_arg;

#define CVT_ARG(w,h,from,to,p) ARG(#p"/"#w"x"#h" "#from"->"#to, w, h, VX_DF_IMAGE_##from, VX_DF_IMAGE_##to, VX_CONVERT_POLICY_##p)

#define PREPEND_SIZE(macro, ...)                \
    CT_EXPAND(macro(1, 1, __VA_ARGS__)),        \
    CT_EXPAND(macro(15, 17, __VA_ARGS__)),      \
    CT_EXPAND(macro(32, 32, __VA_ARGS__)),      \
    CT_EXPAND(macro(640, 480, __VA_ARGS__)),    \
    CT_EXPAND(macro(1231, 1234, __VA_ARGS__))

    /*,
    CT_EXPAND(macro(1280, 720, __VA_ARGS__)),
    CT_EXPAND(macro(1920, 1080, __VA_ARGS__))*/

#define CVT_ARGS                                \
    PREPEND_SIZE(CVT_ARG, U8, S16, SATURATE),   \
    PREPEND_SIZE(CVT_ARG, U8, S16, WRAP),       \
    PREPEND_SIZE(CVT_ARG, S16, U8, SATURATE),   \
    PREPEND_SIZE(CVT_ARG, S16, U8, WRAP)

TEST_WITH_ARG(vxuConvertDepth, BitExact, cvt_depth_arg, CVT_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, refdst, vxdst;
    vx_int32 shift;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, arg_->format_from);
        fillSquence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    ASSERT_VX_OBJECT(dst = vxCreateImage(context, arg_->width, arg_->height, arg_->format_to), VX_TYPE_IMAGE);

    for (shift = VALID_SHIFT_MIN; shift <= VALID_SHIFT_MAX; ++shift)
    {
        ct_update_progress(shift - VALID_SHIFT_MIN, VALID_SHIFT_MAX - VALID_SHIFT_MIN + 1);
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxuConvertDepth(context, src, dst, arg_->policy, shift));

        ASSERT_NO_FAILURE({
            vxdst = ct_image_from_vx_image(dst);
            refdst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
            referenceConvertDepth(ref_src, refdst, shift, arg_->policy);
        });

        EXPECT_EQ_CTIMAGE(refdst, vxdst);
        if (CT_HasFailure())
        {
            printf("Shift value is %d\n", shift);
            break;
        }
    }

    // checked release vx images
    vxReleaseImage(&dst);
    vxReleaseImage(&src);
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src);
}

TEST_WITH_ARG(vxConvertDepth, BitExact, cvt_depth_arg, CVT_ARGS)
{
    vx_image src, dst;
    CT_Image ref_src, refdst, vxdst;
    vx_graph graph;
    vx_node node;
    vx_scalar scalar_shift;
    vx_int32 shift = 0;
    vx_int32 tmp = 0;
    vx_context context = context_->vx_context_;

    ASSERT_NO_FAILURE({
        ref_src = ct_allocate_image(arg_->width, arg_->height, arg_->format_from);
        fillSquence(ref_src, (uint32_t)CT()->seed_);
        src = ct_image_to_vx_image(ref_src, context);
    });

    ASSERT_VX_OBJECT(dst = vxCreateImage(context, arg_->width, arg_->height, arg_->format_to), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scalar_shift = vxCreateScalar(context, VX_TYPE_INT32, &tmp), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxConvertDepthNode(graph, src, dst, arg_->policy, scalar_shift), VX_TYPE_NODE);

    for (shift = VALID_SHIFT_MIN; shift <= VALID_SHIFT_MAX; ++shift)
    {
        ct_update_progress(shift - VALID_SHIFT_MIN, VALID_SHIFT_MAX - VALID_SHIFT_MIN + 1);
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxWriteScalarValue(scalar_shift, &shift));

        // run graph
#ifdef CT_EXECUTE_ASYNC
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
        EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

        ASSERT_NO_FAILURE({
            vxdst = ct_image_from_vx_image(dst);
            refdst = ct_allocate_image(arg_->width, arg_->height, arg_->format_to);
            referenceConvertDepth(ref_src, refdst, shift, arg_->policy);
        });

        EXPECT_EQ_CTIMAGE(refdst, vxdst);
        if (CT_HasFailure())
        {
            printf("Shift value is %d\n", shift);
            break;
        }
    }

    vxReleaseImage(&dst);
    vxReleaseImage(&src);
    vxReleaseScalar(&scalar_shift);
    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
}

TESTCASE_TESTS(vxuConvertDepth, DISABLED_NegativeSizes, BitExact)
TESTCASE_TESTS(vxConvertDepth,  DISABLED_NegativeSizes, BitExact)
