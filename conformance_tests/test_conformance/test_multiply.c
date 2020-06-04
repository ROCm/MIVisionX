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

#include <math.h>

#ifdef _MSC_VER
#define ONE_255 (1.0f/255)
#else
#define ONE_255 0x1.010102p-8f
#endif
#define ONE_2_0 1.0f
#define ONE_2_1 (1.0f/(1<<1))
#define ONE_2_2 (1.0f/(1<<2))
#define ONE_2_3 (1.0f/(1<<3))
#define ONE_2_4 (1.0f/(1<<4))
#define ONE_2_5 (1.0f/(1<<5))
#define ONE_2_6 (1.0f/(1<<6))
#define ONE_2_7 (1.0f/(1<<7))
#define ONE_2_8 (1.0f/(1<<8))
#define ONE_2_9 (1.0f/(1<<9))
#define ONE_2_10 (1.0f/(1<<10))
#define ONE_2_11 (1.0f/(1<<11))
#define ONE_2_12 (1.0f/(1<<12))
#define ONE_2_13 (1.0f/(1<<13))
#define ONE_2_14 (1.0f/(1<<14))
#define ONE_2_15 (1.0f/(1<<15))

#define ONE_255_STR "(1/255)"
#define ONE_2_0_STR "(1/2^0)"
#define ONE_2_1_STR "(1/2^1)"
#define ONE_2_2_STR "(1/2^2)"
#define ONE_2_3_STR "(1/2^3)"
#define ONE_2_4_STR "(1/2^4)"
#define ONE_2_5_STR "(1/2^5)"
#define ONE_2_6_STR "(1/2^6)"
#define ONE_2_7_STR "(1/2^7)"
#define ONE_2_8_STR "(1/2^8)"
#define ONE_2_9_STR "(1/2^9)"
#define ONE_2_10_STR "(1/2^10)"
#define ONE_2_11_STR "(1/2^11)"
#define ONE_2_12_STR "(1/2^12)"
#define ONE_2_13_STR "(1/2^13)"
#define ONE_2_14_STR "(1/2^14)"
#define ONE_2_15_STR "(1/2^15)"

//#define CT_EXECUTE_ASYNC

static void referenceMultiply(CT_Image src0, CT_Image src1, CT_Image dst, CT_Image dst_plus_1, vx_float32 scale, enum vx_convert_policy_e policy)
{
    int32_t min_bound, max_bound;
    uint32_t i, j;
    ASSERT(src0 && src1 && dst && dst_plus_1);
    ASSERT(src0->width = src1->width && src0->width == dst->width && src0->width == dst_plus_1->width);
    ASSERT(src0->height = src1->height && src0->height == dst->height && src0->height == dst_plus_1->height);
    ASSERT(dst->format == dst_plus_1->format);

    switch (policy)
    {
        case VX_CONVERT_POLICY_SATURATE:
            if (dst->format == VX_DF_IMAGE_U8)
            {
                min_bound = 0;
                max_bound = 255;
            }
            else if (dst->format == VX_DF_IMAGE_S16)
            {
                min_bound = -32768;
                max_bound =  32767;
            }
            else
                FAIL("Unsupported result format: (%.4s)", &dst->format);
            break;
        case VX_CONVERT_POLICY_WRAP:
            min_bound = INT32_MIN;
            max_bound = INT32_MAX;
            break;
        default: FAIL("Unknown owerflow policy"); break;
    };

#define MULTIPLY_LOOP(s0, s1, r)                                                                                \
    do{                                                                                                         \
        for (i = 0; i < dst->height; ++i)                                                                       \
            for (j = 0; j < dst->width; ++j)                                                                    \
            {                                                                                                   \
                int32_t val0 = src0->data.s0[i * src0->stride + j];                                             \
                int32_t val1 = src1->data.s1[i * src1->stride + j];                                             \
                /* use double precision because in S16*S16 case (val0*val1) can be not representable as float */\
                int32_t res0 = (int32_t)floor(((double)(val0 * val1)) * scale);                                 \
                int32_t res1 = res0 + 1;                                                                        \
                dst->data.r[i * dst->stride + j] = (res0 < min_bound ? min_bound :                              \
                                                                        (res0 > max_bound ? max_bound : res0)); \
                dst_plus_1->data.r[i * dst_plus_1->stride + j] = (res1 < min_bound ? min_bound :                \
                                                                        (res1 > max_bound ? max_bound : res1)); \
            }                                                                                                   \
    }while(0)

    if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8)
        MULTIPLY_LOOP(y, y, y);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        MULTIPLY_LOOP(y, y, s16);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        MULTIPLY_LOOP(y, s16, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        MULTIPLY_LOOP(s16, y, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        MULTIPLY_LOOP(s16, s16, s16);
    else
        FAIL("Unsupported combination of argument formats: %.4s + %.4s = %.4s", &src0->format, &src1->format, &dst->format);

#undef MULTIPLY_LOOP
}

typedef struct {
    const char* name;
    enum vx_convert_policy_e overflow_policy;
    int width, height;
    vx_df_image arg1_format, arg2_format, result_format;
    enum vx_round_policy_e round_policy;
    vx_float32 scale;
} formats_arg, fuzzy_arg;

#define FORMATS_ARG(owp, f1, f2, fr, rp, scale)                 \
    ARG(#owp "/" #rp " " #f1 "*" #f2 "*" scale##_STR "=" #fr,   \
        VX_CONVERT_POLICY_##owp, 0, 0, VX_DF_IMAGE_##f1, VX_DF_IMAGE_##f2, VX_DF_IMAGE_##fr, VX_ROUND_POLICY_##rp, scale)

#define FUZZY_ARG(owp, w, h, f1, f2, fr, rp, scale)                         \
    ARG(#owp "/" #rp " " #w "x" #h " " #f1 "*" #f2 "*" scale##_STR "=" #fr, \
        VX_CONVERT_POLICY_##owp, w, h, VX_DF_IMAGE_##f1, VX_DF_IMAGE_##f2, VX_DF_IMAGE_##fr, VX_ROUND_POLICY_##rp, scale)

#define APPEND_SCALE(macro, ...)                                \
    CT_EXPAND(macro(__VA_ARGS__, TO_NEAREST_EVEN, ONE_255)),    \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_0)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_1)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_2)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_3)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_4)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_5)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_6)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_7)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_8)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_9)),            \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_10)),           \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_11)),           \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_12)),           \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_13)),           \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_14)),           \
    CT_EXPAND(macro(__VA_ARGS__, TO_ZERO, ONE_2_15))

#define MUL_INVALID_FORMATS(owp)                    \
    APPEND_SCALE(FORMATS_ARG, owp, S16, S16, U8),   \
    APPEND_SCALE(FORMATS_ARG, owp, S16, U8,  U8),   \
    APPEND_SCALE(FORMATS_ARG, owp, U8,  S16, U8)

#define MUL_INFERENCE_FORMATS(opw)                  \
    APPEND_SCALE(FORMATS_ARG, opw, S16, S16, S16),  \
    APPEND_SCALE(FORMATS_ARG, opw, S16, U8,  S16),  \
    APPEND_SCALE(FORMATS_ARG, opw, U8,  S16, S16),  \
    APPEND_SCALE(FORMATS_ARG, opw, U8,  U8,  S16)

#define MUL_VALID_FORMATS(opw) MUL_INFERENCE_FORMATS(opw), APPEND_SCALE(FORMATS_ARG, opw, U8, U8, U8)

#define MUL_FUZZY_ARGS(owp)                                 \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480, U8, U8, U8),     \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480, U8, U8, S16),    \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480, U8, S16, S16),   \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480, S16, U8, S16),   \
    APPEND_SCALE(FUZZY_ARG, owp, 640, 480, S16, S16, S16),  \
                                                            \
    ARG_EXTENDED_BEGIN(),                                   \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15, U8, U8, U8),       \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15, U8, U8, S16),      \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15, U8, S16, S16),     \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15, S16, U8, S16),     \
    APPEND_SCALE(FUZZY_ARG, owp, 15, 15, S16, S16, S16),    \
                                                            \
    APPEND_SCALE(FUZZY_ARG, owp, 1280, 720, U8, U8, U8),    \
    APPEND_SCALE(FUZZY_ARG, owp, 1280, 720, U8, U8, S16),   \
    APPEND_SCALE(FUZZY_ARG, owp, 1280, 720, U8, S16, S16),  \
    APPEND_SCALE(FUZZY_ARG, owp, 1280, 720, S16, U8, S16),  \
    APPEND_SCALE(FUZZY_ARG, owp, 1280, 720, S16, S16, S16), \
    ARG_EXTENDED_END()

TESTCASE(vxuMultiply, CT_VXContext, ct_setup_vx_context, 0)
TESTCASE(vxMultiply,  CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG(vxuMultiply, testNegativeFormat, formats_arg, MUL_INVALID_FORMATS(SATURATE), MUL_INVALID_FORMATS(WRAP))
{
    vx_image src1, src2, dst;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, 32, 32, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, 32, 32, arg_->arg2_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(src1, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2, &CT()->seed_));

    // The output image can be VX_DF_IMAGE_U8 only if both source images are
    // VX_DF_IMAGE_U8 and the output image is explicitly set to VX_DF_IMAGE_U8. It is
    // otherwise VX_DF_IMAGE_S16.
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1, src2, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst));

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
}

TEST_WITH_ARG(vxMultiply, testNegativeFormat, formats_arg, MUL_INVALID_FORMATS(SATURATE), MUL_INVALID_FORMATS(WRAP))
{
    vx_image src1, src2, dst;
    vx_graph graph;
    vx_scalar scale;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(src1  = vxCreateImage(context, 32, 32, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2  = vxCreateImage(context, 32, 32, arg_->arg2_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst   = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale),   VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(vxMultiplyNode(graph, src1, src2, scale, arg_->overflow_policy, arg_->round_policy, dst), VX_TYPE_NODE);

    // The output image can be VX_DF_IMAGE_U8 only if both source images are
    // VX_DF_IMAGE_U8 and the output image is explicitly set to VX_DF_IMAGE_U8. It is
    // otherwise VX_DF_IMAGE_S16.
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseScalar(&scale);
    vxReleaseGraph(&graph);
}

TEST_WITH_ARG(vxuMultiply, testNegativeSizes, formats_arg, MUL_VALID_FORMATS(SATURATE), MUL_VALID_FORMATS(WRAP))
{
    vx_image src1_32x32, src1_64x64, src2_32x32, src2_32x64, dst32x32, dst88x16;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1_32x32 = vxCreateImage(context, 32, 32, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src1_64x64 = vxCreateImage(context, 64, 64, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x32 = vxCreateImage(context, 32, 32, arg_->arg2_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x64 = vxCreateImage(context, 32, 64, arg_->arg2_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst32x32 = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst88x16 = vxCreateImage(context, 88, 16, arg_->result_format), VX_TYPE_IMAGE);

    // initialize to guarantee that images are allocated
    ASSERT_NO_FAILURE(ct_fill_image_random(src1_32x32, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src1_64x64, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2_32x32, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2_32x64, &CT()->seed_));

    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1_32x32, src2_32x32, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst88x16));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1_32x32, src2_32x64, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst32x32));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1_64x64, src2_32x32, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst32x32));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1_64x64, src2_32x64, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst32x32));

    vxReleaseImage(&src1_32x32);
    vxReleaseImage(&src2_32x32);
    vxReleaseImage(&src1_64x64);
    vxReleaseImage(&src2_32x64);
    vxReleaseImage(&dst32x32);
    vxReleaseImage(&dst88x16);
}

TEST_WITH_ARG(vxMultiply, testNegativeSizes, formats_arg, MUL_VALID_FORMATS(SATURATE), MUL_VALID_FORMATS(WRAP))
{
    vx_image src1_32x32, src1_64x64, src2_32x32, src2_32x64, dst32x32, dst88x16;
    vx_graph graph1, graph2, graph3, graph4;
    vx_scalar scale;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1_32x32 = vxCreateImage(context, 32, 32, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src1_64x64 = vxCreateImage(context, 64, 64, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x32 = vxCreateImage(context, 32, 32, arg_->arg2_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x64 = vxCreateImage(context, 32, 64, arg_->arg2_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst32x32 = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst88x16 = vxCreateImage(context, 88, 16, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(graph1 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(vxMultiplyNode(graph1, src1_32x32, src2_32x32, scale, arg_->overflow_policy, arg_->round_policy, dst88x16), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph1));

    ASSERT_VX_OBJECT(graph2 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(vxMultiplyNode(graph2, src1_32x32, src2_32x64, scale, arg_->overflow_policy, arg_->round_policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph2));

    ASSERT_VX_OBJECT(graph3 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(vxMultiplyNode(graph3, src1_64x64, src2_32x32, scale, arg_->overflow_policy, arg_->round_policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph3));

    ASSERT_VX_OBJECT(graph4 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(vxMultiplyNode(graph4, src1_64x64, src2_32x64, scale, arg_->overflow_policy, arg_->round_policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph4));

    vxReleaseImage(&src1_32x32);
    vxReleaseImage(&src2_32x32);
    vxReleaseImage(&src1_64x64);
    vxReleaseImage(&src2_32x64);
    vxReleaseImage(&dst32x32);
    vxReleaseImage(&dst88x16);
    vxReleaseScalar(&scale);
    vxReleaseGraph(&graph1);
    vxReleaseGraph(&graph2);
    vxReleaseGraph(&graph3);
    vxReleaseGraph(&graph4);
}

static vx_image inference_image;
static vx_enum  inference_image_format;
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
    EXPECT_EQ_INT(inference_image_format, format);

    return VX_ACTION_CONTINUE;
}

TEST_WITH_ARG(vxMultiply, testInference, formats_arg, MUL_INFERENCE_FORMATS(SATURATE), MUL_INFERENCE_FORMATS(WRAP))
{
    vx_image src1, src2, dst, gr;
    vx_graph graph;
    vx_scalar scale;
    vx_node n, tmp;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(src1  = vxCreateImage(context, 640, 480, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2  = vxCreateImage(context, 640, 480, arg_->arg2_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst   = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(n     = vxMultiplyNode(graph, src1, src2, scale, arg_->overflow_policy, arg_->round_policy, dst), VX_TYPE_NODE);

    // grounding
    ASSERT_VX_OBJECT(gr    = vxCreateImage(context, 640, 480, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(tmp   = vxAddNode(graph, dst, src2, VX_CONVERT_POLICY_WRAP, gr), VX_TYPE_NODE);

    // test
    inference_image = dst;
    inference_image_format = arg_->result_format;
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxAssignNodeCallback(n, inference_image_test));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));

    vxReleaseNode(&n);
    vxReleaseNode(&tmp);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseImage(&gr);
    vxReleaseScalar(&scale);
    vxReleaseGraph(&graph);
}

TEST_WITH_ARG(vxuMultiply, testFuzzy, fuzzy_arg, MUL_FUZZY_ARGS(SATURATE), MUL_FUZZY_ARGS(WRAP))
{
    vx_image src1, src2, dst;
    CT_Image ref1, ref2, refdst, refdst_plus_1, vxdst;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg2_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, arg_->width, arg_->height, arg_->result_format), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src1, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2, &CT()->seed_));

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuMultiply(context, src1, src2, arg_->scale, arg_->overflow_policy, arg_->round_policy, dst));

    ASSERT_NO_FAILURE({
        ref1  = ct_image_from_vx_image(src1);
        ref2  = ct_image_from_vx_image(src2);
        vxdst = ct_image_from_vx_image(dst);
        refdst        = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);
        refdst_plus_1 = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);

        referenceMultiply(ref1, ref2, refdst, refdst_plus_1, arg_->scale, arg_->overflow_policy);
    });

    if (arg_->scale == ONE_2_0)
        ASSERT_EQ_CTIMAGE(refdst, vxdst);
    else
    {
        // (|ref-v| <= 1 && |ref+1-v| <= 1)  is equivalent to (v == ref || v == ref + 1)
        if (arg_->overflow_policy == VX_CONVERT_POLICY_WRAP)
        {
            EXPECT_CTIMAGE_NEARWRAP(refdst, vxdst, 1, CTIMAGE_ALLOW_WRAP);
            EXPECT_CTIMAGE_NEARWRAP(refdst_plus_1, vxdst, 1, CTIMAGE_ALLOW_WRAP);
        }
        else
        {
            EXPECT_CTIMAGE_NEAR(refdst, vxdst, 1);
            EXPECT_CTIMAGE_NEAR(refdst_plus_1, vxdst, 1);
        }
    }

    // checked release vx images
    vxReleaseImage(&dst);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src1);
    EXPECT_EQ_PTR(NULL, src2);
}

TEST_WITH_ARG(vxMultiply, testFuzzy, fuzzy_arg, MUL_FUZZY_ARGS(SATURATE), MUL_FUZZY_ARGS(WRAP))
{
    vx_image src1, src2, dst;
    vx_graph graph;
    vx_scalar scale = 0;
    CT_Image ref1, ref2, refdst, refdst_plus_1, vxdst;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(dst   = vxCreateImage(context, arg_->width, arg_->height, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->scale), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg2_format),   VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src1, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2, &CT()->seed_));

    // build one-node graph
    ASSERT_VX_OBJECT(vxMultiplyNode(graph, src1, src2, scale, arg_->overflow_policy, arg_->round_policy, dst), VX_TYPE_NODE);

    // run graph
#ifdef CT_EXECUTE_ASYNC
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

    ASSERT_NO_FAILURE({
        ref1  = ct_image_from_vx_image(src1);
        ref2  = ct_image_from_vx_image(src2);
        vxdst = ct_image_from_vx_image(dst);
        refdst        = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);
        refdst_plus_1 = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);

        referenceMultiply(ref1, ref2, refdst, refdst_plus_1, arg_->scale, arg_->overflow_policy);
    });

    if (arg_->scale == ONE_2_0)
        ASSERT_EQ_CTIMAGE(refdst, vxdst);
    else
    {
        // (|ref-v| <= 1 && |ref+1-v| <= 1)  is equivalent to (v == ref || v == ref + 1)
        if (arg_->overflow_policy == VX_CONVERT_POLICY_WRAP)
        {
            EXPECT_CTIMAGE_NEARWRAP(refdst, vxdst, 1, CTIMAGE_ALLOW_WRAP);
            EXPECT_CTIMAGE_NEARWRAP(refdst_plus_1, vxdst, 1, CTIMAGE_ALLOW_WRAP);
        }
        else
        {
            EXPECT_CTIMAGE_NEAR(refdst, vxdst, 1);
            EXPECT_CTIMAGE_NEAR(refdst_plus_1, vxdst, 1);
        }
    }

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseScalar(&scale);
    vxReleaseGraph(&graph);
}

TESTCASE_TESTS(vxuMultiply, DISABLED_testNegativeFormat, DISABLED_testNegativeSizes,                testFuzzy)
TESTCASE_TESTS(vxMultiply,  DISABLED_testNegativeFormat, DISABLED_testNegativeSizes, testInference, testFuzzy)
