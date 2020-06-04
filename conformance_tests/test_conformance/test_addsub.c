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

static void referenceAdd(CT_Image src0, CT_Image src1, CT_Image dst, enum vx_convert_policy_e policy)
{
    int32_t min_bound, max_bound;
    uint32_t i, j;
    ASSERT(src0 && src1 && dst);
    ASSERT(src0->width = src1->width && src0->width == dst->width);
    ASSERT(src0->height = src1->height && src0->height == dst->height);

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

#define ADD_LOOP(s0, s1, r)                                                                                     \
    do{                                                                                                         \
        for (i = 0; i < dst->height; ++i)                                                                       \
            for (j = 0; j < dst->width; ++j)                                                                    \
            {                                                                                                   \
                int32_t val = src0->data.s0[i * src0->stride + j];                                              \
                val += src1->data.s1[i * src1->stride + j];                                                     \
                dst->data.r[i * dst->stride + j] = (val < min_bound ? min_bound :                               \
                                                                        (val > max_bound ? max_bound : val));   \
            }                                                                                                   \
    }while(0)

    if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8)
        ADD_LOOP(y, y, y);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        ADD_LOOP(y, y, s16);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        ADD_LOOP(y, s16, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        ADD_LOOP(s16, y, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        ADD_LOOP(s16, s16, s16);
    else
        FAIL("Unsupported combination of argument formats: %.4s + %.4s = %.4s", &src0->format, &src1->format, &dst->format);

#undef ADD_LOOP
}

static void referenceSubtract(CT_Image src0, CT_Image src1, CT_Image dst, enum vx_convert_policy_e policy)
{
    int32_t min_bound, max_bound;
    uint32_t i, j;
    ASSERT(src0 && src1 && dst);
    ASSERT(src0->width = src1->width && src0->width == dst->width);
    ASSERT(src0->height = src1->height && src0->height == dst->height);

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

#define SUB_LOOP(s0, s1, r)                                                                                     \
    do{                                                                                                         \
        for (i = 0; i < dst->height; ++i)                                                                       \
            for (j = 0; j < dst->width; ++j)                                                                    \
            {                                                                                                   \
                int32_t val = src0->data.s0[i * src0->stride + j];                                              \
                val -= src1->data.s1[i * src1->stride + j];                                                     \
                dst->data.r[i * dst->stride + j] = (val < min_bound ? min_bound :                               \
                                                                        (val > max_bound ? max_bound : val));   \
            }                                                                                                   \
    }while(0)

    if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8)
        SUB_LOOP(y, y, y);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        SUB_LOOP(y, y, s16);
    else if (src0->format == VX_DF_IMAGE_U8 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        SUB_LOOP(y, s16, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_S16)
        SUB_LOOP(s16, y, s16);
    else if (src0->format == VX_DF_IMAGE_S16 && src1->format == VX_DF_IMAGE_S16 && dst->format == VX_DF_IMAGE_S16)
        SUB_LOOP(s16, s16, s16);
    else
        FAIL("Unsupported combination of argument formats: %.4s + %.4s = %.4s", &src0->format, &src1->format, &dst->format);

#undef SUB_LOOP
}

typedef vx_status (VX_API_CALL *vxuArithmFunction)(vx_context, vx_image, vx_image, vx_enum, vx_image);
typedef vx_node   (VX_API_CALL *vxArithmFunction) (vx_graph, vx_image, vx_image, vx_enum, vx_image);
typedef void      (*referenceFunction)(CT_Image, CT_Image, CT_Image, enum vx_convert_policy_e);

#define SGN_Add "+"
#define SGN_Subtract "-"

typedef struct {
    const char* name;
    enum vx_convert_policy_e policy;
    vx_df_image arg1_format;
    int arg1;
    vx_df_image arg2_format;
    int arg2;
    vx_df_image result_format;
    int expected_result;
    vxuArithmFunction vxuFunc;
    vxArithmFunction vxFunc;
} overflow_arg;

#define OVERFLOW_ARG(func, p, f1, f2, fr, v1, v2, vr) \
    ARG(#func ": " #p " " #f1 "(" #v1 ") " SGN_##func " " #f2 "(" #v2 ") = " #fr "(" #vr ")", \
        VX_CONVERT_POLICY_##p, VX_DF_IMAGE_##f1, v1, VX_DF_IMAGE_##f2, v2, VX_DF_IMAGE_##fr, vr, vxu##func, vx##func##Node)\

#define ADDSUB_OVERFLOW_ARGS                                                                        \
    OVERFLOW_ARG(Add, SATURATE, U8, U8, U8, 175, 210, 255), /* with overflow */                     \
    OVERFLOW_ARG(Add, SATURATE, U8, U8, U8, 0,   255, 255), /* normal        */                     \
    OVERFLOW_ARG(Add, WRAP,     U8, U8, U8, 175, 210, 129), /* with overflow */                     \
    OVERFLOW_ARG(Add, WRAP,     U8, U8, U8, 0,   255, 255), /* normal        */                     \
                                                                                                    \
    OVERFLOW_ARG(Add, SATURATE, U8, U8, S16, 175, 210, 385), /* normal */                           \
    OVERFLOW_ARG(Add, WRAP,     U8, U8, S16, 255, 255, 510), /* normal */                           \
                                                                                                    \
    OVERFLOW_ARG(Add, SATURATE, U8, S16, S16, 175, 32760, 32767),  /* with overflow */              \
    OVERFLOW_ARG(Add, SATURATE, U8, S16, S16, 175, -1000, -825),   /* normal        */              \
    OVERFLOW_ARG(Add, WRAP,     U8, S16, S16, 1,   32767, -32768), /* with overflow */              \
    OVERFLOW_ARG(Add, WRAP,     U8, S16, S16, 255, -1000, -745),   /* normal        */              \
                                                                                                    \
    OVERFLOW_ARG(Add, SATURATE, S16, U8, S16, 32513, 255, 32767),  /* with overflow */              \
    OVERFLOW_ARG(Add, SATURATE, S16, U8, S16, 32512, 255, 32767),  /* normal        */              \
    OVERFLOW_ARG(Add, WRAP,     S16, U8, S16, 32514, 255, -32767), /* with overflow */              \
    OVERFLOW_ARG(Add, WRAP,     S16, U8, S16, 0,     0,   0),      /* normal        */              \
                                                                                                    \
    OVERFLOW_ARG(Add, SATURATE, S16, S16, S16,  32000, -32000,  0),     /* normal         */        \
    OVERFLOW_ARG(Add, SATURATE, S16, S16, S16,  32000,  32000,  32767), /* with overflow  */        \
    OVERFLOW_ARG(Add, SATURATE, S16, S16, S16, -32000, -32000, -32768), /* with underflow */        \
    OVERFLOW_ARG(Add, WRAP,     S16, S16, S16, -32768,  32767, -1),     /* normal         */        \
    OVERFLOW_ARG(Add, WRAP,     S16, S16, S16,  32767,  32767, -2),     /* with overflow  */        \
    OVERFLOW_ARG(Add, WRAP,     S16, S16, S16, -17000, -17000,  31536), /* with underflow */        \
                                                                                                    \
    OVERFLOW_ARG(Subtract, SATURATE, U8, U8, U8, 175, 210, 0),   /* with underflow */               \
    OVERFLOW_ARG(Subtract, SATURATE, U8, U8, U8, 255,  55, 200), /* normal         */               \
    OVERFLOW_ARG(Subtract, WRAP,     U8, U8, U8, 175, 210, 221), /* with underflow */               \
    OVERFLOW_ARG(Subtract, WRAP,     U8, U8, U8, 250, 240, 10),  /* normal         */               \
                                                                                                    \
    OVERFLOW_ARG(Subtract, SATURATE, U8, U8, S16, 175, 210, -35), /* normal */                      \
    OVERFLOW_ARG(Subtract, WRAP,     U8, U8, S16, 254, 255, -1),  /* normal */                      \
                                                                                                    \
    OVERFLOW_ARG(Subtract, SATURATE, U8, S16, S16, 175, -32760, 32767),  /* with overflow */        \
    OVERFLOW_ARG(Subtract, SATURATE, U8, S16, S16, 175, 1000, -825),     /* normal        */        \
    OVERFLOW_ARG(Subtract, WRAP,     U8, S16, S16, 1,   -32767, -32768), /* with overflow */        \
    OVERFLOW_ARG(Subtract, WRAP,     U8, S16, S16, 255, 1000, -745),     /* normal        */        \
                                                                                                    \
    OVERFLOW_ARG(Subtract, SATURATE, S16, U8, S16, -32514, 255, -32768), /* with underflow */       \
    OVERFLOW_ARG(Subtract, SATURATE, S16, U8, S16, -32513, 255, -32768), /* normal         */       \
    OVERFLOW_ARG(Subtract, WRAP,     S16, U8, S16, -32514, 255, 32767),  /* with underflow */       \
    OVERFLOW_ARG(Subtract, WRAP,     S16, U8, S16,  0,     0,   0),      /* normal         */       \
                                                                                                    \
    OVERFLOW_ARG(Subtract, SATURATE, S16, S16, S16,  32000,  32000,  0),     /* normal         */   \
    OVERFLOW_ARG(Subtract, SATURATE, S16, S16, S16,  32000, -32000,  32767), /* with overflow  */   \
    OVERFLOW_ARG(Subtract, SATURATE, S16, S16, S16, -32000,  32000, -32768), /* with underflow */   \
    OVERFLOW_ARG(Subtract, WRAP,     S16, S16, S16, -32768, -32767, -1),     /* normal         */   \
    OVERFLOW_ARG(Subtract, WRAP,     S16, S16, S16,  32767, -32767, -2),     /* with overflow  */   \
    OVERFLOW_ARG(Subtract, WRAP,     S16, S16, S16, -17000,  17000,  31536), /* with underflow */

typedef struct {
    const char* name;
    enum vx_convert_policy_e policy;
    int width, height;
    vx_df_image arg1_format, arg2_format, result_format;
    vxuArithmFunction vxuFunc;
    vxArithmFunction vxFunc;
    referenceFunction referenceFunc;
} fuzzy_arg, formats_arg;

#define FUZZY_ARG(func, p, w, h, f1, f2, fr)        \
    ARG(#func ": " #p " " #w "x" #h " " #f1 SGN_##func #f2 "=" #fr,   \
        VX_CONVERT_POLICY_##p, w, h, VX_DF_IMAGE_##f1, VX_DF_IMAGE_##f2, VX_DF_IMAGE_##fr, vxu##func, vx##func##Node, reference##func)

#define FORMATS_ARG(func, p, f1, f2, fr)  \
    ARG(#func ": " #p " " #f1 SGN_##func #f2 "=" #fr, \
        VX_CONVERT_POLICY_##p, 0, 0, VX_DF_IMAGE_##f1, VX_DF_IMAGE_##f2, VX_DF_IMAGE_##fr, vxu##func, vx##func##Node, reference##func)

#define ARITHM_INVALID_FORMATS(func)            \
    FORMATS_ARG(func, SATURATE, S16, S16, U8),  \
    FORMATS_ARG(func, SATURATE, S16, U8,  U8),  \
    FORMATS_ARG(func, SATURATE, U8,  S16, U8),  \
    FORMATS_ARG(func, WRAP, S16, S16, U8),      \
    FORMATS_ARG(func, WRAP, S16, U8,  U8),      \
    FORMATS_ARG(func, WRAP, U8,  S16, U8)

#define ARITHM_INFERENCE_FORMATS(func)          \
    FORMATS_ARG(func, SATURATE, S16, S16, S16), \
    FORMATS_ARG(func, SATURATE, S16, U8,  S16), \
    FORMATS_ARG(func, SATURATE, U8,  S16, S16), \
    FORMATS_ARG(func, SATURATE, U8,  U8,  S16), \
    FORMATS_ARG(func, WRAP, S16, S16, S16),     \
    FORMATS_ARG(func, WRAP, S16, U8,  S16),     \
    FORMATS_ARG(func, WRAP, U8,  S16, S16),     \
    FORMATS_ARG(func, WRAP, U8,  U8,  S16)

#define ARITHM_VALID_FORMATS(func)              \
    ARITHM_INFERENCE_FORMATS(func),             \
    FORMATS_ARG(func, SATURATE, U8, U8, U8),    \
    FORMATS_ARG(func, WRAP, U8, U8, U8)

#define ARITHM_FUZZY_ARGS(func)                         \
    FUZZY_ARG(func, SATURATE, 640, 480, U8, U8, U8),    \
    FUZZY_ARG(func, SATURATE, 640, 480, U8, U8, S16),   \
    FUZZY_ARG(func, SATURATE, 640, 480, U8, S16, S16),  \
    FUZZY_ARG(func, SATURATE, 640, 480, S16, U8, S16),  \
    FUZZY_ARG(func, SATURATE, 640, 480, S16, S16, S16), \
                                                        \
    FUZZY_ARG(func, WRAP, 640, 480, U8, U8, U8),        \
    FUZZY_ARG(func, WRAP, 640, 480, U8, U8, S16),       \
    FUZZY_ARG(func, WRAP, 640, 480, U8, S16, S16),      \
    FUZZY_ARG(func, WRAP, 640, 480, S16, U8, S16),      \
    FUZZY_ARG(func, WRAP, 640, 480, S16, S16, S16),     \
                                                        \
    ARG_EXTENDED_BEGIN(),                               \
    FUZZY_ARG(func, SATURATE, 15, 15, U8, U8, U8),      \
    FUZZY_ARG(func, SATURATE, 15, 15, U8, U8, S16),     \
    FUZZY_ARG(func, SATURATE, 15, 15, U8, S16, S16),    \
    FUZZY_ARG(func, SATURATE, 15, 15, S16, U8, S16),    \
    FUZZY_ARG(func, SATURATE, 15, 15, S16, S16, S16),   \
                                                        \
    FUZZY_ARG(func, SATURATE, 1280, 720, U8, U8, U8),   \
    FUZZY_ARG(func, SATURATE, 1280, 720, U8, U8, S16),  \
    FUZZY_ARG(func, SATURATE, 1280, 720, U8, S16, S16), \
    FUZZY_ARG(func, SATURATE, 1280, 720, S16, U8, S16), \
    FUZZY_ARG(func, SATURATE, 1280, 720, S16, S16, S16),\
                                                        \
    FUZZY_ARG(func, WRAP, 15, 15, U8, U8, U8),          \
    FUZZY_ARG(func, WRAP, 15, 15, U8, U8, S16),         \
    FUZZY_ARG(func, WRAP, 15, 15, U8, S16, S16),        \
    FUZZY_ARG(func, WRAP, 15, 15, S16, U8, S16),        \
    FUZZY_ARG(func, WRAP, 15, 15, S16, S16, S16),       \
                                                        \
    FUZZY_ARG(func, WRAP, 1280, 720, U8, U8, U8),       \
    FUZZY_ARG(func, WRAP, 1280, 720, U8, U8, S16),      \
    FUZZY_ARG(func, WRAP, 1280, 720, U8, S16, S16),     \
    FUZZY_ARG(func, WRAP, 1280, 720, S16, U8, S16),     \
    FUZZY_ARG(func, WRAP, 1280, 720, S16, S16, S16),    \
    ARG_EXTENDED_END()

TESTCASE(vxuAddSub, CT_VXContext, ct_setup_vx_context, 0)
TESTCASE(vxAddSub,  CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG(vxuAddSub, testNegativeFormat, formats_arg, ARITHM_INVALID_FORMATS(Add), ARITHM_INVALID_FORMATS(Subtract))
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
    ASSERT_NE_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1, src2, arg_->policy, dst));

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
}

TEST_WITH_ARG(vxAddSub, testNegativeFormat, formats_arg, ARITHM_INVALID_FORMATS(Add), ARITHM_INVALID_FORMATS(Subtract))
{
    vx_image src1, src2, dst;
    vx_graph graph;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, 32, 32, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, 32, 32, arg_->arg2_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(arg_->vxFunc(graph, src1, src2, arg_->policy, dst), VX_TYPE_NODE);

    // The output image can be VX_DF_IMAGE_U8 only if both source images are
    // VX_DF_IMAGE_U8 and the output image is explicitly set to VX_DF_IMAGE_U8. It is
    // otherwise VX_DF_IMAGE_S16.
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseGraph(&graph);
}

TEST_WITH_ARG(vxuAddSub, testNegativeSizes, formats_arg, ARITHM_VALID_FORMATS(Add), ARITHM_VALID_FORMATS(Subtract))
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

    EXPECT_NE_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1_32x32, src2_32x32, arg_->policy, dst88x16));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1_32x32, src2_32x64, arg_->policy, dst32x32));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1_64x64, src2_32x32, arg_->policy, dst32x32));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1_64x64, src2_32x64, arg_->policy, dst32x32));

    vxReleaseImage(&src1_32x32);
    vxReleaseImage(&src2_32x32);
    vxReleaseImage(&src1_64x64);
    vxReleaseImage(&src2_32x64);
    vxReleaseImage(&dst32x32);
    vxReleaseImage(&dst88x16);
}

TEST_WITH_ARG(vxAddSub, testNegativeSizes, formats_arg, ARITHM_VALID_FORMATS(Add), ARITHM_VALID_FORMATS(Subtract))
{
    vx_image src1_32x32, src1_64x64, src2_32x32, src2_32x64, dst32x32, dst88x16;
    vx_graph graph1, graph2, graph3, graph4;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1_32x32 = vxCreateImage(context, 32, 32, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src1_64x64 = vxCreateImage(context, 64, 64, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x32 = vxCreateImage(context, 32, 32, arg_->arg2_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_32x64 = vxCreateImage(context, 32, 64, arg_->arg2_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst32x32 = vxCreateImage(context, 32, 32, arg_->result_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst88x16 = vxCreateImage(context, 88, 16, arg_->result_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph1 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(arg_->vxFunc(graph1, src1_32x32, src2_32x32, arg_->policy, dst88x16), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph1));

    ASSERT_VX_OBJECT(graph2 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(arg_->vxFunc(graph2, src1_32x32, src2_32x64, arg_->policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph2));

    ASSERT_VX_OBJECT(graph3 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(arg_->vxFunc(graph3, src1_64x64, src2_32x32, arg_->policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph3));

    ASSERT_VX_OBJECT(graph4 = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(arg_->vxFunc(graph4, src1_64x64, src2_32x64, arg_->policy, dst32x32), VX_TYPE_NODE);
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph4));

    vxReleaseImage(&src1_32x32);
    vxReleaseImage(&src2_32x32);
    vxReleaseImage(&src1_64x64);
    vxReleaseImage(&src2_32x64);
    vxReleaseImage(&dst32x32);
    vxReleaseImage(&dst88x16);
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

TEST_WITH_ARG(vxAddSub, testInference, formats_arg, ARITHM_INFERENCE_FORMATS(Add), ARITHM_INFERENCE_FORMATS(Subtract))
{
    vx_image src1, src2, dst, gr;
    vx_graph graph;
    vx_node n, tmp;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(src1  = vxCreateImage(context, 640, 480, arg_->arg1_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2  = vxCreateImage(context, 640, 480, arg_->arg2_format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst   = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(n     = arg_->vxFunc(graph, src1, src2, arg_->policy, dst), VX_TYPE_NODE);

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
    vxReleaseGraph(&graph);
}

TEST_WITH_ARG(vxuAddSub, testOverflowModes, overflow_arg, ADDSUB_OVERFLOW_ARGS)
{
    vx_image src1, src2, dst;
    union { uint8_t u; int16_t s; } init_val;
    vx_context context = context_->vx_context_;

    // allocate and fill images
    ASSERT_VX_OBJECT(dst = vxCreateImage(context, 640, 480, arg_->result_format), VX_TYPE_IMAGE);

    switch (arg_->arg1_format)
    {
        case VX_DF_IMAGE_U8:  init_val.u = arg_->arg1; break;
        case VX_DF_IMAGE_S16: init_val.s = arg_->arg1; break;
        default: FAIL("Bad test argument"); break;
    };
    ASSERT_VX_OBJECT(src1 = vxCreateUniformImage(context, 640, 480, arg_->arg1_format, &init_val), VX_TYPE_IMAGE);

    switch (arg_->arg2_format)
    {
        case VX_DF_IMAGE_U8:  init_val.u = arg_->arg2; break;
        case VX_DF_IMAGE_S16: init_val.s = arg_->arg2; break;
        default: FAIL("Bad test argument"); break;
    };
    ASSERT_VX_OBJECT(src2 = vxCreateUniformImage(context, 640, 480, arg_->arg2_format, &init_val), VX_TYPE_IMAGE);

    // run function
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1, src2, arg_->policy, dst));

    // test bottom right pixel to have an expected value
    {
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        void *pdata = 0;
        void *bottom_right_pixel = 0;
        vx_rectangle_t rect = {0 ,0, 640, 480};

        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxAccessImagePatch(dst, &rect, 0, &addr, &pdata, VX_READ_ONLY));

        bottom_right_pixel = vxFormatImagePatchAddress2d(pdata, addr.dim_x-1, addr.dim_y-1, &addr);
        ASSERT(bottom_right_pixel != NULL);

        switch (arg_->result_format)
        {
            case VX_DF_IMAGE_U8:  EXPECT_EQ_INT(arg_->expected_result, *(uint8_t*)bottom_right_pixel); break;
            case VX_DF_IMAGE_S16: EXPECT_EQ_INT(arg_->expected_result, *(int16_t*)bottom_right_pixel); break;
            default: ADD_FAILURE("Bad test argument"); break;
        };

        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCommitImagePatch(dst, 0, 0, &addr, pdata));
    }

    // checked release vx images
    vxReleaseImage(&dst);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src1);
    EXPECT_EQ_PTR(NULL, src2);
}

TEST_WITH_ARG(vxAddSub, testOverflowModes, overflow_arg, ADDSUB_OVERFLOW_ARGS)
{
    vx_image src1, src2, dst;
    union { uint8_t u; int16_t s; } init_val;
    vx_graph graph;
    vx_context context = context_->vx_context_;

    // allocate and fill images
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(dst   = vxCreateImage(context, 640, 480, arg_->result_format), VX_TYPE_IMAGE);

    switch (arg_->arg1_format)
    {
        case VX_DF_IMAGE_U8:  init_val.u = arg_->arg1; break;
        case VX_DF_IMAGE_S16: init_val.s = arg_->arg1; break;
        default: FAIL("Bad test argument"); break;
    };
    ASSERT_VX_OBJECT(src1 = vxCreateUniformImage(context, 640, 480, arg_->arg1_format, &init_val), VX_TYPE_IMAGE);

    switch (arg_->arg2_format)
    {
        case VX_DF_IMAGE_U8:  init_val.u = arg_->arg2; break;
        case VX_DF_IMAGE_S16: init_val.s = arg_->arg2; break;
        default: FAIL("Bad test argument"); break;
    };
    ASSERT_VX_OBJECT(src2 = vxCreateUniformImage(context, 640, 480, arg_->arg2_format, &init_val), VX_TYPE_IMAGE);

    // build one-node graph
    ASSERT_VX_OBJECT(arg_->vxFunc(graph, src1, src2, arg_->policy, dst), VX_TYPE_NODE);

    // run graph
#ifdef CT_EXECUTE_ASYNC
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

    // test bottom right pixel to have an expected value
    {
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        void *pdata = 0;
        void *bottom_right_pixel = 0;
        vx_rectangle_t rect = {0 ,0, 640, 480};

        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxAccessImagePatch(dst, &rect, 0, &addr, &pdata, VX_READ_ONLY));

        bottom_right_pixel = vxFormatImagePatchAddress2d(pdata, addr.dim_x-1, addr.dim_y-1, &addr);
        ASSERT(bottom_right_pixel != NULL);

        switch (arg_->result_format)
        {
            case VX_DF_IMAGE_U8:  EXPECT_EQ_INT(arg_->expected_result, *(uint8_t*)bottom_right_pixel); break;
            case VX_DF_IMAGE_S16: EXPECT_EQ_INT(arg_->expected_result, *(int16_t*)bottom_right_pixel); break;
            default: ADD_FAILURE("Bad test argument"); break;
        };

        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCommitImagePatch(dst, 0, 0, &addr, pdata));
    }

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseGraph(&graph);
}

TEST_WITH_ARG(vxuAddSub, testFuzzy, fuzzy_arg, ARITHM_FUZZY_ARGS(Add), ARITHM_FUZZY_ARGS(Subtract))
{
    vx_image src1, src2, dst;
    CT_Image ref1, ref2, refdst, vxdst;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg2_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst  = vxCreateImage(context, arg_->width, arg_->height, arg_->result_format), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src1, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2, &CT()->seed_));

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, arg_->vxuFunc(context, src1, src2, arg_->policy, dst));

    ref1  = ct_image_from_vx_image(src1);
    ref2  = ct_image_from_vx_image(src2);
    vxdst = ct_image_from_vx_image(dst);
    refdst = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);

    arg_->referenceFunc(ref1, ref2, refdst, arg_->policy);

    ASSERT_EQ_CTIMAGE(refdst, vxdst);

    // checked release vx images
    vxReleaseImage(&dst);
    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    EXPECT_EQ_PTR(NULL, dst);
    EXPECT_EQ_PTR(NULL, src1);
    EXPECT_EQ_PTR(NULL, src2);
}

TEST_WITH_ARG(vxAddSub, testFuzzy, fuzzy_arg, ARITHM_FUZZY_ARGS(Add), ARITHM_FUZZY_ARGS(Subtract))
{
    vx_image src1, src2, dst;
    vx_graph graph;
    CT_Image ref1, ref2, refdst, vxdst;
    vx_context context = context_->vx_context_;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(dst   = vxCreateImage(context, arg_->width, arg_->height, arg_->result_format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(src1 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg1_format),   VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2 = vxCreateImage(context, arg_->width, arg_->height, arg_->arg2_format),   VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(src1, &CT()->seed_));
    ASSERT_NO_FAILURE(ct_fill_image_random(src2, &CT()->seed_));

    // build one-node graph
    ASSERT_VX_OBJECT(arg_->vxFunc(graph, src1, src2, arg_->policy, dst), VX_TYPE_NODE);

    // run graph
#ifdef CT_EXECUTE_ASYNC
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxScheduleGraph(graph));
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));
#else
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));
#endif

    ref1  = ct_image_from_vx_image(src1);
    ref2  = ct_image_from_vx_image(src2);
    vxdst = ct_image_from_vx_image(dst);
    refdst = ct_allocate_image(arg_->width, arg_->height, arg_->result_format);

    arg_->referenceFunc(ref1, ref2, refdst, arg_->policy);

    ASSERT_EQ_CTIMAGE(refdst, vxdst);

    vxReleaseImage(&src1);
    vxReleaseImage(&src2);
    vxReleaseImage(&dst);
    vxReleaseGraph(&graph);
}

TESTCASE_TESTS(vxuAddSub, DISABLED_testNegativeFormat, DISABLED_testNegativeSizes,                testOverflowModes, testFuzzy)
TESTCASE_TESTS(vxAddSub,  DISABLED_testNegativeFormat, DISABLED_testNegativeSizes, testInference, testOverflowModes, testFuzzy)
