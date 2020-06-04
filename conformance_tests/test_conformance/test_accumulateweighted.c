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


TESTCASE(AccumulateWeighted, CT_VXContext, ct_setup_vx_context, 0)


TEST(AccumulateWeighted, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0, accum = 0;
    vx_float32 alpha = 0.5f;
    vx_scalar alpha_scalar = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(alpha_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &alpha), VX_TYPE_SCALAR);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxAccumulateWeightedImageNode(graph, input, alpha_scalar, accum), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&accum);
    vxReleaseImage(&input);
    vxReleaseScalar(&alpha_scalar);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(accum == 0);
    ASSERT(input == 0);
}


static CT_Image accumulate_weighted_generate_random_8u(int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}


static void accumulate_weighted_reference(CT_Image input, vx_float32 alpha, CT_Image accum)
{
    CT_FILL_IMAGE_8U(return, accum,
            {
                uint8_t* input_data = CT_IMAGE_DATA_PTR_8U(input, x, y);
                vx_float32 res = (1 - alpha) * ((vx_float32)(int32_t)(*dst_data)) + (alpha) * ((vx_float32)(int32_t)(*input_data));
                uint8_t res8 = CT_SATURATE_U8(res);
                *dst_data = res8;
            });
}


static void accumulate_weighted_check(CT_Image input, vx_float32 alpha, CT_Image accum_src, CT_Image accum_dst)
{
    CT_Image accum_ref = NULL;

    ASSERT(input && accum_src && accum_dst);

    ASSERT_NO_FAILURE(accum_ref = ct_image_create_clone(accum_src));

    ASSERT_NO_FAILURE(accumulate_weighted_reference(input, alpha, accum_ref));

    EXPECT_CTIMAGE_NEAR(accum_ref, accum_dst, 1);
#if 0
    if (CT_HasFailure())
    {
        printf("=== Input ===\n");
        ct_dump_image_info(input);
        printf("=== Accum source ===\n");
        ct_dump_image_info(accum_src);
        printf("=== Accum RESULT ===\n");
        ct_dump_image_info(accum_dst);
        printf("=== EXPECTED RESULT ===\n");
        ct_dump_image_info(accum_ref);
    }
#endif
}

typedef struct {
    const char* testName;
    vx_float32 alpha;
    int width, height;
} Arg;


#define PARAMETERS \
    CT_GENERATE_PARAMETERS("random/alpha0.5f", ADD_SIZE_SMALL_SET, ARG, 0.5f), \
    CT_GENERATE_PARAMETERS("random/alpha0.0f", ADD_SIZE_SMALL_SET, ARG, 0.0f), \
    CT_GENERATE_PARAMETERS("random/alpha1.0f", ADD_SIZE_SMALL_SET, ARG, 1.0f), \
    CT_GENERATE_PARAMETERS("random/alpha0.25f", ADD_SIZE_SMALL_SET, ARG, 0.25f), \
    CT_GENERATE_PARAMETERS("random/alpha0.95f", ADD_SIZE_SMALL_SET, ARG, 0.95f), \
    CT_GENERATE_PARAMETERS("random/alpha0.999f", ADD_SIZE_SMALL_SET, ARG, 0.999f), \
    CT_GENERATE_PARAMETERS("random/alpha0.001f", ADD_SIZE_SMALL_SET, ARG, 0.001f)

TEST_WITH_ARG(AccumulateWeighted, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, accum_image = 0;
    vx_scalar alpha_scalar = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image input = NULL, accum_src = NULL, accum_dst = NULL;

    ASSERT_VX_OBJECT(alpha_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->alpha), VX_TYPE_SCALAR);

    ASSERT_NO_FAILURE(input = accumulate_weighted_generate_random_8u(arg_->width, arg_->height));

    ASSERT_NO_FAILURE(accum_src = accumulate_weighted_generate_random_8u(arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum_image = ct_image_to_vx_image(accum_src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxAccumulateWeightedImageNode(graph, input_image, alpha_scalar, accum_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(accum_dst = ct_image_from_vx_image(accum_image));

    ASSERT_NO_FAILURE(accumulate_weighted_check(input, arg_->alpha, accum_src, accum_dst));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&accum_image);
    vxReleaseImage(&input_image);
    vxReleaseScalar(&alpha_scalar);

    ASSERT(accum_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(AccumulateWeighted, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, accum_image = 0;
    vx_scalar alpha_scalar = 0;

    CT_Image input = NULL, accum_src = NULL, accum_dst = NULL;

    ASSERT_VX_OBJECT(alpha_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &arg_->alpha), VX_TYPE_SCALAR);

    ASSERT_NO_FAILURE(input = accumulate_weighted_generate_random_8u(arg_->width, arg_->height));

    ASSERT_NO_FAILURE(accum_src = accumulate_weighted_generate_random_8u(arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum_image = ct_image_to_vx_image(accum_src, context), VX_TYPE_IMAGE);

    VX_CALL(vxuAccumulateWeightedImage(context, input_image, alpha_scalar, accum_image));

    ASSERT_NO_FAILURE(accum_dst = ct_image_from_vx_image(accum_image));

    ASSERT_NO_FAILURE(accumulate_weighted_check(input, arg_->alpha, accum_src, accum_dst));

    vxReleaseImage(&accum_image);
    vxReleaseImage(&input_image);
    vxReleaseScalar(&alpha_scalar);

    ASSERT(accum_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(AccumulateWeighted,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
