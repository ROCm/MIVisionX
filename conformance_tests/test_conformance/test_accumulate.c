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


TESTCASE(Accumulate, CT_VXContext, ct_setup_vx_context, 0)


TEST(Accumulate, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0, accum = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum = vxCreateImage(context, 128, 128, VX_DF_IMAGE_S16), VX_TYPE_IMAGE);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxAccumulateImageNode(graph, input, accum), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&accum);
    vxReleaseImage(&input);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(accum == 0);
    ASSERT(input == 0);
}


static CT_Image accumulate_generate_random_8u(int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}


static CT_Image accumulate_generate_random_16s(int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_S16, &CT()->seed_, -32768, 32768));

    return image;
}


static void accumulate_reference(CT_Image input, CT_Image accum)
{
    CT_FILL_IMAGE_16S(return, accum,
            {
                uint8_t* input_data = CT_IMAGE_DATA_PTR_8U(input, x, y);
                int32_t res32 = ((int32_t)(*dst_data)) + ((int32_t)(*input_data));
                int16_t res = CT_SATURATE_S16(res32);
                *dst_data = res;
            });
}


static void accumulate_check(CT_Image input, CT_Image accum_src, CT_Image accum_dst)
{
    CT_Image accum_ref = NULL;

    ASSERT(input && accum_src && accum_dst);

    ASSERT_NO_FAILURE(accum_ref = ct_image_create_clone(accum_src));

    ASSERT_NO_FAILURE(accumulate_reference(input, accum_ref));

    EXPECT_EQ_CTIMAGE(accum_ref, accum_dst);
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
    int dummy_;
    int width, height;
} Arg;


#define PARAMETERS \
    CT_GENERATE_PARAMETERS("random", ADD_SIZE_SMALL_SET, ARG, 0)

TEST_WITH_ARG(Accumulate, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, accum_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image input = NULL, accum_src = NULL, accum_dst = NULL;

    ASSERT_NO_FAILURE(input = accumulate_generate_random_8u(arg_->width, arg_->height));

    ASSERT_NO_FAILURE(accum_src = accumulate_generate_random_16s(arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum_image = ct_image_to_vx_image(accum_src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxAccumulateImageNode(graph, input_image, accum_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(accum_dst = ct_image_from_vx_image(accum_image));

    ASSERT_NO_FAILURE(accumulate_check(input, accum_src, accum_dst));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&accum_image);
    vxReleaseImage(&input_image);

    ASSERT(accum_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(Accumulate, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, accum_image = 0;

    CT_Image input = NULL, accum_src = NULL, accum_dst = NULL;

    ASSERT_NO_FAILURE(input = accumulate_generate_random_8u(arg_->width, arg_->height));

    ASSERT_NO_FAILURE(accum_src = accumulate_generate_random_16s(arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(accum_image = ct_image_to_vx_image(accum_src, context), VX_TYPE_IMAGE);

    VX_CALL(vxuAccumulateImage(context, input_image, accum_image));

    ASSERT_NO_FAILURE(accum_dst = ct_image_from_vx_image(accum_image));

    ASSERT_NO_FAILURE(accumulate_check(input, accum_src, accum_dst));

    vxReleaseImage(&accum_image);
    vxReleaseImage(&input_image);

    ASSERT(accum_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(Accumulate,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
