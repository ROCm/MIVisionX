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
#include "shared_functions.h"


TESTCASE(Box3x3, CT_VXContext, ct_setup_vx_context, 0)


TEST(Box3x3, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxBox3x3Node(graph, src_image, dst_image), VX_TYPE_NODE);

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}


// Generate input to cover these requirements:
// The input data should have areas that have zero sum and maximum sum
// and everything in between (unless the running time is too high).
static CT_Image box3x3_generate_simple_gradient(const char* fileName, int step_x, int step_y)
{
    CT_Image image = NULL;
    uint32_t x, y;

    ASSERT_(return 0, step_x > 0);
    ASSERT_(return 0, step_y > 0);

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_image(256 * step_x, 256 * step_y, VX_DF_IMAGE_U8));

    for (y = 0; y < image->height; y++)
    {
        for (x = 0; x < image->width; x++)
        {
            uint8_t* ptr = CT_IMAGE_DATA_PTR_8U(image, x, y);
            int v = (y / step_y) + (x / step_x);
            *ptr = (uint8_t)v;
        }
    }

    return image;
}

// Generate input to cover these requirements:
// The input data should contain a bi-level image with every possible
// 3x3 block of pixels taking only the minimum and the maximum intensity values.
static CT_Image box3x3_generate_bi_level(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    uint32_t x, y;
    uint64_t rng;
    int total = width * height;

    CT_RNG_INIT(rng, CT_RNG_NEXT(CT()->seed_));

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_image(width, height, VX_DF_IMAGE_U8));

    for (y = 0; y < image->height; y++)
    {
        for (x = 0; x < image->width; x++)
        {
            uint8_t* ptr = CT_IMAGE_DATA_PTR_8U(image, x, y);
            int v = CT_RNG_NEXT_INT(rng, 0, 3 * total);
            *ptr = (v < (int)(total + x + y)) ? 255 : 0;
        }
    }

    return image;
}

// Generate input to cover these requirements:
// There should be a image with randomly generated pixel intensities.
static CT_Image box3x3_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image box3x3_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}


static uint8_t box3x3_calculate(CT_Image src, uint32_t x, uint32_t y)
{
    uint8_t res = (uint8_t)(ct_floor_u32_no_overflow( ((float)(
            (int16_t)(*CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 0)) +
                      *CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 0) +
                      *CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 0) +
                      *CT_IMAGE_DATA_PTR_8U(src, x + 0, y - 1) +
                      *CT_IMAGE_DATA_PTR_8U(src, x - 1, y - 1) +
                      *CT_IMAGE_DATA_PTR_8U(src, x + 1, y - 1) +
                      *CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 1) +
                      *CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 1) +
                      *CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 1)) )/9 ) );
    return res;
}

static uint8_t box3x3_calculate_replicate(CT_Image src, uint32_t x_, uint32_t y_)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    uint8_t res = (uint8_t)(ct_floor_u32_no_overflow( ((float)(
            (int16_t)(CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 0)) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 0) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 0) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y - 1) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y - 1) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y - 1) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 1) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 1) +
                      CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 1)) )/9 ) );
    return res;
}

static uint8_t box3x3_calculate_constant(CT_Image src, uint32_t x_, uint32_t y_, vx_uint32 constant_value)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    uint8_t res = (uint8_t)(ct_floor_u32_no_overflow( ((float)(
            (int16_t)(CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 0, constant_value)) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 0, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 0, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y - 1, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y - 1, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y - 1, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 1, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 1, constant_value) +
                      CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 1, constant_value)) )/9 ) );
    return res;
}


CT_Image box3x3_create_reference_image(CT_Image src, vx_border_mode_t border)
{
    CT_Image dst;

    CT_ASSERT_(return NULL, src->format == VX_DF_IMAGE_U8);

    dst = ct_allocate_image(src->width, src->height, src->format);

    if (border.mode == VX_BORDER_MODE_UNDEFINED)
    {
        CT_FILL_IMAGE_8U(return 0, dst,
                if (x >= 1 && y >= 1 && x < src->width - 1 && y < src->height - 1)
                {
                    uint8_t res = box3x3_calculate(src, x, y);
                    *dst_data = res;
                });
    }
    else if (border.mode == VX_BORDER_MODE_REPLICATE)
    {
        CT_FILL_IMAGE_8U(return 0, dst,
                {
                    uint8_t res = box3x3_calculate_replicate(src, x, y);
                    *dst_data = res;
                });
    }
    else if (border.mode == VX_BORDER_MODE_CONSTANT)
    {
        vx_uint32 constant_value = border.constant_value;
        CT_FILL_IMAGE_8U(return 0, dst,
                {
                    uint8_t res = box3x3_calculate_constant(src, x, y, constant_value);
                    *dst_data = res;
                });
    }
    else
    {
        ASSERT_(return 0, "NOT IMPLEMENTED");
    }
    return dst;
}


static void box3x3_check(CT_Image src, CT_Image dst, vx_border_mode_t border)
{
    CT_Image dst_ref = NULL;

    ASSERT(src && dst);

    ASSERT_NO_FAILURE(dst_ref = box3x3_create_reference_image(src, border));

    ASSERT_NO_FAILURE(
        if (border.mode == VX_BORDER_MODE_UNDEFINED)
        {
            ct_adjust_roi(dst,  1, 1, 1, 1);
            ct_adjust_roi(dst_ref, 1, 1, 1, 1);
        }
    );

    EXPECT_CTIMAGE_NEAR(dst_ref, dst, 1);
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
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_border_mode_t border;
    int width, height;
} Filter_Arg;

#define ADD_GRADIENT_STEPS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/step=3x3", __VA_ARGS__, 3, 3)), \
    CT_EXPAND(nextmacro(testArgName "/step=4x4", __VA_ARGS__, 4, 4)), \
    CT_EXPAND(nextmacro(testArgName "/step=3x4", __VA_ARGS__, 3, 4)), \
    CT_EXPAND(nextmacro(testArgName "/step=4x3", __VA_ARGS__, 4, 3))

#define BOX_PARAMETERS \
    CT_GENERATE_PARAMETERS("simple_gradient", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_GRADIENT_STEPS, ARG, box3x3_generate_simple_gradient, NULL), \
    CT_GENERATE_PARAMETERS("bi_level", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_SMALL_SET, ARG, box3x3_generate_bi_level, NULL), \
    CT_GENERATE_PARAMETERS("randomInput", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_SMALL_SET, ARG, box3x3_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ARG, box3x3_read_image, "lena.bmp")

TEST_WITH_ARG(Box3x3, testGraphProcessing, Filter_Arg,
    BOX_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = ct_create_similar_image(src_image), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxBox3x3Node(graph, src_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(box3x3_check(src, dst, border));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(Box3x3, testImmediateProcessing, Filter_Arg,
    BOX_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = ct_create_similar_image(src_image), VX_TYPE_IMAGE);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxuBox3x3(context, src_image, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(box3x3_check(src, dst, border));

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TESTCASE_TESTS(Box3x3, testNodeCreation, testGraphProcessing, testImmediateProcessing)
