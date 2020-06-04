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


TESTCASE(LUT, CT_VXContext, ct_setup_vx_context, 0)

static vx_lut lut_create256(vx_context context, vx_uint8* data)
{
    vx_lut lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    void* ptr = NULL;

    ASSERT_VX_OBJECT_(return 0, lut, VX_TYPE_LUT);

    VX_CALL_(return 0, vxAccessLUT(lut, &ptr, VX_WRITE_ONLY));
    ASSERT_(return 0, ptr);
    memcpy(ptr, data, sizeof(data[0])*256);
    VX_CALL_(return 0, vxCommitLUT(lut, ptr));
    return lut;
}

static void lut_data_fill_identity(vx_uint8* data)
{
    int i;

    for (i = 0; i < 256; i++)
        data[i] = i;
}

static void lut_data_fill_random(vx_uint8* data)
{
    uint64_t* seed = &CT()->seed_;
    int i;

    for (i = 0; i < 256; i++)
        data[i] = (vx_uint8)CT_RNG_NEXT_INT(*seed, 0, 256);
}

TEST(LUT, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_uint8 lut_data[256];
    vx_lut lut = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    ASSERT_VX_OBJECT(src_image, VX_TYPE_IMAGE);

    dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    ASSERT_VX_OBJECT(dst_image, VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(lut_data_fill_identity(lut_data));
    ASSERT_VX_OBJECT(lut = lut_create256(context, lut_data), VX_TYPE_LUT);

    {
        vx_enum lut_type;
        vx_size lut_count, lut_size;
        VX_CALL(vxQueryLUT(lut, VX_LUT_ATTRIBUTE_TYPE, &lut_type, sizeof(lut_type)));
        if (VX_TYPE_UINT8 != lut_type)
        {
            CT_FAIL("check for LUT attribute VX_LUT_ATTRIBUTE_TYPE failed\n");
        }
        VX_CALL(vxQueryLUT(lut, VX_LUT_ATTRIBUTE_COUNT, &lut_count, sizeof(lut_count)));
        if (256 != lut_count)
        {
            CT_FAIL("check for LUT attribute VX_LUT_ATTRIBUTE_COUNT failed\n");
        }
        VX_CALL(vxQueryLUT(lut, VX_LUT_ATTRIBUTE_SIZE, &lut_size, sizeof(lut_size)));
        if (256 > lut_size)
        {
            CT_FAIL("check for LUT attribute VX_LUT_ATTRIBUTE_SIZE failed\n");
        }
    }

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    node = vxTableLookupNode(graph, src_image, lut, dst_image);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseLUT(&lut);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(lut == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}


// Generate input to cover these requirements:
// There should be a image with randomly generated pixel intensities.
static CT_Image lut_image_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image lut_image_read(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static vx_uint8 lut_calculate(CT_Image src, uint32_t x, uint32_t y, vx_uint8* lut_data)
{
    vx_uint8 value = *CT_IMAGE_DATA_PTR_8U(src, x, y);
    vx_uint8 res = lut_data[value];
    return res;
}

static CT_Image lut_create_reference_image(CT_Image src, vx_uint8* lut_data)
{
    CT_Image dst;

    CT_ASSERT_(return NULL, src->format == VX_DF_IMAGE_U8);

    dst = ct_allocate_image(src->width, src->height, src->format);

    CT_FILL_IMAGE_8U(return 0, dst,
            {
                uint8_t res = lut_calculate(src, x, y, lut_data);
                *dst_data = res;
            });
    return dst;
}


static void lut_check(CT_Image src, CT_Image dst, vx_uint8* lut_data)
{
    CT_Image dst_ref = NULL;

    ASSERT(src && dst);

    ASSERT_NO_FAILURE(dst_ref = lut_create_reference_image(src, lut_data));

    EXPECT_EQ_CTIMAGE(dst_ref, dst);
#if 0
    if (CT_HasFailure())
    {
        int i = 0;
        printf("=== LUT ===\n");
        for (i = 0; i < 256; i++)
        {
            printf("%3d:%3d ", i, (int)lut_data[i]);
        }
        printf("\n");
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
    void (*lut_generator)(vx_uint8* data);
    int width, height;
} Arg;

#define ADD_LUT_GENERATOR(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/LutIdentity", __VA_ARGS__, lut_data_fill_identity)), \
    CT_EXPAND(nextmacro(testArgName "/LutRandom", __VA_ARGS__, lut_data_fill_random))

#define LUT_PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_LUT_GENERATOR, ADD_SIZE_SMALL_SET, ARG, lut_image_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_LUT_GENERATOR, ADD_SIZE_NONE, ARG, lut_image_read, "lena.bmp")

TEST_WITH_ARG(LUT, testGraphProcessing, Arg,
    LUT_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image src = NULL, dst = NULL;
    vx_uint8 lut_data[256];
    vx_lut lut;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(arg_->lut_generator(lut_data));
    ASSERT_VX_OBJECT(lut = lut_create256(context, lut_data), VX_TYPE_LUT);

    dst_image = ct_create_similar_image(src_image);
    ASSERT_VX_OBJECT(dst_image, VX_TYPE_IMAGE);

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    node = vxTableLookupNode(graph, src_image, lut, dst_image);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(lut_check(src, dst, lut_data));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseLUT(&lut);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(lut == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TEST_WITH_ARG(LUT, testImmediateProcessing, Arg,
    LUT_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;

    CT_Image src = NULL, dst = NULL;
    vx_uint8 lut_data[256];
    vx_lut lut;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(arg_->lut_generator(lut_data));
    ASSERT_VX_OBJECT(lut = lut_create256(context, lut_data), VX_TYPE_LUT);

    dst_image = ct_create_similar_image(src_image);
    ASSERT_VX_OBJECT(dst_image, VX_TYPE_IMAGE);

    VX_CALL(vxuTableLookup(context, src_image, lut, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(lut_check(src, dst, lut_data));

    vxReleaseLUT(&lut);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(lut == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);
}

TESTCASE_TESTS(LUT, testNodeCreation, testGraphProcessing, testImmediateProcessing)
