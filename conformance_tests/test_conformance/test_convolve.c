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

#define MAX_CONV_SIZE 15

TESTCASE(Convolve, CT_VXContext, ct_setup_vx_context, 0)

static vx_convolution convolution_create(vx_context context, int cols, int rows, vx_int16* data, vx_uint32 scale)
{
    vx_convolution convolution = vxCreateConvolution(context, cols, rows);
    vx_size size = 0;

    ASSERT_VX_OBJECT_(return 0, convolution, VX_TYPE_CONVOLUTION);

    VX_CALL_(return 0, vxQueryConvolution(convolution, VX_CONVOLUTION_ATTRIBUTE_SIZE, &size, sizeof(size)));

    VX_CALL_(return 0, vxWriteConvolutionCoefficients(convolution, data));

    VX_CALL_(return 0, vxSetConvolutionAttribute(convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &scale, sizeof(scale)));

    return convolution;
}

static void convolution_data_fill_identity(int cols, int rows, vx_int16* data)
{
    int x = cols / 2, y = rows / 2;
    memset(data, 0, sizeof(vx_int16) * cols * rows);
    data[y * cols + x] = 1;
}

static void convolution_data_fill_random_32768(int cols, int rows, vx_int16* data)
{
    uint64_t* seed = &CT()->seed_;
    int i;

    for (i = 0; i < cols * rows; i++)
        data[i] = (vx_uint8)CT_RNG_NEXT_INT(*seed, -32768, 32768);
}

static void convolution_data_fill_random_128(int cols, int rows, vx_int16* data)
{
    uint64_t* seed = &CT()->seed_;
    int i;

    for (i = 0; i < cols * rows; i++)
        data[i] = (vx_uint8)CT_RNG_NEXT_INT(*seed, -128, 128);
}


TEST(Convolve, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    int cols = 3, rows = 3;
    vx_int16 data[3 * 3] = { 0, 0, 0, 0, 1, 0, 0, 0, 0};
    vx_convolution convolution = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(convolution = convolution_create(context, cols, rows, data, 1), VX_TYPE_CONVOLUTION);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxConvolveNode(graph, src_image, convolution, dst_image), VX_TYPE_NODE);

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);

    vxReleaseConvolution(&convolution);
    ASSERT(convolution == NULL);
}


static CT_Image convolve_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image convolve_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static int32_t convolve_get(CT_Image src, int32_t x, int32_t y, vx_border_mode_t border,
        int cols, int rows, vx_int16* data, vx_uint32 scale, vx_df_image dst_format)
{
    int i;
    int32_t sum = 0, value = 0;
    int32_t src_data[MAX_CONV_SIZE * MAX_CONV_SIZE] = { 0 };

    ASSERT_(return 0, cols <= MAX_CONV_SIZE);
    ASSERT_(return 0, rows <= MAX_CONV_SIZE);

    ASSERT_NO_FAILURE_(return 0,
            ct_image_read_rect_S32(src, src_data, x - cols / 2, y - rows / 2, x + cols / 2, y + rows / 2, border));

    for (i = 0; i < cols * rows; ++i)
        sum += src_data[i] * data[cols * rows - 1 - i];

    value = sum / scale;

    if (dst_format == VX_DF_IMAGE_U8)
    {
        if (value < 0) value = 0;
        else if (value > UINT8_MAX) value = UINT8_MAX;
    }
    else if (dst_format == VX_DF_IMAGE_S16)
    {
        if (value < INT16_MIN) value = INT16_MIN;
        else if (value > INT16_MAX) value = INT16_MAX;
    }

    return value;
}


static CT_Image convolve_create_reference_image(CT_Image src, vx_border_mode_t border,
        int cols, int rows, vx_int16* data, vx_uint32 scale, vx_df_image dst_format)
{
    CT_Image dst;

    CT_ASSERT_(return NULL, src->format == VX_DF_IMAGE_U8);

    dst = ct_allocate_image(src->width, src->height, dst_format);

    if (dst_format == VX_DF_IMAGE_U8)
    {
        CT_FILL_IMAGE_8U(return 0, dst,
                {
                    int32_t res = convolve_get(src, x, y, border, cols, rows, data, scale, dst_format);
                    *dst_data = (vx_uint8)res;
                });
    }
    else if (dst_format == VX_DF_IMAGE_S16)
    {
        CT_FILL_IMAGE_16S(return 0, dst,
                {
                    int32_t res = convolve_get(src, x, y, border, cols, rows, data, scale, dst_format);
                    *dst_data = (vx_int16)res;
                });
    }
    else
    {
        CT_FAIL_(return 0, "NOT IMPLEMENTED");
    }
    return dst;
}


static void convolve_check(CT_Image src, CT_Image dst, vx_border_mode_t border,
        int cols, int rows, vx_int16* data, vx_uint32 scale, vx_df_image dst_format)
{
    CT_Image dst_ref = NULL;

    ASSERT(src && dst);

    ASSERT_NO_FAILURE(dst_ref = convolve_create_reference_image(src, border, cols, rows, data, scale, dst_format));

    ASSERT_NO_FAILURE(
        if (border.mode == VX_BORDER_MODE_UNDEFINED)
        {
            ct_adjust_roi(dst, cols / 2, rows / 2, cols / 2, rows / 2);
            ct_adjust_roi(dst_ref, cols / 2, rows / 2, cols / 2, rows / 2);
        }
    );

    EXPECT_CTIMAGE_NEAR(dst_ref, dst, 1);
#if 0
    if (CT_HasFailure())
    {
        printf("=== SRC ===\n");
        ct_dump_image_info_ex(src, 16, 4);
        printf("=== DST ===\n");
        ct_dump_image_info_ex(dst, 16, 4);
        printf("=== EXPECTED ===\n");
        ct_dump_image_info_ex(dst_ref, 16, 4);
    }
#endif
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height);
    const char* fileName;
    int cols, rows;
    vx_uint32 scale;
    void (*convolution_data_generator)(int cols, int rows, vx_int16* data);
    vx_df_image dst_format;
    vx_border_mode_t border;
    int width, height;
} Arg;


#define ADD_CONV_SIZE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/conv=3x3", __VA_ARGS__, 3, 3)), \
    CT_EXPAND(nextmacro(testArgName "/conv=9x9", __VA_ARGS__, 9, 9)), \
    CT_EXPAND(nextmacro(testArgName "/conv=9x3", __VA_ARGS__, 9, 3)), \
    CT_EXPAND(nextmacro(testArgName "/conv=3x9", __VA_ARGS__, 3, 9)), \
    CT_EXPAND(nextmacro(testArgName "/conv=5x5", __VA_ARGS__, 5, 5)), \
    CT_EXPAND(nextmacro(testArgName "/conv=7x7", __VA_ARGS__, 7, 7))

#define ADD_CONV_SCALE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=1", __VA_ARGS__, 1)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=2", __VA_ARGS__, 2)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=4", __VA_ARGS__, 4)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=8", __VA_ARGS__, 8)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=16", __VA_ARGS__, 16)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=256", __VA_ARGS__, 256)), \
    CT_EXPAND(nextmacro(testArgName "/conv_scale=2^30", __VA_ARGS__, (1ll<<30)))

#define ADD_CONV_GENERATORS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/conv_fill=identity", __VA_ARGS__, convolution_data_fill_identity)), \
    CT_EXPAND(nextmacro(testArgName "/conv_fill=random128", __VA_ARGS__, convolution_data_fill_random_128)), \
    CT_EXPAND(nextmacro(testArgName "/conv_fill=random32768", __VA_ARGS__, convolution_data_fill_random_32768))

#define ADD_CONV_DST_FORMAT(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/dst8U", __VA_ARGS__, VX_DF_IMAGE_U8)), \
    CT_EXPAND(nextmacro(testArgName "/dst16S", __VA_ARGS__, VX_DF_IMAGE_S16))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_CONV_SIZE, ADD_CONV_SCALE, ADD_CONV_GENERATORS, ADD_CONV_DST_FORMAT, ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_64x64, ARG, convolve_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_CONV_SIZE, ADD_CONV_SCALE, ADD_CONV_GENERATORS, ADD_CONV_DST_FORMAT, ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ARG, convolve_read_image, "lena.bmp")

TEST_WITH_ARG(Convolve, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_convolution convolution = 0;
    vx_int16 data[MAX_CONV_SIZE * MAX_CONV_SIZE] = { 0 };
    vx_size conv_max_dim = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, src->width, src->height, arg_->dst_format), VX_TYPE_IMAGE);

    VX_CALL(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION, &conv_max_dim, sizeof(conv_max_dim)));

    if ((vx_size)arg_->cols > conv_max_dim || (vx_size)arg_->rows > conv_max_dim)
    {
        printf("%dx%d convolution is not supported. Skip test\n", (int)arg_->cols, (int)arg_->rows);
        return;
    }

    ASSERT_NO_FAILURE(arg_->convolution_data_generator(arg_->cols, arg_->rows, data));
    ASSERT_NO_FAILURE(convolution = convolution_create(context, arg_->cols, arg_->rows, data, arg_->scale));

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxConvolveNode(graph, src_image, convolution, dst_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(convolve_check(src, dst, border, arg_->cols, arg_->rows, data, arg_->scale, arg_->dst_format));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);

    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);

    vxReleaseConvolution(&convolution);
    ASSERT(convolution == NULL);
}

TEST_WITH_ARG(Convolve, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_convolution convolution = 0;
    vx_int16 data[MAX_CONV_SIZE * MAX_CONV_SIZE] = { 0 };
    vx_size conv_max_dim = 0;

    CT_Image src = NULL, dst = NULL;
    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, src->width, src->height, arg_->dst_format), VX_TYPE_IMAGE);

    VX_CALL(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_CONVOLUTION_MAXIMUM_DIMENSION, &conv_max_dim, sizeof(conv_max_dim)));

    if ((vx_size)arg_->cols > conv_max_dim || (vx_size)arg_->rows > conv_max_dim)
    {
        printf("%dx%d convolution is not supported. Skip test\n", (int)arg_->cols, (int)arg_->rows);
        return;
    }

    ASSERT_NO_FAILURE(arg_->convolution_data_generator(arg_->cols, arg_->rows, data));
    ASSERT_NO_FAILURE(convolution = convolution_create(context, arg_->cols, arg_->rows, data, arg_->scale));

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxuConvolve(context, src_image, convolution, dst_image));

    ASSERT_NO_FAILURE(dst = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(convolve_check(src, dst, border, arg_->cols, arg_->rows, data, arg_->scale, arg_->dst_format));

    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(dst_image == 0);
    ASSERT(src_image == 0);

    vxReleaseConvolution(&convolution);
    ASSERT(convolution == NULL);
}

TESTCASE_TESTS(Convolve, testNodeCreation, testGraphProcessing, testImmediateProcessing)
