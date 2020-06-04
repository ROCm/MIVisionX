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
#include <float.h>

#ifndef M_PI
#define M_PIF   3.14159265358979323846f
#else
#define M_PIF   (vx_float32)M_PI
#endif


TESTCASE(Remap, CT_VXContext, ct_setup_vx_context, 0)

TEST(Remap, testNodeCreation)
{
    vx_uint32 i;
    vx_uint32 j;
    vx_context context = context_->vx_context_;
    vx_image input = 0, output = 0;
    vx_uint32 src_width;
    vx_uint32 src_height;
    vx_uint32 dst_width;
    vx_uint32 dst_height;
    vx_remap map = 0;
    vx_enum attr_name[] =
    {
        VX_REMAP_ATTRIBUTE_SOURCE_WIDTH,
        VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT,
        VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH,
        VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT
    };
    vx_uint32 attr_val[4];
    vx_graph graph = 0;
    vx_node node = 0;
    vx_enum interp = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;

    src_width = 16;
    src_height = 32;
    dst_width = 128;
    dst_height = 64;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, dst_width, dst_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    map = vxCreateRemap(context, src_width, src_height, dst_width, dst_height);
    ASSERT_VX_OBJECT(map, VX_TYPE_REMAP);

    for (i = 0; i < dst_height; i++)
    {
        for (j = 0; j < dst_width; j++)
        {
            VX_CALL(vxSetRemapPoint(map, j, i, (vx_float32)j, (vx_float32)i));
        }
    }

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    node = vxRemapNode(graph, input, map, interp, output);
    ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
    VX_CALL(vxQueryRemap(map, attr_name[0], &attr_val[0], sizeof(attr_val[0])));
    if (attr_val[0] != src_width)
    {
        CT_FAIL("check for remap attribute VX_REMAP_ATTRIBUTE_SOURCE_WIDTH failed\n");
    }
    VX_CALL(vxQueryRemap(map, attr_name[1], &attr_val[1], sizeof(attr_val[1])));
    if (attr_val[1] != src_height)
    {
        CT_FAIL("check for remap attribute VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT failed\n");
    }
    VX_CALL(vxQueryRemap(map, attr_name[2], &attr_val[2], sizeof(attr_val[2])));
    if (attr_val[2] != dst_width)
    {
        CT_FAIL("check for remap attribute VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH failed\n");
    }
    VX_CALL(vxQueryRemap(map, attr_name[3], &attr_val[3], sizeof(attr_val[3])));
    if (attr_val[3] != dst_height)
    {
        CT_FAIL("check for remap attribute VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT failed\n");
    }

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseImage(&output);
    vxReleaseRemap(&map);
    vxReleaseImage(&input);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(output == 0);
    ASSERT(map == 0);
    ASSERT(input == 0);
}

#define SRC_WIDTH       128
#define SRC_HEIGHT      128

#define VX_MAP_IDENT         0
#define VX_MAP_SCALE         1
#define VX_MAP_SCALE_ROTATE  2
#define VX_MAP_RANDOM        3

#define VX_NN_AREA_SIZE         1.5
#define VX_BILINEAR_TOLERANCE   1

static CT_Image remap_read_image_8u(const char* fileName, int width, int height)
{
    CT_Image image = NULL;

    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);

    return image;
}

static CT_Image remap_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

#define RND_FLT(low, high)      (vx_float32)CT_RNG_NEXT_REAL(CT()->seed_, low, high);

static vx_remap remap_generate_map(vx_context context, int src_width, int src_height, int dst_width, int dst_height, int type)
{
    vx_uint32 i;
    vx_uint32 j;
    vx_float32 x;
    vx_float32 y;
    vx_remap map = 0;
    vx_status status;

    map = vxCreateRemap(context, src_width, src_height, dst_width, dst_height);
    if (vxGetStatus((vx_reference)map) == VX_SUCCESS)
    {
        vx_float32 mat[3][2];
        vx_float32 angle, scale_x, scale_y, cos_a, sin_a;
        if (VX_MAP_IDENT == type)
        {
            mat[0][0] = 1.f;
            mat[0][1] = 0.f;

            mat[1][0] = 0.f;
            mat[1][1] = 1.f;

            mat[2][0] = 0.f;
            mat[2][1] = 0.f;
        }
        else if (VX_MAP_SCALE == type)
        {
            scale_x = src_width  / (vx_float32)dst_width;
            scale_y = src_height / (vx_float32)dst_height;

            mat[0][0] = scale_x;
            mat[0][1] = 0.f;

            mat[1][0] = 0.f;
            mat[1][1] = scale_y;

            mat[2][0] = 0.f;
            mat[2][1] = 0.f;
        }
        else if (VX_MAP_SCALE_ROTATE == type)
        {
            angle = M_PIF / RND_FLT(3.f, 6.f);
            scale_x = src_width  / (vx_float32)dst_width;
            scale_y = src_height / (vx_float32)dst_height;
            cos_a = cosf(angle);
            sin_a = sinf(angle);

            mat[0][0] = cos_a * scale_x;
            mat[0][1] = sin_a * scale_y;

            mat[1][0] = -sin_a * scale_x;
            mat[1][1] = cos_a  * scale_y;

            mat[2][0] = 0.f;
            mat[2][1] = 0.f;
        }
        else// if (VX_MATRIX_RANDOM == type)
        {
            angle = M_PIF / RND_FLT(3.f, 6.f);
            scale_x = src_width / (vx_float32)dst_width;
            scale_y = src_height / (vx_float32)dst_height;
            cos_a = cosf(angle);
            sin_a = sinf(angle);

            mat[0][0] = cos_a * RND_FLT(scale_x / 2.f, scale_x);
            mat[0][1] = sin_a * RND_FLT(scale_y / 2.f, scale_y);

            mat[1][0] = -sin_a * RND_FLT(scale_y / 2.f, scale_y);
            mat[1][1] = cos_a  * RND_FLT(scale_x / 2.f, scale_x);

            mat[2][0] = src_width  / 5.f * RND_FLT(-1.f, 1.f);
            mat[2][1] = src_height / 5.f * RND_FLT(-1.f, 1.f);
        }

        for (i = 0; i < (vx_uint32)dst_height; i++)
        {
            for (j = 0; j < (vx_uint32)dst_width; j++)
            {
                x = j * mat[0][0] + i * mat[1][0] + mat[2][0];
                y = j * mat[0][1] + i * mat[1][1] + mat[2][1];
                status = vxSetRemapPoint(map, j, i, x, y);
                if (VX_SUCCESS != status)
                    return 0;
            }
        }
    }

    return map;
}

static int remap_check_pixel(CT_Image input, CT_Image output, int x, int y, vx_enum interp_type, vx_border_mode_t border, vx_remap map)
{
    vx_float32 _x0;
    vx_float32 _y0;
    vx_float64 x0, y0, xlower, ylower, s, t;
    vx_int32 xi, yi;
    int candidate;
    vx_uint8 res = *CT_IMAGE_DATA_PTR_8U(output, x, y);

    VX_CALL_RET(vxGetRemapPoint(map, x, y, &_x0, &_y0));

    x0 = (vx_float64)_x0;
    y0 = (vx_float64)_y0;

    if (VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR == interp_type)
    {
        for (yi = (vx_int32)ceil(y0 - VX_NN_AREA_SIZE); (vx_float64)yi <= y0 + VX_NN_AREA_SIZE; yi++)
        {
            for (xi = (vx_int32)ceil(x0 - VX_NN_AREA_SIZE); (vx_float64)xi <= x0 + VX_NN_AREA_SIZE; xi++)
            {
                if (0 <= xi && 0 <= yi && xi < (vx_int32)input->width && yi < (vx_int32)input->height)
                {
                    candidate = *CT_IMAGE_DATA_PTR_8U(input, xi, yi);
                }
                else if (VX_BORDER_MODE_CONSTANT == border.mode)
                {
                    candidate = border.constant_value;
                }
                else
                {
                    candidate = -1;
                }
                if (candidate == -1 || candidate == res)
                    return 0;
            }
        }
        CT_FAIL_(return 1, "Check failed for pixel (%d, %d): %d", x, y, (int)res);
    }
    else if (VX_INTERPOLATION_TYPE_BILINEAR == interp_type)
    {
        xlower = floor(x0);
        ylower = floor(y0);

        s = x0 - xlower;
        t = y0 - ylower;

        xi = (vx_int32)xlower;
        yi = (vx_int32)ylower;

        candidate = -1;
        if (VX_BORDER_MODE_UNDEFINED == border.mode)
        {
            if (xi >= 0 && yi >= 0 && xi < (vx_int32)input->width - 1 && yi < (vx_int32)input->height - 1)
            {
                candidate = (int)((1. - s) * (1. - t) * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi    , yi    ) +
                                        s  * (1. - t) * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi + 1, yi    ) +
                                  (1. - s) *       t  * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi    , yi + 1) +
                                        s  *       t  * (vx_float64) *CT_IMAGE_DATA_PTR_8U(input, xi + 1, yi + 1));
            }
        }
        else if (VX_BORDER_MODE_CONSTANT == border.mode)
        {
            candidate = (int)((1. - s) * (1. - t) * (vx_float64)CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi    , border.constant_value) +
                                    s  * (1. - t) * (vx_float64)CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi    , border.constant_value) +
                              (1. - s) *       t  * (vx_float64)CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi + 1, border.constant_value) +
                                    s  *       t  * (vx_float64)CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi + 1, border.constant_value));
        }
        if (candidate == -1 || (abs(candidate - res) <= VX_BILINEAR_TOLERANCE))
            return 0;
        return 1;
    }
    CT_FAIL_(return 1, "Interpolation type undefined");
}

static void remap_validate(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_mode_t border, vx_remap map)
{
    vx_uint32 err_count = 0;

    CT_FILL_IMAGE_8U(, output,
            {
                ASSERT_NO_FAILURE(err_count += remap_check_pixel(input, output, x, y, interp_type, border, map));
            });
    if (10 * err_count > output->width * output->height)
        CT_FAIL_(return, "Check failed for %d pixels", err_count);
}

static void remap_check(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_mode_t border, vx_remap map)
{
    ASSERT(input && output);
    ASSERT( (interp_type == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) ||
            (interp_type == VX_INTERPOLATION_TYPE_BILINEAR));

    ASSERT( (border.mode == VX_BORDER_MODE_UNDEFINED) ||
            (border.mode == VX_BORDER_MODE_CONSTANT));

    remap_validate(input, output, interp_type, border, map);
    if (CT_HasFailure())
    {
        printf("=== INPUT ===\n");
        ct_dump_image_info(input);
        printf("=== OUTPUT ===\n");
        ct_dump_image_info(output);
    }
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char*      fileName;
    int width, height;
    vx_border_mode_t border;
    vx_enum interp_type;
    int map_type;
} Arg;

#define ADD_VX_BORDERS_REMAP(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_UNDEFINED", __VA_ARGS__, { VX_BORDER_MODE_UNDEFINED, 0 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=0", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 0 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=1", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 1 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=127", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 127 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=255", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 255 }))

#define ADD_VX_INTERP_TYPE_REMAP(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_BILINEAR", __VA_ARGS__, VX_INTERPOLATION_TYPE_BILINEAR ))

#define ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR))

#define ADD_VX_INTERPOLATION_TYPE_BILINEAR(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_BILINEAR", __VA_ARGS__, VX_INTERPOLATION_TYPE_BILINEAR ))

#define ADD_VX_MAP_PARAM_REMAP(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_MAP_IDENT", __VA_ARGS__,        VX_MAP_IDENT)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MAP_SCALE", __VA_ARGS__,        VX_MAP_SCALE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MAP_SCALE_ROTATE", __VA_ARGS__, VX_MAP_SCALE_ROTATE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MAP_RANDOM", __VA_ARGS__,       VX_MAP_RANDOM))


#define PARAMETERS \
    CT_GENERATE_PARAMETERS("random", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_REMAP, ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, ADD_VX_MAP_PARAM_REMAP, ARG, remap_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_REMAP, ADD_VX_INTERP_TYPE_REMAP, ADD_VX_MAP_PARAM_REMAP, ARG, remap_read_image_8u, "lena.bmp")


TEST_WITH_ARG(Remap, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image input_image = 0, output_image = 0;
    vx_remap map = 0;

    CT_Image input = NULL, output = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, SRC_WIDTH, SRC_HEIGHT));
    ASSERT_NO_FAILURE(output = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(map = remap_generate_map(context, input->width, input->height, arg_->width, arg_->height, arg_->map_type), VX_TYPE_REMAP);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxRemapNode(graph, input_image, map, arg_->interp_type, output_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));
    ASSERT_NO_FAILURE(remap_check(input, output, arg_->interp_type, arg_->border, map));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseRemap(&map);
    vxReleaseImage(&output_image);
    vxReleaseImage(&input_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(map == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(Remap, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, output_image = 0;
    vx_remap map = 0;

    CT_Image input = NULL, output = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, SRC_WIDTH, SRC_HEIGHT));
    ASSERT_NO_FAILURE(output = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(map = remap_generate_map(context, input->width, input->height, arg_->width, arg_->height, arg_->map_type), VX_TYPE_REMAP);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxuRemap(context, input_image, map, arg_->interp_type, output_image));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    ASSERT_NO_FAILURE(remap_check(input, output, arg_->interp_type, arg_->border, map));

    vxReleaseRemap(&map);
    vxReleaseImage(&output_image);
    vxReleaseImage(&input_image);

    ASSERT(map == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(Remap,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
