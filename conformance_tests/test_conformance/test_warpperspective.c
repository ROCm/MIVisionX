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

static CT_Image own_read_image_8u(const char* fileName, int width, int height)
{
    CT_Image image = NULL;

    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);

    return image;
}

static CT_Image own_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}


TESTCASE(WarpPerspective, CT_VXContext, ct_setup_vx_context, 0)

TEST(WarpPerspective, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0, output = 0;
    vx_matrix matrix = 0;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_enum type = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
    const vx_enum matrix_type = VX_TYPE_FLOAT32;
    const vx_size matrix_rows = 3;
    const vx_size matrix_cols = 3;
    const vx_size matrix_data_size = 4 * matrix_rows * matrix_cols;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(matrix = vxCreateMatrix(context, matrix_type, matrix_cols, matrix_rows), VX_TYPE_MATRIX);

    {
        vx_enum ch_matrix_type;
        vx_size ch_matrix_rows, ch_matrix_cols, ch_data_size;

        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_TYPE, &ch_matrix_type, sizeof(ch_matrix_type)));
        if (matrix_type != ch_matrix_type)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_ATTRIBUTE_TYPE failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_ROWS, &ch_matrix_rows, sizeof(ch_matrix_rows)));
        if (matrix_rows != ch_matrix_rows)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_ATTRIBUTE_ROWS failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_COLUMNS, &ch_matrix_cols, sizeof(ch_matrix_cols)));
        if (matrix_cols != ch_matrix_cols)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_ATTRIBUTE_COLUMNS failed\n");
        }
        VX_CALL(vxQueryMatrix(matrix, VX_MATRIX_ATTRIBUTE_SIZE, &ch_data_size, sizeof(ch_data_size)));
        if (matrix_data_size > ch_data_size)
        {
            CT_FAIL("check for Matrix attribute VX_MATRIX_ATTRIBUTE_SIZE failed\n");
        }
    }

    graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxWarpPerspectiveNode(graph, input, matrix, type, output), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseMatrix(&matrix);
    vxReleaseImage(&output);
    vxReleaseImage(&input);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(matrix == 0);
    ASSERT(output == 0);
    ASSERT(input == 0);
}


enum CT_PerspectiveMatrixType {
    VX_MATRIX_IDENT = 0,
    VX_MATRIX_SCALE,
    VX_MATRIX_SCALE_ROTATE,
    VX_MATRIX_RANDOM
};

#define VX_NN_AREA_SIZE         1.5
#define VX_BILINEAR_TOLERANCE   1

static CT_Image warp_perspective_read_image_8u(const char* fileName, int width, int height)
{
    CT_Image image = NULL;

    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);

    return image;
}

static CT_Image warp_perspective_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

#define RND_FLT(low, high)      (vx_float32)CT_RNG_NEXT_REAL(CT()->seed_, low, high);
static void warp_perspective_generate_matrix(vx_float32 *m, int src_width, int src_height, int dst_width, int dst_height, int type)
{
    vx_float32 mat[3][3];
    vx_float32 angle, scale_x, scale_y, cos_a, sin_a;
    if (VX_MATRIX_IDENT == type)
    {
        mat[0][0] = 1.f;
        mat[0][1] = 0.f;
        mat[0][2] = 0.f;

        mat[1][0] = 0.f;
        mat[1][1] = 1.f;
        mat[1][2] = 0.f;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
        mat[2][2] = 1.f;
    }
    else if (VX_MATRIX_SCALE == type)
    {
        scale_x = src_width / (vx_float32)dst_width;
        scale_y = src_height / (vx_float32)dst_height;

        mat[0][0] = scale_x;
        mat[0][1] = 0.f;
        mat[0][2] = 0.f;

        mat[1][0] = 0.f;
        mat[1][1] = scale_y;
        mat[1][2] = 0.f;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
        mat[2][2] = 1.f;
    }
    else if (VX_MATRIX_SCALE_ROTATE == type)
    {
        angle = M_PIF / RND_FLT(3.f, 6.f);
        scale_x = src_width / (vx_float32)dst_width;
        scale_y = src_height / (vx_float32)dst_height;
        cos_a = cosf(angle);
        sin_a = sinf(angle);

        mat[0][0] = cos_a * scale_x;
        mat[0][1] = sin_a * scale_y;
        mat[0][2] = 0.f;

        mat[1][0] = -sin_a * scale_x;
        mat[1][1] = cos_a  * scale_y;
        mat[1][2] = 0.f;

        mat[2][0] = 0.f;
        mat[2][1] = 0.f;
        mat[2][2] = 1.f;
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
        mat[0][2] = RND_FLT(0.f, 0.1f);

        mat[1][0] = -sin_a * RND_FLT(scale_y / 2.f, scale_y);
        mat[1][1] = cos_a  * RND_FLT(scale_x / 2.f, scale_x);
        mat[1][2] = RND_FLT(0.f, 0.1f);

        mat[2][0] = src_width  / 5.f * RND_FLT(-1.f, 1.f);
        mat[2][1] = src_height / 5.f * RND_FLT(-1.f, 1.f);
        mat[2][2] = 1.f;
    }
    memcpy(m, mat, sizeof(mat));
}

static vx_matrix warp_perspective_create_matrix(vx_context context, vx_float32 *m)
{
    vx_matrix matrix;
    matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
    if (vxGetStatus((vx_reference)matrix) == VX_SUCCESS)
    {
        if (VX_SUCCESS != vxWriteMatrix(matrix, m))
        {
            vxReleaseMatrix(&matrix);
            return 0;
        }
    }
    return matrix;
}

static int warp_perspective_check_pixel(CT_Image input, CT_Image output, int x, int y, vx_enum interp_type, vx_border_mode_t border, vx_float32 *m)
{
    vx_float64 x0, y0, z0, xlower, ylower, s, t;
    vx_int32 xi, yi;
    int candidate;
    vx_uint8 res = *CT_IMAGE_DATA_PTR_8U(output, x, y);

    x0 = (vx_float64)m[3 * 0 + 0] * (vx_float64)x + (vx_float64)m[3 * 1 + 0] * (vx_float64)y + (vx_float64)m[3 * 2 + 0];
    y0 = (vx_float64)m[3 * 0 + 1] * (vx_float64)x + (vx_float64)m[3 * 1 + 1] * (vx_float64)y + (vx_float64)m[3 * 2 + 1];
    z0 = (vx_float64)m[3 * 0 + 2] * (vx_float64)x + (vx_float64)m[3 * 1 + 2] * (vx_float64)y + (vx_float64)m[3 * 2 + 2];
    if (fabs(z0) < DBL_MIN)
        return 0;

    x0 = x0 / z0;
    y0 = y0 / z0;
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
            candidate = (int)((1. - s) * (1. - t) * (vx_float32)CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi    , border.constant_value) +
                                    s  * (1. - t) * (vx_float32)CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi    , border.constant_value) +
                              (1. - s) *       t  * (vx_float32)CT_IMAGE_DATA_CONSTANT_8U(input, xi    , yi + 1, border.constant_value) +
                                    s  *       t  * (vx_float32)CT_IMAGE_DATA_CONSTANT_8U(input, xi + 1, yi + 1, border.constant_value));
        }
        if (candidate == -1 || (abs(candidate - res) <= VX_BILINEAR_TOLERANCE))
            return 0;
        return 1;
    }
    CT_FAIL_(return 1, "Interpolation type undefined");
}

static void warp_perspective_validate(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_mode_t border, vx_float32 *m)
{
    vx_uint32 err_count = 0;

    CT_FILL_IMAGE_8U(, output,
            {
                ASSERT_NO_FAILURE(err_count += warp_perspective_check_pixel(input, output, x, y, interp_type, border, m));
            });
    if (10 * err_count > output->width * output->height)
        CT_FAIL_(return, "Check failed for %d pixels", err_count);
}

static void warp_perspective_check(CT_Image input, CT_Image output, vx_enum interp_type, vx_border_mode_t border, vx_float32* m)
{
    ASSERT(input && output);
    ASSERT( (interp_type == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) ||
            (interp_type == VX_INTERPOLATION_TYPE_BILINEAR));

    ASSERT( (border.mode == VX_BORDER_MODE_UNDEFINED) ||
            (border.mode == VX_BORDER_MODE_CONSTANT));

    warp_perspective_validate(input, output, interp_type, border, m);
    if (CT_HasFailure())
    {
        printf("=== INPUT ===\n");
        ct_dump_image_info(input);
        printf("=== OUTPUT ===\n");
        ct_dump_image_info(output);
        printf("Matrix:\n%g %g %g\n%g %g %g\n%g %g %g\n",
                m[0], m[3], m[6],
                m[1], m[4], m[7],
                m[2], m[5], m[8]);
    }
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height);
    const char*      fileName;
    int src_width, src_height;
    int width, height;
    vx_border_mode_t border;
    vx_enum interp_type;
    int matrix_type;
} Arg;

#define ADD_VX_BORDERS_WARP_PERSPECTIVE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_UNDEFINED", __VA_ARGS__, { VX_BORDER_MODE_UNDEFINED, 0 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=0", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 0 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=1", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 1 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=127", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 127 })), \
    CT_EXPAND(nextmacro(testArgName "/VX_BORDER_MODE_CONSTANT=255", __VA_ARGS__, { VX_BORDER_MODE_CONSTANT, 255 }))

#define ADD_VX_INTERP_TYPE_WARP_PERSPECTIVE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR)), \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_BILINEAR", __VA_ARGS__, VX_INTERPOLATION_TYPE_BILINEAR ))

#define ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR", __VA_ARGS__, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR))

#define ADD_VX_MATRIX_PARAM_WARP_PERSPECTIVE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_IDENT", __VA_ARGS__,        VX_MATRIX_IDENT)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_SCALE", __VA_ARGS__,        VX_MATRIX_SCALE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_SCALE_ROTATE", __VA_ARGS__, VX_MATRIX_SCALE_ROTATE)), \
    CT_EXPAND(nextmacro(testArgName "/VX_MATRIX_RANDOM", __VA_ARGS__,       VX_MATRIX_RANDOM))


#define PARAMETERS \
    CT_GENERATE_PARAMETERS("random", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_WARP_PERSPECTIVE, ADD_VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, ADD_VX_MATRIX_PARAM_WARP_PERSPECTIVE, ARG, own_generate_random, NULL, 128, 128), \
    CT_GENERATE_PARAMETERS("lena", ADD_SIZE_SMALL_SET, ADD_VX_BORDERS_WARP_PERSPECTIVE, ADD_VX_INTERP_TYPE_WARP_PERSPECTIVE, ADD_VX_MATRIX_PARAM_WARP_PERSPECTIVE, ARG, own_read_image_8u, "lena.bmp", 0, 0)


TEST_WITH_ARG(WarpPerspective, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_image input_image = 0, output_image = 0;
    vx_matrix matrix = 0;
    vx_float32 m[9];

    CT_Image input = NULL, output = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->src_width, arg_->src_height));
    ASSERT_NO_FAILURE(output = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(warp_perspective_generate_matrix(m, input->width, input->height, arg_->width, arg_->height, arg_->matrix_type));
    ASSERT_VX_OBJECT(matrix = warp_perspective_create_matrix(context, m), VX_TYPE_MATRIX);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxWarpPerspectiveNode(graph, input_image, matrix, arg_->interp_type, output_image), VX_TYPE_NODE);

    VX_CALL(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));
    ASSERT_NO_FAILURE(warp_perspective_check(input, output, arg_->interp_type, arg_->border, m));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseMatrix(&matrix);
    vxReleaseImage(&output_image);
    vxReleaseImage(&input_image);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(matrix == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(WarpPerspective, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image input_image = 0, output_image = 0;
    vx_matrix matrix = 0;
    vx_float32 m[9];

    CT_Image input = NULL, output = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator(arg_->fileName, arg_->src_width, arg_->src_height));
    ASSERT_NO_FAILURE(output = ct_allocate_image(arg_->width, arg_->height, VX_DF_IMAGE_U8));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(output_image = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(warp_perspective_generate_matrix(m, input->width, input->height, arg_->width, arg_->height, arg_->matrix_type));
    ASSERT_VX_OBJECT(matrix = warp_perspective_create_matrix(context, m), VX_TYPE_MATRIX);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxuWarpPerspective(context, input_image, matrix, arg_->interp_type, output_image));

    ASSERT_NO_FAILURE(output = ct_image_from_vx_image(output_image));

    ASSERT_NO_FAILURE(warp_perspective_check(input, output, arg_->interp_type, arg_->border, m));

    vxReleaseMatrix(&matrix);
    vxReleaseImage(&output_image);
    vxReleaseImage(&input_image);

    ASSERT(matrix == 0);
    ASSERT(output_image == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(WarpPerspective,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
