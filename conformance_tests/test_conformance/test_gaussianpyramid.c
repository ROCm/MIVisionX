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

#include "shared_functions.h"

#define VX_GAUSSIAN_PYRAMID_TOLERANCE 1

TESTCASE(GaussianPyramid, CT_VXContext, ct_setup_vx_context, 0)


TEST(GaussianPyramid, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_pyramid pyr = 0;
    vx_graph graph = 0;
    vx_node node = 0;
    const vx_size levels     = 4;
    const vx_float32 scale   = VX_SCALE_PYRAMID_HALF;
    const vx_uint32 width    = 640;
    const vx_uint32 height   = 480;
    const vx_df_image format = VX_DF_IMAGE_U8;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, width, height, format), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(pyr = vxCreatePyramid(context, levels, scale, width, height, format), VX_TYPE_PYRAMID);

    {
        vx_size ch_levels;
        vx_float32 ch_scale;
        vx_uint32 ch_width, ch_height;
        vx_df_image ch_format;

        VX_CALL(vxQueryPyramid(pyr, VX_PYRAMID_ATTRIBUTE_LEVELS, &ch_levels, sizeof(ch_levels)));
        if (levels != ch_levels)
        {
            CT_FAIL("check for pyramid attribute VX_PYRAMID_ATTRIBUTE_LEVELS failed\n");
        }
        VX_CALL(vxQueryPyramid(pyr, VX_PYRAMID_ATTRIBUTE_SCALE, &ch_scale, sizeof(ch_scale)));
        if (scale != ch_scale)
        {
            CT_FAIL("check for pyramid attribute VX_PYRAMID_ATTRIBUTE_SCALE failed\n");
        }
        VX_CALL(vxQueryPyramid(pyr, VX_PYRAMID_ATTRIBUTE_WIDTH, &ch_width, sizeof(ch_width)));
        if (width != ch_width)
        {
            CT_FAIL("check for pyramid attribute VX_PYRAMID_ATTRIBUTE_WIDTH failed\n");
        }
        VX_CALL(vxQueryPyramid(pyr, VX_PYRAMID_ATTRIBUTE_HEIGHT, &ch_height, sizeof(ch_height)));
        if (height != ch_height)
        {
            CT_FAIL("check for pyramid attribute VX_PYRAMID_ATTRIBUTE_HEIGHT failed\n");
        }
        VX_CALL(vxQueryPyramid(pyr, VX_PYRAMID_ATTRIBUTE_FORMAT, &ch_format, sizeof(ch_format)));
        if (format != ch_format)
        {
            CT_FAIL("check for pyramid attribute VX_PYRAMID_ATTRIBUTE_FORMAT failed\n");
        }
    }

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxGaussianPyramidNode(graph, input, pyr), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleasePyramid(&pyr);
    vxReleaseImage(&input);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(pyr == 0);
    ASSERT(input == 0);
}

#define LEVELS_COUNT_MAX    7

static CT_Image gaussian_pyramid_generate_random(const char* fileName, int width, int height)
{
    CT_Image image;

    ASSERT_NO_FAILURE_(return 0,
            image = ct_allocate_ct_image_random(width, height, VX_DF_IMAGE_U8, &CT()->seed_, 0, 256));

    return image;
}

static CT_Image gaussian_pyramid_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static int32_t gaussian5x5_pyramid_get(int32_t *values)
{
    int32_t res = 1 * (values[ 0] + values[ 4] + values[20] + values[24]) +
                  4 * (values[ 1] + values[ 3] + values[ 5] + values[ 9] + values[15] + values[19] + values[21] + values[23]) +
                  6 * (values[ 2] + values[10] + values[14] + values[22]) +
                 16 * (values[ 6] + values[ 8] + values[16] + values[18]) +
                 24 * (values[ 7] + values[11] + values[13] + values[17]) +
                 36 * values[12];
    res = res >> 8;
    return res;
}

static uint8_t gaussian5x5_pyramid_calculate(CT_Image src, uint32_t x, uint32_t y)
{
    int32_t values[25] = {
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 2, y - 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y - 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y - 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y - 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 2, y - 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 2, y - 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y - 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y - 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y - 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 2, y - 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 2, y - 0),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y - 0),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y - 0),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y - 0),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 2, y - 0),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 2, y + 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 2, y + 1),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 2, y + 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x - 1, y + 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 0, y + 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 1, y + 2),
        (int32_t)*CT_IMAGE_DATA_PTR_8U(src, x + 2, y + 2),
    };
    return (uint8_t)gaussian5x5_pyramid_get(values);
}

static uint8_t gaussian5x5_calculate_replicate(CT_Image src, uint32_t x_, uint32_t y_)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    int32_t values[25] = {
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 2, y - 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y - 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y - 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y - 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 2, y - 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 2, y - 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y - 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y - 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y - 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 2, y - 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 2, y - 0),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y - 0),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y - 0),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y - 0),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 2, y - 0),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 2, y + 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 2, y + 1),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 2, y + 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x - 1, y + 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 0, y + 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 1, y + 2),
        (int32_t)CT_IMAGE_DATA_REPLICATE_8U(src, x + 2, y + 2),
    };
    return (uint8_t)gaussian5x5_pyramid_get(values);
}

static uint8_t gaussian5x5_calculate_constant(CT_Image src, uint32_t x_, uint32_t y_, vx_uint32 constant_value)
{
    int32_t x = (int)x_;
    int32_t y = (int)y_;
    int32_t values[25] = {
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 2, y - 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y - 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y - 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y - 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 2, y - 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 2, y - 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y - 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y - 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y - 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 2, y - 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 2, y - 0, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y - 0, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y - 0, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y - 0, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 2, y - 0, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 2, y + 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 2, y + 1, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 2, y + 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x - 1, y + 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 0, y + 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 1, y + 2, constant_value),
        (int32_t)CT_IMAGE_DATA_CONSTANT_8U(src, x + 2, y + 2, constant_value),
    };
    return (uint8_t)gaussian5x5_pyramid_get(values);
}

static vx_int32 gaussian_pyramid_get_pixel(CT_Image input, int x, int y, vx_border_mode_t border)
{
    if (border.mode == VX_BORDER_MODE_UNDEFINED)
    {
        if (x >= 2 && y >= 2 && x < (int)input->width - 2 && y < (int)input->height - 2)
            return gaussian5x5_pyramid_calculate(input, x, y);
        else
            return -1;
    }
    else if (border.mode == VX_BORDER_MODE_REPLICATE)
    {
        return gaussian5x5_calculate_replicate(input, x, y);
    }
    else if (border.mode == VX_BORDER_MODE_CONSTANT)
    {
        return gaussian5x5_calculate_constant(input, x, y, border.constant_value);
    }
    CT_FAIL_(return -1, "NOT IMPLEMENTED");
}

static void gaussian_pyramid_check_pixel(CT_Image input, CT_Image output, int x, int y, vx_border_mode_t border)
{
    vx_uint8 res = *CT_IMAGE_DATA_PTR_8U(output, x, y);

    vx_float64 x_src = (((vx_float64)x + 0.5) * (vx_float64)input->width / (vx_float64)output->width) - 0.5;
    vx_float64 y_src = (((vx_float64)y + 0.5) * (vx_float64)input->height / (vx_float64)output->height) - 0.5;
    int x_min = (int)floor(x_src), y_min = (int)floor(y_src);
    int sx, sy;
    for (sy = 0; sy <= 1; sy++)
    {
        for (sx = 0; sx <= 1; sx++)
        {
            vx_int32 candidate = 0;
            ASSERT_NO_FAILURE_(return, candidate = gaussian_pyramid_get_pixel(input, x_min + sx, y_min + sy, border));
            if (candidate == -1 || abs(candidate - res) <= VX_GAUSSIAN_PYRAMID_TOLERANCE)
                return;
        }
    }
    CT_FAIL_(return, "Check failed for pixel (%d, %d): %d", x, y, (int)res);
}

static void gaussian_pyramid_check_image(CT_Image input, CT_Image output, vx_border_mode_t border, vx_size level)
{
    ASSERT(input && output);
    if (0 == level)
    {
        EXPECT_EQ_CTIMAGE(input, output);
    }
    else
    {
        CT_FILL_IMAGE_8U(, output,
                {
                    ASSERT_NO_FAILURE(gaussian_pyramid_check_pixel(input, output, x, y, border));
                });
    }
}

static const vx_float64 c_orbscale[] =
{
    1.000000000000000000000000000000e+00,
    8.408964152537146130583778358414e-01,
    7.071067811865475727373109293694e-01,
    5.946035575013605134486738279520e-01,
    5.000000000000000000000000000000e-01,
    4.204482076268573065291889179207e-01,
    3.535533905932737308575042334269e-01,
    2.973017787506802567243369139760e-01,
    2.500000000000000000000000000000e-01,
    2.102241038134286532645944589603e-01,
    1.767766952966368654287521167134e-01,
    1.486508893753401283621684569880e-01,
    1.250000000000000000000000000000e-01,
    1.051120519067143266322972294802e-01,
    8.838834764831843271437605835672e-02,
    7.432544468767006418108422849400e-02,
    6.250000000000000000000000000000e-02,
};


static void gaussian_pyramid_check(CT_Image input, vx_pyramid pyr, vx_size levels, vx_float32 scale, vx_border_mode_t border)
{
    vx_uint32 level = 0;
    vx_image output_image = 0;
    CT_Image output_prev = NULL, output_cur = NULL;
    vx_uint32 ref_width = input->width;
    vx_uint32 ref_height = input->height;

    ASSERT(input && pyr && (1 < levels) && (level < sizeof(c_orbscale) / sizeof(float) ));
    ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(pyr, 0), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(output_prev = ct_image_from_vx_image(output_image));
    vxReleaseImage(&output_image);
    ASSERT(output_image == 0);

    gaussian_pyramid_check_image(input, output_prev, border, 0);
    if (CT_HasFailure())
    {
        printf("=== Input ===\n");
        ct_dump_image_info(input);
        printf("=== LEVEL %d ===\n", 0);
        ct_dump_image_info(output_prev);
        return;
    }

    for (level = 1; level < levels; level++)
    {
        ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(pyr, level), VX_TYPE_IMAGE);
        ASSERT_NO_FAILURE(output_cur = ct_image_from_vx_image(output_image));
        vxReleaseImage(&output_image);
        ASSERT(output_image == 0);

        if (VX_SCALE_PYRAMID_ORB == scale)
        {
            vx_float64 orb_scale = c_orbscale[level];
            if ( (output_cur->width  != ceil(orb_scale * ref_width)) ||
                 (output_cur->height != ceil(orb_scale * ref_height)))
            {
                CT_FAIL_(return, "Check failed for size of level: %d", level);
            }
        }
        else
        {
            if ( (output_cur->width != ceil(output_prev->width * scale)) ||
                 (output_cur->height != ceil(output_prev->height * scale)))
            {
                CT_FAIL_(return, "Check failed for size of level: %d", level);
            }
        }

        gaussian_pyramid_check_image(output_prev, output_cur, border, level);
        if (CT_HasFailure())
        {
            printf("=== Input ===\n");
            ct_dump_image_info(output_prev);
            printf("=== LEVEL %d ===\n", level);
            ct_dump_image_info(output_cur);
            return;
        }

        output_prev = output_cur;
    }
}



static vx_uint8 gaussian_pyramid_reference_get_pixel(CT_Image prevLevel, int dst_width, int dst_height, int x, int y, vx_border_mode_t border)
{
    vx_int32 candidate = -1;
    vx_float64 x_src = (((vx_float64)x + 0.5) * (vx_float64)prevLevel->width / (vx_float64)dst_width) - 0.5;
    vx_float64 y_src = (((vx_float64)y + 0.5) * (vx_float64)prevLevel->height / (vx_float64)dst_height) - 0.5;
    int x_int = (int)floor(x_src), y_int = (int)floor(y_src);
    vx_float64 x_f = x_src - x_int, y_f = y_src - y_int;
    if (x_f >= 0.5)
        x_int++;
    if (y_f >= 0.5)
        y_int++;
    if (x_int >= (int)prevLevel->width)
        x_int = prevLevel->width - 1;
    if (y_int >= (int)prevLevel->height)
        y_int = prevLevel->height - 1;
    ASSERT_NO_FAILURE_(return 0, candidate = gaussian_pyramid_get_pixel(prevLevel, x_int, y_int, border));
    if (candidate == -1)
        return 0;
    return CT_CAST_U8(candidate);
}

static CT_Image gaussian_pyramid_create_reference_image(CT_Image input, CT_Image prevLevel, vx_border_mode_t border, vx_float32 scale, vx_size target_level)
{
    vx_uint32 level = 0;
    CT_Image dst;
    vx_uint32 ref_width = input->width;
    vx_uint32 ref_height = input->height;
    vx_uint32 dst_width = input->width;
    vx_uint32 dst_height = input->height;

    ASSERT_(return NULL, scale < 1.0);
    ASSERT_(return NULL, input && (level < (sizeof(c_orbscale) / sizeof(float))));

    ASSERT_(return NULL, input->format == VX_DF_IMAGE_U8);

    if (VX_SCALE_PYRAMID_HALF == scale)
    {
        dst_width = ref_width;
        dst_height = ref_height;
        for (level = 1; level <= target_level; level++)
        {
            dst_width = (vx_uint32)ceil(dst_width * scale);
            dst_height = (vx_uint32)ceil(dst_height * scale);
        }
    }
    else // if (VX_SCALE_PYRAMID_ORB == scale)
    {
        vx_float64 orb_scale = c_orbscale[target_level];
        dst_width = (vx_uint32)ceil(orb_scale * ref_width);
        dst_height = (vx_uint32)ceil(orb_scale * ref_height);
    }

    dst = ct_allocate_image(dst_width, dst_height, input->format);

    if (target_level == 0)
    {
        CT_FILL_IMAGE_8U(return 0, dst,
                {
                    uint8_t res = *CT_IMAGE_DATA_PTR_8U(input, x, y);
                    *dst_data = res;
                });
    }
    else
    {
        CT_FILL_IMAGE_8U(return 0, dst,
                {
                    uint8_t res = gaussian_pyramid_reference_get_pixel(prevLevel, dst_width, dst_height, x, y, border);
                    *dst_data = res;
                });
    }

    return dst;
}

void gaussian_pyramid_fill_reference(CT_Image input, vx_pyramid pyr, vx_size levels, vx_float32 scale, vx_border_mode_t border)
{
    vx_uint32 level = 0;
    vx_image output_image = 0;
    CT_Image output_prev = NULL, output_cur = NULL;
    vx_uint32 ref_width = input->width;
    vx_uint32 ref_height = input->height;

    ASSERT(input && pyr && (levels < sizeof(c_orbscale) / sizeof(float) ));
    ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(pyr, 0), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(output_prev = ct_image_from_vx_image(output_image));

    CT_FILL_IMAGE_8U(return, output_prev,
            {
                uint8_t res = *CT_IMAGE_DATA_PTR_8U(input, x, y);
                *dst_data = res;
            });
    ASSERT_NO_FAILURE(ct_image_copyto_vx_image(output_image, output_prev));

    vxReleaseImage(&output_image);
    ASSERT(output_image == 0);

    for (level = 1; level < levels; level++)
    {
        ASSERT_VX_OBJECT(output_image = vxGetPyramidLevel(pyr, level), VX_TYPE_IMAGE);
        ASSERT_NO_FAILURE(output_cur = ct_image_from_vx_image(output_image));

        if (VX_SCALE_PYRAMID_ORB == scale)
        {
            vx_float64 orb_scale = c_orbscale[level];
            if ( (output_cur->width  != ceil(orb_scale * ref_width)) ||
                 (output_cur->height != ceil(orb_scale * ref_height)))
            {
                CT_FAIL_(return, "Check failed for size of level: %d", level);
            }
        }
        else
        {
            if ( (output_cur->width != ceil(output_prev->width * scale)) ||
                 (output_cur->height != ceil(output_prev->height * scale)))
            {
                CT_FAIL_(return, "Check failed for size of level: %d", level);
            }
        }

        ASSERT_NO_FAILURE(output_cur = gaussian_pyramid_create_reference_image(input, output_prev, border, scale, level));
        ASSERT_NO_FAILURE(ct_image_copyto_vx_image(output_image, output_cur));

        vxReleaseImage(&output_image);
        ASSERT(output_image == 0);

        output_prev = output_cur;
    }
}


static vx_size gaussian_pyramid_calc_max_levels_count(int width, int height, vx_float32 scale)
{
    vx_size level = 1;
    while ((16 <= width) && (16 <= height) && level < LEVELS_COUNT_MAX)
    {
        level++;
        width = (int)ceil((vx_float64)width * scale);
        height = (int)ceil((vx_float64)height * scale);
    }
    return level;
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height);
    const char* fileName;
    vx_border_mode_t border;
    int width, height;
    vx_float32 scale;
} Arg;

#define ADD_VX_SCALE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_SCALE_PYRAMID_HALF", __VA_ARGS__, VX_SCALE_PYRAMID_HALF)), \
    CT_EXPAND(nextmacro(testArgName "/VX_SCALE_PYRAMID_ORB", __VA_ARGS__, VX_SCALE_PYRAMID_ORB))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("randomInput", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_SMALL_SET, ADD_VX_SCALE, ARG, gaussian_pyramid_generate_random, NULL), \
    CT_GENERATE_PARAMETERS("lena", ADD_VX_BORDERS_REQUIRE_UNDEFINED_ONLY, ADD_SIZE_NONE, ADD_VX_SCALE, ARG, gaussian_pyramid_read_image, "lena.bmp")

TEST_WITH_ARG(GaussianPyramid, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_size levels;

    vx_context context = context_->vx_context_;
    vx_image input_image = 0;
    vx_pyramid pyr = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    CT_Image input = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT(arg_->scale < 1.0);

    ASSERT_NO_FAILURE(input = arg_->generator( arg_->fileName, arg_->width, arg_->height));
    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = gaussian_pyramid_calc_max_levels_count(input->width, input->height, arg_->scale);

    ASSERT_VX_OBJECT(pyr = vxCreatePyramid(context, levels, arg_->scale, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node = vxGaussianPyramidNode(graph, input_image, pyr), VX_TYPE_NODE);

    if (border.mode != VX_BORDER_MODE_UNDEFINED)
        VX_CALL(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    CT_ASSERT_NO_FAILURE_(, gaussian_pyramid_check(input, pyr, levels, arg_->scale, arg_->border));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    ASSERT(node == 0);
    ASSERT(graph == 0);

    vxReleasePyramid(&pyr);
    vxReleaseImage(&input_image);
    ASSERT(pyr == 0);
    ASSERT(input_image == 0);
}


TEST_WITH_ARG(GaussianPyramid, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_size levels;

    vx_context context = context_->vx_context_;
    vx_image input_image = 0;
    vx_pyramid pyr = 0;

    CT_Image input = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator( arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = gaussian_pyramid_calc_max_levels_count(input->width, input->height, arg_->scale);

    ASSERT_VX_OBJECT(pyr = vxCreatePyramid(context, levels, arg_->scale, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    VX_CALL(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border)));

    VX_CALL(vxuGaussianPyramid(context, input_image, pyr));

    CT_ASSERT_NO_FAILURE_(, gaussian_pyramid_check(input, pyr, levels, arg_->scale, arg_->border));

    vxReleasePyramid(&pyr);
    vxReleaseImage(&input_image);
    ASSERT(pyr == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(GaussianPyramid, testReference, Arg,
    PARAMETERS
)
{
    vx_size levels;

    vx_context context = context_->vx_context_;
    vx_image input_image = 0;
    vx_pyramid pyr = 0;

    CT_Image input = NULL;

    vx_border_mode_t border = arg_->border;

    ASSERT_NO_FAILURE(input = arg_->generator( arg_->fileName, arg_->width, arg_->height));

    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    levels = gaussian_pyramid_calc_max_levels_count(input->width, input->height, arg_->scale);

    ASSERT_VX_OBJECT(pyr = vxCreatePyramid(context, levels, arg_->scale, input->width, input->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    ASSERT_NO_FAILURE(gaussian_pyramid_fill_reference(input, pyr, levels, arg_->scale, border));

    CT_ASSERT_NO_FAILURE_(, gaussian_pyramid_check(input, pyr, levels, arg_->scale, arg_->border));

    vxReleasePyramid(&pyr);
    vxReleaseImage(&input_image);
    ASSERT(pyr == 0);
    ASSERT(input_image == 0);
}

TESTCASE_TESTS(GaussianPyramid,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing,
        testReference
)



