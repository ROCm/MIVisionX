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
#include <float.h>
#include <math.h>

#define PTS_SEARCH_RADIUS   5

TESTCASE(HarrisCorners, CT_VXContext, ct_setup_vx_context, 0)

TEST(HarrisCorners, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_graph graph = 0;
    vx_node node = 0;

    vx_float32 strength_thresh = 0.001f;
    vx_float32 min_distance = 3.f;
    vx_float32 sensitivity = 0.04f;
    vx_size num_corners = 100;

    vx_scalar strength_thresh_scalar, min_distance_scalar, sensitivity_scalar, num_corners_scalar;
    vx_array corners;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(strength_thresh_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(min_distance_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &min_distance), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(sensitivity_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &sensitivity), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(num_corners_scalar = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 100), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(node = vxHarrisCornersNode(graph, input, strength_thresh_scalar, min_distance_scalar, sensitivity_scalar, 3, 3, corners, num_corners_scalar), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseArray(&corners);
    vxReleaseScalar(&num_corners_scalar);
    vxReleaseScalar(&sensitivity_scalar);
    vxReleaseScalar(&min_distance_scalar);
    vxReleaseScalar(&strength_thresh_scalar);
    vxReleaseImage(&input);

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(corners == 0);
    ASSERT(num_corners_scalar == 0);
    ASSERT(sensitivity_scalar == 0);
    ASSERT(min_distance_scalar == 0);
    ASSERT(strength_thresh_scalar == 0);
    ASSERT(input == 0);
}

typedef struct {
    vx_size         num_corners;
    vx_float32      strength_thresh;
    vx_keypoint_t   *pts;
} TruthData;

static vx_size harris_corner_read_line(const char *data, char *line)
{
    const char* ptr = data;
    int pos_temp = 0;
    while (*ptr && *ptr != '\n')
    {
        line[pos_temp] = *ptr;
        pos_temp++; ptr++;
    }
    line[pos_temp] = 0;
    return (ptr - data);
}

static void harris_corner_read_truth_data(const char *file_path, TruthData *truth_data, float strengthScale)
{
    FILE* f;
    long sz;
    void* buf; char* ptr;
    char temp[1024];
    vx_size ln_size = 0;
    vx_size pts_count = 0;
    vx_keypoint_t *pt;

    ASSERT(truth_data && file_path);

    f = fopen(file_path, "rb");
    ASSERT(f);
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    ASSERT(sz);
    fseek(f, 0, SEEK_SET);

    ASSERT(buf = malloc(sz + 1));
    ASSERT(sz == fread(buf, 1, sz, f));

    fclose(f);

    ptr = (char *)buf;
    ptr[sz] = 0;
    ln_size = harris_corner_read_line(ptr, temp);
    ASSERT(ln_size);
    truth_data->num_corners = atoi(temp);
    ASSERT(truth_data->num_corners);
    ptr+= ln_size + 1;

    ASSERT(truth_data->pts = (vx_keypoint_t *)malloc(truth_data->num_corners * sizeof(vx_keypoint_t)));
    pt = truth_data->pts;
    for (;pts_count < truth_data->num_corners; ptr += ln_size + 1, pt++, pts_count++)
    {
        ln_size = harris_corner_read_line(ptr, temp);
        if (0 == ln_size)
            break;
        sscanf(temp, "%d %d %f", &pt->x, &pt->y, &pt->strength);
        pt->strength *= strengthScale;
    }
    free(buf);

    ASSERT(pts_count == truth_data->num_corners);
    truth_data->strength_thresh = truth_data->pts[truth_data->num_corners - 1].strength - FLT_EPSILON;
}

static int harris_corner_search_truth_point(vx_int32 x, vx_int32 y, vx_float32 strength, const TruthData *truth_data)
{
    vx_int32 xmin = x - PTS_SEARCH_RADIUS;
    vx_int32 xmax = x + PTS_SEARCH_RADIUS;
    vx_int32 ymin = y - PTS_SEARCH_RADIUS;
    vx_int32 ymax = y + PTS_SEARCH_RADIUS;
    vx_size num;
    if (FLT_MIN >= strength)
        return 1;
    for (num = 0; num < truth_data->num_corners; num++)
    {
        if ((xmin <= truth_data->pts[num].x) && (truth_data->pts[num].x <= xmax) &&
            (ymin <= truth_data->pts[num].y) && (truth_data->pts[num].y <= ymax))
        {
            if (fabs(log10(truth_data->pts[num].strength/strength)) < 1.1)
                return 0;
        }
    }
    return 1;
}

static int harris_corner_search_test_point(vx_int32 x, vx_int32 y, vx_float32 strength, char *test_pts, vx_size pts_stride, vx_size num_pts)
{
    vx_int32 xmin = x - PTS_SEARCH_RADIUS;
    vx_int32 xmax = x + PTS_SEARCH_RADIUS;
    vx_int32 ymin = y - PTS_SEARCH_RADIUS;
    vx_int32 ymax = y + PTS_SEARCH_RADIUS;
    vx_size num;
    vx_keypoint_t *pt = NULL;
    if (FLT_MIN >= strength)
        return 1;
    for (num = 0; num < num_pts; num++, test_pts += pts_stride)
    {
        pt = (vx_keypoint_t *)test_pts;
        if ((xmin <= pt->x) && (pt->x <= xmax) &&
            (ymin <= pt->y) && (pt->y <= ymax))
        {
            if (fabs(log10(pt->strength / strength)) < 1.1)
                return 0;
        }
    }
    return 1;
}

static void harris_corner_check(vx_array corners, const TruthData *truth_data)
{
    vx_enum type;
    vx_size i, num_corners, stride;
    char *pts = NULL;
    char *pts_ptr = NULL;
    vx_keypoint_t *pt = NULL;
    vx_int32 fail_count = 0;

    ASSERT(corners && truth_data);
    ASSERT(VX_SUCCESS == vxQueryArray(corners, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)));
    ASSERT(VX_TYPE_KEYPOINT == type);

    ASSERT(VX_SUCCESS == vxQueryArray(corners, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_corners, sizeof(num_corners)));

    ASSERT(num_corners);
    ASSERT(VX_SUCCESS == vxAccessArrayRange(corners, 0, num_corners, &stride, (void **)&pts, VX_READ_ONLY));

    pts_ptr = pts;
    for (i = 0; i < num_corners; i++, pts_ptr += stride)
    {
        pt = (vx_keypoint_t *)pts_ptr;
        if (harris_corner_search_truth_point(pt->x, pt->y, pt->strength, truth_data))
            fail_count++;
    }
    if (100 * fail_count > 10 * (vx_int32)i)
    {
        CT_FAIL_(goto cleanup, "Too much (%d) test points, which don't have corresponding truth data points", fail_count);
    }
    fail_count = 0;
    for (i = 0; i < truth_data->num_corners; i++)
    {
        if (harris_corner_search_test_point(truth_data->pts[i].x, truth_data->pts[i].y, truth_data->pts[i].strength, pts, stride, num_corners))
            fail_count++;
    }
    if (100 * fail_count > 10 * (vx_int32)i)
    {
        CT_FAIL_(goto cleanup, "Too much (%d) truth data points, which don't have corresponding test points", fail_count);
    }
cleanup:
    vxCommitArrayRange(corners, 0, num_corners, pts);
}

typedef struct {
    const char* testName;
    const char* filePrefix;
    vx_float32 min_distance;
    vx_float32 sensitivity;
    vx_int32  gradient_size;
    vx_int32  block_size;
} Arg;

#define ADD_VX_MIN_DISTANCE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/MIN_DISTANCE=3.0", __VA_ARGS__, 3.f)), \
    CT_EXPAND(nextmacro(testArgName "/MIN_DISTANCE=5.0", __VA_ARGS__, 5.f))

#define ADD_VX_SENSITIVITY(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/SENSITIVITY=0.04", __VA_ARGS__, 0.04f)), \
    CT_EXPAND(nextmacro(testArgName "/SENSITIVITY=0.10", __VA_ARGS__, 0.10f)), \
    CT_EXPAND(nextmacro(testArgName "/SENSITIVITY=0.15", __VA_ARGS__, 0.15f))

#define ADD_VX_GRADIENT_SIZE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/GRADIENT_SIZE=3", __VA_ARGS__, 3)), \
    CT_EXPAND(nextmacro(testArgName "/GRADIENT_SIZE=5", __VA_ARGS__, 5)), \
    CT_EXPAND(nextmacro(testArgName "/GRADIENT_SIZE=7_DISABLED", __VA_ARGS__, 7))


#define ADD_VX_BLOCK_SIZE(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/BLOCK_SIZE=3", __VA_ARGS__, 3)), \
    CT_EXPAND(nextmacro(testArgName "/BLOCK_SIZE=5", __VA_ARGS__, 5)), \
    CT_EXPAND(nextmacro(testArgName "/BLOCK_SIZE=7", __VA_ARGS__, 7))

#define PARAMETERS \
    CT_GENERATE_PARAMETERS("few_strong_corners",  ADD_VX_MIN_DISTANCE, ADD_VX_SENSITIVITY, ADD_VX_GRADIENT_SIZE, ADD_VX_BLOCK_SIZE, ARG, "hc_fsc"), \
    CT_GENERATE_PARAMETERS("many_strong_corners", ADD_VX_MIN_DISTANCE, ADD_VX_SENSITIVITY, ADD_VX_GRADIENT_SIZE, ADD_VX_BLOCK_SIZE, ARG, "hc_msc")

TEST_WITH_ARG(HarrisCorners, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_image input_image = 0;
    vx_float32 strength_thresh;
    vx_float32 min_distance = arg_->min_distance + FLT_EPSILON;
    vx_float32 sensitivity = arg_->sensitivity;
    vx_size num_corners;
    vx_graph graph = 0;
    vx_node node = 0;

    vx_scalar strength_thresh_scalar, min_distance_scalar, sensitivity_scalar, num_corners_scalar;
    vx_array corners;

    char filepath[2048];

    CT_Image input = NULL;
    TruthData truth_data;

    double scale = 1.0 / ((1 << (arg_->gradient_size - 1)) * arg_->block_size * 255.0);
    scale = scale * scale * scale * scale;

    sprintf(filepath, "harriscorners/%s_%0.2f_%0.2f_%d_%d.txt", arg_->filePrefix, arg_->min_distance, arg_->sensitivity, arg_->gradient_size, arg_->block_size);
    ASSERT_NO_FAILURE(harris_corner_read_truth_data(filepath, &truth_data, (float)scale));

    strength_thresh = truth_data.strength_thresh;

    sprintf(filepath, "harriscorners/%s.bmp", arg_->filePrefix);

    ASSERT_NO_FAILURE(input = ct_read_image(filepath, 1));
    ASSERT(input && (input->format == VX_DF_IMAGE_U8));
    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    num_corners = input->width * input->height / 10;

    ASSERT_VX_OBJECT(strength_thresh_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(min_distance_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &min_distance), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(sensitivity_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &sensitivity), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(num_corners_scalar = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(corners = vxCreateArray(context, VX_TYPE_KEYPOINT, num_corners), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxHarrisCornersNode(graph, input_image, strength_thresh_scalar, min_distance_scalar,
                                                sensitivity_scalar, arg_->gradient_size, arg_->block_size, corners,
                                                num_corners_scalar), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    CT_ASSERT_NO_FAILURE_(, harris_corner_check(corners, &truth_data));

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    ASSERT(node == 0);
    ASSERT(graph == 0);

    free(truth_data.pts); truth_data.pts = 0;
    vxReleaseArray(&corners);
    vxReleaseScalar(&num_corners_scalar);
    vxReleaseScalar(&sensitivity_scalar);
    vxReleaseScalar(&min_distance_scalar);
    vxReleaseScalar(&strength_thresh_scalar);
    vxReleaseImage(&input_image);

    ASSERT(truth_data.pts == 0);
    ASSERT(corners == 0);
    ASSERT(num_corners_scalar == 0);
    ASSERT(sensitivity_scalar == 0);
    ASSERT(min_distance_scalar == 0);
    ASSERT(strength_thresh_scalar == 0);
    ASSERT(input_image == 0);
}

TEST_WITH_ARG(HarrisCorners, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_image input_image = 0;
    vx_float32 strength_thresh;
    vx_float32 min_distance = arg_->min_distance + FLT_EPSILON;
    vx_float32 sensitivity = arg_->sensitivity;
    vx_size num_corners;

    vx_scalar strength_thresh_scalar, min_distance_scalar, sensitivity_scalar, num_corners_scalar;
    vx_array corners;

    char filepath[2048];

    CT_Image input = NULL;
    TruthData truth_data;

    double scale = 1.0 / ((1 << (arg_->gradient_size - 1)) * arg_->block_size * 255.0);
    scale = scale * scale * scale * scale;

    sprintf(filepath, "harriscorners/%s_%0.2f_%0.2f_%d_%d.txt", arg_->filePrefix, arg_->min_distance, arg_->sensitivity, arg_->gradient_size, arg_->block_size);
    ASSERT_NO_FAILURE(harris_corner_read_truth_data(filepath, &truth_data, (float)scale));

    strength_thresh = truth_data.strength_thresh;

    sprintf(filepath, "harriscorners/%s.bmp", arg_->filePrefix);

    ASSERT_NO_FAILURE(input = ct_read_image(filepath, 1));
    ASSERT(input && (input->format == VX_DF_IMAGE_U8));
    ASSERT_VX_OBJECT(input_image = ct_image_to_vx_image(input, context), VX_TYPE_IMAGE);

    num_corners = input->width * input->height / 10;

    ASSERT_VX_OBJECT(strength_thresh_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(min_distance_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &min_distance), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(sensitivity_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &sensitivity), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(num_corners_scalar = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(corners = vxCreateArray(context, VX_TYPE_KEYPOINT, num_corners), VX_TYPE_ARRAY);

    VX_CALL(vxuHarrisCorners(context, input_image, strength_thresh_scalar, min_distance_scalar, sensitivity_scalar, arg_->gradient_size, arg_->block_size, corners, num_corners_scalar));

    CT_ASSERT_NO_FAILURE_(, harris_corner_check(corners, &truth_data));

    free(truth_data.pts); truth_data.pts = 0;
    vxReleaseArray(&corners);
    vxReleaseScalar(&num_corners_scalar);
    vxReleaseScalar(&sensitivity_scalar);
    vxReleaseScalar(&min_distance_scalar);
    vxReleaseScalar(&strength_thresh_scalar);
    vxReleaseImage(&input_image);

    ASSERT(truth_data.pts == 0);
    ASSERT(corners == 0);
    ASSERT(num_corners_scalar == 0);
    ASSERT(sensitivity_scalar == 0);
    ASSERT(min_distance_scalar == 0);
    ASSERT(strength_thresh_scalar == 0);
    ASSERT(input_image == 0);
}


TESTCASE_TESTS(HarrisCorners,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
)
