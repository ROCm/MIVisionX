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

#define MAX_POINTS 100

TESTCASE(OptFlowPyrLK, CT_VXContext, ct_setup_vx_context, 0)

static vx_array own_create_keypoint_array(vx_context context, vx_size count, vx_keypoint_t* keypoints)
{
    vx_array arr = 0;

    ASSERT_VX_OBJECT_(return 0, arr = vxCreateArray(context, VX_TYPE_KEYPOINT, count), VX_TYPE_ARRAY);

#if 0
    {
    vx_size i;
    vx_size stride = 0;
    void* ptr = 0;

    VX_CALL_(return 0, vxAccessArrayRange(arr, 0, count, &stride, &ptr, VX_WRITE_ONLY));

    for (i = 0; i < count; i++)
    {
        vx_keypoint_t* k = (vx_keypoint_t*)(((char*)ptr) + i * stride);
        memcpy(k, &keypoints[i], sizeof(vx_keypoint_t));
    }

    VX_CALL_(return 0, vxCommitArrayRange(arr, 0, count, ptr));
    }
#else
    VX_CALL_(return 0, vxAddArrayItems(arr, count, keypoints, 0));
#endif

    return arr;
}

TEST(OptFlowPyrLK, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image src_image[2] = {0, 0};
    vx_pyramid src_pyr[2] = {0, 0};
    vx_keypoint_t kp[] = {
            {10, 10, 1, 0, 0, 1, 0},
            {20, 10, 1, 0, 0, 1, 0},
            {20, 20, 1, 0, 0, 1, 0}
    };
    vx_array old_points_arr = 0, new_points_arr = 0;
    vx_float32 eps = 0.01f;
    vx_uint32 num_iter = 10;
    vx_bool use_estimations = vx_true_e;
    vx_scalar vx_eps = 0, vx_num_iter = 0, vx_use_estimations = 0;
    vx_size winSize = 5;
    vx_graph graph = 0;
    vx_node src_pyr_node[2] = {0, 0}, node = 0;

    ASSERT_VX_OBJECT(src_image[0] = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_image[1] = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(old_points_arr = own_create_keypoint_array(context, sizeof(kp) / sizeof(kp[0]), kp), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(new_points_arr = vxCreateArray(context, VX_TYPE_KEYPOINT, sizeof(kp) / sizeof(kp[0])), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(vx_eps = vxCreateScalar(context, VX_TYPE_FLOAT32, &eps), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_num_iter = vxCreateScalar(context, VX_TYPE_UINT32, &num_iter), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_use_estimations = vxCreateScalar(context, VX_TYPE_BOOL, &use_estimations), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(src_pyr[0] = vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(src_pyr[1] = vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_PYRAMID);

    ASSERT_VX_OBJECT(src_pyr_node[0] = vxGaussianPyramidNode(graph, src_image[0], src_pyr[0]), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(src_pyr_node[1] = vxGaussianPyramidNode(graph, src_image[1], src_pyr[1]), VX_TYPE_NODE);

    ASSERT_VX_OBJECT(node = vxOpticalFlowPyrLKNode(graph, src_pyr[0], src_pyr[1],
            old_points_arr, old_points_arr, new_points_arr,
            VX_TERM_CRITERIA_BOTH, vx_eps, vx_num_iter, vx_use_estimations,
            winSize), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    vxReleaseNode(&node); vxReleaseNode(&src_pyr_node[0]); vxReleaseNode(&src_pyr_node[1]);
    vxReleaseGraph(&graph);
    vxReleaseScalar(&vx_eps); vxReleaseScalar(&vx_num_iter); vxReleaseScalar(&vx_use_estimations);
    vxReleaseArray(&old_points_arr); vxReleaseArray(&new_points_arr);
    vxReleasePyramid(&src_pyr[0]); vxReleasePyramid(&src_pyr[1]);
    vxReleaseImage(&src_image[0]); vxReleaseImage(&src_image[1]);
}


static CT_Image optflow_pyrlk_read_image(const char* fileName, int width, int height)
{
    CT_Image image = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    image = ct_read_image(fileName, 1);
    ASSERT_(return 0, image);
    ASSERT_(return 0, image->format == VX_DF_IMAGE_U8);
    return image;
}

static vx_size own_read_keypoints(const char* fileName, vx_keypoint_t** p_old_points, vx_keypoint_t** p_new_points)
{
    size_t sz = 0;
    void* buf = 0;
#if 1
    FILE* f = fopen(fileName, "rb");

    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    ASSERT_(return 0, buf = malloc(sz + 1));
    ASSERT_(return 0, sz == fread(buf, 1, sz, f));
    fclose(f); f = NULL;
    ((char*)buf)[sz] = 0;
#else
    sz = ...
    buf = ...
#endif

    ASSERT_(return 0, *p_old_points = malloc(sizeof(vx_keypoint_t) * MAX_POINTS));
    ASSERT_(return 0, *p_new_points = malloc(sizeof(vx_keypoint_t) * MAX_POINTS));

    {
        int num = 0;
        char* pos = buf;
        char* next = 0;
        while(pos && (next = strchr(pos, '\n')))
        {
            int id = 0, status = 0;
            float x1, y1, x2, y2;

            int res;

            *next = 0;
            res = sscanf(pos, "%d %d %g %g %g %g", &id, &status, &x1, &y1, &x2, &y2);
            pos = next + 1;
            if (res == 6)
            {
                (*p_old_points)[num].x = (vx_int32)x1;
                (*p_old_points)[num].y = (vx_int32)y1;
                (*p_old_points)[num].strength = 1;
                (*p_old_points)[num].scale = 0;
                (*p_old_points)[num].orientation = 0;
                (*p_old_points)[num].tracking_status = 1;
                (*p_old_points)[num].error = 0;

                (*p_new_points)[num].x = (vx_int32)x2;
                (*p_new_points)[num].y = (vx_int32)y2;
                (*p_new_points)[num].strength = 1;
                (*p_new_points)[num].scale = 0;
                (*p_new_points)[num].orientation = 0;
                (*p_new_points)[num].tracking_status = status;
                (*p_new_points)[num].error = 0;

                num++;
            }
            else
                break;
        }

        free(buf);

        return num;
    }
}

static void own_keypoints_check(vx_size num_points,
        vx_keypoint_t* old_points, vx_keypoint_t* new_points_ref, vx_keypoint_t* new_points)
{
    vx_size i;
    int num_valid_points = 0;
    int num_lost = 0;
    int num_errors = 0;
    int num_tracked_points = 0;

    for (i = 0; i < num_points; i++)
    {
        vx_int32 dx, dy;
        if (new_points_ref[i].tracking_status == 0)
            continue;
        num_valid_points++;
        if (new_points[i].tracking_status == 0)
        {
            num_lost++;
            continue;
        }
        num_tracked_points++;
        dx = new_points_ref[i].x - new_points[i].x;
        dy = new_points_ref[i].y - new_points[i].y;
        if ((dx * dx + dy * dy) > 2 * 2)
        {
            num_errors++;
        }
    }

    if (num_lost > (int)(num_valid_points * 0.05f))
        CT_ADD_FAILURE("Too many lost points: %d (threshold %d)\n",
                num_lost, (int)(num_valid_points * 0.05f));
    if (num_errors > (int)(num_tracked_points * 0.10f))
        CT_ADD_FAILURE("Too many bad points: %d (threshold %d, both tracked points %d)\n",
                num_errors, (int)(num_tracked_points * 0.10f), num_tracked_points);

#if 0
    if (CT_HasFailure())
    {
        for (i = 0; i < num_points; i++)
        {
            printf("i=%d status = %d->%d  x =  %d -> %d ? %d    y = %d -> %d ? %d\n", (int)i,
                    new_points_ref[i].tracking_status, new_points[i].tracking_status,
                    old_points[i].x, new_points_ref[i].x, new_points[i].x,
                    old_points[i].y, new_points_ref[i].y, new_points[i].y);
        }
    }
#endif
}

typedef struct {
    const char* testName;
    CT_Image (*generator)(const char* fileName, int width, int height);
    const char* src1_fileName;
    const char* src2_fileName;
    const char* points_fileName;
    vx_size winSize;
    int useReferencePyramid;
} Arg;


#define PARAMETERS \
    ARG("case1/5x5/ReferencePyramid", optflow_pyrlk_read_image, "optflow_00.bmp", "optflow_01.bmp", "optflow_pyrlk_5x5.txt", 5, 1), \
    ARG("case1/9x9/ReferencePyramid", optflow_pyrlk_read_image, "optflow_00.bmp", "optflow_01.bmp", "optflow_pyrlk_9x9.txt", 9, 1), \
    ARG("DISABLED_case1/5x5", optflow_pyrlk_read_image, "optflow_00.bmp", "optflow_01.bmp", "optflow_pyrlk_5x5.txt", 5, 0), \
    ARG("DISABLED_case1/9x9", optflow_pyrlk_read_image, "optflow_00.bmp", "optflow_01.bmp", "optflow_pyrlk_9x9.txt", 9, 0), \

TEST_WITH_ARG(OptFlowPyrLK, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image[2] = {0, 0};
    vx_pyramid src_pyr[2] = {0, 0};
    vx_array old_points_arr = 0, new_points_arr = 0;
    vx_float32 eps = 0.001f;
    vx_uint32 num_iter = 100;
    vx_bool use_estimations = vx_true_e;
    vx_scalar vx_eps = 0, vx_num_iter = 0, vx_use_estimations = 0;
    vx_size winSize = arg_->winSize;
    vx_graph graph = 0;
    vx_node src_pyr_node[2] = {0, 0}, node = 0;

    vx_size num_points = 0;
    vx_keypoint_t *old_points = 0, *new_points_ref = 0, *new_points = 0;
    vx_size new_points_size = 0;

    vx_size max_window_dim = 0;

    CT_Image src_ct_image[2] = {0, 0};

    VX_CALL(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_OPTICAL_FLOW_WINDOW_MAXIMUM_DIMENSION, &max_window_dim, sizeof(max_window_dim)));
    if (winSize > max_window_dim)
    {
        printf("%d window dim is not supported. Skip test\n", (int)winSize);
        return;
    }

    ASSERT_NO_FAILURE(src_ct_image[0] = arg_->generator(arg_->src1_fileName, 0, 0));
    ASSERT_NO_FAILURE(src_ct_image[1] = arg_->generator(arg_->src2_fileName, 0, 0));

    ASSERT_VX_OBJECT(src_image[0] = ct_image_to_vx_image(src_ct_image[0], context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_image[1] = ct_image_to_vx_image(src_ct_image[1], context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(src_pyr[0] = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, src_ct_image[0]->width, src_ct_image[0]->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(src_pyr[1] = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, src_ct_image[0]->width, src_ct_image[0]->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    ASSERT_NO_FAILURE(num_points = own_read_keypoints(arg_->points_fileName, &old_points, &new_points_ref));

    ASSERT_VX_OBJECT(old_points_arr = own_create_keypoint_array(context, num_points, old_points), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(new_points_arr = vxCreateArray(context, VX_TYPE_KEYPOINT, num_points), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(vx_eps = vxCreateScalar(context, VX_TYPE_FLOAT32, &eps), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_num_iter = vxCreateScalar(context, VX_TYPE_UINT32, &num_iter), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_use_estimations = vxCreateScalar(context, VX_TYPE_BOOL, &use_estimations), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    if (arg_->useReferencePyramid)
    {
        vx_border_mode_t border = { VX_BORDER_MODE_UNDEFINED };
        ASSERT_NO_FAILURE(gaussian_pyramid_fill_reference(src_ct_image[0], src_pyr[0], 4, VX_SCALE_PYRAMID_HALF, border));
        ASSERT_NO_FAILURE(gaussian_pyramid_fill_reference(src_ct_image[1], src_pyr[1], 4, VX_SCALE_PYRAMID_HALF, border));
    }
    else
    {
#if 0
        ASSERT_VX_OBJECT(src_pyr_node[0] = vxGaussianPyramidNode(graph, src_image[0], src_pyr[0]), VX_TYPE_NODE);
        ASSERT_VX_OBJECT(src_pyr_node[1] = vxGaussianPyramidNode(graph, src_image[1], src_pyr[1]), VX_TYPE_NODE);
#else
        VX_CALL(vxuGaussianPyramid(context, src_image[0], src_pyr[0]));
        VX_CALL(vxuGaussianPyramid(context, src_image[1], src_pyr[1]));
#endif
    }

    ASSERT_VX_OBJECT(node = vxOpticalFlowPyrLKNode(graph, src_pyr[0], src_pyr[1],
            old_points_arr, old_points_arr, new_points_arr,
            VX_TERM_CRITERIA_BOTH, vx_eps, vx_num_iter, vx_use_estimations,
            winSize), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));
    VX_CALL(vxProcessGraph(graph)); // it is ok to call processing twice, isn't it?

    ASSERT(VX_TYPE_KEYPOINT == ct_read_array(new_points_arr, (void**)&new_points, 0, &new_points_size, 0));
    ASSERT(new_points_size == num_points);

    ASSERT_NO_FAILURE(own_keypoints_check(num_points, old_points, new_points_ref, new_points));

    free(new_points);
    free(new_points_ref);
    free(old_points);
    vxReleaseNode(&node);
    if(src_pyr_node[0])
        vxReleaseNode(&src_pyr_node[0]);
    if(src_pyr_node[1])
        vxReleaseNode(&src_pyr_node[1]);
    vxReleaseGraph(&graph);
    vxReleaseScalar(&vx_eps); vxReleaseScalar(&vx_num_iter); vxReleaseScalar(&vx_use_estimations);
    vxReleaseArray(&old_points_arr); vxReleaseArray(&new_points_arr);
    vxReleasePyramid(&src_pyr[0]); vxReleasePyramid(&src_pyr[1]);
    vxReleaseImage(&src_image[0]); vxReleaseImage(&src_image[1]);
}

TEST_WITH_ARG(OptFlowPyrLK, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image[2] = {0, 0};
    vx_pyramid src_pyr[2] = {0, 0};
    vx_array old_points_arr = 0, new_points_arr = 0;
    vx_float32 eps = 0.001f;
    vx_uint32 num_iter = 100;
    vx_bool use_estimations = vx_true_e;
    vx_scalar vx_eps = 0, vx_num_iter = 0, vx_use_estimations = 0;
    vx_size winSize = arg_->winSize;

    vx_size num_points = 0;
    vx_keypoint_t *old_points = 0, *new_points_ref = 0, *new_points = 0;
    vx_size new_points_size = 0;

    vx_size max_window_dim = 0;

    CT_Image src_ct_image[2] = {0, 0};

    VX_CALL(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_OPTICAL_FLOW_WINDOW_MAXIMUM_DIMENSION, &max_window_dim, sizeof(max_window_dim)));
    if (winSize > max_window_dim)
    {
        printf("%d window dim is not supported. Skip test\n", (int)winSize);
        return;
    }

    ASSERT_NO_FAILURE(src_ct_image[0] = arg_->generator(arg_->src1_fileName, 0, 0));
    ASSERT_NO_FAILURE(src_ct_image[1] = arg_->generator(arg_->src2_fileName, 0, 0));

    ASSERT_VX_OBJECT(src_image[0] = ct_image_to_vx_image(src_ct_image[0], context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_image[1] = ct_image_to_vx_image(src_ct_image[1], context), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(src_pyr[0] = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, src_ct_image[0]->width, src_ct_image[0]->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(src_pyr[1] = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, src_ct_image[0]->width, src_ct_image[0]->height, VX_DF_IMAGE_U8), VX_TYPE_PYRAMID);

    ASSERT_NO_FAILURE(num_points = own_read_keypoints(arg_->points_fileName, &old_points, &new_points_ref));

    ASSERT_VX_OBJECT(old_points_arr = own_create_keypoint_array(context, num_points, old_points), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(new_points_arr = vxCreateArray(context, VX_TYPE_KEYPOINT, num_points), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(vx_eps = vxCreateScalar(context, VX_TYPE_FLOAT32, &eps), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_num_iter = vxCreateScalar(context, VX_TYPE_UINT32, &num_iter), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(vx_use_estimations = vxCreateScalar(context, VX_TYPE_BOOL, &use_estimations), VX_TYPE_SCALAR);

    if (arg_->useReferencePyramid)
    {
        vx_border_mode_t border = { VX_BORDER_MODE_UNDEFINED };
        ASSERT_NO_FAILURE(gaussian_pyramid_fill_reference(src_ct_image[0], src_pyr[0], 4, VX_SCALE_PYRAMID_HALF, border));
        ASSERT_NO_FAILURE(gaussian_pyramid_fill_reference(src_ct_image[1], src_pyr[1], 4, VX_SCALE_PYRAMID_HALF, border));
    }
    else
    {
        VX_CALL(vxuGaussianPyramid(context, src_image[0], src_pyr[0]));
        VX_CALL(vxuGaussianPyramid(context, src_image[1], src_pyr[1]));
    }

    VX_CALL(vxuOpticalFlowPyrLK(context, src_pyr[0], src_pyr[1],
            old_points_arr, old_points_arr, new_points_arr,
            VX_TERM_CRITERIA_BOTH, vx_eps, vx_num_iter, vx_use_estimations,
            winSize));

    ASSERT(VX_TYPE_KEYPOINT == ct_read_array(new_points_arr, (void**)&new_points, 0, &new_points_size, 0));
    ASSERT(new_points_size == num_points);

    ASSERT_NO_FAILURE(own_keypoints_check(num_points, old_points, new_points_ref, new_points));

    free(new_points);
    free(new_points_ref);
    free(old_points);
    vxReleaseScalar(&vx_eps); vxReleaseScalar(&vx_num_iter); vxReleaseScalar(&vx_use_estimations);
    vxReleaseArray(&old_points_arr); vxReleaseArray(&new_points_arr);
    vxReleasePyramid(&src_pyr[0]); vxReleasePyramid(&src_pyr[1]);
    vxReleaseImage(&src_image[0]); vxReleaseImage(&src_image[1]);
}

TESTCASE_TESTS(OptFlowPyrLK,
        testNodeCreation,
        testGraphProcessing,
        testImmediateProcessing
        )
