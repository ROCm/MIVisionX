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

TESTCASE(Graph, CT_VXContext, ct_setup_vx_context, 0)

TEST(Graph, testTwoNodes)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

TEST(Graph, testVirtualImage)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(interm_image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

#ifndef dimof
#  define dimof(arr) (sizeof(arr)/sizeof((arr)[0]))
#endif

// example from specification wrapped into asserts
static void testCornersGraphFactory(vx_context context, vx_graph* graph_out)
{
    vx_uint32 i;
    vx_float32 strength_thresh = 10000.0f;
    vx_float32 r = 1.5f;
    vx_float32 sensitivity = 0.14f;
    vx_int32 window_size = 3;
    vx_int32 block_size = 3;
    vx_graph graph = 0;
    vx_enum channel = VX_CHANNEL_Y;

    *graph_out = NULL;
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    // if (vxGetStatus((vx_reference)graph) == VX_SUCCESS)
    {
        vx_image virts[] = {
            vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT),
            vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT)
        };

        vx_kernel kernels[] = {
            vxGetKernelByEnum(context, VX_KERNEL_CHANNEL_EXTRACT),
            vxGetKernelByEnum(context, VX_KERNEL_MEDIAN_3x3),
            vxGetKernelByEnum(context, VX_KERNEL_HARRIS_CORNERS)
        };

        vx_node nodes[dimof(kernels)] = {
            vxCreateGenericNode(graph, kernels[0]),
            vxCreateGenericNode(graph, kernels[1]),
            vxCreateGenericNode(graph, kernels[2])
        };

        vx_scalar scalars[] = {
            vxCreateScalar(context, VX_TYPE_ENUM, &channel),
            vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh),
            vxCreateScalar(context, VX_TYPE_FLOAT32, &r),
            vxCreateScalar(context, VX_TYPE_FLOAT32, &sensitivity),
            vxCreateScalar(context, VX_TYPE_INT32, &window_size),
            vxCreateScalar(context, VX_TYPE_INT32, &block_size)
        };

        vx_parameter parameters[] = {
            vxGetParameterByIndex(nodes[0], 0),
            vxGetParameterByIndex(nodes[2], 6)
        };

        for (i = 0; i < dimof(virts); i++)
            ASSERT_VX_OBJECT(virts[i], VX_TYPE_IMAGE);
        for (i = 0; i < dimof(kernels); i++)
            ASSERT_VX_OBJECT(kernels[i], VX_TYPE_KERNEL);
        for (i = 0; i < dimof(nodes); i++)
            ASSERT_VX_OBJECT(nodes[i], VX_TYPE_NODE);
        for (i = 0; i < dimof(scalars); i++)
            ASSERT_VX_OBJECT(scalars[i], VX_TYPE_SCALAR);
        for (i = 0; i < dimof(parameters); i++)
            ASSERT_VX_OBJECT(parameters[i], VX_TYPE_PARAMETER);

        // Channel Extract
        VX_CALL(vxAddParameterToGraph(graph, parameters[0]));
        VX_CALL(vxSetParameterByIndex(nodes[0], 1, (vx_reference)scalars[0]));
        VX_CALL(vxSetParameterByIndex(nodes[0], 2, (vx_reference)virts[0]));
        // Median Filter
        VX_CALL(vxSetParameterByIndex(nodes[1], 0, (vx_reference)virts[0]));
        VX_CALL(vxSetParameterByIndex(nodes[1], 1, (vx_reference)virts[1]));
        // Harris Corners
        VX_CALL(vxSetParameterByIndex(nodes[2], 0, (vx_reference)virts[1]));
        VX_CALL(vxSetParameterByIndex(nodes[2], 1, (vx_reference)scalars[1]));
        VX_CALL(vxSetParameterByIndex(nodes[2], 2, (vx_reference)scalars[2]));
        VX_CALL(vxSetParameterByIndex(nodes[2], 3, (vx_reference)scalars[3]));
        VX_CALL(vxSetParameterByIndex(nodes[2], 4, (vx_reference)scalars[4]));
        VX_CALL(vxSetParameterByIndex(nodes[2], 5, (vx_reference)scalars[5]));
        VX_CALL(vxAddParameterToGraph(graph, parameters[1]));

        for (i = 0; i < dimof(scalars); i++)
        {
            VX_CALL(vxReleaseScalar(&scalars[i]));
            ASSERT(scalars[i] == NULL);
        }
        for (i = 0; i < dimof(virts); i++)
        {
            VX_CALL(vxReleaseImage(&virts[i]));
            ASSERT(virts[i] == NULL);
        }
        for (i = 0; i < dimof(kernels); i++)
        {
            VX_CALL(vxReleaseKernel(&kernels[i]));
            ASSERT(kernels[i] == NULL);
        }
        for (i = 0; i < dimof(nodes); i++)
        {
            VX_CALL(vxReleaseNode(&nodes[i]));
            ASSERT(nodes[i] == NULL);
        }
        for (i = 0; i < dimof(parameters); i++)
        {
            VX_CALL(vxReleaseParameter(&parameters[i]));
            ASSERT(parameters[i] == NULL);
        }
    }

    *graph_out = graph;
}

TEST(Graph, testGraphFactory)
{
    vx_context context = context_->vx_context_;
    vx_graph   graph;
    vx_image   source;
    vx_array   points;
    vx_parameter points_param;

    ASSERT_NO_FAILURE(testCornersGraphFactory(context, &graph));

    ASSERT_VX_OBJECT(source = vxCreateImage(context, 640, 480, VX_DF_IMAGE_YUV4), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(points = vxCreateArray(context, VX_TYPE_KEYPOINT, 100), VX_TYPE_ARRAY);

    ASSERT_NO_FAILURE(ct_fill_image_random(source, &CT()->seed_));

    VX_CALL(vxSetGraphParameterByIndex(graph, 0, (vx_reference)source));

    ASSERT_VX_OBJECT(points_param = vxGetGraphParameterByIndex(graph, 1), VX_TYPE_PARAMETER);
    VX_CALL(vxSetParameterByReference(points_param, (vx_reference)points));

    VX_CALL(vxReleaseParameter(&points_param));
    ASSERT(points_param == NULL);
    VX_CALL(vxReleaseImage(&source));
    ASSERT(source == NULL);

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseArray(&points));
    ASSERT(points == NULL);

    VX_CALL(vxReleaseGraph(&graph));
    ASSERT(graph == NULL);
}

#define VX_KERNEL_CONFORMANCE_TEST_TAKE10 (VX_KERNEL_BASE(VX_ID_MAX - 1, 0) + 2)
#define VX_KERNEL_CONFORMANCE_TEST_TAKE10_NAME "org.khronos.openvx.test.array_take_10"

static vx_status VX_CALLBACK take10_InputValidator(vx_node node, vx_uint32 index)
{
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK take10_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_size capacity = 10;
    vx_enum type = VX_TYPE_KEYPOINT;

    // are we really required to set these attributes???
    VX_CALL_(return VX_FAILURE, vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
    VX_CALL_(return VX_FAILURE, vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK take10_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_array input, output;
    vx_size len = 0;
    vx_size stride = 0;
    void* data = 0;

    ASSERT_(return VX_FAILURE, num == 2);
    ASSERT_VX_OBJECT_(return VX_FAILURE, input  = (vx_array)parameters[0], VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT_(return VX_FAILURE, output = (vx_array)parameters[1], VX_TYPE_ARRAY);

    VX_CALL_(return VX_FAILURE, vxTruncateArray(output, 0));
    VX_CALL_(return VX_FAILURE, vxQueryArray(input, VX_ARRAY_ATTRIBUTE_NUMITEMS, &len, sizeof(len)));

    if (len > 10) len = 10;

    VX_CALL_(return VX_FAILURE, vxAccessArrayRange(input,  0, len, &stride, &data, VX_READ_ONLY));
    VX_CALL_(return VX_FAILURE, vxAddArrayItems(output, len, data, stride));
    VX_CALL_(return VX_FAILURE, vxCommitArrayRange(input, 0, len, data));

    return VX_SUCCESS;
}

static void take10_node(vx_graph graph, vx_array in, vx_array out)
{
    vx_kernel kernel = 0;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);

    ASSERT_VX_OBJECT(context, VX_TYPE_CONTEXT);
    ASSERT_VX_OBJECT(kernel = vxAddKernel(
            context,
            VX_KERNEL_CONFORMANCE_TEST_TAKE10_NAME,
            VX_KERNEL_CONFORMANCE_TEST_TAKE10,
            take10_Kernel,
            2,
            take10_InputValidator,
            take10_OutputValidator,
            NULL,
            NULL), VX_TYPE_KERNEL);
    VX_CALL(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    VX_CALL(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    VX_CALL(vxFinalizeKernel(kernel));

    ASSERT_VX_OBJECT(node = vxCreateGenericNode(graph, kernel), VX_TYPE_NODE);
    VX_CALL(vxSetParameterByIndex(node, 0, (vx_reference)in));
    VX_CALL(vxSetParameterByIndex(node, 1, (vx_reference)out));

    VX_CALL(vxReleaseNode(&node));
    ASSERT(node == 0);

    // VX_CALL(vxReleaseKernel(&kernel));
    // ASSERT(kernel == 0);
}

TEST(Graph, testVirtualArray)
{
    vx_bool    use_estimations;
    vx_uint32  num_iter;
    vx_float32 threshold_f;
    vx_float32 eps;
    vx_scalar  fast_thresh;
    vx_scalar  flow_eps;
    vx_scalar  flow_num_iter;
    vx_scalar  flow_use_estimations;
    vx_size    window;
    vx_array   corners;
    vx_array   new_corners, corners10;
    vx_image   frame0 = 0;
    vx_image   frame1 = 0;
    vx_pyramid p0;
    vx_pyramid p1;
    vx_node    n1;
    vx_node    n2;
    vx_node    n3;
    vx_node    n4;
    vx_graph   graph;
    vx_context context;
    CT_Image   src0 = 0, src1 = 0;
    vx_scalar  scalar_fastCorners = 0;
    vx_size    fastCorners = 0;

    ASSERT_VX_OBJECT(context = context_->vx_context_, VX_TYPE_CONTEXT);

    ASSERT_NO_FAILURE(src0 = ct_read_image("optflow_00.bmp", 1));
    ASSERT_NO_FAILURE(src1 = ct_read_image("optflow_01.bmp", 1));

    ASSERT_VX_OBJECT(frame0 = ct_image_to_vx_image(src0, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(frame1 = ct_image_to_vx_image(src1, context), VX_TYPE_IMAGE);

    threshold_f = 80;
    eps = 0.01f;
    num_iter = 10;
    use_estimations = vx_false_e;
    window = 3;

    ASSERT_VX_OBJECT(fast_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &threshold_f), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(flow_eps = vxCreateScalar(context, VX_TYPE_FLOAT32, &eps), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(flow_num_iter = vxCreateScalar(context, VX_TYPE_UINT32, &num_iter), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(flow_use_estimations = vxCreateScalar(context, VX_TYPE_BOOL, &use_estimations), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    // Don't use zero capacity here
    ASSERT_VX_OBJECT(corners = vxCreateVirtualArray(graph, 0, 100), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(new_corners = vxCreateVirtualArray(graph, 0, 0), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(corners10   = vxCreateArray(context, VX_TYPE_KEYPOINT, 10), VX_TYPE_ARRAY);

    ASSERT_VX_OBJECT(scalar_fastCorners = vxCreateScalar(context, VX_TYPE_SIZE, &fastCorners), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(n1 = vxFastCornersNode(graph, frame0, fast_thresh, vx_true_e, corners, scalar_fastCorners), VX_TYPE_NODE);

    ASSERT_VX_OBJECT(p0 = vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_PYRAMID);
    ASSERT_VX_OBJECT(p1 = vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 0, 0, VX_DF_IMAGE_VIRT), VX_TYPE_PYRAMID);

    ASSERT_VX_OBJECT(n2 = vxGaussianPyramidNode(graph, frame0, p0), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(n3 = vxGaussianPyramidNode(graph, frame1, p1), VX_TYPE_NODE);

    ASSERT_VX_OBJECT(n4 = vxOpticalFlowPyrLKNode(graph, p0, p1, corners, corners, new_corners, VX_TERM_CRITERIA_BOTH, flow_eps, flow_num_iter, flow_use_estimations, window), VX_TYPE_NODE);

    ASSERT_NO_FAILURE(take10_node(graph, new_corners, corners10));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReadScalarValue(scalar_fastCorners, &fastCorners));
    ASSERT(fastCorners > 0);

    VX_CALL(vxReleaseScalar(&scalar_fastCorners));
    VX_CALL(vxReleaseScalar(&fast_thresh));
    VX_CALL(vxReleaseScalar(&flow_eps));
    VX_CALL(vxReleaseScalar(&flow_num_iter));
    VX_CALL(vxReleaseScalar(&flow_use_estimations));
    VX_CALL(vxReleaseArray(&corners));
    VX_CALL(vxReleaseArray(&new_corners));
    VX_CALL(vxReleaseArray(&corners10));
    VX_CALL(vxReleasePyramid(&p0));
    VX_CALL(vxReleasePyramid(&p1));
    VX_CALL(vxReleaseImage(&frame0));
    VX_CALL(vxReleaseImage(&frame1));
    VX_CALL(vxReleaseNode(&n1));
    VX_CALL(vxReleaseNode(&n2));
    VX_CALL(vxReleaseNode(&n3));
    VX_CALL(vxReleaseNode(&n4));
    VX_CALL(vxReleaseGraph(&graph));
}


TEST(Graph, testNodeRemove)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    vx_uint32 num_nodes = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_NUMNODES, &num_nodes, sizeof(num_nodes)));
    ASSERT(num_nodes == 2);

    VX_CALL(vxRemoveNode(&node2));
    ASSERT(node2 == 0);

    VX_CALL(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_NUMNODES, &num_nodes, sizeof(num_nodes)));
    ASSERT(num_nodes == 1);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    vxReleaseNode(&node1);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

TEST(Graph, testNodeFromEnum)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_kernel kernel = 0;
    vx_node node = 0;
    vx_uint32 num_params = 0;
    vx_parameter parameter = 0;
    vx_image p_image = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(kernel = vxGetKernelByEnum(context, VX_KERNEL_BOX_3x3), VX_TYPE_KERNEL);

    VX_CALL(vxQueryKernel(kernel, VX_KERNEL_ATTRIBUTE_PARAMETERS, &num_params, sizeof(num_params)));
    ASSERT_EQ_INT(2, num_params);

    ASSERT_VX_OBJECT(node = vxCreateGenericNode(graph, kernel), VX_TYPE_NODE);

    VX_CALL(vxSetParameterByIndex(node, 0, (vx_reference)src_image));

    ASSERT_VX_OBJECT(parameter = vxGetParameterByIndex(node, 0), VX_TYPE_PARAMETER);
    VX_CALL(vxQueryParameter(parameter, VX_PARAMETER_ATTRIBUTE_REF, &p_image, sizeof(p_image)));
    ASSERT(p_image == src_image);
    vxReleaseImage(&p_image);
    vxReleaseParameter(&parameter);

    {
        /* check set node parameter by reference */
        ASSERT_VX_OBJECT(parameter = vxGetParameterByIndex(node, 1), VX_TYPE_PARAMETER);

        /* parameter was not set yet */
        VX_CALL(vxQueryParameter(parameter, VX_PARAMETER_ATTRIBUTE_REF, &p_image, sizeof(p_image)));
        ASSERT(p_image != dst_image);
        VX_CALL(vxSetParameterByReference(parameter, (vx_reference)dst_image));
        /* expect parameter is set to know value */
        VX_CALL(vxQueryParameter(parameter, VX_PARAMETER_ATTRIBUTE_REF, &p_image, sizeof(p_image)));
        ASSERT(p_image == dst_image);
    }

    vxReleaseImage(&p_image);
    vxReleaseParameter(&parameter);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    vxReleaseNode(&node);
    vxReleaseKernel(&kernel);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(kernel == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(src_image == 0);
}

TEST(Graph, testTwoNodesWithSameDst)
{
    vx_context context = context_->vx_context_;
    vx_image src1_image = 0, src2_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;

    ASSERT_VX_OBJECT(src1_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src1_image, dst_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxMedian3x3Node(graph, src2_image, dst_image), VX_TYPE_NODE);

    EXPECT_NE_VX_STATUS(vxVerifyGraph(graph), VX_SUCCESS);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src1_image);
    vxReleaseImage(&src2_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(src1_image == 0); ASSERT(src2_image == 0);
}

TEST(Graph, testCycle)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, dst_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxMedian3x3Node(graph, dst_image, src_image), VX_TYPE_NODE);

    EXPECT_NE_VX_STATUS(vxVerifyGraph(graph), VX_SUCCESS);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(src_image == 0);
}

TEST(Graph, testCycle2)
{
    vx_context context = context_->vx_context_;
    vx_image src1_image = 0, src2_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0, node3 = 0;

    ASSERT_VX_OBJECT(src1_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src2_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxAddNode(graph, src1_image, src2_image, VX_CONVERT_POLICY_SATURATE, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxBox3x3Node(graph, interm_image, dst_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node3 = vxMedian3x3Node(graph, dst_image, src2_image), VX_TYPE_NODE);

    EXPECT_NE_VX_STATUS(vxVerifyGraph(graph), VX_SUCCESS);

    vxReleaseNode(&node1); vxReleaseNode(&node2); vxReleaseNode(&node3);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src1_image);
    vxReleaseImage(&src2_image);
    vxReleaseImage(&interm_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0); ASSERT(node3 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(src1_image == 0); ASSERT(src2_image == 0); ASSERT(interm_image == 0);
}


TEST(Graph, testMultipleRun)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    CT_Image res1 = 0, res2 = 0;
    vx_border_mode_t border = { VX_BORDER_MODE_REPLICATE };

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(ct_fill_image_random(src_image, &CT()->seed_));
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    VX_CALL(vxSetNodeAttribute(node1, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));
    ASSERT_NO_FAILURE(res1 = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(ct_fill_image_random(dst_image, &CT()->seed_));

    VX_CALL(vxProcessGraph(graph));
    ASSERT_NO_FAILURE(res2 = ct_image_from_vx_image(dst_image));

    ASSERT_EQ_CTIMAGE(res1, res2);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}


TEST(Graph, testMultipleRunAsync)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    CT_Image res1 = 0, res2 = 0;
    vx_border_mode_t border = { VX_BORDER_MODE_REPLICATE };

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(ct_fill_image_random(src_image, &CT()->seed_));
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    VX_CALL(vxSetNodeAttribute(node1, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border)));
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxScheduleGraph(graph));
    VX_CALL(vxWaitGraph(graph));
    ASSERT_NO_FAILURE(res1 = ct_image_from_vx_image(dst_image));

    ASSERT_NO_FAILURE(ct_fill_image_random(dst_image, &CT()->seed_));

    VX_CALL(vxScheduleGraph(graph));
    VX_CALL(vxWaitGraph(graph));
    ASSERT_NO_FAILURE(res2 = ct_image_from_vx_image(dst_image));

    ASSERT_EQ_CTIMAGE(res1, res2);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}


TEST(Graph, testAsyncWaitWithoutSchedule)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxWaitGraph(graph));

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}


TEST(Graph, testNodePerformance)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    vx_perf_t perf;
    vx_perf_t perf2;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxQueryNode(node1, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));

    ASSERT(perf.num == 1);
    ASSERT(perf.beg > 0);
    ASSERT(perf.min > 0);

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxQueryNode(node1, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf2, sizeof(perf)));

    ASSERT(perf2.num == 2);
    ASSERT(perf2.beg > perf.end);
    ASSERT(perf2.min > 0);
    ASSERT(perf2.sum > perf.sum);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}


TEST(Graph, testGraphPerformance)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    vx_perf_t perf;
    vx_perf_t perf2;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));

    ASSERT(perf.num == 1);
    ASSERT(perf.beg > 0);
    ASSERT(perf.min > 0);

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf2, sizeof(perf)));

    ASSERT(perf2.num == 2);
    ASSERT(perf2.beg >= perf.end);
    ASSERT(perf2.min > 0);
    ASSERT(perf2.sum >= perf.sum);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

TEST(Graph, testInvalidNode)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_kernel kernel0 = 0;
    vx_kernel kernel1 = 0;
    vx_node node = 0;
    vx_uint32 num_params = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    EXPECT_VX_OBJECT(kernel0 = vxGetKernelByName(context, "org.khronos.openvx.invalid"), VX_TYPE_KERNEL);
    EXPECT_VX_OBJECT(kernel1 = vxGetKernelByEnum(context, VX_KERNEL_INVALID), VX_TYPE_KERNEL);

    EXPECT_EQ_PTR(kernel0, kernel1);

    VX_CALL(vxQueryKernel(kernel0, VX_KERNEL_ATTRIBUTE_PARAMETERS, &num_params, sizeof(num_params)));
    ASSERT_EQ_INT(0, num_params);

    ASSERT_VX_OBJECT(node = vxCreateGenericNode(graph, kernel0), VX_TYPE_NODE);

    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxVerifyGraph(graph));
    EXPECT_NE_VX_STATUS(VX_SUCCESS, vxProcessGraph(graph));

    VX_CALL(vxReleaseKernel(&kernel0));
    VX_CALL(vxReleaseKernel(&kernel1));
    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
}

typedef struct
{
    char* name;
    vx_enum kernel_id;
} kernel_name_arg;

TEST_WITH_ARG(Graph, testKernelName, kernel_name_arg,
    ARG("org.khronos.openvx.invalid",               VX_KERNEL_INVALID),
    ARG("org.khronos.openvx.color_convert",         VX_KERNEL_COLOR_CONVERT),
    ARG("org.khronos.openvx.channel_extract",       VX_KERNEL_CHANNEL_EXTRACT),
    ARG("org.khronos.openvx.channel_combine",       VX_KERNEL_CHANNEL_COMBINE),
    ARG("org.khronos.openvx.sobel_3x3",             VX_KERNEL_SOBEL_3x3),
    ARG("org.khronos.openvx.magnitude",             VX_KERNEL_MAGNITUDE),
    ARG("org.khronos.openvx.phase",                 VX_KERNEL_PHASE),
    ARG("org.khronos.openvx.scale_image",           VX_KERNEL_SCALE_IMAGE),
    ARG("org.khronos.openvx.table_lookup",          VX_KERNEL_TABLE_LOOKUP),
    ARG("org.khronos.openvx.histogram",             VX_KERNEL_HISTOGRAM),
    ARG("org.khronos.openvx.equalize_histogram",    VX_KERNEL_EQUALIZE_HISTOGRAM),
    ARG("org.khronos.openvx.absdiff",               VX_KERNEL_ABSDIFF),
    ARG("org.khronos.openvx.mean_stddev",           VX_KERNEL_MEAN_STDDEV),
    ARG("org.khronos.openvx.threshold",             VX_KERNEL_THRESHOLD),
    ARG("org.khronos.openvx.integral_image",        VX_KERNEL_INTEGRAL_IMAGE),
    ARG("org.khronos.openvx.dilate_3x3",            VX_KERNEL_DILATE_3x3),
    ARG("org.khronos.openvx.erode_3x3",             VX_KERNEL_ERODE_3x3),
    ARG("org.khronos.openvx.median_3x3",            VX_KERNEL_MEDIAN_3x3),
    ARG("org.khronos.openvx.box_3x3",               VX_KERNEL_BOX_3x3),
    ARG("org.khronos.openvx.gaussian_3x3",          VX_KERNEL_GAUSSIAN_3x3),
    ARG("org.khronos.openvx.custom_convolution",    VX_KERNEL_CUSTOM_CONVOLUTION),
    ARG("org.khronos.openvx.gaussian_pyramid",      VX_KERNEL_GAUSSIAN_PYRAMID),
    ARG("org.khronos.openvx.accumulate",            VX_KERNEL_ACCUMULATE),
    ARG("org.khronos.openvx.accumulate_weighted",   VX_KERNEL_ACCUMULATE_WEIGHTED),
    ARG("org.khronos.openvx.accumulate_square",     VX_KERNEL_ACCUMULATE_SQUARE),
    ARG("org.khronos.openvx.minmaxloc",             VX_KERNEL_MINMAXLOC),
    ARG("org.khronos.openvx.convertdepth",          VX_KERNEL_CONVERTDEPTH),
    ARG("org.khronos.openvx.canny_edge_detector",   VX_KERNEL_CANNY_EDGE_DETECTOR),
    ARG("org.khronos.openvx.and",                   VX_KERNEL_AND),
    ARG("org.khronos.openvx.or",                    VX_KERNEL_OR),
    ARG("org.khronos.openvx.xor",                   VX_KERNEL_XOR),
    ARG("org.khronos.openvx.not",                   VX_KERNEL_NOT),
    ARG("org.khronos.openvx.multiply",              VX_KERNEL_MULTIPLY),
    ARG("org.khronos.openvx.add",                   VX_KERNEL_ADD),
    ARG("org.khronos.openvx.subtract",              VX_KERNEL_SUBTRACT),
    ARG("org.khronos.openvx.warp_affine",           VX_KERNEL_WARP_AFFINE),
    ARG("org.khronos.openvx.warp_perspective",      VX_KERNEL_WARP_PERSPECTIVE),
    ARG("org.khronos.openvx.harris_corners",        VX_KERNEL_HARRIS_CORNERS),
    ARG("org.khronos.openvx.fast_corners",          VX_KERNEL_FAST_CORNERS),
    ARG("org.khronos.openvx.optical_flow_pyr_lk",   VX_KERNEL_OPTICAL_FLOW_PYR_LK),
    ARG("org.khronos.openvx.remap",                 VX_KERNEL_REMAP),
    ARG("org.khronos.openvx.halfscale_gaussian",    VX_KERNEL_HALFSCALE_GAUSSIAN),
    )
{
    vx_context context = context_->vx_context_;
    vx_kernel kernel   = 0;
    vx_enum   kernel_id = 0;

    EXPECT_VX_OBJECT(kernel = vxGetKernelByName(context, arg_->name), VX_TYPE_KERNEL);

    if (CT_HasFailure())
    {
        vx_char name[VX_MAX_KERNEL_NAME] = {0};

        ASSERT_VX_OBJECT(kernel = vxGetKernelByEnum(context, arg_->kernel_id), VX_TYPE_KERNEL);
        VX_CALL(vxQueryKernel(kernel, VX_KERNEL_ATTRIBUTE_NAME, &name, sizeof(name)));
        printf("\tExpected kernel name is: %s\n", arg_->name);
        printf("\tActual kernel name is:   %-*s\n", VX_MAX_KERNEL_NAME, name);
    }
    else
    {
        VX_CALL(vxQueryKernel(kernel, VX_KERNEL_ATTRIBUTE_ENUM, &kernel_id, sizeof(kernel_id)));
        EXPECT_EQ_INT(arg_->kernel_id, kernel_id);
    }

    VX_CALL(vxReleaseKernel(&kernel));
}


TESTCASE_TESTS(Graph,
        testTwoNodes,
        testGraphFactory,
        testVirtualImage,
        testVirtualArray,
        testNodeRemove,
        testNodeFromEnum,
        testInvalidNode,
        testTwoNodesWithSameDst,
        testCycle,
        testCycle2,
        testMultipleRun,
        testMultipleRunAsync,
        DISABLED_testAsyncWaitWithoutSchedule,
        testNodePerformance,
        testGraphPerformance,
        testKernelName
        )
