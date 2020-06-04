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

TESTCASE(GraphCallback, CT_VXContext, ct_setup_vx_context, 0)

static vx_bool own_cb_called = vx_false_e;
static vx_action VX_CALLBACK own_node_callback_continue(vx_node node)
{
    own_cb_called = vx_true_e;
    return VX_ACTION_CONTINUE;
}
static vx_action VX_CALLBACK own_node_callback_abandon(vx_node node)
{
    own_cb_called = vx_true_e;
    return VX_ACTION_ABANDON;
}

TEST(GraphCallback, testContinue)
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

    VX_CALL(vxAssignNodeCallback(node1, own_node_callback_continue));
    ASSERT_EQ_PTR(own_node_callback_continue, vxRetrieveNodeCallback(node1));

    VX_CALL(vxVerifyGraph(graph));

    own_cb_called = vx_false_e;
    VX_CALL(vxProcessGraph(graph));
    ASSERT(own_cb_called == vx_true_e);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

TEST(GraphCallback, testAbandon)
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

    VX_CALL(vxAssignNodeCallback(node1, own_node_callback_abandon));

    VX_CALL(vxVerifyGraph(graph));

    own_cb_called = vx_false_e;
    EXPECT_EQ_VX_STATUS(VX_ERROR_GRAPH_ABANDONED, vxProcessGraph(graph));
    ASSERT(own_cb_called == vx_true_e);

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}

static vx_bool own_error = vx_false_e;
static vx_bool own_cb1_called = vx_false_e;
static vx_bool own_cb2_called = vx_false_e;
static vx_action VX_CALLBACK own_callback1(vx_node node)
{
    //printf("callback1 called\n"); fflush(stdout);
    own_cb1_called = vx_true_e;
    if (own_cb2_called == vx_true_e)
    {
        own_error = vx_true_e;
    }
    return VX_ACTION_CONTINUE;
}
static vx_action VX_CALLBACK own_callback2(vx_node node)
{
    //printf("callback2 called\n"); fflush(stdout);
    own_cb2_called = vx_true_e;
    if (own_cb1_called != vx_true_e)
    {
        own_error = vx_true_e;
    }
    return VX_ACTION_CONTINUE;
}

typedef struct {
    const char* testName;
    vx_bool forward;
} Arg;


TEST_WITH_ARG(GraphCallback, testCallbackOrder, Arg,
        CT_ARG("Forward", vx_true_e),
        CT_ARG("Reverse", vx_false_e)
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, interm_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_node node1 = 0, node2 = 0;
    int i = 0;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(interm_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U32), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    if (arg_->forward == vx_true_e)
    {
        ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
        ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);
    }
    else
    {
        ASSERT_VX_OBJECT(node2 = vxIntegralImageNode(graph, interm_image, dst_image), VX_TYPE_NODE);
        ASSERT_VX_OBJECT(node1 = vxBox3x3Node(graph, src_image, interm_image), VX_TYPE_NODE);
    }

    VX_CALL(vxAssignNodeCallback(node1, own_callback1));
    VX_CALL(vxAssignNodeCallback(node2, own_callback2));

    VX_CALL(vxVerifyGraph(graph));

    for (i = 0; i < 10; i++)
    {
        own_cb1_called = vx_false_e;
        own_cb2_called = vx_false_e;
        own_error = vx_false_e;

        VX_CALL(vxProcessGraph(graph));

        ASSERT(own_cb1_called == vx_true_e);
        ASSERT(own_cb2_called == vx_true_e);
        ASSERT(own_error == vx_false_e);
    }

    vxReleaseNode(&node1); vxReleaseNode(&node2);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&interm_image);
    vxReleaseImage(&src_image);

    ASSERT(node1 == 0); ASSERT(node2 == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(interm_image == 0); ASSERT(src_image == 0);
}


TESTCASE_TESTS(GraphCallback,
        testContinue,
        testAbandon,
        testCallbackOrder
        )
