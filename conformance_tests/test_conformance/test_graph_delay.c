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

TESTCASE(GraphDelay, CT_VXContext, ct_setup_vx_context, 0)

TEST(GraphDelay, testSimple)
{
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    int w = 128, h = 128;
    vx_df_image f = VX_DF_IMAGE_U8;
    vx_image images[3];
    vx_node nodes[3];
    vx_delay delay = 0;
    int i;
    vx_size delay_count = 0;

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(images[0] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[1] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(images[2] = vxCreateImage(context, w, h, f), VX_TYPE_IMAGE);

    ASSERT_VX_OBJECT(delay = vxCreateDelay(context, (vx_reference)images[0], 2), VX_TYPE_DELAY);

    VX_CALL(vxQueryDelay(delay, VX_DELAY_ATTRIBUTE_SLOTS, &delay_count, sizeof(delay_count)));
    ASSERT(delay_count == 2);

    ASSERT_VX_OBJECT(nodes[0] = vxBox3x3Node(graph, images[0], (vx_image)vxGetReferenceFromDelay(delay, 0)), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[1] = vxMedian3x3Node(graph, (vx_image)vxGetReferenceFromDelay(delay, -1), images[1]), VX_TYPE_NODE);
    ASSERT_VX_OBJECT(nodes[2] = vxGaussian3x3Node(graph, (vx_image)vxGetReferenceFromDelay(delay, -1), images[2]), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));

    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxAgeDelay(delay));

    VX_CALL(vxProcessGraph(graph));

    for (i = 0; i < 3; i++)
    {
        vxReleaseNode(&nodes[i]);
    }
    for (i = 0; i < 3; i++)
    {
        vxReleaseImage(&images[i]);
    }
    vxReleaseGraph(&graph);
    vxReleaseDelay(&delay);

    ASSERT(graph == 0);
    ASSERT(delay == 0);
}



TESTCASE_TESTS(GraphDelay,
        testSimple
        )
