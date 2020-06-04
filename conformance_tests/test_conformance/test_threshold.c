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
#include <stdlib.h>
#include <limits.h>

static void reference_threshold(CT_Image src, CT_Image dst, vx_enum ttype, int ta, int tb)
{
    uint32_t x, y, width, height, srcstride, dststride;

    ASSERT(src && dst);
    ASSERT(src->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8);
    ASSERT(src->width > 0 && src->height > 0 &&
           src->width == dst->width && src->height == dst->height);
    width = src->width;
    height = src->height;
    srcstride = ct_stride_bytes(src);
    dststride = ct_stride_bytes(dst);

    for( y = 0; y < height; y++ )
    {
        const uint8_t* srcptr = src->data.y + y*srcstride;
        uint8_t* dstptr = dst->data.y + y*dststride;
        if( ttype == VX_THRESHOLD_TYPE_BINARY )
        {
            for( x = 0; x < width; x++ )
                dstptr[x] = srcptr[x] > ta ? 255 : 0;
        }
        else
        {
            for( x = 0; x < width; x++ )
                dstptr[x] = srcptr[x] < ta || srcptr[x] > tb ? 0 : 255;
        }
    }
}


TESTCASE(Threshold, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_enum ttype;
} format_arg;

#define THRESHOLD_CASE(imm, ttype) { #imm "/" #ttype, CT_##imm##_MODE, VX_THRESHOLD_TYPE_##ttype }

TEST_WITH_ARG(Threshold, testOnRandom, format_arg,
              THRESHOLD_CASE(Immediate, BINARY),
              THRESHOLD_CASE(Immediate, RANGE),
              THRESHOLD_CASE(Graph, BINARY),
              THRESHOLD_CASE(Graph, RANGE),
              )
{
    int format = VX_DF_IMAGE_U8;
    int ttype = arg_->ttype;
    int mode = arg_->mode;
    vx_image src, dst;
    vx_threshold vxt;
    CT_Image src0, dst0, dst1;
    vx_node node = 0;
    vx_graph graph = 0;
    vx_context context = context_->vx_context_;
    int iter, niters = 100;
    uint64_t rng;
    int a = 0, b = 256;

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int width, height;

        uint8_t _ta = CT_RNG_NEXT_INT(rng, 0, 256), _tb = CT_RNG_NEXT_INT(rng, 0, 256);
        vx_int32 ta = CT_MIN(_ta, _tb), tb = CT_MAX(_ta, _tb);

        if( ct_check_any_size() )
        {
            width = ct_roundf(ct_log_rng(&rng, 0, 10));
            height = ct_roundf(ct_log_rng(&rng, 0, 10));
            width = CT_MAX(width, 1);
            height = CT_MAX(height, 1);
        }
        else
        {
            width = 640;
            height = 480;
        }

        ct_update_progress(iter, niters);

        ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, format, &rng, a, b));
        if( iter % 20 == 0 )
        {
            uint8_t val = (uint8_t)CT_RNG_NEXT_INT(rng, a, b);
            memset(src0->data.y, val, ct_stride_bytes(src0)*src0->height);
        }
        ASSERT_NO_FAILURE(dst0 = ct_allocate_image(width, height, format));
        ASSERT_NO_FAILURE(reference_threshold(src0, dst0, ttype, ta, tb));

        src = ct_image_to_vx_image(src0, context);
        dst = vxCreateImage(context, width, height, format);
        ASSERT_VX_OBJECT(dst, VX_TYPE_IMAGE);
        vxt = vxCreateThreshold(context, ttype, VX_TYPE_UINT8);
        if( ttype == VX_THRESHOLD_TYPE_BINARY )
        {
            vx_int32 v = 0;
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetThresholdAttribute(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &ta, sizeof(ta)));
            VX_CALL(vxQueryThreshold(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &v, sizeof(v)));
            if (v != ta)
            {
                CT_FAIL("check for query threshold attribute VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE failed\n");
            }
        }
        else
        {
            vx_int32 v1 = 0;
            vx_int32 v2 = 0;
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetThresholdAttribute(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &ta, sizeof(ta)));
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetThresholdAttribute(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &tb, sizeof(tb)));

            VX_CALL(vxQueryThreshold(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &v1, sizeof(v1)));
            if (v1 != ta)
            {
                CT_FAIL("check for query threshold attribute VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER failed\n");
            }

            VX_CALL(vxQueryThreshold(vxt, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &v2, sizeof(v2)));
            if (v2 != tb)
            {
                CT_FAIL("check for query threshold attribute VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER failed\n");
            }
        }
        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuThreshold(context, src, vxt, dst));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxThresholdNode(graph, src, vxt, dst);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            if (vxIsGraphVerified(graph))
            {
                /* do not expect graph to be in verified state before vxGraphVerify call */
                CT_FAIL("check for vxIsGraphVerified() failed\n");
            }
            VX_CALL(vxVerifyGraph(graph));
            if(!vxIsGraphVerified(graph))
            {
                /* expect graph to be in verified state before vxGraphVerify call */
                /* NB, according to the spec, vxProcessGraph may also do graph verification */
                CT_FAIL("check for vxIsGraphVerified() failed\n");
            }
            VX_CALL(vxProcessGraph(graph));
        }

        dst1 = ct_image_from_vx_image(dst);

        ASSERT_CTIMAGE_NEAR(dst0, dst1, 0);
        vxReleaseImage(&src);
        vxReleaseImage(&dst);
        vxReleaseThreshold(&vxt);
        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }
}

TESTCASE_TESTS(Threshold, testOnRandom)
