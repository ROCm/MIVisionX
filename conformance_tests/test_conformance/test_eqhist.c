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

static void reference_eqhist(CT_Image src, CT_Image dst)
{
    int i, j, i0 = -1, hist[256];
    uint32_t x, y, width, height, srcstride, dststride, delta = 0, scale = 1;
    uint8_t lut[256];

    ASSERT(src && dst);
    ASSERT(src->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8);
    ASSERT(src->width > 0 && src->height > 0 &&
           src->width == dst->width && src->height == dst->height);
    width = src->width;
    height = src->height;
    srcstride = ct_stride_bytes(src);
    dststride = ct_stride_bytes(dst);

    for( i = 0; i < 256; i++ )
        hist[i] = 0;

    for( y = 0; y < height; y++ )
    {
        const uint8_t* srcptr = src->data.y + y*srcstride;
        for( x = 0; x < width; x++ )
            hist[srcptr[x]]++;
    }

    for( i = 0; i < 256; i++ )
    {
        if( i0 < 0 && hist[i] > 0 )
        {
            i0 = i;
            delta = hist[i0];
            scale = width*height - delta;
            if( scale == 0 )
            {
                // if all the pixels are equal, leave the image as-is
                for( j = 0; j < 256; j++ )
                    lut[j] = j;
                break;
            }
        }
        hist[i] += i > 0 ? hist[i-1] : 0;
        if( i0 < 0 )
            lut[i] = 0;
        else
        {
            int val = (int)(((hist[i] - delta)*255 + scale/2)/scale);
            lut[i] = CT_CAST_U8(val);
        }
    }

    for( y = 0; y < height; y++ )
    {
        const uint8_t* srcptr = src->data.y + y*srcstride;
        uint8_t* dstptr = dst->data.y + y*dststride;
        for( x = 0; x < width; x++ )
            dstptr[x] = lut[srcptr[x]];
    }
}


TESTCASE(EqualizeHistogram, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_df_image format;
} format_arg;

#define MAX_BINS 256

#define EQHIST_TEST_CASE(imm, tp) \
    {#imm "/" #tp, CT_##imm##_MODE, VX_DF_IMAGE_##tp}

TEST_WITH_ARG(EqualizeHistogram, testOnRandom, format_arg,
              EQHIST_TEST_CASE(Immediate, U8),
              EQHIST_TEST_CASE(Graph, U8),
              )
{
    int format = arg_->format;
    int mode = arg_->mode;
    vx_image src, dst;
    vx_node node = 0;
    vx_graph graph = 0;
    CT_Image src0, dst0, dst1;
    vx_context context = context_->vx_context_;
    int iter, niters = 100;
    uint64_t rng;
    int a = 0, b = 256;

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int width, height;

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
        ASSERT_NO_FAILURE(reference_eqhist(src0, dst0));

        src = ct_image_to_vx_image(src0, context);
        dst = vxCreateImage(context, width, height, format);
        ASSERT_VX_OBJECT(dst, VX_TYPE_IMAGE);

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuEqualizeHist(context, src, dst));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxEqualizeHistNode(graph, src, dst);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
        }
        dst1 = ct_image_from_vx_image(dst);

        ASSERT_CTIMAGE_NEAR(dst0, dst1, 1);
        vxReleaseImage(&src);
        vxReleaseImage(&dst);
        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }
}

TESTCASE_TESTS(EqualizeHistogram, testOnRandom)
