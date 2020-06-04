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

static void reference_histogram(CT_Image src, int32_t* hist, int nbins, int offset, int range)
{
    int i, hist0[256], wsz = range/nbins;
    uint32_t x, y, width = src->width, height = src->height, stride;

    ASSERT(src);
    ASSERT(src->format == VX_DF_IMAGE_U8);
    ASSERT(src->width > 0 && src->height > 0);
    stride = ct_stride_bytes(src);

    for( i = 0; i < 256; i++ )
        hist0[i] = 0;

    for( y = 0; y < height; y++ )
    {
        const uint8_t* ptr = src->data.y + y*stride;
        for( x = 0; x < width; x++ )
            hist0[ptr[x]]++;
    }

    for( i = 0; i < nbins; i++ )
        hist[i] = 0;

    for( i = offset; i < offset + range; i++ )
    {
        int j = (i - offset)/wsz;
        hist[j] = (int32_t)(hist[j] + hist0[i]);
    }
}


TESTCASE(Histogram, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_df_image format;
} format_arg;

#define MAX_BINS 256

#define HIST_TEST_CASE(imm, tp) \
    {#imm "/" #tp, CT_##imm##_MODE, VX_DF_IMAGE_##tp}

TEST_WITH_ARG(Histogram, testOnRandom, format_arg,
              HIST_TEST_CASE(Immediate, U8),
              HIST_TEST_CASE(Graph, U8),
              )
{
    int format = arg_->format;
    int mode = arg_->mode;
    vx_image src;
    CT_Image src0;
    vx_context context = context_->vx_context_;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_distribution dist;
    int iter, niters = 100;
    uint64_t rng;
    int a = 0, b = 256;
    int32_t hist0[MAX_BINS];
    int32_t hist[MAX_BINS];

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int width, height;
        int val0 = CT_RNG_NEXT_INT(rng, 0, 255), val1 = CT_RNG_NEXT_INT(rng, 0, 255);
        int offset = CT_MIN(val0, val1), range = CT_MAX(val0, val1) - offset + 1;
        int i, nbins = CT_RNG_NEXT_INT(rng, 1, range+1);
        void* hptr = 0;

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

        if( iter % 30 == 0 )
        {
            offset = 0;
            range = 256;
            nbins = 1 << CT_RNG_NEXT_INT(rng, 0, 9);
        }
        else
        {
            // make sure the range is divisible by the number of bins,
            // otherwise the histogram will be statistically unbalanced
            range = (range/nbins)*nbins;
        }

        ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, format, &rng, a, b));
        ASSERT_NO_FAILURE(reference_histogram(src0, hist0, nbins, offset, range));

        src = ct_image_to_vx_image(src0, context);
        ASSERT_VX_OBJECT(src, VX_TYPE_IMAGE);

        dist = vxCreateDistribution(context, nbins, offset, range);
        ASSERT_VX_OBJECT(dist, VX_TYPE_DISTRIBUTION);

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuHistogram(context, src, dist));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxHistogramNode(graph, src, dist);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
        }
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxAccessDistribution(dist, &hptr, VX_READ_ONLY));
        memcpy(hist, hptr, nbins*sizeof(hist[0]));
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxCommitDistribution(dist, hptr));

        {
            /* smoke tests for query distribution attributes */
            vx_size   attr_dims = 0;
            vx_int32  attr_offset = 0;
            vx_uint32 attr_range = 0;
            vx_size   attr_bins = 0;
            vx_uint32 attr_window = 0;
            vx_size   attr_size = 0;
            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_DIMENSIONS, &attr_dims, sizeof(attr_dims)));
            if (1 != attr_dims)
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_DIMENSIONS failed\n");

            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_OFFSET, &attr_offset, sizeof(attr_offset)));
            if (attr_offset != offset)
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_OFFSET failed\n");

            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_RANGE, &attr_range, sizeof(attr_range)));
            if (attr_range != range)
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_RANGE failed\n");

            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_BINS, &attr_bins, sizeof(attr_bins)));
            if (attr_bins != nbins)
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_BINS failed\n");

            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_WINDOW, &attr_window, sizeof(attr_window)));
            if (attr_window != range / nbins)
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_WINDOW failed\n");

            VX_CALL(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_SIZE, &attr_size, sizeof(attr_size)));
            if (attr_size < nbins*sizeof(hist[0]))
                CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_ATTRIBUTE_SIZE failed\n");
        }

        for( i = 0; i < nbins; i++ )
        {
            if( hist0[i] != hist[i] )
            {
                CT_RecordFailureAtFormat("Test case %d. width=%d, height=%d, nbins=%d, offset=%d, range=%d\n"
                                         "\tExpected: hist[%d]=%d\n"
                                         "\tActual:   hist[%d]=%d\n",
                                         __FUNCTION__, __FILE__, __LINE__,
                                         iter, width, height, nbins, offset, range,
                                         i, hist0[i], i, hist[i]);
                break;
            }
        }
        if( i < nbins )
            break;
        vxReleaseImage(&src);
        vxReleaseDistribution(&dist);
        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }
}

TESTCASE_TESTS(Histogram, testOnRandom)
