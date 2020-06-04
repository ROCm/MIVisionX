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
#include <math.h>

static const int circle[][2] =
{
    {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
    {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
    {3, 0}, {3, -1}, {2, -2}, {1, -3}, {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
    {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, {0, 3}, {1, 3}, {2, 2}, {3, 1},
};

static int check_pt(const uint8_t* ptr, int32_t stride, int t)
{
    int cval = ptr[0];
    int max_up_count = 0, max_lo_count = 0;
    int i, up_count = 0, lo_count = 0;

    for( i = 0; i < 16+9; i++ )
    {
        int val = ptr[circle[i][0] + circle[i][1]*stride];
        if( val > cval + t )
            up_count++;
        else
        {
            max_up_count = CT_MAX(max_up_count, up_count);
            up_count = 0;
        }
        if( val < cval - t )
            lo_count++;
        else
        {
            max_lo_count = CT_MAX(max_lo_count, lo_count);
            lo_count = 0;
        }
    }
    max_up_count = CT_MAX(max_up_count, up_count);
    max_lo_count = CT_MAX(max_lo_count, lo_count);
    return max_up_count >= 9 || max_lo_count >= 9;
}

static uint32_t reference_fast(CT_Image src, CT_Image dst, CT_Image mask, int threshold, int nonmax_suppression)
{
    const int r = 3;
    int x, y, width, height;
    int32_t srcstride, dststride;
    uint32_t ncorners = 0;

    ASSERT_(return 0, src && dst);
    ASSERT_(return 0, src->format == VX_DF_IMAGE_U8 && dst->format == VX_DF_IMAGE_U8);
    ASSERT_(return 0, src->width > 0 && src->height > 0 &&
           src->width == dst->width && src->height == dst->height);
    width = src->width;
    height = src->height;
    srcstride = (int32_t)ct_stride_bytes(src);
    dststride = (int32_t)ct_stride_bytes(dst);
    memset( dst->data.y, 0, (vx_size)dststride*height );

    for( y = r; y < height - r; y++ )
    {
        const uint8_t* srcptr = src->data.y + y*srcstride;
        uint8_t* dstptr = dst->data.y + y*dststride;
        for( x = r; x < width - r; x++ )
        {
            int is_corner = check_pt(srcptr + x, srcstride, threshold);
            int strength = 0;

            if( is_corner )
            {
                // determine the corner strength using binary search
                int a = threshold;
                int b = 255;
                // loop invariant:
                //    1. point is corner with threshold=a
                //    2. point is not a corner with threshold=b
                while( b - a > 1 )
                {
                    int c = (b + a)/2;
                    is_corner = check_pt(srcptr + x, srcstride, c);
                    if( is_corner )
                        a = c;
                    else
                        b = c;
                }
                strength = a;
                ncorners++;
            }
            dstptr[x] = CT_CAST_U8(strength);
        }
    }

    if( nonmax_suppression )
    {
        int32_t maskstride = (int32_t)ct_stride_bytes(mask);

        for( y = r; y < height - r; y++ )
        {
            const uint8_t* dstptr = dst->data.y + y*dststride;
            uint8_t* mptr = mask->data.y + y*maskstride;
            for( x = r; x < width - r; x++ )
            {
                const uint8_t* ptr = dstptr + x;
                int cval = ptr[0];
                mptr[x] = cval >= ptr[-1-dststride] && cval >= ptr[-dststride] && cval >= ptr[-dststride+1] && cval >= ptr[-1] &&
                          cval >  ptr[-1+dststride] && cval >  ptr[ dststride] && cval >  ptr[ dststride+1] && cval >  ptr[ 1];
            }
        }

        ncorners = 0;
        for( y = r; y < height - r; y++ )
        {
            uint8_t* dstptr = dst->data.y + y*dststride;
            const uint8_t* mptr = mask->data.y + y*maskstride;
            for( x = r; x < width - r; x++ )
            {
                if( mptr[x] )
                    ncorners++;
                else
                    dstptr[x] = 0;
            }
        }
    }
    return ncorners;
}

TESTCASE(FastCorners, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    const char* imgname;
    int threshold;
    int nonmax;
    int mode;
} format_arg;

#define MAX_BINS 256

#define FAST_TEST_CASE(imm, imgname, t, nm) \
    {#imm "/" "image=" #imgname "/" "threshold=" #t "/" "nonmax_suppression=" #nm, #imgname ".bmp", t, nm, CT_##imm##_MODE}

TEST_WITH_ARG(FastCorners, testOnNaturalImages, format_arg,
              FAST_TEST_CASE(Immediate, lena, 10, 0),
              FAST_TEST_CASE(Immediate, lena, 10, 1),
              FAST_TEST_CASE(Immediate, lena, 80, 0),
              FAST_TEST_CASE(Immediate, lena, 80, 1),
              FAST_TEST_CASE(Immediate, baboon, 10, 0),
              FAST_TEST_CASE(Immediate, baboon, 10, 1),
              FAST_TEST_CASE(Immediate, baboon, 80, 0),
              FAST_TEST_CASE(Immediate, baboon, 80, 1),
              FAST_TEST_CASE(Immediate, optflow_00, 10, 0),
              FAST_TEST_CASE(Immediate, optflow_00, 10, 1),
              FAST_TEST_CASE(Immediate, optflow_00, 80, 0),
              FAST_TEST_CASE(Immediate, optflow_00, 80, 1),

              FAST_TEST_CASE(Graph, lena, 10, 0),
              FAST_TEST_CASE(Graph, lena, 10, 1),
              FAST_TEST_CASE(Graph, lena, 80, 0),
              FAST_TEST_CASE(Graph, lena, 80, 1),
              FAST_TEST_CASE(Graph, baboon, 10, 0),
              FAST_TEST_CASE(Graph, baboon, 10, 1),
              FAST_TEST_CASE(Graph, baboon, 80, 0),
              FAST_TEST_CASE(Graph, baboon, 80, 1),
              FAST_TEST_CASE(Graph, optflow_00, 10, 0),
              FAST_TEST_CASE(Graph, optflow_00, 10, 1),
              FAST_TEST_CASE(Graph, optflow_00, 80, 0),
              FAST_TEST_CASE(Graph, optflow_00, 80, 1),
              )
{
    int mode = arg_->mode;
    const char* imgname = arg_->imgname;
    int threshold = arg_->threshold;
    int nonmax = arg_->nonmax;
    vx_image src;
    vx_node node = 0;
    vx_graph graph = 0;
    CT_Image src0, dst0, mask0, dst1;
    vx_context context = context_->vx_context_;
    vx_scalar sthresh;
    vx_array corners;
    uint32_t width, height;
    vx_float32 threshold_f = (vx_float32)threshold;
    uint32_t ncorners0, ncorners;
    vx_size corners_data_size = 0;
    vx_keypoint_t* corners_data = 0;
    uint32_t i, dst1stride;

    ASSERT_NO_FAILURE(src0 = ct_read_image(imgname, 1));
    ASSERT(src0->format == VX_DF_IMAGE_U8);

    width = src0->width;
    height = src0->height;

    ASSERT_NO_FAILURE(dst0 = ct_allocate_image(width, height, VX_DF_IMAGE_U8));
    ASSERT_NO_FAILURE(mask0 = ct_allocate_image(width, height, VX_DF_IMAGE_U8));
    ASSERT_NO_FAILURE(dst1 = ct_allocate_image(width, height, VX_DF_IMAGE_U8));
    dst1stride = ct_stride_bytes(dst1);
    memset(dst1->data.y, 0, (vx_size)dst1stride*height);

    ncorners0 = reference_fast(src0, dst0, mask0, threshold, nonmax);

    src = ct_image_to_vx_image(src0, context);
    sthresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &threshold_f);
    corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 80000);

    if( mode == CT_Immediate_MODE )
    {
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuFastCorners(context, src, sthresh, nonmax ? vx_true_e : vx_false_e,
                                                       corners, 0));
    }
    else
    {
        graph = vxCreateGraph(context);
        ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
        node = vxFastCornersNode(graph, src, sthresh, nonmax ? vx_true_e : vx_false_e, corners, 0);
        ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
        VX_CALL(vxVerifyGraph(graph));
        VX_CALL(vxProcessGraph(graph));

        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
    }

    vxReleaseImage(&src);
    vxReleaseScalar(&sthresh);
    ct_read_array(corners, (void**)&corners_data, 0, &corners_data_size, 0);
    vxReleaseArray(&corners);
    ncorners = (uint32_t)corners_data_size;

    for( i = 0; i < ncorners; i++ )
    {
        vx_keypoint_t* pt = &corners_data[i];
        int ix, iy;
        ASSERT( 0.f <= pt->x && pt->x < (float)width &&
                0.f <= pt->y && pt->y < (float)height );
        ix = (int)(pt->x + 0.5f);
        iy = (int)(pt->y + 0.5f);
        ix = CT_MIN(ix, (int)width-1);
        iy = CT_MIN(iy, (int)height-1);
        ASSERT( !nonmax || (0 < pt->strength && pt->strength <= 255) );
        dst1->data.y[dst1stride*iy + ix] = nonmax ? (uint8_t)(pt->strength + 0.5f) : 1;
    }

    free(corners_data);

    //ASSERT_EQ_CTIMAGE(dst0, dst1);

    {
    const uint32_t border = 3;
    int32_t stride0 = (int32_t)ct_stride_bytes(dst0), stride1 = (int32_t)ct_stride_bytes(dst1);
    uint32_t x, y;
    uint32_t missing0 = 0, missing1 = 0;

    for( y = border; y < height - border; y++ )
    {
        const uint8_t* ptr0 = dst0->data.y + stride0*y;
        const uint8_t* ptr1 = dst1->data.y + stride1*y;

        for( x = border; x < width - border; x++ )
        {
            if( ptr0[x] > 0 && ptr1[x] == 0 )
                missing0++;
            else if( ptr0[x] == 0 && ptr1[x] > 0 )
                missing1++;
            else if( nonmax && ptr0[x] > 0 && ptr1[x] > 0 && fabs(log10((double)ptr0[x]/ptr1[x])) >= 1 )
            {
                missing0++;
                missing1++;
            }
        }
    }

    ASSERT( missing0 <= 0.02*ncorners0 );
    ASSERT( missing1 <= 0.02*ncorners );
    }
}

TESTCASE_TESTS(FastCorners, testOnNaturalImages)
