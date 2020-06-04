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

typedef vx_coordinates2d_t Point;

static void reference_minmaxloc(CT_Image src, int* _minval, int* _maxval,
                                uint32_t* _mincount, uint32_t* _maxcount)
{
    Point pt={0, 0};
    int minval = INT_MAX, maxval = INT_MIN;
    int format = src ? src->format : VX_DF_IMAGE_U8;
    uint32_t mincount = 0, maxcount = 0, stride;

    ASSERT(src);
    ASSERT(src->width > 0 && src->height > 0);
    stride = ct_stride_bytes(src);

#define CASE_MINMAXLOC(format, type) \
    case format: \
    for( pt.y = 0; pt.y < src->height; pt.y++ ) \
    { \
        const type* ptr = (const type*)(src->data.y + stride*pt.y); \
        for( pt.x = 0; pt.x < src->width; pt.x++ ) \
        { \
            int val = ptr[pt.x]; \
            if( val <= minval ) \
            { \
                if(val < minval) \
                { \
                    minval = val; \
                    mincount = 0; \
                } \
                mincount++; \
            } \
            if( val >= maxval ) \
            { \
                if(val > maxval) \
                { \
                    maxval = val; \
                    maxcount = 0; \
                } \
                maxcount++; \
            } \
        } \
    } \
    break

    switch(format)
    {
    CASE_MINMAXLOC(VX_DF_IMAGE_U8, uint8_t);
    CASE_MINMAXLOC(VX_DF_IMAGE_S16, int16_t);
    CASE_MINMAXLOC(VX_DF_IMAGE_S32, int32_t);
    default:
        FAIL("Unsupported image format: (%d)", &src->format);
    }

    *_minval = minval;
    *_maxval = maxval;
    if(_mincount)
        *_mincount = mincount;
    if(_maxcount)
        *_maxcount = maxcount;
}

static void reference_minmax(CT_Image src, int* _minval, int* _maxval)
{
    reference_minmaxloc(src, _minval, _maxval, 0, 0);
}

static int cmp_pt(const void* a, const void* b)
{
    const Point* pa = (const Point*)a;
    const Point* pb = (const Point*)b;
    int d = pa->y - pb->y;
    return d ? d : (int)(pa->x - pb->x);
}

static void ct_sort_points(Point* ptbuf, vx_size npoints)
{
    qsort(ptbuf, npoints, sizeof(ptbuf[0]), cmp_pt);
}

static void ct_set_random_pixels(CT_Image image, uint64_t* rng, int where_count, int what_count, const int* valarr)
{
    int format = image->format, i;
    uint32_t stride = ct_stride_bytes(image);

    #define CASE_SET_RANDOM(format, type, cast_macro) \
    case format: \
        for( i = 0; i < where_count; i++) \
        { \
            int y = CT_RNG_NEXT_INT(*rng, 0, image->height); \
            int x = CT_RNG_NEXT_INT(*rng, 0, image->width); \
            int k = CT_RNG_NEXT_INT(*rng, 0, what_count); \
            int val = valarr[k]; \
            ((type*)(image->data.y + stride*y))[x] = cast_macro(val); \
        } \
        break

    switch(format)
    {
    CASE_SET_RANDOM(VX_DF_IMAGE_U8, uint8_t, CT_CAST_U8);
    CASE_SET_RANDOM(VX_DF_IMAGE_U16, uint16_t, CT_CAST_U16);
    CASE_SET_RANDOM(VX_DF_IMAGE_S16, int16_t, CT_CAST_S16);
    CASE_SET_RANDOM(VX_DF_IMAGE_S32, int32_t, CT_CAST_S32);
    default:
        CT_ADD_FAILURE("unsupported image format %d", format);
    }
}

TESTCASE(MinMaxLoc, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_df_image format;
} format_arg;

#define MINMAXLOC_TEST_CASE(imm, tp) \
    {#imm "/" #tp, CT_##imm##_MODE, VX_DF_IMAGE_##tp}

TEST_WITH_ARG(MinMaxLoc, testOnRandom, format_arg,
              MINMAXLOC_TEST_CASE(Immediate, U8),
              MINMAXLOC_TEST_CASE(Graph, U8),
              MINMAXLOC_TEST_CASE(Immediate, S16),
              MINMAXLOC_TEST_CASE(Graph, S16),
              )
{
    const int MAX_CAP = 300;
    int format = arg_->format;
    int mode = arg_->mode;
    vx_image src;
    CT_Image src0;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;
    int iter, k, niters = 100;
    uint64_t rng;
    int a, b;
    int minval0 = 0, maxval0 = 0, minval = 0, maxval = 0;
    uint32_t mincount0 = 0, maxcount0 = 0, mincount = 0, maxcount = 0;
    vx_scalar minval_, maxval_, mincount_, maxcount_;
    vx_array minloc_ = 0, maxloc_ = 0;
    vx_enum sctype = format == VX_DF_IMAGE_U8 ? VX_TYPE_UINT8 :
                     format == VX_DF_IMAGE_S16 ? VX_TYPE_INT16 :
                     VX_TYPE_INT32;
    uint32_t pixsize = ct_image_bits_per_pixel(format)/8;
    Point* ptbuf = 0;
    vx_size bufbytes = 0, npoints = 0, bufcap = 0;

    if( format == VX_DF_IMAGE_U8 )
        a = 0, b = 256;
    else if( format == VX_DF_IMAGE_S16 )
        a = -32768, b = 32768;
    else
        a = INT_MIN/3, b = INT_MAX/3;

    minval_ = ct_scalar_from_int(context, sctype, 0);
    maxval_ = ct_scalar_from_int(context, sctype, 0);
    mincount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    maxcount_ = ct_scalar_from_int(context, VX_TYPE_UINT32, 0);
    minloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, MAX_CAP);
    maxloc_ = vxCreateArray(context, VX_TYPE_COORDINATES2D, MAX_CAP);
    ASSERT(vxGetStatus((vx_reference)minloc_) == VX_SUCCESS && vxGetStatus((vx_reference)maxloc_) == VX_SUCCESS);

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int return_loc = CT_RNG_NEXT_INT(rng, 0, 2);
        int return_count = CT_RNG_NEXT_INT(rng, 0, 2);
        uint32_t stride;
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

        src0 = ct_allocate_ct_image_random(width, height, format, &rng, a, b);
        stride = ct_stride_bytes(src0);
        if( iter % 3 == 0 )
        {
            int mm[2], maxk;
            reference_minmax(src0, &mm[0], &mm[1]);
            maxk = CT_RNG_NEXT_INT(rng, 0, 100);
            // make sure that there are several pixels with minimum/maximum value
            ct_set_random_pixels(src0, &rng, maxk, 2, mm);
        }
        reference_minmaxloc(src0, &minval0, &maxval0, &mincount0, &maxcount0);
        src = ct_image_to_vx_image(src0, context);

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuMinMaxLoc(context, src, minval_, maxval_,
                         return_loc ? minloc_ : 0,
                         return_loc ? maxloc_ : 0,
                         return_count ? mincount_ : 0,
                         return_count ? maxcount_ : 0));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxMinMaxLocNode(graph, src, minval_, maxval_,
                                   return_loc ? minloc_ : 0,
                                   return_loc ? maxloc_ : 0,
                                   return_count ? mincount_ : 0,
                                   return_count ? maxcount_ : 0);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
        }

        minval = ct_scalar_as_int(minval_);
        maxval = ct_scalar_as_int(maxval_);
        if( return_count )
        {
            mincount = ct_scalar_as_int(mincount_);
            maxcount = ct_scalar_as_int(maxcount_);
        }
        else
        {
            mincount = mincount0;
            maxcount = maxcount0;
        }

        if( minval != minval0 || maxval != maxval0 || mincount != mincount0 || maxcount != maxcount0 )
        {
            CT_RecordFailureAtFormat("Test case %d. width=%d, height=%d,\n"
                                     "\tExpected: minval=%d, maxval=%d, mincount=%d, maxcount=%d\n"
                                     "\tActual:   minval=%d, maxval=%d, mincount=%d, maxcount=%d\n",
                                     __FUNCTION__, __FILE__, __LINE__,
                                     iter, width, height,
                                     minval0, maxval0, mincount0, maxcount0,
                                     minval, maxval, mincount, maxcount);
            break;
        }

        if( return_loc )
        {
            uint8_t* roi_ptr = src0->data.y;
            for( k = 0; k < 2; k++ )
            {
                int val0 = k == 0 ? minval : maxval;
                uint32_t i, count = k == 0 ? mincount : maxcount;
                vx_array loc = k == 0 ? minloc_ : maxloc_;
                vx_enum tp;
                union
                {
                    uint8_t u8;
                    int16_t s16;
                    int32_t s32;
                }
                uval;
                if( format == VX_DF_IMAGE_U8 )
                    uval.u8 = (uint8_t)val0;
                else if( format == VX_DF_IMAGE_S16 )
                    uval.s16 = (int16_t)val0;
                else
                    uval.s32 = (int32_t)val0;

                tp = ct_read_array(loc, (void**)&ptbuf, &bufbytes, &npoints, &bufcap);
                ASSERT(tp == VX_TYPE_COORDINATES2D);
                ASSERT(npoints == CT_MIN(bufcap, (vx_size)count));

                ct_sort_points(ptbuf, npoints);
                for( i = 0; i < npoints; i++ )
                {
                    Point p = ptbuf[i];
                    if( i > 0 )
                    {
                        // all the extrema locations should be different
                        ASSERT(p.x != ptbuf[i-1].x || p.y != ptbuf[i-1].y);
                    }
                    // value at each extrema location should match the extremum value
                    ASSERT(memcmp(roi_ptr + p.y*stride + p.x*pixsize, &uval, pixsize) == 0);
                }
            }
        }

        vxReleaseImage(&src);
        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }

    vxReleaseScalar(&minval_);
    vxReleaseScalar(&maxval_);
    vxReleaseScalar(&mincount_);
    vxReleaseScalar(&maxcount_);
    vxReleaseArray(&minloc_);
    vxReleaseArray(&maxloc_);

    if(ptbuf)
        free(ptbuf);
}

TESTCASE_TESTS(MinMaxLoc, testOnRandom)
