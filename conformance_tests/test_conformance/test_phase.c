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

#include "shared_functions.h"

static void reference_phase(CT_Image dx, CT_Image dy, CT_Image phase)
{
    uint32_t x, y, width, height, dxstride, dystride, phasestride;

    ASSERT(dx && dy && phase);
    ASSERT(dx->format == VX_DF_IMAGE_S16 && dy->format == VX_DF_IMAGE_S16 && phase->format == VX_DF_IMAGE_U8);
    ASSERT(dx->width > 0 && dx->height > 0 &&
           dx->width == dy->width && dx->height == dy->height &&
           dx->width == phase->width && dx->height == phase->height);
    width = dx->width;
    height = dy->height;
    dxstride = ct_stride_bytes(dx);
    dystride = ct_stride_bytes(dy);
    phasestride = ct_stride_bytes(phase);

    for( y = 0; y < height; y++ )
    {
        const int16_t* dxptr = (const int16_t*)(dx->data.y + y*dxstride);
        const int16_t* dyptr = (const int16_t*)(dy->data.y + y*dystride);
        uint8_t* phaseptr = (uint8_t*)(phase->data.y + y*phasestride);
        for( x = 0; x < width; x++ )
        {
            double val = atan2(dyptr[x], dxptr[x])*256/(3.1415926535897932384626433832795*2);
            int ival;
            if( val < 0 )
                val += 256.;
            ival = (int)floor(val + 0.5);
            if( ival >= 256 )
                ival -= 256;
            phaseptr[x] = CT_CAST_U8(ival);
        }
    }
}


TESTCASE(Phase, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    int use_sobel;
    vx_df_image format;
} format_arg;

#undef UseSobel
#undef Random
#define UseSobel 1
#define Random 0

#define PHASE_TEST_CASE(imm, sob, tp) \
    {#imm "/" #sob "/" #tp, CT_##imm##_MODE, sob, VX_DF_IMAGE_##tp}

TEST_WITH_ARG(Phase, testOnRandom, format_arg,
              PHASE_TEST_CASE(Immediate, UseSobel, S16),
              PHASE_TEST_CASE(Immediate, Random, S16),
              PHASE_TEST_CASE(Graph, UseSobel, S16),
              PHASE_TEST_CASE(Graph, Random, S16),
              )
{
    int dxformat = arg_->format;
    int mode = arg_->mode;
    int use_sobel = arg_->use_sobel;
    int srcformat = dxformat == VX_DF_IMAGE_S16 ? VX_DF_IMAGE_U8 : -1;
    int phaseformat = dxformat == VX_DF_IMAGE_S16 ? VX_DF_IMAGE_U8 : -1;
    vx_image presrc=0, src=0, dx=0, dy=0, phase=0;
    CT_Image src0, dx0, dy0, phase0, phase1;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;
    int iter, niters = 100;
    uint64_t rng;
    int srcmin = 0, srcmax = 256;
    int dxmin = -32768, dxmax = 32768;
    vx_border_mode_t border;

    ASSERT( srcformat != -1 && phaseformat != -1 );
    rng = CT()->seed_;
    border.mode = //VX_BORDER_MODE_UNDEFINED;
                  VX_BORDER_MODE_REPLICATE;

    ASSERT_EQ_VX_STATUS(VX_SUCCESS,
                        vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE,
                                              &border, sizeof(border)));
    for( iter = 0; iter < niters; iter++ )
    {
        int width = ct_roundf(ct_log_rng(&rng, 0, 10));
        int height = ct_roundf(ct_log_rng(&rng, 0, 10));

        width = CT_MAX(width, 1);
        height = CT_MAX(height, 1);

        if( !ct_check_any_size() )
        {
            width = CT_MIN((width + 7) & -8, 640);
            height = CT_MIN((height + 7) & -8, 480);
        }

        ct_update_progress(iter, niters);

        if( use_sobel )
        {
            CT_Image box_img;

            width = CT_MAX(width, 3);
            height = CT_MAX(height, 3);

            ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, srcformat, &rng, srcmin, srcmax));
            ASSERT_EQ_INT(VX_DF_IMAGE_S16, dxformat);
            ASSERT_NO_FAILURE(box_img = box3x3_create_reference_image(src0, border));
            ASSERT_NO_FAILURE(sobel3x3_create_reference_image(box_img, border, &dx0, &dy0));

            if( border.mode == VX_BORDER_MODE_UNDEFINED )
            {
                width -= 2;
                height -= 2;
                ASSERT_NO_FAILURE(ct_adjust_roi(dx0, 1, 1, 1, 1));
                ASSERT_NO_FAILURE(ct_adjust_roi(dy0, 1, 1, 1, 1));
            }
            vxReleaseImage(&dx);
            vxReleaseImage(&dy);
        }
        else
        {
            int k, maxk = CT_RNG_NEXT_INT(rng, 0, 20);
            int extreme_vals[] = { dxmin, 0, dxmax };
            ASSERT_NO_FAILURE(dx0 = ct_allocate_ct_image_random(width, height, dxformat, &rng, dxmin, dxmax));
            ASSERT_NO_FAILURE(dy0 = ct_allocate_ct_image_random(width, height, dxformat, &rng, dxmin, dxmax));

            // add some extreme points to the generated Images
            for( k = 0; k < maxk; k++ )
            {
                int x = CT_RNG_NEXT_INT(rng, 0, width);
                int y = CT_RNG_NEXT_INT(rng, 0, height);
                int dxval = extreme_vals[CT_RNG_NEXT_INT(rng, 0, 3)];
                int dyval = extreme_vals[CT_RNG_NEXT_INT(rng, 0, 3)];
                dx0->data.s16[dx0->stride*y + x] = (int16_t)dxval;
                dy0->data.s16[dy0->stride*y + x] = (int16_t)dyval;
            }
            presrc = src = 0;
        }

        dx = ct_image_to_vx_image(dx0, context);
        ASSERT_VX_OBJECT(dx, VX_TYPE_IMAGE);
        dy = ct_image_to_vx_image(dy0, context);
        ASSERT_VX_OBJECT(dy, VX_TYPE_IMAGE);

        ASSERT_NO_FAILURE(phase0 = ct_allocate_image(width, height, phaseformat));
        ASSERT_NO_FAILURE(reference_phase(dx0, dy0, phase0));
        phase = vxCreateImage(context, width, height, phaseformat);
        ASSERT_VX_OBJECT(phase, VX_TYPE_IMAGE);

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuPhase(context, dx, dy, phase));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxPhaseNode(graph, dx, dy, phase);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
        }
        phase1 = ct_image_from_vx_image(phase);

        ASSERT_CTIMAGE_NEARWRAP(phase0, phase1, 1, 0);
        if(presrc)
            vxReleaseImage(&presrc);
        if(src)
            vxReleaseImage(&src);
        vxReleaseImage(&dx);
        vxReleaseImage(&dy);
        vxReleaseImage(&phase);
        if(node)
            vxReleaseNode(&node);
        if(graph)
            vxReleaseGraph(&graph);
        ASSERT(node == 0 && graph == 0);
        CT_CollectGarbage(CT_GC_IMAGE);
    }
}

TESTCASE_TESTS(Phase, testOnRandom)
