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

static int get_yuv_params(CT_Image img, uint8_t** ptrY, uint8_t** ptrU, uint8_t** ptrV,
                           uint32_t* strideY, uint32_t* deltaY,
                           uint32_t* strideC, uint32_t* deltaC,
                           uint32_t* shiftX, uint32_t* shiftY, int* code)
{
    int format = img->format;
    int is_yuv = 0;
    uint32_t stride = ct_stride_bytes(img);
    uint32_t height = img->height;

    *ptrY = img->data.y;
    *strideY = *strideC = stride;
    *deltaY = *deltaC = 1;
    *shiftX = *shiftY = 0;

    if( format == VX_DF_IMAGE_YUV4 )
    {
        *ptrU = *ptrY + stride*height;
        *ptrV = *ptrU + stride*height;
        *shiftX = *shiftY = 0;
        is_yuv = 1;
    }
    else if( format == VX_DF_IMAGE_IYUV )
    {
        *ptrU = *ptrY + stride*height;
        *ptrV = *ptrU + (stride*height)/4;
        *strideC = stride/2;
        *shiftX = *shiftY = 1;
        is_yuv = 1;
    }
    else if( format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21 )
    {
        if( format == VX_DF_IMAGE_NV12 )
        {
            *ptrU = *ptrY + stride*height;
            *ptrV = *ptrU + 1;
        }
        else
        {
            *ptrV = *ptrY + stride*height;
            *ptrU = *ptrV + 1;
        }
        *deltaC = 2;
        *shiftX = *shiftY = 1;
        is_yuv = 1;
    }
    else if( format == VX_DF_IMAGE_YUYV || format == VX_DF_IMAGE_UYVY )
    {
        if( format == VX_DF_IMAGE_YUYV )
        {
            *ptrU = *ptrY + 1;
        }
        else
        {
            *ptrU = *ptrY;
            *ptrY = *ptrU + 1;
        }
        *ptrV = *ptrU + 2;
        *deltaY = 2;
        *deltaC = 4;
        *shiftX = 1;
        *shiftY = 0;
        is_yuv = 1;
    }
    *code = *shiftX == 0 ? 444 : *shiftY == 0 ? 422 : 420;
    return is_yuv;
}

static void rgb2yuv_bt709(uint8_t r, uint8_t g, uint8_t b, uint8_t* y, uint8_t* u, uint8_t* v)
{
    int yval = (int)(r*0.2126f + g*0.7152f + b*0.0722f + 0.5f);
    int uval = (int)(-r*0.1146f - g*0.3854 + b*0.5f + 128.5f);
    int vval = (int)(r*0.5f - g*0.4542f - b*0.0458f + 128.5f);
    *y = CT_CAST_U8(yval);
    *u = CT_CAST_U8(uval);
    *v = CT_CAST_U8(vval);
}

static void yuv2rgb_bt709(uint8_t y, uint8_t u, uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b)
{
    int rval = (int)(y + 1.5748f*(v-128) + 0.5f);
    int gval = (int)(y - 0.1873f*(u-128) - 0.4681f*(v-128) + 0.5f);
    int bval = (int)(y + 1.8556f*(u-128) + 0.5f);
    *r = CT_CAST_U8(rval);
    *g = CT_CAST_U8(gval);
    *b = CT_CAST_U8(bval);
}

static void reference_colorconvert(CT_Image src, CT_Image dst)
{
    uint32_t x, y, width, height, srcstride, dststride;
    int srcformat = src->format;
    int dstformat = dst->format;
    uint8_t *srcptrY=0, *srcptrU=0, *srcptrV=0;
    uint8_t *dstptrY=0, *dstptrU=0, *dstptrV=0;
    uint32_t srcstrideY=0, srcdeltaY=1, srcstrideC=0, srcdeltaC=1;
    uint32_t dststrideY=0, dstdeltaY=1, dststrideC=0, dstdeltaC=1;
    uint32_t srcshiftX = 1, srcshiftY = 1;
    uint32_t dstshiftX = 1, dstshiftY = 1;
    int srcYUV, dstYUV;
    int srccode=0, dstcode=0;

    ASSERT(src && dst);
    ASSERT(src->width > 0 && src->height > 0 &&
           src->width == dst->width && src->height == dst->height);

    width = src->width;
    height = src->height;
    srcstride = ct_stride_bytes(src);
    dststride = ct_stride_bytes(dst);

    srcYUV = get_yuv_params(src, &srcptrY, &srcptrU, &srcptrV, &srcstrideY,
                            &srcdeltaY, &srcstrideC, &srcdeltaC,
                            &srcshiftX, &srcshiftY, &srccode);
    dstYUV = get_yuv_params(dst, &dstptrY, &dstptrU, &dstptrV, &dststrideY,
                            &dstdeltaY, &dststrideC, &dstdeltaC,
                            &dstshiftX, &dstshiftY, &dstcode);

    if( srcformat == VX_DF_IMAGE_RGB || srcformat == VX_DF_IMAGE_RGBX )
    {
        int scn = ct_channels(srcformat);
        if( dstformat == VX_DF_IMAGE_RGB || dstformat == VX_DF_IMAGE_RGBX )
        {
            int dcn = ct_channels(dstformat);

            for( y = 0; y < height; y++ )
            {
                const uint8_t* srcptr = (const uint8_t*)(src->data.y + y*srcstride);
                uint8_t* dstptr = (uint8_t*)(dst->data.y + y*dststride);
                for( x = 0; x < width; x++, srcptr += scn, dstptr += dcn )
                {
                    dstptr[0] = srcptr[0];
                    dstptr[1] = srcptr[1];
                    dstptr[2] = srcptr[2];
                    if(dcn == 4)
                        dstptr[3] = 255;
                }
            }

        }
        else if( dstYUV )
        {
            if( dstcode == 444 )
            {
                for( y = 0; y < height; y++ )
                {
                    const uint8_t* srcptr = (const uint8_t*)(src->data.y + y*srcstride);
                    for( x = 0; x < width; x++, srcptr += scn )
                    {
                        rgb2yuv_bt709(srcptr[0], srcptr[1], srcptr[2],
                                      dstptrY + dststrideY*y + dstdeltaY*x,
                                      dstptrU + dststrideC*y + dstdeltaC*x,
                                      dstptrV + dststrideC*y + dstdeltaC*x);
                    }
                }
            }
            else if( dstcode == 422 )
            {
                for( y = 0; y < height; y++ )
                {
                    const uint8_t* srcptr = (const uint8_t*)(src->data.y + y*srcstride);
                    for( x = 0; x < width; x += 2, srcptr += scn*2 )
                    {
                        uint8_t u0 = 0, v0 = 0, u1 = 0, v1 = 0;
                        rgb2yuv_bt709(srcptr[0], srcptr[1], srcptr[2],
                                      dstptrY + dststrideY*y + dstdeltaY*x, &u0, &v0);
                        rgb2yuv_bt709(srcptr[scn], srcptr[scn+1], srcptr[scn+2],
                                      dstptrY + dststrideY*y + dstdeltaY*(x+1), &u1, &v1);
                        dstptrU[dststrideC*y + dstdeltaC*(x/2)] = (uint8_t)((u0 + u1) >> 1);
                        dstptrV[dststrideC*y + dstdeltaC*(x/2)] = (uint8_t)((v0 + v1) >> 1);
                    }
                }
            }
            else if( dstcode == 420 )
            {
                for( y = 0; y < height; y += 2 )
                {
                    const uint8_t* srcptr = (const uint8_t*)(src->data.y + y*srcstride);
                    for( x = 0; x < width; x += 2, srcptr += scn*2 )
                    {
                        uint8_t u[4], v[4];
                        rgb2yuv_bt709(srcptr[0], srcptr[1], srcptr[2],
                                      dstptrY + dststrideY*y + dstdeltaY*x, &u[0], &v[0]);
                        rgb2yuv_bt709(srcptr[scn], srcptr[scn+1], srcptr[scn+2],
                                      dstptrY + dststrideY*y + dstdeltaY*(x+1), &u[1], &v[1]);
                        rgb2yuv_bt709(srcptr[srcstride+0], srcptr[srcstride+1], srcptr[srcstride+2],
                                      dstptrY + dststrideY*(y+1) + dstdeltaY*x, &u[2], &v[2]);
                        rgb2yuv_bt709(srcptr[srcstride+scn], srcptr[srcstride+scn+1], srcptr[srcstride+scn+2],
                                      dstptrY + dststrideY*(y+1) + dstdeltaY*(x+1), &u[3], &v[3]);
                        dstptrU[dststrideC*(y/2) + dstdeltaC*(x/2)] = (uint8_t)((u[0] + u[1] + u[2] + u[3]) >> 2);
                        dstptrV[dststrideC*(y/2) + dstdeltaC*(x/2)] = (uint8_t)((v[0] + v[1] + v[2] + v[3]) >> 2);
                    }
                }
            }
        }
    }
    else if( srcYUV )
    {
        if( dstformat == VX_DF_IMAGE_RGB || dstformat == VX_DF_IMAGE_RGBX )
        {
            int dcn = ct_channels(dstformat);

            for( y = 0; y < height; y++ )
            {
                uint8_t* dstptr = (uint8_t*)(dst->data.y + y*dststride);
                for( x = 0; x < width; x++, dstptr += dcn )
                {
                    int xc = x >> srcshiftX, yc = y >> srcshiftY;
                    yuv2rgb_bt709(srcptrY[srcstrideY*y + srcdeltaY*x],
                                  srcptrU[srcstrideC*yc + srcdeltaC*xc],
                                  srcptrV[srcstrideC*yc + srcdeltaC*xc],
                                  dstptr, dstptr + 1, dstptr + 2);
                    if( dcn == 4 )
                        dstptr[3] = 255;
                }
            }
        }
        else if( dstYUV )
        {
            if( srccode <= dstcode )
            {
                // if both src and dst are YUV formats and
                // the source image chroma resolution
                // is smaller then we just replicate the chroma components
                for( y = 0; y < height; y++ )
                {
                    for( x = 0; x < width; x++ )
                    {
                        int dstYC = y >> dstshiftY, dstXC = x >> dstshiftX;
                        int srcYC = y >> srcshiftY, srcXC = x >> srcshiftX;
                        dstptrY[dststrideY*y + dstdeltaY*x] = srcptrY[srcstrideY*y + srcdeltaY*x];
                        dstptrU[dststrideC*dstYC + dstdeltaC*dstXC] = srcptrU[srcstrideC*srcYC + srcdeltaC*srcXC];
                        dstptrV[dststrideC*dstYC + dstdeltaC*dstXC] = srcptrV[srcstrideC*srcYC + srcdeltaC*srcXC];
                    }
                }
            }
            else if( srccode == 422 && dstcode == 420 )
            {
                // if both src and dst are YUV formats and
                // the source image chroma resolution
                // is larger then we have to average chroma samples
                for( y = 0; y < height; y += 2 )
                {
                    for( x = 0; x < width; x++ )
                    {
                        int dstYC = y >> dstshiftY, dstXC = x >> dstshiftX;
                        int srcYC = y >> srcshiftY, srcXC = x >> srcshiftX;
                        dstptrY[dststrideY*y + dstdeltaY*x] = srcptrY[srcstrideY*y + srcdeltaY*x];
                        dstptrY[dststrideY*(y+1) + dstdeltaY*x] = srcptrY[srcstrideY*(y+1) + srcdeltaY*x];

                        dstptrU[dststrideC*dstYC + dstdeltaC*dstXC] =
                            (uint8_t)((srcptrU[srcstrideC*srcYC + srcdeltaC*srcXC] +
                                       srcptrU[srcstrideC*(srcYC+1) + srcdeltaC*srcXC]) >> 1);

                        dstptrV[dststrideC*dstYC + dstdeltaC*dstXC] =
                            (uint8_t)((srcptrV[srcstrideC*srcYC + srcdeltaC*srcXC] +
                                       srcptrV[srcstrideC*(srcYC+1) + srcdeltaC*srcXC]) >> 1);
                    }
                }
            }
        }
    }
}


static int cmp_color_images(CT_Image img0, CT_Image img1, int ythresh, int cthresh)
{
    uint32_t x, y, width, height, stride0, stride1;
    int format0 = img0->format;
    int format1 = img1->format;
    uint8_t *ptrY0=0, *ptrU0=0, *ptrV0=0;
    uint8_t *ptrY1=0, *ptrU1=0, *ptrV1=0;
    uint32_t strideY0=0, deltaY0=1, strideC0=0, deltaC0=1;
    uint32_t strideY1=0, deltaY1=1, strideC1=0, deltaC1=1;
    uint32_t shiftX0 = 1, shiftY0 = 1;
    uint32_t shiftX1 = 1, shiftY1 = 1;
    int YUV0, YUV1;
    int code0=0, code1=0;

    ASSERT_(return -1, img0 && img1);
    ASSERT_(return -1, img0->width > 0 && img0->height > 0 &&
           img0->width == img1->width && img0->height == img1->height &&
           format0 == format1);

    width = img0->width;
    height = img0->height;
    stride0 = ct_stride_bytes(img0);
    stride1 = ct_stride_bytes(img1);

    YUV0 = get_yuv_params(img0, &ptrY0, &ptrU0, &ptrV0, &strideY0,
                            &deltaY0, &strideC0, &deltaC0,
                            &shiftX0, &shiftY0, &code0);
    YUV1 = get_yuv_params(img1, &ptrY1, &ptrU1, &ptrV1, &strideY1,
                          &deltaY1, &strideC1, &deltaC1,
                          &shiftX1, &shiftY1, &code1);

    if( format0 == VX_DF_IMAGE_RGB || format0 == VX_DF_IMAGE_RGBX )
    {
        int cn = ct_channels(format0);
        for( y = 0; y < height; y++ )
        {
            const uint8_t* ptr0 = (const uint8_t*)(img0->data.y + y*stride0);
            const uint8_t* ptr1 = (const uint8_t*)(img1->data.y + y*stride1);
            for( x = 0; x < width*cn; x++ )
            {
                if( abs(ptr0[x] - ptr1[x]) > ythresh )
                {
                    printf("images are very different at (%d, %d): %d vs %d\n", x, y, ptr0[x], ptr1[x]);
                    return -1;
                }
            }
        }
    }
    else
    {
        ASSERT_(return -1, YUV0 != 0 && YUV1 != 0 && code0 == code1);
        for( y = 0; y < height; y++ )
        {
            const uint8_t* tempptrY0 = (const uint8_t*)(ptrY0 + y*strideY0);
            const uint8_t* tempptrY1 = (const uint8_t*)(ptrY1 + y*strideY1);
            const uint8_t* tempptrU0_row = (const uint8_t*)(ptrU0 + (y>>shiftY0)*strideC0);
            const uint8_t* tempptrU1_row = (const uint8_t*)(ptrU1 + (y>>shiftY1)*strideC1);
            const uint8_t* tempptrV0_row = (const uint8_t*)(ptrV0 + (y>>shiftY0)*strideC0);
            const uint8_t* tempptrV1_row = (const uint8_t*)(ptrV1 + (y>>shiftY1)*strideC1);
            for( x = 0; x < width; x++, tempptrY0 += deltaY0, tempptrY1 += deltaY1 )
            {
                const uint8_t* tempptrU0 = tempptrU0_row + (x >> shiftX0)*deltaC0;
                const uint8_t* tempptrU1 = tempptrU1_row + (x >> shiftX1)*deltaC1;
                const uint8_t* tempptrV0 = tempptrV0_row + (x >> shiftX0)*deltaC0;
                const uint8_t* tempptrV1 = tempptrV1_row + (x >> shiftX1)*deltaC1;

                if( abs(tempptrY0[0] - tempptrY1[0]) > ythresh ||
                    abs(tempptrU0[0] - tempptrU1[0]) > cthresh ||
                    abs(tempptrV0[0] - tempptrV1[0]) > cthresh )
                {
                    printf("images are very different at (%d, %d): (%d, %d, %d) vs (%d, %d, %d)\n",
                           x, y, tempptrY0[0], tempptrU0[0], tempptrV0[0], tempptrY1[0], tempptrU1[0], tempptrV1[0]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

TESTCASE(ColorConvert, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    vx_df_image srcformat;
    vx_df_image dstformat;
    int mode;
    int ythresh;
    int cthresh;
} format_arg;

#if 0
static void ct_print_image(CT_Image img, const char* name)
{
    uint32_t p, x, y, nplanes=1, width[3] = {img->width, 0, 0}, height[3] = {img->height, 0, 0};
    uint32_t stride[3] = {ct_stride_bytes(img), 0, 0};
    const uint8_t* ptr = img->data.y;
    int format = img->format;

    if( format == VX_DF_IMAGE_RGB || format == VX_DF_IMAGE_RGBX || format == VX_DF_IMAGE_UYVY || format == VX_DF_IMAGE_YUYV )
        width[0] *= format == VX_DF_IMAGE_RGB ? 3 : 4;
    else if( format == VX_DF_IMAGE_YUV4 || format == VX_DF_IMAGE_IYUV )
    {
        int scale = format == VX_DF_IMAGE_YUV4 ? 1 : 2;
        nplanes = 3;
        width[1] = width[2] = width[0]/scale;
        height[1] = height[2] = height[0]/scale;
        stride[1] = stride[2] = stride[0]/scale;
    }
    else if( format == VX_DF_IMAGE_NV12 || format == VX_DF_IMAGE_NV21 )
    {
        nplanes = 2;
        width[1] = width[0];
        height[1] = height[0]/2;
        stride[1] = stride[0];
    }

    printf("=========== %s =======\n", name);

    for( p = 0; p < nplanes; p++ )
    {
        for( y = 0; y < height[p]; y++, ptr += stride[p] )
        {
            for( x = 0; x < width[p]; x++ )
            {
                printf("%4d", ptr[x]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("---------------------------\n");
}
#endif

#define CVT_CASE_(imm, from, to, ythresh, cthresh) \
    {#imm "/" #from "=>" #to, VX_DF_IMAGE_##from, VX_DF_IMAGE_##to, CT_##imm##_MODE, ythresh, cthresh}

#define CVT_CASE(from, to, ythresh, cthresh) \
    CVT_CASE_(Immediate, from, to, ythresh, cthresh), \
    CVT_CASE_(Graph, from, to, ythresh, cthresh)

TEST_WITH_ARG(ColorConvert, testOnRandomAndNatural, format_arg,
              CVT_CASE(RGB, RGBX, 0, 0),
              CVT_CASE(RGB, NV12, 1, 255),
              CVT_CASE(RGB, IYUV, 1, 255),
              CVT_CASE(RGB, YUV4, 1, 1),

              CVT_CASE(RGBX, RGB, 0, 0),
              CVT_CASE(RGBX, NV12, 1, 255),
              CVT_CASE(RGBX, IYUV, 1, 255),
              CVT_CASE(RGBX, YUV4, 1, 1),

              CVT_CASE(NV12, RGB, 255, 255),
              CVT_CASE(NV12, RGBX, 255, 255),
              CVT_CASE(NV12, IYUV, 0, 0),
              CVT_CASE(NV12, YUV4, 0, 255),

              CVT_CASE(NV21, RGB, 255, 255),
              CVT_CASE(NV21, RGBX, 255, 255),
              CVT_CASE(NV21, IYUV, 0, 0),
              CVT_CASE(NV21, YUV4, 0, 255),

              CVT_CASE(UYVY, RGB, 255, 255),
              CVT_CASE(UYVY, RGBX, 255, 255),
              CVT_CASE(UYVY, NV12, 0, 255),
              CVT_CASE(UYVY, IYUV, 0, 255),

              CVT_CASE(YUYV, RGB, 255, 255),
              CVT_CASE(YUYV, RGBX, 255, 255),
              CVT_CASE(YUYV, NV12, 0, 255),
              CVT_CASE(YUYV, IYUV, 0, 255),

              CVT_CASE(IYUV, RGB, 255, 255),
              CVT_CASE(IYUV, RGBX, 255, 255),
              CVT_CASE(IYUV, NV12, 0, 0),
              CVT_CASE(IYUV, YUV4, 0, 255),
              )
{
    int srcformat = arg_->srcformat;
    int dstformat = arg_->dstformat;
    int ythresh = arg_->ythresh;
    int cthresh = arg_->cthresh;
    int mode = arg_->mode;
    vx_image src=0, dst=0;
    CT_Image src0, dst0, dst1;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;
    int iter, niters = 50;
    uint64_t rng;

    rng = CT()->seed_;

    for( iter = 0; iter < niters; iter++ )
    {
        int width = ct_roundf(ct_log_rng(&rng, 0, 10));
        int height = ct_roundf(ct_log_rng(&rng, 0, 10));
        vx_enum range = VX_CHANNEL_RANGE_FULL;
        vx_enum space = VX_COLOR_SPACE_BT709;

        width = CT_MAX((width+1)&-2, 2);
        height = CT_MAX((height+1)&-2, 2);

        if( !ct_check_any_size() )
        {
            width = CT_MIN((width + 7) & -8, 640);
            height = CT_MIN((height + 7) & -8, 480);
        }

        if( srcformat == VX_DF_IMAGE_RGB || srcformat == VX_DF_IMAGE_RGBX )
        {
            int scn = srcformat == VX_DF_IMAGE_RGB ? 3 : 4;
            if( iter == 0 )
            {
                ASSERT_NO_FAILURE(src0 = ct_read_image("lena.bmp", scn));
                width = src0->width;
                height = src0->height;
            }
            else if( iter == 1 )
            {
                ASSERT_NO_FAILURE(src0 = ct_read_image("colors.bmp", scn));
                width = src0->width;
                height = src0->height;
            }
            else
            {
                ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, srcformat, &rng, 0, 256));
            }
        }
        else
        {
            ASSERT_NO_FAILURE(src0 = ct_allocate_ct_image_random(width, height, srcformat, &rng, 0, 256));
        }
        ASSERT_NO_FAILURE(src = ct_image_to_vx_image(src0, context));
        ASSERT_VX_OBJECT(src, VX_TYPE_IMAGE);
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetImageAttribute(src, VX_IMAGE_ATTRIBUTE_RANGE, &range, sizeof(range)));
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetImageAttribute(src, VX_IMAGE_ATTRIBUTE_SPACE, &space, sizeof(space)));

        ASSERT_NO_FAILURE(dst0 = ct_allocate_image(width, height, dstformat));
        ASSERT_NO_FAILURE(reference_colorconvert(src0, dst0));
        dst = vxCreateImage(context, width, height, dstformat);
        ASSERT_VX_OBJECT(dst, VX_TYPE_IMAGE);
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetImageAttribute(dst, VX_IMAGE_ATTRIBUTE_RANGE, &range, sizeof(range)));
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxSetImageAttribute(dst, VX_IMAGE_ATTRIBUTE_SPACE, &space, sizeof(space)));

        if( mode == CT_Immediate_MODE )
        {
            ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuColorConvert(context, src, dst));
        }
        else
        {
            graph = vxCreateGraph(context);
            ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
            node = vxColorConvertNode(graph, src, dst);
            ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
            VX_CALL(vxVerifyGraph(graph));
            VX_CALL(vxProcessGraph(graph));
        }
        dst1 = ct_image_from_vx_image(dst);

        //ASSERT_CTIMAGE_NEAR(dst0, dst1, threshold);
        ASSERT(cmp_color_images(dst0, dst1, ythresh, cthresh) >= 0);
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

TESTCASE_TESTS(ColorConvert, testOnRandomAndNatural)
