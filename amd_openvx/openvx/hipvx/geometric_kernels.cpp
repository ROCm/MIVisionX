/*
Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/



#include "hip_common.h"
#include "hip_host_decls.h"

// ----------------------------------------------------------------------------
// VxScaleImage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    pSrcImage += srcImageStrideInBytes * (uint)fmaf((float)y, scaleInfo.y, scaleInfo.w);
    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);

    dst.x  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 24;

    fx += scaleInfo.x;

    dst.y  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ScaleImage_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    float fx, fy, fint, frac, fy0, fy1;
    float4 f;
    fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
    fy0 = floor(fy);
    fy1 = fy - fy0;
    fy0 = 1.0f - fy1;
    pSrcImage += ((uint)fy * srcImageStrideInBytes);

    fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    fint = floor(fx);
    frac = fx - fint;
    f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    dst.x = pack_(f);

    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floor(fx);
    frac = fx - fint;
    f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ScaleImage_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Replicate(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    float fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);

    if (fx >= 0.0f && fy >= 0.0f && fmaf(8.0f, scaleInfo.x, fx) < (dstWidth - 1) && fmaf(1.0f, scaleInfo.y, fy) < (dstHeight - 1)) {
        uint2 dst;
        float fx, fy, fint, frac, fy0, fy1;
        float4 f;
        fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
        fy0 = floor(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        pSrcImage += ((uint)fy * srcImageStrideInBytes);

        fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
        fint = floor(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.x = pack_(f);

        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.y = pack_(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
    else {
        float fxlimit = (float)(dstWidth - 1);
        float fylimit = (float)(dstHeight - 1);
        float fy0, fy1;
        fy0 = floor(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        uint2 ycoord = hip_clamp_pixel_coordinates_to_border(fy, dstHeight - 1, srcImageStrideInBytes);
        pSrcImage += (ycoord.x * srcImageStrideInBytes);
        float frac;
        uint2 xcoord;
        uint xlimit = dstWidth - 1;

        uint2 dst;
        float4 f;

        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.x = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.y = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.z = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.w = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        dst.x = pack_(f);

        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.x = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.y = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.z = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floor(fx);
        f.w = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        dst.y = pack_(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ScaleImage_U8_U8_Bilinear_Replicate(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Replicate, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    float fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);

    if (fx >= 0.0f && fy >= 0.0f && fmaf(8.0f, scaleInfo.x, fx) < (dstWidth - 1) && fmaf(1.0f, scaleInfo.y, fy) < (dstHeight - 1)) {
        uint2 dst;
        float fx, fy, fint, frac, fy0, fy1;
        float4 f;
        fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
        fy0 = floor(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        pSrcImage += ((uint)fy * srcImageStrideInBytes);

        fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
        fint = floor(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.x = pack_(f);

        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floor(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.y = pack_(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
    else {
        float fy1 = fy - floor(fy);
        float fy0 = 1.0f - fy1;
        int sy = (int) floor(fy);
        float frac;
        uint2 dst;
        float4 f;
        frac = fx - floor(fx);

        f.x = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.y = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.z = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.w = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        dst.x = pack_(f);

        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.x = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.y = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.z = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floor(fx);
        f.w = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floor(fx), sy, dstWidth, dstHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        dst.y = pack_(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ScaleImage_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset, (uint) borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    pSrcImage += srcImageStrideInBytes * (uint)fmaf((float)y, scaleInfo.y, scaleInfo.w);
    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);

    dst.x  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 24;

    fx += scaleInfo.x;

    dst.y  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ScaleImage_U8_U8_Area(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpAffine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrixLoc) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_affine_matrix_t affineMatrix = *affineMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float sx, sy;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix.matrix[1][0], affineMatrix.matrix[2][0]);
    sx = fmaf(dx, affineMatrix.matrix[0][0], sx);
    sy = fmaf(dy, affineMatrix.matrix[1][1], affineMatrix.matrix[2][1]);
    sy = fmaf(dx, affineMatrix.matrix[0][1], sy);

    dst.x = pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)];
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 8;
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 16;
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 24;

    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];

    dst.y  = pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)];
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 8;
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 16;
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)sy, (uint)sx)] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrixLoc) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrixLoc);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrixLoc, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_affine_matrix_t affineMatrix = *affineMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float sx, sy;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix.matrix[1][0], affineMatrix.matrix[2][0]);
    sx = fmaf(dx, affineMatrix.matrix[0][0], sx);
    sy = fmaf(dy, affineMatrix.matrix[1][1], affineMatrix.matrix[2][1]);
    sy = fmaf(dx, affineMatrix.matrix[0][1], sy);

    x = (uint)(int)sx;
    y = (uint)(int)sy;
    dstWidth -= 2;
    dstHeight -= 2;

    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x = v;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= v << 8;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= v << 16;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= v << 24;

    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;

    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y = v;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= v << 8;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= v << 16;
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= v << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Nearest_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrixLoc, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrixLoc, (uint) borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrixLoc) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_affine_matrix_t affineMatrix = *affineMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix.matrix[1][0], affineMatrix.matrix[2][0]);
    sx = fmaf(dx, affineMatrix.matrix[0][0], sx);
    sy = fmaf(dy, affineMatrix.matrix[1][1], affineMatrix.matrix[2][1]);
    sy = fmaf(dx, affineMatrix.matrix[0][1], sy);

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    dst.x = pack_(f);

    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix.matrix[0][0];
    sy += affineMatrix.matrix[0][1];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx, sy);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrixLoc) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrixLoc);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrixLoc, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_affine_matrix_t affineMatrix = *affineMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix.matrix[1][0], affineMatrix.matrix[2][0]);
    sx = fmaf(dx, affineMatrix.matrix[0][0], sx);
    sy = fmaf(dy, affineMatrix.matrix[1][1], affineMatrix.matrix[2][1]);
    sy = fmaf(dx, affineMatrix.matrix[0][1], sy);

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    dst.x = pack_(f);

    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1];

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix.matrix[0][0]; sy += affineMatrix.matrix[0][1]; f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrixLoc, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrixLoc, (uint) borderValue);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpPerspective kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrixLoc) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_perspective_matrix_t perspectiveMatrix = *perspectiveMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float sx, sy, sz, isz;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix.matrix[1][0], perspectiveMatrix.matrix[2][0]);
    sx = fmaf(dx, perspectiveMatrix.matrix[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix.matrix[1][1], perspectiveMatrix.matrix[2][1]);
    sy = fmaf(dx, perspectiveMatrix.matrix[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix.matrix[1][2], perspectiveMatrix.matrix[2][2]);
    sz = fmaf(dx, perspectiveMatrix.matrix[0][2], sz);

    isz = 1.0f / sz;

    dst.x = pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))];
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 8;
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 16;
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.x |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 24;
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.y  = pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))];
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 8;
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 16;
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    dst.y |= pSrcImage[(int)fmaf(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz))] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrixLoc) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrixLoc);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrixLoc, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_perspective_matrix_t perspectiveMatrix = *perspectiveMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float sx, sy, sz, isz;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix.matrix[1][0], perspectiveMatrix.matrix[2][0]);
    sx = fmaf(dx, perspectiveMatrix.matrix[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix.matrix[1][1], perspectiveMatrix.matrix[2][1]);
    sy = fmaf(dx, perspectiveMatrix.matrix[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix.matrix[1][2], perspectiveMatrix.matrix[2][2]);
    sz = fmaf(dx, perspectiveMatrix.matrix[0][2], sz);

    isz = 1.0f / sz;

    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);

    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x = v;
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= (v << 8);
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= (v << 16);
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.x |= (v << 24);

    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);

    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y = v;
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= (v << 8);
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= (v << 16);
    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx * isz); y = (uint)(int)(sy * isz);
    mask = ((int)(x | (dstWidth - x) | y | (dstHeight - y))) >> 31; mask = ~mask;
    x &= mask; y &= mask; v = pSrcImage[(int)fmaf(srcImageStrideInBytes, y, x)]; v = HIPSELECT(borderValue, v, mask); dst.y |= (v << 24);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Nearest_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrixLoc, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrixLoc, (uint) borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrixLoc) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_perspective_matrix_t perspectiveMatrix = *perspectiveMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy, sz, isz;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix.matrix[1][0], perspectiveMatrix.matrix[2][0]);
    sx = fmaf(dx, perspectiveMatrix.matrix[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix.matrix[1][1], perspectiveMatrix.matrix[2][1]);
    sy = fmaf(dx, perspectiveMatrix.matrix[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix.matrix[1][2], perspectiveMatrix.matrix[2][2]);
    sz = fmaf(dx, perspectiveMatrix.matrix[0][2], sz);

    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    dst.x = pack_(f);

    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageStrideInBytes, sx * isz, sy * isz);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrixLoc) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrixLoc);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrixLoc, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    d_perspective_matrix_t perspectiveMatrix = *perspectiveMatrixLoc;

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy, sz, isz;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix.matrix[1][0], perspectiveMatrix.matrix[2][0]);
    sx = fmaf(dx, perspectiveMatrix.matrix[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix.matrix[1][1], perspectiveMatrix.matrix[2][1]);
    sy = fmaf(dx, perspectiveMatrix.matrix[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix.matrix[1][2], perspectiveMatrix.matrix[2][2]);
    sz = fmaf(dx, perspectiveMatrix.matrix[0][2], sz);

    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    dst.x = pack_(f);

    sx += perspectiveMatrix.matrix[0][0]; sy += perspectiveMatrix.matrix[0][1]; sz += perspectiveMatrix.matrix[0][2]; isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix.matrix[0][0];
    sy += perspectiveMatrix.matrix[0][1];
    sz += perspectiveMatrix.matrix[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx * isz, sy * isz, borderValue);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrixLoc, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrixLoc, (uint) borderValue);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxRemap kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *map, unsigned int mapStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    int xSrc = (int)PIXELROUNDF32(map[y * (dstWidth * 2) + (x*2) + 0]);
    int ySrc = (int)PIXELROUNDF32(map[y * (dstWidth * 2) + (x*2) + 1]);
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_Remap_U8_U8_Nearest(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    // Printing Passed Remap Table

    // printf("\n\n--------------------- Passed Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", mapStrideInBytes);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%d,%d  ", map[i].y, map[i].x);
    // }
    // printf("\n\n");

    // Generating New Remap Table
    _vx_coordinates2df_t Remap_remapTable_coordinates2df[dstWidth * dstHeight];
	vx_size Remap_remapTableStrideY_size = dstWidth * 8;

    for (int i = 0; i < dstHeight; i ++) {
		for (int j = 0; j < dstWidth; j++) {
			if ((j < srcWidth) && (i < srcHeight)) {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = j;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = i;
			}
			else {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = 0;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = 0;
			}
		}
	}

    // Printing Generated Remap Table

    // printf("\n\n--------------------- Generated Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", (vx_uint32)Remap_remapTableStrideY_size);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%0.1f,%0.1f  ", Remap_remapTable_coordinates2df[i].y, Remap_remapTable_coordinates2df[i].x);
    // }
    // printf("\n\n");

    float *remapTable_float = (float*) Remap_remapTable_coordinates2df;

    /*printf("\n\n");
    for (int i = 0; i < (dstWidth * dstHeight * 2); i+=2)
    {
        printf("%0.1f,%0.1f  ", remapTable_float[i+1], remapTable_float[i]);
    }
    printf("\n\n");*/

    vx_uint32 bufferSize = dstWidth * dstHeight * 64;
    vx_uint8 *hipRemapTable_float;
    hipMalloc(&hipRemapTable_float, bufferSize);
    hipMemcpy(hipRemapTable_float, remapTable_float, bufferSize, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Nearest,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)hipRemapTable_float, mapStrideInBytes);
    hipFree(&hipRemapTable_float);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *map, unsigned int mapStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    float xSrcFloat = map[y * (dstWidth * 2) + (x*2) + 0];
    float ySrcFloat = map[y * (dstWidth * 2) + (x*2) + 1];
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] +
        (s) * (1-t) * pSrcImage[srcIdxTopRight] +
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] +
        (s) * (t) * pSrcImage[srcIdxBottomRight]
        );
    }
}
int HipExec_Remap_U8_U8_Bilinear(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    // Printing Passed Remap Table

    // printf("\n\n--------------------- Passed Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", mapStrideInBytes);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%d,%d  ", map[i].y, map[i].x);
    // }
    // printf("\n\n");

    // Generating New Remap Table

    _vx_coordinates2df_t Remap_remapTable_coordinates2df[dstWidth * dstHeight];
	vx_size Remap_remapTableStrideY_size = dstWidth * 8;

    for (int i = 0; i < dstHeight; i ++) {
		for (int j = 0; j < dstWidth; j++) {
			if ((j < srcWidth) && (i < srcHeight)) {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = j;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = i;
			}
			else {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = 0;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = 0;
			}
		}
	}

    // Printing Generated Remap Table

    // printf("\n\n--------------------- Generated Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", (vx_uint32)Remap_remapTableStrideY_size);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%0.1f,%0.1f  ", Remap_remapTable_coordinates2df[i].y, Remap_remapTable_coordinates2df[i].x);
    // }
    // printf("\n\n");

    float *remapTable_float = (float*) Remap_remapTable_coordinates2df;

    /*printf("\n\n");
    for (int i = 0; i < (dstWidth * dstHeight * 2); i+=2)
    {
        printf("%0.1f,%0.1f  ", remapTable_float[i+1], remapTable_float[i]);
    }
    printf("\n\n");*/

    vx_uint32 bufferSize = dstWidth * dstHeight * 64;
    vx_uint8 *hipRemapTable_float;
    hipMalloc(&hipRemapTable_float, bufferSize);
    hipMemcpy(hipRemapTable_float, remapTable_float, bufferSize, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Bilinear,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)hipRemapTable_float, mapStrideInBytes);
    hipFree(&hipRemapTable_float);
    return VX_SUCCESS;
}