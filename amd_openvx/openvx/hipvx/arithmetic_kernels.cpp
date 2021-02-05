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

#include "hip_kernels.h"

#define PIXELSATURATEU8(pixel)  (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
#define PIXELROUNDF32(value)    ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))

#define atan2_p0    (0.273*0.3183098862f)
#define atan2_p1    (0.9997878412794807f*57.29577951308232f)
#define atan2_p3    (-0.3258083974640975f*57.29577951308232f)
#define atan2_p5    (0.1555786518463281f*57.29577951308232f)
#define atan2_p7    (-0.04432655554792128f*57.29577951308232f)
#define DBL_EPSILON __DBL_EPSILON__

__device__ __forceinline__ float4 uchars_to_float4(uint src) {
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ float4 s16s_to_float4_grouped(int src1, int src2) {
    return make_float4((float)(src1&0xFFFF), (float)((src1&0xFFFF0000)>>16), (float)(src2&0xFFFF), (float)((src2&0xFFFF0000)>>16));
}

__device__ __forceinline__ float4 s16s_to_float4_ungrouped(const short int *src, unsigned int srcIdx) {
    short4 srcs4 = *((short4 *)(&src[srcIdx]));
    return make_float4((float)srcs4.x, (float)srcs4.y, (float)srcs4.z, (float)srcs4.w);
}

__device__ __forceinline__ uint float4_to_uchars(float4 src) {
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ void float4_to_s16s(short int *dst_s16s, unsigned int dstIdx, float4 dst_float4) {
    *((short4 *)(&dst_s16s[dstIdx])) = make_short4(dst_float4.x, dst_float4.y, dst_float4.z, dst_float4.w);
}

__device__ __forceinline__ float4 generic_mod_float4(float4 src, int b) {
    src.x = (float) ((int)src.x % b < 0 ? (int)src.x % b + b : (int)src.x % b);
    src.y = (float) ((int)src.y % b < 0 ? (int)src.y % b + b : (int)src.y % b);
    src.z = (float) ((int)src.z % b < 0 ? (int)src.z % b + b : (int)src.z % b);
    src.w = (float) ((int)src.w % b < 0 ? (int)src.w % b + b : (int)src.w % b);
    return src;
}

__device__ float Norm_Atan2_deg (float Gx, float Gy) {
    float scale = (float)128 / 180.f;
    vx_uint16 ax, ay;
    ax = fabsf(Gx), ay = fabsf(Gy);
    float a, c, c2;
    if (ax >= ay) {
        c = (float)ay / ((float)ax + (float)DBL_EPSILON);
        c2 = c*c;
        a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    else {
        c = (float)ax / ((float)ay + (float)DBL_EPSILON);
        c2 = c*c;
        a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    if (Gx < 0)
    a = 180.f - a;
    if (Gy < 0)
    a = 360.f - a;

    // normalize and copy to dst
    float arct_norm = (a*scale + 0.5);
    return arct_norm;
}
// ----------------------------------------------------------------------------
// VxAbsDiff kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_U8_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((fabs4(unpack_(src1.x) - unpack_(src2.x))));
    dst.y = pack_((fabs4(unpack_(src1.y) - unpack_(src2.y))));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_AbsDiff_U8_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_U8_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint  dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx = y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = min(abs(((src1.x << 16) >> 16) - ((src2.x << 16) >> 16)), 32767u);
    dst.x |= (int)min(abs((src1.x    >> 16) - ( src2.x        >> 16)), 32767u) << 16;
    dst.y  = min(abs(((src1.y << 16) >> 16) - ((src2.y << 16) >> 16)), 32767u);
    dst.y |= (int) min(abs((src1.y   >> 16) - ( src2.y        >> 16)), 32767u) << 16;
    dst.z  = min(abs(((src1.z << 16) >> 16) - ((src2.z << 16) >> 16)), 32767u);
    dst.z |= (int)min(abs(( src1.z   >> 16) - ( src2.z        >> 16)), 32767u) << 16;
    dst.w  = min(abs(((src1.w << 16) >> 16) - ((src2.w << 16) >> 16)), 32767u);
    dst.w |= (int)min(abs(( src1.w   >> 16) - ( src2.w        >> 16)), 32767u) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_AbsDiff_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                    (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar*)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxAdd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = (src1.x +  src2.x              ) & 0x000000ff;
    dst.x |= (src1.x + (src2.x & 0x0000ff00)) & 0x0000ff00;
    dst.x |= (src1.x + (src2.x & 0x00ff0000)) & 0x00ff0000;
    dst.x |= (src1.x + (src2.x & 0xff000000)) & 0xff000000;
    dst.y  = (src1.y +  src2.y              ) & 0x000000ff;
    dst.y |= (src1.y + (src2.y & 0x0000ff00)) & 0x0000ff00;
    dst.y |= (src1.y + (src2.y & 0x00ff0000)) & 0x00ff0000;
    dst.y |= (src1.y + (src2.y & 0xff000000)) & 0xff000000;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_U8_U8U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((unpack_(src1.x) + unpack_(src2.x)));
    dst.y = pack_((unpack_(src1.y) + unpack_(src2.y)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_U8_U8U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((src1.x & 0x000000ff) + (src2.x & 0x000000ff));
    dst.x |= ((src1.x & 0x0000ff00) + (src2.x & 0x0000ff00)) <<  8;
    dst.y  = ((src1.x & 0x00ff0000) + (src2.x & 0x00ff0000)) >> 16;
    dst.y |= ((src1.x >>        24) + (src2.x >>        24)) << 16;
    dst.z  = ((src1.y & 0x000000ff) + (src2.y & 0x000000ff));
    dst.z |= ((src1.y & 0x0000ff00) + (src2.y & 0x0000ff00)) <<  8;
    dst.w  = ((src1.y & 0x00ff0000) + (src2.y & 0x00ff0000)) >> 16;
    dst.w |= ((src1.y >>        24) + (src2.y >>        24)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((((int)(src1.x) << 16) >> 16) + ( src2.x        & 0x000000ff)) & 0x0000ffff;
    dst.x |= ((       src1.x  & 0xffff0000) + ((src2.x <<  8) & 0x00ff0000));
    dst.y  = ((((int)(src1.y) << 16) >> 16) + ((src2.x >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.y |= ((       src1.y  & 0xffff0000) + ((src2.x >>  8) & 0x00ff0000));
    dst.z  = ((((int)(src1.z) << 16) >> 16) + ( src2.y        & 0x000000ff)) & 0x0000ffff;
    dst.z |= ((       src1.z  & 0xffff0000) + ((src2.y <<  8) & 0x00ff0000));
    dst.w  = ((((int)(src1.w) << 16) >> 16) + ((src2.y >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.w |= ((       src1.w  & 0xffff0000) + ((src2.y >>  8) & 0x00ff0000));

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (int)(hip_clamp((float)(((int)(src1.x) << 16) >> 16) + unpackedSrc2X.x, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.x |= (int)(hip_clamp((float)( (int)(src1.x)        >> 16) + unpackedSrc2X.y, -32768.0f, 32767.0f)) << 16;
    dst.y  = (int)(hip_clamp((float)(((int)(src1.y) << 16) >> 16) + unpackedSrc2X.z, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.y |= (int)(hip_clamp((float)( (int)(src1.y)        >> 16) + unpackedSrc2X.w, -32768.0f, 32767.0f)) << 16;
    dst.z  = (int)(hip_clamp((float)(((int)(src1.z) << 16) >> 16) + unpackedSrc2Y.x, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.z |= (int)(hip_clamp((float)( (int)(src1.z)        >> 16) + unpackedSrc2Y.y, -32768.0f, 32767.0f)) << 16;
    dst.w  = (int)(hip_clamp((float)(((int)(src1.w) << 16) >> 16) + unpackedSrc2Y.z, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.w |= (int)(hip_clamp((float)( (int)(src1.w)        >> 16) + unpackedSrc2Y.w, -32768.0f, 32767.0f)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (src1.x +  src2.x              ) & 0x0000ffff;
    dst.x |= (src1.x + (src2.x & 0xffff0000)) & 0xffff0000;
    dst.y  = (src1.y +  src2.y              ) & 0x0000ffff;
    dst.y |= (src1.y + (src2.y & 0xffff0000)) & 0xffff0000;
    dst.z  = (src1.z +  src2.z              ) & 0x0000ffff;
    dst.z |= (src1.z + (src2.z & 0xffff0000)) & 0xffff0000;
    dst.w  = (src1.w +  src2.w              ) & 0x0000ffff;
    dst.w |= (src1.w + (src2.w & 0xffff0000)) & 0xffff0000;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = hip_clamp((((int)(src1.x) << 16) >> 16) + (((int)(src2.x) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.x |= hip_clamp(( (int)(src1.x)        >> 16) + ( (int)(src2.x)        >> 16), -32768, 32767) << 16;
    dst.y  = hip_clamp((((int)(src1.y) << 16) >> 16) + (((int)(src2.y) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.y |= hip_clamp(( (int)(src1.y)        >> 16) + ( (int)(src2.y)        >> 16), -32768, 32767) << 16;
    dst.z  = hip_clamp((((int)(src1.z) << 16) >> 16) + (((int)(src2.z) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.z |= hip_clamp(( (int)(src1.z)        >> 16) + ( (int)(src2.z)        >> 16), -32768, 32767) << 16;
    dst.w  = hip_clamp((((int)(src1.w) << 16) >> 16) + (((int)(src2.w) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.w |= hip_clamp(( (int)(src1.w)        >> 16) + ( (int)(src2.w)        >> 16), -32768, 32767) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxSub kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = (src1.x -  src2.x              ) & 0x000000ff;
    dst.x |= (src1.x - (src2.x & 0x0000ff00)) & 0x0000ff00;
    dst.x |= (src1.x - (src2.x & 0x00ff0000)) & 0x00ff0000;
    dst.x |= (src1.x - (src2.x & 0xff000000)) & 0xff000000;
    dst.y  = (src1.y -  src2.y              ) & 0x000000ff;
    dst.y |= (src1.y - (src2.y & 0x0000ff00)) & 0x0000ff00;
    dst.y |= (src1.y - (src2.y & 0x00ff0000)) & 0x00ff0000;
    dst.y |= (src1.y - (src2.y & 0xff000000)) & 0xff000000;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_U8_U8U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((unpack_(src1.x) - unpack_(src2.x)));
    dst.y = pack_((unpack_(src1.y) - unpack_(src2.y)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_U8_U8U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((src1.x & 0x000000ff) - (src2.x & 0x000000ff));
    dst.x |= ((src1.x & 0x0000ff00) - (src2.x & 0x0000ff00)) <<  8;
    dst.y  = ((src1.x & 0x00ff0000) - (src2.x & 0x00ff0000)) >> 16;
    dst.y |= ((src1.x >>        24) - (src2.x >>        24)) << 16;
    dst.z  = ((src1.y & 0x000000ff) - (src2.y & 0x000000ff));
    dst.z |= ((src1.y & 0x0000ff00) - (src2.y & 0x0000ff00)) <<  8;
    dst.w  = ((src1.y & 0x00ff0000) - (src2.y & 0x00ff0000)) >> 16;
    dst.w |= ((src1.y >>        24) - (src2.y >>        24)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((((int)(src1.x) << 16) >> 16) - ( src2.x        & 0x000000ff)) & 0x0000ffff;
    dst.x |= ((       src1.x  & 0xffff0000) - ((src2.x <<  8) & 0x00ff0000));
    dst.y  = ((((int)(src1.y) << 16) >> 16) - ((src2.x >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.y |= ((       src1.y  & 0xffff0000) - ((src2.x >>  8) & 0x00ff0000));
    dst.z  = ((((int)(src1.z) << 16) >> 16) - ( src2.y        & 0x000000ff)) & 0x0000ffff;
    dst.z |= ((       src1.z  & 0xffff0000) - ((src2.y <<  8) & 0x00ff0000));
    dst.w  = ((((int)(src1.w) << 16) >> 16) - ((src2.y >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.w |= ((       src1.w  & 0xffff0000) - ((src2.y >>  8) & 0x00ff0000));

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (int)(hip_clamp((float)(((int)(src1.x) << 16) >> 16) - unpackedSrc2X.x, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.x |= (int)(hip_clamp((float)( (int)(src1.x)        >> 16) - unpackedSrc2X.y, -32768.0f, 32767.0f)) << 16;
    dst.y  = (int)(hip_clamp((float)(((int)(src1.y) << 16) >> 16) - unpackedSrc2X.z, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.y |= (int)(hip_clamp((float)( (int)(src1.y)        >> 16) - unpackedSrc2X.w, -32768.0f, 32767.0f)) << 16;
    dst.z  = (int)(hip_clamp((float)(((int)(src1.z) << 16) >> 16) - unpackedSrc2Y.x, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.z |= (int)(hip_clamp((float)( (int)(src1.z)        >> 16) - unpackedSrc2Y.y, -32768.0f, 32767.0f)) << 16;
    dst.w  = (int)(hip_clamp((float)(((int)(src1.w) << 16) >> 16) - unpackedSrc2Y.z, -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.w |= (int)(hip_clamp((float)( (int)(src1.w)        >> 16) - unpackedSrc2Y.w, -32768.0f, 32767.0f)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (src1.x -  src2.x              ) & 0x0000ffff;
    dst.x |= (src1.x - (src2.x & 0xffff0000)) & 0xffff0000;
    dst.y  = (src1.y -  src2.y              ) & 0x0000ffff;
    dst.y |= (src1.y - (src2.y & 0xffff0000)) & 0xffff0000;
    dst.z  = (src1.z -  src2.z              ) & 0x0000ffff;
    dst.z |= (src1.z - (src2.z & 0xffff0000)) & 0xffff0000;
    dst.w  = (src1.w -  src2.w              ) & 0x0000ffff;
    dst.w |= (src1.w - (src2.w & 0xffff0000)) & 0xffff0000;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = hip_clamp((((int)(src1.x) << 16) >> 16) - (((int)(src2.x) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.x |= hip_clamp(( (int)(src1.x)        >> 16) - ( (int)(src2.x)        >> 16), -32768, 32767) << 16;
    dst.y  = hip_clamp((((int)(src1.y) << 16) >> 16) - (((int)(src2.y) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.y |= hip_clamp(( (int)(src1.y)        >> 16) - ( (int)(src2.y)        >> 16), -32768, 32767) << 16;
    dst.z  = hip_clamp((((int)(src1.z) << 16) >> 16) - (((int)(src2.z) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.z |= hip_clamp(( (int)(src1.z)        >> 16) - ( (int)(src2.z)        >> 16), -32768, 32767) << 16;
    dst.w  = hip_clamp((((int)(src1.w) << 16) >> 16) - (((int)(src2.w) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.w |= hip_clamp(( (int)(src1.w)        >> 16) - ( (int)(src2.w)        >> 16), -32768, 32767) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMul kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = ((int)(scale * unpackedSrc1X.x * unpackedSrc2X.x) & 0x000000ff)      ;
    dst.x |= ((int)(scale * unpackedSrc1X.y * unpackedSrc2X.y) & 0x000000ff) <<  8;
    dst.x |= ((int)(scale * unpackedSrc1X.z * unpackedSrc2X.z) & 0x000000ff) << 16;
    dst.x |= ((int)(scale * unpackedSrc1X.w * unpackedSrc2X.w)             ) << 24;
    dst.y  = ((int)(scale * unpackedSrc1Y.x * unpackedSrc2Y.x) & 0x000000ff)      ;
    dst.y |= ((int)(scale * unpackedSrc1Y.y * unpackedSrc2Y.y) & 0x000000ff) <<  8;
    dst.y |= ((int)(scale * unpackedSrc1Y.z * unpackedSrc2Y.z) & 0x000000ff) << 16;
    dst.y |= ((int)(scale * unpackedSrc1Y.w * unpackedSrc2Y.w)             ) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = ((int)(scale * unpackedSrc1X.x * unpackedSrc2X.x + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;
    dst.x |= ((int)(scale * unpackedSrc1X.y * unpackedSrc2X.y + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;
    dst.x |= ((int)(scale * unpackedSrc1X.z * unpackedSrc2X.z + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;
    dst.x |= ((int)(scale * unpackedSrc1X.w * unpackedSrc2X.w + (0.5f - 0.00006103515625f))             ) << 24;
    dst.y  = ((int)(scale * unpackedSrc1Y.x * unpackedSrc2Y.x + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;
    dst.y |= ((int)(scale * unpackedSrc1Y.y * unpackedSrc2Y.y + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;
    dst.y |= ((int)(scale * unpackedSrc1Y.z * unpackedSrc2Y.z + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;
    dst.y |= ((int)(scale * unpackedSrc1Y.w * unpackedSrc2Y.w + (0.5f - 0.00006103515625f))             ) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    float4 f;
    f.x = scale * unpackedSrc1X.x * unpackedSrc2X.x - (0.5f - 0.00006103515625f);
    f.y = scale * unpackedSrc1X.y * unpackedSrc2X.y - (0.5f - 0.00006103515625f);
    f.z = scale * unpackedSrc1X.z * unpackedSrc2X.z - (0.5f - 0.00006103515625f);
    f.w = scale * unpackedSrc1X.w * unpackedSrc2X.w - (0.5f - 0.00006103515625f);
    dst.x = pack_(f);
    f.x = scale * unpackedSrc1Y.x * unpackedSrc2Y.x - (0.5f - 0.00006103515625f);
    f.y = scale * unpackedSrc1Y.y * unpackedSrc2Y.y - (0.5f - 0.00006103515625f);
    f.z = scale * unpackedSrc1Y.z * unpackedSrc2Y.z - (0.5f - 0.00006103515625f);
    f.w = scale * unpackedSrc1Y.w * unpackedSrc2Y.w - (0.5f - 0.00006103515625f);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    float4 f;
    f.x = scale * unpackedSrc1X.x * unpackedSrc2X.x;
    f.y = scale * unpackedSrc1X.y * unpackedSrc2X.y;
    f.z = scale * unpackedSrc1X.z * unpackedSrc2X.z;
    f.w = scale * unpackedSrc1X.w * unpackedSrc2X.w;
    dst.x = pack_(f);
    f.x = scale * unpackedSrc1Y.x * unpackedSrc2Y.x;
    f.y = scale * unpackedSrc1Y.y * unpackedSrc2Y.y;
    f.z = scale * unpackedSrc1Y.z * unpackedSrc2Y.z;
    f.w = scale * unpackedSrc1Y.w * unpackedSrc2Y.w;
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)(scale * unpackedSrc1X.x * unpackedSrc2X.x)) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * unpackedSrc1X.y * unpackedSrc2X.y))             ) << 16;
    dst.y  = (((int)(scale * unpackedSrc1X.z * unpackedSrc2X.z)) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * unpackedSrc1X.w * unpackedSrc2X.w))             ) << 16;
    dst.z  = (((int)(scale * unpackedSrc1Y.x * unpackedSrc2Y.x)) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * unpackedSrc1Y.y * unpackedSrc2Y.y))             ) << 16;
    dst.w  = (((int)(scale * unpackedSrc1Y.z * unpackedSrc2Y.z)) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * unpackedSrc1Y.w * unpackedSrc2Y.w))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)(scale * unpackedSrc1X.x * unpackedSrc2X.x + 0.5f)) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * unpackedSrc1X.y * unpackedSrc2X.y + 0.5f))             ) << 16;
    dst.y  = (((int)(scale * unpackedSrc1X.z * unpackedSrc2X.z + 0.5f)) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * unpackedSrc1X.w * unpackedSrc2X.w + 0.5f))             ) << 16;
    dst.z  = (((int)(scale * unpackedSrc1Y.x * unpackedSrc2Y.x + 0.5f)) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * unpackedSrc1Y.y * unpackedSrc2Y.y + 0.5f))             ) << 16;
    dst.w  = (((int)(scale * unpackedSrc1Y.z * unpackedSrc2Y.z + 0.5f)) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * unpackedSrc1Y.w * unpackedSrc2Y.w + 0.5f))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)(hip_clamp(scale * unpackedSrc1X.x * unpackedSrc2X.x, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_clamp(scale * unpackedSrc1X.y * unpackedSrc2X.y, -32768.0f, 32767.0f)))             ) << 16;
    dst.y  = (((int)(hip_clamp(scale * unpackedSrc1X.z * unpackedSrc2X.z, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_clamp(scale * unpackedSrc1X.w * unpackedSrc2X.w, -32768.0f, 32767.0f)))             ) << 16;
    dst.z  = (((int)(hip_clamp(scale * unpackedSrc1Y.x * unpackedSrc2Y.x, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_clamp(scale * unpackedSrc1Y.y * unpackedSrc2Y.y, -32768.0f, 32767.0f)))             ) << 16;
    dst.w  = (((int)(hip_clamp(scale * unpackedSrc1Y.z * unpackedSrc2Y.z, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_clamp(scale * unpackedSrc1Y.w * unpackedSrc2Y.w, -32768.0f, 32767.0f)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc1X = unpack_(src1.x);
    float4 unpackedSrc1Y = unpack_(src1.y);
    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)(hip_clamp(scale * unpackedSrc1X.x * unpackedSrc2X.x + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_clamp(scale * unpackedSrc1X.y * unpackedSrc2X.y + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.y  = (((int)(hip_clamp(scale * unpackedSrc1X.z * unpackedSrc2X.z + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_clamp(scale * unpackedSrc1X.w * unpackedSrc2X.w + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.z  = (((int)(hip_clamp(scale * unpackedSrc1Y.x * unpackedSrc2Y.x + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_clamp(scale * unpackedSrc1Y.y * unpackedSrc2Y.y + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.w  = (((int)(hip_clamp(scale * unpackedSrc1Y.z * unpackedSrc2Y.z + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_clamp(scale * unpackedSrc1Y.w * unpackedSrc2Y.w + 0.5f, -32768.0f, 32767.0f)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)(scale * (float)(((int)(src1.x) << 16) >> 16) * unpackedSrc2X.x)) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * (float)( (int)(src1.x)        >> 16) * unpackedSrc2X.y))             ) << 16;
    dst.y  = (((int)(scale * (float)(((int)(src1.y) << 16) >> 16) * unpackedSrc2X.z)) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * (float)( (int)(src1.y)        >> 16) * unpackedSrc2X.w))             ) << 16;
    dst.z  = (((int)(scale * (float)(((int)(src1.z) << 16) >> 16) * unpackedSrc2Y.x)) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * (float)( (int)(src1.z)        >> 16) * unpackedSrc2Y.y))             ) << 16;
    dst.w  = (((int)(scale * (float)(((int)(src1.w) << 16) >> 16) * unpackedSrc2Y.z)) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * (float)( (int)(src1.w)        >> 16) * unpackedSrc2Y.w))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.x) << 16) >> 16) * unpackedSrc2X.x)) & 0x0000ffff)      ;
    dst.x |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.x)        >> 16) * unpackedSrc2X.y))             ) << 16;
    dst.y  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.y) << 16) >> 16) * unpackedSrc2X.z)) & 0x0000ffff)      ;
    dst.y |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.y)        >> 16) * unpackedSrc2X.w))             ) << 16;
    dst.z  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.z) << 16) >> 16) * unpackedSrc2Y.x)) & 0x0000ffff)      ;
    dst.z |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.z)        >> 16) * unpackedSrc2Y.y))             ) << 16;
    dst.w  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.w) << 16) >> 16) * unpackedSrc2Y.z)) & 0x0000ffff)      ;
    dst.w |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.w)        >> 16) * unpackedSrc2Y.w))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    float f;
    f = hip_clamp(scale * (float)(((int)(src1.x) << 16) >> 16) * unpackedSrc2X.x, -32768.0f, 32767.0f); dst.x  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.x)        >> 16) * unpackedSrc2X.y, -32768.0f, 32767.0f); dst.x |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.y) << 16) >> 16) * unpackedSrc2X.z, -32768.0f, 32767.0f); dst.y  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.y)        >> 16) * unpackedSrc2X.w, -32768.0f, 32767.0f); dst.y |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.z) << 16) >> 16) * unpackedSrc2Y.x, -32768.0f, 32767.0f); dst.z  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.z)        >> 16) * unpackedSrc2Y.y, -32768.0f, 32767.0f); dst.z |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.w) << 16) >> 16) * unpackedSrc2Y.z, -32768.0f, 32767.0f); dst.w  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.w)        >> 16) * unpackedSrc2Y.w, -32768.0f, 32767.0f); dst.w |= ((int)(f)             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float4 unpackedSrc2X = unpack_(src2.x);
    float4 unpackedSrc2Y = unpack_(src2.y);
    dst.x =  (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.x) << 16) >> 16) * unpackedSrc2X.x))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.x)  >> 16)	     * unpackedSrc2X.y)))             ) << 16;
    dst.y  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.y) << 16) >> 16) * unpackedSrc2X.z))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.y)  >> 16)        * unpackedSrc2X.w))))              << 16;
    dst.z  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.z) << 16) >> 16) * unpackedSrc2Y.x))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.z)  >> 16)        * unpackedSrc2Y.y))))              << 16;
    dst.w  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.w) << 16) >> 16) * unpackedSrc2Y.z))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.w)  >> 16)        * unpackedSrc2Y.w))))              << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((int)(scale * (double)((((int)(src1.x)) << 16) >> 16) * (double)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff;
    dst.x |= ((int)(scale * (double)(( (int)(src1.x)) >> 16)        * (double)(( (int)(src2.x)) >> 16)))		  << 16;
    dst.y  = ((int)(scale * (double)((((int)(src1.y)) << 16) >> 16) * (double)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff;
    dst.y |= ((int)(scale * (double)(( (int)(src1.y)) >> 16)        * (double)(( (int)(src2.y)) >> 16)))		  << 16;
    dst.z  = ((int)(scale * (double)((((int)(src1.z)) << 16) >> 16) * (double)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff;
    dst.z |= ((int)(scale * (double)(( (int)(src1.z)) >> 16)        * (double)(( (int)(src2.z)) >> 16)))		  << 16;
    dst.w  = ((int)(scale * (double)((((int)(src1.w)) << 16) >> 16) * (double)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff;
    dst.w |= ((int)(scale * (double)(( (int)(src1.w)) >> 16)        * (double)(( (int)(src2.w)) >> 16)))		  << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff;
    dst.x |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.x)) >> 16)        * (float)(( (int)(src2.x)) >> 16)))		  << 16;
    dst.y  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff;
    dst.y |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.y)) >> 16)        * (float)(( (int)(src2.y)) >> 16)))		  << 16;
    dst.z  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff;
    dst.z |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.z)) >> 16)        * (float)(( (int)(src2.z)) >> 16)))		  << 16;
    dst.w  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff;
    dst.w |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.w)) >> 16)        * (float)(( (int)(src2.w)) >> 16)))		  << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)hip_clamp((scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.x |= (((int)hip_clamp((scale * (float)(( (int)(src1.x))        >> 16) * (float)(( (int)(src2.x))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.y  = (((int)hip_clamp((scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.y |= (((int)hip_clamp((scale * (float)(( (int)(src1.y))        >> 16) * (float)(( (int)(src2.y))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.z  = (((int)hip_clamp((scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.z |= (((int)hip_clamp((scale * (float)(( (int)(src1.z))        >> 16) * (float)(( (int)(src2.z))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.w  = (((int)hip_clamp((scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.w |= (((int)hip_clamp((scale * (float)(( (int)(src1.w))        >> 16) * (float)(( (int)(src2.w))        >> 16)), -32768.0f, 32767.0f))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.x |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.x))        >> 16) * (float)(( (int)(src2.x))        >> 16)))             ) << 16;
    dst.y  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.y |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.y))        >> 16) * (float)(( (int)(src2.y))        >> 16)))             ) << 16;
    dst.z  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.z |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.z))        >> 16) * (float)(( (int)(src2.z))        >> 16)))             ) << 16;
    dst.w  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.w |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.w))        >> 16) * (float)(( (int)(src2.w))        >> 16)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWeightedAverage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WeightedAverage_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float alpha, float invAlpha
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src2.x*invAlpha+src1.x*alpha, src2.y*invAlpha+src1.y*alpha, src2.z*invAlpha+src1.z*alpha, src2.w*invAlpha+src1.w*alpha);
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_WeightedAverage_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    vx_float32 alpha
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;
    vx_float32 invAlpha = (vx_float32)1 - alpha;

    hipLaunchKernelGGL(Hip_WeightedAverage_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, alpha, invAlpha);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMagnitude kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Magnitude_S16_S16S16(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    // To Modify
    // float2 f;
    // f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3 |= (uint)(f.s0) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Magnitude_S16_S16S16(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Magnitude_S16_S16S16, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxPhase kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Phase_U8_S16S16(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    // To Modify
    // float2 f; float4 p4;
    // f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;
    // f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;
    // p4 = select(p4, (float4) 0.0f, p4 > 255.5f);
    // r.s0 = amd_pack(p4);
    // f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;
    // f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;
    // p4 = select(p4, (float4) 0.0f, p4 > 255.5f);
    // r.s1 = amd_pack(p4);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Phase_U8_S16S16(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Phase_U8_S16S16, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}