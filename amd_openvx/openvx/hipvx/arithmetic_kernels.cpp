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



#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
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
Hip_AbsDiff_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;

    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(fabsf(src1.x-src2.x), fabsf(src1.y-src2.y), fabsf(src1.z-src2.z), fabsf(src1.w-src2.w));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_AbsDiff_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_S16_S16S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int  dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x*4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x*4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(fabsf(src1.x-src2.x)), PIXELSATURATES16(fabsf(src1.y-src2.y)), PIXELSATURATES16(fabsf(src1.z-src2.z)), PIXELSATURATES16(fabsf(src1.w-src2.w)));

    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_AbsDiff_S16_S16S16_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_S16_S16S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxAdd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Add_U8_U8U8_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATEU8(src1.x+src2.x), PIXELSATURATEU8(src1.y+src2.y), PIXELSATURATEU8(src1.z+src2.z), PIXELSATURATEU8(src1.w+src2.w));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Add_U8_U8U8_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Add_S16_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Add_S16_S16U8_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(src1.x+src2.x), PIXELSATURATES16(src1.y+src2.y), PIXELSATURATES16(src1.z+src2.z), PIXELSATURATES16(src1.w+src2.w));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Add_S16_S16U8_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int  dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x*4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x*4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4((src1.x+src2.x), (src1.y+src2.y), (src1.z+src2.z), (src1.w+src2.w));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Add_S16_S16S16_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int  dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x*4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x*4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(src1.x+src2.x), PIXELSATURATES16(src1.y+src2.y), PIXELSATURATES16(src1.z+src2.z), PIXELSATURATES16(src1.w+src2.w));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Add_S16_S16S16_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxSub kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    dst = generic_mod_float4(dst, 256);
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Sub_U8_U8U8_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATEU8(src1.x-src2.x), PIXELSATURATEU8(src1.y-src2.y), PIXELSATURATEU8(src1.z-src2.z), PIXELSATURATEU8(src1.w-src2.w));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Sub_U8_U8U8_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2 ,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    
    hipLaunchKernelGGL(Hip_Sub_S16_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_S16U8_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(src1.x-src2.x), PIXELSATURATES16(src1.y-src2.y), PIXELSATURATES16(src1.z-src2.z), PIXELSATURATES16(src1.w-src2.w));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_S16U8_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_U8S16_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8S16_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(src1.x-src2.x), PIXELSATURATES16(src1.y-src2.y), PIXELSATURATES16(src1.z-src2.z), PIXELSATURATES16(src1.w-src2.w)); //doesnt work for neg numbers
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_U8S16_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_S16S16_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes,
                    (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(src1.x-src2.x), PIXELSATURATES16(src1.y-src2.y), PIXELSATURATES16(src1.z-src2.z), PIXELSATURATES16(src1.w-src2.w)); //doesnt work for neg numbers
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Sub_S16_S16S16_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes,
                    (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMul kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x*src2.x*scale, src1.y*src2.y*scale, src1.z*src2.z*scale, src1.w*src2.w*scale);
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Mul_U8_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELROUNDF32(src1.x*src2.x*scale), PIXELROUNDF32(src1.y*src2.y*scale), PIXELROUNDF32(src1.z*src2.z*scale), PIXELROUNDF32(src1.w*src2.w*scale));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Mul_U8_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATEU8(src1.x*src2.x*scale), PIXELSATURATEU8(src1.y*src2.y*scale), PIXELSATURATEU8(src1.z*src2.z*scale), PIXELSATURATEU8(src1.w*src2.w*scale));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Mul_U8_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATEU8(PIXELROUNDF32(src1.x*src2.x*scale)), PIXELSATURATEU8(PIXELROUNDF32(src1.y*src2.y*scale)), PIXELSATURATEU8(PIXELROUNDF32(src1.z*src2.z*scale)), PIXELSATURATEU8(PIXELROUNDF32(src1.w*src2.w*scale)));
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_Mul_U8_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x*src2.x*scale, src1.y*src2.y*scale, src1.z*src2.z*scale, src1.w*src2.w*scale);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELROUNDF32(src1.x*src2.x*scale), PIXELROUNDF32(src1.y*src2.y*scale), PIXELROUNDF32(src1.z*src2.z*scale), PIXELROUNDF32(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(src1.x*src2.x*scale), PIXELSATURATES16(src1.y*src2.y*scale), PIXELSATURATES16(src1.z*src2.z*scale), PIXELSATURATES16(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(PIXELROUNDF32(src1.x*src2.x*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.y*src2.y*scale)),
        PIXELSATURATES16(PIXELROUNDF32(src1.z*src2.z*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.w*src2.w*scale)));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(src1.x*src2.x*scale, src1.y*src2.y*scale, src1.z*src2.z*scale, src1.w*src2.w*scale);
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16S16_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELROUNDF32(src1.x*src2.x*scale), PIXELROUNDF32(src1.y*src2.y*scale), PIXELROUNDF32(src1.z*src2.z*scale), PIXELROUNDF32(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16S16_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(src1.x*src2.x*scale), PIXELSATURATES16(src1.y*src2.y*scale), PIXELSATURATES16(src1.z*src2.z*scale), PIXELSATURATES16(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16S16_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x * 4);
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELSATURATES16(PIXELROUNDF32(src1.x*src2.x*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.y*src2.y*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.z*src2.z*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.w*src2.w*scale)));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16S16_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 =  s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4((src1.x*src2.x*scale), (src1.y*src2.y*scale), (src1.z*src2.z*scale),(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);

    float4 dst = make_float4(PIXELROUNDF32(src1.x*src2.x*scale), PIXELROUNDF32(src1.y*src2.y*scale), PIXELROUNDF32(src1.z*src2.z*scale), PIXELROUNDF32(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Trunc(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(src1.x*src2.x*scale), PIXELSATURATES16(src1.y*src2.y*scale), PIXELSATURATES16(src1.z*src2.z*scale), PIXELSATURATES16(src1.w*src2.w*scale));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Trunc,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Round(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float scale
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x * 4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(PIXELSATURATES16(PIXELROUNDF32(src1.x*src2.x*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.y*src2.y*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.z*src2.z*scale)), PIXELSATURATES16(PIXELROUNDF32(src1.w*src2.w*scale)));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Mul_S16_S16U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Round,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, scale);
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
Hip_Magnitude_S16_S16S16(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int  dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>1) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x*4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x*4);

    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(PIXELROUNDF32(hypotf(src1.x,src2.x)), PIXELROUNDF32(hypotf(src1.y,src2.y)), PIXELROUNDF32(hypotf(src1.z,src2.z)), PIXELROUNDF32(hypotf(src1.w,src2.w)));
    float4_to_s16s(pDstImage, dstIdx, dst);
}
int HipExec_Magnitude_S16_S16S16(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Magnitude_S16_S16S16,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}
// ----------------------------------------------------------------------------
// VxPhase kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Phase_U8_S16S16(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const short int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const short int *pSrcImage2, unsigned int srcImage2StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>1) + (x*4);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>1) + (x*4);

    float4 src1 = s16s_to_float4_ungrouped(pSrcImage1, src1Idx);
    float4 src2 = s16s_to_float4_ungrouped(pSrcImage2, src2Idx);
    float4 dst = make_float4(Norm_Atan2_deg(src1.x,src2.x), Norm_Atan2_deg(src1.y,src2.y), Norm_Atan2_deg(src1.z,src2.z), Norm_Atan2_deg(src1.w,src2.w));
    pDstImage[dstIdx] = float4_to_uchars(dst);

}
int HipExec_Phase_U8_S16S16(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {

    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Phase_U8_S16S16,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const short int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const short int *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

