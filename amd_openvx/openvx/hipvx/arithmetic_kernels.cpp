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



//#include "../ago/ago_internal.h"
#include "hip_kernels.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"

__device__ __forceinline__ float4 ucharTofloat4(unsigned int src)
{
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ float2 s16Tofloat2( int src)

{
    return make_float2((float)(src&0xFFFF), (float)((src&0xFFFF0000)>>16));
    // return make_float2((float)(src&0xFFFF), (float)((src&0xFFFF0000)>>16));
    //  return make_float4((float)(src&0xFFFF), (float)((src&0xFFFF0000)>>16), (float)((src&0xFFFF00000000)>>32), (float)((src&0xFFFF000000000000)>>48));
}

__device__ __forceinline__ uint float4ToUint(float4 src)
{
  return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
}

// ----------------------------------------------------------------------------
// VxAbsDiff kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(fabsf(src1.x-src2.x), fabsf(src1.y-src2.y), fabsf(src1.z-src2.z), fabsf(src1.w-src2.w));
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_AbsDiff_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_AbsDiff_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_AbsDiff_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxAdd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Add_U8_U8U8_Wrap(vx_uint32 dstWidth, vx_uint32 dstHeight, vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_Add_U8_U8U8_Wrap: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Add_U8_U8U8_Wrap: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight, unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Add_U8_U8U8_Sat(vx_uint32 dstWidth, vx_uint32 dstHeight, vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_Add_U8_U8U8_Sat: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Add_U8_U8U8_Sat: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1.z+src2.z, src1.w+src2.w);
    // pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);

    pDstImage[dstIdx] = ((int)dst.x&0xFFFF) | (((int)dst.y&0xFFFF)<<16);
    pDstImage[dstIdx+1] = ((int)dst.z&0xFFFF) | (((int)dst.w&0xFFFF)<<16);
}

int HipExec_Add_S16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Add_S16_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Add_S16_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// **NOT YET WORKING VERSION** //
__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, int *pDstImage, unsigned int dstImageStrideInBytes,
    int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    
    float2 src1 = s16Tofloat2(pSrcImage1[src1Idx]);
    float2 src1_2 = s16Tofloat2(pSrcImage1[src1Idx+1]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x+src2.x, src1.y+src2.y, src1_2.x+src2.z, src1_2.y+src2.w);
    // pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);

    pDstImage[dstIdx] = ((int)dst.x&0xFFFF) | (((int)dst.y&0xFFFF)<<16);
    pDstImage[dstIdx+1] = ((int)dst.z&0xFFFF) | (((int)dst.w&0xFFFF)<<16);
}

int HipExec_Add_S16_S16U8_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (int *)pHipDstImage , dstImageStrideInBytes, (int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Add_S16_S16U8_Wrap: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}
// **NOT YET WORKING VERSION** //





// ----------------------------------------------------------------------------
// VxSub kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Subtract_U8_U8U8_Wrap
	(
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage1,
        unsigned int     srcImage1StrideInBytes,
        const unsigned int    * pSrcImage2,
        unsigned int     srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}
int HipExec_Subtract_U8_U8U8_Wrap
    (
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8     * pHipDstImage,
    vx_uint32     dstImageStrideInBytes,
    const vx_uint8    * pHipSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    const vx_uint8    * pHipSrcImage2,
    vx_uint32     srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Subtract_U8_U8U8_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Subtract_U8_U8U8_Wrap: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// VxSubtract kernel for hip backend
__global__ void __attribute__((visibility("default")))
Hip_Subtract_U8_U8U8_Sat
	(
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage1,
        unsigned int     srcImage1StrideInBytes,
        const unsigned int    * pSrcImage2,
        unsigned int     srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}
int HipExec_Subtract_U8_U8U8_Sat
    (
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8     * pHipDstImage,
    vx_uint32     dstImageStrideInBytes,
    const vx_uint8    * pHipSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    const vx_uint8    * pHipSrcImage2,
    vx_uint32     srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Subtract_U8_U8U8_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Subtract_U8_U8U8_Sat: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMul kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxMagnitude kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxPhase kernels for hip backend
// ----------------------------------------------------------------------------












/*
// VxSubtract kernel for hip backend
__global__ void __attribute__((visibility("default")))
Hip_Subtract_U8_U8U8
	(
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage1,
        unsigned int     srcImage1StrideInBytes,
        const unsigned int    * pSrcImage2,
        unsigned int     srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src1.x-src2.x, src1.y-src2.y, src1.z-src2.z, src1.w-src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Subtract_U8_U8U8
    (
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8     * pHipDstImage,
    vx_uint32     dstImageStrideInBytes,
    const vx_uint8    * pHipSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    const vx_uint8    * pHipSrcImage2,
    vx_uint32     srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3) >> 2,   globalThreads_y = dstHeight;
    //printf("HipExec_AbsDiff_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Subtract_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Subtract_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}
*/
