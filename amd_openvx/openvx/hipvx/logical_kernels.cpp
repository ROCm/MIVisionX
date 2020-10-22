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

__device__ __forceinline__ uint float4ToUint(float4 src)
{
  return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
}

// ----------------------------------------------------------------------------
// VxAnd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U8U8
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
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = ucharTofloat4(pSrcImage1[src1Idx]);
    float4 src2 = ucharTofloat4(pSrcImage2[src2Idx]);
    float4 dst = make_float4((int)src1.x&(int)src2.x, (int)src1.y&(int)src2.y,(int) src1.z&(int)src2.z, (int)src1.w&(int)src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}
int HipExec_And_U8_U8U8
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
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_And_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_And_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_And_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxOr kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U8U8(
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
    float4 dst = make_float4((int)src1.x|(int)src2.x, (int)src1.y|(int)src2.y, (int)src1.z|(int)src2.z,(int)src1.w|(int)src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Or_U8_U8U8(vx_uint32 dstWidth, vx_uint32 dstHeight, vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_Or_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Or_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Or_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxXor kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U8U8(
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
    float4 dst = make_float4((int)src1.x^(int)src2.x, (int)src1.y^(int)src2.y, (int)src1.z^(int)src2.z, (int)src1.w^(int)src2.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Xor_U8_U8U8(vx_uint32 dstWidth, vx_uint32 dstHeight, vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    //printf("HipExec_Xor_U8_U8U8: dst: %p src1: %p src2: %p <%dx%d> stride <%dx%dx%d>\n", pHipDstImage, pHipSrcImage1, pHipSrcImage2,
     //       dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Xor_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Xor_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxNand kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxNor kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxXnor kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxNot kernels for hip backend
// ----------------------------------------------------------------------------
// VxNot(Bitwise NOT) kernel for hip backend
__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U8U8
	(
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage,
        unsigned int     srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x>= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + x;
    float4 src = ucharTofloat4(pSrcImage[srcIdx]);
    float4 dst = make_float4(~(int)src.x, ~(int)src.y, ~(int)src.z,~(int) src.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Not_U8_U8U8
    (
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8     * pHipDstImage,
    vx_uint32     dstImageStrideInBytes,
    const vx_uint8    * pHipSrcImage,
    vx_uint32     srcImage1StrideInBytes
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
    hipLaunchKernelGGL(Hip_Not_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Not_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}




/*
// VxNot(Bitwise NOT) kernel for hip backend
__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U8U8
	(
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage,
        unsigned int     srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + x;
    float4 src = ucharTofloat4(pSrcImage[srcIdx]);
    float4 dst = make_float4(~(int)src.x, ~(int)src.y, (int)src.z,(int) src.w);
    pDstImage[dstIdx] = ((int)dst.x&0xFF) | (((int)dst.y&0xFF)<<8) | (((int)dst.z&0xFF)<<16)| (((int)dst.w&0xFF) << 24);
}

int HipExec_Not_U8_U8U8
    (
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8     * pHipDstImage,
    vx_uint32     dstImageStrideInBytes,
    const vx_uint8    * pHipSrcImage,
    vx_uint32     srcImage1StrideInBytes
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
    hipLaunchKernelGGL(Hip_Not_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("HipExec_Not_U8_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}
*/
