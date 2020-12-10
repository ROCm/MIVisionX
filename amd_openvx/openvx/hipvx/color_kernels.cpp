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

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)

__device__ __forceinline__ float4 uchars_to_float4(uint src)
{
    return make_float4((float)(src & 0xFF), (float)((src & 0xFF00) >> 8), (float)((src & 0xFF0000) >> 16), (float)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ uint float4_to_uchars(float4 src)
{
    return ((uint)src.x & 0xFF) | (((uint)src.y & 0xFF) << 8) | (((uint)src.z & 0xFF) << 16) | (((uint)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint float4_to_uchars_u32(float4 src)
{
    // return ((uint)src.x&0xFF)<<24 | (((uint)src.y&0xFF)<<16) | (((uint)src.z&0xFF)<<8)| (((uint)src.w&0xFF));
    return ((uint)src.x & 0xFF) | (((uint)src.y & 0xFF) << 8) | (((uint)src.z & 0xFF) << 16) | (((uint)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint4 uchars_to_uint4(unsigned int src)
{
    printf("\nuchars_to_uint4 %d, %d, %d, %d", (unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8), (unsigned int)((src & 0xFF0000) >> 16), (unsigned int)((src & 0xFF000000) >> 24));
    return make_uint4((unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8), (unsigned int)((src & 0xFF0000) >> 16), (unsigned int)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ unsigned int uint4_to_uchars(uint4 src)
{
    printf("\nuint4_to_uchars %d, %d, %d, %d", ((unsigned char)src.x & 0xFF), ((unsigned char)src.y & 0xFF), ((unsigned char)src.z & 0xFF), ((unsigned char)src.w & 0xFF));
    return ((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8) | (((unsigned char)src.z & 0xFF) << 16) | (((unsigned char)src.w & 0xFF) << 24);
}

__device__ __forceinline__ float FLOAT_MAX(float f1, float f2)
{
    if (f1 >= f2)
        return f1;
    else
        return f2;
}

__device__ __forceinline__ float FLOAT_MIN(float f1, float f2)
{
    if (f1 <= f2)
        return f1;
    else
        return f2;
}
// ----------------------------------------------------------------------------
// VxLut kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *lut)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight))
        return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 4);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);
    for (int i = 0; i < 4; i++)
        pDstImage[dstIdx + i] = lut[pSrcImage1[src1Idx + i]];
}
int HipExec_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_uint8 *lut)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    vx_uint8 *hipLut;
    hipMalloc(&hipLut, 2048);
    hipMemcpy(hipLut, lut, 2048, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Lut_U8_U8,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes, (const unsigned char *)hipLut);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipLut);

    printf("\nHipExec_Lut_U8_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorDepth kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes>>1) + x;
    pDstImage[dstIdx] = (unsigned char)(pSrcImage[srcIdx] >> shift);
}
int HipExec_ColorDepth_U8_S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const short int *)pHipSrcImage, srcImageStrideInBytes,
                    shift);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ColorDepth_U8_S16_Wrap: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes>>1) + x;
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(pSrcImage[srcIdx] >> shift);
}
int HipExec_ColorDepth_U8_S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const short int *)pHipSrcImage, srcImageStrideInBytes,
                    shift);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ColorDepth_U8_S16_Sat: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * (dstImageStrideInBytes>>1) + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes) + x;
    pDstImage[dstIdx] = ((short int)pSrcImage[srcIdx]) << shift;
}
int HipExec_ColorDepth_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorDepth_S16_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    shift);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ColorDepth_S16_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelExtract kernels for hip backend
// ----------------------------------------------------------------------------

//**********************************
//ChannelExtract_U8_U16_Pos0
//**********************************
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight))
        return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x * 4;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 8;

    for (int i = 0; i < 4; i++)
        pDstImage[dstIdx + i] = pSrcImage1[src1Idx + i * 2];
}
int HipExec_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U16_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U16_Pos1
**********************************/

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight))
        return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x * 4;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 8;

    for (int i = 0; i < 4; i++)
        pDstImage[dstIdx + i] = pSrcImage1[src1Idx + i * 2 + 1];
    // printf("\n&pDstImage[dstIdx], &pDstImage[dstIdx + 1]: %p, %p", (void*)(&pDstImage[dstIdx]), (void*)(&pDstImage[dstIdx + 1]));
}
int HipExec_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U16_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos0
**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;

    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    pDstImage[dstIdx] = pSrcImage1[src1Idx];
}
int HipExec_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos1

**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;
    // unsigned char
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 4;

    pDstImage[dstIdx] = pSrcImage1[src1Idx + 1];
}
int HipExec_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos2

**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;
    // unsigned char
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 4;
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 2];
}
int HipExec_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos2,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos2: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos3
**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;
    // unsigned char
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 4;

    pDstImage[dstIdx] = pSrcImage1[src1Idx + 3];
}
int HipExec_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos3,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos0

*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + x * 3;

    pDstImage[dstIdx] = pSrcImage1[src1Idx];
}
int HipExec_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos1

*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *3 ;

    pDstImage[dstIdx] = pSrcImage1[src1Idx + 1];
}
int HipExec_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos2

*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *3 ;

    pDstImage[dstIdx] = pSrcImage1[src1Idx + 2];
}
int HipExec_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos2,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelCombine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx];  
}
int HipExec_ChannelCombine_U16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d , srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelCombine_U16_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelCombine_U16_U8U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U24_U8U8U8_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*3);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx]; 
    pDstImage[dstIdx+2]  = pSrcImage3[src3Idx];  
}
int HipExec_ChannelCombine_U24_U8U8U8_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelCombine_U24_U8U8U8_RGB,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelCombine_U24_U8U8U8_RGB: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage2[src2Idx];
    pDstImage[dstIdx+1]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+2]  = pSrcImage3[src3Idx]; 
    pDstImage[dstIdx+3]  = pSrcImage1[src1Idx+1];  
}
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_UYVY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelCombine_U32_U8U8U8_UYVY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx];
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx];
    pDstImage[dstIdx+2]  = pSrcImage1[src1Idx+1]; 
    pDstImage[dstIdx+3]  = pSrcImage3[src3Idx];
}
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_YUYV,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelCombine_U32_U8U8U8_YUYV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8U8_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes,
    const unsigned char *pSrcImage4, unsigned int srcImage4StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    unsigned int src4Idx =  y*(srcImage4StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx]; 
    pDstImage[dstIdx+2]  = pSrcImage3[src3Idx];
    pDstImage[dstIdx+3]  = pSrcImage4[src4Idx];  
}
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
    const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8U8_RGBX,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes,
                    (const unsigned char *)pHipSrcImage4, srcImage4StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelCombine_U32_U8U8U8U8_RGBX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorConvert kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 4);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 3);

    pDstImage[dstIdx] = pSrcImage1[src1Idx];
    pDstImage[dstIdx + 1] = pSrcImage1[src1Idx + 1];
    pDstImage[dstIdx + 2] = pSrcImage1[src1Idx + 2];
    pDstImage[dstIdx + 3] = (unsigned char)255;
}
int HipExec_ColorConvert_RGBX_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_RGB: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 3);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    pDstImage[dstIdx] = pSrcImage1[src1Idx];
    pDstImage[dstIdx + 1] = pSrcImage1[src1Idx + 1];
    pDstImage[dstIdx + 2] = pSrcImage1[src1Idx + 2];
}
int HipExec_ColorConvert_RGB_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_RGBX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 6);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
    Ypix1 = (float )pSrcImage1[src1Idx];
    Upix = (float )pSrcImage1[src1Idx + 1] - 128.0f;
    Ypix2 = (float )pSrcImage1[src1Idx + 2];
    Vpix = (float )pSrcImage1[src1Idx + 3] - 128.0f;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx] = Rpix;
    pDstImage[dstIdx+1] = Gpix;
    pDstImage[dstIdx+2] = Bpix;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx+3] = Rpix;
    pDstImage[dstIdx+4] = Gpix;
    pDstImage[dstIdx+5] = Bpix;
}
int HipExec_ColorConvert_RGB_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_YUYV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}



__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 6);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
    Upix  = (float )pSrcImage1[src1Idx] - 128.0f;
    Ypix1 = (float )pSrcImage1[src1Idx + 1] ;
    Vpix  = (float )pSrcImage1[src1Idx + 2] - 128.0f;
    Ypix2 = (float )pSrcImage1[src1Idx + 3] ;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx] = Rpix;
    pDstImage[dstIdx+1] = Gpix;
    pDstImage[dstIdx+2] = Bpix;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);
    

    pDstImage[dstIdx+3] = Rpix;
    pDstImage[dstIdx+4] = Gpix;
    pDstImage[dstIdx+5] = Bpix;

}
int HipExec_ColorConvert_RGB_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_UYVY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 8);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
    Ypix1 = (float )pSrcImage1[src1Idx];
    Upix = (float )pSrcImage1[src1Idx + 1] - 128.0f;
    Ypix2 = (float )pSrcImage1[src1Idx + 2];
    Vpix = (float )pSrcImage1[src1Idx + 3] - 128.0f;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx] = Rpix;
    pDstImage[dstIdx+1] = Gpix;
    pDstImage[dstIdx+2] = Bpix;
    pDstImage[dstIdx+3] = 255;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx+4] = Rpix;
    pDstImage[dstIdx+5] = Gpix;
    pDstImage[dstIdx+6] = Bpix;
    pDstImage[dstIdx+7] = 255;
}
int HipExec_ColorConvert_RGBX_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_YUYV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + (x * 8);
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);

    float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
    Upix  = (float )pSrcImage1[src1Idx] - 128.0f;
    Ypix1 = (float )pSrcImage1[src1Idx + 1] ;
    Vpix  = (float )pSrcImage1[src1Idx + 2] - 128.0f;
    Ypix2 = (float )pSrcImage1[src1Idx + 3] ;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

    pDstImage[dstIdx] = Rpix;
    pDstImage[dstIdx+1] = Gpix;
    pDstImage[dstIdx+2] = Bpix;
    pDstImage[dstIdx+3] = 255;

    Rpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
    Gpix = FLOAT_MIN(FLOAT_MAX(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
    Bpix = FLOAT_MIN(FLOAT_MAX(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);
    

    pDstImage[dstIdx+4] = Rpix;
    pDstImage[dstIdx+5] = Gpix;
    pDstImage[dstIdx+6] = Bpix;
    pDstImage[dstIdx+7] = 255;
}
int HipExec_ColorConvert_RGBX_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_UYVY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcYImage, unsigned int srcYImageStrideInBytes,
    const unsigned char *pSrcUImage, unsigned int srcUImageStrideInBytes,
    const unsigned char *pSrcVImage, unsigned int srcVImageStrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 6);
    unsigned int srcYIdx = y * (srcYImageStrideInBytes<<1) + (x * 2);
    unsigned int srcUIdx = y * (srcUImageStrideInBytes) + x;
    unsigned int srcVIdx = y * (srcVImageStrideInBytes) + x;

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcYImage[srcYIdx];
    Bpix = (float)pSrcUImage[srcUIdx] - 128.0f;
    Rpix = (float)pSrcVImage[srcVIdx] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcYImage[srcYIdx + 1];

    pDstImage[dstIdx + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcYImage[srcYIdx + srcYImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcYImage[srcYIdx + srcYImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
}
int HipExec_ColorConvert_RGB_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcYImageStrideInBytes = %d srcUImageStrideInBytes = %d srcVImageStrideInBytes = %d\n", 
                    dstWidth, dstHeight, dstImageStrideInBytes, srcYImageStrideInBytes, srcUImageStrideInBytes, srcVImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_IYUV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcYImage, srcYImageStrideInBytes,
                       (const unsigned char *)pHipSrcUImage, srcUImageStrideInBytes,
                       (const unsigned char *)pHipSrcVImage, srcVImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_IYUV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned char *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 6);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes<<1) + (x * 2);
    unsigned int srcChromaIdx = y * (srcChromaImageStrideInBytes) + (x * 2);

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcLumaImage[srcLumaIdx];
    Bpix = (float)pSrcChromaImage[srcChromaIdx] - 128.0f;
    Rpix = (float)pSrcChromaImage[srcChromaIdx + 1] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + 1];

    pDstImage[dstIdx + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
}
int HipExec_ColorConvert_RGB_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcLumaImageStrideInBytes = %d , srcChromaImageStrideInBytes = %d \n", dstWidth, dstHeight, dstImageStrideInBytes, srcLumaImageStrideInBytes, srcChromaImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV12,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned char *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_NV12: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned char *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 6);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes<<1) + (x * 2);
    unsigned int srcChromaIdx = y * (srcChromaImageStrideInBytes) + (x * 2);

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcLumaImage[srcLumaIdx];
    Rpix = (float)pSrcChromaImage[srcChromaIdx] - 128.0f;
    Bpix = (float)pSrcChromaImage[srcChromaIdx + 1] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + 1];

    pDstImage[dstIdx + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
}
int HipExec_ColorConvert_RGB_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcLumaImageStrideInBytes = %d , srcChromaImageStrideInBytes = %d \n", dstWidth, dstHeight, dstImageStrideInBytes, srcLumaImageStrideInBytes, srcChromaImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV21,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned char *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGB_NV21: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcYImage, unsigned int srcYImageStrideInBytes,
    const unsigned char *pSrcUImage, unsigned int srcUImageStrideInBytes,
    const unsigned char *pSrcVImage, unsigned int srcVImageStrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 8);
    unsigned int srcYIdx = y * (srcYImageStrideInBytes<<1) + (x * 2);
    unsigned int srcUIdx = y * (srcUImageStrideInBytes) + x;
    unsigned int srcVIdx = y * (srcVImageStrideInBytes) + x;

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcYImage[srcYIdx];
    Bpix = (float)pSrcUImage[srcUIdx] - 128.0f;
    Rpix = (float)pSrcVImage[srcVIdx] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 3] = 255;

    Ypix = (float)pSrcYImage[srcYIdx + 1];

    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 7] = 255;

    Ypix = (float)pSrcYImage[srcYIdx + srcYImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = 255;

    Ypix = (float)pSrcYImage[srcYIdx + srcYImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 7] = 255;
}
int HipExec_ColorConvert_RGBX_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3)>>1 , globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcYImageStrideInBytes = %d srcUImageStrideInBytes = %d srcVImageStrideInBytes = %d\n", 
                    dstWidth, dstHeight, dstImageStrideInBytes, srcYImageStrideInBytes, srcUImageStrideInBytes, srcVImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_IYUV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcYImage, srcYImageStrideInBytes,
                       (const unsigned char *)pHipSrcUImage, srcUImageStrideInBytes,
                       (const unsigned char *)pHipSrcVImage, srcVImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_IYUV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned char *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 8);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes<<1) + (x * 2);
    unsigned int srcChromaIdx = y * (srcChromaImageStrideInBytes) + (x * 2);

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcLumaImage[srcLumaIdx];
    Bpix = (float)pSrcChromaImage[srcChromaIdx] - 128.0f;
    Rpix = (float)pSrcChromaImage[srcChromaIdx + 1] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 3] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + 1];

    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 7] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 7] = 255;
}
int HipExec_ColorConvert_RGBX_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcLumaImageStrideInBytes = %d , srcChromaImageStrideInBytes = %d \n", dstWidth, dstHeight, dstImageStrideInBytes, srcLumaImageStrideInBytes, srcChromaImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV12,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned char *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_NV12: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned char *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes<<1) + (x * 8);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes<<1) + (x * 2);
    unsigned int srcChromaIdx = y * (srcChromaImageStrideInBytes) + (x * 2);

    float Ypix, Rpix, Gpix, Bpix;
    
    Ypix = (float)pSrcLumaImage[srcLumaIdx];
    Rpix = (float)pSrcChromaImage[srcChromaIdx] - 128.0f;
    Bpix = (float)pSrcChromaImage[srcChromaIdx + 1] - 128.0f;

    Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
    Rpix *= 1.5748f;
    Bpix *= 1.8556f;

    pDstImage[dstIdx] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0);
    pDstImage[dstIdx + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 3] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + 1];

    pDstImage[dstIdx + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + 7] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes];
    pDstImage[dstIdx + dstImageStrideInBytes] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 1] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 2] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 3] = 255;

    Ypix = (float)pSrcLumaImage[srcLumaIdx + srcLumaImageStrideInBytes + 1];
    pDstImage[dstIdx + dstImageStrideInBytes + 4] = FLOAT_MIN(FLOAT_MAX(Ypix + Rpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 5] = FLOAT_MIN(FLOAT_MAX(Ypix - Gpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 6] = FLOAT_MIN(FLOAT_MAX(Ypix + Bpix, 0.0f), 255.0f);
    pDstImage[dstIdx + dstImageStrideInBytes + 7] = 255;
}
int HipExec_ColorConvert_RGBX_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcLumaImageStrideInBytes = %d , srcChromaImageStrideInBytes = %d \n", dstWidth, dstHeight, dstImageStrideInBytes, srcLumaImageStrideInBytes, srcChromaImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV21,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned char *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_RGBX_NV21: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_NV12_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImageLuma, unsigned int dstImageLumaStrideInBytes,
    unsigned char *pDstImageChroma, unsigned int dstImageChromaStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight))
        return;
    unsigned int dstIdxLuma = y * (dstImageLumaStrideInBytes<<1) + x * 2;
    unsigned int dstIdxChroma = y * (dstImageChromaStrideInBytes ) + x *2;
    unsigned int src1Idx = y * (srcImage1StrideInBytes<<1) + x * 6;

    float R, G, B, U , V;
    // //first row
    R = (float)pSrcImage1[src1Idx];
	G = (float)pSrcImage1[src1Idx+1];
	B = (float)pSrcImage1[src1Idx+2];

    pDstImageLuma[dstIdxLuma] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
	V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

    R = (float)pSrcImage1[src1Idx+3];
	G = (float)pSrcImage1[src1Idx+4];
	B = (float)pSrcImage1[src1Idx+5];

	pDstImageLuma[dstIdxLuma + 1] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);


    // //second row
    R = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes ];
	G = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +1];
	B = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +2];

	pDstImageLuma[dstIdxLuma + dstImageLumaStrideInBytes] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

	R = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +3];
	G = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +4];
	B = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +5];

	pDstImageLuma[dstIdxLuma + dstImageLumaStrideInBytes + 1] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

    U /= 4.0;	V /= 4.0;

    pDstImageChroma[dstIdxChroma] = (unsigned char) U;
    pDstImageChroma[dstIdxChroma + 1] = (unsigned char) V;
			


}
int HipExec_ColorConvert_NV12_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageChromaStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageChromaStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImageLuma, dstImageLumaStrideInBytes,
                       (unsigned char *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_NV12_RGB: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))

Hip_ColorConvert_NV12_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImageLuma, unsigned int dstImageLumaStrideInBytes,
    unsigned char *pDstImageChroma, unsigned int dstImageChromaStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight))
        return;
    unsigned int dstIdxLuma = y * (dstImageLumaStrideInBytes<<1) + x * 2;
    unsigned int dstIdxChroma = y * (dstImageChromaStrideInBytes ) + x *2;
    unsigned int src1Idx = y * (srcImage1StrideInBytes<<1) + x * 8;

    float R, G, B, U , V;
    // //first row
    R = (float)pSrcImage1[src1Idx];
	G = (float)pSrcImage1[src1Idx+1];
	B = (float)pSrcImage1[src1Idx+2];

    pDstImageLuma[dstIdxLuma] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
	V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

    R = (float)pSrcImage1[src1Idx+4];
	G = (float)pSrcImage1[src1Idx+5];
	B = (float)pSrcImage1[src1Idx+6];

	pDstImageLuma[dstIdxLuma + 1] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);


    // //second row
    R = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes ];
	G = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +1];
	B = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +2];

	pDstImageLuma[dstIdxLuma + dstImageLumaStrideInBytes] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

	R = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +4];
	G = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +5];
	B = (float)pSrcImage1[src1Idx + srcImage1StrideInBytes +6];

	pDstImageLuma[dstIdxLuma + dstImageLumaStrideInBytes + 1] = (unsigned char)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
	U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
	V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

    U /= 4.0;	V /= 4.0;

    pDstImageChroma[dstIdxChroma] = (unsigned char) U;
    pDstImageChroma[dstIdxChroma + 1] = (unsigned char) V;
			


}
int HipExec_ColorConvert_NV12_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstImageChromaStrideInBytes = %d, srcImage1StrideInBytes = %d\n", dstWidth, dstHeight, dstImageChromaStrideInBytes, srcImage1StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImageLuma, dstImageLumaStrideInBytes,
                       (unsigned char *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_NV12_RGBX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes<<1) + (x * 2);
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes<<1) + (x * 6);

    float Upix, Vpix, Rpix, Gpix, Bpix;
    
    Rpix = (float)pSrcImage[srcIdx];
    Gpix = (float)pSrcImage[srcIdx + 1];
    Bpix = (float)pSrcImage[srcIdx + 2];

    pDstYImage[dstYIdx] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix = ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix = ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);

    Rpix = (float)pSrcImage[srcIdx + 3];
    Gpix = (float)pSrcImage[srcIdx + 4];
    Bpix = (float)pSrcImage[srcIdx + 5];

    pDstYImage[dstYIdx + 1] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);
    
    Rpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes];
    Gpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 1];
    Bpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 2];

    pDstYImage[dstYIdx + dstYImageStrideInBytes] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);

    Rpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 3];
    Gpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 4];
    Bpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 5];

    pDstYImage[dstYIdx + dstYImageStrideInBytes + 1] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);

    Upix /= 4.0f;
    Vpix /= 4.0f;

    pDstUImage[dstUIdx] = Upix;
    pDstVImage[dstVIdx] = Vpix;
}
int HipExec_ColorConvert_IYUV_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_IYUV_RGB: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes<<1) + (x * 2);
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes<<1) + (x * 8);

    float Upix, Vpix, Rpix, Gpix, Bpix;
    
    Rpix = (float)pSrcImage[srcIdx];
    Gpix = (float)pSrcImage[srcIdx + 1];
    Bpix = (float)pSrcImage[srcIdx + 2];

    pDstYImage[dstYIdx] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix = (Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f;
    Vpix = (Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f;

    Rpix = (float)pSrcImage[srcIdx + 4];
    Gpix = (float)pSrcImage[srcIdx + 5];
    Bpix = (float)pSrcImage[srcIdx + 6];

    pDstYImage[dstYIdx + 1] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);
    
    Rpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes];
    Gpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 1];
    Bpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 2];

    pDstYImage[dstYIdx + dstYImageStrideInBytes] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);

    Rpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 4];
    Gpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 5];
    Bpix = (float)pSrcImage[srcIdx + srcImageStrideInBytes + 6];

    pDstYImage[dstYIdx + dstYImageStrideInBytes + 1] = ((Rpix * 0.2126f) + (Gpix * 0.7152f) + (Bpix * 0.0722f));
    Upix += ((Rpix * -0.1146f) + (Gpix * -0.3854f) + (Bpix * 0.5f) + 128.0f);
    Vpix += ((Rpix * 0.5f) + (Gpix * -0.4542f) + (Bpix * -0.0458f) + 128.0f);

    Upix /= 4.0f;
    Vpix /= 4.0f;

    pDstUImage[dstUIdx] = Upix;
    pDstVImage[dstVIdx] = Vpix;
}
int HipExec_ColorConvert_IYUV_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_IYUV_RGBX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}
// ----------------------------------------------------------------------------
// VxFormatConvert kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstLumaImage, unsigned int dstLumaImageStrideInBytes,
    unsigned char *pDstChromaImage, unsigned int dstChromaImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdxLuma = y * (2 * dstLumaImageStrideInBytes) + (x * 2);
    unsigned int dstIdxChroma = y * (dstChromaImageStrideInBytes) + (x * 2);
    unsigned int srcIdx = y * (2 * srcImageStrideInBytes) + (x * 4);

    pDstChromaImage[dstIdxChroma] = (pSrcImage[srcIdx] + pSrcImage[srcIdx + srcImageStrideInBytes]) / 2;  //U
    pDstLumaImage[dstIdxLuma] = pSrcImage[srcIdx + 1];                                                    //Y
    pDstLumaImage[dstIdxLuma + dstLumaImageStrideInBytes] = pSrcImage[srcIdx + srcImageStrideInBytes + 1] ;// Y next Row
    pDstChromaImage[dstIdxChroma + 1] = (pSrcImage[srcIdx + 2] + pSrcImage[srcIdx + srcImageStrideInBytes + 2]) / 2; //V
    pDstLumaImage[dstIdxLuma + 1] = pSrcImage[srcIdx + 3];                                                 //Y
    pDstLumaImage[dstIdxLuma + dstLumaImageStrideInBytes + 1] = pSrcImage[srcIdx + srcImageStrideInBytes + 3]  ;// Y next Row

}
int HipExec_FormatConvert_NV12_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
    vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstLumaImageStrideInBytes = %d, srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstLumaImageStrideInBytes , srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_FormatConvert_NV12_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pDstLumaImage, dstLumaImageStrideInBytes,
                        (unsigned char *)pDstChromaImage, dstChromaImageStrideInBytes,
                       (const unsigned char *)pSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_FormatConvert_NV12_UYVY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstLumaImage, unsigned int dstLumaImageStrideInBytes,
    unsigned char *pDstChromaImage, unsigned int dstChromaImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdxLuma = y * (2 * dstLumaImageStrideInBytes) + (x * 2);
    unsigned int dstIdxChroma = y * (dstChromaImageStrideInBytes) + (x * 2);
    unsigned int srcIdx = y * (2 * srcImageStrideInBytes) + (x * 4);

    pDstLumaImage[dstIdxLuma] = pSrcImage[srcIdx ];   //Y
    pDstLumaImage[dstIdxLuma + dstLumaImageStrideInBytes] =  pSrcImage[srcIdx + srcImageStrideInBytes]; //Y next Row
    pDstChromaImage[dstIdxChroma] = (pSrcImage[srcIdx + 1] + pSrcImage[srcIdx + srcImageStrideInBytes + 1]) / 2;  //U
    pDstLumaImage[dstIdxLuma + 1] = pSrcImage[srcIdx + 2];   //Y
    pDstLumaImage[dstIdxLuma + dstLumaImageStrideInBytes + 1] =  pSrcImage[srcIdx + srcImageStrideInBytes + 2]; //Y next Row
    pDstChromaImage[dstIdxChroma + 1] = (pSrcImage[srcIdx + 3] + pSrcImage[srcIdx + srcImageStrideInBytes + 3]) / 2;  //V


}
int HipExec_FormatConvert_NV12_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
    vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    printf("\ndstWidth = %d, dstHeight = %d\ndstLumaImageStrideInBytes = %d, srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstLumaImageStrideInBytes , srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_FormatConvert_NV12_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pDstLumaImage, dstLumaImageStrideInBytes,
                        (unsigned char *)pDstChromaImage, dstChromaImageStrideInBytes,
                       (const unsigned char *)pSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_FormatConvert_NV12_YUYV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes<<1) + (x * 2);
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes<<1) + (x * 4);

    pDstUImage[dstUIdx] = (pSrcImage[srcIdx] + pSrcImage[srcIdx + srcImageStrideInBytes]) >> 1;
    pDstYImage[dstYIdx] = pSrcImage[srcIdx + 1];
    pDstYImage[dstYIdx + dstYImageStrideInBytes] = pSrcImage[srcIdx + srcImageStrideInBytes + 1];
    pDstVImage[dstVIdx] = (pSrcImage[srcIdx + 2] + pSrcImage[srcIdx + srcImageStrideInBytes + 2]) >> 1;
    pDstYImage[dstYIdx + 1] = pSrcImage[srcIdx + 3];
    pDstYImage[dstYIdx + dstYImageStrideInBytes + 1] = pSrcImage[srcIdx + srcImageStrideInBytes + 3];
}
int HipExec_FormatConvert_IYUV_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_FormatConvert_IYUV_UYVY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes<<1) + (x * 2);
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes<<1) + (x * 4);

    pDstYImage[dstYIdx] = pSrcImage[srcIdx];
    pDstYImage[dstYIdx + dstYImageStrideInBytes] = pSrcImage[srcIdx + srcImageStrideInBytes];
    pDstUImage[dstUIdx] = (pSrcImage[srcIdx + 1] + pSrcImage[srcIdx + srcImageStrideInBytes + 1]) >> 1;
    pDstYImage[dstYIdx + 1] = pSrcImage[srcIdx + 2];
    pDstYImage[dstYIdx + dstYImageStrideInBytes + 1] = pSrcImage[srcIdx + srcImageStrideInBytes + 2];
    pDstVImage[dstVIdx] = (pSrcImage[srcIdx + 3] + pSrcImage[srcIdx + srcImageStrideInBytes + 3]) >> 1;
}
int HipExec_FormatConvert_IYUV_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_FormatConvert_IYUV_YUYV: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_YUV4_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x *4 >= dstWidth) || (y  >= dstHeight))
        return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes) + x*4;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x*4;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x*4;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + (x * 16);

    float R, G, B;
    int j=0;
    for (int i=0;i<4;i++)
    {
        R = pSrcImage[srcIdx+ j];
        G = pSrcImage[srcIdx+ 1+j];
        B = pSrcImage[srcIdx + 2+j];
        pDstYImage[dstYIdx+i] = ((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
        pDstUImage[dstUIdx+i] = ((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f);
        pDstVImage[dstVIdx+i] = ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);
        j =j +4;
    }
}
int HipExec_ColorConvert_YUV4_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

    printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ColorConvert_YUV4_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, 0, dstWidth, dstHeight,
                       (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ColorConvert_YUV4_RGBX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}



// __global__ void __attribute__((visibility("default")))
// Hip_ColorConvert_YUV4_NV12(
//     vx_uint32 dstWidth, vx_uint32 dstHeight,
//     unsigned char *pDstYImage, unsigned int dstYImageStrideInBytes,
//     unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
//     unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
//     const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes)
// {
//     int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
//     int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
//     if ((x *4 >= dstWidth) || (y  >= dstHeight))
//         return;
//     unsigned int dstYIdx = y * (dstYImageStrideInBytes) + x*4;
//     unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x*4;
//     unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x*4;
//     unsigned int srcIdx = y * (srcImageStrideInBytes) + (x * 16);

//     float R, G, B;
//     int j=0;
//     for (int i=0;i<4;i++)
//     {
//         R = pSrcImage[srcIdx+ j];
//         G = pSrcImage[srcIdx+ 1+j];
//         B = pSrcImage[srcIdx + 2+j];
//         pDstYImage[dstYIdx+i] = ((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
//         pDstUImage[dstUIdx+i] = ((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f);
//         pDstVImage[dstVIdx+i] = ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);
//         j =j +4;
//     }
// }
// int HipExec_ColorConvert_YUV4_NV12(
//     vx_uint32 dstWidth, vx_uint32 dstHeight,
//     vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
//     vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
//     vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
//     const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
//     )
// {
//     hipEvent_t start, stop;
//     int localThreads_x = 16, localThreads_y = 16;
//     int globalThreads_x = (dstWidth+3)>>1, globalThreads_y = (dstHeight+3)>>1;

//     printf("\ndstWidth = %d, dstHeight = %d\ndstYImageStrideInBytes = %d, dstUImageStrideInBytes = %d, dstVImageStrideInBytes = %d srcImageStrideInBytes = %d\n", dstWidth, dstHeight, dstYImageStrideInBytes, dstUImageStrideInBytes, dstVImageStrideInBytes, srcImageStrideInBytes);

//     hipEventCreate(&start);
//     hipEventCreate(&stop);
//     float eventMs = 1.0f;
//     hipEventRecord(start, NULL);
//     hipLaunchKernelGGL(Hip_ColorConvert_YUV4_NV12,
//                        dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
//                        dim3(localThreads_x, localThreads_y),
//                        0, 0, dstWidth, dstHeight,
//                        (unsigned char *)pHipDstYImage, dstYImageStrideInBytes,
//                        (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
//                        (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
//                        (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

//     hipEventRecord(stop, NULL);
//     hipEventSynchronize(stop);
//     hipEventElapsedTime(&eventMs, start, stop);
//     printf("\nHipExec_ColorConvert_YUV4_NV12: Kernel time: %f\n", eventMs);
//     return VX_SUCCESS;
// }