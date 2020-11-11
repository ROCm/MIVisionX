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
#include <stdlib.h>

// __device__ __forceinline__ float4 uchars_to_float4(uint src)
// {
//     return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
// }

// __device__ __forceinline__ uint float4_to_uchars(float4 src)
// {
//     return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
// }

#define HIPVXMAX3(a,b,c)  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define HIPVXMIN3(a,b,c)  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))

// ----------------------------------------------------------------------------
// VxBox kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Box_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    int sum = 0;
    sum += ((int)*(pSrcImage + srcIdx) + (int)*(pSrcImage + srcIdxTopRow) + (int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum += ((int)*(pSrcImage + srcIdx - 1) + (int)*(pSrcImage + srcIdxTopRow - 1) + (int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += ((int)*(pSrcImage + srcIdx + 1) + (int)*(pSrcImage + srcIdxTopRow + 1) + (int)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)((float)sum / 9);
}
int HipExec_Box_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Box_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Box_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxDilate kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Dilate_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    unsigned char valCol0 = 0, valCol1 = 0, valCol2 = 0;
    valCol1 = HIPVXMAX3(*(pSrcImage + srcIdx), *(pSrcImage + srcIdxTopRow), *(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      valCol0 = HIPVXMAX3(*(pSrcImage + srcIdx - 1), *(pSrcImage + srcIdxTopRow - 1), *(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      valCol2 = HIPVXMAX3(*(pSrcImage + srcIdx + 1), *(pSrcImage + srcIdxTopRow + 1), *(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)HIPVXMAX3(valCol0, valCol1, valCol2);
}
int HipExec_Dilate_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Dilate_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Dilate_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxErode kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Erode_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    unsigned char valCol0 = 0, valCol1 = 0, valCol2 = 0;
    valCol1 = HIPVXMIN3(*(pSrcImage + srcIdx), *(pSrcImage + srcIdxTopRow), *(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      valCol0 = HIPVXMIN3(*(pSrcImage + srcIdx - 1), *(pSrcImage + srcIdxTopRow - 1), *(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      valCol2 = HIPVXMIN3(*(pSrcImage + srcIdx + 1), *(pSrcImage + srcIdxTopRow + 1), *(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)HIPVXMIN3(valCol0, valCol1, valCol2);
}
int HipExec_Erode_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Erode_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Erode_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMedian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Median_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    int pixelArr[9] = {0,0,0,0,0,0,0,0,0};
    pixelArr[1] = (int) *(pSrcImage + srcIdxTopRow);
    pixelArr[4] = (int) *(pSrcImage + srcIdx);
    pixelArr[7] = (int) *(pSrcImage + srcIdxBottomRow);
    if (x != 0)
    {
      pixelArr[0] = (int) *(pSrcImage + srcIdxTopRow - 1);
      pixelArr[3] = (int) *(pSrcImage + srcIdx - 1);
      pixelArr[6] = (int) *(pSrcImage + srcIdxBottomRow - 1);
    }
    if (x != (dstWidth - 1))
    {
      pixelArr[2] = (int) *(pSrcImage + srcIdxTopRow + 1);
      pixelArr[5] = (int) *(pSrcImage + srcIdx + 1);
      pixelArr[8] = (int) *(pSrcImage + srcIdxBottomRow + 1);
    }
    int i, j, min_idx;
    int n = 9;
    for (i = 0; i < n-1; i++)  
    {  
      min_idx = i;
      for (j = i+1; j < n; j++)  
      if (pixelArr[j] < pixelArr[min_idx])  
        min_idx = j;
      int temp = *(pixelArr + min_idx);
      *(pixelArr + min_idx) = *(pixelArr + i);
      *(pixelArr + i) = temp;
    }
    pDstImage[dstIdx] = (unsigned char)pixelArr[4];
}
int HipExec_Median_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Median_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Median_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxGaussian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Gaussian_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    int sum = 0;
    sum += ((int)*(pSrcImage + srcIdx) + (int)*(pSrcImage + srcIdxTopRow) + (int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum += ((int)*(pSrcImage + srcIdx - 1) + (int)*(pSrcImage + srcIdxTopRow - 1) + (int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += ((int)*(pSrcImage + srcIdx + 1) + (int)*(pSrcImage + srcIdxTopRow + 1) + (int)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)((float)sum / 9);
}
int HipExec_Gaussian_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Gaussian_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Gaussian_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxConvolve kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxLinearFilter kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxSobel kernels for hip backend
// ----------------------------------------------------------------------------










