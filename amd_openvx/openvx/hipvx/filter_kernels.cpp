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

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
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
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *gaussian
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
    float sum = 0;
    sum += (gaussian[4] * (float)*(pSrcImage + srcIdx) + gaussian[1] * (float)*(pSrcImage + srcIdxTopRow) + gaussian[7] * (float)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum += (gaussian[3] * (float)*(pSrcImage + srcIdx - 1) + gaussian[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + gaussian[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += (gaussian[5] * (float)*(pSrcImage + srcIdx + 1) + gaussian[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + gaussian[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
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

    float gaussian[9] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
    float *hipGaussian;
    hipMalloc(&hipGaussian, 288);
    hipMemcpy(hipGaussian, gaussian, 288, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Gaussian_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes,
                    (const float *)hipGaussian);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGaussian);

    printf("\nHipExec_Gaussian_U8_U8_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxConvolve kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv, 
    const unsigned int convolutionWidth, const unsigned int convolutionHeight
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int bound = convolutionHeight / 2;
    int dstIdx =  y*dstImageStrideInBytes + x;
    int srcIdx =  (y - bound)*(srcImageStrideInBytes) + (x - bound);
    int indices[25];
    short int sum = 0;
    for (int i = 0; i < convolutionHeight; i++)
    {
      indices[i] = srcIdx + (i * srcImageStrideInBytes);
    }
    for (int i = 0; i < convolutionWidth; i++)
    {
      if (x <= bound)
      {
        if ((i >= bound - x) && (i < convolutionHeight))
        {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else if (x >= dstWidth - bound)
      {
        if ((i >= 0) && (i < dstWidth - (x - bound)))
        {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else
      {
        for (int j = 0; j < convolutionHeight; j++)
          sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
      }
    }
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
}
int HipExec_Convolve_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    )
{
    hipEvent_t start, stop;
    int bound = convolutionHeight / 2;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - (2 * bound);

    float *hipConv;
    hipMalloc(&hipConv, 16 * convolutionWidth * convolutionHeight);
    hipMemcpy(hipConv, conv, 16 * convolutionWidth * convolutionHeight, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Convolve_U8_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv, convolutionWidth, convolutionHeight);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipConv);

    printf("\nHipExec_Convolve_U8_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv, 
    const unsigned int convolutionWidth, const unsigned int convolutionHeight
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int bound = convolutionHeight / 2;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  (y - bound)*(srcImageStrideInBytes) + (x - bound);
    if (y >= dstHeight - (2 * bound))
    {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    int indices[25];
    short int sum = 0;
    for (int i = 0; i < convolutionHeight; i++)
    {
      indices[i] = srcIdx + (i * srcImageStrideInBytes);
    }
    for (int i = 0; i < convolutionWidth; i++)
    {
      if (x <= bound)
      {
        if ((i >= bound - x) && (i < convolutionHeight))
        {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else if (x >= dstWidth - bound)
      {
        if ((i >= 0) && (i < dstWidth - (x - bound)))
        {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else
      {
        for (int j = 0; j < convolutionHeight; j++)
          sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
      }
    }
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum);
}
int HipExec_Convolve_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    )
{
    hipEvent_t start, stop;
    int bound = convolutionHeight / 2;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    float *hipConv;
    hipMalloc(&hipConv, 16 * convolutionWidth * convolutionHeight);
    hipMemcpy(hipConv, conv, 16 * convolutionWidth * convolutionHeight, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Convolve_S16_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv, convolutionWidth, convolutionHeight);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipConv);

    printf("\nHipExec_Convolve_S16_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxSobel kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16S16_U8_3x3_GXY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    short int *pDstImage1, unsigned int dstImage1StrideInBytes,
    short int *pDstImage2, unsigned int dstImage2StrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dst1Idx =  y*(dstImage1StrideInBytes>>1) + x;
    int dst2Idx =  y*(dstImage2StrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2)
    {
      pDstImage1[dst1Idx] = (short int)0;
      pDstImage2[dst2Idx] = (short int)0;
      return;
    }
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    
    short int sum1 = 0;
    sum1 += (gx[4] * (short int)*(pSrcImage + srcIdx) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow) + gx[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum1 += (gx[3] * (short int)*(pSrcImage + srcIdx - 1) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gx[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum1 += (gx[5] * (short int)*(pSrcImage + srcIdx + 1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gx[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    short int sum2 = 0;
    sum2 += (gy[4] * (short int)*(pSrcImage + srcIdx) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow) + gy[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum2 += (gy[3] * (short int)*(pSrcImage + srcIdx - 1) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gy[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum2 += (gy[5] * (short int)*(pSrcImage + srcIdx + 1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gy[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage1[dst1Idx] = (short int)PIXELSATURATES16(sum2);
    pDstImage2[dst2Idx] = (short int)PIXELSATURATES16(sum1);
}
int HipExec_Sobel_S16S16_U8_3x3_GXY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
    vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    short int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 144);
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Sobel_S16S16_U8_3x3_GXY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (short int *)pHipDstImage1 + (dstImage1StrideInBytes>>1) , dstImage1StrideInBytes, 
                    (short int *)pHipDstImage2 + (dstImage2StrideInBytes>>1) , dstImage2StrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_Sobel_S16S16_U8_3x3_GXY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GX(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2)
    {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    
    short int sum1 = 0;
    sum1 += (gx[4] * (short int)*(pSrcImage + srcIdx) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow) + gx[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum1 += (gx[3] * (short int)*(pSrcImage + srcIdx - 1) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gx[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum1 += (gx[5] * (short int)*(pSrcImage + srcIdx + 1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gx[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum1);
}
int HipExec_Sobel_S16_U8_3x3_GX(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    short int *hipGx;
    hipMalloc(&hipGx, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GX,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (short int *)pHipDstImage + (dstImageStrideInBytes>>1) , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes,
                    (const short int *)hipGx);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);

    printf("\nHipExec_Sobel_S16_U8_3x3_GX: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2)
    {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    
    short int sum2 = 0;
    sum2 += (gy[4] * (short int)*(pSrcImage + srcIdx) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow) + gy[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum2 += (gy[3] * (short int)*(pSrcImage + srcIdx - 1) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gy[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum2 += (gy[5] * (short int)*(pSrcImage + srcIdx + 1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gy[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum2);
}
int HipExec_Sobel_S16_U8_3x3_GY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    short int *hipGy;
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (short int *)pHipDstImage + (dstImageStrideInBytes>>1) , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes,
                    (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGy);

    printf("\nHipExec_Sobel_S16_U8_3x3_GY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}