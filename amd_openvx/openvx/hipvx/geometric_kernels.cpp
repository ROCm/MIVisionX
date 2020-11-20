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

// __device__ __forceinline__ float4 ucharTofloat4(unsigned int src)
// {
//     return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
// }

// __device__ __forceinline__ uint float4ToUint(float4 src)
// {
//   return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
// }

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELROUNDF32(value)        ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))

// ----------------------------------------------------------------------------
// VxRemap kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxWarpAffine kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxWarpPerspective kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxScaleImage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Nearest(
    const float dstWidth, const float dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight, 
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int xSrc = (int)PIXELROUNDF32(((x + 0.5) * (srcWidth/dstWidth)) - 0.5);
    int ySrc = (int)PIXELROUNDF32(((y + 0.5) * (srcHeight/dstHeight)) - 0.5);
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrc*(srcImageStrideInBytes) + xSrc;
    pDstImage[dstIdx] = pSrcImage[srcIdx];
}
int HipExec_ScaleImage_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Nearest,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, 
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ScaleImage_U8_U8_Nearest: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear(
    const float dstWidth, const float dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight, 
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    float xSrcFloat = ((x + 0.5) * (srcWidth/dstWidth)) - 0.5;
    float ySrcFloat = ((y + 0.5) * (srcHeight/dstHeight)) - 0.5;
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    float s = xSrcFloat - xSrcLower;
    float t = ySrcFloat - ySrcLower;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
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
int HipExec_ScaleImage_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, 
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ScaleImage_U8_U8_Bilinear: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area(
    const float dstWidth, const float dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight, 
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;

    int xSrcLow = (int)(((float)x - (srcWidth / dstWidth)) - 0.5);
    int ySrcLow = (int)(((float)y - (srcHeight / dstHeight)) - 0.5);
    int xSrcHigh = (int)(((float)(x+1) - (srcWidth / dstWidth)) - 0.5);
    int ySrcHigh = (int)(((float)(y+1) - (srcHeight / dstHeight)) - 0.5);

    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrcLow * (srcImageStrideInBytes) + xSrcLow;
    int srcIdxRow = srcIdx;
    int srcIdxCol = srcIdxRow;

    int sum = 0, count = 0;
    for (int y = ySrcLow; y < ySrcHigh; y++)
    {
        for (int x = xSrcLow; x < xSrcHigh; x++)
        {
            sum += pSrcImage[srcIdxCol];
            srcIdxCol += 1;
            count += 1;
        }
        srcIdxRow += srcImageStrideInBytes;
        srcIdxCol = srcIdxRow;
    }

    // int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);


    // float xSrcFloat = ((x + 0.5) * (srcWidth/dstWidth)) - 0.5;
    // float ySrcFloat = ((y + 0.5) * (srcHeight/dstHeight)) - 0.5;
    // int xSrcLower = (int)xSrcFloat;
    // int ySrcLower = (int)ySrcFloat;
    // float s = xSrcFloat - xSrcLower;
    // float t = ySrcFloat - ySrcLower;
    // int dstIdx =  y*(dstImageStrideInBytes) + x;
    // int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
    // int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
    // int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
    // int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
    // pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
    //   (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
    //   (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
    //   (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
    //   (s) * (t) * pSrcImage[srcIdxBottomRight]
    // );
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8((float)sum / (float)count);
}
int HipExec_ScaleImage_U8_U8_Area(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, 
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ScaleImage_U8_U8_Area: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}