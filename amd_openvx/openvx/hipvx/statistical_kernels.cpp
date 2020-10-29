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

#define PIXELTHRESHOLDBINARY(pixel, thresholdValue)     ((pixel > thresholdValue) ? 255 : 0)
#define PIXELTHRESHOLDRANGE(pixel, thresholdLower, thresholdUpper)     ((pixel > thresholdUpper) ? 0 : ((pixel < thresholdLower) ? 0 : 255))
#define PIXELTHRESHOLDNOTBINARY(pixel, thresholdValue)     ((pixel > thresholdValue) ? 0 : 255)
#define PIXELTHRESHOLDNOTRANGE(pixel, thresholdLower, thresholdUpper)     ((pixel > thresholdUpper) ? 255 : ((pixel < thresholdLower) ? 255 : 0))
#define PIXELCHECKU1(pixel) (pixel == (vx_int32)0) ? ((vx_uint32)0) : ((vx_uint32)1)

__device__ __forceinline__ int4 uchars_to_int4(uint src)
{
    return make_int4((int)(src&0xFF), (int)((src&0xFF00)>>8), (int)((src&0xFF0000)>>16), (int)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint int4_to_uchars(int4 src)
{
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

// ----------------------------------------------------------------------------
// VxThreshold kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdValue
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4((int)PIXELTHRESHOLDBINARY(src1.x,thresholdValue),(int)PIXELTHRESHOLDBINARY(src1.y,thresholdValue),
                        (int)PIXELTHRESHOLDBINARY(src1.z,thresholdValue),(int)PIXELTHRESHOLDBINARY(src1.w,thresholdValue));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Threshold_U8_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Threshold_U8_U8_Binary: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdLower, int thresholdUpper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4((int)PIXELTHRESHOLDRANGE(src1.x,thresholdLower,thresholdUpper),(int)PIXELTHRESHOLDRANGE(src1.y,thresholdLower,thresholdUpper),
                        (int)PIXELTHRESHOLDRANGE(src1.z,thresholdLower,thresholdUpper),(int)PIXELTHRESHOLDRANGE(src1.w,thresholdLower,thresholdUpper));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Threshold_U8_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Threshold_U8_U8_Range: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdValue
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4(PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.x,thresholdValue)),PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.y,thresholdValue)),
                        PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.z,thresholdValue)),PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.w,thresholdValue)));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Threshold_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Threshold_U1_U8_Binary: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdLower, int thresholdUpper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4(PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.x,thresholdLower,thresholdUpper)),PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.y,thresholdLower,thresholdUpper)),
                        PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.z,thresholdLower,thresholdUpper)),PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.w,thresholdLower,thresholdUpper)));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Threshold_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Threshold_U1_U8_Range: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ThresholdNot_U8_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdValue
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4((int)PIXELTHRESHOLDNOTBINARY(src1.x,thresholdValue),(int)PIXELTHRESHOLDNOTBINARY(src1.y,thresholdValue),
                        (int)PIXELTHRESHOLDNOTBINARY(src1.z,thresholdValue),(int)PIXELTHRESHOLDNOTBINARY(src1.w,thresholdValue));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_ThresholdNot_U8_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ThresholdNot_U8_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ThresholdNot_U8_U8_Binary: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ThresholdNot_U8_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdLower, int thresholdUpper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4((int)PIXELTHRESHOLDNOTRANGE(src1.x,thresholdLower,thresholdUpper),(int)PIXELTHRESHOLDNOTRANGE(src1.y,thresholdLower,thresholdUpper),
                        (int)PIXELTHRESHOLDNOTRANGE(src1.z,thresholdLower,thresholdUpper),(int)PIXELTHRESHOLDNOTRANGE(src1.w,thresholdLower,thresholdUpper));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_ThresholdNot_U8_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ThresholdNot_U8_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ThresholdNot_U8_U8_Range: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ThresholdNot_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdValue
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4(PIXELCHECKU1((int)PIXELTHRESHOLDNOTBINARY(src1.x,thresholdValue)),PIXELCHECKU1((int)PIXELTHRESHOLDNOTBINARY(src1.y,thresholdValue)),
                        PIXELCHECKU1((int)PIXELTHRESHOLDNOTBINARY(src1.z,thresholdValue)),PIXELCHECKU1((int)PIXELTHRESHOLDNOTBINARY(src1.w,thresholdValue)));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_ThresholdNot_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ThresholdNot_U1_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ThresholdNot_U1_U8_Binary: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ThresholdNot_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdLower, int thresholdUpper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 dst = make_int4(PIXELCHECKU1((int)PIXELTHRESHOLDNOTRANGE(src1.x,thresholdLower,thresholdUpper)),PIXELCHECKU1((int)PIXELTHRESHOLDNOTRANGE(src1.y,thresholdLower,thresholdUpper)),
                        PIXELCHECKU1((int)PIXELTHRESHOLDNOTRANGE(src1.z,thresholdLower,thresholdUpper)),PIXELCHECKU1((int)PIXELTHRESHOLDNOTRANGE(src1.w,thresholdLower,thresholdUpper)));
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_ThresholdNot_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ThresholdNot_U1_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_ThresholdNot_U1_U8_Range: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxIntegralImage kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxHistogram kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxMeanStdDev kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxMinMax kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxEqualize kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxHistogramMerge kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxMeanStdDevMerge kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxMinMaxLoc kernels for hip backend
// ----------------------------------------------------------------------------


