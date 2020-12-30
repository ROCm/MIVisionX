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
#define PIXELCHECKU1(pixel) (pixel == (vx_int32)0) ? ((vx_uint32)0) : ((vx_uint32)1)

__device__ __forceinline__ int4 uchars_to_int4(uint src)
{
    return make_int4((int)(src&0xFF), (int)((src&0xFF00)>>8), (int)((src&0xFF0000)>>16), (int)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint int4_to_uchars(int4 src)
{
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ void prefixSum(unsigned int* output, unsigned int* input, int w, int nextpow2)
{
    extern __shared__ int temp[];

    const int tdx = threadIdx.x;
    int offset = 1;
    const int tdx2 = 2*tdx;
    const int tdx2p = tdx2 + 1;

    temp[tdx2] =  tdx2 < w ? input[tdx2] : 0;
    temp[tdx2p] = tdx2p < w ? input[tdx2p] : 0;

    for(int d = nextpow2>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    int last = temp[nextpow2 - 1];
    if(tdx == 0) temp[nextpow2 - 1] = 0;

    for(int d = 1; d < nextpow2; d *= 2) {
        offset >>= 1;

        __syncthreads();

        if(tdx < d )
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            int t  = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if(tdx2 < w)  output[tdx2 - 1] = temp[tdx2];
    if(tdx2p < w) output[tdx2p - 1] = temp[tdx2p];
    if(tdx2p < w) output[w - 1] = last;
}

__device__ __forceinline__ int pixelcheckU1(uint bit, uint pixel)
{
    return (bit ? 255 : pixel);
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
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
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
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes,
    int thresholdValue
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + (x*2);
    int4 src1 = uchars_to_int4(pSrcImage[srcIdx]);
    int4 src2 = uchars_to_int4(pSrcImage[srcIdx + 1]);
    pDstImage[dstIdx] =  PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.x,thresholdValue)) | (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.y,thresholdValue)) << 1) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.z,thresholdValue)) << 2) | (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src1.w,thresholdValue)) << 3) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src2.x,thresholdValue)) << 4) | (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src2.y,thresholdValue)) << 5) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src2.z,thresholdValue)) << 6) | (PIXELCHECKU1((int)PIXELTHRESHOLDBINARY(src2.w,thresholdValue)) << 7) ;
}
int HipExec_Threshold_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    )
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes,
    int thresholdLower, int thresholdUpper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + (x*2);
    int4 src1 = uchars_to_int4(pSrcImage[srcIdx]);
    int4 src2 = uchars_to_int4(pSrcImage[srcIdx + 1]);
    pDstImage[dstIdx] =  PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.x,thresholdLower,thresholdUpper)) | (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.y,thresholdLower,thresholdUpper)) << 1) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.z,thresholdLower,thresholdUpper)) << 2) | (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src1.w,thresholdLower,thresholdUpper)) << 3) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src2.x,thresholdLower,thresholdUpper)) << 4) | (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src2.y,thresholdLower,thresholdUpper)) << 5) |
                        (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src2.z,thresholdLower,thresholdUpper)) << 6) | (PIXELCHECKU1((int)PIXELTHRESHOLDRANGE(src2.w,thresholdLower,thresholdUpper)) << 7) ;
}
int HipExec_Threshold_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    )
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMinMaxLoc kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_MinMax_DATA_U8(
    vx_int32  * pDstMinValue, vx_int32  * pDstMaxValue,
    vx_uint32  srcWidth, vx_uint32  srcHeight,
    unsigned char *pSrcImage, vx_uint32 srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= srcWidth) || (y >= srcHeight)) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*4);
    for(int i=0; i<4; i++)
    {
        atomicMin(pDstMinValue, (int)pSrcImage[srcIdx + i]);
        atomicMax(pDstMaxValue, (int)pSrcImage[srcIdx + i]);
    }
}
int HipExec_MinMax_DATA_U8(
    vx_int32    * pHipDstMinValue, vx_int32    * pHipDstMaxValue,
    vx_uint32     srcWidth,  vx_uint32     srcHeight,
    vx_uint8    * pHipSrcImage, vx_uint32     srcImageStrideInBytes
    )
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (srcWidth+3)>>2, globalThreads_y = srcHeight;

    vx_int32 *dstMinVal, *dstMaxVal;
    hipMalloc((void**)&dstMinVal, sizeof(vx_int32));
    // hipMemset(dstMinVal, 255, sizeof(vx_int32));
    hipMalloc((void**)&dstMaxVal, sizeof(vx_int32));
    // hipMemset(dstMaxVal, 0, sizeof(vx_int32));

    hipLaunchKernelGGL(Hip_MinMax_DATA_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstMinVal, dstMaxVal, srcWidth, srcHeight,
                    (unsigned char*)pHipSrcImage , srcImageStrideInBytes);

    hipMemcpyDtoH(pHipDstMinValue, dstMinVal, sizeof(vx_int32));
    hipMemcpyDtoH(pHipDstMinValue, dstMaxVal, sizeof(vx_int32));

    return VX_SUCCESS;
}
// ----------------------------------------------------------------------------
// VxMeanStdDev kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_MeanStdDev_DATA_U8(
    vx_float32  * pSum, vx_float32  * pSumOfSquared,
    vx_uint32  srcWidth, vx_uint32  srcHeight,
    unsigned char *pSrcImage, vx_uint32 srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= srcWidth) || (y >= srcHeight)) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*4);
    for(int i=0; i<4; i++)
    {
        atomicAdd(pSum, (float)pSrcImage[srcIdx + i]);
        atomicAdd(pSumOfSquared, (float)pSrcImage[srcIdx + i] * (float)pSrcImage[srcIdx + i]);
    }
}
int HipExec_MeanStdDev_DATA_U8(
    vx_float32  * pHipSum,  vx_float32  * pHipSumOfSquared,
    vx_uint32     srcWidth,  vx_uint32     srcHeight,
    vx_uint8    * pHipSrcImage, vx_uint32     srcImageStrideInBytes
    )
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (srcWidth+3)>>2, globalThreads_y = srcHeight;

    vx_float32 *Sum, *SumOfSquared;
    hipMalloc((void**)&Sum, sizeof(vx_float32));
    hipMalloc((void**)&SumOfSquared, sizeof(vx_float32));

    hipLaunchKernelGGL(Hip_MeanStdDev_DATA_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, Sum, SumOfSquared, srcWidth, srcHeight,
                    (unsigned char*)pHipSrcImage , srcImageStrideInBytes);

    hipMemcpyDtoH(pHipSum, Sum, sizeof(vx_int32));
    hipMemcpyDtoH(pHipSumOfSquared, SumOfSquared, sizeof(vx_int32));

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxIntegralImage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_ConvertDepth(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx =  y*srcImageStrideInBytes + x;
    pDstImage[dstIdx] = (unsigned int) pSrcImage[srcIdx];
}
__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_PrefixSumRows(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int row = hipBlockIdx_y;
    prefixSum(pDstImage+row*(dstImageStrideInBytes>>2), pSrcImage+row*(srcImageStrideInBytes>>2), dstWidth, 2*hipBlockDim_x );
}
__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_PrefixSumRowsTrans(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    int row = hipBlockIdx_y;
    prefixSum(pDstImage+row*(dstWidth+1), pSrcImage+row*dstWidth, dstWidth, 2*hipBlockDim_x );
}
__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_Transpose(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
	)
{
    __shared__ int temp[16][16];
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if((x < dstWidth) && (y < dstHeight))
    {
        int id_in = y * (srcImageStrideInBytes>>2) + x;
        temp[hipThreadIdx_y][hipThreadIdx_x] = pSrcImage[id_in];
    }
    __syncthreads();
    x = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_x;
    y = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_y;
    if((x < dstHeight) && (y < dstWidth))
    {
        int id_out = y * dstHeight + x;
        pDstImage[id_out] = temp[hipThreadIdx_x][hipThreadIdx_y];
    }
}
int HipExec_IntegralImage_U32_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint32 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x_ConvertDepth = 16, localThreads_y_ConvertDepth = 16;
    int globalThreads_x_ConvertDepth = dstWidth,   globalThreads_y_ConvertDepth = dstHeight;
    
    int localThreads_x_PrefixSumRows = 16, localThreads_y_PrefixSumRows = 1;
    int globalThreads_x_PrefixSumRows = 16,   globalThreads_y_PrefixSumRows = dstHeight;

    int localThreads_x_Transpose1 = 16, localThreads_y_Transpose1 = 16;
    int globalThreads_x_Transpose1 = dstWidth + localThreads_x_Transpose1 - 1,   globalThreads_y_Transpose1 = dstHeight + localThreads_y_Transpose1 - 1;
    
    int localThreads_x_PrefixSumRowsTrans = 16, localThreads_y_PrefixSumRowsTrans = 1;
    int globalThreads_x_PrefixSumRowsTrans = 16,   globalThreads_y_PrefixSumRowsTrans = dstWidth;

    int localThreads_x_Transpose2 = 16, localThreads_y_Transpose2 = 16;
    int globalThreads_x_Transpose2 = dstWidth+1 + localThreads_x_Transpose2 - 1,   globalThreads_y_Transpose2 = dstHeight+1 + localThreads_y_Transpose2 - 1;
    
    unsigned int *pHipDstImageBuffer, *pHipDstImageBufferT;
    vx_uint32 dstImageBufferStrideInBytes = dstImageStrideInBytes + 4;
    hipMalloc(&pHipDstImageBuffer, (dstHeight + 1) * dstImageBufferStrideInBytes * sizeof(unsigned int));
    hipMalloc(&pHipDstImageBufferT, (dstHeight + 1) * dstImageBufferStrideInBytes * sizeof(unsigned int));
    hipMemset(pHipDstImageBuffer, 0, (dstHeight + 1) * dstImageBufferStrideInBytes * sizeof(unsigned int));
    hipMemset(pHipDstImageBufferT, 0, (dstHeight + 1) * dstImageBufferStrideInBytes * sizeof(unsigned int));
    
    printf("\n\ndstImageBufferStrideInBytes, dstImageStrideInBytes, srcImageStrideInBytes = %d, %d, %d", dstImageBufferStrideInBytes, dstImageStrideInBytes, srcImageStrideInBytes);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_ConvertDepth,
                dim3(ceil((float)globalThreads_x_ConvertDepth/localThreads_x_ConvertDepth), ceil((float)globalThreads_y_ConvertDepth/localThreads_y_ConvertDepth)),
                dim3(localThreads_x_ConvertDepth, localThreads_y_ConvertDepth),
                0, 0, dstWidth, dstHeight,
                (unsigned int *)pHipDstImage, dstImageStrideInBytes, 
                (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_PrefixSumRows,
                    dim3(ceil((float)globalThreads_x_PrefixSumRows/localThreads_x_PrefixSumRows), ceil((float)globalThreads_y_PrefixSumRows/localThreads_y_PrefixSumRows)),
                    dim3(localThreads_x_PrefixSumRows, localThreads_y_PrefixSumRows),
                    2 * sizeof(int) * localThreads_x_PrefixSumRows, 0, 
                    dstWidth, dstHeight,
                    (unsigned int *)pHipDstImageBuffer, dstImageStrideInBytes, 
                    (unsigned int *)pHipDstImage, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_Transpose,
                    dim3(ceil((float)globalThreads_x_Transpose1/localThreads_x_Transpose1), ceil((float)globalThreads_y_Transpose1/localThreads_y_Transpose1)),
                    dim3(localThreads_x_Transpose1, localThreads_y_Transpose1),
                    0, 0, 
                    dstWidth, dstHeight,
                    (unsigned int *)pHipDstImageBufferT, dstImageStrideInBytes, 
                    (unsigned int *)pHipDstImageBuffer, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipMemset(pHipDstImageBuffer, 0, (dstHeight + 1) * sizeof(int));
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_PrefixSumRowsTrans,
                    dim3(ceil((float)globalThreads_x_PrefixSumRowsTrans/localThreads_x_PrefixSumRowsTrans), ceil((float)globalThreads_y_PrefixSumRowsTrans/localThreads_y_PrefixSumRowsTrans)),
                    dim3(localThreads_x_PrefixSumRowsTrans, localThreads_y_PrefixSumRowsTrans),
                    2 * sizeof(int) * localThreads_x_PrefixSumRowsTrans, 0, 
                    dstHeight, dstWidth, 
                    (unsigned int *)pHipDstImageBuffer, dstImageBufferStrideInBytes, 
                    (unsigned int *)pHipDstImageBufferT, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_Transpose,
                    dim3(ceil((float)globalThreads_x_Transpose2/localThreads_x_Transpose2), ceil((float)globalThreads_y_Transpose2/localThreads_y_Transpose2)),
                    dim3(localThreads_x_Transpose2, localThreads_y_Transpose2),
                    0, 0, 
                    dstHeight+1, dstWidth+1, 
                    (unsigned int *)pHipDstImageBufferT, dstImageBufferStrideInBytes, 
                    (unsigned int *)pHipDstImageBuffer, (vx_uint32)((dstHeight+1)*sizeof(unsigned int)));
    hipDeviceSynchronize();
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    // hipMemcpy(pHipDstImage, pHipDstImageBuffer, dstHeight * dstImageStrideInBytes * sizeof(unsigned int), hipMemcpyDeviceToDevice);
    hipMemcpy(pHipDstImage, pHipDstImageBufferT, dstHeight * dstImageStrideInBytes * sizeof(unsigned int), hipMemcpyDeviceToDevice);
    hipFree(&pHipDstImageBuffer);
    hipFree(&pHipDstImageBufferT);

    printf("\nHipExec_IntegralImage_U32_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxHistogram kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxEqualize kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxHistogramMerge kernels for hip backend
// ----------------------------------------------------------------------------



