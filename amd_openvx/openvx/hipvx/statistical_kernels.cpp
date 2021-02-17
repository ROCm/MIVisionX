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



#include "hip_common.h"
#include "hip_host_decls.h"

// ----------------------------------------------------------------------------
// VxThreshold kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    int thresholdValue
	) {
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
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
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
	) {
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
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdLower, thresholdUpper);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Binary(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    int thresholdValue
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*8);
    for(int bit=0; bit < 8; bit++)
    {
        if(((x*8) + bit) >= dstWidth)   return;
        pDstImage[dstIdx] >>= 1;
        if (PIXELTHRESHOLDBINARY((int)pSrcImage[srcIdx + bit],thresholdValue))
            pDstImage[dstIdx] |= 0x80;
    }
}
int HipExec_Threshold_U1_U8_Binary(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdValue
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Binary,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    thresholdValue);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Range(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    int thresholdLower, int thresholdUpper
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*8);
    for(int bit=0; bit < 8; bit++)
    {
        if(((x*8) + bit) >= dstWidth)   return;
        pDstImage[dstIdx] >>= 1;
        if (PIXELTHRESHOLDRANGE((int)pSrcImage[srcIdx + bit],thresholdLower, thresholdUpper))
            pDstImage[dstIdx] |= 0x80;
    }

}
int HipExec_Threshold_U1_U8_Range(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Range,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
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
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= srcWidth) || (y >= srcHeight)) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*4);
    for(int i=0; i<4; i++) {
        atomicMin(pDstMinValue, (int)pSrcImage[srcIdx + i]);
        atomicMax(pDstMaxValue, (int)pSrcImage[srcIdx + i]);
    }
}
int HipExec_MinMax_DATA_U8(
    hipStream_t stream, vx_int32 * pHipDstMinValue, vx_int32 * pHipDstMaxValue,
    vx_uint32     srcWidth,  vx_uint32     srcHeight,
    vx_uint8    * pHipSrcImage, vx_uint32     srcImageStrideInBytes
    ) {
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
                    0, stream, dstMinVal, dstMaxVal, srcWidth, srcHeight,
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
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= srcWidth) || (y >= srcHeight)) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + (x*4);
    for(int i=0; i<4; i++) {
        atomicAdd(pSum, (float)pSrcImage[srcIdx + i]);
        atomicAdd(pSumOfSquared, (float)pSrcImage[srcIdx + i] * (float)pSrcImage[srcIdx + i]);
    }
}
int HipExec_MeanStdDev_DATA_U8(
    hipStream_t stream, vx_float32  * pHipSum,  vx_float32  * pHipSumOfSquared,
    vx_uint32     srcWidth,  vx_uint32     srcHeight,
    vx_uint8    * pHipSrcImage, vx_uint32     srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (srcWidth+3)>>2, globalThreads_y = srcHeight;
    vx_float32 *Sum, *SumOfSquared;
    hipMalloc((void**)&Sum, sizeof(vx_float32));
    hipMalloc((void**)&SumOfSquared, sizeof(vx_float32));

    hipLaunchKernelGGL(Hip_MeanStdDev_DATA_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, Sum, SumOfSquared, srcWidth, srcHeight,
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
	) {
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
    ) {
    int row = hipBlockIdx_y;
    prefixSum(pDstImage+row*(dstImageStrideInBytes>>2), pSrcImage+row*(srcImageStrideInBytes>>2), dstWidth, 2*hipBlockDim_x );
}

__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_PrefixSumRowsTrans(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int row = hipBlockIdx_y;
    prefixSum(pDstImage+row*(dstWidth+1), pSrcImage+row*dstWidth, dstWidth, 2*hipBlockDim_x );
}
__global__ void __attribute__((visibility("default")))
Hip_IntegralImage_U32_U8_Transpose(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageHeightStrideInPixels,
    unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    __shared__ int temp[16][16];
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if((x < dstWidth) && (y < dstHeight)) {
        int id_in = y * (srcImageStrideInBytes>>2) + x;
        temp[hipThreadIdx_y][hipThreadIdx_x] = pSrcImage[id_in];
    }
    __syncthreads();
    x = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_x;
    y = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_y;
    if((x < dstHeight) && (y < dstWidth)) {
        int id_out = y * dstImageHeightStrideInPixels + x;
        pDstImage[id_out] = temp[hipThreadIdx_x][hipThreadIdx_y];
    }
}
int HipExec_IntegralImage_U32_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint32 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
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
    
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_ConvertDepth,
                dim3(ceil((float)globalThreads_x_ConvertDepth/localThreads_x_ConvertDepth), ceil((float)globalThreads_y_ConvertDepth/localThreads_y_ConvertDepth)),
                dim3(localThreads_x_ConvertDepth, localThreads_y_ConvertDepth),
                0, stream, dstWidth, dstHeight,
                (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_PrefixSumRows,
                    dim3(ceil((float)globalThreads_x_PrefixSumRows/localThreads_x_PrefixSumRows), ceil((float)globalThreads_y_PrefixSumRows/localThreads_y_PrefixSumRows)),
                    dim3(localThreads_x_PrefixSumRows, localThreads_y_PrefixSumRows),
                    2 * sizeof(int) * localThreads_x_PrefixSumRows, stream, // TODO
                    dstWidth, dstHeight,
                    (unsigned int *)pHipDstImageBuffer, dstImageStrideInBytes,
                    (unsigned int *)pHipDstImage, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_Transpose,
                    dim3(ceil((float)globalThreads_x_Transpose1/localThreads_x_Transpose1), ceil((float)globalThreads_y_Transpose1/localThreads_y_Transpose1)),
                    dim3(localThreads_x_Transpose1, localThreads_y_Transpose1),
                    0, stream,
                    dstWidth, dstHeight,
                    (unsigned int *)pHipDstImageBufferT, dstHeight,
                    (unsigned int *)pHipDstImageBuffer, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipMemset(pHipDstImageBuffer, 0, (dstHeight + 1) * dstImageBufferStrideInBytes * sizeof(int));
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_PrefixSumRowsTrans,
                    dim3(ceil((float)globalThreads_x_PrefixSumRowsTrans/localThreads_x_PrefixSumRowsTrans), ceil((float)globalThreads_y_PrefixSumRowsTrans/localThreads_y_PrefixSumRowsTrans)),
                    dim3(localThreads_x_PrefixSumRowsTrans, localThreads_y_PrefixSumRowsTrans),
                    2 * sizeof(int) * localThreads_x_PrefixSumRowsTrans, stream,  // TODO
                    dstHeight, dstWidth,
                    (unsigned int *)pHipDstImageBuffer, dstImageBufferStrideInBytes,
                    (unsigned int *)pHipDstImageBufferT, dstImageStrideInBytes);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(Hip_IntegralImage_U32_U8_Transpose,
                    dim3(ceil((float)globalThreads_x_Transpose2/localThreads_x_Transpose2), ceil((float)globalThreads_y_Transpose2/localThreads_y_Transpose2)),
                    dim3(localThreads_x_Transpose2, localThreads_y_Transpose2),
                    0, stream,
                    dstHeight+1, dstWidth+1,
                    (unsigned int *)pHipDstImage, dstImageStrideInBytes>>2,
                    (unsigned int *)pHipDstImageBuffer, (vx_uint32)((dstHeight+1)*sizeof(unsigned int)));
    hipDeviceSynchronize();
    // hipMemcpy(pHipDstImage, pHipDstImageBufferT, dstHeight * dstImageStrideInBytes * sizeof(unsigned int), hipMemcpyDeviceToDevice);
    hipFree(&pHipDstImageBuffer);
    hipFree(&pHipDstImageBufferT);

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



