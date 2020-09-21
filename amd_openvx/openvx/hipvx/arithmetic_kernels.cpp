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

// VxAbsDiff kernel for hip backend
__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_U8_U8U8
	(
		vx_uint8     * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		const vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		const vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	)
{
    size_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    pSrcImage1 += (y*srcImage1StrideInBytes + x*4);
    pSrcImage2 += (y*srcImage2StrideInBytes + x*4);
    unsigned int dstIdx =  y*dstImageStrideInBytes + x;
    float4 src1 = make_float4(pSrcImage1[0], pSrcImage1[1], pSrcImage1[2], pSrcImage1[3]);
    float4 src2 = make_float4(pSrcImage2[0], pSrcImage2[1], pSrcImage2[2], pSrcImage2[3]);
    float4 dst = make_float4(fabsf(src1.x-src2.x), fabsf(src1.y-src2.y), fabsf(src1.x-src2.x), fabsf(src1.x-src2.x));
    *((unsigned int *)pDstImage + dstIdx) = (uchar)dst.x | ((uchar)dst.y<<8) | ((uchar)dst.z<<16)| ((uchar)dst.w << 24);
}

int HipExec_AbsDiff_U8_U8U8
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

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(Hip_AbsDiff_U8_U8U8,
                    dim3(globalThreads_x/localThreads_x, globalThreads_y/localThreads_y),
                    dim3(localThreads_x, localThreads_y),
                    0, 0,
                    pHipDstImage , dstImageStrideInBytes, pHipSrcImage1, srcImage1StrideInBytes,
                    pHipSrcImage2, srcImage2StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    return VX_SUCCESS;
}
