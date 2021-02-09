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
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_kernels.h"

__global__ void __attribute__((visibility("default")))
Hip_Copy_U8_U8 (
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        unsigned int     * pDstImage,
        unsigned int     dstImageStrideInBytes,
        const unsigned int    * pSrcImage,
        vx_uint32     srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + (x*2);
    pDstImage[dstIdx] = pSrcImage[srcIdx];
    pDstImage[dstIdx+1] = pSrcImage[srcIdx+1];
}
int HipExec_ChannelCopy (
        hipStream_t  stream,
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        vx_uint8     * pHipDstImage,
        vx_uint32     dstImageStrideInBytes,
        const vx_uint8    * pHipSrcImage,
        vx_uint32     srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3, globalThreads_y = dstHeight;
#if 1    
    hipLaunchKernelGGL(Hip_Copy_U8_U8,
                       dim3(ceil((float) globalThreads_x / localThreads_x), ceil((float) globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *) pHipDstImage, dstImageStrideInBytes, (const unsigned int *) pHipSrcImage,
                       srcImageStrideInBytes);
#endif
    return VX_SUCCESS;
}
