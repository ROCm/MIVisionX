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

#ifndef MIVISIONX_HIP_KERNELS_H
#define MIVISIONX_HIP_KERNELS_H

#include <VX/vx.h>

#define ENABLE_KERNEL_EVENT_TIMING    0

#if ENABLE_KERNEL_EVENT_TIMING
#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
        #define __FUNCTION_NAME__   __FUNCTION__
#else          //*NIX
        #define __FUNCTION_NAME__   __func__
#endif
#endif //#ifndef __FUNCTION_NAME__
#define HIP_KERNEL_TIMING_START(start, stop, t)   { hipEventCreate(&start); hipEventCreate(&stop); t = 1.0f; hipEventRecord(start, NULL); }
#define HIP_KERNEL_TIMING_STOP(start, stop, t)   { hipEventRecord(stop, NULL); hipEventSynchronize(stop);  hipEventElapsedTime(&eventMs, start, stop); printf("%s: Kernel time: %f ms\n", __FUNCTION_NAME__, t);}
#else
#define HIP_KERNEL_TIMING_START(start, stop, t);
#define HIP_KERNEL_TIMING_STOP(start, stop, t);
#endif

int HipExec_AbsDiff_U8_U8U8
(
        hipStream_t  stream,
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        vx_uint8     * pHipDstImage,
        vx_uint32     dstImageStrideInBytes,
        const vx_uint8    * pHipSrcImage1,
        vx_uint32     srcImage1StrideInBytes,
        const vx_uint8    * pHipSrcImage2,
        vx_uint32     srcImage2StrideInBytes
);
int HipExec_ChannelCopy
(
        hipStream_t  stream,
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        vx_uint8     * pHipDstImage,
        vx_uint32     dstImageStrideInBytes,
        const vx_uint8    * pHipSrcImage,
        vx_uint32     srcImageStrideInBytes
);

#endif //MIVISIONX_HIP_KERNELS_H
