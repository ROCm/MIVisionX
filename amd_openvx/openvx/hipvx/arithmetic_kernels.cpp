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

#include "hip_kernels.h"

// ----------------------------------------------------------------------------
// VxAbsDiff kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_U8_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((fabs4(unpack_(src1.x) - unpack_(src2.x))));
    dst.y = pack_((fabs4(unpack_(src1.y) - unpack_(src2.y))));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_AbsDiff_U8_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_U8_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_AbsDiff_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint  dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx = y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = min(abs(((src1.x << 16) >> 16) - ((src2.x << 16) >> 16)), 32767u);
    dst.x |= (int)min(abs((src1.x    >> 16) - ( src2.x        >> 16)), 32767u) << 16;
    dst.y  = min(abs(((src1.y << 16) >> 16) - ((src2.y << 16) >> 16)), 32767u);
    dst.y |= (int) min(abs((src1.y   >> 16) - ( src2.y        >> 16)), 32767u) << 16;
    dst.z  = min(abs(((src1.z << 16) >> 16) - ((src2.z << 16) >> 16)), 32767u);
    dst.z |= (int)min(abs(( src1.z   >> 16) - ( src2.z        >> 16)), 32767u) << 16;
    dst.w  = min(abs(((src1.w << 16) >> 16) - ((src2.w << 16) >> 16)), 32767u);
    dst.w |= (int)min(abs(( src1.w   >> 16) - ( src2.w        >> 16)), 32767u) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_AbsDiff_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_AbsDiff_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                    (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar*)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxAdd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = (src1.x +  src2.x              ) & 0x000000ff;
    dst.x |= (src1.x + (src2.x & 0x0000ff00)) & 0x0000ff00;
    dst.x |= (src1.x + (src2.x & 0x00ff0000)) & 0x00ff0000;
    dst.x |= (src1.x + (src2.x & 0xff000000)) & 0xff000000;
    dst.y  = (src1.y +  src2.y              ) & 0x000000ff;
    dst.y |= (src1.y + (src2.y & 0x0000ff00)) & 0x0000ff00;
    dst.y |= (src1.y + (src2.y & 0x00ff0000)) & 0x00ff0000;
    dst.y |= (src1.y + (src2.y & 0xff000000)) & 0xff000000;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_U8_U8U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_U8_U8U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((unpack_(src1.x) + unpack_(src2.x)));
    dst.y = pack_((unpack_(src1.y) + unpack_(src2.y)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_U8_U8U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_U8_U8U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((src1.x & 0x000000ff) + (src2.x & 0x000000ff));
    dst.x |= ((src1.x & 0x0000ff00) + (src2.x & 0x0000ff00)) <<  8;
    dst.y  = ((src1.x & 0x00ff0000) + (src2.x & 0x00ff0000)) >> 16;
    dst.y |= ((src1.x >>        24) + (src2.x >>        24)) << 16;
    dst.z  = ((src1.y & 0x000000ff) + (src2.y & 0x000000ff));
    dst.z |= ((src1.y & 0x0000ff00) + (src2.y & 0x0000ff00)) <<  8;
    dst.w  = ((src1.y & 0x00ff0000) + (src2.y & 0x00ff0000)) >> 16;
    dst.w |= ((src1.y >>        24) + (src2.y >>        24)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((((int)(src1.x) << 16) >> 16) + ( src2.x        & 0x000000ff)) & 0x0000ffff;
    dst.x |= ((       src1.x  & 0xffff0000) + ((src2.x <<  8) & 0x00ff0000));
    dst.y  = ((((int)(src1.y) << 16) >> 16) + ((src2.x >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.y |= ((       src1.y  & 0xffff0000) + ((src2.x >>  8) & 0x00ff0000));
    dst.z  = ((((int)(src1.z) << 16) >> 16) + ( src2.y        & 0x000000ff)) & 0x0000ffff;
    dst.z |= ((       src1.z  & 0xffff0000) + ((src2.y <<  8) & 0x00ff0000));
    dst.w  = ((((int)(src1.w) << 16) >> 16) + ((src2.y >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.w |= ((       src1.w  & 0xffff0000) + ((src2.y >>  8) & 0x00ff0000));

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (int)(hip_clamp((float)(((int)(src1.x) << 16) >> 16) + unpack0_(src2.x), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.x |= (int)(hip_clamp((float)( (int)(src1.x)        >> 16) + unpack1_(src2.x), -32768.0f, 32767.0f)) << 16;
    dst.y  = (int)(hip_clamp((float)(((int)(src1.y) << 16) >> 16) + unpack2_(src2.x), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.y |= (int)(hip_clamp((float)( (int)(src1.y)        >> 16) + unpack3_(src2.x), -32768.0f, 32767.0f)) << 16;
    dst.z  = (int)(hip_clamp((float)(((int)(src1.z) << 16) >> 16) + unpack0_(src2.y), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.z |= (int)(hip_clamp((float)( (int)(src1.z)        >> 16) + unpack1_(src2.y), -32768.0f, 32767.0f)) << 16;
    dst.w  = (int)(hip_clamp((float)(((int)(src1.w) << 16) >> 16) + unpack2_(src2.y), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.w |= (int)(hip_clamp((float)( (int)(src1.w)        >> 16) + unpack3_(src2.y), -32768.0f, 32767.0f)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (src1.x +  src2.x              ) & 0x0000ffff;
    dst.x |= (src1.x + (src2.x & 0xffff0000)) & 0xffff0000;
    dst.y  = (src1.y +  src2.y              ) & 0x0000ffff;
    dst.y |= (src1.y + (src2.y & 0xffff0000)) & 0xffff0000;
    dst.z  = (src1.z +  src2.z              ) & 0x0000ffff;
    dst.z |= (src1.z + (src2.z & 0xffff0000)) & 0xffff0000;
    dst.w  = (src1.w +  src2.w              ) & 0x0000ffff;
    dst.w |= (src1.w + (src2.w & 0xffff0000)) & 0xffff0000;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Add_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = hip_clamp((((int)(src1.x) << 16) >> 16) + (((int)(src2.x) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.x |= hip_clamp(( (int)(src1.x)        >> 16) + ( (int)(src2.x)        >> 16), -32768, 32767) << 16;
    dst.y  = hip_clamp((((int)(src1.y) << 16) >> 16) + (((int)(src2.y) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.y |= hip_clamp(( (int)(src1.y)        >> 16) + ( (int)(src2.y)        >> 16), -32768, 32767) << 16;
    dst.z  = hip_clamp((((int)(src1.z) << 16) >> 16) + (((int)(src2.z) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.z |= hip_clamp(( (int)(src1.z)        >> 16) + ( (int)(src2.z)        >> 16), -32768, 32767) << 16;
    dst.w  = hip_clamp((((int)(src1.w) << 16) >> 16) + (((int)(src2.w) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.w |= hip_clamp(( (int)(src1.w)        >> 16) + ( (int)(src2.w)        >> 16), -32768, 32767) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Add_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Add_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxSub kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = (src1.x -  src2.x              ) & 0x000000ff;
    dst.x |= (src1.x - (src2.x & 0x0000ff00)) & 0x0000ff00;
    dst.x |= (src1.x - (src2.x & 0x00ff0000)) & 0x00ff0000;
    dst.x |= (src1.x - (src2.x & 0xff000000)) & 0xff000000;
    dst.y  = (src1.y -  src2.y              ) & 0x000000ff;
    dst.y |= (src1.y - (src2.y & 0x0000ff00)) & 0x0000ff00;
    dst.y |= (src1.y - (src2.y & 0x00ff0000)) & 0x00ff0000;
    dst.y |= (src1.y - (src2.y & 0xff000000)) & 0xff000000;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_U8_U8U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_U8_U8U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x = pack_((unpack_(src1.x) - unpack_(src2.x)));
    dst.y = pack_((unpack_(src1.y) - unpack_(src2.y)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_U8_U8U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_U8_U8U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((src1.x & 0x000000ff) - (src2.x & 0x000000ff)) & 0x0000ffff;
    dst.x |= ((src1.x & 0x0000ff00) - (src2.x & 0x0000ff00)) <<  8;
    dst.y  = ((src1.x & 0x00ff0000) - (src2.x & 0x00ff0000)) >> 16;
    dst.y |= ((src1.x >>        24) - (src2.x >>        24)) << 16;
    dst.z  = ((src1.y & 0x000000ff) - (src2.y & 0x000000ff)) & 0x0000ffff;
    dst.z |= ((src1.y & 0x0000ff00) - (src2.y & 0x0000ff00)) <<  8;
    dst.w  = ((src1.y & 0x00ff0000) - (src2.y & 0x00ff0000)) >> 16;
    dst.w |= ((src1.y >>        24) - (src2.y >>        24)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((((int)(src1.x) << 16) >> 16) - ( src2.x        & 0x000000ff)) & 0x0000ffff;
    dst.x |= ((       src1.x  & 0xffff0000) - ((src2.x <<  8) & 0x00ff0000));
    dst.y  = ((((int)(src1.y) << 16) >> 16) - ((src2.x >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.y |= ((       src1.y  & 0xffff0000) - ((src2.x >>  8) & 0x00ff0000));
    dst.z  = ((((int)(src1.z) << 16) >> 16) - ( src2.y        & 0x000000ff)) & 0x0000ffff;
    dst.z |= ((       src1.z  & 0xffff0000) - ((src2.y <<  8) & 0x00ff0000));
    dst.w  = ((((int)(src1.w) << 16) >> 16) - ((src2.y >> 16) & 0x000000ff)) & 0x0000ffff;
    dst.w |= ((       src1.w  & 0xffff0000) - ((src2.y >>  8) & 0x00ff0000));

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16U8_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16U8_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (int)(hip_clamp((float)(((int)(src1.x) << 16) >> 16) - unpack0_(src2.x), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.x |= (int)(hip_clamp((float)( (int)(src1.x)        >> 16) - unpack1_(src2.x), -32768.0f, 32767.0f)) << 16;
    dst.y  = (int)(hip_clamp((float)(((int)(src1.y) << 16) >> 16) - unpack2_(src2.x), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.y |= (int)(hip_clamp((float)( (int)(src1.y)        >> 16) - unpack3_(src2.x), -32768.0f, 32767.0f)) << 16;
    dst.z  = (int)(hip_clamp((float)(((int)(src1.z) << 16) >> 16) - unpack0_(src2.y), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.z |= (int)(hip_clamp((float)( (int)(src1.z)        >> 16) - unpack1_(src2.y), -32768.0f, 32767.0f)) << 16;
    dst.w  = (int)(hip_clamp((float)(((int)(src1.w) << 16) >> 16) - unpack2_(src2.y), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.w |= (int)(hip_clamp((float)( (int)(src1.w)        >> 16) - unpack3_(src2.y), -32768.0f, 32767.0f)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16U8_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16U8_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (( src1.x        & 0x000000ff) - (((int)(src2.x) << 16) >> 16)) & 0x0000ffff;
    dst.x |= (((src1.x <<  8) & 0x00ff0000) - (       src2.x  & 0xffff0000));
    dst.y  = (((src1.x >> 16) & 0x000000ff) - (((int)(src2.y) << 16) >> 16)) & 0x0000ffff;
    dst.y |= (((src1.x >>  8) & 0x00ff0000) - (       src2.y  & 0xffff0000));
    dst.z  = (( src1.y        & 0x000000ff) - (((int)(src2.z) << 16) >> 16)) & 0x0000ffff;
    dst.z |= (((src1.y <<  8) & 0x00ff0000) - (       src2.z  & 0xffff0000));
    dst.w  = (((src1.y >> 16) & 0x000000ff) - (((int)(src2.w) << 16) >> 16)) & 0x0000ffff;
    dst.w |= (((src1.y >>  8) & 0x00ff0000) - (       src2.w  & 0xffff0000));

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_U8S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_U8S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (int)(hip_clamp(unpack0_(src1.x) - (float)(((int)(src2.x) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.x |= (int)(hip_clamp(unpack1_(src1.x) - (float)( (int)(src2.x)        >> 16), -32768.0f, 32767.0f)) << 16;
    dst.y  = (int)(hip_clamp(unpack2_(src1.x) - (float)(((int)(src2.y) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.y |= (int)(hip_clamp(unpack3_(src1.x) - (float)( (int)(src2.y)        >> 16), -32768.0f, 32767.0f)) << 16;
    dst.z  = (int)(hip_clamp(unpack0_(src1.y) - (float)(((int)(src2.z) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.z |= (int)(hip_clamp(unpack1_(src1.y) - (float)( (int)(src2.z)        >> 16), -32768.0f, 32767.0f)) << 16;
    dst.w  = (int)(hip_clamp(unpack2_(src1.y) - (float)(((int)(src2.w) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;
    dst.w |= (int)(hip_clamp(unpack3_(src1.y) - (float)( (int)(src2.w)        >> 16), -32768.0f, 32767.0f)) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_U8S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_U8S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (src1.x -  src2.x              ) & 0x0000ffff;
    dst.x |= (src1.x - (src2.x & 0xffff0000)) & 0xffff0000;
    dst.y  = (src1.y -  src2.y              ) & 0x0000ffff;
    dst.y |= (src1.y - (src2.y & 0xffff0000)) & 0xffff0000;
    dst.z  = (src1.z -  src2.z              ) & 0x0000ffff;
    dst.z |= (src1.z - (src2.z & 0xffff0000)) & 0xffff0000;
    dst.w  = (src1.w -  src2.w              ) & 0x0000ffff;
    dst.w |= (src1.w - (src2.w & 0xffff0000)) & 0xffff0000;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sub_S16_S16S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = hip_clamp((((int)(src1.x) << 16) >> 16) - (((int)(src2.x) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.x |= hip_clamp(( (int)(src1.x)        >> 16) - ( (int)(src2.x)        >> 16), -32768, 32767) << 16;
    dst.y  = hip_clamp((((int)(src1.y) << 16) >> 16) - (((int)(src2.y) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.y |= hip_clamp(( (int)(src1.y)        >> 16) - ( (int)(src2.y)        >> 16), -32768, 32767) << 16;
    dst.z  = hip_clamp((((int)(src1.z) << 16) >> 16) - (((int)(src2.z) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.z |= hip_clamp(( (int)(src1.z)        >> 16) - ( (int)(src2.z)        >> 16), -32768, 32767) << 16;
    dst.w  = hip_clamp((((int)(src1.w) << 16) >> 16) - (((int)(src2.w) << 16) >> 16), -32768, 32767) & 0x0000ffff;
    dst.w |= hip_clamp(( (int)(src1.w)        >> 16) - ( (int)(src2.w)        >> 16), -32768, 32767) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Sub_S16_S16S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sub_S16_S16S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMul kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = ((int)(scale * unpack0_(src1.x) * unpack0_(src2.x)) & 0x000000ff)      ;
    dst.x |= ((int)(scale * unpack1_(src1.x) * unpack1_(src2.x)) & 0x000000ff) <<  8;
    dst.x |= ((int)(scale * unpack2_(src1.x) * unpack2_(src2.x)) & 0x000000ff) << 16;
    dst.x |= ((int)(scale * unpack3_(src1.x) * unpack3_(src2.x))             ) << 24;
    dst.y  = ((int)(scale * unpack0_(src1.y) * unpack0_(src2.y)) & 0x000000ff)      ;
    dst.y |= ((int)(scale * unpack1_(src1.y) * unpack1_(src2.y)) & 0x000000ff) <<  8;
    dst.y |= ((int)(scale * unpack2_(src1.y) * unpack2_(src2.y)) & 0x000000ff) << 16;
    dst.y |= ((int)(scale * unpack3_(src1.y) * unpack3_(src2.y))             ) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    dst.x  = ((int)(scale * unpack0_(src1.x) * unpack0_(src2.x) + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;
    dst.x |= ((int)(scale * unpack1_(src1.x) * unpack1_(src2.x) + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;
    dst.x |= ((int)(scale * unpack2_(src1.x) * unpack2_(src2.x) + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;
    dst.x |= ((int)(scale * unpack3_(src1.x) * unpack3_(src2.x) + (0.5f - 0.00006103515625f))             ) << 24;
    dst.y  = ((int)(scale * unpack0_(src1.y) * unpack0_(src2.y) + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;
    dst.y |= ((int)(scale * unpack1_(src1.y) * unpack1_(src2.y) + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;
    dst.y |= ((int)(scale * unpack2_(src1.y) * unpack2_(src2.y) + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;
    dst.y |= ((int)(scale * unpack3_(src1.y) * unpack3_(src2.y) + (0.5f - 0.00006103515625f))             ) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 f;
    f.x = scale * unpack0_(src1.x) * unpack0_(src2.x) - (0.5f - 0.00006103515625f);
    f.y = scale * unpack1_(src1.x) * unpack1_(src2.x) - (0.5f - 0.00006103515625f);
    f.z = scale * unpack2_(src1.x) * unpack2_(src2.x) - (0.5f - 0.00006103515625f);
    f.w = scale * unpack3_(src1.x) * unpack3_(src2.x) - (0.5f - 0.00006103515625f);
    dst.x = pack_(f);
    f.x = scale * unpack0_(src1.y) * unpack0_(src2.y) - (0.5f - 0.00006103515625f);
    f.y = scale * unpack1_(src1.y) * unpack1_(src2.y) - (0.5f - 0.00006103515625f);
    f.z = scale * unpack2_(src1.y) * unpack2_(src2.y) - (0.5f - 0.00006103515625f);
    f.w = scale * unpack3_(src1.y) * unpack3_(src2.y) - (0.5f - 0.00006103515625f);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_U8_U8U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    float4 f;
    f.x = scale * unpack0_(src1.x) * unpack0_(src2.x);
    f.y = scale * unpack1_(src1.x) * unpack1_(src2.x);
    f.z = scale * unpack2_(src1.x) * unpack2_(src2.x);
    f.w = scale * unpack3_(src1.x) * unpack3_(src2.x);
    dst.x = pack_(f);
    f.x = scale * unpack0_(src1.y) * unpack0_(src2.y);
    f.y = scale * unpack1_(src1.y) * unpack1_(src2.y);
    f.z = scale * unpack2_(src1.y) * unpack2_(src2.y);
    f.w = scale * unpack3_(src1.y) * unpack3_(src2.y);
    dst.y = pack_(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_U8_U8U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_U8_U8U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)(scale * unpack0_(src1.x) * unpack0_(src2.x))) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * unpack1_(src1.x) * unpack1_(src2.x)))             ) << 16;
    dst.y  = (((int)(scale * unpack2_(src1.x) * unpack2_(src2.x))) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * unpack3_(src1.x) * unpack3_(src2.x)))             ) << 16;
    dst.z  = (((int)(scale * unpack0_(src1.y) * unpack0_(src2.y))) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * unpack1_(src1.y) * unpack1_(src2.y)))             ) << 16;
    dst.w  = (((int)(scale * unpack2_(src1.y) * unpack2_(src2.y))) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * unpack3_(src1.y) * unpack3_(src2.y)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)(scale * unpack0_(src1.x) * unpack0_(src2.x) + 0.5f)) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * unpack1_(src1.x) * unpack1_(src2.x) + 0.5f))             ) << 16;
    dst.y  = (((int)(scale * unpack2_(src1.x) * unpack2_(src2.x) + 0.5f)) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * unpack3_(src1.x) * unpack3_(src2.x) + 0.5f))             ) << 16;
    dst.z  = (((int)(scale * unpack0_(src1.y) * unpack0_(src2.y) + 0.5f)) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * unpack1_(src1.y) * unpack1_(src2.y) + 0.5f))             ) << 16;
    dst.w  = (((int)(scale * unpack2_(src1.y) * unpack2_(src2.y) + 0.5f)) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * unpack3_(src1.y) * unpack3_(src2.y) + 0.5f))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)(hip_clamp(scale * unpack0_(src1.x) * unpack0_(src2.x), -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_clamp(scale * unpack1_(src1.x) * unpack1_(src2.x), -32768.0f, 32767.0f)))             ) << 16;
    dst.y  = (((int)(hip_clamp(scale * unpack2_(src1.x) * unpack2_(src2.x), -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_clamp(scale * unpack3_(src1.x) * unpack3_(src2.x), -32768.0f, 32767.0f)))             ) << 16;
    dst.z  = (((int)(hip_clamp(scale * unpack0_(src1.y) * unpack0_(src2.y), -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_clamp(scale * unpack1_(src1.y) * unpack1_(src2.y), -32768.0f, 32767.0f)))             ) << 16;
    dst.w  = (((int)(hip_clamp(scale * unpack2_(src1.y) * unpack2_(src2.y), -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_clamp(scale * unpack3_(src1.y) * unpack3_(src2.y), -32768.0f, 32767.0f)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_U8U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)(hip_clamp(scale * unpack0_(src1.x) * unpack0_(src2.x) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_clamp(scale * unpack1_(src1.x) * unpack1_(src2.x) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.y  = (((int)(hip_clamp(scale * unpack2_(src1.x) * unpack2_(src2.x) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_clamp(scale * unpack3_(src1.x) * unpack3_(src2.x) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.z  = (((int)(hip_clamp(scale * unpack0_(src1.y) * unpack0_(src2.y) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_clamp(scale * unpack1_(src1.y) * unpack1_(src2.y) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;
    dst.w  = (((int)(hip_clamp(scale * unpack2_(src1.y) * unpack2_(src2.y) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_clamp(scale * unpack3_(src1.y) * unpack3_(src2.y) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_U8U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_U8U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)(scale * (float)(((int)(src1.x) << 16) >> 16) * unpack0_(src2.x))) & 0x0000ffff)      ;
    dst.x |= (((int)(scale * (float)( (int)(src1.x)        >> 16) * unpack1_(src2.x)))             ) << 16;
    dst.y  = (((int)(scale * (float)(((int)(src1.y) << 16) >> 16) * unpack2_(src2.x))) & 0x0000ffff)      ;
    dst.y |= (((int)(scale * (float)( (int)(src1.y)        >> 16) * unpack3_(src2.x)))             ) << 16;
    dst.z  = (((int)(scale * (float)(((int)(src1.z) << 16) >> 16) * unpack0_(src2.y))) & 0x0000ffff)      ;
    dst.z |= (((int)(scale * (float)( (int)(src1.z)        >> 16) * unpack1_(src2.y)))             ) << 16;
    dst.w  = (((int)(scale * (float)(((int)(src1.w) << 16) >> 16) * unpack2_(src2.y))) & 0x0000ffff)      ;
    dst.w |= (((int)(scale * (float)( (int)(src1.w)        >> 16) * unpack3_(src2.y)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.x) << 16) >> 16) * unpack0_(src2.x))) & 0x0000ffff)      ;
    dst.x |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.x)        >> 16) * unpack1_(src2.x)))             ) << 16;
    dst.y  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.y) << 16) >> 16) * unpack2_(src2.x))) & 0x0000ffff)      ;
    dst.y |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.y)        >> 16) * unpack3_(src2.x)))             ) << 16;
    dst.z  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.z) << 16) >> 16) * unpack0_(src2.y))) & 0x0000ffff)      ;
    dst.z |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.z)        >> 16) * unpack1_(src2.y)))             ) << 16;
    dst.w  = (((int)hip_convert_short_rte(scale * (float)(((int)(src1.w) << 16) >> 16) * unpack2_(src2.y))) & 0x0000ffff)      ;
    dst.w |= (((int)hip_convert_short_rte(scale * (float)( (int)(src1.w)        >> 16) * unpack3_(src2.y)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    float f;
    f = hip_clamp(scale * (float)(((int)(src1.x) << 16) >> 16) * unpack0_(src2.x), -32768.0f, 32767.0f); dst.x  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.x)        >> 16) * unpack1_(src2.x), -32768.0f, 32767.0f); dst.x |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.y) << 16) >> 16) * unpack2_(src2.x), -32768.0f, 32767.0f); dst.y  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.y)        >> 16) * unpack3_(src2.x), -32768.0f, 32767.0f); dst.y |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.z) << 16) >> 16) * unpack0_(src2.y), -32768.0f, 32767.0f); dst.z  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.z)        >> 16) * unpack1_(src2.y), -32768.0f, 32767.0f); dst.z |= ((int)(f)             ) << 16;
    f = hip_clamp(scale * (float)(((int)(src1.w) << 16) >> 16) * unpack2_(src2.y), -32768.0f, 32767.0f); dst.w  = ((int)(f) & 0x0000ffff)      ;
    f = hip_clamp(scale * (float)( (int)(src1.w)        >> 16) * unpack3_(src2.y), -32768.0f, 32767.0f); dst.w |= ((int)(f)             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16U8_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x =  (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.x) << 16) >> 16) * unpack0_(src2.x)))) & 0x0000ffff)      ;
    dst.x |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.x)  >> 16)	     * unpack1_(src2.x))))             ) << 16;
    dst.y  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.y) << 16) >> 16) * unpack2_(src2.x)))) & 0x0000ffff)      ;
    dst.y |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.y)  >> 16)        * unpack3_(src2.x)))))              << 16;
    dst.z  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.z) << 16) >> 16) * unpack0_(src2.y)))) & 0x0000ffff)      ;
    dst.z |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.z)  >> 16)        * unpack1_(src2.y)))))              << 16;
    dst.w  = (((int)(hip_convert_short_sat_rte(scale * (float)(((int)(src1.w) << 16) >> 16) * unpack2_(src2.y)))) & 0x0000ffff)      ;
    dst.w |= (((int)(hip_convert_short_sat_rte(scale * (float)((int)(src1.w)  >> 16)        * unpack3_(src2.y)))))              << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16U8_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16U8_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((int)(scale * (double)((((int)(src1.x)) << 16) >> 16) * (double)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff;
    dst.x |= ((int)(scale * (double)(( (int)(src1.x)) >> 16)        * (double)(( (int)(src2.x)) >> 16)))		  << 16;
    dst.y  = ((int)(scale * (double)((((int)(src1.y)) << 16) >> 16) * (double)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff;
    dst.y |= ((int)(scale * (double)(( (int)(src1.y)) >> 16)        * (double)(( (int)(src2.y)) >> 16)))		  << 16;
    dst.z  = ((int)(scale * (double)((((int)(src1.z)) << 16) >> 16) * (double)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff;
    dst.z |= ((int)(scale * (double)(( (int)(src1.z)) >> 16)        * (double)(( (int)(src2.z)) >> 16)))		  << 16;
    dst.w  = ((int)(scale * (double)((((int)(src1.w)) << 16) >> 16) * (double)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff;
    dst.w |= ((int)(scale * (double)(( (int)(src1.w)) >> 16)        * (double)(( (int)(src2.w)) >> 16)))		  << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Wrap_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Wrap_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff;
    dst.x |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.x)) >> 16)        * (float)(( (int)(src2.x)) >> 16)))		  << 16;
    dst.y  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff;
    dst.y |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.y)) >> 16)        * (float)(( (int)(src2.y)) >> 16)))		  << 16;
    dst.z  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff;
    dst.z |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.z)) >> 16)        * (float)(( (int)(src2.z)) >> 16)))		  << 16;
    dst.w  = ((int)hip_convert_short_rte(scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff;
    dst.w |= ((int)hip_convert_short_rte(scale * (float)(( (int)(src1.w)) >> 16)        * (float)(( (int)(src2.w)) >> 16)))		  << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Wrap_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Wrap_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Trunc(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)hip_clamp((scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.x |= (((int)hip_clamp((scale * (float)(( (int)(src1.x))        >> 16) * (float)(( (int)(src2.x))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.y  = (((int)hip_clamp((scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.y |= (((int)hip_clamp((scale * (float)(( (int)(src1.y))        >> 16) * (float)(( (int)(src2.y))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.z  = (((int)hip_clamp((scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.z |= (((int)hip_clamp((scale * (float)(( (int)(src1.z))        >> 16) * (float)(( (int)(src2.z))        >> 16)), -32768.0f, 32767.0f))             ) << 16;
    dst.w  = (((int)hip_clamp((scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;
    dst.w |= (((int)hip_clamp((scale * (float)(( (int)(src1.w))        >> 16) * (float)(( (int)(src2.w))        >> 16)), -32768.0f, 32767.0f))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Sat_Trunc(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Trunc, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Mul_S16_S16S16_Sat_Round(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes, 
    float scale) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    dst.x  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.x)) << 16) >> 16) * (float)((((int)(src2.x)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.x |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.x))        >> 16) * (float)(( (int)(src2.x))        >> 16)))             ) << 16;
    dst.y  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.y)) << 16) >> 16) * (float)((((int)(src2.y)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.y |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.y))        >> 16) * (float)(( (int)(src2.y))        >> 16)))             ) << 16;
    dst.z  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.z)) << 16) >> 16) * (float)((((int)(src2.z)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.z |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.z))        >> 16) * (float)(( (int)(src2.z))        >> 16)))             ) << 16;
    dst.w  = (((int)hip_convert_short_sat_rte(scale * (float)((((int)(src1.w)) << 16) >> 16) * (float)((((int)(src2.w)) << 16) >> 16))) & 0x0000ffff)      ;
    dst.w |= (((int)hip_convert_short_sat_rte(scale * (float)(( (int)(src1.w))        >> 16) * (float)(( (int)(src2.w))        >> 16)))             ) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Mul_S16_S16S16_Sat_Round(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes, 
    vx_float32 scale) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Mul_S16_S16S16_Sat_Round, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, 
                        scale);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWeightedAverage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WeightedAverage_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes,
    float alpha, float invAlpha
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    float4 src1 = uchars_to_float4(pSrcImage1[src1Idx]);
    float4 src2 = uchars_to_float4(pSrcImage2[src2Idx]);
    float4 dst = make_float4(src2.x*invAlpha+src1.x*alpha, src2.y*invAlpha+src1.y*alpha, src2.z*invAlpha+src1.z*alpha, src2.w*invAlpha+src1.w*alpha);
    pDstImage[dstIdx] = float4_to_uchars(dst);
}
int HipExec_WeightedAverage_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    vx_float32 alpha
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;
    vx_float32 invAlpha = (vx_float32)1 - alpha;

    hipLaunchKernelGGL(Hip_WeightedAverage_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes, alpha, invAlpha);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMagnitude kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Magnitude_S16_S16S16(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    int4 dst;

    // To Modify
    // float2 f;
    // f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2 |= (uint)(f.s0) << 16;
    // f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3  = (uint)(f.s0);
    // f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3 |= (uint)(f.s0) << 16;

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Magnitude_S16_S16S16(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Magnitude_S16_S16S16, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxPhase kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Phase_U8_S16S16(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x + x;
    uint src2Idx = y * srcImage2StrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src1 = *((int4 *)(&pSrcImage1[src1Idx]));
    int4 src2 = *((int4 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    // To Modify
    // float2 f; float4 p4;
    // f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;
    // f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;
    // p4 = select(p4, (float4) 0.0f, p4 > 255.5f);
    // r.s0 = amd_pack(p4);
    // f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;
    // f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;
    // f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;
    // p4 = select(p4, (float4) 0.0f, p4 > 255.5f);
    // r.s1 = amd_pack(p4);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Phase_U8_S16S16(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Phase_U8_S16S16, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes, 
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}