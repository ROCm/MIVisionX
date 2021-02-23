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
// VxBox kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Box_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Box_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Box_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxDilate kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Dilate_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    uint4 pix;
    float4 val;
    uint2 *pixLoc0 = (uint2*)&pix.x;
    uint2 *pixLoc2 = (uint2*)&pix.z;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pixLoc0 = lbufptr[0];
    *pixLoc2 = lbufptr[1];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    sum.data[0] = max3_(val.x, val.y, val.z);
    val.x = unpack2_(pix.y);
    sum.data[1] = max3_(val.x, val.y, val.z);
    val.y = unpack3_(pix.y);
    sum.data[2] = max3_(val.x, val.y, val.z);
    val.z = unpack0_(pix.z);
    sum.data[3] = max3_(val.x, val.y, val.z);
    val.x = unpack1_(pix.z);
    sum.data[4] = max3_(val.x, val.y, val.z);
    val.y = unpack2_(pix.z);
    sum.data[5] = max3_(val.x, val.y, val.z);
    val.z = unpack3_(pix.z);
    sum.data[6] = max3_(val.x, val.y, val.z);
    val.x = unpack0_(pix.w);
    sum.data[7] = max3_(val.x, val.y, val.z);
    *pixLoc0 = lbufptr[17];
    *pixLoc2 = lbufptr[18];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[0] = max(sum.data[0], val.w);
    val.x = unpack2_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[1] = max(sum.data[1], val.w);
    val.y = unpack3_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[2] = max(sum.data[2], val.w);
    val.z = unpack0_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[3] = max(sum.data[3], val.w);
    val.x = unpack1_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[4] = max(sum.data[4], val.w);
    val.y = unpack2_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[5] = max(sum.data[5], val.w);
    val.z = unpack3_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[6] = max(sum.data[6], val.w);
    val.x = unpack0_(pix.w);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[7] = max(sum.data[7], val.w);
    *pixLoc0 = lbufptr[34];
    *pixLoc2 = lbufptr[35];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[0] = max(sum.data[0], val.w);
    val.x = unpack2_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[1] = max(sum.data[1], val.w);
    val.y = unpack3_(pix.y);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[2] = max(sum.data[2], val.w);
    val.z = unpack0_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[3] = max(sum.data[3], val.w);
    val.x = unpack1_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[4] = max(sum.data[4], val.w);
    val.y = unpack2_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[5] = max(sum.data[5], val.w);
    val.z = unpack3_(pix.z);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[6] = max(sum.data[6], val.w);
    val.x = unpack0_(pix.w);
    val.w = max3_(val.x, val.y, val.z);
    sum.data[7] = max(sum.data[7], val.w);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Dilate_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Dilate_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxErode kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Erode_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    uint4 pix;
    float4 val;
    uint2 *pixLoc0 = (uint2*)&pix.x;
    uint2 *pixLoc2 = (uint2*)&pix.z;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pixLoc0 = lbufptr[0];
    *pixLoc2 = lbufptr[1];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    sum.data[0] = min3_(val.x, val.y, val.z);
    val.x = unpack2_(pix.y);
    sum.data[1] = min3_(val.x, val.y, val.z);
    val.y = unpack3_(pix.y);
    sum.data[2] = min3_(val.x, val.y, val.z);
    val.z = unpack0_(pix.z);
    sum.data[3] = min3_(val.x, val.y, val.z);
    val.x = unpack1_(pix.z);
    sum.data[4] = min3_(val.x, val.y, val.z);
    val.y = unpack2_(pix.z);
    sum.data[5] = min3_(val.x, val.y, val.z);
    val.z = unpack3_(pix.z);
    sum.data[6] = min3_(val.x, val.y, val.z);
    val.x = unpack0_(pix.w);
    sum.data[7] = min3_(val.x, val.y, val.z);
    *pixLoc0 = lbufptr[17];
    *pixLoc2 = lbufptr[18];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[0] = min(sum.data[0], val.w);
    val.x = unpack2_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[1] = min(sum.data[1], val.w);
    val.y = unpack3_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[2] = min(sum.data[2], val.w);
    val.z = unpack0_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[3] = min(sum.data[3], val.w);
    val.x = unpack1_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[4] = min(sum.data[4], val.w);
    val.y = unpack2_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[5] = min(sum.data[5], val.w);
    val.z = unpack3_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[6] = min(sum.data[6], val.w);
    val.x = unpack0_(pix.w);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[7] = min(sum.data[7], val.w);
    *pixLoc0 = lbufptr[34];
    *pixLoc2 = lbufptr[35];
    val.x = unpack3_(pix.x);
    val.y = unpack0_(pix.y);
    val.z = unpack1_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[0] = min(sum.data[0], val.w);
    val.x = unpack2_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[1] = min(sum.data[1], val.w);
    val.y = unpack3_(pix.y);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[2] = min(sum.data[2], val.w);
    val.z = unpack0_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[3] = min(sum.data[3], val.w);
    val.x = unpack1_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[4] = min(sum.data[4], val.w);
    val.y = unpack2_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[5] = min(sum.data[5], val.w);
    val.z = unpack3_(pix.z);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[6] = min(sum.data[6], val.w);
    val.x = unpack0_(pix.w);
    val.w = min3_(val.x, val.y, val.z);
    sum.data[7] = min(sum.data[7], val.w);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Erode_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Erode_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMedian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Median_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    float4 val0, val1, val2, valz;
    uint4 pix0, pix1, pix2;
    uint2 *pix0Loc0 = (uint2*)&pix0.x;
    uint2 *pix0Loc2 = (uint2*)&pix0.z;
    uint2 *pix1Loc0 = (uint2*)&pix1.x;
    uint2 *pix1Loc2 = (uint2*)&pix1.z;
    uint2 *pix2Loc0 = (uint2*)&pix2.x;
    uint2 *pix2Loc2 = (uint2*)&pix2.z;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pix0Loc0 = lbufptr[0];
    *pix0Loc2 = lbufptr[1];
    *pix1Loc0 = lbufptr[17];
    *pix1Loc2 = lbufptr[18];
    *pix2Loc0 = lbufptr[34];
    *pix2Loc2 = lbufptr[35];
    // pixel 0
    valz.x = unpack3_(pix0.x);
    valz.y = unpack0_(pix0.y);
    valz.z = unpack1_(pix0.y);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack3_(pix1.x);
    valz.y = unpack0_(pix1.y);
    valz.z = unpack1_(pix1.y);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack3_(pix2.x);
    valz.y = unpack0_(pix2.y);
    valz.z = unpack1_(pix2.y);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[0] = median3_(valz.x, valz.y, valz.z);
    // pixel 1
    valz.x = unpack0_(pix0.y);
    valz.y = unpack1_(pix0.y);
    valz.z = unpack2_(pix0.y);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack0_(pix1.y);
    valz.y = unpack1_(pix1.y);
    valz.z = unpack2_(pix1.y);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack0_(pix2.y);
    valz.y = unpack1_(pix2.y);
    valz.z = unpack2_(pix2.y);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[1] = median3_(valz.x, valz.y, valz.z);
    // pixel 2
    valz.x = unpack1_(pix0.y);
    valz.y = unpack2_(pix0.y);
    valz.z = unpack3_(pix0.y);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack1_(pix1.y);
    valz.y = unpack2_(pix1.y);
    valz.z = unpack3_(pix1.y);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack1_(pix2.y);
    valz.y = unpack2_(pix2.y);
    valz.z = unpack3_(pix2.y);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[2] = median3_(valz.x, valz.y, valz.z);
    // pixel 3
    valz.x = unpack2_(pix0.y);
    valz.y = unpack3_(pix0.y);
    valz.z = unpack0_(pix0.z);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack2_(pix1.y);
    valz.y = unpack3_(pix1.y);
    valz.z = unpack0_(pix1.z);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack2_(pix2.y);
    valz.y = unpack3_(pix2.y);
    valz.z = unpack0_(pix2.z);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[3] = median3_(valz.x, valz.y, valz.z);
    // pixel 4
    valz.x = unpack3_(pix0.y);
    valz.y = unpack0_(pix0.z);
    valz.z = unpack1_(pix0.z);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack3_(pix1.y);
    valz.y = unpack0_(pix1.z);
    valz.z = unpack1_(pix1.z);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack3_(pix2.y);
    valz.y = unpack0_(pix2.z);
    valz.z = unpack1_(pix2.z);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[4] = median3_(valz.x, valz.y, valz.z);
    // pixel 5
    valz.x = unpack0_(pix0.z);
    valz.y = unpack1_(pix0.z);
    valz.z = unpack2_(pix0.z);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack0_(pix1.z);
    valz.y = unpack1_(pix1.z);
    valz.z = unpack2_(pix1.z);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack0_(pix2.z);
    valz.y = unpack1_(pix2.z);
    valz.z = unpack2_(pix2.z);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[5] = median3_(valz.x, valz.y, valz.z);
    // pixel 6
    valz.x = unpack1_(pix0.z);
    valz.y = unpack2_(pix0.z);
    valz.z = unpack3_(pix0.z);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack1_(pix1.z);
    valz.y = unpack2_(pix1.z);
    valz.z = unpack3_(pix1.z);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack1_(pix2.z);
    valz.y = unpack2_(pix2.z);
    valz.z = unpack3_(pix2.z);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[6] = median3_(valz.x, valz.y, valz.z);
    // pixel 7
    valz.x = unpack2_(pix0.z);
    valz.y = unpack3_(pix0.z);
    valz.z = unpack0_(pix0.w);
    val0.x = min3_(valz.x, valz.y, valz.z);
    val0.y = median3_(valz.x, valz.y, valz.z);
    val0.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack2_(pix1.z);
    valz.y = unpack3_(pix1.z);
    valz.z = unpack0_(pix1.w);
    val1.x = min3_(valz.x, valz.y, valz.z);
    val1.y = median3_(valz.x, valz.y, valz.z);
    val1.z = max3_(valz.x, valz.y, valz.z);
    valz.x = unpack2_(pix2.z);
    valz.y = unpack3_(pix2.z);
    valz.z = unpack0_(pix2.w);
    val2.x = min3_(valz.x, valz.y, valz.z);
    val2.y = median3_(valz.x, valz.y, valz.z);
    val2.z = max3_(valz.x, valz.y, valz.z);
    valz.x = max3_(val0.x, val1.x, val2.x);
    valz.y = median3_(val0.y, val1.y, val2.y);
    valz.z = min3_(val0.z, val1.z, val2.z);
    sum.data[7] = median3_(valz.x, valz.y, valz.z);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Median_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Median_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxGaussian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Gaussian_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 2.500000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 2.500000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 2.500000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 2.500000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 2.500000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 2.500000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 2.500000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 2.500000000000e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Gaussian_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Gaussian_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxConvolve kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const float *convf32) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[0], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[1], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[0], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[2], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[1], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[0], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[2], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[1], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[0], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[2], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[1], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[0], sum.data[4]);
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[2], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[1], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[0], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[2], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[1], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[0], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[2], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[1], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[0], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[2], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[1], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[3], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[4], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[3], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[5], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[4], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[3], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[5], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[4], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[3], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[5], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[4], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[3], sum.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[5], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[4], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[3], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[5], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[4], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[3], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[5], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[4], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[3], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[5], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[4], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[6], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[7], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[6], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[8], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[7], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[6], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[8], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[7], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[6], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[8], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[7], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[6], sum.data[4]);
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[8], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[7], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[6], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[8], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[7], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[6], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[8], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[7], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[6], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[8], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[7], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[8], sum.data[7]);

    uint2 dst;
    dst.x = pack_(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = pack_(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Convolve_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    const vx_float32 *convf32 = (vx_float32*) conv;

    if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            convf32);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const float *convf32) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[0], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[1], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[0], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[2], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[1], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[0], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[2], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[1], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[0], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[2], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[1], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[0], sum.data[4]);
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[2], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[1], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[0], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[2], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[1], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[0], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[2], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[1], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[0], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[2], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[1], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[3], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[4], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[3], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[5], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[4], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[3], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[5], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[4], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[3], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[5], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[4], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[3], sum.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[5], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[4], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[3], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[5], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[4], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[3], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[5], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[4], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[3], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[5], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[4], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, convf32[6], sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, convf32[7], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[6], sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, convf32[8], sum.data[0]);
    sum.data[1] = fmaf(fval, convf32[7], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[6], sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, convf32[8], sum.data[1]);
    sum.data[2] = fmaf(fval, convf32[7], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[6], sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, convf32[8], sum.data[2]);
    sum.data[3] = fmaf(fval, convf32[7], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[6], sum.data[4]);
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, convf32[8], sum.data[3]);
    sum.data[4] = fmaf(fval, convf32[7], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[6], sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, convf32[8], sum.data[4]);
    sum.data[5] = fmaf(fval, convf32[7], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[6], sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, convf32[8], sum.data[5]);
    sum.data[6] = fmaf(fval, convf32[7], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[6], sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, convf32[8], sum.data[6]);
    sum.data[7] = fmaf(fval, convf32[7], sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, convf32[8], sum.data[7]);

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Convolve_S16_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    const vx_float32 *convf32 = (vx_float32*) conv;

    if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            convf32);
    }

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxSobel kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] -= fval;
    fval = unpack0_(pix.y);
    sum.data[1] -= fval;
    fval = unpack1_(pix.y);
    sum.data[0] += fval;
    sum.data[2] -= fval;
    fval = unpack2_(pix.y);
    sum.data[1] += fval;
    sum.data[3] -= fval;
    fval = unpack3_(pix.y);
    sum.data[2] += fval;
    sum.data[4] -= fval;
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] += fval;
    sum.data[5] -= fval;
    fval = unpack1_(pix.x);
    sum.data[4] += fval;
    sum.data[6] -= fval;
    fval = unpack2_(pix.x);
    sum.data[5] += fval;
    sum.data[7] -= fval;
    fval = unpack3_(pix.x);
    sum.data[6] += fval;
    fval = unpack0_(pix.y);
    sum.data[7] += fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum.data[0] = fmaf(fval, -2.000000000000e+00f, sum.data[0]);
    fval = unpack0_(pix.y);
    sum.data[1] = fmaf(fval, -2.000000000000e+00f, sum.data[1]);
    fval = unpack1_(pix.y);
    sum.data[0] = fmaf(fval, 2.000000000000e+00f, sum.data[0]);
    sum.data[2] = fmaf(fval, -2.000000000000e+00f, sum.data[2]);
    fval = unpack2_(pix.y);
    sum.data[1] = fmaf(fval, 2.000000000000e+00f, sum.data[1]);
    sum.data[3] = fmaf(fval, -2.000000000000e+00f, sum.data[3]);
    fval = unpack3_(pix.y);
    sum.data[2] = fmaf(fval, 2.000000000000e+00f, sum.data[2]);
    sum.data[4] = fmaf(fval, -2.000000000000e+00f, sum.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum.data[3] = fmaf(fval, 2.000000000000e+00f, sum.data[3]);
    sum.data[5] = fmaf(fval, -2.000000000000e+00f, sum.data[5]);
    fval = unpack1_(pix.x);
    sum.data[4] = fmaf(fval, 2.000000000000e+00f, sum.data[4]);
    sum.data[6] = fmaf(fval, -2.000000000000e+00f, sum.data[6]);
    fval = unpack2_(pix.x);
    sum.data[5] = fmaf(fval, 2.000000000000e+00f, sum.data[5]);
    sum.data[7] = fmaf(fval, -2.000000000000e+00f, sum.data[7]);
    fval = unpack3_(pix.x);
    sum.data[6] = fmaf(fval, 2.000000000000e+00f, sum.data[6]);
    fval = unpack0_(pix.y);
    sum.data[7] = fmaf(fval, 2.000000000000e+00f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] -= fval;
    fval = unpack0_(pix.y);
    sum.data[1] -= fval;
    fval = unpack1_(pix.y);
    sum.data[0] += fval;
    sum.data[2] -= fval;
    fval = unpack2_(pix.y);
    sum.data[1] += fval;
    sum.data[3] -= fval;
    fval = unpack3_(pix.y);
    sum.data[2] += fval;
    sum.data[4] -= fval;
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] += fval;
    sum.data[5] -= fval;
    fval = unpack1_(pix.x);
    sum.data[4] += fval;
    sum.data[6] -= fval;
    fval = unpack2_(pix.x);
    sum.data[5] += fval;
    sum.data[7] -= fval;
    fval = unpack3_(pix.x);
    sum.data[6] += fval;
    fval = unpack0_(pix.y);
    sum.data[7] += fval;

    int4 dst;
    dst.x = ((int)sum.data[0]) & 0xffff;
    dst.x |= ((int)sum.data[1]) << 16;
    dst.y = ((int)sum.data[2]) & 0xffff;
    dst.y |= ((int)sum.data[3]) << 16;
    dst.z = ((int)sum.data[4]) & 0xffff;
    dst.z |= ((int)sum.data[5]) << 16;
    dst.w = ((int)sum.data[6]) & 0xffff;
    dst.w |= ((int)sum.data[7]) << 16;

    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Sobel_S16_U8_3x3_GX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GX, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum.data[0] -= fval;
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, -2.000000000000e+00f, sum.data[0]);
    sum.data[1] -= fval;
    fval = unpack1_(pix.y);
    sum.data[0] -= fval;
    sum.data[1] = fmaf(fval, -2.000000000000e+00f, sum.data[1]);
    sum.data[2] -= fval;
    fval = unpack2_(pix.y);
    sum.data[1] -= fval;
    sum.data[2] = fmaf(fval, -2.000000000000e+00f, sum.data[2]);
    sum.data[3] -= fval;
    fval = unpack3_(pix.y);
    sum.data[2] -= fval;
    sum.data[3] = fmaf(fval, -2.000000000000e+00f, sum.data[3]);
    sum.data[4] -= fval;
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum.data[3] -= fval;
    sum.data[4] = fmaf(fval, -2.000000000000e+00f, sum.data[4]);
    sum.data[5] -= fval;
    fval = unpack1_(pix.x);
    sum.data[4] -= fval;
    sum.data[5] = fmaf(fval, -2.000000000000e+00f, sum.data[5]);
    sum.data[6] -= fval;
    fval = unpack2_(pix.x);
    sum.data[5] -= fval;
    sum.data[6] = fmaf(fval, -2.000000000000e+00f, sum.data[6]);
    sum.data[7] -= fval;
    fval = unpack3_(pix.x);
    sum.data[6] -= fval;
    sum.data[7] = fmaf(fval, -2.000000000000e+00f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] -= fval;
    // filterRow = 1
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum.data[0] += fval;
    fval = unpack0_(pix.y);
    sum.data[0] = fmaf(fval, 2.000000000000e+00f, sum.data[0]);
    sum.data[1] += fval;
    fval = unpack1_(pix.y);
    sum.data[0] += fval;
    sum.data[1] = fmaf(fval, 2.000000000000e+00f, sum.data[1]);
    sum.data[2] += fval;
    fval = unpack2_(pix.y);
    sum.data[1] += fval;
    sum.data[2] = fmaf(fval, 2.000000000000e+00f, sum.data[2]);
    sum.data[3] += fval;
    fval = unpack3_(pix.y);
    sum.data[2] += fval;
    sum.data[3] = fmaf(fval, 2.000000000000e+00f, sum.data[3]);
    sum.data[4] += fval;
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum.data[3] += fval;
    sum.data[4] = fmaf(fval, 2.000000000000e+00f, sum.data[4]);
    sum.data[5] += fval;
    fval = unpack1_(pix.x);
    sum.data[4] += fval;
    sum.data[5] = fmaf(fval, 2.000000000000e+00f, sum.data[5]);
    sum.data[6] += fval;
    fval = unpack2_(pix.x);
    sum.data[5] += fval;
    sum.data[6] = fmaf(fval, 2.000000000000e+00f, sum.data[6]);
    sum.data[7] += fval;
    fval = unpack3_(pix.x);
    sum.data[6] += fval;
    sum.data[7] = fmaf(fval, 2.000000000000e+00f, sum.data[7]);
    fval = unpack0_(pix.y);
    sum.data[7] += fval;

    int4 dst;
    dst.x = ((int)sum.data[0]) & 0xffff;
    dst.x |= ((int)sum.data[1]) << 16;
    dst.y = ((int)sum.data[2]) & 0xffff;
    dst.y |= ((int)sum.data[3]) << 16;
    dst.z = ((int)sum.data[4]) & 0xffff;
    dst.z |= ((int)sum.data[5]) << 16;
    dst.w = ((int)sum.data[6]) & 0xffff;
    dst.w |= ((int)sum.data[7]) << 16;

    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Sobel_S16_U8_3x3_GY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GY, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16S16_U8_3x3_GXY(uint dstWidth, uint dstHeight,
    uchar *pDstImage1, uint dstImage1StrideInBytes,
    uchar *pDstImage2, uint dstImage2StrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = unpack3_(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = unpack0_(pix.y);
    sum2.data[0] = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = unpack1_(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum2.data[1] = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = unpack2_(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum2.data[2] = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = unpack3_(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum2.data[3] = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    pix = lbufptr[1];
    fval = unpack0_(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum2.data[4] = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = unpack1_(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum2.data[5] = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = unpack2_(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum2.data[6] = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = unpack3_(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum2.data[7] = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = unpack0_(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = unpack3_(pix.x);
    sum1.data[0] = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    fval = unpack0_(pix.y);
    sum1.data[1] = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    fval = unpack1_(pix.y);
    sum1.data[0] = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum1.data[2] = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    fval = unpack2_(pix.y);
    sum1.data[1] = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum1.data[3] = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    fval = unpack3_(pix.y);
    sum1.data[2] = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum1.data[4] = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    pix = lbufptr[18];
    fval = unpack0_(pix.x);
    sum1.data[3] = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum1.data[5] = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    fval = unpack1_(pix.x);
    sum1.data[4] = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum1.data[6] = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    fval = unpack2_(pix.x);
    sum1.data[5] = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum1.data[7] = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    fval = unpack3_(pix.x);
    sum1.data[6] = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    fval = unpack0_(pix.y);
    sum1.data[7] = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = unpack3_(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = unpack0_(pix.y);
    sum2.data[0] = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = unpack1_(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum2.data[1] = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = unpack2_(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum2.data[2] = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = unpack3_(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum2.data[3] = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    pix = lbufptr[35];
    fval = unpack0_(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum2.data[4] = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = unpack1_(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum2.data[5] = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = unpack2_(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum2.data[6] = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = unpack3_(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum2.data[7] = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = unpack0_(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    int4 dst1;
    dst1.x = ((int)sum1.data[0]) & 0xffff;
    dst1.x |= ((int)sum1.data[1]) << 16;
    dst1.y = ((int)sum1.data[2]) & 0xffff;
    dst1.y |= ((int)sum1.data[3]) << 16;
    dst1.z = ((int)sum1.data[4]) & 0xffff;
    dst1.z |= ((int)sum1.data[5]) << 16;
    dst1.w = ((int)sum1.data[6]) & 0xffff;
    dst1.w |= ((int)sum1.data[7]) << 16;

    int4 dst2;
    dst2.x = ((int)sum2.data[0]) & 0xffff;
    dst2.x |= ((int)sum2.data[1]) << 16;
    dst2.y = ((int)sum2.data[2]) & 0xffff;
    dst2.y |= ((int)sum2.data[3]) << 16;
    dst2.z = ((int)sum2.data[4]) & 0xffff;
    dst2.z |= ((int)sum2.data[5]) << 16;
    dst2.w = ((int)sum2.data[6]) & 0xffff;
    dst2.w |= ((int)sum2.data[7]) << 16;

    uint dst1Idx =  y * dstImage1StrideInBytes + x + x;
    uint dst2Idx =  y * dstImage2StrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage1[dst1Idx])) = dst1;
        *((int4 *)(&pDstImage2[dst2Idx])) = dst2;
    }
}
int HipExec_Sobel_S16S16_U8_3x3_GXY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
    vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16S16_U8_3x3_GXY, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, dstImage1StrideInBytes, (uchar *)pHipDstImage2, dstImage2StrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxScaleGaussianHalf kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleGaussianHalf_U8_U8_3x3(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *gaussian
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int xSrc = (int)PIXELROUNDF32(((x + 0.5) * (srcWidth/dstWidth)) - 0.5);
    int ySrc = (int)PIXELROUNDF32(((y + 0.5) * (srcHeight/dstHeight)) - 0.5);
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrc*(srcImageStrideInBytes) + xSrc;
    if ((ySrc > 1) && (ySrc < srcHeight - 2)) {
        int srcIdxTopRow, srcIdxBottomRow;
        srcIdxTopRow = srcIdx - srcImageStrideInBytes;
        srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
        float sum = 0;
        sum += (gaussian[4] * (float)*(pSrcImage + srcIdx) + gaussian[1] * (float)*(pSrcImage + srcIdxTopRow) + gaussian[7] * (float)*(pSrcImage + srcIdxBottomRow));
        if (xSrc != 0)
            sum += (gaussian[3] * (float)*(pSrcImage + srcIdx - 1) + gaussian[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + gaussian[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
        if (xSrc != (srcWidth - 1))
            sum += (gaussian[5] * (float)*(pSrcImage + srcIdx + 1) + gaussian[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + gaussian[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
    }
    else {
        pDstImage[dstIdx] = 0;
    }
}
int HipExec_ScaleGaussianHalf_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    float gaussian[9] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
    float *hipGaussian;
    hipMalloc(&hipGaussian, 288);
    hipMemcpy(hipGaussian, gaussian, 288, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_ScaleGaussianHalf_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const float *)hipGaussian);
    hipFree(&hipGaussian);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleGaussianHalf_U8_U8_5x5(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *gaussian
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int xSrc = (int)PIXELROUNDF32(((x + 0.5) * (srcWidth/dstWidth)) - 0.5);
    int ySrc = (int)PIXELROUNDF32(((y + 0.5) * (srcHeight/dstHeight)) - 0.5);

    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrc*(srcImageStrideInBytes) + xSrc;

    if ((ySrc > 1) && (ySrc < srcHeight - 2)) {
        int srcIdxTopRowOuter, srcIdxTopRowInner, srcIdxBottomRowInner, srcIdxBottomRowOuter;
        srcIdxTopRowInner = srcIdx - srcImageStrideInBytes;
        srcIdxTopRowOuter = srcIdx - (2 * srcImageStrideInBytes);
        srcIdxBottomRowInner = srcIdx + srcImageStrideInBytes;
        srcIdxBottomRowOuter = srcIdx + (2 * srcImageStrideInBytes);
        float sum = 0;
        sum += (
            gaussian[12] * (float)*(pSrcImage + srcIdx) +
            gaussian[7] * (float)*(pSrcImage + srcIdxTopRowInner) +
            gaussian[2] * (float)*(pSrcImage + srcIdxTopRowOuter) +
            gaussian[17] * (float)*(pSrcImage + srcIdxBottomRowInner) +
            gaussian[22] * (float)*(pSrcImage + srcIdxBottomRowOuter)
            );
        if (xSrc >= 1)
            sum += (
                gaussian[11] * (float)*(pSrcImage + srcIdx - 1) +
                gaussian[6] * (float)*(pSrcImage + srcIdxTopRowInner - 1) +
                gaussian[1] * (float)*(pSrcImage + srcIdxTopRowOuter - 1) +
                gaussian[16] * (float)*(pSrcImage + srcIdxBottomRowInner - 1) +
                gaussian[21] * (float)*(pSrcImage + srcIdxBottomRowOuter - 1)
                );
        if (xSrc >= 2)
            sum += (
                gaussian[10] * (float)*(pSrcImage + srcIdx - 2) +
                gaussian[5] * (float)*(pSrcImage + srcIdxTopRowInner - 2) +
                gaussian[0] * (float)*(pSrcImage + srcIdxTopRowOuter - 2) +
                gaussian[15] * (float)*(pSrcImage + srcIdxBottomRowInner - 2) +
                gaussian[20] * (float)*(pSrcImage + srcIdxBottomRowOuter - 2)
                );
        if (xSrc < (srcWidth - 1))
            sum += (
                gaussian[13] * (float)*(pSrcImage + srcIdx + 1) +
                gaussian[8] * (float)*(pSrcImage + srcIdxTopRowInner + 1) +
                gaussian[3] * (float)*(pSrcImage + srcIdxTopRowOuter + 1) +
                gaussian[18] * (float)*(pSrcImage + srcIdxBottomRowInner + 1) +
                gaussian[23] * (float)*(pSrcImage + srcIdxBottomRowOuter + 1)
                );
        if (xSrc < (srcWidth - 2))
            sum += (
                gaussian[14] * (float)*(pSrcImage + srcIdx + 2) +
                gaussian[9] * (float)*(pSrcImage + srcIdxTopRowInner + 2) +
                gaussian[4] * (float)*(pSrcImage + srcIdxTopRowOuter + 2) +
                gaussian[19] * (float)*(pSrcImage + srcIdxBottomRowInner + 2) +
                gaussian[24] * (float)*(pSrcImage + srcIdxBottomRowOuter + 2)
                );
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
    }
    else {
        pDstImage[dstIdx] = 0;
    }
}
int HipExec_ScaleGaussianHalf_U8_U8_5x5(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    float gaussian[25] = {
        0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
        0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
        0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
        0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
        0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625};

    float *hipGaussian;
    hipMalloc(&hipGaussian, 800);
    hipMemcpy(hipGaussian, gaussian, 800, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_ScaleGaussianHalf_U8_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const float *)hipGaussian);
    return VX_SUCCESS;
}