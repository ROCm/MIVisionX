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

#include "hip_common_funcs.h"
#include "hip_host_decls.h"

// ----------------------------------------------------------------------------
// VxCannyEdgeDetector kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L1NORM(uint dstWidth, uint dstHeight,
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
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum2.data[1] = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum2.data[2] = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum2.data[3] = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum2.data[4] = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum2.data[5] = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum2.data[6] = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum2.data[7] = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum1.data[0] = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack0(pix.y);
    sum1.data[1] = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack1(pix.y);
    sum1.data[0] = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum1.data[2] = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack2(pix.y);
    sum1.data[1] = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum1.data[3] = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack3(pix.y);
    sum1.data[2] = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum1.data[4] = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[3] = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum1.data[5] = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    fval = hip_unpack1(pix.x);
    sum1.data[4] = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum1.data[6] = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack2(pix.x);
    sum1.data[5] = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum1.data[7] = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[6] = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum2.data[1] = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum2.data[2] = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum2.data[3] = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum2.data[4] = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum2.data[5] = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum2.data[6] = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum2.data[7] = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 1);
    mask = HIPSELECT(0u, mask, y < 719);
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L1(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < 1274u);
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < 1273u);
    dst.w  =  mp;
    mp = hip_canny_mag_phase_L1(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < 1272u);
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_CannySobel_U16_U8_3x3_L1NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L1NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L2NORM(uint dstWidth, uint dstHeight,
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
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum2.data[1] = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum2.data[2] = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum2.data[3] = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum2.data[4] = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum2.data[5] = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum2.data[6] = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum2.data[7] = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum1.data[0] = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack0(pix.y);
    sum1.data[1] = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack1(pix.y);
    sum1.data[0] = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum1.data[2] = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack2(pix.y);
    sum1.data[1] = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum1.data[3] = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack3(pix.y);
    sum1.data[2] = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum1.data[4] = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[3] = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum1.data[5] = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    fval = hip_unpack1(pix.x);
    sum1.data[4] = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum1.data[6] = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack2(pix.x);
    sum1.data[5] = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum1.data[7] = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[6] = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum2.data[1] = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum2.data[2] = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum2.data[3] = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum2.data[4] = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum2.data[5] = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum2.data[6] = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum2.data[7] = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 1);
    mask = HIPSELECT(0u, mask, y < 719);
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L2(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < 1274u);
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < 1273u);
    dst.w  =  mp;
    mp = hip_canny_mag_phase_L2(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < 1272u);
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_CannySobel_U16_U8_3x3_L2NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L2NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const uchar *xyStack, uint capacityOfXY, uint2 hyst,
    uint dstWidthComp) {

    __shared__ uchar lbuf[2448];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidthComp) && (y < dstHeight);

    uint dstIdx =  y * dstImageStrideInBytes + (x << 2);

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

    uchar *lbuf_ptr = lbuf + ly * 136 + (lx << 3);
    uint4 L0 = *((uint4 *)(&lbuf_ptr[0]));
    uint4 L1 = *((uint4 *)(&lbuf_ptr[136]));
    uint4 L2 = *((uint4 *)(&lbuf_ptr[272]));
    uint3 NA, NB, NC;
    uint T, M1, M2;
    uint4 M;

    NA.x = L0.x >> 18;
    NA.y = L1.x >> 18;
    NA.z = L2.x >> 18;
    NB.x = hip_bfe(L0.y, 2, 14);
    NB.y = hip_bfe(L1.y, 2, 14);
    NB.z = hip_bfe(L2.y, 2, 14);
    NC.x = L0.y >> 18;
    NC.y = L1.y >> 18;
    NC.z = L2.y >> 18;
    T = hip_bfe(L1.y,  0, 2);
    M1 = HIPSELECT(NA.y, NA.x, T > 0);
    M1 = HIPSELECT(M1, NB.x, T > 1);
    M1 = HIPSELECT(M1, NA.z, T > 2);
    M2 = HIPSELECT(NC.y, NC.z+1, T > 0);
    M2 = HIPSELECT(M2, NB.z, T > 1);
    M2 = HIPSELECT(M2, NC.x+1, T > 2);
    M.x = HIPSELECT(0u, NB.y, NB.y > M1);
    M.x = HIPSELECT(0u, M.x, NB.y >= M2);

    NA.x = hip_bfe(L0.z, 2, 14);
    NA.y = hip_bfe(L1.z, 2, 14);
    NA.z = hip_bfe(L2.z, 2, 14);
    T = hip_bfe(L1.y, 16, 2);
    M1 = HIPSELECT(NB.y, NB.x, T > 0);
    M1 = HIPSELECT(M1, NC.x, T > 1);
    M1 = HIPSELECT(M1, NB.z, T > 2);
    M2 = HIPSELECT(NA.y, NA.z+1, T > 0);
    M2 = HIPSELECT(M2, NC.z, T > 1);
    M2 = HIPSELECT(M2, NA.x+1, T > 2);
    M.y = HIPSELECT(0u, NC.y, NC.y > M1);
    M.y = HIPSELECT(0u, M.y, NC.y >= M2);
    NB.x = L0.z >> 18;
    NB.y = L1.z >> 18;
    NB.z = L2.z >> 18;
    T = hip_bfe(L1.z, 0, 2);
    M1 = HIPSELECT(NC.y, NC.x, T > 0);
    M1 = HIPSELECT(M1, NA.x, T > 1);
    M1 = HIPSELECT(M1, NC.z, T > 2);
    M2 = HIPSELECT(NB.y, NB.z+1, T > 0);
    M2 = HIPSELECT(M2, NA.z, T > 1);
    M2 = HIPSELECT(M2, NB.x+1, T > 2);
    M.z = HIPSELECT(0u, NA.y, NA.y > M1);
    M.z = HIPSELECT(0u, M.z, NA.y >= M2);
    NC.x = hip_bfe(L0.w, 2, 14);
    NC.y = hip_bfe(L1.w, 2, 14);
    NC.z = hip_bfe(L2.w, 2, 14);
    T = hip_bfe(L1.z, 16, 2);
    M1 = HIPSELECT(NA.y, NA.x, T > 0);
    M1 = HIPSELECT(M1, NB.x, T > 1);
    M1 = HIPSELECT(M1, NA.z, T > 2);
    M2 = HIPSELECT(NC.y, NC.z+1, T > 0);
    M2 = HIPSELECT(M2, NB.z, T > 1);
    M2 = HIPSELECT(M2, NC.x+1, T > 2);
    M.w = HIPSELECT(0u, NB.y, NB.y > M1);
    M.w = HIPSELECT(0u, M.w, NB.y >= M2);

    uint mask = HIPSELECT(0u, 0xffffffffu, x < 320u);
    mask = HIPSELECT(0u, mask, y < 720u);
    M.x &= mask;
    M.y &= mask;
    M.z &= mask;
    M.w &= mask;
    uint4 P;
    P.x = HIPSELECT(0u, 127u, M.x > hyst.x);
    P.y = HIPSELECT(0u, 127u, M.y > hyst.x);
    P.z = HIPSELECT(0u, 127u, M.z > hyst.x);
    P.w = HIPSELECT(0u, 127u, M.w > hyst.x);
    P.x = HIPSELECT(P.x, 255u, M.x > hyst.y);
    P.y = HIPSELECT(P.y, 255u, M.y > hyst.y);
    P.z = HIPSELECT(P.z, 255u, M.z > hyst.y);
    P.w = HIPSELECT(P.w, 255u, M.w > hyst.y);
    uint p0 = P.x;
    p0 += P.y << 8;
    p0 += P.z << 16;
    p0 += P.w << 24;

    if (valid) {
        *((uint *)(&pDstImage[dstIdx])) = p0;
        uint stack_icount;
        stack_icount  = HIPSELECT(0u, 1u, P.x == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.y == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.z == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.w == 255u);
        if (stack_icount > 0) {
            uint pos = atomicAdd((uint *)xyStack, stack_icount);
            uint *xyStackPtr = (uint *)&xyStack[0];
            uint xyloc = (y << 16) + (x << 2);
            if(pos < capacityOfXY && P.x == 255u)
                xyStackPtr[pos++] = xyloc;
            if(pos < capacityOfXY && P.y == 255u)
                xyStackPtr[pos++] = xyloc + 1;
            if(pos < capacityOfXY && P.z == 255u)
                xyStackPtr[pos++] = xyloc + 2;
            if(pos < capacityOfXY && P.w == 255u)
                xyStackPtr[pos++] = xyloc + 3;
        }
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_3x3(hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    uint2 hyst;
    hyst.x = (uint) hyst_lower;
    hyst.y = (uint) hyst_upper;

    uint dstWidthComp = (dstWidth + 3) / 4;

    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (const uchar *)xyStack, capacityOfXY, hyst,
                        dstWidthComp);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxFastCorners kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_NoSupression(uint capacityOfDstCorner, char *pDstCorners,
    uint srcWidth, uint srcHeight,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float strength_threshold) {

    int idx = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) + 3;
    int idy = (hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y) + 3;
    int stride = (int) srcImageStrideInBytes;
    if((idx > (int)srcWidth - 3) || (idy > (int)srcHeight - 3))
        return;
    const uchar *pTempImg = pSrcImage + hip_mad24(idy, stride, idx);
    int centerPixel_neg = pTempImg[0];
    int centerPixel_pos = centerPixel_neg + (int)strength_threshold;
    centerPixel_neg -= (int)strength_threshold;
    int candp, candn, pos_mask, neg_mask;
    candp = pTempImg[3];
    candn = pTempImg[-3];
    neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);
    pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);
    int offs = -stride*3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));
    pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));
    if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0)
        return;

    offs = -stride*3 + 1;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));
    pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));

    offs = -stride*3 - 1;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));
    pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));

    offs = -(stride << 1) + 2;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));
    pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));

    offs = -(stride << 1) - 2;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));
    pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));

    offs = -stride + 3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));
    pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));

    offs = -stride - 3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));
    pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));

    pos_mask |= (pos_mask << 16);
    neg_mask |= (neg_mask << 16);

    int cornerMask = 511, isCorner = 0;

    for(int i = 0; i < 16; i++)	{
        isCorner += ((pos_mask & cornerMask) == cornerMask);
        isCorner += ((neg_mask & cornerMask) == cornerMask);
        pos_mask >>= 1;
        neg_mask >>= 1;
    }

    uint *numKeypoints = (uint *) pDstCorners;
    d_KeyPt *keypt_list = (d_KeyPt *) pDstCorners;
    if(isCorner) {
        uint old_idx = atomicInc(numKeypoints, 1);
        if(old_idx < capacityOfDstCorner) {
            keypt_list[old_idx].x = idx;
            keypt_list[old_idx].y = idy;
            keypt_list[old_idx].strength = strength_threshold;
            keypt_list[old_idx].scale = 0;
            keypt_list[old_idx].orientation = 0;
            keypt_list[old_idx].tracking_status = 1;
            keypt_list[old_idx].error = 0;
        }
    }
}
int HipExec_FastCorners_XY_U8_NoSupression(hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_keypoint_t pHipDstCorner[], vx_uint32 *pHipDstCornerCount,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_float32 strength_threshold) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = ceil((srcWidth - 4) / 14) * 16;
    int globalThreads_y = ceil((srcHeight - 4) / 14) * 16;

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_NoSupression, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, capacityOfDstCorner, (char *) pHipDstCorner,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        strength_threshold);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_Supression(uint capacityOfDstCorner, char *pDstCorners,
    uint srcWidth, uint srcHeight,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float strength_threshold) {

    int lidx = hipThreadIdx_x;
    int lidy = hipThreadIdx_y;
    int gidx = hipBlockIdx_x;
    int gidy = hipBlockIdx_y;
    int xoffset = gidx * 14 + lidx + 2;
    int yoffset = gidy * 14 + lidy + 2;
    const uchar *pTempImg = pSrcImage + hip_mad24(yoffset, (int)srcImageStrideInBytes, xoffset);
    int pLocalStrengthShare[16][16];
    bool doCompute = true;
    if((xoffset > (int)srcWidth - 3) || (yoffset > (int)srcHeight - 3) || (xoffset < 3) || (yoffset < 3)) {
        doCompute = false;
        pLocalStrengthShare[lidy][lidx] = 0;
    }

    int local_strength;
    if(doCompute) {
        int boundary[16];
        int pos_mask, neg_mask, offs;
        int centerPixel_neg = pTempImg[0];
        for(int i = 0; i < 16; i++)
            boundary[i] = centerPixel_neg;
        int centerPixel_pos = centerPixel_neg + (int)strength_threshold;
        centerPixel_neg -= (int) strength_threshold;
        int candp = pTempImg[3];
        int candn = pTempImg[-3];
        neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);
        pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);
        boundary[0] -= candp;
        boundary[8] -= candn;
        offs = -srcImageStrideInBytes * 3;
        candp = pTempImg[offs];
        candn = pTempImg[-offs];
        neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));
        pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));
        boundary[4] -= candp;
        boundary[12] -= candn;
        if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0) {
            pLocalStrengthShare[lidy][lidx] = 0;
            doCompute = false;
        }
        else {
            offs = -srcImageStrideInBytes*3 + 1;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));
            pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));
            boundary[3] -= candp;
            boundary[11] -= candn;

            offs = -srcImageStrideInBytes*3 - 1;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));
            pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));
            boundary[5] -= candp;
            boundary[13] -= candn;

            offs = -(srcImageStrideInBytes<<1) + 2;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));
            pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));
            boundary[2] -= candp;
            boundary[10] -= candn;

            offs = -(srcImageStrideInBytes<<1) - 2;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));
            pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));
            boundary[6] -= candp;
            boundary[14] -= candn;

            offs = -srcImageStrideInBytes + 3;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));
            pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));
            boundary[1] -= candp;
            boundary[9] -= candn;

            offs = -srcImageStrideInBytes - 3;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));
            pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));
            boundary[7] -= candp;
            boundary[15] -= candn;

            pos_mask |= (pos_mask << 16);
            neg_mask |= (neg_mask << 16);

            int cornerMask = 511;
            int isCorner = 0;

            for (int i = 0; i < 16; i++) {
                isCorner += ((pos_mask & cornerMask) == cornerMask);
                isCorner += ((neg_mask & cornerMask) == cornerMask);
                pos_mask >>= 1;
                neg_mask >>= 1;
            }

            if(isCorner == 0) {
                pLocalStrengthShare[lidy][lidx] = 0;
                doCompute = false;
            }
            else {
                int strength;
                int tmp = 0;
                for (int i = 0; i < 16; i += 2)	{
                    int s = min(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);
                    s = min(s, boundary[(i + 3) & 15]);
                    s = min(s, boundary[(i + 4) & 15]);
                    s = min(s, boundary[(i + 5) & 15]);
                    s = min(s, boundary[(i + 6) & 15]);
                    s = min(s, boundary[(i + 7) & 15]);
                    s = min(s, boundary[(i + 8) & 15]);
                    tmp = max(tmp, min(s, boundary[i & 15]));
                    tmp = max(tmp, min(s, boundary[(i + 9) & 15]));
                }
                strength = -tmp;
                for (int i = 0; i < 16; i += 2)	{
                    int s = max(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);
                    s = max(s, boundary[(i + 3) & 15]);
                    s = max(s, boundary[(i + 4) & 15]);
                    s = max(s, boundary[(i + 5) & 15]);
                    s = max(s, boundary[(i + 6) & 15]);
                    s = max(s, boundary[(i + 7) & 15]);
                    s = max(s, boundary[(i + 8) & 15]);
                    strength = min(strength, max(s, boundary[i & 15]));
                    strength = min(strength, max(s, boundary[(i + 9) & 15]));
                }

                local_strength = -strength - 1;
                pLocalStrengthShare[lidy][lidx] = local_strength;
            }
        }
    }
    __syncthreads();

    bool writeCorner = doCompute &&
                        (local_strength >= pLocalStrengthShare[lidy-1][lidx-1]) &&
                        (local_strength >= pLocalStrengthShare[lidy-1][lidx]) &&
                        (local_strength >= pLocalStrengthShare[lidy-1][lidx+1]) &&
                        (local_strength >= pLocalStrengthShare[lidy][lidx-1]) &&
                        (local_strength > pLocalStrengthShare[lidy][lidx+1]) &&
                        (local_strength > pLocalStrengthShare[lidy+1][lidx-1]) &&
                        (local_strength > pLocalStrengthShare[lidy+1][lidx]) &&
                        (local_strength >= pLocalStrengthShare[lidy+1][lidx+1]) &&
                        (lidx > 0) &&
                        (lidy > 0) &&
                        (lidx < 15) &&
                        (lidy < 15);

    uint *numKeypoints = (uint *) pDstCorners;
    d_KeyPt *keypt_list = (d_KeyPt *) pDstCorners;
    if(writeCorner)	{
        uint old_idx = atomicInc(numKeypoints, 1);
        if(old_idx < capacityOfDstCorner) {
            keypt_list[old_idx].x = xoffset;
            keypt_list[old_idx].y = yoffset;
            keypt_list[old_idx].strength = (float) local_strength;
            keypt_list[old_idx].scale = 0;
            keypt_list[old_idx].orientation = 0;
            keypt_list[old_idx].tracking_status = 1;
            keypt_list[old_idx].error = 0;
        }
    }
}
int HipExec_FastCorners_XY_U8_Supression(hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_keypoint_t pHipDstCorner[], vx_uint32 *pHipDstCornerCount,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_float32 strength_threshold, vx_uint8 *pHipScratch) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = ceil((srcWidth - 4) / 14) * 16;
    int globalThreads_y = ceil((srcHeight - 4) / 14) * 16;

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_Supression, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, capacityOfDstCorner, (char *) pHipDstCorner,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        strength_threshold);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxHarrisCorners kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstGxy, uint dstGxyStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp1, uint dstWidthComp2) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) << 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

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
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum2.data[1] = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum2.data[2] = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum2.data[3] = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum2.data[4] = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum2.data[5] = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum2.data[6] = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum2.data[7] = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum1.data[0] = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack0(pix.y);
    sum1.data[1] = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack1(pix.y);
    sum1.data[0] = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum1.data[2] = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack2(pix.y);
    sum1.data[1] = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum1.data[3] = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack3(pix.y);
    sum1.data[2] = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum1.data[4] = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[3] = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum1.data[5] = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    fval = hip_unpack1(pix.x);
    sum1.data[4] = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum1.data[6] = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack2(pix.x);
    sum1.data[5] = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum1.data[7] = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[6] = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum2.data[1] = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum2.data[2] = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum2.data[3] = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum2.data[4] = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum2.data[5] = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum2.data[6] = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum2.data[7] = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    if ((x < dstWidth) && (y < dstHeight)) {
        uint dstIdx =  y * dstGxyStrideInBytes + (x << 2);

        float4 sum1X, sum1Y, sum2X, sum2Y;
        sum1X = make_float4(sum1.data[0], sum1.data[1], sum1.data[2], sum1.data[3]);
        sum1Y = make_float4(sum1.data[4], sum1.data[5], sum1.data[6], sum1.data[7]);
        sum2X = make_float4(sum2.data[0], sum2.data[1], sum2.data[2], sum2.data[3]);
        sum2Y = make_float4(sum2.data[4], sum2.data[5], sum2.data[6], sum2.data[7]);

        d_float8 dst;

        *(float4 *)(&dst.data[0]) = sum1X * sum1X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum1Y;
        *((d_float8 *)(&pDstGxy[dstIdx])) = dst;

        *(float4 *)(&dst.data[0]) = sum1X * sum2X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp1])) = dst;

        *(float4 *)(&dst.data[0]) = sum2X * sum2X;
        *(float4 *)(&dst.data[4]) = sum2Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp2])) = dst;
    }

}
int HipExec_HarrisSobel_HG3_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstGxy, dstGxyStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstVc, uint dstVcStrideInBytes,
    uchar *pSrcGxy, uint srcGxyStrideInBytes,
    float sensitivity, float strength_threshold,
    int border, float normFactor,
    uint dstWidthComp1, uint dstWidthComp2) {

    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[4896];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;

    int gstride = srcGxyStrideInBytes;
    uint dstIdx =  gy * dstVcStrideInBytes + (gx << 4);

    uchar *gbuf;
    gbuf = pSrcGxy;
    __shared__ uchar *lbuf_ptr;
    float2 v2;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum0;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum0.x  = v2.x;
    sum0.x += v2.y;
    sum0.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z  = v2.x;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.w += v2.y;
    *(float4 *)lbuf_ptr = sum0;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum0.x  = v2.x;
        sum0.x += v2.y;
        sum0.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z  = v2.x;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.w += v2.y;
        *((float4 *)&lbuf_ptr[4352]) = sum0;
    }

    __syncthreads();

    sum0  = *(float4 *)lbuf_ptr;
    sum0 += *(float4 *)&lbuf_ptr[272];
    sum0 += *(float4 *)&lbuf_ptr[544];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp1;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum1;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum1.x  = v2.x;
    sum1.x += v2.y;
    sum1.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z  = v2.x;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.w += v2.y;
    *(float4 *)lbuf_ptr = sum1;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum1.x  = v2.x;
        sum1.x += v2.y;
        sum1.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z  = v2.x;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum1;
    }

    __syncthreads();

    sum1  = *(float4 *)lbuf_ptr;
    sum1 += *(float4 *)&lbuf_ptr[272];
    sum1 += *(float4 *)&lbuf_ptr[544];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp2;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum2;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum2.x  = v2.x;
    sum2.x += v2.y;
    sum2.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z  = v2.x;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.w += v2.y;
    *(float4 *)lbuf_ptr = sum2;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum2.x  = v2.x;
        sum2.x += v2.y;
        sum2.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z  = v2.x;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum2;
    }

    __syncthreads();

    sum2  = *(float4 *)lbuf_ptr;
    sum2 += *(float4 *)&lbuf_ptr[272];
    sum2 += *(float4 *)&lbuf_ptr[544];

    gx = gx << 2;
    if ((gx < dstWidth) && (gy < dstHeight)) {
        float4 score = (float4)0.0f;
        if ((gy >= border) && (gy < dstHeight - border)) {
            score = sum0 * sum2 - sum1 * sum1;
            sum0 += sum2;
            sum0 *= sum0;
            score.x = fmaf(sum0.x, -sensitivity, score.x);
            score.y = fmaf(sum0.y, -sensitivity, score.y);
            score.z = fmaf(sum0.z, -sensitivity, score.z);
            score.w = fmaf(sum0.w, -sensitivity, score.w);
            score *= (float4)normFactor;
            score = HIPSELECT((float4)0.0f, score, (
                score.x > strength_threshold &&
                score.y > strength_threshold &&
                score.z > strength_threshold &&
                score.w > strength_threshold));
            score.x = HIPSELECT(score.x, 0.0f, gx < border);
            score.y = HIPSELECT(score.y, 0.0f, gx < border - 1);
            score.z = HIPSELECT(score.z, 0.0f, gx < border - 2);
            score.w = HIPSELECT(score.w, 0.0f, gx < border - 3);
            score.x = HIPSELECT(score.x, 0.0f, gx > dstWidth - 1 - border);
            score.y = HIPSELECT(score.y, 0.0f, gx > dstWidth - 2 - border);
            score.z = HIPSELECT(score.z, 0.0f, gx > dstWidth - 3 - border);
            score.w = HIPSELECT(score.w, 0.0f, gx > dstWidth - 4 - border);
        }
        *((float4 *)(&pDstVc[dstIdx])) = score;
    }
}
int HipExec_HarrisScore_HVC_HG3_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes, vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold, vx_int32 border, vx_float32 normFactor) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstVc, dstVcStrideInBytes,
                        (uchar *)pHipSrcGxy, srcGxyStrideInBytes, sensitivity, strength_threshold, border, normFactor,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_NonMaxSupp_XY_ANY_3x3(char *pDstList, uint capacityOfList,
    uint srcWidth, uint srcHeight,
    uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcWidthComp1, uint srcWidthComp2) {

    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int gstride = srcImageStrideInBytes;
    uchar *gbuf = pSrcImage;
    __shared__ uchar lbuf[2448];
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (gy - 1) * srcImageStrideInBytes + gx - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (gy - ly + id - 1) * srcImageStrideInBytes + (((gx >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    __shared__ uchar *lbuf_ptr;
    lbuf_ptr = lbuf + ly * 136 + (lx << 3);
    float4 L0 = *(float4 *)lbuf_ptr;
    float4 L1 = *(float4 *)&lbuf_ptr[136];
    float4 L2 = *(float4 *)&lbuf_ptr[272];
    float2 T = *(float2 *)&L1;
    T.x = HIPSELECT(0.0f, T.x, T.x >= L1.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L1.z);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.y);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.z);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.y);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L1.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L1.w);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.w);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.w);
    T.x = HIPSELECT(0.0f, T.x, gx < srcWidthComp1);
    T.y = HIPSELECT(0.0f, T.y, gx < srcWidthComp2);
    T.x = HIPSELECT(0.0f, T.x, gy < srcHeight);
    T.y = HIPSELECT(0.0f, T.y, gy < srcHeight);
    gx = gx + gx + HIPSELECT(0, 1, T.y > 0.0f);
    T.x = HIPSELECT(T.x, T.y, T.y > 0.0f);
    if (T.x > 0.0f) {
        uint pos = atomicInc((uint *)pDstList, 1);
        if(pos < capacityOfList) {
            *((uint2 *)(&pDstList[pos << 3])) = make_uint2(gx | (gy << 16), (uint)(T.x));
        }
    }
}
int HipExec_NonMaxSupp_XY_ANY_3x3(hipStream_t stream, vx_uint32 capacityOfList, ago_keypoint_xys_t *pHipDstList,
    vx_uint32 srcWidth, vx_uint32 srcHeight, vx_float32 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = srcWidth;
    int globalThreads_y = srcHeight;

    vx_uint32 srcWidthComp1 = (srcWidth + 1) / 2;
    vx_uint32 srcWidthComp2 = srcWidth / 2;

    hipLaunchKernelGGL(Hip_NonMaxSupp_XY_ANY_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, (char *)pHipDstList, capacityOfList,
                        srcWidth, srcHeight, (uchar *)pHipSrcImage, srcImageStrideInBytes,
                        srcWidthComp1, srcWidthComp2);

    return VX_SUCCESS;
}


























// VxHarrisCorners old kernels

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_5x5(
    unsigned int  dstWidth, unsigned int  dstHeight,
    float * pDstGxy_,unsigned int  dstGxyStrideInBytes,
    const unsigned char  * pSrcImage ,unsigned int srcImageStrideInBytes,
    float * gx, float *gy
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >dstWidth) || (x<0)|| (y > dstHeight-2) || y<2)	return;
    unsigned int dstIdx = y * (dstGxyStrideInBytes ) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + x;
    ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)( pDstGxy_ );
    float div_factor = 1; // 4.0f * 255;

    int srcIdxTopRow1, srcIdxTopRow2, srcIdxBottomRow1, srcIdxBottomRow2;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    float sum_x = 0;
    sum_x = (gx[12] * (float)*(pSrcImage + srcIdx) + gx[7] * (float)*(pSrcImage + srcIdxTopRow1) + gx[2] * (float)*(pSrcImage + srcIdxTopRow2) + gx[17] * (float)*(pSrcImage + srcIdxBottomRow1) + gx[22] * (float)*(pSrcImage + srcIdxBottomRow2) +
            gx[11] * (float)*(pSrcImage + srcIdx - 1) + gx[6] * (float)*(pSrcImage + srcIdxTopRow1 - 1) + gx[1] * (float)*(pSrcImage + srcIdxTopRow2 - 1) + gx[16] * (float)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[21] * (float)*(pSrcImage + srcIdxBottomRow2 - 1) +
            gx[10] * (float)*(pSrcImage + srcIdx - 2) + gx[5] * (float)*(pSrcImage + srcIdxTopRow1 - 2) + gx[0] * (float)*(pSrcImage + srcIdxTopRow2 - 2) + gx[15] * (float)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[20] * (float)*(pSrcImage + srcIdxBottomRow2 - 2) +
            gx[13] * (float)*(pSrcImage + srcIdx + 1) + gx[8] * (float)*(pSrcImage + srcIdxTopRow1 + 1) + gx[3] * (float)*(pSrcImage + srcIdxTopRow2 + 1) + gx[18] * (float)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[23] * (float)*(pSrcImage + srcIdxBottomRow2 + 1) +
            gx[14] * (float)*(pSrcImage + srcIdx + 2) + gx[9] * (float)*(pSrcImage + srcIdxTopRow1 + 2) + gx[4] * (float)*(pSrcImage + srcIdxTopRow2 + 2) + gx[19] * (float)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[24] * (float)*(pSrcImage + srcIdxBottomRow2 + 2));
    float sum_y = 0;
    sum_y = (gy[12] * (float)*(pSrcImage + srcIdx) + gy[7] * (float)*(pSrcImage + srcIdxTopRow1) + gy[2] * (float)*(pSrcImage + srcIdxTopRow2) + gy[17] * (float)*(pSrcImage + srcIdxBottomRow1) + gy[22] * (float)*(pSrcImage + srcIdxBottomRow2) +
            gy[11] * (float)*(pSrcImage + srcIdx - 1) + gy[6] * (float)*(pSrcImage + srcIdxTopRow1 - 1) + gy[1] * (float)*(pSrcImage + srcIdxTopRow2 - 1) + gy[16] * (float)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[21] * (float)*(pSrcImage + srcIdxBottomRow2 - 1) +
            gy[10] * (float)*(pSrcImage + srcIdx - 2) + gy[5] * (float)*(pSrcImage + srcIdxTopRow1 - 2) + gy[0] * (float)*(pSrcImage + srcIdxTopRow2 - 2) + gy[15] * (float)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[20] * (float)*(pSrcImage + srcIdxBottomRow2 - 2) +
            gy[13] * (float)*(pSrcImage + srcIdx + 1) + gy[8] * (float)*(pSrcImage + srcIdxTopRow1 + 1) + gy[3] * (float)*(pSrcImage + srcIdxTopRow2 + 1) + gy[18] * (float)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[23] * (float)*(pSrcImage + srcIdxBottomRow2 + 1) +
            gy[14] * (float)*(pSrcImage + srcIdx + 2) + gy[9] * (float)*(pSrcImage + srcIdxTopRow1 + 2) + gy[4] * (float)*(pSrcImage + srcIdxTopRow2 + 2) + gy[19] * (float)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[24] * (float)*(pSrcImage + srcIdxBottomRow2 + 2));


    pDstGxy[dstIdx].GxGx = sum_x * sum_x;
    pDstGxy[dstIdx].GxGy = sum_x * sum_y;
    pDstGxy[dstIdx].GyGy = sum_y * sum_y;
}
int HipExec_HarrisSobel_HG3_U8_5x5(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth),   globalThreads_y = dstHeight;

    float gx[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    float gy[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    float *hipGx, *hipGy;
    hipMalloc(&hipGx, 400);
    hipMalloc(&hipGy, 400);
    hipMemcpy(hipGx, gx, 400, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 400, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstGxy_ , (dstGxyStrideInBytes/sizeof(ago_harris_Gxy_t)),
                    (const unsigned char *)pSrcImage, srcImageStrideInBytes,
                    (float *)hipGx, (float *)hipGy);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_7x7(
    unsigned int  dstWidth, unsigned int  dstHeight,
    float * pDstGxy_,unsigned int  dstGxyStrideInBytes,
    const unsigned char  * pSrcImage ,unsigned int srcImageStrideInBytes,
    float * gx, float *gy
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >dstWidth) || (x<0)|| (y > dstHeight-3) || y<3)	return;
    unsigned int dstIdx = y * (dstGxyStrideInBytes ) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + x;
    ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)( pDstGxy_ );
    float div_factor = 1; // 4.0f * 255;

    int srcIdxTopRow1, srcIdxTopRow2, srcIdxTopRow3, srcIdxBottomRow1, srcIdxBottomRow2, srcIdxBottomRow3;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxTopRow3 = srcIdx - (3 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3 * srcImageStrideInBytes);
    float sum_x = 0;
    sum_x = (gx[24] * (float)*(pSrcImage + srcIdx) + gx[17] * (float)*(pSrcImage + srcIdxTopRow1) + gx[10] * (float)*(pSrcImage + srcIdxTopRow2) + gx[3] * (float)*(pSrcImage + srcIdxTopRow3) + gx[31] * (float)*(pSrcImage + srcIdxBottomRow1) + gx[38] * (float)*(pSrcImage + srcIdxBottomRow2) + gx[45] * (float)*(pSrcImage + srcIdxBottomRow3) +
            gx[23] * (float)*(pSrcImage + srcIdx - 1) + gx[16] * (float)*(pSrcImage + srcIdxTopRow1 - 1) + gx[9] * (float)*(pSrcImage + srcIdxTopRow2 - 1) + gx[2] * (float)*(pSrcImage + srcIdxTopRow3 - 1) + gx[30] * (float)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[37] * (float)*(pSrcImage + srcIdxBottomRow2 - 1) + gx[44] * (float)*(pSrcImage + srcIdxBottomRow3 - 1) +
            gx[22] * (float)*(pSrcImage + srcIdx - 2) + gx[15] * (float)*(pSrcImage + srcIdxTopRow1 - 2) + gx[8] * (float)*(pSrcImage + srcIdxTopRow2 - 2) + gx[1] * (float)*(pSrcImage + srcIdxTopRow3 - 2) + gx[29] * (float)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[36] * (float)*(pSrcImage + srcIdxBottomRow2 - 2) + gx[43] * (float)*(pSrcImage + srcIdxBottomRow3 - 2) +
            gx[21] * (float)*(pSrcImage + srcIdx - 3) + gx[14] * (float)*(pSrcImage + srcIdxTopRow1 - 3) + gx[7] * (float)*(pSrcImage + srcIdxTopRow2 - 3) + gx[0] * (float)*(pSrcImage + srcIdxTopRow3 - 3) + gx[28] * (float)*(pSrcImage + srcIdxBottomRow1 - 3) + gx[35] * (float)*(pSrcImage + srcIdxBottomRow2 - 3) + gx[42] * (float)*(pSrcImage + srcIdxBottomRow3 - 3) +
            gx[25] * (float)*(pSrcImage + srcIdx + 1) + gx[18] * (float)*(pSrcImage + srcIdxTopRow1 + 1) + gx[11] * (float)*(pSrcImage + srcIdxTopRow2 + 1) + gx[4] * (float)*(pSrcImage + srcIdxTopRow3 + 1) + gx[32] * (float)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[39] * (float)*(pSrcImage + srcIdxBottomRow2 + 1) + gx[46] * (float)*(pSrcImage + srcIdxBottomRow3 + 1) +
            gx[26] * (float)*(pSrcImage + srcIdx + 2) + gx[19] * (float)*(pSrcImage + srcIdxTopRow1 + 2) + gx[12] * (float)*(pSrcImage + srcIdxTopRow2 + 2) + gx[5] * (float)*(pSrcImage + srcIdxTopRow3 + 2) + gx[33] * (float)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[40] * (float)*(pSrcImage + srcIdxBottomRow2 + 2) + gx[47] * (float)*(pSrcImage + srcIdxBottomRow3 + 2) +
            gx[27] * (float)*(pSrcImage + srcIdx + 3) + gx[20] * (float)*(pSrcImage + srcIdxTopRow1 + 3) + gx[13] * (float)*(pSrcImage + srcIdxTopRow2 + 3) + gx[6] * (float)*(pSrcImage + srcIdxTopRow3 + 3) + gx[34] * (float)*(pSrcImage + srcIdxBottomRow1 + 3) + gx[41] * (float)*(pSrcImage + srcIdxBottomRow2 + 3) + gx[48] * (float)*(pSrcImage + srcIdxBottomRow3 + 3));
    float sum_y = 0;
    sum_y = (gy[24] * (float)*(pSrcImage + srcIdx) + gy[17] * (float)*(pSrcImage + srcIdxTopRow1) + gy[10] * (float)*(pSrcImage + srcIdxTopRow2) + gy[3] * (float)*(pSrcImage + srcIdxTopRow3) + gy[31] * (float)*(pSrcImage + srcIdxBottomRow1) + gy[38] * (float)*(pSrcImage + srcIdxBottomRow2) + gy[45] * (float)*(pSrcImage + srcIdxBottomRow3) +
            gy[23] * (float)*(pSrcImage + srcIdx - 1) + gy[16] * (float)*(pSrcImage + srcIdxTopRow1 - 1) + gy[9] * (float)*(pSrcImage + srcIdxTopRow2 - 1) + gy[2] * (float)*(pSrcImage + srcIdxTopRow3 - 1) + gy[30] * (float)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[37] * (float)*(pSrcImage + srcIdxBottomRow2 - 1) + gy[44] * (float)*(pSrcImage + srcIdxBottomRow3 - 1) +
            gy[22] * (float)*(pSrcImage + srcIdx - 2) + gy[15] * (float)*(pSrcImage + srcIdxTopRow1 - 2) + gy[8] * (float)*(pSrcImage + srcIdxTopRow2 - 2) + gy[1] * (float)*(pSrcImage + srcIdxTopRow3 - 2) + gy[29] * (float)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[36] * (float)*(pSrcImage + srcIdxBottomRow2 - 2) + gy[43] * (float)*(pSrcImage + srcIdxBottomRow3 - 2) +
            gy[21] * (float)*(pSrcImage + srcIdx - 3) + gy[14] * (float)*(pSrcImage + srcIdxTopRow1 - 3) + gy[7] * (float)*(pSrcImage + srcIdxTopRow2 - 3) + gy[0] * (float)*(pSrcImage + srcIdxTopRow3 - 3) + gy[28] * (float)*(pSrcImage + srcIdxBottomRow1 - 3) + gy[35] * (float)*(pSrcImage + srcIdxBottomRow2 - 3) + gy[42] * (float)*(pSrcImage + srcIdxBottomRow3 - 3) +
            gy[25] * (float)*(pSrcImage + srcIdx + 1) + gy[18] * (float)*(pSrcImage + srcIdxTopRow1 + 1) + gy[11] * (float)*(pSrcImage + srcIdxTopRow2 + 1) + gy[4] * (float)*(pSrcImage + srcIdxTopRow3 + 1) + gy[32] * (float)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[39] * (float)*(pSrcImage + srcIdxBottomRow2 + 1) + gy[46] * (float)*(pSrcImage + srcIdxBottomRow3 + 1) +
            gy[26] * (float)*(pSrcImage + srcIdx + 2) + gy[19] * (float)*(pSrcImage + srcIdxTopRow1 + 2) + gy[12] * (float)*(pSrcImage + srcIdxTopRow2 + 2) + gy[5] * (float)*(pSrcImage + srcIdxTopRow3 + 2) + gy[33] * (float)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[40] * (float)*(pSrcImage + srcIdxBottomRow2 + 2) + gy[47] * (float)*(pSrcImage + srcIdxBottomRow3 + 2) +
            gy[27] * (float)*(pSrcImage + srcIdx + 3) + gy[20] * (float)*(pSrcImage + srcIdxTopRow1 + 3) + gy[13] * (float)*(pSrcImage + srcIdxTopRow2 + 3) + gy[6] * (float)*(pSrcImage + srcIdxTopRow3 + 3) + gy[34] * (float)*(pSrcImage + srcIdxBottomRow1 + 3) + gy[41] * (float)*(pSrcImage + srcIdxBottomRow2 + 3) + gy[48] * (float)*(pSrcImage + srcIdxBottomRow3 + 3));

    pDstGxy[dstIdx].GxGx = sum_x * sum_x;
    pDstGxy[dstIdx].GxGy = sum_x * sum_y;
    pDstGxy[dstIdx].GyGy = sum_y * sum_y;
}
int HipExec_HarrisSobel_HG3_U8_7x7(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth),   globalThreads_y = dstHeight;

    float gx[49] = {-1,-4,-5,0,5,4,1,-6,-24,-30,0,30,24,6,-15,-60,-75,0,75,60,15,-20,-80,-100,0,100,80,20,-15,-60,-75,0,75,60,15,-6,-24,-30,0,30,24,6,-1,-4,-5,0,5,4,1};
    float gy[49] = {-1,-6,-15,-20,-15,-6,-1,-4,-24,-60,-80,-60,-24,-4,-5,-30,-75,-100,-75,-30,-5,0,0,0,0,0,0,0,5,30,75,100,75,30,5,4,24,60,80,60,24,4,1,6,15,20,15,6,1};
    float *hipGx, *hipGy;
    hipMalloc(&hipGx, 784);
    hipMalloc(&hipGy, 784);
    hipMemcpy(hipGx, gx, 784, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 784, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstGxy_ , (dstGxyStrideInBytes/sizeof(ago_harris_Gxy_t)),
                    (const unsigned char *)pSrcImage, srcImageStrideInBytes,
                    (float *)hipGx, (float *)hipGy);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_5x5(
    unsigned int dstWidth, unsigned int dstHeight,
    float *pDstVc, unsigned int dstVcStrideInBytes,
    float *pSrcGxy_, unsigned int srcGxyStrideInBytes,
    float sensitivity, float strength_threshold,
    float normalization_factor
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    unsigned int dstIdx = y * (dstVcStrideInBytes) + x;
    unsigned int srcIdx = y * (srcGxyStrideInBytes) + x;

    if ((x >= dstWidth-2) || (x <= 2) || (y >= dstHeight-2 ) || y <= 2)	{
        pDstVc[dstIdx] = (float)0;
        return;
    }

    float gx2 = 0, gy2 = 0, gxy2 = 0;
    float traceA =0, detA =0, Mc =0;
    ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
    //Prev Row + Current Row + Next Row sum of gx2, gxy2, gy2
    int srcIdxTopRow1, srcIdxBottomRow1,srcIdxBottomRow2,srcIdxTopRow2;
    srcIdxTopRow2 = srcIdx - (2*srcGxyStrideInBytes);
    srcIdxTopRow1 = srcIdx - srcGxyStrideInBytes;
    srcIdxBottomRow1 = srcIdx + srcGxyStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2*srcGxyStrideInBytes);
    gx2 =
    (float)pSrcGxy[srcIdxTopRow2 - 2].GxGx + (float)pSrcGxy[srcIdxTopRow2 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow2].GxGx +(float)pSrcGxy[srcIdxTopRow2 + 1].GxGx + (float)pSrcGxy[srcIdxTopRow2 + 2].GxGx +
    (float)pSrcGxy[srcIdxTopRow1 - 2].GxGx + (float)pSrcGxy[srcIdxTopRow1 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow1].GxGx +(float)pSrcGxy[srcIdxTopRow1 + 1].GxGx + (float)pSrcGxy[srcIdxTopRow1 + 2].GxGx +
    (float)pSrcGxy[srcIdx-2].GxGx + (float)pSrcGxy[srcIdx-1].GxGx + (float)pSrcGxy[srcIdx].GxGx + (float)pSrcGxy[srcIdx+1].GxGx + (float)pSrcGxy[srcIdx+2].GxGx +
    (float)pSrcGxy[srcIdxBottomRow1 -2].GxGx + (float)pSrcGxy[srcIdxBottomRow1 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow1].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 2].GxGx +
    (float)pSrcGxy[srcIdxBottomRow2 -2].GxGx + (float)pSrcGxy[srcIdxBottomRow2 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow2].GxGx + (float)pSrcGxy[srcIdxBottomRow2 + 1].GxGx + (float)pSrcGxy[srcIdxBottomRow2 + 2].GxGx ;

    gxy2 =
    (float)pSrcGxy[srcIdxTopRow2 - 2].GxGy + (float)pSrcGxy[srcIdxTopRow2 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow2].GxGy +(float)pSrcGxy[srcIdxTopRow2 + 1].GxGy + (float)pSrcGxy[srcIdxTopRow2 + 2].GxGy +
    (float)pSrcGxy[srcIdxTopRow1 - 2].GxGy + (float)pSrcGxy[srcIdxTopRow1 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow1].GxGy +(float)pSrcGxy[srcIdxTopRow1 + 1].GxGy + (float)pSrcGxy[srcIdxTopRow1 + 2].GxGy +
    (float)pSrcGxy[srcIdx-2].GxGy + (float)pSrcGxy[srcIdx-1].GxGy + (float)pSrcGxy[srcIdx].GxGy + (float)pSrcGxy[srcIdx+1].GxGy + (float)pSrcGxy[srcIdx+2].GxGy +
    (float)pSrcGxy[srcIdxBottomRow1 -2].GxGy + (float)pSrcGxy[srcIdxBottomRow1 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow1].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 2].GxGy +
    (float)pSrcGxy[srcIdxBottomRow2 -2].GxGy + (float)pSrcGxy[srcIdxBottomRow2 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow2].GxGy + (float)pSrcGxy[srcIdxBottomRow2 + 1].GxGy + (float)pSrcGxy[srcIdxBottomRow2 + 2].GxGy ;

    gy2 =
    (float)pSrcGxy[srcIdxTopRow2 - 2].GyGy + (float)pSrcGxy[srcIdxTopRow2 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow2].GyGy +(float)pSrcGxy[srcIdxTopRow2 + 1].GyGy + (float)pSrcGxy[srcIdxTopRow2 + 2].GyGy +
    (float)pSrcGxy[srcIdxTopRow1 - 2].GyGy + (float)pSrcGxy[srcIdxTopRow1 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow1].GyGy +(float)pSrcGxy[srcIdxTopRow1 + 1].GyGy + (float)pSrcGxy[srcIdxTopRow1 + 2].GyGy +
    (float)pSrcGxy[srcIdx-2].GyGy + (float)pSrcGxy[srcIdx-1].GyGy + (float)pSrcGxy[srcIdx].GyGy + (float)pSrcGxy[srcIdx+1].GyGy + (float)pSrcGxy[srcIdx+2].GyGy +
    (float)pSrcGxy[srcIdxBottomRow1 -2].GyGy + (float)pSrcGxy[srcIdxBottomRow1 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow1].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 2].GyGy +
    (float)pSrcGxy[srcIdxBottomRow2 -2].GyGy + (float)pSrcGxy[srcIdxBottomRow2 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow2].GyGy + (float)pSrcGxy[srcIdxBottomRow2 + 1].GyGy + (float)pSrcGxy[srcIdxBottomRow2 + 2].GyGy ;

    traceA = gx2 + gy2;
    detA = (gx2 * gy2) - (gxy2 * gxy2);
    Mc = detA - (sensitivity * traceA * traceA);
    Mc /= normalization_factor;
    if(Mc > strength_threshold) {
        pDstVc[dstIdx] = (float)Mc;
    }
    else {
        pDstVc[dstIdx] = (float)0;
    }
}
int HipExec_HarrisScore_HVC_HG3_5x5(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
    vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold,
    vx_float32 normalization_factor
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth),   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstVc , (dstVcStrideInBytes/sizeof(float)),
                    (float *)pSrcGxy_, (srcGxyStrideInBytes/sizeof(ago_harris_Gxy_t)),
                    sensitivity, strength_threshold,normalization_factor );

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_7x7(
    unsigned int dstWidth, unsigned int dstHeight,
    float *pDstVc, unsigned int dstVcStrideInBytes,
    float *pSrcGxy_, unsigned int srcGxyStrideInBytes,
    float sensitivity, float strength_threshold,
    float normalization_factor
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    unsigned int dstIdx = y * (dstVcStrideInBytes) + x;
    unsigned int srcIdx = y * (srcGxyStrideInBytes) + x;

    if ((x >= dstWidth-3) || (x <= 3) || (y >= dstHeight-3 ) || y <= 3)	{
        pDstVc[dstIdx] = (float)0;
        return;
    }

    float gx2 = 0, gy2 = 0, gxy2 = 0;
    float traceA =0, detA =0, Mc =0;
    ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
    //Prev Row3 +Prev Row2 +Prev Row1 + Current Row + Next Row1+ Next Row2++ Next Row3 sum of gx2, gxy2, gy2
    int srcIdxTopRow1, srcIdxBottomRow1,srcIdxBottomRow2,srcIdxTopRow2,srcIdxTopRow3, srcIdxBottomRow3;
    srcIdxTopRow3 = srcIdx - (3*srcGxyStrideInBytes);
    srcIdxTopRow2 = srcIdx - (2*srcGxyStrideInBytes);
    srcIdxTopRow1 = srcIdx - srcGxyStrideInBytes;
    srcIdxBottomRow1 = srcIdx + srcGxyStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2*srcGxyStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3*srcGxyStrideInBytes);

    gx2 =
        (float)pSrcGxy[srcIdxTopRow3 - 3].GxGx + (float)pSrcGxy[srcIdxTopRow3 - 2].GxGx + (float)pSrcGxy[srcIdxTopRow3 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow3].GxGx +(float)pSrcGxy[srcIdxTopRow3 + 1].GxGx + (float)pSrcGxy[srcIdxTopRow3 + 2].GxGx + (float)pSrcGxy[srcIdxTopRow3 + 3].GxGx +
        (float)pSrcGxy[srcIdxTopRow2 - 3].GxGx + (float)pSrcGxy[srcIdxTopRow2 - 2].GxGx + (float)pSrcGxy[srcIdxTopRow2 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow2].GxGx +(float)pSrcGxy[srcIdxTopRow2 + 1].GxGx + (float)pSrcGxy[srcIdxTopRow2 + 2].GxGx + (float)pSrcGxy[srcIdxTopRow2 + 3].GxGx +
        (float)pSrcGxy[srcIdxTopRow1 - 3].GxGx +(float)pSrcGxy[srcIdxTopRow1 - 2].GxGx + (float)pSrcGxy[srcIdxTopRow1 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow1].GxGx +(float)pSrcGxy[srcIdxTopRow1 + 1].GxGx + (float)pSrcGxy[srcIdxTopRow1 + 2].GxGx +(float)pSrcGxy[srcIdxTopRow1 + 3].GxGx +
        (float)pSrcGxy[srcIdx-3].GxGx +(float)pSrcGxy[srcIdx-2].GxGx + (float)pSrcGxy[srcIdx-1].GxGx + (float)pSrcGxy[srcIdx].GxGx + (float)pSrcGxy[srcIdx+1].GxGx + (float)pSrcGxy[srcIdx+2].GxGx +(float)pSrcGxy[srcIdx+3].GxGx +
        (float)pSrcGxy[srcIdxBottomRow1 -3].GxGx + (float)pSrcGxy[srcIdxBottomRow1 -2].GxGx + (float)pSrcGxy[srcIdxBottomRow1 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow1].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 2].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 3].GxGx +
        (float)pSrcGxy[srcIdxBottomRow2 -3].GxGx + (float)pSrcGxy[srcIdxBottomRow2 -2].GxGx + (float)pSrcGxy[srcIdxBottomRow2 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow2].GxGx + (float)pSrcGxy[srcIdxBottomRow2 + 1].GxGx + (float)pSrcGxy[srcIdxBottomRow2 + 2].GxGx + (float)pSrcGxy[srcIdxBottomRow2 + 3].GxGx +
        (float)pSrcGxy[srcIdxBottomRow3 -3].GxGx + (float)pSrcGxy[srcIdxBottomRow3 -2].GxGx + (float)pSrcGxy[srcIdxBottomRow3 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow3].GxGx + (float)pSrcGxy[srcIdxBottomRow3 + 1].GxGx + (float)pSrcGxy[srcIdxBottomRow3 + 2].GxGx + (float)pSrcGxy[srcIdxBottomRow3 + 3].GxGx;

    gxy2 =
        (float)pSrcGxy[srcIdxTopRow3 - 3].GxGy + (float)pSrcGxy[srcIdxTopRow3 - 2].GxGy + (float)pSrcGxy[srcIdxTopRow3 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow3].GxGy +(float)pSrcGxy[srcIdxTopRow3 + 1].GxGy + (float)pSrcGxy[srcIdxTopRow3 + 2].GxGy + (float)pSrcGxy[srcIdxTopRow3 + 3].GxGy +
        (float)pSrcGxy[srcIdxTopRow2 - 3].GxGy + (float)pSrcGxy[srcIdxTopRow2 - 2].GxGy + (float)pSrcGxy[srcIdxTopRow2 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow2].GxGy +(float)pSrcGxy[srcIdxTopRow2 + 1].GxGy + (float)pSrcGxy[srcIdxTopRow2 + 2].GxGy + (float)pSrcGxy[srcIdxTopRow2 + 3].GxGy +
        (float)pSrcGxy[srcIdxTopRow1 - 3].GxGy +(float)pSrcGxy[srcIdxTopRow1 - 2].GxGy + (float)pSrcGxy[srcIdxTopRow1 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow1].GxGy +(float)pSrcGxy[srcIdxTopRow1 + 1].GxGy + (float)pSrcGxy[srcIdxTopRow1 + 2].GxGy +(float)pSrcGxy[srcIdxTopRow1 + 3].GxGy +
        (float)pSrcGxy[srcIdx-3].GxGy +(float)pSrcGxy[srcIdx-2].GxGy + (float)pSrcGxy[srcIdx-1].GxGy + (float)pSrcGxy[srcIdx].GxGy + (float)pSrcGxy[srcIdx+1].GxGy + (float)pSrcGxy[srcIdx+2].GxGy +(float)pSrcGxy[srcIdx+3].GxGy +
        (float)pSrcGxy[srcIdxBottomRow1 -3].GxGy + (float)pSrcGxy[srcIdxBottomRow1 -2].GxGy + (float)pSrcGxy[srcIdxBottomRow1 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow1].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 2].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 3].GxGy +
        (float)pSrcGxy[srcIdxBottomRow2 -3].GxGy + (float)pSrcGxy[srcIdxBottomRow2 -2].GxGy + (float)pSrcGxy[srcIdxBottomRow2 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow2].GxGy + (float)pSrcGxy[srcIdxBottomRow2 + 1].GxGy + (float)pSrcGxy[srcIdxBottomRow2 + 2].GxGy + (float)pSrcGxy[srcIdxBottomRow2 + 3].GxGy +
        (float)pSrcGxy[srcIdxBottomRow3 -3].GxGy + (float)pSrcGxy[srcIdxBottomRow3 -2].GxGy + (float)pSrcGxy[srcIdxBottomRow3 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow3].GxGy + (float)pSrcGxy[srcIdxBottomRow3 + 1].GxGy + (float)pSrcGxy[srcIdxBottomRow3 + 2].GxGy + (float)pSrcGxy[srcIdxBottomRow3 + 3].GxGy;

    gy2 =
        (float)pSrcGxy[srcIdxTopRow3 - 3].GyGy + (float)pSrcGxy[srcIdxTopRow3 - 2].GyGy + (float)pSrcGxy[srcIdxTopRow3 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow3].GyGy +(float)pSrcGxy[srcIdxTopRow3 + 1].GyGy + (float)pSrcGxy[srcIdxTopRow3 + 2].GyGy + (float)pSrcGxy[srcIdxTopRow3 + 3].GyGy +
        (float)pSrcGxy[srcIdxTopRow2 - 3].GyGy + (float)pSrcGxy[srcIdxTopRow2 - 2].GyGy + (float)pSrcGxy[srcIdxTopRow2 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow2].GyGy +(float)pSrcGxy[srcIdxTopRow2 + 1].GyGy + (float)pSrcGxy[srcIdxTopRow2 + 2].GyGy + (float)pSrcGxy[srcIdxTopRow2 + 3].GyGy +
        (float)pSrcGxy[srcIdxTopRow1 - 3].GyGy +(float)pSrcGxy[srcIdxTopRow1 - 2].GyGy + (float)pSrcGxy[srcIdxTopRow1 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow1].GyGy +(float)pSrcGxy[srcIdxTopRow1 + 1].GyGy + (float)pSrcGxy[srcIdxTopRow1 + 2].GyGy +(float)pSrcGxy[srcIdxTopRow1 + 3].GyGy +
        (float)pSrcGxy[srcIdx-3].GyGy +(float)pSrcGxy[srcIdx-2].GyGy + (float)pSrcGxy[srcIdx-1].GyGy + (float)pSrcGxy[srcIdx].GyGy + (float)pSrcGxy[srcIdx+1].GyGy + (float)pSrcGxy[srcIdx+2].GyGy +(float)pSrcGxy[srcIdx+3].GyGy +
        (float)pSrcGxy[srcIdxBottomRow1 -3].GyGy + (float)pSrcGxy[srcIdxBottomRow1 -2].GyGy + (float)pSrcGxy[srcIdxBottomRow1 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow1].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 2].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 3].GyGy +
        (float)pSrcGxy[srcIdxBottomRow2 -3].GyGy + (float)pSrcGxy[srcIdxBottomRow2 -2].GyGy + (float)pSrcGxy[srcIdxBottomRow2 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow2].GyGy + (float)pSrcGxy[srcIdxBottomRow2 + 1].GyGy + (float)pSrcGxy[srcIdxBottomRow2 + 2].GyGy + (float)pSrcGxy[srcIdxBottomRow2 + 3].GyGy +
        (float)pSrcGxy[srcIdxBottomRow3 -3].GyGy + (float)pSrcGxy[srcIdxBottomRow3 -2].GyGy + (float)pSrcGxy[srcIdxBottomRow3 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow3].GyGy + (float)pSrcGxy[srcIdxBottomRow3 + 1].GyGy + (float)pSrcGxy[srcIdxBottomRow3 + 2].GyGy + (float)pSrcGxy[srcIdxBottomRow3 + 3].GyGy;

    traceA = gx2 + gy2;
    detA = (gx2 * gy2) - (gxy2 * gxy2);
    Mc = detA - (sensitivity * traceA * traceA);
    Mc /= normalization_factor;
    if(Mc > strength_threshold) {
        pDstVc[dstIdx] = (float)Mc;
    }
    else {
        pDstVc[dstIdx] = (float)0;
    }
}
int HipExec_HarrisScore_HVC_HG3_7x7(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
    vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold,
    vx_float32 normalization_factor
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth),   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstVc , (dstVcStrideInBytes/sizeof(float)),
                    (float *)pSrcGxy_, (srcGxyStrideInBytes/sizeof(ago_harris_Gxy_t)),
                    sensitivity, strength_threshold,normalization_factor );

    return VX_SUCCESS;
}




















// VxCannyEdgeDetector OLD kernels

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 2) || (y <= 1) || (x >= dstWidth - 2) || (x <= 1)) {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxBottomRow1, srcIdxBottomRow2;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    short int sum1 = 0;
    sum1 = (
      gx[12] * (short int)*(pSrcImage + srcIdx) + gx[7] * (short int)*(pSrcImage + srcIdxTopRow1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow2) + gx[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + gx[22] * (short int)*(pSrcImage + srcIdxBottomRow2) +
      gx[11] * (short int)*(pSrcImage + srcIdx - 1) + gx[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gx[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) +
      gx[10] * (short int)*(pSrcImage + srcIdx - 2) + gx[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gx[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) +
      gx[13] * (short int)*(pSrcImage + srcIdx + 1) + gx[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gx[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gx[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) +
      gx[14] * (short int)*(pSrcImage + srcIdx + 2) + gx[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gx[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gx[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
    short int sum2 = 0;
    sum2 = (
      gy[12] * (short int)*(pSrcImage + srcIdx) + gy[7] * (short int)*(pSrcImage + srcIdxTopRow1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow2) + gy[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + gy[22] * (short int)*(pSrcImage + srcIdxBottomRow2) +
      gy[11] * (short int)*(pSrcImage + srcIdx - 1) + gy[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gy[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) +
      gy[10] * (short int)*(pSrcImage + srcIdx - 2) + gy[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gy[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) +
      gy[13] * (short int)*(pSrcImage + srcIdx + 1) + gy[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gy[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gy[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) +
      gy[14] * (short int)*(pSrcImage + srcIdx + 2) + gy[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gy[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gy[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
    sum2 = ~sum2 + 1;
    unsigned short int tmp = abs(sum1) + abs(sum2);
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = tmp;
}
int HipExec_CannySobel_U16_U8_5x5_L1NORM(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    short int gy[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 400);
    hipMalloc(&hipGy, 400);
    hipMemcpy(hipGx, gx, 400, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 400, hipMemcpyHostToDevice);

	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipFree(&hipGx);
    hipFree(&hipGy);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 2) || (y <= 1) || (x >= dstWidth - 2) || (x <= 1)) {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxBottomRow1, srcIdxBottomRow2;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    short int sum1 = 0;
    sum1 = (
      gx[12] * (short int)*(pSrcImage + srcIdx) + gx[7] * (short int)*(pSrcImage + srcIdxTopRow1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow2) + gx[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + gx[22] * (short int)*(pSrcImage + srcIdxBottomRow2) +
      gx[11] * (short int)*(pSrcImage + srcIdx - 1) + gx[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gx[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) +
      gx[10] * (short int)*(pSrcImage + srcIdx - 2) + gx[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gx[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) +
      gx[13] * (short int)*(pSrcImage + srcIdx + 1) + gx[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gx[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gx[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) +
      gx[14] * (short int)*(pSrcImage + srcIdx + 2) + gx[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gx[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gx[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
    short int sum2 = 0;
    sum2 = (
      gy[12] * (short int)*(pSrcImage + srcIdx) + gy[7] * (short int)*(pSrcImage + srcIdxTopRow1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow2) + gy[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + gy[22] * (short int)*(pSrcImage + srcIdxBottomRow2) +
      gy[11] * (short int)*(pSrcImage + srcIdx - 1) + gy[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gy[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) +
      gy[10] * (short int)*(pSrcImage + srcIdx - 2) + gy[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gy[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) +
      gy[13] * (short int)*(pSrcImage + srcIdx + 1) + gy[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gy[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gy[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) +
      gy[14] * (short int)*(pSrcImage + srcIdx + 2) + gy[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gy[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gy[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
    sum2 = ~sum2 + 1;
    int tmp = (vx_int16)sqrt((sum1*sum1) + (sum2*sum2));
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = (unsigned short int) tmp;
}
int HipExec_CannySobel_U16_U8_5x5_L2NORM(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    short int gy[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 400);
    hipMalloc(&hipGy, 400);
    hipMemcpy(hipGx, gx, 400, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 400, hipMemcpyHostToDevice);

	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L2NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipFree(&hipGx);
    hipFree(&hipGy);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 3) || (y <= 2) || (x >= dstWidth - 3) || (x <= 2)) {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxTopRow3, srcIdxBottomRow1, srcIdxBottomRow2, srcIdxBottomRow3;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxTopRow3 = srcIdx - (3 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3 * srcImageStrideInBytes);
    int sum1 = 0;
    sum1 = (
      gx[24] * (int)*(pSrcImage + srcIdx) + gx[17] * (int)*(pSrcImage + srcIdxTopRow1) + gx[10] * (int)*(pSrcImage + srcIdxTopRow2) + gx[3] * (int)*(pSrcImage + srcIdxTopRow3) + gx[31] * (int)*(pSrcImage + srcIdxBottomRow1) + gx[38] * (int)*(pSrcImage + srcIdxBottomRow2) + gx[45] * (int)*(pSrcImage + srcIdxBottomRow3) +
      gx[23] * (int)*(pSrcImage + srcIdx - 1) + gx[16] * (int)*(pSrcImage + srcIdxTopRow1 - 1) + gx[9] * (int)*(pSrcImage + srcIdxTopRow2 - 1) + gx[2] * (int)*(pSrcImage + srcIdxTopRow3 - 1) + gx[30] * (int)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[37] * (int)*(pSrcImage + srcIdxBottomRow2 - 1) + gx[44] * (int)*(pSrcImage + srcIdxBottomRow3 - 1) +
      gx[22] * (int)*(pSrcImage + srcIdx - 2) + gx[15] * (int)*(pSrcImage + srcIdxTopRow1 - 2) + gx[8] * (int)*(pSrcImage + srcIdxTopRow2 - 2) + gx[1] * (int)*(pSrcImage + srcIdxTopRow3 - 2) + gx[29] * (int)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[36] * (int)*(pSrcImage + srcIdxBottomRow2 - 2) + gx[43] * (int)*(pSrcImage + srcIdxBottomRow3 - 2) +
      gx[21] * (int)*(pSrcImage + srcIdx - 3) + gx[14] * (int)*(pSrcImage + srcIdxTopRow1 - 3) + gx[7] * (int)*(pSrcImage + srcIdxTopRow2 - 3) + gx[0] * (int)*(pSrcImage + srcIdxTopRow3 - 3) + gx[28] * (int)*(pSrcImage + srcIdxBottomRow1 - 3) + gx[35] * (int)*(pSrcImage + srcIdxBottomRow2 - 3) + gx[42] * (int)*(pSrcImage + srcIdxBottomRow3 - 3) +
      gx[25] * (int)*(pSrcImage + srcIdx + 1) + gx[18] * (int)*(pSrcImage + srcIdxTopRow1 + 1) + gx[11] * (int)*(pSrcImage + srcIdxTopRow2 + 1) + gx[4] * (int)*(pSrcImage + srcIdxTopRow3 + 1) + gx[32] * (int)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[39] * (int)*(pSrcImage + srcIdxBottomRow2 + 1) + gx[46] * (int)*(pSrcImage + srcIdxBottomRow3 + 1) +
      gx[26] * (int)*(pSrcImage + srcIdx + 2) + gx[19] * (int)*(pSrcImage + srcIdxTopRow1 + 2) + gx[12] * (int)*(pSrcImage + srcIdxTopRow2 + 2) + gx[5] * (int)*(pSrcImage + srcIdxTopRow3 + 2) + gx[33] * (int)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[40] * (int)*(pSrcImage + srcIdxBottomRow2 + 2) + gx[47] * (int)*(pSrcImage + srcIdxBottomRow3 + 2) +
      gx[27] * (int)*(pSrcImage + srcIdx + 3) + gx[20] * (int)*(pSrcImage + srcIdxTopRow1 + 3) + gx[13] * (int)*(pSrcImage + srcIdxTopRow2 + 3) + gx[6] * (int)*(pSrcImage + srcIdxTopRow3 + 3) + gx[34] * (int)*(pSrcImage + srcIdxBottomRow1 + 3) + gx[41] * (int)*(pSrcImage + srcIdxBottomRow2 + 3) + gx[48] * (int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
    int sum2 = 0;
    sum2 = (
      gy[24] * (int)*(pSrcImage + srcIdx) + gy[17] * (int)*(pSrcImage + srcIdxTopRow1) + gy[10] * (int)*(pSrcImage + srcIdxTopRow2) + gy[3] * (int)*(pSrcImage + srcIdxTopRow3) + gy[31] * (int)*(pSrcImage + srcIdxBottomRow1) + gy[38] * (int)*(pSrcImage + srcIdxBottomRow2) + gy[45] * (int)*(pSrcImage + srcIdxBottomRow3) +
      gy[23] * (int)*(pSrcImage + srcIdx - 1) + gy[16] * (int)*(pSrcImage + srcIdxTopRow1 - 1) + gy[9] * (int)*(pSrcImage + srcIdxTopRow2 - 1) + gy[2] * (int)*(pSrcImage + srcIdxTopRow3 - 1) + gy[30] * (int)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[37] * (int)*(pSrcImage + srcIdxBottomRow2 - 1) + gy[44] * (int)*(pSrcImage + srcIdxBottomRow3 - 1) +
      gy[22] * (int)*(pSrcImage + srcIdx - 2) + gy[15] * (int)*(pSrcImage + srcIdxTopRow1 - 2) + gy[8] * (int)*(pSrcImage + srcIdxTopRow2 - 2) + gy[1] * (int)*(pSrcImage + srcIdxTopRow3 - 2) + gy[29] * (int)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[36] * (int)*(pSrcImage + srcIdxBottomRow2 - 2) + gy[43] * (int)*(pSrcImage + srcIdxBottomRow3 - 2) +
      gy[21] * (int)*(pSrcImage + srcIdx - 3) + gy[14] * (int)*(pSrcImage + srcIdxTopRow1 - 3) + gy[7] * (int)*(pSrcImage + srcIdxTopRow2 - 3) + gy[0] * (int)*(pSrcImage + srcIdxTopRow3 - 3) + gy[28] * (int)*(pSrcImage + srcIdxBottomRow1 - 3) + gy[35] * (int)*(pSrcImage + srcIdxBottomRow2 - 3) + gy[42] * (int)*(pSrcImage + srcIdxBottomRow3 - 3) +
      gy[25] * (int)*(pSrcImage + srcIdx + 1) + gy[18] * (int)*(pSrcImage + srcIdxTopRow1 + 1) + gy[11] * (int)*(pSrcImage + srcIdxTopRow2 + 1) + gy[4] * (int)*(pSrcImage + srcIdxTopRow3 + 1) + gy[32] * (int)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[39] * (int)*(pSrcImage + srcIdxBottomRow2 + 1) + gy[46] * (int)*(pSrcImage + srcIdxBottomRow3 + 1) +
      gy[26] * (int)*(pSrcImage + srcIdx + 2) + gy[19] * (int)*(pSrcImage + srcIdxTopRow1 + 2) + gy[12] * (int)*(pSrcImage + srcIdxTopRow2 + 2) + gy[5] * (int)*(pSrcImage + srcIdxTopRow3 + 2) + gy[33] * (int)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[40] * (int)*(pSrcImage + srcIdxBottomRow2 + 2) + gy[47] * (int)*(pSrcImage + srcIdxBottomRow3 + 2) +
      gy[27] * (int)*(pSrcImage + srcIdx + 3) + gy[20] * (int)*(pSrcImage + srcIdxTopRow1 + 3) + gy[13] * (int)*(pSrcImage + srcIdxTopRow2 + 3) + gy[6] * (int)*(pSrcImage + srcIdxTopRow3 + 3) + gy[34] * (int)*(pSrcImage + srcIdxBottomRow1 + 3) + gy[41] * (int)*(pSrcImage + srcIdxBottomRow2 + 3) + gy[48] * (int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
    sum2 = ~sum2 + 1;
    int tmp = abs(sum1) + abs(sum2);
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = (unsigned short int)tmp;
}
int HipExec_CannySobel_U16_U8_7x7_L1NORM(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[49] = {-1,-4,-5,0,5,4,1,-6,-24,-30,0,30,24,6,-15,-60,-75,0,75,60,15,-20,-80,-100,0,100,80,20,-15,-60,-75,0,75,60,15,-6,-24,-30,0,30,24,6,-1,-4,-5,0,5,4,1};
    short int gy[49] = {-1,-6,-15,-20,-15,-6,-1,-4,-24,-60,-80,-60,-24,-4,-5,-30,-75,-100,-75,-30,-5,0,0,0,0,0,0,0,5,30,75,100,75,30,5,4,24,60,80,60,24,4,1,6,15,20,15,6,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 784);
    hipMalloc(&hipGy, 784);
    hipMemcpy(hipGx, gx, 784, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 784, hipMemcpyHostToDevice);

	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipFree(&hipGx);
    hipFree(&hipGy);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 3) || (y <= 2) || (x >= dstWidth - 3) || (x <= 2)) {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxTopRow3, srcIdxBottomRow1, srcIdxBottomRow2, srcIdxBottomRow3;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxTopRow3 = srcIdx - (3 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3 * srcImageStrideInBytes);
    short int sum1 = 0;
    sum1 = (
      gx[24] * (short int)*(pSrcImage + srcIdx) + gx[17] * (short int)*(pSrcImage + srcIdxTopRow1) + gx[10] * (short int)*(pSrcImage + srcIdxTopRow2) + gx[3] * (short int)*(pSrcImage + srcIdxTopRow3) + gx[31] * (short int)*(pSrcImage + srcIdxBottomRow1) + gx[38] * (short int)*(pSrcImage + srcIdxBottomRow2) + gx[45] * (short int)*(pSrcImage + srcIdxBottomRow3) +
      gx[23] * (short int)*(pSrcImage + srcIdx - 1) + gx[16] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gx[9] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow3 - 1) + gx[30] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gx[37] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) + gx[44] * (short int)*(pSrcImage + srcIdxBottomRow3 - 1) +
      gx[22] * (short int)*(pSrcImage + srcIdx - 2) + gx[15] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gx[8] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow3 - 2) + gx[29] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gx[36] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) + gx[43] * (short int)*(pSrcImage + srcIdxBottomRow3 - 2) +
      gx[21] * (short int)*(pSrcImage + srcIdx - 3) + gx[14] * (short int)*(pSrcImage + srcIdxTopRow1 - 3) + gx[7] * (short int)*(pSrcImage + srcIdxTopRow2 - 3) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow3 - 3) + gx[28] * (short int)*(pSrcImage + srcIdxBottomRow1 - 3) + gx[35] * (short int)*(pSrcImage + srcIdxBottomRow2 - 3) + gx[42] * (short int)*(pSrcImage + srcIdxBottomRow3 - 3) +
      gx[25] * (short int)*(pSrcImage + srcIdx + 1) + gx[18] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gx[11] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gx[4] * (short int)*(pSrcImage + srcIdxTopRow3 + 1) + gx[32] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gx[39] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) + gx[46] * (short int)*(pSrcImage + srcIdxBottomRow3 + 1) +
      gx[26] * (short int)*(pSrcImage + srcIdx + 2) + gx[19] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gx[12] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gx[5] * (short int)*(pSrcImage + srcIdxTopRow3 + 2) + gx[33] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gx[40] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2) + gx[47] * (short int)*(pSrcImage + srcIdxBottomRow3 + 2) +
      gx[27] * (short int)*(pSrcImage + srcIdx + 3) + gx[20] * (short int)*(pSrcImage + srcIdxTopRow1 + 3) + gx[13] * (short int)*(pSrcImage + srcIdxTopRow2 + 3) + gx[6] * (short int)*(pSrcImage + srcIdxTopRow3 + 3) + gx[34] * (short int)*(pSrcImage + srcIdxBottomRow1 + 3) + gx[41] * (short int)*(pSrcImage + srcIdxBottomRow2 + 3) + gx[48] * (short int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
    short int sum2 = 0;
    sum2 = (
      gy[24] * (short int)*(pSrcImage + srcIdx) + gy[17] * (short int)*(pSrcImage + srcIdxTopRow1) + gy[10] * (short int)*(pSrcImage + srcIdxTopRow2) + gy[3] * (short int)*(pSrcImage + srcIdxTopRow3) + gy[31] * (short int)*(pSrcImage + srcIdxBottomRow1) + gy[38] * (short int)*(pSrcImage + srcIdxBottomRow2) + gy[45] * (short int)*(pSrcImage + srcIdxBottomRow3) +
      gy[23] * (short int)*(pSrcImage + srcIdx - 1) + gy[16] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + gy[9] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow3 - 1) + gy[30] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + gy[37] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) + gy[44] * (short int)*(pSrcImage + srcIdxBottomRow3 - 1) +
      gy[22] * (short int)*(pSrcImage + srcIdx - 2) + gy[15] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + gy[8] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow3 - 2) + gy[29] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + gy[36] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) + gy[43] * (short int)*(pSrcImage + srcIdxBottomRow3 - 2) +
      gy[21] * (short int)*(pSrcImage + srcIdx - 3) + gy[14] * (short int)*(pSrcImage + srcIdxTopRow1 - 3) + gy[7] * (short int)*(pSrcImage + srcIdxTopRow2 - 3) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow3 - 3) + gy[28] * (short int)*(pSrcImage + srcIdxBottomRow1 - 3) + gy[35] * (short int)*(pSrcImage + srcIdxBottomRow2 - 3) + gy[42] * (short int)*(pSrcImage + srcIdxBottomRow3 - 3) +
      gy[25] * (short int)*(pSrcImage + srcIdx + 1) + gy[18] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + gy[11] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + gy[4] * (short int)*(pSrcImage + srcIdxTopRow3 + 1) + gy[32] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + gy[39] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) + gy[46] * (short int)*(pSrcImage + srcIdxBottomRow3 + 1) +
      gy[26] * (short int)*(pSrcImage + srcIdx + 2) + gy[19] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + gy[12] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + gy[5] * (short int)*(pSrcImage + srcIdxTopRow3 + 2) + gy[33] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + gy[40] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2) + gy[47] * (short int)*(pSrcImage + srcIdxBottomRow3 + 2) +
      gy[27] * (short int)*(pSrcImage + srcIdx + 3) + gy[20] * (short int)*(pSrcImage + srcIdxTopRow1 + 3) + gy[13] * (short int)*(pSrcImage + srcIdxTopRow2 + 3) + gy[6] * (short int)*(pSrcImage + srcIdxTopRow3 + 3) + gy[34] * (short int)*(pSrcImage + srcIdxBottomRow1 + 3) + gy[41] * (short int)*(pSrcImage + srcIdxBottomRow2 + 3) + gy[48] * (short int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
    sum2 = ~sum2 + 1;
    int tmp = (vx_int16)sqrt((sum1*sum1) + (sum2*sum2));
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = (unsigned short int) tmp;
}
int HipExec_CannySobel_U16_U8_7x7_L2NORM(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[49] = {-1,-4,-5,0,5,4,1,-6,-24,-30,0,30,24,6,-15,-60,-75,0,75,60,15,-20,-80,-100,0,100,80,20,-15,-60,-75,0,75,60,15,-6,-24,-30,0,30,24,6,-1,-4,-5,0,5,4,1};
    short int gy[49] = {-1,-6,-15,-20,-15,-6,-1,-4,-24,-60,-80,-60,-24,-4,-5,-30,-75,-100,-75,-30,-5,0,0,0,0,0,0,0,5,30,75,100,75,30,5,4,24,60,80,60,24,4,1,6,15,20,15,6,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 784);
    hipMalloc(&hipGy, 784);
    hipMemcpy(hipGx, gx, 784, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 784, hipMemcpyHostToDevice);

	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L2NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipFree(&hipGx);
    hipFree(&hipGy);
    return VX_SUCCESS;
}




















// VxCannyEdgeDetector OLD + UNUSED kernels

int HipExec_CannySuppThreshold_U8XY_U16_7x7(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
    return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
  return VX_ERROR_NOT_IMPLEMENTED;
}