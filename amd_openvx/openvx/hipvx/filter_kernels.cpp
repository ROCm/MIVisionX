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

int HipExec_Box_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Box_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (uchar *)pHipDstImage, dstImageStrideInBytes,
                    (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxDilate kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Dilate_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    unsigned char valCol0 = 0, valCol1 = 0, valCol2 = 0;
    valCol1 = HIPVXMAX3(*(pSrcImageCurrent), *(pSrcImageTop), *(pSrcImageBottom));
    if (x != 0)
      valCol0 = HIPVXMAX3(*(pSrcImageCurrent - 1), *(pSrcImageTop - 1), *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      valCol2 = HIPVXMAX3(*(pSrcImageCurrent + 1), *(pSrcImageTop + 1), *(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (unsigned char)HIPVXMAX3(valCol0, valCol1, valCol2);
}
int HipExec_Dilate_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipLaunchKernelGGL(Hip_Dilate_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxErode kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Erode_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    unsigned char valCol0 = 0, valCol1 = 0, valCol2 = 0;
    valCol1 = HIPVXMIN3(*(pSrcImageCurrent), *(pSrcImageTop), *(pSrcImageBottom));
    if (x != 0)
      valCol0 = HIPVXMIN3(*(pSrcImageCurrent - 1), *(pSrcImageTop - 1), *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      valCol2 = HIPVXMIN3(*(pSrcImageCurrent + 1), *(pSrcImageTop + 1), *(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (unsigned char)HIPVXMIN3(valCol0, valCol1, valCol2);
}
int HipExec_Erode_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipLaunchKernelGGL(Hip_Erode_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMedian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Median_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int col = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int row = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    __shared__ unsigned char sharedmem[(16+2)]  [(16+2)];
    bool is_x_left = (hipThreadIdx_x == 0), is_x_right = (hipThreadIdx_x == 16-1);
    bool is_y_top = (hipThreadIdx_y == 0), is_y_bottom = (hipThreadIdx_y == 16-1);
    if(is_x_left)
      sharedmem[hipThreadIdx_x][hipThreadIdx_y+1] = 0;
    else if(is_x_right)
      sharedmem[hipThreadIdx_x + 2][hipThreadIdx_y+1]=0;
    if (is_y_top){
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y] = 0;
      if(is_x_left)
        sharedmem[hipThreadIdx_x][hipThreadIdx_y] = 0;
      else if(is_x_right)
        sharedmem[hipThreadIdx_x+2][hipThreadIdx_y] = 0;
    }
    else if (is_y_bottom){
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y+2] = 0;
      if(is_x_right)
        sharedmem[hipThreadIdx_x+2][hipThreadIdx_y+2] = 0;
      else if(is_x_left)
        sharedmem[hipThreadIdx_x][hipThreadIdx_y+2] = 0;
	  }
    sharedmem[hipThreadIdx_x+1][hipThreadIdx_y+1] = pSrcImage[row*srcImageStrideInBytes+col];
    if(is_x_left && (col>0))
      sharedmem[hipThreadIdx_x][hipThreadIdx_y+1] = pSrcImage[row*srcImageStrideInBytes+(col-1)];
    else if(is_x_right && (col<dstWidth-1))
      sharedmem[hipThreadIdx_x + 2][hipThreadIdx_y+1]= pSrcImage[row*srcImageStrideInBytes+(col+1)];
    if (is_y_top && (row>0)){
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y] = pSrcImage[(row-1)*srcImageStrideInBytes+col];
      if(is_x_left)
        sharedmem[hipThreadIdx_x][hipThreadIdx_y] = pSrcImage[(row-1)*srcImageStrideInBytes+(col-1)];
      else if(is_x_right )
        sharedmem[hipThreadIdx_x+2][hipThreadIdx_y] = pSrcImage[(row-1)*srcImageStrideInBytes+(col+1)];
    }
    else if (is_y_bottom && (row<dstHeight-1)){
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y+2] = pSrcImage[(row+1)*srcImageStrideInBytes + col];
      if(is_x_right)
        sharedmem[hipThreadIdx_x+2][hipThreadIdx_y+2] = pSrcImage[(row+1)*srcImageStrideInBytes+(col+1)];
      else if(is_x_left)
        sharedmem[hipThreadIdx_x][hipThreadIdx_y+2] = pSrcImage[(row+1)*srcImageStrideInBytes+(col-1)];
    }
    __syncthreads();
    unsigned char filterVector[9] = {
      sharedmem[hipThreadIdx_x][hipThreadIdx_y], 
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y], 
      sharedmem[hipThreadIdx_x+2][hipThreadIdx_y],
      sharedmem[hipThreadIdx_x][hipThreadIdx_y+1], 
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y+1], 
      sharedmem[hipThreadIdx_x+2][hipThreadIdx_y+1],
      sharedmem[hipThreadIdx_x][hipThreadIdx_y+2], 
      sharedmem[hipThreadIdx_x+1][hipThreadIdx_y+2], 
      sharedmem[hipThreadIdx_x+2][hipThreadIdx_y+2]};

		for (int i = 0; i < 9; i++) {
      for (int j = i + 1; j < 9; j++) {
        if (filterVector[i] > filterVector[j]) { 
          char tmp = filterVector[i];
          filterVector[i] = filterVector[j];
          filterVector[j] = tmp;
        }
      }
    }
	  pDstImage[row*dstImageStrideInBytes+col] = filterVector[4];
}
int HipExec_Median_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Median_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxGaussian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Gaussian_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    int sum = 0;
    sum += (4 * *(pSrcImageCurrent) + 2 * *(pSrcImageTop) + 2 * *(pSrcImageBottom));
    if (x != 0)
      sum += (2 * *(pSrcImageCurrent - 1) + *(pSrcImageTop - 1) + *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum += (2 * *(pSrcImageCurrent + 1) + *(pSrcImageTop + 1) + *(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum/16);
}
int HipExec_Gaussian_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

	  hipLaunchKernelGGL(Hip_Gaussian_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxConvolve kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*dstImageStrideInBytes + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow, *pSrcImageBottomRow;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageBottomRow = pSrcImageCurrentRow + srcImageStrideInBytes;
    short int sum = 0;
    sum += (conv[4] * *(pSrcImageCurrentRow) + conv[1] * *(pSrcImageTopRow) + conv[7] * *(pSrcImageBottomRow));
    if (x != 0)
      sum += (conv[3] * *(pSrcImageCurrentRow - 1) + conv[0] * *(pSrcImageTopRow - 1) + conv[6] * *(pSrcImageBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += (conv[5] * *(pSrcImageCurrentRow + 1) + conv[2] * *(pSrcImageTopRow + 1) + conv[8] * *(pSrcImageBottomRow + 1));
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_5x5(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*dstImageStrideInBytes + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow1, *pSrcImageTopRow2, *pSrcImageBottomRow1, *pSrcImageBottomRow2;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow1 = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageTopRow2 = pSrcImageTopRow1 - srcImageStrideInBytes;
    pSrcImageBottomRow1 = pSrcImageCurrentRow + srcImageStrideInBytes;
    pSrcImageBottomRow2 = pSrcImageBottomRow1 + srcImageStrideInBytes;
    short int sum = 0;
    sum += (conv[12] * (short int)*(pSrcImageCurrentRow) + conv[7] * (short int)*(pSrcImageTopRow1) + conv[2] * (short int)*(pSrcImageTopRow2) + conv[17] * (short int)*(pSrcImageBottomRow1) + conv[22] * (short int)*(pSrcImageBottomRow2));
    if (x > 0)
      sum += (conv[11] * (short int)*(pSrcImageCurrentRow - 1) + conv[6] * (short int)*(pSrcImageTopRow1 - 1) + conv[1] * (short int)*(pSrcImageTopRow2 - 1) + conv[16] * (short int)*(pSrcImageBottomRow1 - 1) + conv[21] * (short int)*(pSrcImageBottomRow2 - 1));
    if (x > 1)
      sum += (conv[10] * (short int)*(pSrcImageCurrentRow - 2) + conv[5] * (short int)*(pSrcImageTopRow1 - 2) + conv[0] * (short int)*(pSrcImageTopRow2 - 2) + conv[15] * (short int)*(pSrcImageBottomRow1 - 2) + conv[20] * (short int)*(pSrcImageBottomRow2 - 2));
    if (x < dstWidth - 1)
      sum += (conv[13] * (short int)*(pSrcImageCurrentRow + 1) + conv[8] * (short int)*(pSrcImageTopRow1 + 1) + conv[3] * (short int)*(pSrcImageTopRow2 + 1) + conv[18] * (short int)*(pSrcImageBottomRow1 + 1) + conv[23] * (short int)*(pSrcImageBottomRow2 + 1));
    if (x < dstWidth - 2)
      sum += (conv[14] * (short int)*(pSrcImageCurrentRow + 2) + conv[9] * (short int)*(pSrcImageTopRow1 + 2) + conv[4] * (short int)*(pSrcImageTopRow2 + 2) + conv[19] * (short int)*(pSrcImageBottomRow1 + 2) + conv[24] * (short int)*(pSrcImageBottomRow2 + 2));
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_7x7(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*dstImageStrideInBytes + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow1, *pSrcImageTopRow2, *pSrcImageTopRow3, *pSrcImageBottomRow1, *pSrcImageBottomRow2, *pSrcImageBottomRow3;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow1 = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageTopRow2 = pSrcImageTopRow1 - srcImageStrideInBytes;
    pSrcImageTopRow3 = pSrcImageTopRow2 - srcImageStrideInBytes;
    pSrcImageBottomRow1 = pSrcImageCurrentRow + srcImageStrideInBytes;
    pSrcImageBottomRow2 = pSrcImageBottomRow1 + srcImageStrideInBytes;
    pSrcImageBottomRow3 = pSrcImageBottomRow2 + srcImageStrideInBytes;
    int sum = 0;
    sum += (conv[24] * (int)*(pSrcImageCurrentRow) + conv[17] * (int)*(pSrcImageTopRow1) + conv[10] * (int)*(pSrcImageTopRow2) + conv[3] * (int)*(pSrcImageTopRow3) + conv[31] * (int)*(pSrcImageBottomRow1) + conv[38] * (int)*(pSrcImageBottomRow2) + conv[45] * (int)*(pSrcImageBottomRow3));
    if (x > 0)
      sum += (conv[23] * (int)*(pSrcImageCurrentRow - 1) + conv[16] * (int)*(pSrcImageTopRow1 - 1) + conv[9] * (int)*(pSrcImageTopRow2 - 1) + conv[2] * (int)*(pSrcImageTopRow3 - 1) + conv[30] * (int)*(pSrcImageBottomRow1 - 1) + conv[37] * (int)*(pSrcImageBottomRow2 - 1) + conv[44] * (int)*(pSrcImageBottomRow3 - 1));
    if (x > 1)
      sum += (conv[22] * (int)*(pSrcImageCurrentRow - 2) + conv[15] * (int)*(pSrcImageTopRow1 - 2) + conv[8] * (int)*(pSrcImageTopRow2 - 2) + conv[1] * (int)*(pSrcImageTopRow3 - 2) + conv[29] * (int)*(pSrcImageBottomRow1 - 2) + conv[36] * (int)*(pSrcImageBottomRow2 - 2) + conv[43] * (int)*(pSrcImageBottomRow3 - 2));
    if (x > 2)
      sum += (conv[21] * (int)*(pSrcImageCurrentRow - 3) + conv[14] * (int)*(pSrcImageTopRow1 - 3) + conv[7] * (int)*(pSrcImageTopRow2 - 3) + conv[0] * (int)*(pSrcImageTopRow3 - 3) + conv[28] * (int)*(pSrcImageBottomRow1 - 3) + conv[35] * (int)*(pSrcImageBottomRow2 - 3) + conv[42] * (int)*(pSrcImageBottomRow3 - 3));
    if (x < dstWidth - 1)
      sum += (conv[25] * (int)*(pSrcImageCurrentRow + 1) + conv[18] * (int)*(pSrcImageTopRow1 + 1) + conv[11] * (int)*(pSrcImageTopRow2 + 1) + conv[4] * (int)*(pSrcImageTopRow3 + 1) + conv[32] * (int)*(pSrcImageBottomRow1 + 1) + conv[39] * (int)*(pSrcImageBottomRow2 + 1) + conv[46] * (int)*(pSrcImageBottomRow3 + 1));
    if (x < dstWidth - 2)
      sum += (conv[26] * (int)*(pSrcImageCurrentRow + 2) + conv[19] * (int)*(pSrcImageTopRow1 + 2) + conv[12] * (int)*(pSrcImageTopRow2 + 2) + conv[5] * (int)*(pSrcImageTopRow3 + 2) + conv[33] * (int)*(pSrcImageBottomRow1 + 2) + conv[40] * (int)*(pSrcImageBottomRow2 + 2) + conv[47] * (int)*(pSrcImageBottomRow3 + 2));
    if (x < dstWidth - 3)
      sum += (conv[27] * (int)*(pSrcImageCurrentRow + 3) + conv[20] * (int)*(pSrcImageTopRow1 + 3) + conv[13] * (int)*(pSrcImageTopRow2 + 3) + conv[6] * (int)*(pSrcImageTopRow3 + 3) + conv[34] * (int)*(pSrcImageBottomRow1 + 3) + conv[41] * (int)*(pSrcImageBottomRow2 + 3) + conv[48] * (int)*(pSrcImageBottomRow3 + 3));
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv,
    const unsigned int convolutionWidth, const unsigned int convolutionHeight
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int bound = convolutionHeight / 2;
    int dstIdx =  y*dstImageStrideInBytes + x;
    int srcIdx =  (y - bound)*(srcImageStrideInBytes) + (x - bound);
    int indices[25];
    short int sum = 0;
    for (int i = 0; i < convolutionHeight; i++) {
      indices[i] = srcIdx + (i * srcImageStrideInBytes);
    }
    for (int i = 0; i < convolutionWidth; i++) {
      if (x <= bound) {
        if ((i >= bound - x) && (i < convolutionHeight)) {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else if (x >= dstWidth - bound) {
        if ((i >= 0) && (i < dstWidth - (x - bound))) {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else {
        for (int j = 0; j < convolutionHeight; j++)
          sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
      }
    }
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(sum);
}
int HipExec_Convolve_U8_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    ) {
    int bound = convolutionHeight / 2;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - (2 * bound);
	  if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else if ((convolutionWidth == 5) && (convolutionHeight == 5)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else if ((convolutionWidth == 7) && (convolutionHeight == 7)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv, 
                    convolutionWidth, convolutionHeight);
    }
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow, *pSrcImageBottomRow;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageBottomRow = pSrcImageCurrentRow + srcImageStrideInBytes;
    short int sum = 0;
    sum += (conv[4] * *(pSrcImageCurrentRow) + conv[1] * *(pSrcImageTopRow) + conv[7] * *(pSrcImageBottomRow));
    if (x != 0)
      sum += (conv[3] * *(pSrcImageCurrentRow - 1) + conv[0] * *(pSrcImageTopRow - 1) + conv[6] * *(pSrcImageBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += (conv[5] * *(pSrcImageCurrentRow + 1) + conv[2] * *(pSrcImageTopRow + 1) + conv[8] * *(pSrcImageBottomRow + 1));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_5x5(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow1, *pSrcImageTopRow2, *pSrcImageBottomRow1, *pSrcImageBottomRow2;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow1 = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageTopRow2 = pSrcImageTopRow1 - srcImageStrideInBytes;
    pSrcImageBottomRow1 = pSrcImageCurrentRow + srcImageStrideInBytes;
    pSrcImageBottomRow2 = pSrcImageBottomRow1 + srcImageStrideInBytes;
    short int sum = 0;
    sum += (conv[12] * (short int)*(pSrcImageCurrentRow) + conv[7] * (short int)*(pSrcImageTopRow1) + conv[2] * (short int)*(pSrcImageTopRow2) + conv[17] * (short int)*(pSrcImageBottomRow1) + conv[22] * (short int)*(pSrcImageBottomRow2));
    if (x > 0)
      sum += (conv[11] * (short int)*(pSrcImageCurrentRow - 1) + conv[6] * (short int)*(pSrcImageTopRow1 - 1) + conv[1] * (short int)*(pSrcImageTopRow2 - 1) + conv[16] * (short int)*(pSrcImageBottomRow1 - 1) + conv[21] * (short int)*(pSrcImageBottomRow2 - 1));
    if (x > 1)
      sum += (conv[10] * (short int)*(pSrcImageCurrentRow - 2) + conv[5] * (short int)*(pSrcImageTopRow1 - 2) + conv[0] * (short int)*(pSrcImageTopRow2 - 2) + conv[15] * (short int)*(pSrcImageBottomRow1 - 2) + conv[20] * (short int)*(pSrcImageBottomRow2 - 2));
    if (x < dstWidth - 1)
      sum += (conv[13] * (short int)*(pSrcImageCurrentRow + 1) + conv[8] * (short int)*(pSrcImageTopRow1 + 1) + conv[3] * (short int)*(pSrcImageTopRow2 + 1) + conv[18] * (short int)*(pSrcImageBottomRow1 + 1) + conv[23] * (short int)*(pSrcImageBottomRow2 + 1));
    if (x < dstWidth - 2)
      sum += (conv[14] * (short int)*(pSrcImageCurrentRow + 2) + conv[9] * (short int)*(pSrcImageTopRow1 + 2) + conv[4] * (short int)*(pSrcImageTopRow2 + 2) + conv[19] * (short int)*(pSrcImageBottomRow1 + 2) + conv[24] * (short int)*(pSrcImageBottomRow2 + 2));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_7x7(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*srcImageStrideInBytes + x;
    const unsigned char *pSrcImageCurrentRow, *pSrcImageTopRow1, *pSrcImageTopRow2, *pSrcImageTopRow3, *pSrcImageBottomRow1, *pSrcImageBottomRow2, *pSrcImageBottomRow3;
    pSrcImageCurrentRow = pSrcImage + srcIdx;
    pSrcImageTopRow1 = pSrcImageCurrentRow - srcImageStrideInBytes;
    pSrcImageTopRow2 = pSrcImageTopRow1 - srcImageStrideInBytes;
    pSrcImageTopRow3 = pSrcImageTopRow2 - srcImageStrideInBytes;
    pSrcImageBottomRow1 = pSrcImageCurrentRow + srcImageStrideInBytes;
    pSrcImageBottomRow2 = pSrcImageBottomRow1 + srcImageStrideInBytes;
    pSrcImageBottomRow3 = pSrcImageBottomRow2 + srcImageStrideInBytes;
    int sum = 0;
    sum += (conv[24] * (int)*(pSrcImageCurrentRow) + conv[17] * (int)*(pSrcImageTopRow1) + conv[10] * (int)*(pSrcImageTopRow2) + conv[3] * (int)*(pSrcImageTopRow3) + conv[31] * (int)*(pSrcImageBottomRow1) + conv[38] * (int)*(pSrcImageBottomRow2) + conv[45] * (int)*(pSrcImageBottomRow3));
    if (x > 0)
      sum += (conv[23] * (int)*(pSrcImageCurrentRow - 1) + conv[16] * (int)*(pSrcImageTopRow1 - 1) + conv[9] * (int)*(pSrcImageTopRow2 - 1) + conv[2] * (int)*(pSrcImageTopRow3 - 1) + conv[30] * (int)*(pSrcImageBottomRow1 - 1) + conv[37] * (int)*(pSrcImageBottomRow2 - 1) + conv[44] * (int)*(pSrcImageBottomRow3 - 1));
    if (x > 1)
      sum += (conv[22] * (int)*(pSrcImageCurrentRow - 2) + conv[15] * (int)*(pSrcImageTopRow1 - 2) + conv[8] * (int)*(pSrcImageTopRow2 - 2) + conv[1] * (int)*(pSrcImageTopRow3 - 2) + conv[29] * (int)*(pSrcImageBottomRow1 - 2) + conv[36] * (int)*(pSrcImageBottomRow2 - 2) + conv[43] * (int)*(pSrcImageBottomRow3 - 2));
    if (x > 2)
      sum += (conv[21] * (int)*(pSrcImageCurrentRow - 3) + conv[14] * (int)*(pSrcImageTopRow1 - 3) + conv[7] * (int)*(pSrcImageTopRow2 - 3) + conv[0] * (int)*(pSrcImageTopRow3 - 3) + conv[28] * (int)*(pSrcImageBottomRow1 - 3) + conv[35] * (int)*(pSrcImageBottomRow2 - 3) + conv[42] * (int)*(pSrcImageBottomRow3 - 3));
    if (x < dstWidth - 1)
      sum += (conv[25] * (int)*(pSrcImageCurrentRow + 1) + conv[18] * (int)*(pSrcImageTopRow1 + 1) + conv[11] * (int)*(pSrcImageTopRow2 + 1) + conv[4] * (int)*(pSrcImageTopRow3 + 1) + conv[32] * (int)*(pSrcImageBottomRow1 + 1) + conv[39] * (int)*(pSrcImageBottomRow2 + 1) + conv[46] * (int)*(pSrcImageBottomRow3 + 1));
    if (x < dstWidth - 2)
      sum += (conv[26] * (int)*(pSrcImageCurrentRow + 2) + conv[19] * (int)*(pSrcImageTopRow1 + 2) + conv[12] * (int)*(pSrcImageTopRow2 + 2) + conv[5] * (int)*(pSrcImageTopRow3 + 2) + conv[33] * (int)*(pSrcImageBottomRow1 + 2) + conv[40] * (int)*(pSrcImageBottomRow2 + 2) + conv[47] * (int)*(pSrcImageBottomRow3 + 2));
    if (x < dstWidth - 3)
      sum += (conv[27] * (int)*(pSrcImageCurrentRow + 3) + conv[20] * (int)*(pSrcImageTopRow1 + 3) + conv[13] * (int)*(pSrcImageTopRow2 + 3) + conv[6] * (int)*(pSrcImageTopRow3 + 3) + conv[34] * (int)*(pSrcImageBottomRow1 + 3) + conv[41] * (int)*(pSrcImageBottomRow2 + 3) + conv[48] * (int)*(pSrcImageBottomRow3 + 3));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum);
}
__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *conv,
    const unsigned int convolutionWidth, const unsigned int convolutionHeight
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int bound = convolutionHeight / 2;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  (y - bound)*(srcImageStrideInBytes) + (x - bound);
    if (y >= dstHeight - (2 * bound)) {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    int indices[25];
    short int sum = 0;
    for (int i = 0; i < convolutionHeight; i++) {
      indices[i] = srcIdx + (i * srcImageStrideInBytes);
    }
    for (int i = 0; i < convolutionWidth; i++) {
      if (x <= bound) {
        if ((i >= bound - x) && (i < convolutionHeight)) {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else if (x >= dstWidth - bound) {
        if ((i >= 0) && (i < dstWidth - (x - bound))) {
          for (int j = 0; j < convolutionHeight; j++)
            sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
        }
      }
      else {
        for (int j = 0; j < convolutionHeight; j++)
          sum += (conv[(j * convolutionWidth) + i] * *(pSrcImage + indices[j] + i));
      }
    }
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum);
}
int HipExec_Convolve_S16_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
    ) {

    int bound = convolutionHeight / 2;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - (2 * bound);
    if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else if ((convolutionWidth == 5) && (convolutionHeight == 5)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else if ((convolutionWidth == 7) && (convolutionHeight == 7)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv);
    }
    else {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)conv, convolutionWidth, convolutionHeight);
    }
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxSobel kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16S16_U8_3x3_GXY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage1, unsigned int dstImage1StrideInBytes,
    short int *pDstImage2, unsigned int dstImage2StrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dst1Idx =  y*(dstImage1StrideInBytes>>1) + x;
    int dst2Idx =  y*(dstImage2StrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2) {
      pDstImage1[dst1Idx] = 0;
      pDstImage2[dst2Idx] = 0;
      return;
    }
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    short int sum1 = 0;
    if (x != 0)
      sum1 += (-2 * *(pSrcImageCurrent - 1) - *(pSrcImageTop - 1) - *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum1 += (2 * *(pSrcImageCurrent + 1) + *(pSrcImageTop + 1) + *(pSrcImageBottom + 1));
    short int sum2 = 0;
    sum2 += (- 2 * *(pSrcImageTop) + 2 * *(pSrcImageBottom));
    if (x != 0)
      sum2 += (- *(pSrcImageTop - 1) + *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum2 += (- *(pSrcImageTop + 1) + *(pSrcImageBottom + 1));
    pDstImage1[dst1Idx] = PIXELSATURATES16(sum2);
    pDstImage2[dst2Idx] = PIXELSATURATES16(sum1);
}
int HipExec_Sobel_S16S16_U8_3x3_GXY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
    vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

	  hipLaunchKernelGGL(Hip_Sobel_S16S16_U8_3x3_GXY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage1 + (dstImage1StrideInBytes>>1) , dstImage1StrideInBytes,
                    (short int *)pHipDstImage2 + (dstImage2StrideInBytes>>1) , dstImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2) {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    short int sum1 = 0;
    if (x != 0)
      sum1 += (-2 * (short int)*(pSrcImageCurrent - 1) - (short int)*(pSrcImageTop - 1) - (short int)*(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum1 += (2 * (short int)*(pSrcImageCurrent + 1) + (short int)*(pSrcImageTop + 1) + (short int)*(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum1);
}
int HipExec_Sobel_S16_U8_3x3_GX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

	  hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GX,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage + (dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if (y >= dstHeight - 2) {
      pDstImage[dstIdx] = (short int)0;
      return;
    }
    const unsigned char *pSrcImageTop, *pSrcImageCurrent, *pSrcImageBottom;
    pSrcImageCurrent = pSrcImage + srcIdx;
    pSrcImageTop = pSrcImageCurrent - srcImageStrideInBytes;
    pSrcImageBottom = pSrcImageCurrent + srcImageStrideInBytes;
    short int sum2 = 0;
    sum2 += (- 2 * (short int)*(pSrcImageTop) + 2 * (short int)*(pSrcImageBottom));
    if (x != 0)
      sum2 += (- (short int)*(pSrcImageTop - 1) + (short int)*(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum2 += (- (short int)*(pSrcImageTop + 1) + (short int)*(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (short int)PIXELSATURATES16(sum2);
}
int HipExec_Sobel_S16_U8_3x3_GY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage + (dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
    return VX_SUCCESS;
}