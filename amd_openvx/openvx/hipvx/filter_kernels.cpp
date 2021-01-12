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
#include <stdlib.h>

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
#define HIPVXMAX3(a,b,c)  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define HIPVXMIN3(a,b,c)  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))

// ----------------------------------------------------------------------------
// VxBox kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Box_U8_U8_3x3(
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
    sum += (*(pSrcImageCurrent) + *(pSrcImageTop) + *(pSrcImageBottom));
    if (x != 0)
      sum += (*(pSrcImageCurrent - 1) + *(pSrcImageTop - 1) + *(pSrcImageBottom - 1));
    if (x != (dstWidth - 1))
      sum += (*(pSrcImageCurrent + 1) + *(pSrcImageTop + 1) + *(pSrcImageBottom + 1));
    pDstImage[dstIdx] = (unsigned char)(sum / 9);
}
int HipExec_Box_U8_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight - 2;

    hipLaunchKernelGGL(Hip_Box_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - 2,
                    (unsigned char *)pHipDstImage + dstImageStrideInBytes , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + srcImageStrideInBytes, srcImageStrideInBytes);
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
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    float sum = 0;
    sum += (conv[4] * (float)*(pSrcImage + srcIdx) + conv[1] * (float)*(pSrcImage + srcIdxTopRow) + conv[7] * (float)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum += (conv[3] * (float)*(pSrcImage + srcIdx - 1) + conv[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + conv[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += (conv[5] * (float)*(pSrcImage + srcIdx + 1) + conv[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + conv[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));
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
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxBottomRow1, srcIdxBottomRow2;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    short int sum = 0;
    sum = (
      conv[12] * (short int)*(pSrcImage + srcIdx) + conv[7] * (short int)*(pSrcImage + srcIdxTopRow1) + conv[2] * (short int)*(pSrcImage + srcIdxTopRow2) + conv[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + conv[22] * (short int)*(pSrcImage + srcIdxBottomRow2) + 
      conv[11] * (short int)*(pSrcImage + srcIdx - 1) + conv[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + conv[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + conv[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + conv[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) + 
      conv[10] * (short int)*(pSrcImage + srcIdx - 2) + conv[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + conv[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + conv[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + conv[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) + 
      conv[13] * (short int)*(pSrcImage + srcIdx + 1) + conv[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + conv[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + conv[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + conv[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) + 
      conv[14] * (short int)*(pSrcImage + srcIdx + 2) + conv[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + conv[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + conv[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + conv[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
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
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxTopRow3, srcIdxBottomRow1, srcIdxBottomRow2, srcIdxBottomRow3;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxTopRow3 = srcIdx - (3 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3 * srcImageStrideInBytes);
    short int sum = 0;
    sum = (
      conv[24] * (int)*(pSrcImage + srcIdx) + conv[17] * (int)*(pSrcImage + srcIdxTopRow1) + conv[10] * (int)*(pSrcImage + srcIdxTopRow2) + conv[3] * (int)*(pSrcImage + srcIdxTopRow3) + conv[31] * (int)*(pSrcImage + srcIdxBottomRow1) + conv[38] * (int)*(pSrcImage + srcIdxBottomRow2) + conv[45] * (int)*(pSrcImage + srcIdxBottomRow3) + 
      conv[23] * (int)*(pSrcImage + srcIdx - 1) + conv[16] * (int)*(pSrcImage + srcIdxTopRow1 - 1) + conv[9] * (int)*(pSrcImage + srcIdxTopRow2 - 1) + conv[2] * (int)*(pSrcImage + srcIdxTopRow3 - 1) + conv[30] * (int)*(pSrcImage + srcIdxBottomRow1 - 1) + conv[37] * (int)*(pSrcImage + srcIdxBottomRow2 - 1) + conv[44] * (int)*(pSrcImage + srcIdxBottomRow3 - 1) + 
      conv[22] * (int)*(pSrcImage + srcIdx - 2) + conv[15] * (int)*(pSrcImage + srcIdxTopRow1 - 2) + conv[8] * (int)*(pSrcImage + srcIdxTopRow2 - 2) + conv[1] * (int)*(pSrcImage + srcIdxTopRow3 - 2) + conv[29] * (int)*(pSrcImage + srcIdxBottomRow1 - 2) + conv[36] * (int)*(pSrcImage + srcIdxBottomRow2 - 2) + conv[43] * (int)*(pSrcImage + srcIdxBottomRow3 - 2) + 
      conv[21] * (int)*(pSrcImage + srcIdx - 3) + conv[14] * (int)*(pSrcImage + srcIdxTopRow1 - 3) + conv[7] * (int)*(pSrcImage + srcIdxTopRow2 - 3) + conv[0] * (int)*(pSrcImage + srcIdxTopRow3 - 3) + conv[28] * (int)*(pSrcImage + srcIdxBottomRow1 - 3) + conv[35] * (int)*(pSrcImage + srcIdxBottomRow2 - 3) + conv[42] * (int)*(pSrcImage + srcIdxBottomRow3 - 3) + 
      conv[25] * (int)*(pSrcImage + srcIdx + 1) + conv[18] * (int)*(pSrcImage + srcIdxTopRow1 + 1) + conv[11] * (int)*(pSrcImage + srcIdxTopRow2 + 1) + conv[4] * (int)*(pSrcImage + srcIdxTopRow3 + 1) + conv[32] * (int)*(pSrcImage + srcIdxBottomRow1 + 1) + conv[39] * (int)*(pSrcImage + srcIdxBottomRow2 + 1) + conv[46] * (int)*(pSrcImage + srcIdxBottomRow3 + 1) + 
      conv[26] * (int)*(pSrcImage + srcIdx + 2) + conv[19] * (int)*(pSrcImage + srcIdxTopRow1 + 2) + conv[12] * (int)*(pSrcImage + srcIdxTopRow2 + 2) + conv[5] * (int)*(pSrcImage + srcIdxTopRow3 + 2) + conv[33] * (int)*(pSrcImage + srcIdxBottomRow1 + 2) + conv[40] * (int)*(pSrcImage + srcIdxBottomRow2 + 2) + conv[47] * (int)*(pSrcImage + srcIdxBottomRow3 + 2) + 
      conv[27] * (int)*(pSrcImage + srcIdx + 3) + conv[20] * (int)*(pSrcImage + srcIdxTopRow1 + 3) + conv[13] * (int)*(pSrcImage + srcIdxTopRow2 + 3) + conv[6] * (int)*(pSrcImage + srcIdxTopRow3 + 3) + conv[34] * (int)*(pSrcImage + srcIdxBottomRow1 + 3) + conv[41] * (int)*(pSrcImage + srcIdxBottomRow2 + 3) + conv[48] * (int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
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

    float *hipConv;
    hipMalloc(&hipConv, 16 * convolutionWidth * convolutionHeight);
    hipMemcpy(hipConv, conv, 16 * convolutionWidth * convolutionHeight, hipMemcpyHostToDevice);
    
	  if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else if ((convolutionWidth == 5) && (convolutionHeight == 5)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else if ((convolutionWidth == 7) && (convolutionHeight == 7)) {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else {
      hipLaunchKernelGGL(Hip_Convolve_U8_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (unsigned char *)pHipDstImage + (bound * dstImageStrideInBytes) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv, convolutionWidth, convolutionHeight);
    }
    hipFree(&hipConv);
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
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    float sum = 0;
    sum += (conv[4] * (float)*(pSrcImage + srcIdx) + conv[1] * (float)*(pSrcImage + srcIdxTopRow) + conv[7] * (float)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum += (conv[3] * (float)*(pSrcImage + srcIdx - 1) + conv[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + conv[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum += (conv[5] * (float)*(pSrcImage + srcIdx + 1) + conv[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + conv[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));
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
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxBottomRow1, srcIdxBottomRow2;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    short int sum = 0;
    sum = (
      conv[12] * (short int)*(pSrcImage + srcIdx) + conv[7] * (short int)*(pSrcImage + srcIdxTopRow1) + conv[2] * (short int)*(pSrcImage + srcIdxTopRow2) + conv[17] * (short int)*(pSrcImage + srcIdxBottomRow1) + conv[22] * (short int)*(pSrcImage + srcIdxBottomRow2) + 
      conv[11] * (short int)*(pSrcImage + srcIdx - 1) + conv[6] * (short int)*(pSrcImage + srcIdxTopRow1 - 1) + conv[1] * (short int)*(pSrcImage + srcIdxTopRow2 - 1) + conv[16] * (short int)*(pSrcImage + srcIdxBottomRow1 - 1) + conv[21] * (short int)*(pSrcImage + srcIdxBottomRow2 - 1) + 
      conv[10] * (short int)*(pSrcImage + srcIdx - 2) + conv[5] * (short int)*(pSrcImage + srcIdxTopRow1 - 2) + conv[0] * (short int)*(pSrcImage + srcIdxTopRow2 - 2) + conv[15] * (short int)*(pSrcImage + srcIdxBottomRow1 - 2) + conv[20] * (short int)*(pSrcImage + srcIdxBottomRow2 - 2) + 
      conv[13] * (short int)*(pSrcImage + srcIdx + 1) + conv[8] * (short int)*(pSrcImage + srcIdxTopRow1 + 1) + conv[3] * (short int)*(pSrcImage + srcIdxTopRow2 + 1) + conv[18] * (short int)*(pSrcImage + srcIdxBottomRow1 + 1) + conv[23] * (short int)*(pSrcImage + srcIdxBottomRow2 + 1) + 
      conv[14] * (short int)*(pSrcImage + srcIdx + 2) + conv[9] * (short int)*(pSrcImage + srcIdxTopRow1 + 2) + conv[4] * (short int)*(pSrcImage + srcIdxTopRow2 + 2) + conv[19] * (short int)*(pSrcImage + srcIdxBottomRow1 + 2) + conv[24] * (short int)*(pSrcImage + srcIdxBottomRow2 + 2)
    );
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
    int srcIdxTopRow1, srcIdxTopRow2, srcIdxTopRow3, srcIdxBottomRow1, srcIdxBottomRow2, srcIdxBottomRow3;
    srcIdxTopRow1 = srcIdx - srcImageStrideInBytes;
    srcIdxTopRow2 = srcIdx - (2 * srcImageStrideInBytes);
    srcIdxTopRow3 = srcIdx - (3 * srcImageStrideInBytes);
    srcIdxBottomRow1 = srcIdx + srcImageStrideInBytes;
    srcIdxBottomRow2 = srcIdx + (2 * srcImageStrideInBytes);
    srcIdxBottomRow3 = srcIdx + (3 * srcImageStrideInBytes);
    short int sum = 0;
    sum = (
      conv[24] * (int)*(pSrcImage + srcIdx) + conv[17] * (int)*(pSrcImage + srcIdxTopRow1) + conv[10] * (int)*(pSrcImage + srcIdxTopRow2) + conv[3] * (int)*(pSrcImage + srcIdxTopRow3) + conv[31] * (int)*(pSrcImage + srcIdxBottomRow1) + conv[38] * (int)*(pSrcImage + srcIdxBottomRow2) + conv[45] * (int)*(pSrcImage + srcIdxBottomRow3) + 
      conv[23] * (int)*(pSrcImage + srcIdx - 1) + conv[16] * (int)*(pSrcImage + srcIdxTopRow1 - 1) + conv[9] * (int)*(pSrcImage + srcIdxTopRow2 - 1) + conv[2] * (int)*(pSrcImage + srcIdxTopRow3 - 1) + conv[30] * (int)*(pSrcImage + srcIdxBottomRow1 - 1) + conv[37] * (int)*(pSrcImage + srcIdxBottomRow2 - 1) + conv[44] * (int)*(pSrcImage + srcIdxBottomRow3 - 1) + 
      conv[22] * (int)*(pSrcImage + srcIdx - 2) + conv[15] * (int)*(pSrcImage + srcIdxTopRow1 - 2) + conv[8] * (int)*(pSrcImage + srcIdxTopRow2 - 2) + conv[1] * (int)*(pSrcImage + srcIdxTopRow3 - 2) + conv[29] * (int)*(pSrcImage + srcIdxBottomRow1 - 2) + conv[36] * (int)*(pSrcImage + srcIdxBottomRow2 - 2) + conv[43] * (int)*(pSrcImage + srcIdxBottomRow3 - 2) + 
      conv[21] * (int)*(pSrcImage + srcIdx - 3) + conv[14] * (int)*(pSrcImage + srcIdxTopRow1 - 3) + conv[7] * (int)*(pSrcImage + srcIdxTopRow2 - 3) + conv[0] * (int)*(pSrcImage + srcIdxTopRow3 - 3) + conv[28] * (int)*(pSrcImage + srcIdxBottomRow1 - 3) + conv[35] * (int)*(pSrcImage + srcIdxBottomRow2 - 3) + conv[42] * (int)*(pSrcImage + srcIdxBottomRow3 - 3) + 
      conv[25] * (int)*(pSrcImage + srcIdx + 1) + conv[18] * (int)*(pSrcImage + srcIdxTopRow1 + 1) + conv[11] * (int)*(pSrcImage + srcIdxTopRow2 + 1) + conv[4] * (int)*(pSrcImage + srcIdxTopRow3 + 1) + conv[32] * (int)*(pSrcImage + srcIdxBottomRow1 + 1) + conv[39] * (int)*(pSrcImage + srcIdxBottomRow2 + 1) + conv[46] * (int)*(pSrcImage + srcIdxBottomRow3 + 1) + 
      conv[26] * (int)*(pSrcImage + srcIdx + 2) + conv[19] * (int)*(pSrcImage + srcIdxTopRow1 + 2) + conv[12] * (int)*(pSrcImage + srcIdxTopRow2 + 2) + conv[5] * (int)*(pSrcImage + srcIdxTopRow3 + 2) + conv[33] * (int)*(pSrcImage + srcIdxBottomRow1 + 2) + conv[40] * (int)*(pSrcImage + srcIdxBottomRow2 + 2) + conv[47] * (int)*(pSrcImage + srcIdxBottomRow3 + 2) + 
      conv[27] * (int)*(pSrcImage + srcIdx + 3) + conv[20] * (int)*(pSrcImage + srcIdxTopRow1 + 3) + conv[13] * (int)*(pSrcImage + srcIdxTopRow2 + 3) + conv[6] * (int)*(pSrcImage + srcIdxTopRow3 + 3) + conv[34] * (int)*(pSrcImage + srcIdxBottomRow1 + 3) + conv[41] * (int)*(pSrcImage + srcIdxBottomRow2 + 3) + conv[48] * (int)*(pSrcImage + srcIdxBottomRow3 + 3)
    );
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

    float *hipConv;
    hipMalloc(&hipConv, 16 * convolutionWidth * convolutionHeight);
    hipMemcpy(hipConv, conv, 16 * convolutionWidth * convolutionHeight, hipMemcpyHostToDevice);
    
    if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else if ((convolutionWidth == 5) && (convolutionHeight == 5)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_5x5,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else if ((convolutionWidth == 7) && (convolutionHeight == 7)) {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv);
    }
    else {
      hipLaunchKernelGGL(Hip_Convolve_S16_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight - (2 * bound),
                    (short int *)pHipDstImage + (bound * dstImageStrideInBytes>>1) , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage + (bound * srcImageStrideInBytes), srcImageStrideInBytes,
                    (const short int *)hipConv, convolutionWidth, convolutionHeight);
    }
    hipFree(&hipConv);
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