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

// __device__ __forceinline__ float4 ucharTofloat4(unsigned int src)
// {
//     return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
// }

// __device__ __forceinline__ uint float4ToUint(float4 src)
// {
//   return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
// }

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
#define HIPVXMAX3(a,b,c)  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define HIPVXMIN3(a,b,c)  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))

__device__ int FastAtan2_Canny(short int Gx, short int Gy)
{
	unsigned int ret;
	unsigned short int ax, ay;
	ax = std::abs(Gx), ay = std::abs(Gy);	// todo:: check if math.h function is faster
	float d1 = (float)ax*0.4142135623730950488016887242097f;
	float d2 = (float)ax*2.4142135623730950488016887242097f;
	ret = (Gx*Gy) < 0 ? 3 : 1;
	if (ay <= d1)
		ret = 0;
	if (ay >= d2)
		ret = 2;
	return ret;
}

// ----------------------------------------------------------------------------
// VxFastCorners kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxHarrisCorners kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxCannyEdgeDetector kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 1) || (y <= 0))
    {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    short int sum1 = 0;
    sum1 += (gx[4] * (short int)*(pSrcImage + srcIdx) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow) + gx[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum1 += (gx[3] * (short int)*(pSrcImage + srcIdx - 1) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gx[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum1 += (gx[5] * (short int)*(pSrcImage + srcIdx + 1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gx[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    short int sum2 = 0;
    sum2 += (gy[4] * (short int)*(pSrcImage + srcIdx) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow) + gy[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum2 += (gy[3] * (short int)*(pSrcImage + srcIdx - 1) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gy[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum2 += (gy[5] * (short int)*(pSrcImage + srcIdx + 1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gy[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    sum2 = ~sum2 + 1;
    unsigned short int tmp = abs(sum1) + abs(sum2);
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = tmp;
}
int HipExec_CannySobel_U16_U8_3x3_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    // printf("\nPrinting source before first Canny:");
    // unsigned char *pHostSrcImage = (unsigned char *) calloc(dstWidth * dstHeight, sizeof(unsigned char));
    // hipMemcpy(pHostSrcImage, pHipSrcImage, dstWidth * dstHeight * sizeof(unsigned char), hipMemcpyDeviceToHost);
    
    // for (int i = 0; i < dstHeight; i++)
    // {
    //   printf("\n");
    //   for (int j = 0; j < dstWidth; j++)
    //   {
    //     printf("%d\t", pHostSrcImage[i * srcImageStrideInBytes + j]);
    //   }
    // }
    // free(pHostSrcImage);

    short int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    short int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 144);
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    // printf("\nPrinting after first Canny:");
    // vx_uint32 dstride = dstImageStrideInBytes>>1;
    // int *pHostDstImage = (int *) calloc(dstWidth * dstHeight, sizeof(int));
    // short int *pHostDstImageShort;
    // pHostDstImageShort = (short int *)pHostDstImage;
    // hipMemcpy(pHostDstImageShort, pHipDstImage, dstWidth * dstHeight * sizeof(int), hipMemcpyDeviceToHost);
    
    // for (int i = 0; i < dstHeight; i++)
    // {
    //   printf("\n");
    //   for (int j = 0; j < dstWidth; j++)
    //   {
    //     printf("%d\t", pHostDstImageShort[i * dstride + j]);
    //   }
    // }
    // free(pHostDstImage);

    printf("\nHipExec_CannySobel_U16_U8_3x3_L1NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 1) || (y <= 0))
    {
      pDstImage[dstIdx] = (unsigned short int)0;
      return;
    }
    int srcIdxTopRow, srcIdxBottomRow;
    srcIdxTopRow = srcIdx - srcImageStrideInBytes;
    srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
    short int sum1 = 0;
    sum1 += (gx[4] * (short int)*(pSrcImage + srcIdx) + gx[1] * (short int)*(pSrcImage + srcIdxTopRow) + gx[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum1 += (gx[3] * (short int)*(pSrcImage + srcIdx - 1) + gx[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gx[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum1 += (gx[5] * (short int)*(pSrcImage + srcIdx + 1) + gx[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gx[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    short int sum2 = 0;
    sum2 += (gy[4] * (short int)*(pSrcImage + srcIdx) + gy[1] * (short int)*(pSrcImage + srcIdxTopRow) + gy[7] * (short int)*(pSrcImage + srcIdxBottomRow));
    if (x != 0)
      sum2 += (gy[3] * (short int)*(pSrcImage + srcIdx - 1) + gy[0] * (short int)*(pSrcImage + srcIdxTopRow - 1) + gy[6] * (short int)*(pSrcImage + srcIdxBottomRow - 1));
    if (x != (dstWidth - 1))
      sum2 += (gy[5] * (short int)*(pSrcImage + srcIdx + 1) + gy[2] * (short int)*(pSrcImage + srcIdxTopRow + 1) + gy[8] * (short int)*(pSrcImage + srcIdxBottomRow + 1));
    sum2 = ~sum2 + 1;
    unsigned short int tmp = (vx_int16)sqrt((sum1*sum1) + (sum2*sum2));
    tmp <<= 2;
    tmp |= (FastAtan2_Canny(sum1, sum2) & 3);
    pDstImage[dstIdx] = tmp;
}
int HipExec_CannySobel_U16_U8_3x3_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    short int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 144);
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L2NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_CannySobel_U16_U8_3x3_L2NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 2) || (y <= 1) || (x >= dstWidth - 2) || (x <= 1))
    {
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
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    short int gy[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 400);
    hipMalloc(&hipGy, 400);
    hipMemcpy(hipGx, gx, 400, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 400, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_CannySobel_U16_U8_5x5_L1NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 2) || (y <= 1) || (x >= dstWidth - 2) || (x <= 1))
    {
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
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    short int gy[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 400);
    hipMalloc(&hipGy, 400);
    hipMemcpy(hipGx, gx, 400, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 400, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L2NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_CannySobel_U16_U8_5x5_L2NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L1NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 3) || (y <= 2) || (x >= dstWidth - 3) || (x <= 2))
    {
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
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[49] = {-1,-4,-5,0,5,4,1,-6,-24,-30,0,30,24,6,-15,-60,-75,0,75,60,15,-20,-80,-100,0,100,80,20,-15,-60,-75,0,75,60,15,-6,-24,-30,0,30,24,6,-1,-4,-5,0,5,4,1};
    short int gy[49] = {-1,-6,-15,-20,-15,-6,-1,-4,-24,-60,-80,-60,-24,-4,-5,-30,-75,-100,-75,-30,-5,0,0,0,0,0,0,0,5,30,75,100,75,30,5,4,24,60,80,60,24,4,1,6,15,20,15,6,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 784);
    hipMalloc(&hipGy, 784);
    hipMemcpy(hipGx, gx, 784, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 784, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_CannySobel_U16_U8_7x7_L1NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L2NORM(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const short int *gx, const short int *gy
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes>>1) + x;
    int srcIdx =  y*(srcImageStrideInBytes) + x;
    if ((y >= dstHeight - 3) || (y <= 2) || (x >= dstWidth - 3) || (x <= 2))
    {
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
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[49] = {-1,-4,-5,0,5,4,1,-6,-24,-30,0,30,24,6,-15,-60,-75,0,75,60,15,-20,-80,-100,0,100,80,20,-15,-60,-75,0,75,60,15,-6,-24,-30,0,30,24,6,-1,-4,-5,0,5,4,1};
    short int gy[49] = {-1,-6,-15,-20,-15,-6,-1,-4,-24,-60,-80,-60,-24,-4,-5,-30,-75,-100,-75,-30,-5,0,0,0,0,0,0,0,5,30,75,100,75,30,5,4,24,60,80,60,24,4,1,6,15,20,15,6,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 784);
    hipMalloc(&hipGy, 784);
    hipMemcpy(hipGx, gx, 784, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 784, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L2NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    hipFree(&hipGx);
    hipFree(&hipGy);

    printf("\nHipExec_CannySobel_U16_U8_7x7_L2NORM: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack, 
    const unsigned short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const unsigned short int hyst_lower, const unsigned short int hyst_upper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1))
    {
      pDstImage[dstIdx] = (unsigned char)0;
      return;
    }
    int srcIdx =  y*(srcImageStrideInBytes>>1) + x;
    int xyStackStride = dstWidth * 2;
    int xyStackIdx = (y * xyStackStride) + (x * 2);
    static const int n_offset[4][2][2] = {
      {{-1, 0}, {1, 0}}, 
      {{1, -1}, {-1, 1}}, 
      {{0, -1}, {0, 1}}, 
      {{-1, -1}, {1, 1}}
    };
    const unsigned short int *pLocSrc = pSrcImage + srcIdx;
    unsigned short int mag = (pLocSrc[0] >> 2);
    unsigned short int ang = pLocSrc[0] & 3;
    int offset0 = n_offset[ang][0][1] * (srcImageStrideInBytes>>1) + n_offset[ang][0][0];
		int offset1 = n_offset[ang][1][1] * (srcImageStrideInBytes>>1) + n_offset[ang][1][0];
    unsigned short int edge = ((mag >(pLocSrc[offset0] >> 2)) && (mag >(pLocSrc[offset1] >> 2))) ? mag : 0;
    if (edge > hyst_upper)
    {
      pDstImage[dstIdx] = (unsigned char)255;
      xyStack[xyStackIdx] = x;
      xyStack[xyStackIdx + 1] = y;
		}
		else if (edge <= hyst_lower)
    {
			pDstImage[dstIdx] = (unsigned char)0;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
		}
		else
    {
      pDstImage[dstIdx] = (unsigned char)127;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_3x3(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    (unsigned short int *) xyStack, 
                    (const unsigned short int *)pHipSrcImage, srcImageStrideInBytes,
                    (const unsigned short int)hyst_lower, (const unsigned short int)hyst_upper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    *pxyStackTop = (vx_uint32)(dstWidth * dstHeight - 1);

    // printf("\nPrinting after second Canny:");
    // vx_uint32 dstride = dstImageStrideInBytes;
    // unsigned char *pHostDstImage = (unsigned char *) calloc(dstWidth * dstHeight, sizeof(unsigned char));
    // hipMemcpy(pHostDstImage, pHipDstImage, dstWidth * dstHeight * sizeof(unsigned char), hipMemcpyDeviceToHost);
    // for (int i = 0; i < dstHeight; i++)
    // {
    //   printf("\n");
    //   for (int j = 0; j < dstWidth; j++)
    //   {
    //     printf("%d\t", pHostDstImage[i * dstride + j]);
    //   }
    // }
    // free(pHostDstImage);

    // printf("\nPrinting stack after second Canny:");
    // unsigned short int *xyStackHost = (unsigned short int *) calloc(dstWidth * dstHeight * 2, sizeof(unsigned short int));
    // hipMemcpy(xyStackHost, xyStack, dstWidth * dstHeight * 2 * sizeof(unsigned short int), hipMemcpyDeviceToHost);
    // for (int i = 0; i < dstHeight; i++)
    // {
    //   printf("\n");
    //   for (int j = 0; j < dstWidth; j++)
    //   {
    //     printf("%d,%d\t", xyStackHost[i*dstWidth*2 + j*2], xyStackHost[i*dstWidth*2 + j*2 + 1]);
    //   }
    // }
    // free(xyStackHost);

    printf("\nHipExec_CannySuppThreshold_U8XY_U16_3x3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_7x7(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack, 
    const unsigned short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const unsigned short int hyst_lower, const unsigned short int hyst_upper
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1))
    {
      pDstImage[dstIdx] = (unsigned char)0;
      return;
    }
    int srcIdx =  y*(srcImageStrideInBytes>>1) + x;
    int xyStackStride = dstWidth * 2;
    int xyStackIdx = (y * xyStackStride) + (x * 2);
    static const int n_offset[4][2][2] = {
      {{-1, 0}, {1, 0}}, 
      {{1, -1}, {-1, 1}}, 
      {{0, -1}, {0, 1}}, 
      {{-1, -1}, {1, 1}}
    };
    const unsigned short int *pLocSrc = pSrcImage + srcIdx;
    unsigned short int mag = (pLocSrc[0] >> 2);
    unsigned short int ang = pLocSrc[0] & 3;
    int offset0 = n_offset[ang][0][1] * (srcImageStrideInBytes>>1) + n_offset[ang][0][0];
		int offset1 = n_offset[ang][1][1] * (srcImageStrideInBytes>>1) + n_offset[ang][1][0];
    unsigned short int edge = ((mag >(pLocSrc[offset0] >> 2)) && (mag >(pLocSrc[offset1] >> 2))) ? mag : 0;
    if (edge > hyst_upper)
    {
      pDstImage[dstIdx] = (unsigned char)255;
      xyStack[xyStackIdx] = x;
      xyStack[xyStackIdx + 1] = y;
		}
		else if (edge <= hyst_lower)
    {
			pDstImage[dstIdx] = (unsigned char)0;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
		}
		else
    {
      pDstImage[dstIdx] = (unsigned char)127;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_7x7(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    (unsigned short int *) xyStack, 
                    (const unsigned short int *)pHipSrcImage, srcImageStrideInBytes,
                    (const unsigned short int)hyst_lower, (const unsigned short int)hyst_upper);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    *pxyStackTop = (vx_uint32)(dstWidth * dstHeight - 1);

    printf("\nHipExec_CannySuppThreshold_U8XY_U16_7x7: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop, 
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, 
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    )
{
  return VX_ERROR_NOT_IMPLEMENTED;
}

__global__ void __attribute__((visibility("default")))
Hip_CannyEdgeTrace_U8_U8XY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1))
      return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int xyStackStride = dstWidth * 2;
    int xyStackIdx = (y * xyStackStride) + (x * 2);
    if ((xyStack[xyStackIdx] == 0) && (xyStack[xyStackIdx + 1] == 0))
      return;
    else
    {
      unsigned short int xLoc = xyStack[xyStackIdx];
      unsigned short int yLoc = xyStack[xyStackIdx + 1];
      static const ago_coord2d_short_t dir_offsets[8] = {
        {(vx_int16)(-1), (vx_int16)(-1)}, 
        {(vx_int16)0, (vx_int16)(-1)}, 
        {(vx_int16)1, (vx_int16)(-1)}, 
        {(vx_int16)(-1), (vx_int16)0}, 
        {(vx_int16)1, (vx_int16)0}, 
        {(vx_int16)(-1), (vx_int16)1}, 
        {(vx_int16)0, (vx_int16)1}, 
        {(vx_int16)1, (vx_int16)1}
      };
      for (int i = 0; i < 8; i++)
      {
        const ago_coord2d_short_t offs = dir_offsets[i];
        unsigned short int x1 = x + offs.x;
			  unsigned short int y1 = y + offs.y;
        int dstIdxNeighbor =  y1*(dstImageStrideInBytes) + x1;
        if (pDstImage[dstIdxNeighbor] == 127)
        {
          pDstImage[dstIdxNeighbor] = (unsigned char)255;
          int xyStackIdxNeighbor = (y1 * xyStackStride) + (x1 * 2);
          xyStack[xyStackIdxNeighbor] = x1;
          xyStack[xyStackIdxNeighbor + 1] = y1;
        }
      }
    }
    if (pDstImage[dstIdx] == 127)
      pDstImage[dstIdx] = (unsigned char)0;
}
int HipExec_CannyEdgeTrace_U8_U8XY(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 xyStackTop
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_CannyEdgeTrace_U8_U8XY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes, 
                    (unsigned short int *) xyStack);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_CannyEdgeTrace_U8_U8XY: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxNonMaxSupp kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxOpticalFlow kernels for hip backend
// ----------------------------------------------------------------------------

