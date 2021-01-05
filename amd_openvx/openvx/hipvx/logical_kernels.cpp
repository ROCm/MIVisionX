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

#define PIXELCHECKU1(value, pixel) (value ? 255 : pixel)

__device__ __forceinline__ int4 uchars_to_int4(uint src) {
    return make_int4((int)(src&0xFF), (int)((src&0xFF00)>>8), (int)((src&0xFF0000)>>16), (int)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint int4_to_uchars(int4 src) {
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ unsigned char extractMSB(int4 src1, int4 src2) {
    return ((((src1.x>>7)&1)<<7) | (((src1.y>>7)&1)<<6) | (((src1.z>>7)&1)<<5) | (((src1.w>>7)&1)<<4)
            | (((src2.x>>7)&1)<<3) | (((src2.y>>7)&1)<<2) | (((src2.z>>7)&1)<<1) | ((src2.w>>7)&1));
}

__device__ __forceinline__ int dataConvertU1ToU8_4bytes(uint nibble) {
    return (((nibble&1) * 0xFF) | ((((nibble>>1)&1) * 0xFF)<<8) |  ((((nibble>>2)&1) * 0xFF)<<16) | ((((nibble>>3)&1) * 0xFF)<<24));
}

// ----------------------------------------------------------------------------
// VxAnd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src2 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 dst = make_int4(src1.x&src2.x, src1.y&src2.y, src1.z&src2.z, src1.w&src2.w);
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_And_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U8U1( 
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    int src2 = (int)pSrcImage2[src2Idx];
    int4 dst1 = make_int4((src2 & 1) * src10.x, ((src2 >> 1)&1) * src10.y,
                        ((src2 >> 2)&1) * src10.z, ((src2 >> 3)&1) * src10.w);
    int4 dst2 = make_int4(((src2 >> 4) & 1) * src11.x, ((src2 >> 5)&1) * src11.y,
                        ((src2 >> 6)&1) * src11.z, ((src2 >> 7)&1) * src11.w);
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);

}
int HipExec_And_U8_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    // unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*8);
    // unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    // unsigned int src2Idx =  y*(srcImage2StrideInBytes) + (x*8);
    // for (int i = 0; i < 8; i++)
    //     pDstImage[dstIdx + i] = ((pSrcImage1[src1Idx] >> i) & 1) * pSrcImage2[src2Idx + i];

    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int src1 = (int)pSrcImage1[src1Idx];
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx + 1]);
    int4 dst1 = make_int4((src1 & 1) * src20.x, ((src1 >> 1)&1) * src20.y,
                        ((src1 >> 2)&1) * src20.z, ((src1 >> 3)&1) * src20.w);
    int4 dst2 = make_int4(((src1 >> 4) & 1) * src21.x, ((src1 >> 5)&1) * src21.y,
                        ((src1 >> 6)&1) * src21.z, ((src1 >> 7)&1) * src21.w);
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);
}
int HipExec_And_U8_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default"))) 
Hip_And_U8_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned char srcPixel = pSrcImage1[src1Idx] & pSrcImage2[src2Idx];
    int dst1 = dataConvertU1ToU8_4bytes(srcPixel & 0xF);
    int dst2 = dataConvertU1ToU8_4bytes((srcPixel>>4) & 0xF);
    pDstImage[dstIdx] = dst1;
    pDstImage[dstIdx+1] = dst2;
}
int HipExec_And_U8_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx+1]);
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    int4 dst1 = make_int4(src10.x&src20.x, src10.y&src20.y, src10.z&src20.z, src10.w&src20.w);
    int4 dst2 = make_int4(src11.x&src21.x, src11.y&src21.y, src11.z&src21.z, src11.w&src21.w);
    pDstImage[dstIdx] = PIXELCHECKU1(extractMSB(dst1, dst2), 0);
}
int HipExec_And_U1_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U8U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;

    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    unsigned char src2 = pSrcImage2[src2Idx]; 
    unsigned char srcByte = extractMSB(src10, src11);
    pDstImage[dstIdx] = PIXELCHECKU1(srcByte & src2, 0);
}
int HipExec_And_U1_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    unsigned char src1 = pSrcImage1[src1Idx]; 
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    unsigned char srcByte = extractMSB(src20, src21);
    pDstImage[dstIdx] = PIXELCHECKU1(src1 & srcByte, 0);
}
int HipExec_And_U1_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    pDstImage[dstIdx] = pSrcImage1[src1Idx] & pSrcImage2[src2Idx];
}
int HipExec_And_U1_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxOr kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src2 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 dst = make_int4(src1.x|src2.x, src1.y|src2.y, src1.z|src2.z, src1.w|src2.w);
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Or_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U8U1( 
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    int src2 = (int)pSrcImage2[src2Idx];
    // int4 dst1 = make_int4((src2 & 1) ? 255 : src10.x, ((src2 >> 1)&1) ? 255 : src10.y,
    //                     ((src2 >> 2)&1) ? 255 : src10.z, ((src2 >> 3)&1) ? 255 : src10.w);
    // int4 dst2 = make_int4(((src2 >> 4) & 1) ? 255 : src11.x, ((src2 >> 5)&1) ? 255 : src11.y,
    //                     ((src2 >> 6)&1) ? 255 : src11.z, ((src2 >> 7)&1) ? 255 : src11.w);
    int4 dst1 = make_int4(PIXELCHECKU1((src2 & 1), src10.x), PIXELCHECKU1(((src2 >> 1)&1), src10.y),
                         PIXELCHECKU1(((src2 >> 2)&1), src10.z), PIXELCHECKU1(((src2 >> 3)&1), src10.w));
    int4 dst2 = make_int4(PIXELCHECKU1(((src2 >> 4)&1), src11.x), PIXELCHECKU1(((src2 >> 5)&1), src11.y),
                         PIXELCHECKU1(((src2 >> 6)&1), src11.z), PIXELCHECKU1(((src2 >> 7)&1), src11.w));
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);

}
int HipExec_Or_U8_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int src1 = (int)pSrcImage1[src1Idx];
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx + 1]);
    int4 dst1 = make_int4(PIXELCHECKU1((src1 & 1), src20.x), PIXELCHECKU1(((src1 >> 1)&1), src20.y),
                         PIXELCHECKU1(((src1 >> 2)&1), src20.z), PIXELCHECKU1(((src1 >> 3)&1), src20.w));
    int4 dst2 = make_int4(PIXELCHECKU1(((src1 >> 4)&1), src21.x), PIXELCHECKU1(((src1 >> 5)&1), src21.y),
                         PIXELCHECKU1(((src1 >> 6)&1), src21.z), PIXELCHECKU1(((src1 >> 7)&1), src21.w));
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);
}
int HipExec_Or_U8_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default"))) 
Hip_Or_U8_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned char srcPixel = pSrcImage1[src1Idx] | pSrcImage2[src2Idx];
    int dst1 = dataConvertU1ToU8_4bytes(srcPixel & 0xF);
    int dst2 = dataConvertU1ToU8_4bytes((srcPixel>>4) & 0xF);
    pDstImage[dstIdx] = dst1;
    pDstImage[dstIdx+1] = dst2;
}
int HipExec_Or_U8_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx+1]);
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    int4 dst1 = make_int4(src10.x|src20.x, src10.y|src20.y, src10.z|src20.z, src10.w|src20.w);
    int4 dst2 = make_int4(src11.x|src21.x, src11.y|src21.y, src11.z|src21.z, src11.w|src21.w);
    pDstImage[dstIdx] = PIXELCHECKU1(extractMSB(dst1, dst2), 0);
}
int HipExec_Or_U1_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U8U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;

    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    unsigned char src2 = pSrcImage2[src2Idx]; 
    unsigned char srcByte = extractMSB(src10, src11);
    pDstImage[dstIdx] = PIXELCHECKU1(srcByte | src2, 0);
}
int HipExec_Or_U1_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    unsigned char src1 = pSrcImage1[src1Idx]; 
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    unsigned char srcByte = extractMSB(src20, src21);
    pDstImage[dstIdx] = PIXELCHECKU1(src1 | srcByte, 0);
}
int HipExec_Or_U1_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    pDstImage[dstIdx] = pSrcImage1[src1Idx] | pSrcImage2[src2Idx];
}
int HipExec_Or_U1_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxXor kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + x;
    int4 src1 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src2 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 dst = make_int4(src1.x^src2.x, src1.y^src2.y, src1.z^src2.z, src1.w^src2.w);
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Xor_U8_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U8U1( 
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    int src2 = (int)pSrcImage2[src2Idx];
    int4 dst1 = make_int4(PIXELCHECKU1((src2 & 1), 0)^ src10.x, PIXELCHECKU1(((src2 >> 1)&1), 0) ^ src10.y,
                        PIXELCHECKU1(((src2 >> 2)&1), 0) ^ src10.z, PIXELCHECKU1(((src2 >> 3)&1), 0) ^ src10.w);
    int4 dst2 = make_int4(PIXELCHECKU1(((src2 >> 4) & 1), 0) ^ src11.x, PIXELCHECKU1(((src2 >> 5)&1), 0) ^ src11.y,
                        PIXELCHECKU1(((src2 >> 6)&1), 0) ^ src11.z, PIXELCHECKU1(((src2 >> 7)&1), 0) ^ src11.w);
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);

}
int HipExec_Xor_U8_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int src1 = (int)pSrcImage1[src1Idx];
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx + 1]);
    int4 dst1 = make_int4(PIXELCHECKU1((src1 & 1), 0 )^ src20.x, PIXELCHECKU1(((src1 >> 1)&1), 0) ^ src20.y,
                        PIXELCHECKU1(((src1 >> 2)&1), 0) ^ src20.z, PIXELCHECKU1(((src1 >> 3)&1), 0) ^ src20.w);
    int4 dst2 = make_int4(PIXELCHECKU1(((src1 >> 4) & 1), 0) ^ src21.x, PIXELCHECKU1(((src1 >> 5)&1),0) ^ src21.y,
                        PIXELCHECKU1(((src1 >> 6)&1), 0) ^ src21.z, PIXELCHECKU1(((src1 >> 7)&1), 0) ^ src21.w);
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);
}
int HipExec_Xor_U8_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default"))) 
Hip_Xor_U8_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned char srcPixel = pSrcImage1[src1Idx] ^ pSrcImage2[src2Idx];
    int dst1 = dataConvertU1ToU8_4bytes(srcPixel & 0xF);
    int dst2 = dataConvertU1ToU8_4bytes((srcPixel>>4) & 0xF);
    pDstImage[dstIdx] = dst1;
    pDstImage[dstIdx+1] = dst2;
}
int HipExec_Xor_U8_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx+1]);
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    int4 dst1 = make_int4(src10.x^src20.x, src10.y^src20.y, src10.z^src20.z, src10.w^src20.w);
    int4 dst2 = make_int4(src11.x^src21.x, src11.y^src21.y, src11.z^src21.z, src11.w^src21.w);
    pDstImage[dstIdx] = PIXELCHECKU1(extractMSB(dst1, dst2), 0);
}
int HipExec_Xor_U1_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U8U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;

    int4 src10 = uchars_to_int4(pSrcImage1[src1Idx]);
    int4 src11 = uchars_to_int4(pSrcImage1[src1Idx + 1]);
    unsigned char src2 = pSrcImage2[src2Idx]; 
    unsigned char srcByte = extractMSB(src10, src11);
    pDstImage[dstIdx] = PIXELCHECKU1(srcByte ^ src2, 0);
}
int HipExec_Xor_U1_U8U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U8U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U1U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned int *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes>>2) + (x*2);
    unsigned char src1 = pSrcImage1[src1Idx]; 
    int4 src20 = uchars_to_int4(pSrcImage2[src2Idx]);
    int4 src21 = uchars_to_int4(pSrcImage2[src2Idx+1]);
    unsigned char srcByte = extractMSB(src20, src21);
    pDstImage[dstIdx] = PIXELCHECKU1(src1 ^ srcByte, 0);
}
int HipExec_Xor_U1_U1U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U1U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned int *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U1U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    pDstImage[dstIdx] = pSrcImage1[src1Idx] ^ pSrcImage2[src2Idx];
}
int HipExec_Xor_U1_U1U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U1U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes);
    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxNand kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxNor kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxXnor kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxNot kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int* pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int * pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + x;
    int4 src = uchars_to_int4(pSrcImage[srcIdx]);
    int4 dst = make_int4(~src.x, ~src.y, ~src.z, ~src.w);
    pDstImage[dstIdx] = int4_to_uchars(dst);
}
int HipExec_Not_U8_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 * pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 * pHipSrcImage, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U8_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage, srcImage1StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int * pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char * pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + (x*2);
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + x;
    unsigned char src = ~ pSrcImage[srcIdx];
    int4 dst1 = make_int4(PIXELCHECKU1((src & 1), 0 ), PIXELCHECKU1(((src >> 1)&1), 0),
                        PIXELCHECKU1(((src >> 2)&1), 0), PIXELCHECKU1(((src >> 3)&1), 0));
    int4 dst2 = make_int4(PIXELCHECKU1(((src >> 4)&1), 0), PIXELCHECKU1(((src >> 5)&1), 0),
                        PIXELCHECKU1(((src >> 6)&1), 0), PIXELCHECKU1(((src >> 7)&1), 0));
    pDstImage[dstIdx] = int4_to_uchars(dst1);
    pDstImage[dstIdx+1] = int4_to_uchars(dst2);
}
int HipExec_Not_U8_U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 * pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 * pHipSrcImage, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U8_U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage, srcImage1StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U1_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char * pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int * pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes>>2) + (x*2);
    int4 src10 = uchars_to_int4(pSrcImage[srcIdx]);
    int4 src11 = uchars_to_int4(pSrcImage[srcIdx + 1]);
    unsigned char src = PIXELCHECKU1(extractMSB(src10, src11), 0);
    pDstImage[dstIdx] = ~src;
}
int HipExec_Not_U1_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 * pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 * pHipSrcImage, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U1_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned int *)pHipSrcImage, srcImage1StrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U1_U1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char * pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char * pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*8 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + x;
    pDstImage[dstIdx] = ~pSrcImage[srcIdx];
}
int HipExec_Not_U1_U1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 * pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 * pHipSrcImage, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+7)>>3,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U1_U1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, (const unsigned char *)pHipSrcImage, srcImage1StrideInBytes);
    return VX_SUCCESS;
}