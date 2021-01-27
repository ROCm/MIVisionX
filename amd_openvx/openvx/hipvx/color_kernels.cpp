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

#define PIXELSATURATEU8(pixel)      (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELROUNDU8(value)        ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))
#define FLOAT_MAX(f1, f2)   (f1 >= f2 ? f1 : f2)
#define FLOAT_MIN(f1, f2)   (f1 <= f2 ? f1 : f2)

#define RGB2Y(R,G,B) ((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f))
#define RGB2U(R,G,B) ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f)
#define RGB2V(R,G,B) ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f)

#define YUV2R(Y,U,V) (Y + ((V - 128.0f) * 1.5748f))
#define YUV2G(Y,U,V) (Y - ((U - 128.0f) * 0.1873f) - ((V - 128.0f) * 0.4681f))
#define YUV2B(Y,U,V) (Y + ((U - 128.0f) * 1.8556f))

__device__ __forceinline__ float4 uchars_to_float4 (uint src) {
    return make_float4((float)(src & 0xFF), (float)((src & 0xFF00) >> 8), (float)((src & 0xFF0000) >> 16), (float)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ uint float4_to_uchars (float4 src) {
    return ((uint)src.x & 0xFF) | (((uint)src.y & 0xFF) << 8) | (((uint)src.z & 0xFF) << 16) | (((uint)src.w & 0xFF) << 24);
}

__device__ __forceinline__ float2 uchars_to_float2 (uint src) {
    return make_float2((float)(src & 0xFF), (float)((src & 0xFF00) >> 8));
}

__device__ __forceinline__ uint float2_to_uchars (float2 src) {
    return (((uint)src.x & 0xFF) | (((uint)src.y & 0xFF) << 8) );
}

__device__ __forceinline__ uint float4_to_uchars_u32 (float4 src) {
    return ((uint)src.x & 0xFF) | (((uint)src.y & 0xFF) << 8) | (((uint)src.z & 0xFF) << 16) | (((uint)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint4 uchars_to_uint4 (unsigned int src) {
    return make_uint4((unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8), (unsigned int)((src & 0xFF0000) >> 16), (unsigned int)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ unsigned int uint4_to_uchars (uint4 src) {
    return ((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8) | (((unsigned char)src.z & 0xFF) << 16) | (((unsigned char)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint2 uchars_to_uint2 (unsigned int src) {
    return make_uint2((unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8));
}

__device__ __forceinline__ unsigned int uint2_to_uchars (uint2 src) {
    return (((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8));
}

__device__ __forceinline__ uchar4 uchars_to_uchar4(unsigned int src) {
    return make_uchar4((unsigned char)(src & 0xFF), (unsigned char)((src & 0xFF00) >> 8), (unsigned char)((src & 0xFF0000) >> 16), (unsigned char)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ unsigned int uchar4_to_uchars(uchar4 src) {
    return ((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8) | (((unsigned char)src.z & 0xFF) << 16) | (((unsigned char)src.w & 0xFF) << 24);
}
// ----------------------------------------------------------------------------
// VxLut kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *lut
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight))
        return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes >> 2) + x;
    uchar4 src = uchars_to_uchar4(pSrcImage1[src1Idx]);
    pDstImage[dstIdx] = uchar4_to_uchars(make_uchar4(lut[src.x], lut[src.y], lut[src.z], lut[src.w]));
}
int HipExec_Lut_U8_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_uint8 *lut
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Lut_U8_U8,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes, 
                       (const unsigned char *)lut);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorDepth kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Wrap(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift) {
    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * dstImageStrideInBytes + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes >> 1) + x;

    int4 src = *((int4 *)(&pSrcImage[srcIdx]));

    uint2 dst;
    int sr = shift;
    sr += 16;
    dst.x  = ((src.x   << 16) >> sr) & 0xff;
    dst.x |= ((src.x          >> sr) & 0xff) <<  8;
    dst.x |= (((src.y  << 16) >> sr) & 0xff) << 16;
    dst.x |= ((src.y          >> sr) & 0xff) << 24;
    dst.y  = ((src.z   << 16) >> sr) & 0xff;
    dst.y |= ((src.z          >> sr) & 0xff) <<  8;
    dst.y |= (((src.w  << 16) >> sr) & 0xff) << 16;
    dst.y |= ((src.w          >> sr) & 0xff) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}

int HipExec_ColorDepth_U8_S16_Wrap(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Wrap,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const short int *)pHipSrcImage, srcImageStrideInBytes,
                    shift);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Sat(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift) {
    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes>>1) + x;
    int4 src = *((int4 *)(&pSrcImage[srcIdx]));

    uint2 dst;
    int sr = shift;
    sr += 16;
    float4 f;
    f.x = (float)((src.x << 16) >> sr);
    f.y = (float)( src.x        >> sr);
    f.z = (float)((src.y << 16) >> sr);
    f.w = (float)( src.y        >> sr);
    dst.x = float4_to_uchars_u32(f);
    f.x = (float)((src.z << 16) >> sr);
    f.y = (float)( src.z        >> sr);
    f.z = (float)((src.w << 16) >> sr);
    f.w = (float)( src.w        >> sr);
    dst.y = float4_to_uchars_u32(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;

}
int HipExec_ColorDepth_U8_S16_Sat(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Sat,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const short int *)pHipSrcImage, srcImageStrideInBytes,
                    shift);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_S16_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    short int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const int shift) {
    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y * (dstImageStrideInBytes>>1) + x;
    unsigned int srcIdx =  y * (srcImageStrideInBytes) + x;

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));

    int4 dst;
    dst.x  =  (src.x & 0x000000ff) <<       shift ;
    dst.x |=  (src.x & 0x0000ff00) << ( 8 + shift);
    dst.y  =  (src.x & 0x00ff0000) >> (16 - shift);
    dst.y |=  (src.x & 0xff000000) >> ( 8 - shift);
    dst.z  =  (src.y & 0x000000ff) <<       shift ;
    dst.z |=  (src.y & 0x0000ff00) << ( 8 + shift);
    dst.w  =  (src.y & 0x00ff0000) >> (16 - shift);
    dst.w |=  (src.y & 0xff000000) >> ( 8 - shift);

    *((int4 *)(&pDstImage[dstIdx])) = dst;

}
int HipExec_ColorDepth_S16_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift
    ) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_S16_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (short int *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    shift);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelExtract kernels for hip backend
// ----------------------------------------------------------------------------

//**********************************
//ChannelExtract_U8_U16_Pos0
//**********************************
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes>>2) + (x*2);

    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float4 src1 = uchars_to_float4(pSrcImage[srcIdx + 1]);
    pDstImage[dstIdx] = float4_to_uchars(make_float4(src0.x, src0.z, src1.x, src0.z));
}
int HipExec_ChannelExtract_U8_U16_Pos0(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U16_Pos1
**********************************/

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes>>2) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes>>2) + (x*2);

    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float4 src1 = uchars_to_float4(pSrcImage[srcIdx + 1]);
    pDstImage[dstIdx] = float4_to_uchars(make_float4(src0.y, src0.w, src1.y, src0.w));
}
int HipExec_ChannelExtract_U8_U16_Pos1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos0
**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);
    pDstImage[dstIdx] = pSrcImage1[src1Idx];
}
int HipExec_ChannelExtract_U8_U32_Pos0(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos1

**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 1];
}
int HipExec_ChannelExtract_U8_U32_Pos1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos2
**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 2];
}
int HipExec_ChannelExtract_U8_U32_Pos2(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos2,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos3
**********************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 4);
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 3];
}
int HipExec_ChannelExtract_U8_U32_Pos3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth), globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos3,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos0
*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    unsigned int src1Idx = y * (srcImage1StrideInBytes) + (x * 3);
    pDstImage[dstIdx] = pSrcImage1[src1Idx];
}
int HipExec_ChannelExtract_U8_U24_Pos0(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos0,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos1
*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x *3);
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 1];
}
int HipExec_ChannelExtract_U8_U24_Pos1(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos1,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U24_Pos2
*****************************/
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x *3);
    pDstImage[dstIdx] = pSrcImage1[src1Idx + 2];
}
int HipExec_ChannelExtract_U8_U24_Pos2(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos2,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8_U24(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage0, unsigned char *pDstImage1, unsigned char *pDstImage2,
    unsigned int dstImageStrideInBytes, const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dst0Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst1Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst2Idx = y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + (x * 3);
    pDstImage0[dst0Idx] = pSrcImage[srcIdx];
    pDstImage1[dst1Idx] = pSrcImage[srcIdx + 1];
    pDstImage2[dst2Idx] = pSrcImage[srcIdx + 2];
}
int HipExec_ChannelExtract_U8U8U8_U24(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
    vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8_U24,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage0, (unsigned char *)pHipDstImage1, (unsigned char *)pHipDstImage2,
                        dstImageStrideInBytes, (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8_U32(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage0, unsigned char *pDstImage1, unsigned char *pDstImage2,
    unsigned int dstImageStrideInBytes, const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dst0Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst1Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst2Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst3Idx = y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + (x * 4);
    pDstImage0[dst0Idx] = pSrcImage[srcIdx];
    pDstImage1[dst1Idx] = pSrcImage[srcIdx + 1];
    pDstImage2[dst2Idx] = pSrcImage[srcIdx + 2];
}
int HipExec_ChannelExtract_U8U8U8_U32(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
    vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8_U32,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage0, (unsigned char *)pHipDstImage1, (unsigned char *)pHipDstImage2,
                        dstImageStrideInBytes, (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8U8_U32(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage0, unsigned char *pDstImage1, unsigned char *pDstImage2, unsigned char *pDstImage3,
    unsigned int dstImageStrideInBytes, const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dst0Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst1Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst2Idx = y * (dstImageStrideInBytes) + x;
    unsigned int dst3Idx = y * (dstImageStrideInBytes) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes) + (x * 4);
    pDstImage0[dst0Idx] = pSrcImage[srcIdx];
    pDstImage1[dst1Idx] = pSrcImage[srcIdx + 1];
    pDstImage2[dst2Idx] = pSrcImage[srcIdx + 2];
    pDstImage3[dst3Idx] = pSrcImage[srcIdx + 3];
}
int HipExec_ChannelExtract_U8U8U8U8_U32(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3,
    vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8U8_U32,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage0, (unsigned char *)pHipDstImage1, (unsigned char *)pHipDstImage2, (unsigned char *)pHipDstImage3,
                        dstImageStrideInBytes, (const unsigned char *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelCombine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U16_U8U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*2);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx];  
}
int HipExec_ChannelCombine_U16_U8U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U16_U8U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U24_U8U8U8_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*3);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx]; 
    pDstImage[dstIdx+2]  = pSrcImage3[src3Idx];  
}
int HipExec_ChannelCombine_U24_U8U8U8_RGB(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U24_U8U8U8_RGB,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage2[src2Idx];
    pDstImage[dstIdx+1]  = pSrcImage1[src1Idx]; 
    pDstImage[dstIdx+2]  = pSrcImage3[src3Idx]; 
    pDstImage[dstIdx+3]  = pSrcImage1[src1Idx+1];  
}
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_UYVY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x*4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x*2);
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx];
    pDstImage[dstIdx+1]  = pSrcImage2[src2Idx];
    pDstImage[dstIdx+2]  = pSrcImage1[src1Idx+1]; 
    pDstImage[dstIdx+3]  = pSrcImage3[src3Idx];
}
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_YUYV,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8U8_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *pSrcImage2, unsigned int srcImage2StrideInBytes,
    const unsigned char *pSrcImage3, unsigned int srcImage3StrideInBytes,
    const unsigned char *pSrcImage4, unsigned int srcImage4StrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x;
    unsigned int src2Idx =  y*(srcImage2StrideInBytes) + x;
    unsigned int src3Idx =  y*(srcImage3StrideInBytes) + x;
    unsigned int src4Idx =  y*(srcImage4StrideInBytes) + x;

    float4 dst = make_float4((float)pSrcImage1[src1Idx], (float)pSrcImage2[src2Idx], (float)pSrcImage3[src3Idx], (float)pSrcImage4[src4Idx]);
    pDstImage[dstIdx] = float4_to_uchars(dst); 
}
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
    const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8U8_RGBX,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned int *)pHipDstImage , dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes,
                    (const unsigned char *)pHipSrcImage2, srcImage2StrideInBytes,
                    (const unsigned char *)pHipSrcImage3, srcImage3StrideInBytes,
                    (const unsigned char *)pHipSrcImage4, srcImage4StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorConvert kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 1) + (x * 3);

    uint2 src0 = uchars_to_uint2(pSrcImage[srcIdx]);
    uint2 src1 = uchars_to_uint2(pSrcImage[srcIdx + 1]);
    uint2 src2 = uchars_to_uint2(pSrcImage[srcIdx + 2]);
    uint4 dst0 = make_uint4(src0.x, src0.y, src1.x, 255);
    uint4 dst1 = make_uint4(src1.y, src2.x, src2.y, 255);
    pDstImage[dstIdx] = uint4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = uint4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_RGB(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage1, unsigned int srcImage1StrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 3);
    unsigned int src1Idx = y * (srcImage1StrideInBytes >> 2) + (x * 4);

    uint4 src0 = uchars_to_uint4(pSrcImage1[src1Idx]);
    uint4 src1 = uchars_to_uint4(pSrcImage1[src1Idx + 1]);
    uint4 src2 = uchars_to_uint4(pSrcImage1[src1Idx + 2]);
    uint4 src3 = uchars_to_uint4(pSrcImage1[src1Idx + 3]);
    uint4 dst1 = make_uint4(src0.x, src0.y, src0.z, src1.x);
    uint4 dst2 = make_uint4(src1.y, src1.z, src2.x, src2.y);
    uint4 dst3 = make_uint4(src2.z, src3.x, src3.y, src3.z);
    pDstImage[dstIdx] = uint4_to_uchars(dst1);
    pDstImage[dstIdx + 1] = uint4_to_uchars(dst2);
    pDstImage[dstIdx + 2] = uint4_to_uchars(dst3);
}
int HipExec_ColorConvert_RGB_RGBX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 1) + (x * 3);
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 2) + x;

    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float2 dst0 = make_float2(PIXELSATURATEU8(YUV2R(src0.x, src0.y, src0.w)), PIXELSATURATEU8(YUV2G(src0.x, src0.y, src0.w)));
    float2 dst1 = make_float2(PIXELSATURATEU8(YUV2B(src0.x, src0.y, src0.w)), PIXELSATURATEU8(YUV2R(src0.z, src0.y, src0.w)));
    float2 dst2 = make_float2(PIXELSATURATEU8(YUV2G(src0.z, src0.y, src0.w)), PIXELSATURATEU8(YUV2B(src0.z, src0.y, src0.w)));
    pDstImage[dstIdx] = float2_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float2_to_uchars(dst1);
    pDstImage[dstIdx + 2] = float2_to_uchars(dst2);

    /*Uses Integer destination process 4 pixels in one thread*/

    // int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    // unsigned int dstIdx = y * (dstImageStrideInBytes>>2) + (x * 3);
    // unsigned int src1Idx = y * (srcImage1StrideInBytes>>2) + (x * 2);

    // float4 src0 = uchars_to_float4(pSrcImage1[src1Idx]);
    // float4 src1 = uchars_to_float4(pSrcImage1[src1Idx + 1]);

    // float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(src0.x, src0.y, src0.w)), PIXELSATURATEU8(YUV2G(src0.x, src0.y, src0.w)),
    //                         PIXELSATURATEU8(YUV2B(src0.x, src0.y, src0.w)), PIXELSATURATEU8(YUV2R(src0.z, src0.y, src0.w)));
    // float4 dst1 = make_float4(PIXELSATURATEU8(YUV2G(src0.z, src0.y, src0.w)), PIXELSATURATEU8(YUV2B(src0.z, src0.y, src0.w)),
    //                         PIXELSATURATEU8(YUV2R(src1.x, src1.y, src1.w)), PIXELSATURATEU8(YUV2G(src1.x, src1.y, src1.w)));
    // float4 dst2 = make_float4(PIXELSATURATEU8(YUV2B(src1.x, src1.y, src1.w)), PIXELSATURATEU8(YUV2R(src1.z, src1.y, src1.w)),
    //                         PIXELSATURATEU8(YUV2G(src1.z, src1.y, src1.w)), PIXELSATURATEU8(YUV2B(src1.z, src1.y, src1.w)));

    // pDstImage[dstIdx] = float4_to_uchars(dst0);
    // pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
    // pDstImage[dstIdx + 2] = float4_to_uchars(dst2);
}
int HipExec_ColorConvert_RGB_YUYV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 1) + (x * 3);
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 2) + x;

    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float2 dst0 = make_float2(PIXELSATURATEU8(YUV2R(src0.y, src0.x, src0.z)), PIXELSATURATEU8(YUV2G(src0.y, src0.x, src0.z)));
    float2 dst1 = make_float2(PIXELSATURATEU8(YUV2B(src0.y, src0.x, src0.z)), PIXELSATURATEU8(YUV2R(src0.w, src0.x, src0.z)));
    float2 dst2 = make_float2(PIXELSATURATEU8(YUV2G(src0.w, src0.x, src0.z)), PIXELSATURATEU8(YUV2B(src0.w, src0.x, src0.z)));
    pDstImage[dstIdx] = float2_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float2_to_uchars(dst1);
    pDstImage[dstIdx + 2] = float2_to_uchars(dst2); 

    /*Uses Integer destination process 4 pixels in one thread*/

    // int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    // unsigned int dstIdx = y * (dstImageStrideInBytes>>2) + (x * 3);
    // unsigned int src1Idx = y * (srcImage1StrideInBytes>>2) + (x * 2);

    // float4 src0 = uchars_to_float4(pSrcImage1[src1Idx]);
    // float4 src1 = uchars_to_float4(pSrcImage1[src1Idx + 1]);

    // float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(src0.y, src0.x, src0.z)), PIXELSATURATEU8(YUV2G(src0.y, src0.x, src0.z)),
    //                         PIXELSATURATEU8(YUV2B(src0.y, src0.x, src0.z)), PIXELSATURATEU8(YUV2R(src0.w, src0.x, src0.z)));
    // float4 dst1 = make_float4(PIXELSATURATEU8(YUV2G(src0.w, src0.x, src0.z)), PIXELSATURATEU8(YUV2B(src0.w, src0.x, src0.z)),
    //                         PIXELSATURATEU8(YUV2R(src1.y, src1.x, src1.z)), PIXELSATURATEU8(YUV2G(src1.y, src1.x, src1.z)));
    // float4 dst2 = make_float4(PIXELSATURATEU8(YUV2B(src1.y, src1.x, src1.z)), PIXELSATURATEU8(YUV2R(src1.w, src1.x, src1.z)),
    //                         PIXELSATURATEU8(YUV2G(src1.w, src1.x, src1.z)), PIXELSATURATEU8(YUV2B(src1.w, src1.x, src1.z)));

    // pDstImage[dstIdx] = float4_to_uchars(dst0);
    // pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
    // pDstImage[dstIdx + 2] = float4_to_uchars(dst2);                       
}
int HipExec_ColorConvert_RGB_UYVY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 2) + x;

    float4 src = uchars_to_float4(pSrcImage[srcIdx]);
    float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(src.x,src.y,src.w)),PIXELSATURATEU8(YUV2G(src.x,src.y,src.w)),
                            PIXELSATURATEU8(YUV2B(src.x,src.y,src.w)), 255.0);
    float4 dst1 = make_float4(PIXELSATURATEU8(YUV2R(src.z,src.y,src.w)),PIXELSATURATEU8(YUV2G(src.z,src.y,src.w)),
                            PIXELSATURATEU8(YUV2B(src.z,src.y,src.w)), 255.0);
    pDstImage[dstIdx] = float4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_YUYV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 2) + x;

    float4 src = uchars_to_float4(pSrcImage[srcIdx]);
    float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(src.y,src.x,src.z)),PIXELSATURATEU8(YUV2G(src.y,src.x,src.z)),
                            PIXELSATURATEU8(YUV2B(src.y,src.x,src.z)), 255.0);
    float4 dst1 = make_float4(PIXELSATURATEU8(YUV2R(src.w,src.x,src.z)),PIXELSATURATEU8(YUV2G(src.w,src.x,src.z)),
                            PIXELSATURATEU8(YUV2B(src.w,src.x,src.z)), 255.0);
    pDstImage[dstIdx] = float4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_UYVY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcYImage, unsigned int srcYImageStrideInBytes,
    const unsigned char *pSrcUImage, unsigned int srcUImageStrideInBytes,
    const unsigned char *pSrcVImage, unsigned int srcVImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 1) + (x * 3);
    unsigned int srcYIdx = y * (srcYImageStrideInBytes >> 1) + x;
    unsigned int srcUIdx = (y >> 1) * (srcUImageStrideInBytes) + x;
    unsigned int srcVIdx = (y >> 1) * (srcVImageStrideInBytes) + x;

    float2 srcY = uchars_to_float2(pSrcYImage[srcYIdx]);
    float srcU = pSrcUImage[srcUIdx];
    float srcV = pSrcVImage[srcVIdx];
    float2 dst0 = make_float2(PIXELSATURATEU8(YUV2R(srcY.x, srcU, srcV)), PIXELSATURATEU8(YUV2G(srcY.x, srcU, srcV)));
    float2 dst1 = make_float2(PIXELSATURATEU8(YUV2B(srcY.x, srcU, srcV)), PIXELSATURATEU8(YUV2R(srcY.y, srcU, srcV)));
    float2 dst2 = make_float2(PIXELSATURATEU8(YUV2G(srcY.y, srcU, srcV)), PIXELSATURATEU8(YUV2B(srcY.y, srcU, srcV)));
    pDstImage[dstIdx] = float2_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float2_to_uchars(dst1);
    pDstImage[dstIdx + 2] = float2_to_uchars(dst2);

    /*Uses Integer destination process 4 pixels in one thread*/

    // int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    // if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    // unsigned int dstIdx = y * (dstImageStrideInBytes>>2) + (x * 3);
    // unsigned int srcYIdx = y * (srcYImageStrideInBytes>>2) + x;
    // unsigned int srcUIdx = (y>>1) * (srcUImageStrideInBytes>>1) + x;
    // unsigned int srcVIdx = (y>>1) * (srcVImageStrideInBytes>>1) + x;

    // float4 srcY = uchars_to_float4(pSrcYImage[srcYIdx]);
    // float2 srcU = uchars_to_float2(pSrcUImage[srcUIdx]);
    // float2 srcV = uchars_to_float2(pSrcVImage[srcVIdx]);

    // float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(srcY.x, srcU.x, srcV.x)), PIXELSATURATEU8(YUV2G(srcY.x, srcU.x, srcV.x)),
    //                         PIXELSATURATEU8(YUV2B(srcY.x, srcU.x, srcV.x)), PIXELSATURATEU8(YUV2R(srcY.y, srcU.x, srcV.x)));
    // float4 dst1 = make_float4(PIXELSATURATEU8(YUV2G(srcY.y, srcU.x, srcV.x)), PIXELSATURATEU8(YUV2B(srcY.y, srcU.x, srcV.x)),
    //                         PIXELSATURATEU8(YUV2R(srcY.z, srcU.y, srcV.y)), PIXELSATURATEU8(YUV2G(srcY.z, srcU.y, srcV.y)));
    // float4 dst2 = make_float4(PIXELSATURATEU8(YUV2B(srcY.z, srcU.y, srcV.y)), PIXELSATURATEU8(YUV2R(srcY.w, srcU.y, srcV.y)),
    //                         PIXELSATURATEU8(YUV2G(srcY.w, srcU.y, srcV.y)), PIXELSATURATEU8(YUV2B(srcY.w, srcU.y, srcV.y)));

    // pDstImage[dstIdx] = float4_to_uchars(dst0);
    // pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
    // pDstImage[dstIdx + 2] = float4_to_uchars(dst2); 
}
int HipExec_ColorConvert_RGB_IYUV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_IYUV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcYImage, srcYImageStrideInBytes,
                       (const unsigned char *)pHipSrcUImage, srcUImageStrideInBytes,
                       (const unsigned char *)pHipSrcVImage, srcVImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned short *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 1) + (x * 3);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes >> 1) + x;
    unsigned int srcChromaIdx = (y >> 1) * (srcChromaImageStrideInBytes >> 1) + x;

    float2 srcLuma = uchars_to_float2(pSrcLumaImage[srcLumaIdx]);
    float2 srcChroma = uchars_to_float2(pSrcChromaImage[srcChromaIdx]);
    float2 dst0 = make_float2(PIXELSATURATEU8(YUV2R(srcLuma.x, srcChroma.x, srcChroma.y)), PIXELSATURATEU8(YUV2G(srcLuma.x, srcChroma.x, srcChroma.y)));
    float2 dst1 = make_float2(PIXELSATURATEU8(YUV2B(srcLuma.x, srcChroma.x, srcChroma.y)), PIXELSATURATEU8(YUV2R(srcLuma.y, srcChroma.x, srcChroma.y)));
    float2 dst2 = make_float2(PIXELSATURATEU8(YUV2G(srcLuma.y, srcChroma.x, srcChroma.y)), PIXELSATURATEU8(YUV2B(srcLuma.y, srcChroma.x, srcChroma.y)));
    pDstImage[dstIdx] = float2_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float2_to_uchars(dst1);
    pDstImage[dstIdx + 2] = float2_to_uchars(dst2);
}
int HipExec_ColorConvert_RGB_NV12(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV12,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned short *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned short *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 1) + (x * 3);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes >> 1) + x;
    unsigned int srcChromaIdx = (y >> 1) * (srcChromaImageStrideInBytes >> 1) + x;

    float2 srcLuma = uchars_to_float2(pSrcLumaImage[srcLumaIdx]);
    float2 srcChroma = uchars_to_float2(pSrcChromaImage[srcChromaIdx]);
    float2 dst0 = make_float2(PIXELSATURATEU8(YUV2R(srcLuma.x, srcChroma.y, srcChroma.x)), PIXELSATURATEU8(YUV2G(srcLuma.x, srcChroma.y, srcChroma.x)));
    float2 dst1 = make_float2(PIXELSATURATEU8(YUV2B(srcLuma.x, srcChroma.y, srcChroma.x)), PIXELSATURATEU8(YUV2R(srcLuma.y, srcChroma.y, srcChroma.x)));
    float2 dst2 = make_float2(PIXELSATURATEU8(YUV2G(srcLuma.y, srcChroma.y, srcChroma.x)), PIXELSATURATEU8(YUV2B(srcLuma.y, srcChroma.y, srcChroma.x)));
    pDstImage[dstIdx] = float2_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float2_to_uchars(dst1);
    pDstImage[dstIdx + 2] = float2_to_uchars(dst2);
}
int HipExec_ColorConvert_RGB_NV21(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV21,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned short *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_IYUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned char *pSrcYImage, unsigned int srcYImageStrideInBytes,
    const unsigned char *pSrcUImage, unsigned int srcUImageStrideInBytes,
    const unsigned char *pSrcVImage, unsigned int srcVImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcYIdx = y * (srcYImageStrideInBytes) + (x * 2);
    unsigned int srcUIdx = (y >> 1) * (srcUImageStrideInBytes) + x;
    unsigned int srcVIdx = (y >> 1) * (srcVImageStrideInBytes) + x;

    unsigned char Y0, Y1, U, V;
    Y0 = pSrcYImage[srcYIdx];
    Y1 = pSrcYImage[srcYIdx + 1];
    U = pSrcUImage[srcUIdx];
    V = pSrcVImage[srcVIdx];
    float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(Y0,U,V)), PIXELSATURATEU8(YUV2G(Y0,U,V)),
                            PIXELSATURATEU8(YUV2B(Y0,U,V)), 255);
    float4 dst1 = make_float4(PIXELSATURATEU8(YUV2R(Y1,U,V)), PIXELSATURATEU8(YUV2G(Y1,U,V)),
                            PIXELSATURATEU8(YUV2B(Y1,U,V)), 255);
    pDstImage[dstIdx] = float4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_IYUV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1 , globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_IYUV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned char *)pHipSrcYImage, srcYImageStrideInBytes,
                       (const unsigned char *)pHipSrcUImage, srcUImageStrideInBytes,
                       (const unsigned char *)pHipSrcVImage, srcVImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned short *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes >> 1) + x;
    unsigned int srcChromaIdx = (y >> 1) * (srcChromaImageStrideInBytes >> 1) + x;

    float2 srcLuma = uchars_to_float2(pSrcLumaImage[srcLumaIdx]);
    float2 srcChroma = uchars_to_float2(pSrcChromaImage[srcChromaIdx]);
    float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(srcLuma.x,srcChroma.x,srcChroma.y)), PIXELSATURATEU8(YUV2G(srcLuma.x,srcChroma.x,srcChroma.y)),
                            PIXELSATURATEU8(YUV2B(srcLuma.x,srcChroma.x,srcChroma.y)), 255);
    float4 dst1 = make_float4(PIXELSATURATEU8(YUV2R(srcLuma.y,srcChroma.x,srcChroma.y)), PIXELSATURATEU8(YUV2G(srcLuma.y,srcChroma.x,srcChroma.y)),
                            PIXELSATURATEU8(YUV2B(srcLuma.y,srcChroma.x,srcChroma.y)), 255);
    pDstImage[dstIdx] = float4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_NV12(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV12,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned short *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV21(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcLumaImage, unsigned int srcLumaImageStrideInBytes,
    const unsigned short *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + (x * 2);
    unsigned int srcLumaIdx = y * (srcLumaImageStrideInBytes >> 1) + x;
    unsigned int srcChromaIdx = (y >> 1) * (srcChromaImageStrideInBytes >> 1) + x;

    float2 srcLuma = uchars_to_float2(pSrcLumaImage[srcLumaIdx]);
    float2 srcChroma = uchars_to_float2(pSrcChromaImage[srcChromaIdx]);
    float4 dst0 = make_float4(PIXELSATURATEU8(YUV2R(srcLuma.x,srcChroma.y,srcChroma.x)), PIXELSATURATEU8(YUV2G(srcLuma.x,srcChroma.y,srcChroma.x)),
                            PIXELSATURATEU8(YUV2B(srcLuma.x,srcChroma.y,srcChroma.x)), 255);
    float4 dst1 = make_float4(PIXELSATURATEU8(YUV2R(srcLuma.y,srcChroma.y,srcChroma.x)), PIXELSATURATEU8(YUV2G(srcLuma.y,srcChroma.y,srcChroma.x)),
                            PIXELSATURATEU8(YUV2B(srcLuma.y,srcChroma.y,srcChroma.x)), 255);
    pDstImage[dstIdx] = float4_to_uchars(dst0);
    pDstImage[dstIdx + 1] = float4_to_uchars(dst1);
}
int HipExec_ColorConvert_RGBX_NV21(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV21,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstImageStrideInBytes,
                       (const unsigned short *)pHipSrcLumaImage, srcLumaImageStrideInBytes,
                       (const unsigned short *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_NV12_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImageLuma, unsigned int dstImageLumaStrideInBytes,
    unsigned short *pDstImageChroma, unsigned int dstImageChromaStrideInBytes,
    const unsigned short *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y * 2 >= dstHeight))    return;
    unsigned int dstIdxLuma = y * ((dstImageLumaStrideInBytes >> 1) << 1) + x;
    unsigned int dstIdxChroma = y * (dstImageChromaStrideInBytes >> 1) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 1) << 1) + (x * 3);

    float Upix, Vpix;
    float2 src0 = uchars_to_float2(pSrcImage[srcIdx]);
    float2 src1 = uchars_to_float2(pSrcImage[srcIdx + 1]);
    float2 src2 = uchars_to_float2(pSrcImage[srcIdx + 2]);
    pDstImageLuma[dstIdxLuma] = float2_to_uchars(make_float2(RGB2Y(src0.x,src0.y,src1.x), RGB2Y(src1.y,src2.x,src2.y)));
    Upix = RGB2U(src0.x,src0.y,src1.x) + RGB2U(src1.y,src2.x,src2.y);
    Vpix = RGB2V(src0.x,src0.y,src1.x) + RGB2V(src1.y,src2.x,src2.y);
    src0 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1)]);
    src1 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1) + 1]);
    src2 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1) + 2]);
    pDstImageLuma[dstIdxLuma + (dstImageLumaStrideInBytes >> 1)] = float2_to_uchars(make_float2(RGB2Y(src0.x,src0.y,src1.x), RGB2Y(src1.y,src2.x,src2.y)));
    Upix += (RGB2U(src0.x,src0.y,src1.x) + RGB2U(src1.y,src2.x,src2.y));
    Vpix += (RGB2V(src0.x,src0.y,src1.x) + RGB2V(src1.y,src2.x,src2.y));
    pDstImageChroma[dstIdxChroma] = float2_to_uchars(make_float2(Upix/4, Vpix/4));
}
int HipExec_ColorConvert_NV12_RGB(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImageLuma, dstImageLumaStrideInBytes,
                       (unsigned short *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                       (const unsigned short *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_NV12_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstImageLuma, unsigned int dstImageLumaStrideInBytes,
    unsigned short *pDstImageChroma, unsigned int dstImageChromaStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y * 2 >= dstHeight))    return;
    unsigned int dstIdxLuma = y * ((dstImageLumaStrideInBytes >> 1) << 1) + x;
    unsigned int dstIdxChroma = y * (dstImageChromaStrideInBytes >> 1) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 2) << 1) + (x * 2);
    
    float Upix, Vpix;
    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float4 src1 = uchars_to_float4(pSrcImage[srcIdx + 1]);
    pDstImageLuma[dstIdxLuma] = float2_to_uchars(make_float2(nearbyintf(RGB2Y(src0.x,src0.y,src0.z)), nearbyintf(RGB2Y(src1.x,src1.y,src1.z))));
    Upix = RGB2U(src0.x,src0.y,src0.z) + RGB2U(src1.x,src1.y,src1.z);
    Vpix = RGB2V(src0.x,src0.y,src0.z) + RGB2V(src1.x,src1.y,src1.z);
    src0 = uchars_to_float4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    src1 = uchars_to_float4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    pDstImageLuma[dstIdxLuma + (dstImageLumaStrideInBytes >> 1)] = float2_to_uchars(make_float2(nearbyintf(RGB2Y(src0.x,src0.y,src0.z)), nearbyintf(RGB2Y(src1.x,src1.y,src1.z))));
    Upix += (RGB2U(src0.x,src0.y,src0.z) + RGB2U(src1.x,src1.y,src1.z));
    Vpix += (RGB2V(src0.x,src0.y,src0.z) + RGB2V(src1.x,src1.y,src1.z));
    pDstImageChroma[dstIdxChroma] = float2_to_uchars(make_float2(nearbyintf(Upix/4), nearbyintf(Vpix/4)));
}
int HipExec_ColorConvert_NV12_RGBX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstImageLuma, dstImageLumaStrideInBytes,
                       (unsigned short *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                       (const unsigned int *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned short *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y * 2 >= dstHeight)) return;
    unsigned int dstYIdx = y * ((dstYImageStrideInBytes >> 1) << 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 1) << 1) + (x * 3);

    float Upix, Vpix;
    float2 src0 = uchars_to_float2(pSrcImage[srcIdx]);
    float2 src1 = uchars_to_float2(pSrcImage[srcIdx + 1]);
    float2 src2 = uchars_to_float2(pSrcImage[srcIdx + 2]);
    pDstYImage[dstYIdx] = float2_to_uchars(make_float2(RGB2Y(src0.x,src0.y,src1.x), RGB2Y(src1.y,src2.x,src2.y)));
    Upix = RGB2U(src0.x,src0.y,src1.x) + RGB2U(src1.y,src2.x,src2.y);
    Vpix = RGB2V(src0.x,src0.y,src1.x) + RGB2V(src1.y,src2.x,src2.y);
    src0 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1)]);
    src1 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1) + 1]);
    src2 = uchars_to_float2(pSrcImage[srcIdx + (srcImageStrideInBytes >> 1) + 2]);
    pDstYImage[dstYIdx + (dstYImageStrideInBytes >> 1)] = float2_to_uchars(make_float2(RGB2Y(src0.x,src0.y,src1.x), RGB2Y(src1.y,src2.x,src2.y)));
    Upix += (RGB2U(src0.x,src0.y,src1.x) + RGB2U(src1.y,src2.x,src2.y));
    Vpix += (RGB2V(src0.x,src0.y,src1.x) + RGB2V(src1.y,src2.x,src2.y));
    pDstUImage[dstUIdx] = Upix/4;
    pDstVImage[dstVIdx] = Vpix/4;
}
int HipExec_ColorConvert_IYUV_RGB(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned short *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y * 2 >= dstHeight)) return;
    unsigned int dstYIdx = y * ((dstYImageStrideInBytes >> 1) << 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 2) << 1) + (x * 2);

    float Upix, Vpix;
    float4 src0 = uchars_to_float4(pSrcImage[srcIdx]);
    float4 src1 = uchars_to_float4(pSrcImage[srcIdx + 1]);
    pDstYImage[dstYIdx] = float2_to_uchars(make_float2(nearbyintf(RGB2Y(src0.x,src0.y,src0.z)), nearbyintf(RGB2Y(src1.x,src1.y,src1.z))));
    Upix = RGB2U(src0.x,src0.y,src0.z) + RGB2U(src1.x,src1.y,src1.z);
    Vpix = RGB2V(src0.x,src0.y,src0.z) + RGB2V(src1.x,src1.y,src1.z);
    src0 = uchars_to_float4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    src1 = uchars_to_float4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    pDstYImage[dstYIdx + (dstYImageStrideInBytes >> 1)] = float2_to_uchars(make_float2(nearbyintf(RGB2Y(src0.x,src0.y,src0.z)), nearbyintf(RGB2Y(src1.x,src1.y,src1.z))));
    Upix += (RGB2U(src0.x,src0.y,src0.z) + RGB2U(src1.x,src1.y,src1.z));
    Vpix += (RGB2V(src0.x,src0.y,src0.z) + RGB2V(src1.x,src1.y,src1.z));
    pDstUImage[dstUIdx] = nearbyintf(Upix/4);
    pDstVImage[dstVIdx] = nearbyintf(Vpix/4);
}
int HipExec_ColorConvert_IYUV_RGBX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_YUV4_RGB(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned short *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned short *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned short *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y  >= dstHeight))    return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes >> 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes >> 1) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes >> 1) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 1) + (x * 3);

    float2 src0 = uchars_to_float2(pSrcImage[srcIdx]);
    float2 src1 = uchars_to_float2(pSrcImage[srcIdx + 1]);
    float2 src2 = uchars_to_float2(pSrcImage[srcIdx + 2]);
    float2 dstY = make_float2(nearbyintf(RGB2Y(src0.x, src0.y, src1.x)),nearbyintf(RGB2Y(src1.y, src2.x, src2.y)));
    float2 dstU = make_float2(nearbyintf(RGB2U(src0.x, src0.y, src1.x)),nearbyintf(RGB2U(src1.y, src2.x, src2.y)));
    float2 dstV = make_float2(nearbyintf(RGB2V(src0.x, src0.y, src1.x)),nearbyintf(RGB2V(src1.y, src2.x, src2.y)));
    pDstYImage[dstYIdx] = float2_to_uchars(dstY);
    pDstUImage[dstUIdx] = float2_to_uchars(dstU);
    pDstVImage[dstVIdx] = float2_to_uchars(dstV);
}
int HipExec_ColorConvert_YUV4_RGB(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;
    
    hipLaunchKernelGGL(Hip_ColorConvert_YUV4_RGB,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned short *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned short *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned short *)pHipSrcImage, srcImageStrideInBytes);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_YUV4_RGBX(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned short *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned short *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y  >= dstHeight))    return;
    unsigned int dstYIdx = y * (dstYImageStrideInBytes >> 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes >> 1) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes >> 1) + x;
    unsigned int srcIdx = y * (srcImageStrideInBytes >> 2) + (x * 2);

    float4 rgb0 = uchars_to_float4(pSrcImage[srcIdx]);
    float4 rgb1 = uchars_to_float4(pSrcImage[srcIdx+1]);
    float2 dstY = make_float2(nearbyintf(RGB2Y(rgb0.x, rgb0.y, rgb0.z)),nearbyintf(RGB2Y(rgb1.x, rgb1.y, rgb1.z)));
    float2 dstU = make_float2(nearbyintf(RGB2U(rgb0.x, rgb0.y, rgb0.z)),nearbyintf(RGB2U(rgb1.x, rgb1.y, rgb1.z)));
    float2 dstV = make_float2(nearbyintf(RGB2V(rgb0.x, rgb0.y, rgb0.z)),nearbyintf(RGB2V(rgb1.x, rgb1.y, rgb1.z)));
    pDstYImage[dstYIdx] = float2_to_uchars(dstY);
    pDstUImage[dstUIdx] = float2_to_uchars(dstU);
    pDstVImage[dstVIdx] = float2_to_uchars(dstV);
}
int HipExec_ColorConvert_YUV4_RGBX(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_YUV4_RGBX,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned short *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned short *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;   
}

// ----------------------------------------------------------------------------
// VxFormatConvert kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstLumaImage, unsigned int dstLumaImageStrideInBytes,
    unsigned int *pDstChromaImage, unsigned int dstChromaImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y*2 >= dstHeight)) return;
    unsigned int dstIdxLuma = y * ((dstLumaImageStrideInBytes>>2)<<1) + x;
    unsigned int dstIdxChroma = y * (dstChromaImageStrideInBytes>>2) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes>>2)<<1) + (x * 2);

    uint4 src0 = uchars_to_uint4(pSrcImage[srcIdx]);
    uint4 src1 = uchars_to_uint4(pSrcImage[srcIdx + 1]);
    uint4 src2 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    uint4 src3 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    uint4 dstY0 = make_uint4(src0.y, src0.w, src1.y, src1.w);
    uint4 dstY1 = make_uint4(src2.y, src2.w, src3.y, src3.w);
    uint4 dstUV = make_uint4((src0.x + src2.x)/2 , (src0.z + src2.z)/2,
                              (src1.x + src3.x)/2, (src1.z + src3.z)/2);
    pDstLumaImage[dstIdxLuma] = uint4_to_uchars(dstY0);
    pDstLumaImage[dstIdxLuma + (dstLumaImageStrideInBytes >> 2)] = uint4_to_uchars(dstY1);
    pDstChromaImage[dstIdxChroma] = uint4_to_uchars(dstUV);
}
int HipExec_FormatConvert_NV12_UYVY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
    vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_FormatConvert_NV12_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pDstLumaImage, dstLumaImageStrideInBytes,
                       (unsigned int *)pDstChromaImage, dstChromaImageStrideInBytes,
                       (const unsigned int *)pSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstLumaImage, unsigned int dstLumaImageStrideInBytes,
    unsigned int *pDstChromaImage, unsigned int dstChromaImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y * 2 >= dstHeight)) return;
    unsigned int dstIdxLuma = y * ((dstLumaImageStrideInBytes >> 2) << 1) + x;
    unsigned int dstIdxChroma = y * (dstChromaImageStrideInBytes >> 2) + x;
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 2) << 1) + (x * 2);

    uint4 src0 = uchars_to_uint4(pSrcImage[srcIdx]);
    uint4 src1 = uchars_to_uint4(pSrcImage[srcIdx + 1]);
    uint4 src2 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    uint4 src3 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    uint4 dstY0 = make_uint4(src0.x, src0.z, src1.x, src1.z);
    uint4 dstY1 = make_uint4(src2.x, src2.z, src3.x, src3.z);
    uint4 dstUV = make_uint4((src0.y + src2.y) / 2 , (src0.w + src2.w) / 2 ,
                              (src1.y + src3.y) / 2, (src1.w + src3.w) / 2);
    pDstLumaImage[dstIdxLuma] = uint4_to_uchars(dstY0);
    pDstLumaImage[dstIdxLuma + (dstLumaImageStrideInBytes >> 2)] = uint4_to_uchars(dstY1);
    pDstChromaImage[dstIdxChroma] = uint4_to_uchars(dstUV);
}
int HipExec_FormatConvert_NV12_YUYV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
    vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_FormatConvert_NV12_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pDstLumaImage, dstLumaImageStrideInBytes,
                       (unsigned int *)pDstChromaImage, dstChromaImageStrideInBytes,
                       (const unsigned int *)pSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_UYVY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y * 2 >= dstHeight)) return;
    unsigned int dstYIdx = y * ((dstYImageStrideInBytes >> 2) << 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + (x * 2);
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + (x * 2);
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 2) << 1) + (x * 2);

    uint4 src0 = uchars_to_uint4(pSrcImage[srcIdx]);
    uint4 src1 = uchars_to_uint4(pSrcImage[srcIdx + 1]);
    uint4 src2 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    uint4 src3 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    uint4 dstY0 = make_uint4(src0.y, src0.w, src1.y, src1.w);
    uint4 dstY1 = make_uint4(src2.y, src2.w, src3.y, src3.w);
    pDstYImage[dstYIdx] = uint4_to_uchars(dstY0);
    pDstYImage[dstYIdx + (dstYImageStrideInBytes>>2)] = uint4_to_uchars(dstY1);
    pDstUImage[dstUIdx] = (src0.x + src2.x) / 2;
    pDstUImage[dstUIdx + 1] = (src1.x + src3.x) / 2;    
    pDstVImage[dstVIdx] = (src0.z + src2.z) / 2;
    pDstVImage[dstVIdx + 1] = (src1.z + src3.z) / 2;
}
int HipExec_FormatConvert_IYUV_UYVY(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_UYVY,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_YUYV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstYImage, unsigned int dstYImageStrideInBytes,
    unsigned char *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned char *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned int *pSrcImage, unsigned int srcImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y * 2 >= dstHeight)) return;
    unsigned int dstYIdx = y * ((dstYImageStrideInBytes >> 2) << 1) + x;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes) + (x * 2);
    unsigned int dstVIdx = y * (dstVImageStrideInBytes) + (x * 2);
    unsigned int srcIdx = y * ((srcImageStrideInBytes >> 2) << 1) + (x * 2);

    uint4 src0 = uchars_to_uint4(pSrcImage[srcIdx]);
    uint4 src1 = uchars_to_uint4(pSrcImage[srcIdx + 1]);
    uint4 src2 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2)]);
    uint4 src3 = uchars_to_uint4(pSrcImage[srcIdx + (srcImageStrideInBytes >> 2) + 1]);
    uint4 dstY0 = make_uint4(src0.x, src0.z, src1.x, src1.z);
    uint4 dstY1 = make_uint4(src2.x, src2.z, src3.x, src3.z);
    pDstYImage[dstYIdx] = uint4_to_uchars(dstY0);
    pDstYImage[dstYIdx + (dstYImageStrideInBytes >> 2)] = uint4_to_uchars(dstY1);
    pDstUImage[dstUIdx] = (src0.y + src2.y) / 2;
    pDstUImage[dstUIdx + 1] = (src1.y + src3.y) / 2;    
    pDstVImage[dstVIdx] = (src0.w + src2.w) / 2;
    pDstVImage[dstVIdx + 1] = (src1.w + src3.w) / 2;
}
int HipExec_FormatConvert_IYUV_YUYV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = (dstHeight + 3) >> 1;

    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_YUYV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstYImage, dstYImageStrideInBytes,
                       (unsigned char *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned char *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned int *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IUV_UV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned short *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned int *pSrcChromaImage, unsigned int srcChromaImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes>>1) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes>>1) + x;
    unsigned int srcChromaIdx = y * (srcChromaImageStrideInBytes>>2) + x;

    uint4 src = uchars_to_uint4(pSrcChromaImage[srcChromaIdx]);
    uint2 dstU = make_uint2(src.x, src.z);
    uint2 dstV = make_uint2(src.y, src.w);
    pDstUImage[dstUIdx] = uint2_to_uchars(dstU);
    pDstVImage[dstVIdx] = uint2_to_uchars(dstV);
}
int HipExec_FormatConvert_IUV_UV12(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_FormatConvert_IUV_UV12,
                    dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short *)pHipDstUImage, dstUImageStrideInBytes,
                    (unsigned short *)pHipDstVImage, dstVImageStrideInBytes,
                    (const unsigned int *)pHipSrcChromaImage, srcChromaImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_UV12_IUV(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstChromaImage, unsigned int dstChromaImageStrideInBytes,
    const unsigned short *pSrcUImage, unsigned int srcUImageStrideInBytes,
    const unsigned short *pSrcVImage, unsigned int srcVImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstChromaIdx = y * (dstChromaImageStrideInBytes >> 2) + x;
    unsigned int srcUIdx = y * (srcUImageStrideInBytes >> 1) + x;
    unsigned int srcVIdx = y * (srcVImageStrideInBytes >> 1) + x;

    uint2 srcU = uchars_to_uint2(pSrcUImage[srcUIdx]);
    uint2 srcV = uchars_to_uint2(pSrcVImage[srcVIdx]);
    uint4 dst = make_uint4(srcU.x, srcV.x, srcU.y, srcV.y);
    pDstChromaImage[dstChromaIdx] = uint4_to_uchars(dst);
}
int HipExec_FormatConvert_UV12_IUV(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_FormatConvert_UV12_IUV,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstChromaImage, dstChromaImageStrideInBytes,
                       (const unsigned short *)pHipSrcUImage, srcUImageStrideInBytes,
                       (const unsigned short *)pHipSrcVImage, srcVImageStrideInBytes);

    return VX_SUCCESS;   
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_UV_UV12(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned short *pDstUImage, unsigned int dstUImageStrideInBytes,
    unsigned short *pDstVImage, unsigned int dstVImageStrideInBytes,
    const unsigned short *pSrcCImage, unsigned int srcCImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 2 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstUIdx = y * (dstUImageStrideInBytes >> 1) + x;
    unsigned int dstVIdx = y * (dstVImageStrideInBytes >> 1) + x;
    unsigned int srcIdx = (y >> 1) * (srcCImageStrideInBytes >> 1) + x;

    uint2 src = uchars_to_uint2(pSrcCImage[srcIdx]);
    uint2 dstU = make_uint2(src.x, src.x);
    uint2 dstV = make_uint2(src.y, src.y);
    pDstUImage[dstUIdx] = uint2_to_uchars(dstU);
    pDstVImage[dstVIdx] = uint2_to_uchars(dstV);
}
int HipExec_FormatConvert_UV_UV12(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcCImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 1, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_FormatConvert_UV_UV12,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned short *)pHipDstUImage, dstUImageStrideInBytes,
                       (unsigned short *)pHipDstVImage, dstVImageStrideInBytes,
                       (const unsigned short *)pHipSrcCImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleUp2x2_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned int *pDstImage, unsigned int dstImageStrideInBytes,
    const unsigned short *pSrcImage, unsigned int srcCImageStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x * 4 >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes >> 2) + x;
    unsigned int srcIdx = (y >> 1) * (srcCImageStrideInBytes >> 1) + x;

    uint2 src = uchars_to_uint2(pSrcImage[srcIdx]);
    uint4 dst = make_uint4(src.x, src.x, src.y, src.y);
    pDstImage[dstIdx] = uint4_to_uchars(dst);
}
int HipExec_ScaleUp2x2_U8_U8(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstUImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ScaleUp2x2_U8_U8,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned int *)pHipDstImage, dstUImageStrideInBytes,
                       (const unsigned short *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}
