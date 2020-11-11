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

__device__ __forceinline__ float4 uchars_to_float4(uint src)
{
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint float4_to_uchars(float4 src)
{
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ uint float4_to_uchars_u32(float4 src)
{
    // return ((uint)src.x&0xFF)<<24 | (((uint)src.y&0xFF)<<16) | (((uint)src.z&0xFF)<<8)| (((uint)src.w&0xFF));
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}


__device__ __forceinline__ uint4 uchars_to_uint4(unsigned int src)
{
    printf("\nuchars_to_uint4 %d, %d, %d, %d", (unsigned int)(src&0xFF), (unsigned int)((src&0xFF00)>>8), (unsigned int)((src&0xFF0000)>>16), (unsigned int)((src&0xFF000000)>>24));
    return make_uint4((unsigned int)(src&0xFF), (unsigned int)((src&0xFF00)>>8), (unsigned int)((src&0xFF0000)>>16), (unsigned int)((src&0xFF000000)>>24));
}

__device__ __forceinline__ unsigned int uint4_to_uchars(uint4 src)
{
    printf("\nuint4_to_uchars %d, %d, %d, %d", ((unsigned char)src.x&0xFF), ((unsigned char)src.y&0xFF), ((unsigned char)src.z&0xFF), ((unsigned char)src.w&0xFF));
    return ((unsigned char)src.x&0xFF) | (((unsigned char)src.y&0xFF)<<8) | (((unsigned char)src.z&0xFF)<<16) | (((unsigned char)src.w&0xFF) << 24);
}

// ----------------------------------------------------------------------------
// VxLut kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes,
    const unsigned char *lut
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + (x * 4);
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + (x * 4);
    for (int i = 0; i < 4; i++)
        pDstImage[dstIdx + i] = lut[pSrcImage1[src1Idx + i]];
}
int HipExec_Lut_U8_U8(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_uint8 *lut
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    vx_uint8 *hipLut;
    hipMalloc(&hipLut, 2048);
    hipMemcpy(hipLut, lut, 2048, hipMemcpyHostToDevice);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_Lut_U8_U8,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes, (const unsigned char *)hipLut);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

    printf("\nHipExec_Lut_U8_U8: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorDepth kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxChannelExtract kernels for hip backend
// ----------------------------------------------------------------------------

//**********************************
//ChannelExtract_U8_U16_Pos0
//**********************************
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x*2;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x*4;

    for (int i = 0; i < 2; i++)
        pDstImage[dstIdx + i] = pSrcImage1[src1Idx + i*2];
}
int HipExec_ChannelExtract_U8_U16_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos0,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U16_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U16_Pos1
**********************************/

__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x*2;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x*4;

    for (int i = 0; i < 2; i++)
        pDstImage[dstIdx + i] = pSrcImage1[src1Idx + i*2 + 1];
    // printf("\n&pDstImage[dstIdx], &pDstImage[dstIdx + 1]: %p, %p", (void*)(&pDstImage[dstIdx]), (void*)(&pDstImage[dstIdx + 1]));
}
int HipExec_ChannelExtract_U8_U16_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U16_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


/**********************************
    ChannelExtract_U8_U32_Pos0
**********************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
//unsigned int
    // unsigned int dstIdx =  y*(dstImageStrideInBytes>>2) + x;
    // unsigned int src1Idx =  y*(srcImage1StrideInBytes>>2) + x * 4;
    
    // float4 dst1 = uchars_to_float4(pSrcImage1[src1Idx]) ;
    // float4 dst2 = uchars_to_float4(pSrcImage1[src1Idx+1]);
    // float4 dst3 = uchars_to_float4(pSrcImage1[src1Idx+2] )  ;
    // float4 dst4 = uchars_to_float4(pSrcImage1[src1Idx+3]);

    // float4 dst = make_float4(dst1.x,dst2.x,dst3.x,dst4.x);
    // pDstImage[dstIdx] = float4_to_uchars(dst);
    

// unsigned char
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x * 4 ;

    for(int i =0;i<1;i++)
        pDstImage[dstIdx+i]  = pSrcImage1[src1Idx+(i*4)];   
}
int HipExec_ChannelExtract_U8_U32_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos0,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

/**********************************
    ChannelExtract_U8_U32_Pos1
*****
*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    // unsigned char
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x*2;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *4 ;

    for(int i =0;i<4;i++)
        pDstImage[dstIdx+i]  = pSrcImage1[src1Idx+(i*4)+1]; 
}
int HipExec_ChannelExtract_U8_U32_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


/**********************************
    ChannelExtract_U8_U32_Pos2
*****
*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    // unsigned char
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *4 ;
    pDstImage[dstIdx]  = pSrcImage1[src1Idx+2]; 
}
int HipExec_ChannelExtract_U8_U32_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos2,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos2: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}



/**********************************
    ChannelExtract_U8_U32_Pos3
*****
*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    // unsigned char
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x*2;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *4 ;

    for(int i =0;i<4;i++)
        pDstImage[dstIdx+i]  = pSrcImage1[src1Idx+(i*4)+3]; 
}
int HipExec_ChannelExtract_U8_U32_Pos3(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U32_Pos3: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}



/**********************************
    ChannelExtract_U8_U24_Pos0

*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *3 ;

    pDstImage[dstIdx]  = pSrcImage1[src1Idx]; 
}
int HipExec_ChannelExtract_U8_U24_Pos0(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos0,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos0: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


/**********************************
    ChannelExtract_U8_U24_Pos1

*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *3 ;

    pDstImage[dstIdx]  = pSrcImage1[src1Idx+1]; 
}
int HipExec_ChannelExtract_U8_U24_Pos1(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos1,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}



/**********************************
    ChannelExtract_U8_U24_Pos2

*****************************/
__global__ void __attribute__((visibility("default")))

Hip_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    unsigned char *pDstImage, unsigned int  dstImageStrideInBytes,
    const unsigned char *pSrcImage1, unsigned int srcImage1StrideInBytes
	)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x*4 >= dstWidth*4) || (y >= dstHeight)) return;
    unsigned int dstIdx =  y*(dstImageStrideInBytes) + x;
    unsigned int src1Idx =  y*(srcImage1StrideInBytes) + x *3 ;

    pDstImage[dstIdx]  = pSrcImage1[src1Idx+2]; 
}
int HipExec_ChannelExtract_U8_U24_Pos2(
    vx_uint32 dstWidth, vx_uint32 dstHeight, 
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
    )
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth+3)>>2,   globalThreads_y = dstHeight;

    // printf("\ndstWidth = %d, dstHeight = %d\ndstImageStrideInBytes = %d, srcImage1StrideInBytes = %d, srcImage2StrideInBytes = %d\n", dstWidth, dstHeight, dstImageStrideInBytes, srcImage1StrideInBytes, srcImage2StrideInBytes);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos2,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage , dstImageStrideInBytes, 
                    (const unsigned char *)pHipSrcImage1, srcImage1StrideInBytes);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);
    printf("\nHipExec_ChannelExtract_U8_U24_Pos1: Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}


// ----------------------------------------------------------------------------
// VxChannelCombine kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxColorConvert kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxFormatConvert kernels for hip backend
// ----------------------------------------------------------------------------


