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
#define PIXELROUNDF32(value)        ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))

// ----------------------------------------------------------------------------
// VxRemap kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *map, unsigned int mapStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    int xSrc = (int)PIXELROUNDF32(map[y * (dstWidth * 2) + (x*2) + 0]);
    int ySrc = (int)PIXELROUNDF32(map[y * (dstWidth * 2) + (x*2) + 1]);
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_Remap_U8_U8_Nearest(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    // Printing Passed Remap Table

    // printf("\n\n--------------------- Passed Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", mapStrideInBytes);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%d,%d  ", map[i].y, map[i].x);
    // }
    // printf("\n\n");

    // Generating New Remap Table
    _vx_coordinates2df_t Remap_remapTable_coordinates2df[dstWidth * dstHeight];
	vx_size Remap_remapTableStrideY_size = dstWidth * 8;

    for (int i = 0; i < dstHeight; i ++) {
		for (int j = 0; j < dstWidth; j++) {
			if ((j < srcWidth) && (i < srcHeight)) {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = j;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = i;
			}
			else {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = 0;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = 0;
			}
		}
	}

    // Printing Generated Remap Table

    // printf("\n\n--------------------- Generated Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", (vx_uint32)Remap_remapTableStrideY_size);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%0.1f,%0.1f  ", Remap_remapTable_coordinates2df[i].y, Remap_remapTable_coordinates2df[i].x);
    // }
    // printf("\n\n");

    float *remapTable_float = (float*) Remap_remapTable_coordinates2df;

    /*printf("\n\n");
    for (int i = 0; i < (dstWidth * dstHeight * 2); i+=2)
    {
        printf("%0.1f,%0.1f  ", remapTable_float[i+1], remapTable_float[i]);
    }
    printf("\n\n");*/

    vx_uint32 bufferSize = dstWidth * dstHeight * 64;
    vx_uint8 *hipRemapTable_float;
    hipMalloc(&hipRemapTable_float, bufferSize);
    hipMemcpy(hipRemapTable_float, remapTable_float, bufferSize, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Nearest,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)hipRemapTable_float, mapStrideInBytes);
    hipFree(&hipRemapTable_float);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *map, unsigned int mapStrideInBytes
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    float xSrcFloat = map[y * (dstWidth * 2) + (x*2) + 0];
    float ySrcFloat = map[y * (dstWidth * 2) + (x*2) + 1];
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]
        );
    }
}
int HipExec_Remap_U8_U8_Bilinear(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    // Printing Passed Remap Table

    // printf("\n\n--------------------- Passed Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", mapStrideInBytes);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%d,%d  ", map[i].y, map[i].x);
    // }
    // printf("\n\n");

    // Generating New Remap Table

    _vx_coordinates2df_t Remap_remapTable_coordinates2df[dstWidth * dstHeight];
	vx_size Remap_remapTableStrideY_size = dstWidth * 8;

    for (int i = 0; i < dstHeight; i ++) {
		for (int j = 0; j < dstWidth; j++) {
			if ((j < srcWidth) && (i < srcHeight)) {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = j;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = i;
			}
			else {
				Remap_remapTable_coordinates2df[i*dstWidth + j].x = 0;
				Remap_remapTable_coordinates2df[i*dstWidth + j].y = 0;
			}
		}
	}

    // Printing Generated Remap Table

    // printf("\n\n--------------------- Generated Remap Table ---------------------\n");
    // printf("\nmapStrideInBytes = %d", (vx_uint32)Remap_remapTableStrideY_size);
    // printf("\n\n");
    // for (int i = 0; i < (dstWidth * dstHeight); i++)
    // {
    //     printf("%0.1f,%0.1f  ", Remap_remapTable_coordinates2df[i].y, Remap_remapTable_coordinates2df[i].x);
    // }
    // printf("\n\n");

    float *remapTable_float = (float*) Remap_remapTable_coordinates2df;

    /*printf("\n\n");
    for (int i = 0; i < (dstWidth * dstHeight * 2); i+=2)
    {
        printf("%0.1f,%0.1f  ", remapTable_float[i+1], remapTable_float[i]);
    }
    printf("\n\n");*/

    vx_uint32 bufferSize = dstWidth * dstHeight * 64;
    vx_uint8 *hipRemapTable_float;
    hipMalloc(&hipRemapTable_float, bufferSize);
    hipMemcpy(hipRemapTable_float, remapTable_float, bufferSize, hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(Hip_Remap_U8_U8_Bilinear,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)hipRemapTable_float, mapStrideInBytes);
    hipFree(&hipRemapTable_float);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpAffine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *affineMatrix
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    int xSrc = (int)PIXELROUNDF32((affineMatrix[0] * x) + (affineMatrix[2] * y) + affineMatrix[4]);
    int ySrc = (int)PIXELROUNDF32((affineMatrix[1] * x) + (affineMatrix[3] * y) + affineMatrix[5]);
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_WarpAffine_U8_U8_Nearest(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;
    
    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)affineMatrix);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest_Constant(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *affineMatrix,
    const unsigned char border
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    int xSrc = (int)PIXELROUNDF32((affineMatrix[0] * x) + (affineMatrix[2] * y) + affineMatrix[4]);
    int ySrc = (int)PIXELROUNDF32((affineMatrix[1] * x) + (affineMatrix[3] * y) + affineMatrix[5]);
    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = border;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_WarpAffine_U8_U8_Nearest_Constant(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix,
    vx_uint8 border
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest_Constant,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)affineMatrix,
                       border);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *affineMatrix
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    float xSrcFloat = ((affineMatrix[0] * x) + (affineMatrix[2] * y) + affineMatrix[4]);
    float ySrcFloat = ((affineMatrix[1] * x) + (affineMatrix[3] * y) + affineMatrix[5]);
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]);
    }
}
int HipExec_WarpAffine_U8_U8_Bilinear(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)affineMatrix);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear_Constant(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *affineMatrix,
    const unsigned char border
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))
        return;
    float xSrcFloat = ((affineMatrix[0] * x) + (affineMatrix[2] * y) + affineMatrix[4]);
    float ySrcFloat = ((affineMatrix[1] * x) + (affineMatrix[3] * y) + affineMatrix[5]);
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = border;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]));
    }
}
int HipExec_WarpAffine_U8_U8_Bilinear_Constant(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix,
    vx_uint8 border
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear_Constant,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)affineMatrix,
                       border);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpPerspective kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *perspectiveMatrix
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    float z = (perspectiveMatrix[2] * x) + (perspectiveMatrix[5] * y) + perspectiveMatrix[8];
    int xSrc = (int)PIXELROUNDF32(((perspectiveMatrix[0] * x) + (perspectiveMatrix[3] * y) + perspectiveMatrix[6]) / z);
    int ySrc = (int)PIXELROUNDF32(((perspectiveMatrix[1] * x) + (perspectiveMatrix[4] * y) + perspectiveMatrix[7]) / z);

    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_WarpPerspective_U8_U8_Nearest(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)perspectiveMatrix);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest_Constant(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *perspectiveMatrix,
    const unsigned char border
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    float z = (perspectiveMatrix[2] * x) + (perspectiveMatrix[5] * y) + perspectiveMatrix[8];
    int xSrc = (int)PIXELROUNDF32(((perspectiveMatrix[0] * x) + (perspectiveMatrix[3] * y) + perspectiveMatrix[6]) / z);
    int ySrc = (int)PIXELROUNDF32(((perspectiveMatrix[1] * x) + (perspectiveMatrix[4] * y) + perspectiveMatrix[7]) / z);

    unsigned int dstIdx = y * (dstImageStrideInBytes) + x;
    if ((xSrc < 0) || (xSrc >= srcWidth) || (ySrc < 0) || (ySrc >= srcHeight)) {
        pDstImage[dstIdx] = border;
    }
    else {
        unsigned int srcIdx = ySrc * (srcImageStrideInBytes) + xSrc;
        pDstImage[dstIdx] = pSrcImage[srcIdx];
    }
}
int HipExec_WarpPerspective_U8_U8_Nearest_Constant(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrix,
    vx_uint8 border
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest_Constant,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)perspectiveMatrix,
                       border);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *perspectiveMatrix
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    float z = (perspectiveMatrix[2] * x) + (perspectiveMatrix[5] * y) + perspectiveMatrix[8];
    float xSrcFloat = (((perspectiveMatrix[0] * x) + (perspectiveMatrix[3] * y) + perspectiveMatrix[6]) / z);
    float ySrcFloat = (((perspectiveMatrix[1] * x) + (perspectiveMatrix[4] * y) + perspectiveMatrix[7]) / z);
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = 0;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]);
    }
}
int HipExec_WarpPerspective_U8_U8_Bilinear(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)perspectiveMatrix);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear_Constant(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float *perspectiveMatrix,
    const unsigned char border
    ) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight))    return;
    float z = (perspectiveMatrix[2] * x) + (perspectiveMatrix[5] * y) + perspectiveMatrix[8];
    float xSrcFloat = (((perspectiveMatrix[0] * x) + (perspectiveMatrix[3] * y) + perspectiveMatrix[6]) / z);
    float ySrcFloat = (((perspectiveMatrix[1] * x) + (perspectiveMatrix[4] * y) + perspectiveMatrix[7]) / z);
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((xSrcLower < 0) || (ySrcLower < 0) || (xSrcLower >= srcWidth) || (ySrcLower >= srcHeight)) {
        pDstImage[dstIdx] = border;
    }
    else {
        float s = xSrcFloat - xSrcLower;
        float t = ySrcFloat - ySrcLower;
        int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
        int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
        pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]);
    }
}
int HipExec_WarpPerspective_U8_U8_Bilinear_Constant(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrix,
    vx_uint8 border
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth, globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear_Constant,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                       dim3(localThreads_x, localThreads_y),
                       0, stream, dstWidth, dstHeight,
                       (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                       srcWidth, srcHeight,
                       (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                       (const float *)perspectiveMatrix,
                       border);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxScaleImage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Nearest(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float widthRatio, const float heightRatio,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float widthParam, const float heightParam
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    // int xSrc = (int)PIXELROUNDF32(((x + 0.5) * widthRatio) - 0.5);
    // int ySrc = (int)PIXELROUNDF32(((y + 0.5) * heightRatio) - 0.5);
    int xSrc = (int)PIXELROUNDF32(x * widthRatio + widthParam);
    int ySrc = (int)PIXELROUNDF32(y * heightRatio + heightParam);
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrc*(srcImageStrideInBytes) + xSrc;
    pDstImage[dstIdx] = pSrcImage[srcIdx];
}
int HipExec_ScaleImage_U8_U8_Nearest(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;
    float widthRatio = srcWidthFloat / dstWidthFloat;
    float heightRatio = srcHeightFloat / dstHeightFloat;
    float widthParam = 0.5 * widthRatio - 0.5;
    float heightParam = 0.5 * heightRatio - 0.5;

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Nearest,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    widthRatio, heightRatio,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    widthParam, heightParam);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const float widthRatio, const float heightRatio,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float widthParam, const float heightParam
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    // float xSrcFloat = ((x + 0.5) * (srcWidth/dstWidth)) - 0.5;
    // float ySrcFloat = ((y + 0.5) * (srcHeight/dstHeight)) - 0.5;
    float xSrcFloat = x * widthRatio + widthParam;
    float ySrcFloat = y * heightRatio + heightParam;
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    float s = xSrcFloat - xSrcLower;
    float t = ySrcFloat - ySrcLower;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdxTopLeft, srcIdxTopRight, srcIdxBottomLeft, srcIdxBottomRight;
    srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
    srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
    if (ySrcLower + 1 < srcHeight) {
        srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
        srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
    }
    else {
        srcIdxBottomLeft =  (ySrcLower) * (srcImageStrideInBytes) + xSrcLower;
        srcIdxBottomRight =  (ySrcLower) * (srcImageStrideInBytes) + (xSrcLower + 1);
    }
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32(
        (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
        (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
        (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
        (s) * (t) * pSrcImage[srcIdxBottomRight]));
}
int HipExec_ScaleImage_U8_U8_Bilinear(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;
    float widthRatio = srcWidthFloat / dstWidthFloat;
    float heightRatio = srcHeightFloat / dstHeightFloat;
    float widthParam = 0.5 * widthRatio - 0.5;
    float heightParam = 0.5 * heightRatio - 0.5;

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    widthRatio, heightRatio,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    widthParam, heightParam);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Replicate(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const float widthRatio, const float heightRatio,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float widthParam, const float heightParam
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    // float xSrcFloat = ((x + 0.5) * (srcWidth/dstWidth)) - 0.5;
    // float ySrcFloat = ((y + 0.5) * (srcHeight/dstHeight)) - 0.5;
    float xSrcFloat = x * widthRatio + widthParam;
    float ySrcFloat = y * heightRatio + heightParam;
    int xSrcLower = (int)xSrcFloat;
    int ySrcLower = (int)ySrcFloat;
    float s = xSrcFloat - xSrcLower;
    float t = ySrcFloat - ySrcLower;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
    int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
    int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
    int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
    if (ySrcLower < 0) {
        srcIdxTopLeft += srcImageStrideInBytes;
        srcIdxTopRight += srcImageStrideInBytes;
    }
    if (ySrcLower + 1 >= srcHeight) {
        srcIdxBottomLeft -= srcImageStrideInBytes;
        srcIdxBottomRight -= srcImageStrideInBytes;
    }
    if (xSrcLower < 0) {
        srcIdxTopLeft += 1;
        srcIdxBottomLeft += 1;
    }
    if (xSrcLower + 1 >= srcWidth) {
        srcIdxTopRight -= 1;
        srcIdxBottomRight -= 1;
    }
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32(
      (1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + 
      (s) * (1-t) * pSrcImage[srcIdxTopRight] + 
      (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + 
      (s) * (t) * pSrcImage[srcIdxBottomRight]));
}
int HipExec_ScaleImage_U8_U8_Bilinear_Replicate(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;
    float widthRatio = srcWidthFloat / dstWidthFloat;
    float heightRatio = srcHeightFloat / dstHeightFloat;
    float widthParam = 0.5 * widthRatio - 0.5;
    float heightParam = 0.5 * heightRatio - 0.5;

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Replicate,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    widthRatio, heightRatio,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    widthParam, heightParam);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Constant(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const float widthRatio, const float heightRatio,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
    const float widthParam, const float heightParam,
    const unsigned char border
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    // float xSrcFloat = ((x + 0.5) * (srcWidth/dstWidth)) - 0.5;
    // float ySrcFloat = ((y + 0.5) * (srcHeight/dstHeight)) - 0.5;
    float xSrcFloat = x * widthRatio + widthParam;
    float ySrcFloat = y * heightRatio + heightParam;
    int xSrcLower, ySrcLower;
    if (xSrcFloat >= 0)
        xSrcLower = (int)xSrcFloat;
    else
        xSrcLower = (int)xSrcFloat - 1;
    if (ySrcFloat >= 0)
        ySrcLower = (int)ySrcFloat;
    else
        ySrcLower = (int)ySrcFloat - 1;
    float s = xSrcFloat - xSrcLower;
    float t = ySrcFloat - ySrcLower;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdxTopLeft =  ySrcLower * (srcImageStrideInBytes) + xSrcLower;
    int srcIdxTopRight =  ySrcLower * (srcImageStrideInBytes) + (xSrcLower + 1);
    int srcIdxBottomLeft =  (ySrcLower + 1) * (srcImageStrideInBytes) + xSrcLower;
    int srcIdxBottomRight =  (ySrcLower + 1) * (srcImageStrideInBytes) + (xSrcLower + 1);
    if (ySrcFloat >= 0)
        if (ySrcFloat + 1 < srcHeight)
            if (xSrcFloat >= 0)
                if (xSrcFloat + 1 < srcWidth)
                    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + (s) * (1-t) * pSrcImage[srcIdxTopRight] + (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + (s) * (t) * pSrcImage[srcIdxBottomRight]));
                else
                    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + (s) * (1-t) * border + (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + (s) * (t) * border));
            else
                pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * border + (s) * (1-t) * pSrcImage[srcIdxTopRight] + (1-s) * (t) * border + (s) * (t) * pSrcImage[srcIdxBottomRight]));
        else
            if (xSrcFloat >= 0)
                if (xSrcFloat + 1 < srcWidth)
                    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + (s) * (1-t) * pSrcImage[srcIdxTopRight] + (1-s) * (t) * border + (s) * (t) * border));
                else
                    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * pSrcImage[srcIdxTopLeft] + (s) * (1-t) * border + (1-s) * (t) * border + (s) * (t) * border));
            else
                pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * border + (s) * (1-t) * pSrcImage[srcIdxTopRight] + (1-s) * (t) * border + (s) * (t) * border));
    else
        if (xSrcFloat >= 0)
            if (xSrcFloat + 1 < srcWidth)
                pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * border + (s) * (1-t) * border + (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + (s) * (t) * pSrcImage[srcIdxBottomRight]));
            else
                pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * border + (s) * (1-t) * border + (1-s) * (t) * pSrcImage[srcIdxBottomLeft] + (s) * (t) * border));
        else
            pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8(PIXELROUNDF32((1-s) * (1-t) * border + (s) * (1-t) * border + (1-s) * (t) * border + (s) * (t) * pSrcImage[srcIdxBottomRight]));
}
int HipExec_ScaleImage_U8_U8_Bilinear_Constant(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix,
    const vx_uint8 border
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;
    float widthRatio = srcWidthFloat / dstWidthFloat;
    float heightRatio = srcHeightFloat / dstHeightFloat;
    float widthParam = 0.5 * widthRatio - 0.5;
    float heightParam = 0.5 * heightRatio - 0.5;

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Constant,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    widthRatio, heightRatio,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    widthParam, heightParam,
                    border);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area(
    const float dstWidth, const float dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    const float srcWidth, const float srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int xSrcLow = (int)(((float)x - (srcWidth / dstWidth)) - 0.5);
    int ySrcLow = (int)(((float)y - (srcHeight / dstHeight)) - 0.5);
    int xSrcHigh = (int)(((float)(x+1) - (srcWidth / dstWidth)) - 0.5);
    int ySrcHigh = (int)(((float)(y+1) - (srcHeight / dstHeight)) - 0.5);
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int srcIdx =  ySrcLow * (srcImageStrideInBytes) + xSrcLow;
    int srcIdxRow = srcIdx;
    int srcIdxCol = srcIdxRow;
    int sum = 0, count = 0;
    for (int y = ySrcLow; y < ySrcHigh; y++) {
        for (int x = xSrcLow; x < xSrcHigh; x++) {
            sum += pSrcImage[srcIdxCol];
            srcIdxCol += 1;
            count += 1;
        }
        srcIdxRow += srcImageStrideInBytes;
        srcIdxCol = srcIdxRow;
    }
    pDstImage[dstIdx] = (unsigned char)PIXELSATURATEU8((float)sum / (float)count);
}
int HipExec_ScaleImage_U8_U8_Area(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const ago_scale_matrix_t *matrix
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;
    float dstWidthFloat, dstHeightFloat, srcWidthFloat, srcHeightFloat;
    dstWidthFloat = (float)dstWidth;
    dstHeightFloat = (float)dstHeight;
    srcWidthFloat = (float)srcWidth;
    srcHeightFloat = (float)srcHeight;

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidthFloat, dstHeightFloat,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    srcWidthFloat, srcHeightFloat,
                    (unsigned char *)pHipSrcImage, srcImageStrideInBytes);
    return VX_SUCCESS;
}

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