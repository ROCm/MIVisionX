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


__device__ int FastAtan2_Canny(short int Gx, short int Gy) {
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

__device__ __forceinline__ int isCorner(int mask) {
	int cornerMask = 0x1FF;									// Nine 1's in the LSB
	if (mask) {
		mask = mask | (mask << 16);
		for (int i = 0; i < 16; i++) {
			if ((mask & cornerMask) == cornerMask)
				return 1;
			mask >>= 1;
		}
	}
	return 0;
}

__device__ __forceinline__ int isCornerPlus(short candidate, short * boundary, short t) {
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;
	candidate += t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++) {
		if (boundary[i] > candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}
	return isCorner(mask);
}

__device__ __forceinline__ int isCornerMinus(short candidate, short * boundary, short t) {
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;
	candidate -= t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++) {
		if (boundary[i] < candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}
	return isCorner(mask);
}

typedef struct {
	vx_float32 GxGx;
	vx_float32 GxGy;
	vx_float32 GyGy;
} ago_harris_Gxy_t;

// ----------------------------------------------------------------------------
// VxFastCorners kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_NoSupression(
    vx_uint32 capacityOfDstCorner, vx_keypoint_t *pDstCorner, vx_uint32 *cornercount,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes, vx_float32 threshold
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y + 3;
    if ((x >= (srcWidth - 3)) || (y >= (srcHeight - 3))) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + x;

    int offsets[16] = { 0 };
    offsets[0] = srcIdx + (-3 * srcImageStrideInBytes);
    offsets[15] = offsets[0] - 1;
    offsets[1] = offsets[0] + 1;
    offsets[2] = srcIdx - (srcImageStrideInBytes << 1) + 2;
    offsets[14] = offsets[2] - 4;
    offsets[3] = srcIdx - srcImageStrideInBytes + 3;
    offsets[13] = offsets[3] - 6;
    offsets[4] = srcIdx + 3;
    offsets[12] = srcIdx - 3;
    offsets[5] = srcIdx + srcImageStrideInBytes + 3;
    offsets[11] = offsets[5] - 6;
    offsets[6] = srcIdx + (srcImageStrideInBytes << 1) + 2;
    offsets[10] = offsets[6] - 4;
    offsets[7] = srcIdx + (3 * srcImageStrideInBytes) + 1;
    offsets[8] = offsets[7] - 1;
    offsets[9] = offsets[8] - 1;

    int mask_max=0, mask_min=0;
	// Early exit conditions
	if ((abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[0]]) < (short)threshold) && (abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[8]]) < (short)threshold))					// Pixels 1 and 9 within (short) threshold of the candidate
		return;
	if ((abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[4]]) < (short)threshold) && (abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[12]]) < (short)threshold))				// Pixels 5 and 13 within (short) threshold of the candidate
		return;

	// Check for I_p + t
	short cand = (short)pSrcImage[srcIdx] + (short)threshold;

	mask_max = (CHECKMAX(pSrcImage[offsets[0]], cand)) | (CHECKMAX(pSrcImage[offsets[1]], cand) << 1) | (CHECKMAX(pSrcImage[offsets[2]], cand) << 2) | (CHECKMAX(pSrcImage[offsets[3]], cand) << 3) |
				(CHECKMAX(pSrcImage[offsets[4]], cand) << 4) | (CHECKMAX(pSrcImage[offsets[5]], cand) << 5) | (CHECKMAX(pSrcImage[offsets[6]], cand) << 6) | (CHECKMAX(pSrcImage[offsets[7]], cand) << 7) |
				(CHECKMAX(pSrcImage[offsets[8]], cand) << 8) | (CHECKMAX(pSrcImage[offsets[9]], cand) << 9) | (CHECKMAX(pSrcImage[offsets[10]], cand) << 10) | (CHECKMAX(pSrcImage[offsets[11]], cand) << 11) |
				(CHECKMAX(pSrcImage[offsets[12]], cand) << 12) | (CHECKMAX(pSrcImage[offsets[13]], cand) << 13) | (CHECKMAX(pSrcImage[offsets[14]], cand) << 14) | (CHECKMAX(pSrcImage[offsets[15]], cand) << 15);

	// Check for I_p - t
	cand = (short)pSrcImage[srcIdx] - (short)threshold;

	mask_min = (CHECKMIN(pSrcImage[offsets[0]], cand)) | (CHECKMIN(pSrcImage[offsets[1]], cand) << 1) | (CHECKMIN(pSrcImage[offsets[2]], cand) << 2) | (CHECKMIN(pSrcImage[offsets[3]], cand) << 3) |
				(CHECKMIN(pSrcImage[offsets[4]], cand) << 4) | (CHECKMIN(pSrcImage[offsets[5]], cand) << 5) | (CHECKMIN(pSrcImage[offsets[6]], cand) << 6) | (CHECKMIN(pSrcImage[offsets[7]], cand) << 7) |
				(CHECKMIN(pSrcImage[offsets[8]], cand) << 8) | (CHECKMIN(pSrcImage[offsets[9]], cand) << 9) | (CHECKMIN(pSrcImage[offsets[10]], cand) << 10) | (CHECKMIN(pSrcImage[offsets[11]], cand) << 11) |
				(CHECKMIN(pSrcImage[offsets[12]], cand) << 12) | (CHECKMIN(pSrcImage[offsets[13]], cand) << 13) | (CHECKMIN(pSrcImage[offsets[14]], cand) << 14) | (CHECKMIN(pSrcImage[offsets[15]], cand) << 15);
	
	int cornerMask = 511, isCorner = 0;
	if (mask_max || mask_min) {
		mask_max = mask_max | (mask_max << 16);
		mask_min = mask_min | (mask_min << 16);
		for (int i = 0; i < 16; i++) {
			if (((mask_max & cornerMask) == cornerMask) || ((mask_min & cornerMask) == cornerMask)) {
				isCorner = 1;
				break;
			}
			mask_max >>= 1;
			mask_min >>= 1;
		}
	}
	
	if(isCorner) {
		unsigned int old_idx = atomicAdd(cornercount, 1);
		if (old_idx < capacityOfDstCorner) {
			pDstCorner[old_idx].y = y;
			pDstCorner[old_idx].x = x;
			pDstCorner[old_idx].strength = threshold;			// Undefined as per the 1.0.1 spec
			pDstCorner[old_idx].scale = 0;
			pDstCorner[old_idx].orientation = 0;
			pDstCorner[old_idx].error = 0;
			pDstCorner[old_idx].tracking_status = 1;
		}
	}
}
int HipExec_FastCorners_XY_U8_NoSupression(
		hipStream_t stream,
		vx_uint32  capacityOfDstCorner,
    	vx_keypoint_t   pHipDstCorner[],
		vx_uint32  *pHipDstCornerCount,
		vx_uint32  srcWidth, vx_uint32 srcHeight,
		vx_uint8   *pHipSrcImage,
		vx_uint32   srcImageStrideInBytes,
		vx_float32  strength_threshold
	) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = srcWidth-6,   globalThreads_y = srcHeight-6;
    
	vx_uint32 *cornerCount;
	hipMalloc(&cornerCount, sizeof(vx_uint32));
	hipMemcpy(cornerCount, pHipDstCornerCount, sizeof(vx_uint32), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_NoSupression,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
                    srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold
					);

	hipMemcpyDtoH(pHipDstCornerCount, cornerCount, sizeof(vx_uint32));
	hipFree(cornerCount);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_Supression(
    vx_uint32 capacityOfDstCorner, vx_keypoint_t *pDstCorner, vx_uint32 *cornercount,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
	vx_float32 threshold, unsigned char *pScratch
	) {
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y + 3;
    if ((x >= (srcWidth - 3)) || (y >= (srcHeight - 3))) return;
    unsigned int srcIdx =  y*(srcImageStrideInBytes) + x;

	int offsets[16] = { 0 };
    offsets[0] = srcIdx + (-3 * srcImageStrideInBytes);
    offsets[15] = offsets[0] - 1;
    offsets[1] = offsets[0] + 1;
    offsets[2] = srcIdx - (srcImageStrideInBytes << 1) + 2;
    offsets[14] = offsets[2] - 4;
    offsets[3] = srcIdx - srcImageStrideInBytes + 3;
    offsets[13] = offsets[3] - 6;
    offsets[4] = srcIdx + 3;
    offsets[12] = srcIdx - 3;
    offsets[5] = srcIdx + srcImageStrideInBytes + 3;
    offsets[11] = offsets[5] - 6;
    offsets[6] = srcIdx + (srcImageStrideInBytes << 1) + 2;
    offsets[10] = offsets[6] - 4;
    offsets[7] = srcIdx + (3 * srcImageStrideInBytes) + 1;
    offsets[8] = offsets[7] - 1;
    offsets[9] = offsets[8] - 1;

	short strength = 0;

	// Early exit conditions
	if ((abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[0]]) < threshold) && (abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[8]]) < threshold))					// Pixels 1 and 9 within t of the candidate
		return;
	if ((abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[4]]) < threshold) && (abs((short)pSrcImage[srcIdx] - (short)pSrcImage[offsets[12]]) < threshold))				// Pixels 5 and 13 within t of the candidate
		return;

	// Get boundary
	short boundary[16];
	boundary[0] = (short)pSrcImage[offsets[0]];
	boundary[1] = (short)pSrcImage[offsets[1]];
	boundary[2] = (short)pSrcImage[offsets[2]];
	boundary[3] = (short)pSrcImage[offsets[3]];
	boundary[4] = (short)pSrcImage[offsets[4]];
	boundary[5] = (short)pSrcImage[offsets[5]];
	boundary[6] = (short)pSrcImage[offsets[6]];
	boundary[7] = (short)pSrcImage[offsets[7]];
	boundary[8] = (short)pSrcImage[offsets[8]];
	boundary[9] = (short)pSrcImage[offsets[9]];
	boundary[10] = (short)pSrcImage[offsets[10]];
	boundary[11] = (short)pSrcImage[offsets[11]];
	boundary[12] = (short)pSrcImage[offsets[12]];
	boundary[13] = (short)pSrcImage[offsets[13]];
	boundary[14] = (short)pSrcImage[offsets[14]];
	boundary[15] = (short)pSrcImage[offsets[15]];

	// Check for I_p + t
	short cand = (short)pSrcImage[srcIdx] + threshold;
	int maskP = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++) {
		if (boundary[i] > cand)
			maskP |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_pos = 0;
	cand = pSrcImage[srcIdx];
	if (isCorner(maskP)) {
		short thresh_upper = 255;
		short thresh_lower = threshold;
		
		while (thresh_upper - thresh_lower > 1)	{					// Binary search 
			strength_pos = (thresh_upper + thresh_lower) >> 1;
			if (isCornerPlus(cand, boundary, strength_pos))  
				thresh_lower = strength_pos;
			else
				thresh_upper = strength_pos;
		}
		strength_pos = thresh_lower;
	}


	// Check for I_p - t
	cand = (short)pSrcImage[srcIdx] - threshold;;
	int maskN = 0;
	iterMask = 1;
	for (int i = 0; i < 16; i++) {
		if (boundary[i] < cand)
			maskN |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_neg = 0;
	cand = pSrcImage[srcIdx];
	if (isCorner(maskN)) {
		short thresh_upper = 255;
		short thresh_lower = threshold;
		
		while (thresh_upper - thresh_lower > 1) {						// Binary search
			strength_neg = (thresh_upper + thresh_lower) >> 1;
			if (isCornerMinus(cand, boundary, strength_neg))
				thresh_lower = strength_neg;
			else
				thresh_upper = strength_neg;
		}
		strength_neg = thresh_lower;
	}

	if (maskP || maskN) {
		strength = strength_pos > strength_neg ? strength_pos : strength_neg;
		pScratch[y * srcWidth + x] = (vx_uint8)strength;
	}
}

__global__ void __attribute__((visibility("default")))
Hip_NonMaximumSupression_3x3(
    vx_uint32 capacityOfDstCorner, vx_keypoint_t *pDstCorner, vx_uint32 *cornercount,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes,
	vx_float32 threshold, unsigned char *pScratch
	) {
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y + 3;
    if ((x >= (srcWidth - 3)) || (y >= (srcHeight - 3))) return;
    unsigned int srcIdx =  y*(srcWidth) + x;	
	unsigned int srcIdxTopRow = srcIdx - srcWidth;
    unsigned int srcIdxBottomRow = srcIdx + srcWidth;

	if(pScratch[srcIdx] == 0 || pScratch[srcIdx] < pScratch [srcIdxTopRow - 1] || pScratch[srcIdx] <= pScratch[srcIdxBottomRow - 1]
		|| pScratch[srcIdx] < pScratch [srcIdxTopRow] || pScratch[srcIdx] <= pScratch[srcIdxBottomRow]
		|| pScratch[srcIdx] < pScratch [srcIdxTopRow + 1] || pScratch[srcIdx] <= pScratch[srcIdxBottomRow + 1]
		|| pScratch[srcIdx] < pScratch[srcIdx - 1] || pScratch[srcIdx] <= pScratch[srcIdx + 1])	    
			return;

	unsigned int old_idx = atomicAdd(cornercount, 1);
	if (old_idx < capacityOfDstCorner) {
		pDstCorner[old_idx].y = y;
		pDstCorner[old_idx].x = x;
		pDstCorner[old_idx].strength = threshold;			// Undefined as per the 1.0.1 spec
		pDstCorner[old_idx].scale = 0;
		pDstCorner[old_idx].orientation = 0;
		pDstCorner[old_idx].error = 0;
		pDstCorner[old_idx].tracking_status = 1;
	}
}

int HipExec_FastCorners_XY_U8_Supression(
	hipStream_t stream,
	vx_uint32  capacityOfDstCorner,
	vx_keypoint_t   pHipDstCorner[],
	vx_uint32  *pHipDstCornerCount,
	vx_uint32  srcWidth, vx_uint32 srcHeight,
	vx_uint8   *pHipSrcImage,
	vx_uint32   srcImageStrideInBytes,
	vx_float32  strength_threshold,
	vx_uint8   *pHipScratch
	) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = srcWidth,   globalThreads_y = srcHeight;

	vx_uint32 *cornerCount;
	hipMalloc(&cornerCount, sizeof(vx_uint32));
	hipMemcpy(cornerCount, pHipDstCornerCount, sizeof(vx_uint32), hipMemcpyHostToDevice);

	vx_uint8 * Scratch;
	hipMalloc(&Scratch, sizeof(vx_uint8) * srcWidth * srcHeight);
	hipMemset(Scratch, 0, sizeof(vx_uint8) * srcWidth * srcHeight);

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_Supression,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
                    srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold, (unsigned char *)Scratch);

	hipLaunchKernelGGL(Hip_NonMaximumSupression_3x3,
					dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
					dim3(localThreads_x, localThreads_y),
					0, stream, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
					srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold, (unsigned char *)Scratch);
	
	hipMemcpyDtoH(pHipScratch, Scratch, sizeof(vx_uint8) * srcWidth * srcHeight);
	hipMemcpyDtoH(pHipDstCornerCount, cornerCount, sizeof(vx_uint32));
	hipFree(cornerCount);
	hipFree(Scratch);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxHarrisCorners kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_3x3(
    unsigned int  dstWidth, unsigned int  dstHeight,
    float * pDstGxy_,unsigned int  dstGxyStrideInBytes,
    const unsigned char  * pSrcImage ,unsigned int srcImageStrideInBytes,
    float * gx, float *gy
	) {
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
	if ((x >dstWidth) || (x<0)|| (y >= dstHeight) || y<=0)	return;
	unsigned int dstIdx = y * (dstGxyStrideInBytes /sizeof(ago_harris_Gxy_t)) + x;
	unsigned int srcIdx = y * (srcImageStrideInBytes) + x;
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)( pDstGxy_ );
	float div_factor = 1; // 4.0f * 255;

	int srcIdxTopRow = srcIdx - srcImageStrideInBytes;
	int srcIdxBottomRow = srcIdx + srcImageStrideInBytes;
	float sum_x = 0;
	sum_x += (gx[4] * (float)*(pSrcImage + srcIdx) + gx[1] * (float)*(pSrcImage + srcIdxTopRow) + gx[7] * (float)*(pSrcImage + srcIdxBottomRow));
	sum_x += (gx[3] * (float)*(pSrcImage + srcIdx - 1) + gx[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + gx[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
	sum_x += (gx[5] * (float)*(pSrcImage + srcIdx + 1) + gx[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + gx[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));
	float sum_y = 0;
	sum_y += (gy[4] * (float)*(pSrcImage + srcIdx) + gy[1] * (float)*(pSrcImage + srcIdxTopRow) + gy[7] * (float)*(pSrcImage + srcIdxBottomRow));
	sum_y += (gy[3] * (float)*(pSrcImage + srcIdx - 1) + gy[0] * (float)*(pSrcImage + srcIdxTopRow - 1) + gy[6] * (float)*(pSrcImage + srcIdxBottomRow - 1));
	sum_y += (gy[5] * (float)*(pSrcImage + srcIdx + 1) + gy[2] * (float)*(pSrcImage + srcIdxTopRow + 1) + gy[8] * (float)*(pSrcImage + srcIdxBottomRow + 1));

	pDstGxy[dstIdx].GxGx = sum_x*sum_x;
	pDstGxy[dstIdx].GxGy = sum_x*sum_y;
	pDstGxy[dstIdx].GyGy = sum_y*sum_y;
}
int HipExec_HarrisSobel_HG3_U8_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth),   globalThreads_y = dstHeight;

    float gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    float gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    float *hipGx, *hipGy;
    hipMalloc(&hipGx, 144);
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstGxy_ , dstGxyStrideInBytes,
                    (const unsigned char *)pSrcImage, srcImageStrideInBytes,
                    (float *)hipGx, (float *)hipGy);
                    
/* Printing Outputs for verification */
    /*ago_harris_Gxy_t *DstGxy;
    DstGxy = (ago_harris_Gxy_t *)malloc(dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    hipError_t status = hipMemcpyDtoH(DstGxy, pDstGxy_, dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    if (status != hipSuccess)
      //printf("Copy mem dev to host failed\n");
    for (int j = 1; j < dstHeight-1 ; j++) {
      for (int i = 0; i < dstWidth; i++) {
        int idx = j*(dstGxyStrideInBytes/sizeof(ago_harris_Gxy_t)) + i;
        //printf("<row, col>: <%d,%d>", j,i);
        //printf("The GXGX : %f \t and \tGYGY : %f \t and \t GXGY : %f\n", DstGxy[idx].GxGx, DstGxy[idx].GyGy, DstGxy[idx].GxGy);
      }
    }
    hipFree(DstGxy);*/
    return VX_SUCCESS;
}

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
                    
/* Printing Outputs for verification */
    /*ago_harris_Gxy_t *DstGxy;
    DstGxy = (ago_harris_Gxy_t *)malloc(dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    hipError_t status = hipMemcpyDtoH(DstGxy, pDstGxy_, dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    if (status != hipSuccess)
      printf("Copy mem dev to host failed\n");
    for (int j = 2; j < dstHeight-2 ; j++)
    {
      for (int i = 0; i < dstWidth; i++)
      {
        int idx = j*(dstGxyStrideInBytes/sizeof(ago_harris_Gxy_t)) + i;
        printf("<row, col>: <%d,%d>", j,i);
        printf("The GXGX : %f \t and \t GYGY : %f \t and \t GXGY : %f\n", DstGxy[idx].GxGx, DstGxy[idx].GyGy, DstGxy[idx].GxGy);
      }
    }
    hipFree(DstGxy);*/
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
                    
/* Printing Outputs for verification */
    /*ago_harris_Gxy_t *DstGxy;
    DstGxy = (ago_harris_Gxy_t *)malloc(dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    hipError_t status = hipMemcpyDtoH(DstGxy, pDstGxy_, dstWidth * dstHeight * sizeof(ago_harris_Gxy_t));
    if (status != hipSuccess)
      printf("Copy mem dev to host failed\n");
    for (int j = 3; j < dstHeight-3 ; j++)
    {
      for (int i = 0; i < dstWidth; i++)
      {
        int idx = j*(dstGxyStrideInBytes/sizeof(ago_harris_Gxy_t)) + i;
        printf("<row, col>: <%d,%d>", j,i);
        printf("The GXGX : %f \t and \t GYGY : %f \t and \t GXGY : %f\n", DstGxy[idx].GxGx, DstGxy[idx].GyGy, DstGxy[idx].GxGy);
      }
    }
    hipFree(DstGxy);*/
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_3x3(
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

	if ((x >= dstWidth-1) || (x <= 0) || (y >= dstHeight-1 ) || y <= 0) {
		pDstVc[dstIdx] = (float)0;
		return;
	}

	float gx2 = 0, gy2 = 0, gxy2 = 0;
	float traceA =0, detA =0, Mc =0;
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	//Prev Row + Current Row + Next Row sum of gx2, gxy2, gy2
	int srcIdxTopRow1, srcIdxBottomRow1;
	srcIdxTopRow1 = srcIdx - srcGxyStrideInBytes;
	srcIdxBottomRow1 = srcIdx + srcGxyStrideInBytes;
	gx2 =  
	(float)pSrcGxy[srcIdxTopRow1 - 1].GxGx + (float)pSrcGxy[srcIdxTopRow1].GxGx +(float)pSrcGxy[srcIdxTopRow1 + 1].GxGx +
	(float)pSrcGxy[srcIdx-1].GxGx + (float)pSrcGxy[srcIdx].GxGx + (float)pSrcGxy[srcIdx+1].GxGx + 
	(float)pSrcGxy[srcIdxBottomRow1 -1].GxGx + (float)pSrcGxy[srcIdxBottomRow1].GxGx + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGx;

	gxy2 = 
	(float)pSrcGxy[srcIdxTopRow1 - 1].GxGy + (float)pSrcGxy[srcIdxTopRow1].GxGy + (float)pSrcGxy[srcIdxTopRow1 + 1].GxGy + 
	(float)pSrcGxy[srcIdx-1].GxGy + (float)pSrcGxy[srcIdx].GxGy + (float)pSrcGxy[srcIdx+1].GxGy +
	(float)pSrcGxy[srcIdxBottomRow1 -1].GxGy + (float)pSrcGxy[srcIdxBottomRow1].GxGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GxGy ;

	gy2 = 
	(float)pSrcGxy[srcIdxTopRow1 - 1].GyGy + (float)pSrcGxy[srcIdxTopRow1].GyGy + (float)pSrcGxy[srcIdxTopRow1 + 1].GyGy +
	(float)pSrcGxy[srcIdx-1].GyGy + (float)pSrcGxy[srcIdx].GyGy + (float)pSrcGxy[srcIdx+1].GyGy +
	(float)pSrcGxy[srcIdxBottomRow1 -1].GyGy + (float)pSrcGxy[srcIdxBottomRow1].GyGy + (float)pSrcGxy[srcIdxBottomRow1 + 1].GyGy;

	traceA = gx2 + gy2;
	detA = (gx2 * gy2) - (gxy2 * gxy2);
	Mc = detA - (sensitivity * traceA * traceA);
	Mc /= normalization_factor;
	if(Mc > strength_threshold){
		pDstVc[dstIdx] = (float)Mc; 
	}
	else{
		pDstVc[dstIdx] = (float)0;
	}
}
int HipExec_HarrisScore_HVC_HG3_3x3(
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
    vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold,
    vx_float32 normalization_factor
	) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream,
                    dstWidth, dstHeight,
                    (float *)pDstVc , (dstVcStrideInBytes/sizeof(float)),
                    (float *)pSrcGxy_, (srcGxyStrideInBytes/sizeof(ago_harris_Gxy_t)),
                    sensitivity, strength_threshold,normalization_factor );
                    
/* Printing Outputs for verification */
    /*float *pDstVc_;
    pDstVc_ = (float *)malloc(dstWidth * dstHeight * sizeof(float));
    hipError_t status = hipMemcpyDtoH(pDstVc_, pDstVc, dstWidth * dstHeight * sizeof(float));
    if (status != hipSuccess)
      printf("Copy mem dev to host failed\n");
    for (int j = 1; j < dstHeight-1 ; j++)
    {
      for (int i = 1; i < dstWidth-1; i++)
      {
        int idx = j*(dstVcStrideInBytes/sizeof(float)) + i;
        printf("\n <row, col>: <%d,%d>", j,i);
        printf(" \t Mc: <%f>",pDstVc_[idx]);
        // printf("The GXGX : %f \n", pDstVc_[idx]);
      }
    }
    hipFree(pDstVc_);*/
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
                    
/* Printing Outputs for verification */
    /*
	float *pDstVc_;
    pDstVc_ = (float *)malloc(dstWidth * dstHeight * sizeof(float));
    hipError_t status = hipMemcpyDtoH(pDstVc_, pDstVc, dstWidth * dstHeight * sizeof(float));
    if (status != hipSuccess)
      printf("Copy mem dev to host failed\n");
    for (int j = 2; j < dstHeight-2 ; j++)
    {
      for (int i = 2; i < dstWidth-2; i++)
      {
        int idx = j*(dstVcStrideInBytes/sizeof(float)) + i;
        printf("\n <row, col>: <%d,%d>", j,i);
        printf(" \t Mc: <%f>",pDstVc_[idx]);
      }
    }
    hipFree(pDstVc_);
	*/
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
                    
/* Printing Outputs for verification */
    /*
	float *pDstVc_;
    pDstVc_ = (float *)malloc(dstWidth * dstHeight * sizeof(float));
    hipError_t status = hipMemcpyDtoH(pDstVc_, pDstVc, dstWidth * dstHeight * sizeof(float));
    if (status != hipSuccess)
      printf("Copy mem dev to host failed\n");
    for (int j = 3; j < dstHeight-3 ; j++)
    {
      for (int i = 3; i < dstWidth-3; i++)
      {
        int idx = j*(dstVcStrideInBytes/sizeof(float)) + i;
        printf("\n <row, col>: <%d,%d>", j,i);
        printf(" \t Mc: <%f>",pDstVc_[idx]);
      }
    }
    hipFree(pDstVc_);
	*/
    return VX_SUCCESS;
}













// ----------------------------------------------------------------------------
// VxCannyEdgeDetector kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L1NORM(
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
    if ((y >= dstHeight - 1) || (y <= 0)) {
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
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
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
    
	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L1NORM,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned short int *)pHipDstImage, dstImageStrideInBytes,
                    (const unsigned char *)pHipSrcImage, srcImageStrideInBytes,
                    (const short int *)hipGx, (const short int *)hipGy);
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

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L2NORM(
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
    if ((y >= dstHeight - 1) || (y <= 0)) {
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
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    short int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    short int gy[9] = {-1,-2,-1,0,0,0,1,2,1};
    short int *hipGx, *hipGy;
    hipMalloc(&hipGx, 144);
    hipMalloc(&hipGy, 144);
    hipMemcpy(hipGx, gx, 144, hipMemcpyHostToDevice);
    hipMemcpy(hipGy, gy, 144, hipMemcpyHostToDevice);
    
	hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L2NORM,
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

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_3x3(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack,
    const unsigned short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const unsigned short int hyst_lower, const unsigned short int hyst_upper
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1)) {
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
    if (edge > hyst_upper) {
		pDstImage[dstIdx] = (unsigned char)255;
		xyStack[xyStackIdx] = x;
		xyStack[xyStackIdx + 1] = y;
	}
	else if (edge <= hyst_lower) {
		pDstImage[dstIdx] = (unsigned char)0;
		xyStack[xyStackIdx] = 0;
		xyStack[xyStackIdx + 1] = 0;
		}
	else {
      pDstImage[dstIdx] = (unsigned char)127;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_3x3(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_3x3,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    (unsigned short int *) xyStack,
                    (const unsigned short int *)pHipSrcImage, srcImageStrideInBytes,
                    (const unsigned short int)hyst_lower, (const unsigned short int)hyst_upper);
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
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_7x7(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack,
    const unsigned short int *pSrcImage, unsigned int srcImageStrideInBytes,
    const unsigned short int hyst_lower, const unsigned short int hyst_upper
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= dstWidth) || (y >= dstHeight)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1)) {
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
    if (edge > hyst_upper){
      pDstImage[dstIdx] = (unsigned char)255;
      xyStack[xyStackIdx] = x;
      xyStack[xyStackIdx + 1] = y;
	}
	else if (edge <= hyst_lower) {
		pDstImage[dstIdx] = (unsigned char)0;
		xyStack[xyStackIdx] = 0;
      	xyStack[xyStackIdx + 1] = 0;
	}
	else {
      pDstImage[dstIdx] = (unsigned char)127;
      xyStack[xyStackIdx] = 0;
      xyStack[xyStackIdx + 1] = 0;
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_7x7(
    hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_7x7,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    (unsigned short int *) xyStack,
                    (const unsigned short int *)pHipSrcImage, srcImageStrideInBytes,
                    (const unsigned short int)hyst_lower, (const unsigned short int)hyst_upper);
    *pxyStackTop = (vx_uint32)(dstWidth * dstHeight - 1);

    return VX_SUCCESS;
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

__global__ void __attribute__((visibility("default")))
Hip_CannyEdgeTrace_U8_U8XY(
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    unsigned char *pDstImage, unsigned int dstImageStrideInBytes,
    unsigned short int *xyStack
	) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((y <= 0) || (y >= dstHeight - 1) || (x <= 0) || (x >= dstWidth - 1)) return;
    int dstIdx =  y*(dstImageStrideInBytes) + x;
    int xyStackStride = dstWidth * 2;
    int xyStackIdx = (y * xyStackStride) + (x * 2);
    if ((xyStack[xyStackIdx] == 0) && (xyStack[xyStackIdx + 1] == 0))  return;
    else {
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
      for (int i = 0; i < 8; i++) {
        const ago_coord2d_short_t offs = dir_offsets[i];
        unsigned short int x1 = x + offs.x;
		unsigned short int y1 = y + offs.y;
        int dstIdxNeighbor =  y1*(dstImageStrideInBytes) + x1;
        if (pDstImage[dstIdxNeighbor] == 127) {
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
    hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 xyStackTop
    ) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = dstWidth,   globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannyEdgeTrace_U8_U8XY,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, stream, dstWidth, dstHeight,
                    (unsigned char *)pHipDstImage, dstImageStrideInBytes,
                    (unsigned short int *) xyStack);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxNonMaxSupp kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxOpticalFlow kernels for hip backend
// ----------------------------------------------------------------------------

