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

#define CHECKMAX(a, b) (a > b ? 1 : 0)
#define CHECKMIN(a, b) (a < b ? 1 : 0)

__device__ __forceinline__ float4 ucharTofloat4(unsigned int src)
{
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint float4ToUint(float4 src)
{
  return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
}

__device__ __forceinline__ int isCorner(int mask)
{
	int cornerMask = 0x1FF;									// Nine 1's in the LSB

	if (mask)
	{
		mask = mask | (mask << 16);
		for (int i = 0; i < 16; i++)
		{
			if ((mask & cornerMask) == cornerMask)
				return 1;
			mask >>= 1;
		}
	}
	return 0;
}

__device__ __forceinline__ int isCornerPlus(short candidate, short * boundary, short t)
{
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate += t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] > candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}

	return isCorner(mask);
}

__device__ __forceinline__ int isCornerMinus(short candidate, short * boundary, short t)
{
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate -= t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] < candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}

	return isCorner(mask);
}

// ----------------------------------------------------------------------------
// VxFastCorners kernels for hip backend
// ----------------------------------------------------------------------------
__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_NoSupression(
    vx_uint32 capacityOfDstCorner, vx_keypoint_t *pDstCorner, vx_uint32 *cornercount,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes, vx_float32 threshold
	)
{
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
	if (mask_max || mask_min)
	{
		mask_max = mask_max | (mask_max << 16);
		mask_min = mask_min | (mask_min << 16);

		for (int i = 0; i < 16; i++)
		{
			if (((mask_max & cornerMask) == cornerMask) || ((mask_min & cornerMask) == cornerMask))
			{
				isCorner = 1;
				break;
			}
			mask_max >>= 1;
			mask_min >>= 1;
		}
	}

	
	if(isCorner)
	{
		unsigned int old_idx = atomicAdd(cornercount, 1);
		if (old_idx < capacityOfDstCorner)
		{
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
		vx_uint32  capacityOfDstCorner, 
    	vx_keypoint_t   pHipDstCorner[],
		vx_uint32  *pHipDstCornerCount,
		vx_uint32  srcWidth, vx_uint32 srcHeight,
		vx_uint8   *pHipSrcImage,
		vx_uint32   srcImageStrideInBytes,
		vx_float32  strength_threshold
)
{
    hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = srcWidth-6,   globalThreads_y = srcHeight-6;

    printf("srcWidth : %d srcHeight : %d\nsrcImageStrideInBytes : %d\n Capacity: %d\n",srcWidth, srcHeight, srcImageStrideInBytes, capacityOfDstCorner);
    
	vx_uint32 *cornerCount;
	hipMalloc(&cornerCount, sizeof(vx_uint32));
	hipMemcpy(cornerCount, pHipDstCornerCount, sizeof(vx_uint32), hipMemcpyHostToDevice);

	hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_NoSupression,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
                    srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold
					);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

	hipMemcpyDtoH(pHipDstCornerCount, cornerCount, sizeof(vx_uint32));

	hipFree(cornerCount);

    printf("\nHipExec_FastCorners_XY_U8_NoSupression Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_Supression(
    vx_uint32 capacityOfDstCorner, vx_keypoint_t *pDstCorner, vx_uint32 *cornercount,
    vx_uint32 srcWidth, vx_uint32 srcHeight, 
    const unsigned char *pSrcImage, unsigned int srcImageStrideInBytes, 
	vx_float32 threshold, unsigned char *pScratch
	)
{
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
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] > cand)
			maskP |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_pos = 0;
	cand = pSrcImage[srcIdx];
	if (isCorner(maskP))
	{
		short thresh_upper = 255;
		short thresh_lower = threshold;
		
		while (thresh_upper - thresh_lower > 1)						// Binary search
		{
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
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] < cand)
			maskN |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_neg = 0;
	cand = pSrcImage[srcIdx];
	if (isCorner(maskN))
	{
		short thresh_upper = 255;
		short thresh_lower = threshold;
		
		while (thresh_upper - thresh_lower > 1)						// Binary search
		{
			strength_neg = (thresh_upper + thresh_lower) >> 1;
			if (isCornerMinus(cand, boundary, strength_neg))
				thresh_lower = strength_neg;
			else
				thresh_upper = strength_neg;
		}
		strength_neg = thresh_lower;
	}

	if (maskP || maskN)
	{
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
	)
{
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
	if (old_idx < capacityOfDstCorner)
	{
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
		vx_uint32  capacityOfDstCorner, 
    	vx_keypoint_t   pHipDstCorner[],
		vx_uint32  *pHipDstCornerCount,
		vx_uint32  srcWidth, vx_uint32 srcHeight,
		vx_uint8   *pHipSrcImage,
		vx_uint32   srcImageStrideInBytes,
		vx_float32  strength_threshold,
		vx_uint8   *pHipScratch
)
{
	hipEvent_t start, stop;
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = srcWidth,   globalThreads_y = srcHeight;

    printf("srcWidth : %d srcHeight : %d\nsrcImageStrideInBytes : %d\n Capacity: %d\n",srcWidth, srcHeight, srcImageStrideInBytes, capacityOfDstCorner);

	vx_uint32 *cornerCount;
	hipMalloc(&cornerCount, sizeof(vx_uint32));
	hipMemcpy(cornerCount, pHipDstCornerCount, sizeof(vx_uint32), hipMemcpyHostToDevice);

	vx_uint8 * Scratch;
	hipMalloc(&Scratch, sizeof(vx_uint8) * srcWidth * srcHeight);
	hipMemset(Scratch, 0, sizeof(vx_uint8) * srcWidth * srcHeight);

	hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_Supression,
                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                    dim3(localThreads_x, localThreads_y),
                    0, 0, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
                    srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold, (unsigned char *)Scratch);

	hipLaunchKernelGGL(Hip_NonMaximumSupression_3x3,
				dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
				dim3(localThreads_x, localThreads_y),
				0, 0, capacityOfDstCorner,(vx_keypoint_t *) pHipDstCorner, (vx_uint32 *)cornerCount,
				srcWidth, srcHeight, (const unsigned char*) pHipSrcImage, srcImageStrideInBytes, strength_threshold, (unsigned char *)Scratch);
	

	hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&eventMs, start, stop);

	hipMemcpyDtoH(pHipScratch, Scratch, sizeof(vx_uint8) * srcWidth * srcHeight);
	hipMemcpyDtoH(pHipDstCornerCount, cornerCount, sizeof(vx_uint32));

	hipFree(cornerCount);
	hipFree(Scratch);
	
	
	printf("\nHipExec_FastCorners_XY_U8_Supression Kernel time: %f\n", eventMs);
    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxHarrisSobel kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxCannySobelSuppThreshold kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxCannySobel kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxCannySuppThreshold kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxNonMaxSupp kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxOpticalFlow kernels for hip backend
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// VxHarrisMergeSortAndPick kernels for hip backend
// ----------------------------------------------------------------------------




