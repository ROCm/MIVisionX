/* 
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 
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


#include "ago_internal.h"

#include <vector>

typedef struct {
	vx_float32 GxGx;
	vx_float32 GxGy;
	vx_float32 GyGy;
} ago_harris_Gxy_t;

void insertAtLocation(vx_uint32 listCapacity, vx_keypoint_t * pList, vx_uint32 * cornerCount, vx_keypoint_t itemToBeAdded, vx_uint32 loc)
{
	vx_keypoint_t incoming_keypt = itemToBeAdded;
	vx_keypoint_t temp;

	for (int i = (int)loc; i <= (int)*cornerCount; i++)
	{
		temp = pList[i];
		pList[i] = incoming_keypt;
		incoming_keypt = temp;
	}

	*cornerCount = *cornerCount + 1;
}

void AddToTheSortedKeypointList(vx_uint32 listCapacity, vx_keypoint_t * pList, vx_uint32 * cornerCount, vx_keypoint_t itemToBeAdded)
{
	if (*cornerCount == 0)									// Add the item to the head
	{
		pList[0] = itemToBeAdded;
		*cornerCount = 1;
	}
	else
	{
		if (itemToBeAdded.strength <= pList[*cornerCount - 1].strength)
		{
			if (*cornerCount == listCapacity)
				return;
			else
			{
				pList[*cornerCount] = itemToBeAdded;
				*cornerCount = *cornerCount + 1;
			}
		}
		else
		{
			int idx = 0;
			while (pList[idx].strength > itemToBeAdded.strength)
				idx++;
			insertAtLocation(listCapacity, pList, cornerCount, itemToBeAdded, idx);
		}
	}
}

// Using Separable filter:
// For Gx:
//	-1	0	1		-1	0	1		1
//	-2	0	2	=					2
//	-1	0	1						1
// For Gy:
//	-1	-2	-1		1	2	1		-1
//	 0	 0	 0	=					 0
//	 1	 2	 1						 1
int HafCpu_HarrisSobel_HG3_U8_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *) pDstGxy_ + dstGxyStrideInBytes);

	// SoA layout: 3 ping-pong rows each with separate gx, gy i16 buffers.
	// Total size: 6 * paddedWidth * sizeof(vx_int16), same as the legacy AoS layout.
	int paddedWidth = (dstWidth + 15) & ~15;
	vx_int16 * pPrev_gx = (vx_int16*)pScratch;
	vx_int16 * pPrev_gy = pPrev_gx + paddedWidth;
	vx_int16 * pCurr_gx = pPrev_gy + paddedWidth;
	vx_int16 * pCurr_gy = pCurr_gx + paddedWidth;
	vx_int16 * pNext_gx = pCurr_gy + paddedWidth;
	vx_int16 * pNext_gy = pNext_gx + paddedWidth;

	// Helper: horizontal Sobel-style filtering for one source row.
	// gx[x] = src[x+1] - src[x-1] (i16)
	// gy[x] = src[x-1] + 2*src[x] + src[x+1] (i16)
	auto compute_row = [](const vx_uint8 *src, vx_int16 *gx, vx_int16 *gy, int width)
	{
		int x = 0;
#if USE_AVX
		// AVX2: 32 bytes at a time -> 32 i16 outputs
		for (; x + 32 <= width; x += 32)
		{
			__m256i s_lo = _mm256_loadu_si256((const __m256i *)(src + x - 1));   // x-1 .. x+30
			__m256i s_md = _mm256_loadu_si256((const __m256i *)(src + x));       // x   .. x+31
			__m256i s_hi = _mm256_loadu_si256((const __m256i *)(src + x + 1));   // x+1 .. x+32

			// Widen each 32-byte block to two 16x16-bit halves. AVX2 unpack works lane-wise,
			// so first extract 128-bit halves and use SSE-style cvtepu8_epi16 for sequential output.
			__m128i s_lo_a = _mm256_castsi256_si128(s_lo);
			__m128i s_lo_b = _mm256_extracti128_si256(s_lo, 1);
			__m128i s_md_a = _mm256_castsi256_si128(s_md);
			__m128i s_md_b = _mm256_extracti128_si256(s_md, 1);
			__m128i s_hi_a = _mm256_castsi256_si128(s_hi);
			__m128i s_hi_b = _mm256_extracti128_si256(s_hi, 1);

			__m256i v_lo_a = _mm256_cvtepu8_epi16(s_lo_a);
			__m256i v_lo_b = _mm256_cvtepu8_epi16(s_lo_b);
			__m256i v_md_a = _mm256_cvtepu8_epi16(s_md_a);
			__m256i v_md_b = _mm256_cvtepu8_epi16(s_md_b);
			__m256i v_hi_a = _mm256_cvtepu8_epi16(s_hi_a);
			__m256i v_hi_b = _mm256_cvtepu8_epi16(s_hi_b);

			__m256i gx_a = _mm256_sub_epi16(v_hi_a, v_lo_a);
			__m256i gx_b = _mm256_sub_epi16(v_hi_b, v_lo_b);
			__m256i md2_a = _mm256_slli_epi16(v_md_a, 1);
			__m256i md2_b = _mm256_slli_epi16(v_md_b, 1);
			__m256i gy_a = _mm256_add_epi16(_mm256_add_epi16(v_lo_a, md2_a), v_hi_a);
			__m256i gy_b = _mm256_add_epi16(_mm256_add_epi16(v_lo_b, md2_b), v_hi_b);

			_mm256_storeu_si256((__m256i *)(gx + x),      gx_a);
			_mm256_storeu_si256((__m256i *)(gx + x + 16), gx_b);
			_mm256_storeu_si256((__m256i *)(gy + x),      gy_a);
			_mm256_storeu_si256((__m256i *)(gy + x + 16), gy_b);
		}
#endif
		for (; x + 16 <= width; x += 16)
		{
			__m128i s_lo = _mm_loadu_si128((const __m128i *)(src + x - 1));
			__m128i s_md = _mm_loadu_si128((const __m128i *)(src + x));
			__m128i s_hi = _mm_loadu_si128((const __m128i *)(src + x + 1));
			__m128i lo_lo = _mm_cvtepu8_epi16(s_lo);
			__m128i lo_hi = _mm_cvtepu8_epi16(_mm_srli_si128(s_lo, 8));
			__m128i md_lo = _mm_cvtepu8_epi16(s_md);
			__m128i md_hi = _mm_cvtepu8_epi16(_mm_srli_si128(s_md, 8));
			__m128i hi_lo = _mm_cvtepu8_epi16(s_hi);
			__m128i hi_hi = _mm_cvtepu8_epi16(_mm_srli_si128(s_hi, 8));
			__m128i gx_lo_v = _mm_sub_epi16(hi_lo, lo_lo);
			__m128i gx_hi_v = _mm_sub_epi16(hi_hi, lo_hi);
			__m128i md2_lo = _mm_slli_epi16(md_lo, 1);
			__m128i md2_hi = _mm_slli_epi16(md_hi, 1);
			__m128i gy_lo_v = _mm_add_epi16(_mm_add_epi16(lo_lo, md2_lo), hi_lo);
			__m128i gy_hi_v = _mm_add_epi16(_mm_add_epi16(lo_hi, md2_hi), hi_hi);
			_mm_storeu_si128((__m128i *)(gx + x),      gx_lo_v);
			_mm_storeu_si128((__m128i *)(gx + x + 8),  gx_hi_v);
			_mm_storeu_si128((__m128i *)(gy + x),      gy_lo_v);
			_mm_storeu_si128((__m128i *)(gy + x + 8),  gy_hi_v);
		}
		for (; x < width; x++)
		{
			gx[x] = (vx_int16)src[x + 1] - (vx_int16)src[x - 1];
			gy[x] = (vx_int16)src[x - 1] + ((vx_int16)src[x] << 1) + (vx_int16)src[x + 1];
		}
	};

	// Initial: fill prev and curr rows from src rows 0 and 1.
	compute_row(pSrcImage, pPrev_gx, pPrev_gy, (int)dstWidth);
	pSrcImage += srcImageStrideInBytes;
	compute_row(pSrcImage, pCurr_gx, pCurr_gy, (int)dstWidth);
	pSrcImage += srcImageStrideInBytes;

	const int W = (int)dstWidth;
	for (int y = 0; y < (int)dstHeight - 2; y++)
	{
		// Compute horizontal filter for the "next" row (one below current).
		compute_row(pSrcImage, pNext_gx, pNext_gy, W);

		ago_harris_Gxy_t * pLocalDst = pDstGxy;
		int x = 0;
#if USE_AVX
		// AVX2: 8 pixels at a time.
		for (; x + 8 <= W; x += 8)
		{
			__m128i gx_next_i16 = _mm_loadu_si128((const __m128i *)(pNext_gx + x));
			__m128i gx_curr_i16 = _mm_loadu_si128((const __m128i *)(pCurr_gx + x));
			__m128i gx_prev_i16 = _mm_loadu_si128((const __m128i *)(pPrev_gx + x));
			__m128i gy_next_i16 = _mm_loadu_si128((const __m128i *)(pNext_gy + x));
			__m128i gy_prev_i16 = _mm_loadu_si128((const __m128i *)(pPrev_gy + x));

			__m256i gx_next = _mm256_cvtepi16_epi32(gx_next_i16);
			__m256i gx_curr = _mm256_cvtepi16_epi32(gx_curr_i16);
			__m256i gx_prev = _mm256_cvtepi16_epi32(gx_prev_i16);
			__m256i gy_next = _mm256_cvtepi16_epi32(gy_next_i16);
			__m256i gy_prev = _mm256_cvtepi16_epi32(gy_prev_i16);

			__m256i gx_curr_x2 = _mm256_slli_epi32(gx_curr, 1);
			__m256i gx_final_i = _mm256_add_epi32(_mm256_add_epi32(gx_prev, gx_curr_x2), gx_next);
			__m256i gy_final_i = _mm256_sub_epi32(gy_next, gy_prev);

			__m256 gx_f = _mm256_cvtepi32_ps(gx_final_i);
			__m256 gy_f = _mm256_cvtepi32_ps(gy_final_i);
			__m256 gxgx = _mm256_mul_ps(gx_f, gx_f);
			__m256 gxgy = _mm256_mul_ps(gx_f, gy_f);
			__m256 gygy = _mm256_mul_ps(gy_f, gy_f);

			alignas(32) float gxgx_arr[8], gxgy_arr[8], gygy_arr[8];
			_mm256_store_ps(gxgx_arr, gxgx);
			_mm256_store_ps(gxgy_arr, gxgy);
			_mm256_store_ps(gygy_arr, gygy);
			for (int k = 0; k < 8; k++)
			{
				pLocalDst[k].GxGx = gxgx_arr[k];
				pLocalDst[k].GxGy = gxgy_arr[k];
				pLocalDst[k].GyGy = gygy_arr[k];
			}
			pLocalDst += 8;
		}
#endif
		// SSE: 4 pixels at a time.
		for (; x + 4 <= W; x += 4)
		{
			__m128i gx_next_i16 = _mm_loadl_epi64((const __m128i *)(pNext_gx + x));
			__m128i gx_curr_i16 = _mm_loadl_epi64((const __m128i *)(pCurr_gx + x));
			__m128i gx_prev_i16 = _mm_loadl_epi64((const __m128i *)(pPrev_gx + x));
			__m128i gy_next_i16 = _mm_loadl_epi64((const __m128i *)(pNext_gy + x));
			__m128i gy_prev_i16 = _mm_loadl_epi64((const __m128i *)(pPrev_gy + x));

			__m128i gx_next = _mm_cvtepi16_epi32(gx_next_i16);
			__m128i gx_curr = _mm_cvtepi16_epi32(gx_curr_i16);
			__m128i gx_prev = _mm_cvtepi16_epi32(gx_prev_i16);
			__m128i gy_next = _mm_cvtepi16_epi32(gy_next_i16);
			__m128i gy_prev = _mm_cvtepi16_epi32(gy_prev_i16);

			__m128i gx_curr_x2 = _mm_slli_epi32(gx_curr, 1);
			__m128i gx_final_i = _mm_add_epi32(_mm_add_epi32(gx_prev, gx_curr_x2), gx_next);
			__m128i gy_final_i = _mm_sub_epi32(gy_next, gy_prev);

			__m128 gx_f = _mm_cvtepi32_ps(gx_final_i);
			__m128 gy_f = _mm_cvtepi32_ps(gy_final_i);
			__m128 gxgx = _mm_mul_ps(gx_f, gx_f);
			__m128 gxgy = _mm_mul_ps(gx_f, gy_f);
			__m128 gygy = _mm_mul_ps(gy_f, gy_f);

			alignas(16) float gxgx_arr[4], gxgy_arr[4], gygy_arr[4];
			_mm_store_ps(gxgx_arr, gxgx);
			_mm_store_ps(gxgy_arr, gxgy);
			_mm_store_ps(gygy_arr, gygy);
			for (int k = 0; k < 4; k++)
			{
				pLocalDst[k].GxGx = gxgx_arr[k];
				pLocalDst[k].GxGy = gxgy_arr[k];
				pLocalDst[k].GyGy = gygy_arr[k];
			}
			pLocalDst += 4;
		}
		for (; x < W; x++)
		{
			vx_int32 gx = (vx_int32)pPrev_gx[x] + ((vx_int32)pCurr_gx[x] << 1) + (vx_int32)pNext_gx[x];
			vx_int32 gy = (vx_int32)pNext_gy[x] - (vx_int32)pPrev_gy[x];
			pLocalDst->GxGx = (vx_float32)gx * (vx_float32)gx;
			pLocalDst->GxGy = (vx_float32)gx * (vx_float32)gy;
			pLocalDst->GyGy = (vx_float32)gy * (vx_float32)gy;
			pLocalDst++;
		}

		// Rotate buffers
		vx_int16 *tmp_gx = pPrev_gx, *tmp_gy = pPrev_gy;
		pPrev_gx = pCurr_gx; pPrev_gy = pCurr_gy;
		pCurr_gx = pNext_gx; pCurr_gy = pNext_gy;
		pNext_gx = tmp_gx;   pNext_gy = tmp_gy;

		pSrcImage += srcImageStrideInBytes;
		pDstGxy += (dstGxyStrideInBytes / sizeof(ago_harris_Gxy_t));
	}


#if 0
	pSrcImage += srcImageStrideInBytes;										// First row not processed
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	__declspec(align(16)) short r0[3840 * 2], r1[3840 * 2], r2[3840 * 2];	// Intermideate buffers to store results between horizontally filtered rows - [GxL GxH GyL GyH]

	__m128i * pPrevRow = (__m128i *) r0;
	__m128i * pCurrRow = (__m128i *) r1;
	__m128i * pNextRow = (__m128i *) r2;

	__m128i row0, temp0, temp1, temp2, temp3, Gx, Gy;
	__m128i zeromask = _mm_setzero_si128();

	__m128i * pLocalPrevRow = pPrevRow;
	__m128i * pLocalCurrRow = pCurrRow;
	__m128i * pLocalNextRow = pNextRow;
	//__m128i * pTemp;

	int alignedWidth = dstWidth & ~15;								// Sixteen pixels processed in a go for first two rows
	int postfixWidth = dstWidth & 15;
	int srcStride = (int)srcImageStrideInBytes;

	// Process first two rows
	// Process first two rows - Horizontal filtering
	for (int x = 0; x < (int)(alignedWidth >> 4); x++)
	{
		__m128i shiftedR, shiftedL;

		// row above
		row0 = _mm_load_si128((__m128i *)(pLocalSrc - srcStride));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride + 1));

		temp0 = _mm_unpackhi_epi8(row0, zeromask);
		temp0 = _mm_slli_epi16(temp0, 1);								// GyH: 2 * (0,-1)
		Gy = _mm_cvtepu8_epi16(row0);
		Gy = _mm_slli_epi16(Gy, 1);										// GyL: 2 * (0,-1)

		Gx = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		temp1 = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		temp1 = _mm_add_epi16(temp0, temp1);
		Gy = _mm_add_epi16(Gy, Gx);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		temp1 = _mm_sub_epi16(shiftedL, temp1);
		Gx = _mm_sub_epi16(shiftedR, Gx);
		temp0 = _mm_add_epi16(temp0, shiftedL);
		Gy = _mm_add_epi16(Gy, shiftedR);

		_mm_store_si128(pLocalPrevRow++, Gx);
		_mm_store_si128(pLocalPrevRow++, temp1);
		_mm_store_si128(pLocalPrevRow++, Gy);
		_mm_store_si128(pLocalPrevRow++, temp0);

		// current row
		row0 = _mm_load_si128((__m128i *)pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		temp0 = _mm_unpackhi_epi8(row0, zeromask);
		temp0 = _mm_slli_epi16(temp0, 1);								// GyH: 2 * (-1, 0)
		Gy = _mm_cvtepu8_epi16(row0);
		Gy = _mm_slli_epi16(Gy, 1);										// GyL: 2 * (-1, 0)

		Gx = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		temp1 = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		temp0 = _mm_add_epi16(temp0, temp1);
		Gy = _mm_add_epi16(Gy, Gx);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		temp1 = _mm_sub_epi16(shiftedL, temp1);
		Gx = _mm_sub_epi16(shiftedR, Gx);
		temp0 = _mm_add_epi16(temp0, shiftedL);
		Gy = _mm_add_epi16(Gy, shiftedR);

		_mm_store_si128(pLocalCurrRow++, Gx);
		_mm_store_si128(pLocalCurrRow++, temp1);
		_mm_store_si128(pLocalCurrRow++, Gy);
		_mm_store_si128(pLocalCurrRow++, temp0);

		pLocalSrc += 16;
	}

	short * pShort_Prev = (short *)pLocalPrevRow;
	short * pShort_Curr = (short *)pLocalCurrRow;
	for (int x = 0; x < postfixWidth; x++)
	{
		// Row above
		*pShort_Prev++ = (short)pLocalSrc[-srcStride + 1] - (short)pLocalSrc[-srcStride - 1];										// Gx
		*pShort_Prev++ = (short)pLocalSrc[-srcStride + 1] + (short)pLocalSrc[-srcStride] + (short)pLocalSrc[-srcStride - 1];		// Gy

		// Current row
		*pShort_Curr++ = (short)pLocalSrc[1] - (short)pLocalSrc[-1];										// Gx
		*pShort_Curr++ = (short)pLocalSrc[1] + (short)pLocalSrc[0] + (short)pLocalSrc[-1];					// Gy
	}
	
	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)(dstHeight - 2);
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below

		int width = (int)(alignedWidth >> 3);											// Eight pixels processed in a go
		while (width)
		{
			__m128i prevRowFiltered, currRowFiltered;

			// Horizontal filtering - next row
			row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			Gx = _mm_cvtepu8_epi16(row0);												// 1 * (-1,1)
			Gy = _mm_add_epi16(Gx, zeromask);											// 1 * (-1,1)

			prevRowFiltered = _mm_load_si128(pLocalPrevRow++);

			row0 = _mm_srli_si128(row0, 1);
			temp0 = _mm_cvtepu8_epi16(row0);
			temp0 = _mm_slli_epi16(temp0, 1);											// 2 * (0,1)
			Gy = _mm_add_epi16(Gy, temp0);

			currRowFiltered = _mm_load_si128(pLocalCurrRow++);

			row0 = _mm_srli_si128(row0, 1);
			temp0 = _mm_cvtepu8_epi16(row0);											// 1 * (1,1)
			Gx = _mm_sub_epi16(temp0, Gx);
			Gy = _mm_add_epi16(Gy, temp0);

			currRowFiltered = _mm_slli_epi16(currRowFiltered, 1);						// 2 * filteredCurrRow
			Gx = _mm_add_epi16(Gx, currRowFiltered);

			Gx = _mm_add_epi16(Gx, prevRowFiltered);									// Gx0 Gx1 Gx2 Gx3 Gx4 Gx5 Gx6 Gx7
			Gy = _mm_subs_epi16(Gy, prevRowFiltered);									// Gy0 Gy1 Gy2 Gy3 Gy4 Gy5 Gy6 Gy7

			prevRowFiltered = _mm_cvtepi16_epi32(Gx);									// Gx0 Gx1 Gx2 Gx3
			currRowFiltered = _mm_cvtepi16_epi32(Gy);									// Gy0 Gy1 Gy2 Gy3

			temp0 = _mm_shuffle_epi32(prevRowFiltered, 64);								// Gx0 Gx0 Gx0 Gx1
			temp1 = _mm_shuffle_epi32(currRowFiltered, 64);								// Gy0 Gy0 Gy0 Gy1

			temp2 = _mm_blend_epi16(temp0, temp1, 0x10);								// Gx0 Gx0 Gy0 Gx1
			temp3 = _mm_blend_epi32(temp0, temp1, 0x14);								// Gx0 Gy0 Gy0 Gx1

			
			width--;

		}
		height--;
	}
#endif
	return AGO_SUCCESS;
}

// Using separable filter
//			-1	-2	0	2	1			1
//										4
//  Gx =								6
//										4
//										1
int HafCpu_HarrisSobel_HG3_U8_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *)pDstGxy_ + 2*dstGxyStrideInBytes);

	int tmpWidth = (dstWidth + 15) & ~15;
	tmpWidth <<= 1;
	vx_int16 * pRowMinus2 = (vx_int16*)pScratch;
	vx_int16 * pRowMinus1 = ((vx_int16*)pScratch) + tmpWidth;
	vx_int16 * pRowCurr = ((vx_int16*)pScratch) + (2*tmpWidth);
	vx_int16 * pRowPlus1 = ((vx_int16*)pScratch) + (3*tmpWidth);
	vx_int16 * pRowPlus2 = ((vx_int16*)pScratch) + (4*tmpWidth);

	vx_int16 * pLocalRowMinus2 = pRowMinus2;
	vx_int16 * pLocalRowMinus1 = pRowMinus1;
	vx_int16 * pLocalRowCurr = pRowCurr;
	vx_int16 * pLocalRowPlus1 = pRowPlus1;
	vx_int16 * pLocalRowPlus2 = pRowPlus2;

	// Horizontal filtering for the first row - row 0
	vx_uint8 * pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus2++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowMinus2++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 1
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus1++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowMinus1++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 2
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowCurr++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowCurr++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 3
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus1++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowPlus1++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	pSrcImage += srcImageStrideInBytes;
	
	pLocalRowMinus2 = pRowMinus2;
	pLocalRowMinus1 = pRowMinus1;
	pLocalRowCurr = pRowCurr;
	pLocalRowPlus1 = pRowPlus1;

	// Process rows 4 until end
	for (int y = 0; y < (int)dstHeight - 4; y++)
	{
		pLocalSrc = pSrcImage;
		for (int x = 0; x < (int)dstWidth; x++)
		{
			vx_int16 gx, gy;
			
			gx = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
			gy = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);

			*pLocalRowPlus2++ = gx;
			*pLocalRowPlus2++ = gy;

			gx += *pLocalRowMinus2++ + ((*pLocalRowMinus1++ + *pLocalRowCurr + *pLocalRowPlus1++) << 2) + (*pLocalRowCurr << 1);
			gy += ((*pLocalRowPlus1++ - *pLocalRowMinus1++) << 1) - *pLocalRowMinus2++;
			pLocalRowCurr += 2;

			pDstGxy->GxGx = ((vx_float32)gx * (vx_float32)gx);	// / 16.0f;
			pDstGxy->GxGy = ((vx_float32)gx * (vx_float32)gy);	// / 16.0f;
			pDstGxy->GyGy = ((vx_float32)gy * (vx_float32)gy);	// / 16.0f;

			pDstGxy++;
			pLocalSrc++;
		}

		vx_int16 * pTemp = pRowMinus2;
		pRowMinus2 = pRowMinus1;
		pRowMinus1 = pRowCurr;
		pRowCurr = pRowPlus1;
		pRowPlus1 = pRowPlus2;
		pRowPlus2 = pTemp;

		pLocalRowMinus2 = pRowMinus2;
		pLocalRowMinus1 = pRowMinus1;
		pLocalRowCurr = pRowCurr;
		pLocalRowPlus1 = pRowPlus1;
		pLocalRowPlus2 = pRowPlus2;

		pSrcImage += srcImageStrideInBytes;
	}
	
	return AGO_SUCCESS;
}

// Using separable filter
//				-1	-4	-5	0	5	4	1			1
//													6
//													15
//		Gx =										20	
//													15
//													6
//													1
int HafCpu_HarrisSobel_HG3_U8_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *)pDstGxy_ + 3*dstGxyStrideInBytes);

	int tmpWidth = (dstWidth + 15) & ~15;
	tmpWidth <<= 1;
	vx_int32 * pRowMinus3 = (vx_int32*)pScratch;
	vx_int32 * pRowMinus2 = ((vx_int32*)pScratch) + tmpWidth;
	vx_int32 * pRowMinus1 = ((vx_int32*)pScratch) + (2 * tmpWidth);
	vx_int32 * pRowCurr = ((vx_int32*)pScratch) + (3 * tmpWidth);
	vx_int32 * pRowPlus1 = ((vx_int32*)pScratch) + (4 * tmpWidth);
	vx_int32 * pRowPlus2 = ((vx_int32*)pScratch) + (5 * tmpWidth);
	vx_int32 * pRowPlus3 = ((vx_int32*)pScratch) + (6 * tmpWidth);

	vx_int32 * pLocalRowMinus3 = pRowMinus3;
	vx_int32 * pLocalRowMinus2 = pRowMinus2;
	vx_int32 * pLocalRowMinus1 = pRowMinus1;
	vx_int32 * pLocalRowCurr = pRowCurr;
	vx_int32 * pLocalRowPlus1 = pRowPlus1;
	vx_int32 * pLocalRowPlus2 = pRowPlus2;
	vx_int32 * pLocalRowPlus3 = pRowPlus3;

	// Horizontal filtering for the first row - row 0
	vx_uint8 * pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus3++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus3++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 1
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus2++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus2++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 2
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus1++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus1++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 3
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowCurr++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowCurr++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 4
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus1++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowPlus1++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 5
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus2++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowPlus2++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	pSrcImage += srcImageStrideInBytes;

	pLocalRowMinus3 = pRowMinus3;
	pLocalRowMinus2 = pRowMinus2;
	pLocalRowMinus1 = pRowMinus1;
	pLocalRowCurr = pRowCurr;
	pLocalRowPlus1 = pRowPlus1;
	pLocalRowPlus2 = pRowPlus2;

	// Process rows 4 until end
	for (int y = 0; y < (int)dstHeight - 6; y++)
	{
		pLocalSrc = pSrcImage;
		for (int x = 0; x < (int)dstWidth; x++)
		{
			vx_int32 gx, gy;

			gx = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
			gy = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);

			*pLocalRowPlus3++ = gx;
			*pLocalRowPlus3++ = gy;

			gx += *pLocalRowMinus3++ + ((*pLocalRowMinus2++ + *pLocalRowPlus2++) * 6) + ((*pLocalRowMinus1++ + *pLocalRowPlus1++) * 15) + (*pLocalRowCurr++ * 20);
			gy += ((*pLocalRowPlus2++ - *pLocalRowMinus2++) << 2) + ((*pLocalRowPlus1++ - *pLocalRowMinus1++) * 5) - *pLocalRowMinus3++;
			pLocalRowCurr++;

			pDstGxy->GxGx = ((vx_float32)gx * (vx_float32)gx);	// / 64.0f;
			pDstGxy->GxGy = ((vx_float32)gx * (vx_float32)gy);	// / 64.0f;
			pDstGxy->GyGy = ((vx_float32)gy * (vx_float32)gy);	// / 64.0f;
			pDstGxy++;
			pLocalSrc++;
		}

		vx_int32 * pTemp = pRowMinus3;
		pRowMinus3 = pRowMinus2;
		pRowMinus2 = pRowMinus1;
		pRowMinus1 = pRowCurr;
		pRowCurr = pRowPlus1;
		pRowPlus1 = pRowPlus2;
		pRowPlus2 = pRowPlus3;
		pRowPlus3 = pTemp;

		pLocalRowMinus3 = pRowMinus3;
		pLocalRowMinus2 = pRowMinus2;
		pLocalRowMinus1 = pRowMinus1;
		pLocalRowCurr = pRowCurr;
		pLocalRowPlus1 = pRowPlus1;
		pLocalRowPlus2 = pRowPlus2;
		pLocalRowPlus3 = pRowPlus3;

		pSrcImage += srcImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	memset(pDstVc, 0, dstVcStrideInBytes);											// Zero the thresholds of first row
	pDstVc += dstStride;

	// Column-sum scratch: per source column, holds (gx2_sum, gxy2_sum, gy2_sum, 0)
	// summed vertically over the current 3-row window. We index it as plain
	// float* and use unaligned SSE load/store; std::vector<float> storage is not
	// guaranteed 16-byte aligned per the standard so __m128 array indexing would
	// be undefined.
	std::vector<float> colSumBuf((size_t)dstWidth * 4 + 16);
	float *colSum = colSumBuf.data();
	auto colSumLoad  = [&](int idx) { return _mm_loadu_ps(colSum + (size_t)idx * 4); };
	auto colSumStore = [&](int idx, __m128 v) { _mm_storeu_ps(colSum + (size_t)idx * 4, v); };

	const __m128 invNorm = _mm_set1_ps(1.0f / normalization_factor);
	const __m128 sens    = _mm_set1_ps(sensitivity);
	const __m128 Tc_vec  = _mm_set1_ps(Tc);

	for (int y = 1; y < (int)dstHeight - 1; y++)
	{
		vx_float32 * pLocalDst = pDstVc;
		ago_harris_Gxy_t * pRow0 = pSrcGxy + (y - 1) * srcStride;
		ago_harris_Gxy_t * pRow1 = pRow0 + srcStride;
		ago_harris_Gxy_t * pRow2 = pRow1 + srcStride;

		// Build vertical column sums for this row triplet. Each Gxy_t is 3 floats; the
		// 4-float load picks up the next pixel's GxGx in lane 3 which we discard (only
		// lanes 0..2 are summed into the window). Last column uses a scalar gather to
		// avoid reading past the row buffer.
		int x = 0;
		for (; x < (int)dstWidth - 1; x++)
		{
			__m128 a = _mm_loadu_ps(&pRow0[x].GxGx);
			__m128 b = _mm_loadu_ps(&pRow1[x].GxGx);
			__m128 c = _mm_loadu_ps(&pRow2[x].GxGx);
			colSumStore(x, _mm_add_ps(_mm_add_ps(a, b), c));
		}
		{
			int xl = (int)dstWidth - 1;
			vx_float32 v0 = pRow0[xl].GxGx + pRow1[xl].GxGx + pRow2[xl].GxGx;
			vx_float32 v1 = pRow0[xl].GxGy + pRow1[xl].GxGy + pRow2[xl].GxGy;
			vx_float32 v2 = pRow0[xl].GyGy + pRow1[xl].GyGy + pRow2[xl].GyGy;
			colSumStore(xl, _mm_setr_ps(v0, v1, v2, 0.0f));
		}

		*pLocalDst++ = 0.0f;													// First column Vc = 0;

		// Process 4 output pixels at a time. Each iteration reads 6 column sums,
		// builds 4 windows, transposes them into SoA (gx2/gxy2/gy2/junk vectors)
		// and computes the Harris score vector-wide.
		x = 1;
		for (; x + 4 <= (int)dstWidth - 1; x += 4)
		{
			__m128 c0 = colSumLoad(x - 1);
			__m128 c1 = colSumLoad(x);
			__m128 c2 = colSumLoad(x + 1);
			__m128 c3 = colSumLoad(x + 2);
			__m128 c4 = colSumLoad(x + 3);
			__m128 c5 = colSumLoad(x + 4);

			__m128 w0 = _mm_add_ps(_mm_add_ps(c0, c1), c2);
			__m128 w1 = _mm_add_ps(_mm_add_ps(c1, c2), c3);
			__m128 w2 = _mm_add_ps(_mm_add_ps(c2, c3), c4);
			__m128 w3 = _mm_add_ps(_mm_add_ps(c3, c4), c5);

			_MM_TRANSPOSE4_PS(w0, w1, w2, w3);
			// w0 = gx2 for 4 pixels, w1 = gxy2 for 4 pixels, w2 = gy2 for 4 pixels.
			__m128 trace = _mm_add_ps(w0, w2);
			__m128 det   = _mm_sub_ps(_mm_mul_ps(w0, w2), _mm_mul_ps(w1, w1));
			__m128 Mc    = _mm_sub_ps(det, _mm_mul_ps(_mm_mul_ps(sens, trace), trace));
			Mc = _mm_mul_ps(Mc, invNorm);
			__m128 keep = _mm_cmpgt_ps(Mc, Tc_vec);
			Mc = _mm_and_ps(Mc, keep);
			_mm_storeu_ps(pLocalDst, Mc);
			pLocalDst += 4;
		}
		for (; x < (int)dstWidth - 1; x++)
		{
			__m128 win = _mm_add_ps(_mm_add_ps(colSumLoad(x - 1), colSumLoad(x)), colSumLoad(x + 1));
			alignas(16) float w[4];
			_mm_store_ps(w, win);
			vx_float32 traceA = w[0] + w[2];
			vx_float32 detA   = w[0] * w[2] - w[1] * w[1];
			vx_float32 Mc     = (detA - sensitivity * traceA * traceA) / normalization_factor;
			*pLocalDst++ = (Mc > Tc) ? Mc : 0.0f;
		}

		*pLocalDst = 0.0f;														// Last column Vc = 0;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes);											// Zero the thresholds of last row
	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	pSrcGxy += (srcStride + srcStride);									// Skip first two rows
	memset(pDstVc, 0, dstVcStrideInBytes + dstVcStrideInBytes);			// Zero the thresholds of first two rows
	pDstVc += (dstStride + dstStride);

	for (int y = 2; y < (int)dstHeight - 2; y++)
	{
		ago_harris_Gxy_t * pLocalSrc = pSrcGxy;
		vx_float32  * pLocalDst = pDstVc;

		*pLocalDst = 0;															// First column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Second column Vc = 0;
		pLocalDst++;
		pLocalSrc += 2;

		for (int x = 2; x < (int)dstWidth - 2; x++)
		{
			vx_float32 gx2 = 0;
			vx_float32 gy2 = 0;
			vx_float32 gxy2 = 0;

			// Windowing
			for (int j = -2; j <= 2; j++)
			{
				for (int i = -2; i <= 2; i++)
				{
					gx2 += pLocalSrc[j * srcStride + i].GxGx;
					gxy2 += pLocalSrc[j * srcStride + i].GxGy;
					gy2 += pLocalSrc[j * srcStride + i].GyGy;
				}
			}

			vx_float32 traceA = gx2 + gy2;
			vx_float32 detA = (gx2 * gy2) - (gxy2 * gxy2);
			vx_float32 Mc = detA - (sensitivity * traceA * traceA);
			Mc /= normalization_factor;
			*pLocalDst = (Mc > Tc) ? Mc : 0;

			pLocalSrc++;
			pLocalDst++;
		}

		*pLocalDst = 0;															// second to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// last column Vc = 0;

		pSrcGxy += srcStride;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes + dstVcStrideInBytes);					// Zero the thresholds of last rows
	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	pSrcGxy += (srcStride * 3);																// Skip first three rows
	memset(pDstVc, 0, dstVcStrideInBytes * 3);												// Zero the thresholds of first three rows
	pDstVc += (dstStride * 3);

	for (int y = 3; y < (int)dstHeight - 3; y++)
	{
		ago_harris_Gxy_t * pLocalSrc = pSrcGxy;
		vx_float32 * pLocalDst = pDstVc;

		*pLocalDst = 0;															// First column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Second column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Third column Vc = 0;
		pLocalSrc += 3;

		for (int x = 3; x < (int)dstWidth - 3; x++)
		{
			vx_float32 gx2 = 0;
			vx_float32 gy2 = 0;
			vx_float32 gxy2 = 0;

			// Windowing
			for (int j = -3; j <= 3; j++)
			{
				for (int i = -3; i <= 3; i++)
				{
					gx2 += pLocalSrc[j * srcStride + i].GxGx;
					gxy2 += pLocalSrc[j * srcStride + i].GxGy;
					gy2 += pLocalSrc[j * srcStride + i].GyGy;
				}
			}

			vx_float32 traceA = gx2 + gy2;
			vx_float32 detA = (gx2 * gy2) - (gxy2 * gxy2);
			vx_float32 Mc = detA - (sensitivity * traceA * traceA);
			Mc /= normalization_factor;
			*pLocalDst = (Mc > Tc) ? Mc : 0;

			pLocalSrc++;
			pLocalDst++;
		}

		*pLocalDst = 0;															// third to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// second to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// last column Vc = 0;

		pSrcGxy += srcStride;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes * 3);											// Zero the thresholds of last rows
	return AGO_SUCCESS;
}

int HafCpu_HarrisMergeSortAndPick_XY_HVC
	(
		vx_uint32         capacityOfDstCorner,
		vx_keypoint_t     dstCorner[],
		vx_uint32       * pDstCornerCount,
		vx_uint32         srcWidth,
		vx_uint32         srcHeight,
		vx_float32      * pSrcVc,
		vx_uint32         srcVcStrideInBytes,
		vx_float32        min_distance
	)
{
	vx_float32      * pLocalSrc;
	vx_float32      * pSrcVc_NMS = pSrcVc;
	vx_int32 radius = (vx_int32) min_distance;

	// Non max supression
	for (vx_int32 y = 0; y < (vx_int32)srcHeight; y++)
	{
		pLocalSrc = pSrcVc_NMS;
		for (vx_int32 x = 0; x < (vx_int32)srcWidth; x++)
		{
			vx_float32 Vc = *pLocalSrc;
			if (Vc)
			{
				
				for (vx_int32 i = max(y - radius, 0); i <= min(y + radius, (vx_int32) srcHeight - 1); i++)
				{
					for (vx_int32 j = max(x - radius, 0); j <= min(x + radius, (vx_int32) srcWidth - 1); j++)
					{
						if ((vx_float32)((y-i)*(y-i)) + (vx_float32)((x-j)*(x-j)) <= radius*radius)
						{
							vx_float32 * neighbor = (vx_float32 *)(((char *)pLocalSrc) + (i - y) * (vx_int32)srcVcStrideInBytes + (j - x) * sizeof(vx_float32));
							if (*neighbor < Vc)
								*neighbor = 0;
						}
					}
				}
			}
			pLocalSrc++;
		}

		pSrcVc_NMS = (vx_float32 *)((char *)pSrcVc_NMS + srcVcStrideInBytes);
	}	

	// Populate the sorted list
	vx_keypoint_t cand;
	vx_uint32 numCorners = 0;
	
	for (vx_uint32 y = 0; y < srcHeight; y++)
	{
		pLocalSrc = pSrcVc;
		for (vx_uint32 x = 0; x < srcWidth; x++)
		{
			if (*pLocalSrc)
			{
				cand.x = x;
				cand.y = y;
				cand.strength = *pLocalSrc;
				cand.scale = 0;
				cand.orientation = 0;
				cand.error = 0;
				cand.tracking_status = 1;
				if (numCorners < capacityOfDstCorner)
					AddToTheSortedKeypointList(capacityOfDstCorner, dstCorner, &numCorners, cand);
				else
					numCorners++;
			}
			pLocalSrc++;
		}
		pSrcVc = (vx_float32 *)((char *)pSrcVc + srcVcStrideInBytes);
	}

	*pDstCornerCount = numCorners;

	return AGO_SUCCESS;
}

int HafCpu_NonMaxSupp_XY_ANY_3x3
	(
		vx_uint32               capacityOfList,
		ago_keypoint_xys_t    * dstList,
		vx_uint32             * pDstListCount,
		vx_uint32               srcWidth,
		vx_uint32               srcHeight,
		vx_float32            * pSrcImg,
		vx_uint32               srcStrideInBytes
	)
{
	vx_uint32 count = 0;
	const vx_uint8 * pImg = (const vx_uint8 *)pSrcImg;
	const vx_int32 W = (vx_int32)srcWidth;
	const vx_int32 H = (vx_int32)srcHeight;
	for (vx_int32 y = 1; y < H - 1; y++, pImg += srcStrideInBytes) {
		if (count >= capacityOfList)
			break;
		const vx_float32 * p9 = (const vx_float32 *)&pImg[0];
		const vx_float32 * p0 = (const vx_float32 *)&pImg[srcStrideInBytes];
		const vx_float32 * p1 = (const vx_float32 *)&pImg[srcStrideInBytes << 1];

		vx_int32 x = 1;
#if USE_AVX
		// SIMD prefilter: in the Harris pipeline most pixels are 0 because the
		// score kernel zeros sub-threshold scores. Scan 8 candidates at a time
		// and only run the full 9-tap check for non-zero centers.
		const __m256 zero8 = _mm256_setzero_ps();
		for (; x + 8 <= W - 1; x += 8)
		{
			__m256 c = _mm256_loadu_ps(p0 + 1);  // 8 center values at columns x..x+7
			__m256 nz_mask = _mm256_cmp_ps(c, zero8, _CMP_GT_OQ);
			int nz = _mm256_movemask_ps(nz_mask);
			if (nz)
			{
				// SIMD-screen vs. easy neighbors (left/up-left/up-center: must be <=)
				__m256 ul = _mm256_loadu_ps(p9 + 0);
				__m256 uc = _mm256_loadu_ps(p9 + 1);
				__m256 ur = _mm256_loadu_ps(p9 + 2);
				__m256 ll = _mm256_loadu_ps(p0 + 0);
				__m256 rr = _mm256_loadu_ps(p0 + 2);
				__m256 dl = _mm256_loadu_ps(p1 + 0);
				__m256 dc = _mm256_loadu_ps(p1 + 1);
				__m256 dr = _mm256_loadu_ps(p1 + 2);
				__m256 m;
				m = _mm256_cmp_ps(c, ul, _CMP_GE_OQ);
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, uc, _CMP_GE_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, ur, _CMP_GE_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, ll, _CMP_GE_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, rr, _CMP_GT_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, dl, _CMP_GT_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, dc, _CMP_GT_OQ));
				m = _mm256_and_ps(m, _mm256_cmp_ps(c, dr, _CMP_GT_OQ));
				m = _mm256_and_ps(m, nz_mask);
				unsigned int ok_mask = (unsigned int)_mm256_movemask_ps(m);
				while (ok_mask)
				{
					int lane = agoCtz32(ok_mask);
					ok_mask &= ok_mask - 1u;
					if (count >= capacityOfList) break;
					dstList->x = (vx_uint16)(x + lane);
					dstList->y = (vx_uint16)y;
					dstList->s = p0[1 + lane];
					dstList++;
					count++;
				}
				if (count >= capacityOfList) break;
			}
			p9 += 8;
			p0 += 8;
			p1 += 8;
		}
#endif
		for (; x < W - 1; x++) {
			if (p0[1] >= p9[0] && p0[1] >= p9[1] && p0[1] >= p9[2] &&
				p0[1] >= p0[0]                   && p0[1] >  p0[2] &&
				p0[1] >  p1[0] && p0[1] >  p1[1] && p0[1] >  p1[2])
			{
				dstList->x = (vx_uint16)x;
				dstList->y = (vx_uint16)y;
				dstList->s = p0[1];
				dstList++;
				count++;
				if (count >= capacityOfList)
					break;
			}
			p9++;
			p0++;
			p1++;
		}
	}
	*pDstListCount = count;
	return AGO_SUCCESS;
}

int HafCpu_HarrisMergeSortAndPick_XY_XYS
	(
		vx_uint32                  capacityOfDstCorner,
		vx_keypoint_t            * dstCorner,
		vx_uint32                * pDstCornerCount,
		ago_keypoint_xys_t       * srcList,
		vx_uint32                  srcListCount,
		vx_float32                 min_distance,
		ago_harris_grid_header_t * gridInfo,
		ago_coord2d_short_t      * gridBuf
	)
{
	// sort the keypoint XYS list
	std::sort((vx_int64 *)&srcList[0], (vx_int64 *)&srcList[srcListCount], std::greater<vx_int64>());
	// extract useful keypoints from XYS list into corners array
	vx_uint32 count = 0;
	if (gridInfo) {
		// get grid info and initialize grid buffer if (-1,-1) coordinate values indicating no presence of values
		vx_uint32 gridWidth = gridInfo->width;
		vx_uint32 gridHeight = gridInfo->height;
		vx_uint32 cellSize = gridInfo->cellSize;
		HafCpu_MemSet_U32(gridInfo->gridBufSize >> 2, (vx_uint32 *)gridBuf, (vx_uint32)-1);
		// filter the keypoints with min_distance
		vx_int32 min_dist2 = (vx_int32)ceilf(min_distance * min_distance);
		vx_keypoint_t * corner = dstCorner;
		for (vx_uint32 i = 0; i < srcListCount; i++) {
			vx_uint32 x = srcList[i].x, y = srcList[i].y;
			bool found = true;
			vx_int32 cx = (vx_int32)x / cellSize, cy = (vx_int32)y / cellSize;
			ago_coord2d_short_t * cgrid = gridBuf + cy * gridWidth + cx;
			if (cgrid->x < 0) {
				vx_int32 cxmin = max(cx - 2, 0), cxmax = min(cx + 2, (vx_int32)gridWidth - 1), cw = cxmax - cxmin + 1;
				vx_int32 cymin = max(cy - 2, 0), cymax = min(cy + 2, (vx_int32)gridHeight - 1), ch = cymax - cymin + 1;
				ago_coord2d_short_t * grid = gridBuf + cxmin + cymin * gridWidth;
				for (vx_int32 icy = 0; icy < ch; icy++, grid += gridWidth) {
					for (vx_int32 icx = 0; icx < cw; icx++) {
						int ix = grid[icx].x;
						if (ix >= 0) {
							int iy = grid[icx].y;
							ix -= x; iy -= y;
							int dist2 = ix*ix + iy*iy;
							if (dist2 < min_dist2) {
								goto search_done;
							}
						}
					}
				}
				found = false;
			}
		search_done:
			if (!found) {
				if (count < capacityOfDstCorner) {
					corner->x = x;
					corner->y = y;
					corner->strength = srcList[i].s;
					corner->tracking_status = 1;
					corner->error = 0;
					corner->scale = 0.0f;
					corner->orientation = 0.0f;
					corner++;
				}
				count++;
				cgrid->x = x;
				cgrid->y = y;
			}
		}
	}
	else {
		// copy all points into output array
		count = (srcListCount < capacityOfDstCorner) ? srcListCount : capacityOfDstCorner;
		for (vx_uint32 i = 0; i < count; i++, dstCorner++, srcList++) {
			dstCorner->x = srcList->x;
			dstCorner->y = srcList->y;
			dstCorner->strength = srcList->s;
			dstCorner->tracking_status = 1;
			dstCorner->error = 0;
			dstCorner->scale = 0.0f;
			dstCorner->orientation = 0.0f;
		}
	}
	*pDstCornerCount = count;
	return AGO_SUCCESS;
}