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

// Vectorized version of HafCpu_FastAtan2_Canny for 8 int16 (Gx, Gy) lanes.
// Returns 8 angle codes in [0,3] packed into int16 lanes.
// Selection order matches the scalar reference: angle codes are
//   0 if |Gy| <= |Gx| * tan(22.5deg)   (nearly horizontal)
//   2 if |Gy| >= |Gx| * tan(67.5deg)   (nearly vertical)
//   1 if Gx*Gy >= 0 in between (diagonal, same sign)
//   3 otherwise                        (anti-diagonal, opposite signs)
static inline __m128i HafCpu_FastAtan2_Canny_8(__m128i Gx, __m128i Gy)
{
	const __m128i z = _mm_setzero_si128();
	const __m128 c_p414 = _mm_set1_ps(0.4142135623730950488f);
	const __m128 c_p2414 = _mm_set1_ps(2.4142135623730950488f);

	__m128i ax = _mm_abs_epi16(Gx);
	__m128i ay = _mm_abs_epi16(Gy);

	__m128 axLoF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ax, z));
	__m128 axHiF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ax, z));
	__m128 ayLoF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ay, z));
	__m128 ayHiF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(ay, z));

	__m128 d1Lo = _mm_mul_ps(axLoF, c_p414);
	__m128 d1Hi = _mm_mul_ps(axHiF, c_p414);
	__m128 d2Lo = _mm_mul_ps(axLoF, c_p2414);
	__m128 d2Hi = _mm_mul_ps(axHiF, c_p2414);

	__m128i leD1 = _mm_packs_epi32(
		_mm_castps_si128(_mm_cmple_ps(ayLoF, d1Lo)),
		_mm_castps_si128(_mm_cmple_ps(ayHiF, d1Hi)));
	__m128i geD2 = _mm_packs_epi32(
		_mm_castps_si128(_mm_cmpge_ps(ayLoF, d2Lo)),
		_mm_castps_si128(_mm_cmpge_ps(ayHiF, d2Hi)));

	__m128i signProd = _mm_xor_si128(_mm_srai_epi16(Gx, 15), _mm_srai_epi16(Gy, 15));

	__m128i quad13 = _mm_blendv_epi8(_mm_set1_epi16(1), _mm_set1_epi16(3), signProd);
	__m128i quadOr2 = _mm_blendv_epi8(quad13, _mm_set1_epi16(2), geD2);
	return _mm_blendv_epi8(quadOr2, z, leD1);
}

static const int n_offset[][2][2] = {
	{ { -1, 0 }, { 1, 0 } },
	{ { 1, -1 }, { -1, 1 } },
	{ { 0, -1 }, { 0, 1 } },
	{ { -1, -1 }, { 1, 1 } },
};
static const ago_coord2d_short_t dir_offsets[8] = {
	{ -1, -1 },
	{ 0, -1 },
	{ +1, -1 },
	{ -1, 0 },
	{ +1, 0 },
	{ -1, +1 },
	{ 0, +1 },
	{ +1, +1 },
};


int HafCpu_CannySobel_U16_U8_3x3_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	pSrcImage += srcImageStrideInBytes;
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += dstride;		// don't care about border. start processing from row2
	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 1; y < (int)dstHeight - 1; y++)
	{
		const vx_uint8* srow0 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage;
		const vx_uint8* srow2 = pSrcImage + srcImageStrideInBytes;
		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_add_epi16(s2, s4), _mm_slli_epi16(s3, 1));
			t1 = _mm_sub_epi16(z, t1);
			s1 = HafCpu_FastAtan2_Canny_8(t0, t1);
			t0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			// pack with signed saturation
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

// Using separable filter
//			-1	-2	0	2	1			1
//										4
//  Gx =								6
//										4
//										1

int HafCpu_CannySobel_U16_U8_5x5_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 2 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 2 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 2; y < (int)dstHeight - 2; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage;
		const vx_uint8* srow3 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow4 = pSrcImage + 2 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- ((vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(s1, s3), 2), _mm_mullo_epi16(s2, c6));
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s4));

			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s3, s1), 1);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s4, s0));
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));

			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s7 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			__m128i s8 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));

			__m128i t0 = _mm_slli_epi16(_mm_sub_epi16(s2, s1), 1);
			t0 = _mm_adds_epi16(t0, _mm_sub_epi16(s3, s0));
			__m128i t1 = _mm_slli_epi16(_mm_add_epi16(s5, s7), 2);
			s0 = _mm_mullo_epi16(s6, c6);
			t1 = _mm_add_epi16(t1, _mm_add_epi16(s4, s8));
			t1 = _mm_add_epi16(t1, s0);
			t1 = _mm_sub_epi16(z, t1);
			// find magnitude
			s0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			//s0 = _mm_min_epi16(s0, clamp);
			t0 = HafCpu_FastAtan2_Canny_8(t0, t1);
			s0 = _mm_or_si128(_mm_slli_epi16(s0, 2), t0);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), s0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- ((vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}

	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_7x7_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c5 = _mm_set1_epi16(5), c6 = _mm_set1_epi16(6);
	__m128i c15 = _mm_set1_epi16(15), c20 = _mm_set1_epi16(20);
	__m128i clamp = _mm_set1_epi16(0x3FFF);

	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 3 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 3 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 3; y < (int)dstHeight - 3; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 3 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow3 = pSrcImage;
		const vx_uint8* srow4 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow5 = pSrcImage + 2 * srcImageStrideInBytes;
		const vx_uint8* srow6 = pSrcImage + 3 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);
			__m128i s5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow5 + x)), z);
			__m128i s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow6 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s6));
			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s5, s1), 2);
			t0 = _mm_add_epi16(t0, t2);

			t2 = _mm_mullo_epi16(_mm_sub_epi16(s4, s2), c5);
			t0 = _mm_srai_epi16(t0, 2);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s6, s0));
			t1 = _mm_add_epi16(t1, t2);
			t1 = _mm_srai_epi16(t1, 2);

			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 3));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r0 + x + 3));


			__m128i t0 = _mm_slli_epi16(_mm_subs_epi16(s4, s1), 2);
			__m128i t1 = _mm_mullo_epi16(_mm_subs_epi16(s3, s2), c5);
			t0 = _mm_adds_epi16(t0, _mm_subs_epi16(s5, s0));
			t0 = _mm_adds_epi16(t0, t1);

			s0 = _mm_loadu_si128((const __m128i*)(r1 + x - 3));
			s1 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			s5 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x + 3));


			t1 = _mm_adds_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t1 = _mm_adds_epi16(t1, _mm_adds_epi16(s0, s6));
			t1 = _mm_adds_epi16(t1, t2);
			t1 = _mm_subs_epi16(z, t1);
			// find magnitude
			s0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			s0 = _mm_min_epi16(s0, clamp);
			t0 = HafCpu_FastAtan2_Canny_8(t0, t1);
			s0 = _mm_or_si128(_mm_slli_epi16(s0, 2), t0);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), s0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}


int HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper,
		vx_uint8			 * pScratch
	)
{
	vx_int16 *Gx, *Gy;
	vx_uint8 * pTemp;
	vx_uint32 dstride = ((dstWidth + 15)&~15);
	// Gx and Gy each need (dstride * dstHeight) vx_int16 elements; the original code
	// only spaced them by `dstride` u16 elements, aliasing Gy row 0 onto Gx row 1.
	Gx = (vx_int16 *)pScratch;
	Gy = (vx_int16 *)(pScratch + dstride * dstHeight * sizeof(vx_int16));
	pTemp = pScratch + 2 * dstride * dstHeight * sizeof(vx_int16);
	// compute Sobel gradients
	HafCpu_Sobel_S16S16_U8_3x3_GXY(dstWidth, dstHeight - 2, Gx + dstride, dstride * 2, Gy + dstride, dstride * 2, pSrcImage + srcImageStrideInBytes, srcImageStrideInBytes, pTemp);
	
	// compute L1 norm and phase
	// Vectorize the scalar HafCpu_FastAtan2_deg loop using HafCpu_FastAtan2_Canny_8 which
	// produces the same 0..3 angle bucket directly from (Gx, Gy) vectors. The 8-wide path
	// packs (mag << 2) | orientation into u16 lanes and stores them as a __m128i.
	//
	// NOTE on Gy sign convention: HafCpu_Sobel_S16S16_U8_3x3_GXY emits Gy = src(y+1) - src(y-1)
	// (positive downward in screen coordinates) while the angle table (n_offset) and the
	// HafCpu_FastAtan2_Canny_8 quadrant logic expect the Canny-classic Gy = src(y-1) - src(y+1)
	// (positive upward). Negate Gy before feeding the angle classifier to keep the
	// diagonal angle buckets (1 and 3) correctly aligned with the right neighbour pair.
	//
	// The Sobel kernel populates rows 1..dstHeight-2 of Gx/Gy in-place (called with
	// dstHeight-2 as the output height). Write the packed mag+angle back into the same
	// (Gx) row we read from, so that the subsequent NMS loop indexes correctly by row.
	vx_int16 *pGx = Gx + dstride;
	vx_int16 *pGy = Gy + dstride;
	vx_int16 *pMag = Gx + dstride;
	const __m128i sse_zero = _mm_setzero_si128();
	for (unsigned int y = 1; y < dstHeight - 1; y++)
	{
		vx_uint16 *pdst = (vx_uint16*)pMag;		// to store the result

		unsigned int x = 1;
		for (; x + 8 <= dstWidth; x += 8)
		{
			__m128i gxv = _mm_loadu_si128((const __m128i *)(pGx + x));
			__m128i gyv = _mm_loadu_si128((const __m128i *)(pGy + x));
			__m128i gy_neg = _mm_sub_epi16(sse_zero, gyv);
			__m128i orn = HafCpu_FastAtan2_Canny_8(gxv, gy_neg);
			__m128i ax = _mm_abs_epi16(gxv);
			__m128i ay = _mm_abs_epi16(gyv);
			__m128i mag = _mm_add_epi16(ax, ay);
			__m128i out = _mm_or_si128(_mm_slli_epi16(mag, 2), orn);
			_mm_storeu_si128((__m128i *)(pdst + x), out);
		}
		for (; x < dstWidth; x++)
		{
			vx_uint8 orn;	// orientation

			float scale = (float)128 / 180.f;
			float arct = HafCpu_FastAtan2_deg(pGx[x], (vx_int16)(-pGy[x]));
			// normalize and convert to degrees 0-180
			orn = (((int)(arct*scale) + 16) >> 5)&7;		// quantize to 8 (22.5 degrees)
			if (orn >= 4)orn -= 4;
			vx_int16 val = (vx_int16)(abs(pGx[x]) + abs(pGy[x]));
			pdst[x] = (vx_uint16)((val << 2) | orn);				// store both mag and orientation
		}
		pGx += dstride;
		pGy += dstride;
		pMag += dstride;
	}

	// do minmax suppression: from Gx
	ago_coord2d_ushort_t *pxyStack = xyStack;
	for (int y = 1; y < dstHeight - 1; y++)
	{
		vx_uint8* pOut = pDst + y*dstStrideInBytes;
		vx_int16 *pSrc = (vx_int16 *)(Gx + y * dstride);	// we are processing from 2nd row
		for (unsigned int x = 1; x < dstWidth - 1; x++, pSrc++)
		{
			vx_int32 edge;
			// get the Mag and angle
			int mag = (pSrc[0] >> 2);
			int ang = pSrc[0] & 3;
			int offset0 = n_offset[ang][0][1] * dstride + n_offset[ang][0][0];
			int offset1 = n_offset[ang][1][1] * dstride + n_offset[ang][1][0];
			edge = ((mag >(pSrc[offset0] >> 2)) && (mag >(pSrc[offset1] >> 2))) ? mag : 0;
			if (edge > hyst_upper){
				pOut[x] = (vx_int8)255;
				// add the cordinates to stacktop
				pxyStack->x = x;	// store x and y co-ordinates
				pxyStack->y = y;	// store x and y co-ordinates
				pxyStack++;
			}
			else if (edge <= hyst_lower){
				pOut[x] = 0;
			}
			else pOut[x] = 127;
		}
	}
	*pxyStackTop = (vx_uint32)(pxyStack - xyStack);

	return AGO_SUCCESS;
}

int HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper,
		vx_uint8			 * pScratch
	)
{
	vx_int16 *Gx, *Gy;
	vx_uint8 * pTemp;
	vx_uint32 dstride = ((dstWidth + 15)&~15);
	// Same layout as the L1NORM 3x3 fused path. Gx and Gy each occupy
	// dstride*dstHeight vx_int16 elements; pTemp follows.
	Gx = (vx_int16 *)pScratch;
	Gy = (vx_int16 *)(pScratch + dstride * dstHeight * sizeof(vx_int16));
	pTemp = pScratch + 2 * dstride * dstHeight * sizeof(vx_int16);
	HafCpu_Sobel_S16S16_U8_3x3_GXY(dstWidth, dstHeight - 2, Gx + dstride, dstride * 2, Gy + dstride, dstride * 2, pSrcImage + srcImageStrideInBytes, srcImageStrideInBytes, pTemp);

	// Compute L2 norm magnitude (sqrt(Gx^2 + Gy^2)) and angle bucket, packing
	// (mag << 2) | orientation back into the Gx rows. Mirrors the SIMD trick used
	// by HafCpu_CannySobel_U16_U8_3x3_L2NORM (unpacklo/unpackhi + madd + sqrt_ps).
	vx_int16 *pGx = Gx + dstride;
	vx_int16 *pGy = Gy + dstride;
	vx_int16 *pMag = Gx + dstride;
	const __m128i sse_zero = _mm_setzero_si128();
	for (unsigned int y = 1; y < dstHeight - 1; y++)
	{
		vx_uint16 *pdst = (vx_uint16*)pMag;
		unsigned int x = 1;
		for (; x + 8 <= dstWidth; x += 8)
		{
			__m128i gxv = _mm_loadu_si128((const __m128i *)(pGx + x));
			__m128i gyv = _mm_loadu_si128((const __m128i *)(pGy + x));
			__m128i gy_neg = _mm_sub_epi16(sse_zero, gyv);
			__m128i orn = HafCpu_FastAtan2_Canny_8(gxv, gy_neg);
			__m128i lo = _mm_unpacklo_epi16(gxv, gyv);
			__m128i hi = _mm_unpackhi_epi16(gxv, gyv);
			lo = _mm_madd_epi16(lo, lo);
			hi = _mm_madd_epi16(hi, hi);
			__m128 flo = _mm_sqrt_ps(_mm_cvtepi32_ps(lo));
			__m128 fhi = _mm_sqrt_ps(_mm_cvtepi32_ps(hi));
			__m128i mlo = _mm_cvtps_epi32(flo);
			__m128i mhi = _mm_cvtps_epi32(fhi);
			__m128i mag = _mm_packus_epi32(mlo, mhi);
			__m128i out = _mm_or_si128(_mm_slli_epi16(mag, 2), orn);
			_mm_storeu_si128((__m128i *)(pdst + x), out);
		}
		for (; x < dstWidth; x++)
		{
			float scale = (float)128 / 180.f;
			float arct = HafCpu_FastAtan2_deg(pGx[x], (vx_int16)(-pGy[x]));
			vx_uint8 orn = (((int)(arct*scale) + 16) >> 5) & 7;
			if (orn >= 4) orn -= 4;
			vx_int32 g2 = (vx_int32)pGx[x] * pGx[x] + (vx_int32)pGy[x] * pGy[x];
			vx_int16 val = (vx_int16)sqrtf((float)g2);
			pdst[x] = (vx_uint16)((val << 2) | orn);
		}
		pGx += dstride;
		pGy += dstride;
		pMag += dstride;
	}

	// Non-max suppression + dual-threshold (identical to L1 variant).
	ago_coord2d_ushort_t *pxyStack = xyStack;
	for (int y = 1; y < (int)dstHeight - 1; y++)
	{
		vx_uint8* pOut = pDst + y*dstStrideInBytes;
		vx_int16 *pSrc = (vx_int16 *)(Gx + y * dstride);
		for (unsigned int x = 1; x < dstWidth - 1; x++, pSrc++)
		{
			int mag = (pSrc[0] >> 2);
			int ang = pSrc[0] & 3;
			int offset0 = n_offset[ang][0][1] * dstride + n_offset[ang][0][0];
			int offset1 = n_offset[ang][1][1] * dstride + n_offset[ang][1][0];
			vx_int32 edge = ((mag > (pSrc[offset0] >> 2)) && (mag > (pSrc[offset1] >> 2))) ? mag : 0;
			if (edge > hyst_upper) {
				pOut[x] = (vx_int8)255;
				pxyStack->x = x;
				pxyStack->y = y;
				pxyStack++;
			}
			else if (edge <= hyst_lower) {
				pOut[x] = 0;
			}
			else pOut[x] = 127;
		}
	}
	*pxyStackTop = (vx_uint32)(pxyStack - xyStack);

	return AGO_SUCCESS;
}

int HafCpu_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_CannySuppThreshold_U8XY_U16_3x3
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint16            * pSrc,
		vx_uint32              srcStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	)
{
	// Non-max suppression + hysteresis classification, vectorized 8 pixels at a time.
	// The original scalar loop indexed neighbors via a small table (n_offset[ang][...])
	// which forced a per-pixel data-dependent address. The SIMD path instead loads all
	// four candidate neighbor offsets unconditionally and then selects the right pair
	// via 16-bit masks derived from the angle bucket. The hysteresis classification
	// (255/127/0) is computed branchlessly; only the rare stack push for "strong" edges
	// falls back to a per-lane scalar loop using a movemask of the upper-threshold mask.
	vx_uint32 sstride = srcStrideInBytes>>1;
	ago_coord2d_ushort_t *pxyStack = xyStack;
	ago_coord2d_ushort_t *pxyStackEnd = xyStack + capacityOfXY;

	const __m128i ang_mask = _mm_set1_epi16(3);
	const __m128i one_v = _mm_set1_epi16(1);
	const __m128i two_v = _mm_set1_epi16(2);
	const __m128i three_v = _mm_set1_epi16(3);
	const __m128i hyst_upper_v = _mm_set1_epi16((short)hyst_upper);
	const __m128i hyst_lower_v = _mm_set1_epi16((short)hyst_lower);
	const __m128i k255 = _mm_set1_epi16(255);
	const __m128i k127 = _mm_set1_epi16(127);
#if USE_AVX
	const __m256i ang_mask256 = _mm256_set1_epi16(3);
	const __m256i one256 = _mm256_set1_epi16(1);
	const __m256i two256 = _mm256_set1_epi16(2);
	const __m256i three256 = _mm256_set1_epi16(3);
	const __m256i hyst_upper256 = _mm256_set1_epi16((short)hyst_upper);
	const __m256i hyst_lower256 = _mm256_set1_epi16((short)hyst_lower);
	const __m256i k255_256 = _mm256_set1_epi16(255);
	const __m256i k127_256 = _mm256_set1_epi16(127);
#endif

	for (unsigned int y = 1; y < dstHeight - 1; y++)
	{
		vx_uint8* pOut = pDst + y*dstStrideInBytes;
		vx_uint16 *pLocSrc = pSrc + y * sstride;	// row pointer

		unsigned int x = 1;
#if USE_AVX
		// AVX2: 16 pixels per iteration.
		for (; x + 16 <= dstWidth - 1; x += 16)
		{
			__m256i pix = _mm256_loadu_si256((const __m256i *)(pLocSrc + x));
			__m256i mag = _mm256_srli_epi16(pix, 2);
			__m256i ang = _mm256_and_si256(pix, ang_mask256);

			__m256i n0_a0 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x - 1));
			__m256i n1_a0 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x + 1));
			__m256i n0_a1 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x + 1 - sstride));
			__m256i n1_a1 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x - 1 + sstride));
			__m256i n0_a2 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x     - sstride));
			__m256i n1_a2 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x     + sstride));
			__m256i n0_a3 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x - 1 - sstride));
			__m256i n1_a3 = _mm256_loadu_si256((const __m256i *)(pLocSrc + x + 1 + sstride));

			__m256i m0 = _mm256_cmpeq_epi16(ang, _mm256_setzero_si256());
			__m256i m1 = _mm256_cmpeq_epi16(ang, one256);
			__m256i m2 = _mm256_cmpeq_epi16(ang, two256);
			__m256i m3 = _mm256_cmpeq_epi16(ang, three256);

			__m256i n0 = _mm256_or_si256(
				_mm256_or_si256(_mm256_and_si256(m0, n0_a0), _mm256_and_si256(m1, n0_a1)),
				_mm256_or_si256(_mm256_and_si256(m2, n0_a2), _mm256_and_si256(m3, n0_a3)));
			__m256i n1 = _mm256_or_si256(
				_mm256_or_si256(_mm256_and_si256(m0, n1_a0), _mm256_and_si256(m1, n1_a1)),
				_mm256_or_si256(_mm256_and_si256(m2, n1_a2), _mm256_and_si256(m3, n1_a3)));

			__m256i n0m = _mm256_srli_epi16(n0, 2);
			__m256i n1m = _mm256_srli_epi16(n1, 2);

			__m256i is_max = _mm256_and_si256(_mm256_cmpgt_epi16(mag, n0m), _mm256_cmpgt_epi16(mag, n1m));
			__m256i edge = _mm256_and_si256(is_max, mag);

			__m256i gt_upper = _mm256_cmpgt_epi16(edge, hyst_upper256);
			__m256i gt_lower = _mm256_cmpgt_epi16(edge, hyst_lower256);
			__m256i out_u16 = _mm256_or_si256(
				_mm256_and_si256(gt_upper, k255_256),
				_mm256_and_si256(_mm256_andnot_si256(gt_upper, gt_lower), k127_256));
			// packus on AVX2 is lane-wise: 16 i16 -> 32 u8 with [lane0 lo, lane1 lo, lane0 hi, lane1 hi]
			// Use unpacked store via permute to get sequential 16 bytes.
			__m128i lo = _mm256_castsi256_si128(out_u16);
			__m128i hi = _mm256_extracti128_si256(out_u16, 1);
			__m128i packed = _mm_packus_epi16(lo, hi);
			_mm_storeu_si128((__m128i *)(pOut + x), packed);

			// Stack push: gt_upper has 16 i16 lanes -> need 16-bit mask
			__m128i upper_lo = _mm256_castsi256_si128(gt_upper);
			__m128i upper_hi = _mm256_extracti128_si256(gt_upper, 1);
			__m128i upper_pack = _mm_packs_epi16(upper_lo, upper_hi);
			int upper_mask = _mm_movemask_epi8(upper_pack);
			while (upper_mask)
			{
				int b = __builtin_ctz(upper_mask);
				upper_mask &= upper_mask - 1;
				if (pxyStack < pxyStackEnd)
				{
					pxyStack->x = (vx_uint16)(x + b);
					pxyStack->y = (vx_uint16)y;
					pxyStack++;
				}
			}
		}
#endif
		for (; x + 8 <= dstWidth - 1; x += 8)
		{
			__m128i pix    = _mm_loadu_si128((const __m128i *)(pLocSrc + x));
			__m128i mag    = _mm_srli_epi16(pix, 2);
			__m128i ang    = _mm_and_si128(pix, ang_mask);

			// Pre-load all four possible (n0, n1) neighbor pair candidates.
			__m128i n0_a0 = _mm_loadu_si128((const __m128i *)(pLocSrc + x - 1));
			__m128i n1_a0 = _mm_loadu_si128((const __m128i *)(pLocSrc + x + 1));
			__m128i n0_a1 = _mm_loadu_si128((const __m128i *)(pLocSrc + x + 1 - sstride));
			__m128i n1_a1 = _mm_loadu_si128((const __m128i *)(pLocSrc + x - 1 + sstride));
			__m128i n0_a2 = _mm_loadu_si128((const __m128i *)(pLocSrc + x     - sstride));
			__m128i n1_a2 = _mm_loadu_si128((const __m128i *)(pLocSrc + x     + sstride));
			__m128i n0_a3 = _mm_loadu_si128((const __m128i *)(pLocSrc + x - 1 - sstride));
			__m128i n1_a3 = _mm_loadu_si128((const __m128i *)(pLocSrc + x + 1 + sstride));

			__m128i m0 = _mm_cmpeq_epi16(ang, _mm_setzero_si128());
			__m128i m1 = _mm_cmpeq_epi16(ang, one_v);
			__m128i m2 = _mm_cmpeq_epi16(ang, two_v);
			__m128i m3 = _mm_cmpeq_epi16(ang, three_v);

			__m128i n0 = _mm_or_si128(_mm_or_si128(_mm_and_si128(m0, n0_a0), _mm_and_si128(m1, n0_a1)),
			                          _mm_or_si128(_mm_and_si128(m2, n0_a2), _mm_and_si128(m3, n0_a3)));
			__m128i n1 = _mm_or_si128(_mm_or_si128(_mm_and_si128(m0, n1_a0), _mm_and_si128(m1, n1_a1)),
			                          _mm_or_si128(_mm_and_si128(m2, n1_a2), _mm_and_si128(m3, n1_a3)));

			__m128i n0m = _mm_srli_epi16(n0, 2);
			__m128i n1m = _mm_srli_epi16(n1, 2);

			__m128i is_max = _mm_and_si128(_mm_cmpgt_epi16(mag, n0m), _mm_cmpgt_epi16(mag, n1m));
			__m128i edge = _mm_and_si128(is_max, mag);

			__m128i gt_upper = _mm_cmpgt_epi16(edge, hyst_upper_v);
			__m128i gt_lower = _mm_cmpgt_epi16(edge, hyst_lower_v);

			__m128i out_u16 = _mm_or_si128(_mm_and_si128(gt_upper, k255),
			                                _mm_and_si128(_mm_andnot_si128(gt_upper, gt_lower), k127));
			__m128i out_u8  = _mm_packus_epi16(out_u16, out_u16);
			_mm_storel_epi64((__m128i *)(pOut + x), out_u8);

			// Stack push for lanes where edge > hyst_upper. movemask of an 8-lane
			// i16 packed compare gives one byte per lane.
			__m128i upper_pack = _mm_packs_epi16(gt_upper, gt_upper);
			int upper_mask = _mm_movemask_epi8(upper_pack) & 0xFF;
			while (upper_mask)
			{
				int b = __builtin_ctz(upper_mask);
				upper_mask &= upper_mask - 1;
				if (pxyStack < pxyStackEnd)
				{
					pxyStack->x = (vx_uint16)(x + b);
					pxyStack->y = (vx_uint16)y;
					pxyStack++;
				}
			}
		}

		// Scalar tail (final 1-7 pixels plus the original loop boundary at dstWidth-1).
		vx_uint16 *pLocSrcS = pSrc + y * sstride + x;
		for (; x < dstWidth - 1; x++, pLocSrcS++)
		{
			vx_int32 edge;
			int mag = (pLocSrcS[0] >> 2);
			int ang = pLocSrcS[0] & 3;
			int offset0 = n_offset[ang][0][1] * (int)sstride + n_offset[ang][0][0];
			int offset1 = n_offset[ang][1][1] * (int)sstride + n_offset[ang][1][0];
			edge = ((mag >(pLocSrcS[offset0] >> 2)) && (mag >(pLocSrcS[offset1] >> 2))) ? mag : 0;
			if (edge > hyst_upper) {
				pOut[x] = (vx_uint8)255;
				if (pxyStack < pxyStackEnd) {
					pxyStack->x = (vx_uint16)x;
					pxyStack->y = (vx_uint16)y;
					pxyStack++;
				}
			} else if (edge <= hyst_lower) {
				pOut[x] = 0;
			} else {
				pOut[x] = 127;
			}
		}
	}
	*pxyStackTop = (vx_uint32)(pxyStack - xyStack);
	return AGO_SUCCESS;
}

int HafCpu_CannyEdgeTrace_U8_U8XY
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32              xyStackTop
	)
{
	// Clamp stackTop to capacity to prevent reading beyond buffer
	if (xyStackTop > capacityOfXY) {
		xyStackTop = capacityOfXY;
	}

	ago_coord2d_ushort_t *pxyStack = xyStack + xyStackTop;
	ago_coord2d_ushort_t *pxyStackEnd = xyStack + capacityOfXY;

	while (pxyStack != xyStack){
			pxyStack--;
			vx_uint16 x = pxyStack->x;
			vx_uint16 y = pxyStack->y;
			// look at all the neighbors for strong edge value
		for (int i = 0; i < 8; i++){
			const ago_coord2d_short_t offs = dir_offsets[i];
			vx_int16 x1 = x + offs.x;
			vx_int16 y1 = y + offs.y;

			// Add bounds checking for image access
			if (x1 >= 0 && x1 < (vx_int16)dstWidth &&
			    y1 >= 0 && y1 < (vx_int16)dstHeight) {
				vx_uint8 *pDst = pDstImage + y1*dstImageStrideInBytes + x1;
				if (*pDst == 127)
				{
					*pDst |= 0x80;		// *pDst = 255

					// Check capacity before pushing to stack
					if (pxyStack < pxyStackEnd) {
						*((unsigned *)pxyStack) = (y1<<16)|x1;
						pxyStack++;
					}
				}
			}
		}
	}
	// go through the entire destination and convert all 127 to 0
	const __m128i mm127 = _mm_set1_epi8((char)127);
	for (unsigned int y = 0; y < dstHeight; y++) {
		__m128i * src = (__m128i *)pDstImage;
		vx_uint32 width = (dstWidth + 15) >> 4;

		for (unsigned int x = 0; x < width; x++) {
			__m128i mask;
			__m128i pixels = _mm_load_si128(src);
			mask = _mm_cmpeq_epi8(pixels, mm127);
			pixels = _mm_andnot_si128(mask, pixels);
			_mm_store_si128(src++, pixels);
		}
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_3x3_L2NORM
(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	pSrcImage += srcImageStrideInBytes;
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += dstride;		// don't care about border. start processing from row2
	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 1; y < (int)dstHeight - 1; y++)
	{
		const vx_uint8* srow0 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage;
		const vx_uint8* srow2 = pSrcImage + srcImageStrideInBytes;
		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_add_epi16(s2, s4), _mm_slli_epi16(s3, 1));
			t1 = _mm_sub_epi16(z, t1);
			s0 = _mm_mullo_epi16(t0, t0);
			s1 = _mm_mullo_epi16(t1, t1);
			// unpack to dwords for multiplication
			s2 = _mm_unpackhi_epi16(s0, z);
			s0 = _mm_unpacklo_epi16(s0, z);
			s3 = _mm_unpackhi_epi16(s1, z);
			s1 = _mm_unpacklo_epi16(s1, z);
			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			__m128 f2 = _mm_cvtepi32_ps(s1);
			__m128 f3 = _mm_cvtepi32_ps(s3);
			f0 = _mm_add_ps(f0, f2);
			f1 = _mm_add_ps(f1, f3);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);

			s1 = HafCpu_FastAtan2_Canny_8(t0, t1);
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_5x5_L2NORM
	(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 2 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 2 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 2; y < (int)dstHeight - 2; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage;
		const vx_uint8* srow3 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow4 = pSrcImage + 2 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- (vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2];
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(s1, s3), 2), _mm_mullo_epi16(s2, c6));
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s4));

			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s3, s1), 1);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s4, s0));
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));

			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s7 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			__m128i s8 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));

			__m128i t0 = _mm_slli_epi16(_mm_sub_epi16(s2, s1), 1);
			t0 = _mm_adds_epi16(t0, _mm_sub_epi16(s3, s0));
			__m128i t1 = _mm_slli_epi16(_mm_add_epi16(s5, s7), 2);
			s0 = _mm_mullo_epi16(s6, c6);
			t1 = _mm_add_epi16(t1, _mm_add_epi16(s4, s8));
			t1 = _mm_adds_epi16(t1, s0);
			t1 = _mm_sub_epi16(z, t1);
			// unpack for multiplication
			s0 = _mm_unpacklo_epi16(t0, t1);
			s2 = _mm_unpackhi_epi16(t0, t1);
			s0 = _mm_madd_epi16(s0, s0);
			s2 = _mm_madd_epi16(s2, s2);

			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);

			s1 = HafCpu_FastAtan2_Canny_8(t0, t1);
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- (vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2];
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_7x7_L2NORM
	(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c5 = _mm_set1_epi16(5), c6 = _mm_set1_epi16(6);
	__m128i c15 = _mm_set1_epi16(15), c20 = _mm_set1_epi16(20);
	__m128i clamp = _mm_set1_epi16(0x3FFF);

	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 3 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 3 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 3; y < (int)dstHeight - 3; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 3 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow3 = pSrcImage;
		const vx_uint8* srow4 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow5 = pSrcImage + 2 * srcImageStrideInBytes;
		const vx_uint8* srow6 = pSrcImage + 3 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);
			__m128i s5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow5 + x)), z);
			__m128i s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow6 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s6));
			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s5, s1), 2);
			t0 = _mm_add_epi16(t0, t2);

			t2 = _mm_mullo_epi16(_mm_sub_epi16(s4, s2), c5);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s6, s0));
			t0 = _mm_srai_epi16(t0, 2);
			t1 = _mm_add_epi16(t1, t2);
			t1 = _mm_srai_epi16(t1, 2);

			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 3));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r0 + x + 3));


			__m128i t0 = _mm_slli_epi16(_mm_subs_epi16(s4, s1), 2);
			__m128i t1 = _mm_mullo_epi16(_mm_subs_epi16(s3, s2), c5);
			t0 = _mm_adds_epi16(t0, _mm_subs_epi16(s5, s0));
			t0 = _mm_adds_epi16(t0, t1);

			s0 = _mm_loadu_si128((const __m128i*)(r1 + x - 3));
			s1 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			s5 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x + 3));


			t1 = _mm_adds_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t1 = _mm_adds_epi16(t1, _mm_adds_epi16(s0, s6));
			t1 = _mm_adds_epi16(t1, t2);
			t1 = _mm_subs_epi16(z, t1);
			// unpack for multiplication
			s0 = _mm_unpacklo_epi16(t0, t1);
			s2 = _mm_unpackhi_epi16(t0, t1);
			s0 = _mm_madd_epi16(s0, s0);
			s2 = _mm_madd_epi16(s2, s2);

			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);
			s1 = HafCpu_FastAtan2_Canny_8(t0, t1);
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}