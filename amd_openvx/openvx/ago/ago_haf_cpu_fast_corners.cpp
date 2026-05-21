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

DECL_ALIGN(16) unsigned char dataFastCornersPixelMask[7 * 16] ATTR_ALIGN(16) = {
	  1,   2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
	255, 255,   4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255,
	255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255,
	255, 255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255, 255,
	255, 255, 255, 255, 255,   6, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   4, 255, 255, 255,   0, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   2,   1,   0, 255, 255, 255, 255, 255, 255
};

// Position-dependent FAST boundary shuffle masks. The original layout above is
// applied via _mm_shuffle_epi8 to seven contiguous row loads; the SSE inner loop
// then increments the candidate column by shifting each row right by one byte
// each iteration (7 srli_si128 ops × 8 candidates = 56 unconditional shifts per
// outer iteration). Instead, we precompute an entire 7-mask set for each of the
// 8 candidate offsets (each non-255 byte index in the template is incremented
// by i), letting the per-candidate path read directly from the original row
// loads without any shifts.
DECL_ALIGN(16) unsigned char dataFastCornersPixelMaskAll[8 * 7 * 16] ATTR_ALIGN(16) = {
	// offset 0
	  1,   2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
	255, 255,   4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255,
	255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255,
	255, 255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255, 255,
	255, 255, 255, 255, 255,   6, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   4, 255, 255, 255,   0, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   2,   1,   0, 255, 255, 255, 255, 255, 255,
	// offset 1
	  2,   3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   1,
	255, 255,   5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   1, 255,
	255, 255, 255,   7, 255, 255, 255, 255, 255, 255, 255, 255, 255,   1, 255, 255,
	255, 255, 255, 255,   7, 255, 255, 255, 255, 255, 255, 255,   1, 255, 255, 255,
	255, 255, 255, 255, 255,   7, 255, 255, 255, 255, 255,   1, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   5, 255, 255, 255,   1, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   3,   2,   1, 255, 255, 255, 255, 255, 255,
	// offset 2
	  3,   4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2,
	255, 255,   6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2, 255,
	255, 255, 255,   8, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2, 255, 255,
	255, 255, 255, 255,   8, 255, 255, 255, 255, 255, 255, 255,   2, 255, 255, 255,
	255, 255, 255, 255, 255,   8, 255, 255, 255, 255, 255,   2, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   6, 255, 255, 255,   2, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   4,   3,   2, 255, 255, 255, 255, 255, 255,
	// offset 3
	  4,   5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3,
	255, 255,   7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255,
	255, 255, 255,   9, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255,
	255, 255, 255, 255,   9, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255, 255,
	255, 255, 255, 255, 255,   9, 255, 255, 255, 255, 255,   3, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   7, 255, 255, 255,   3, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   5,   4,   3, 255, 255, 255, 255, 255, 255,
	// offset 4
	  5,   6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   4,
	255, 255,   8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   4, 255,
	255, 255, 255,  10, 255, 255, 255, 255, 255, 255, 255, 255, 255,   4, 255, 255,
	255, 255, 255, 255,  10, 255, 255, 255, 255, 255, 255, 255,   4, 255, 255, 255,
	255, 255, 255, 255, 255,  10, 255, 255, 255, 255, 255,   4, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   8, 255, 255, 255,   4, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   6,   5,   4, 255, 255, 255, 255, 255, 255,
	// offset 5
	  6,   7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   5,
	255, 255,   9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   5, 255,
	255, 255, 255,  11, 255, 255, 255, 255, 255, 255, 255, 255, 255,   5, 255, 255,
	255, 255, 255, 255,  11, 255, 255, 255, 255, 255, 255, 255,   5, 255, 255, 255,
	255, 255, 255, 255, 255,  11, 255, 255, 255, 255, 255,   5, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   9, 255, 255, 255,   5, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   7,   6,   5, 255, 255, 255, 255, 255, 255,
	// offset 6
	  7,   8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   6,
	255, 255,  10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   6, 255,
	255, 255, 255,  12, 255, 255, 255, 255, 255, 255, 255, 255, 255,   6, 255, 255,
	255, 255, 255, 255,  12, 255, 255, 255, 255, 255, 255, 255,   6, 255, 255, 255,
	255, 255, 255, 255, 255,  12, 255, 255, 255, 255, 255,   6, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,  10, 255, 255, 255,   6, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   8,   7,   6, 255, 255, 255, 255, 255, 255,
	// offset 7
	  8,   9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   7,
	255, 255,  11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   7, 255,
	255, 255, 255,  13, 255, 255, 255, 255, 255, 255, 255, 255, 255,   7, 255, 255,
	255, 255, 255, 255,  13, 255, 255, 255, 255, 255, 255, 255,   7, 255, 255, 255,
	255, 255, 255, 255, 255,  13, 255, 255, 255, 255, 255,   7, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,  11, 255, 255, 255,   7, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   9,   8,   7, 255, 255, 255, 255, 255, 255
};

static inline void generateOffset(int srcStride, int * offsets)
{
	offsets[0] = -3 * srcStride;
	offsets[15] = offsets[0] - 1;
	offsets[1] = offsets[0] + 1;
	offsets[2] = -(srcStride << 1) + 2;
	offsets[14] = offsets[2] - 4;
	offsets[3] = -srcStride + 3;
	offsets[13] = offsets[3] - 6;
	offsets[4] = 3;
	offsets[12] = -3;
	offsets[5] = srcStride + 3;
	offsets[11] = offsets[5] - 6;
	offsets[6] = (srcStride << 1) + 2;
	offsets[10] = offsets[6] - 4;
	offsets[7] = 3 * srcStride + 1;
	offsets[8] = offsets[7] - 1;
	offsets[9] = offsets[8] - 1;

	return;
}

static inline void generateMasks_C(unsigned char * src, int srcStride, int* offsets, short t, int mask[2])
{
	mask[0] = 0;
	mask[1] = 0;
	int iterMask = 1;

	// Early exit conditions
	if ((abs((short)src[0] - (short)src[offsets[0]]) < t) && (abs((short)src[0] - (short)src[offsets[8]]) < t))					// Pixels 1 and 9 within t of the candidate
		return;
	if ((abs((short)src[0] - (short)src[offsets[4]]) < t) && (abs((short)src[0] - (short)src[offsets[12]]) < t))				// Pixels 5 and 13 within t of the candidate
		return;

	// Check for I_p + t
	short cand = (short)(*src) + t;
	for (int i = 0; i < 16; i++)
	{
		if ((short)src[offsets[i]] > cand)
			mask[0] |= iterMask;
		iterMask <<= 1;
	}

	// Check for I_p - t
	iterMask = 1;
	cand = (short)(*src) - t;
	for (int i = 0; i < 16; i++)
	{
		if ((short)src[offsets[i]] < cand)
			mask[1] |= iterMask;
		iterMask <<= 1;
	}
}

static inline bool isCorner(int mask[2])
{
	int cornerMask = 0x1FF;									// Nine 1's in the LSB

	if (mask[0] || mask[1])
	{
		mask[0] = mask[0] | (mask[0] << 16);
		mask[1] = mask[1] | (mask[1] << 16);

		for (int i = 0; i < 16; i++)
		{
			if (((mask[0] & cornerMask) == cornerMask) || ((mask[1] & cornerMask) == cornerMask))
				return true;
			mask[0] >>= 1;
			mask[1] >>= 1;
		}
	}
	
	return false;
}

static inline bool isCorner(int mask)
{
	int cornerMask = 0x1FF;									// Nine 1's in the LSB

	if (mask)
	{
		mask = mask | (mask << 16);
		for (int i = 0; i < 16; i++)
		{
			if ((mask & cornerMask) == cornerMask)
				return true;
			mask >>= 1;
		}
	}
	return false;
}

static inline bool isCornerPlus(short candidate, short * boundary, short t)
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

static inline bool isCornerPlus_SSE(__m128i candidate, __m128i boundary, short t)
{
	__m128i boundaryH = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());								// Boundary 8..15 (words)
	__m128i boundaryL = _mm_cvtepu8_epi16(boundary);													// Boundary 0..7 (words)
	__m128i threshold = _mm_set1_epi16(t);

	short cand = M128I(candidate).m128i_i16[0];

	// Early exit conditions
	if ((abs(cand - M128I(boundaryL).m128i_i16[0]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[0]) < t))			// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(cand - M128I(boundaryL).m128i_i16[4]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[4]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate = _mm_add_epi16(candidate, threshold);
	boundaryH = _mm_cmpgt_epi16(boundaryH, candidate);
	boundaryL = _mm_cmpgt_epi16(boundaryL, candidate);
	boundaryL = _mm_packs_epi16(boundaryL, boundaryH);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(boundaryL);

	return isCorner(mask);
}

static inline bool isCornerMinus(short candidate, short * boundary, short t)
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

static inline bool isCornerMinus_SSE(__m128i candidate, __m128i boundary, short t)
{
	__m128i boundaryH = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());								// Boundary 8..15 (words)
	__m128i boundaryL = _mm_cvtepu8_epi16(boundary);													// Boundary 0..7 (words)
	__m128i threshold = _mm_set1_epi16(t);

	short cand = M128I(candidate).m128i_i16[0];

	// Early exit conditions
	if ((abs(cand - M128I(boundaryL).m128i_i16[0]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[0]) < t))			// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(cand - M128I(boundaryL).m128i_i16[4]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[4]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate = _mm_sub_epi16(candidate, threshold);
	boundaryH = _mm_cmplt_epi16(boundaryH, candidate);
	boundaryL = _mm_cmplt_epi16(boundaryL, candidate);
	boundaryL = _mm_packs_epi16(boundaryL, boundaryH);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(boundaryL);

	return isCorner(mask);
}

static inline bool checkForCornerAndGetStrength(unsigned char * src, int* offsets, short t, short * strength)
{
	// Early exit conditions
	if ((abs((short)src[0] - (short)src[offsets[0]]) < t) && (abs((short)src[0] - (short)src[offsets[8]]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs((short)src[0] - (short)src[offsets[4]]) < t) && (abs((short)src[0] - (short)src[offsets[12]]) < t))				// Pixels 5 and 13 within t of the candidate
		return false;

	// Get boundary
	unsigned char boundary[16];
	for (int i = 0; i < 16; i++)
		boundary[i] = src[offsets[i]];

	// Direct strength computation by 9-element sliding-window min/max over cyclic
	// 16-element boundary (matches SIMD path above; avoids per-pixel binary search).
	int p = (int)src[0];
	int brighter_max = 0;
	int darker_min   = 255;
	for (int s = 0; s < 16; s++)
	{
		int m = 255;
		int M = 0;
		for (int k = 0; k < 9; k++)
		{
			int v = boundary[(s + k) & 15];
			if (v < m) m = v;
			if (v > M) M = v;
		}
		if (m > brighter_max) brighter_max = m;
		if (M < darker_min)   darker_min   = M;
	}

	int strength_pos = (brighter_max > p) ? (brighter_max - p - 1) : 0;
	int strength_neg = (darker_min  < p) ? (p - darker_min  - 1) : 0;

	bool is_brighter = (strength_pos >= (int)t);
	bool is_darker   = (strength_neg >= (int)t);

	if (is_brighter || is_darker)
	{
		int s_pos = is_brighter ? strength_pos : 0;
		int s_neg = is_darker   ? strength_neg : 0;
		*strength = (short)((s_pos > s_neg) ? s_pos : s_neg);
		return true;
	}
	return false;
}

bool isCorner_SSE(unsigned char pixel, __m128i boundary, __m128i t)
{
	// Check for boundary > pixel + t
	__m128i cand = _mm_set1_epi16((short)pixel);									// The candidate pixel
	cand = _mm_add_epi16(cand, t);													// Pixel + t

	__m128i temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());				// Boundary 8..15 (words)
	__m128i temp1 = _mm_cvtepu8_epi16(boundary);									// Boundary 0..7 (words)

	temp0 = _mm_cmpgt_epi16(temp0, cand);
	temp1 = _mm_cmpgt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(temp1);
	int plusMask = mask | (mask << 16);

	// Check for boundary > pixel - t
	cand = _mm_sub_epi16(cand, t);													// pixel + t - t = pixel
	cand = _mm_sub_epi16(cand, t);													// pixel - t

	temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());						// Boundary 8..15 (words)
	temp1 = _mm_cvtepu8_epi16(boundary);											// Boundary 0..7 (words)

	temp0 = _mm_cmplt_epi16(temp0, cand);
	temp1 = _mm_cmplt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	mask = _mm_movemask_epi8(temp1);
	int minusMask = mask | (mask << 16);

	if (plusMask || minusMask)
	{
		int cornerMask = 0x1FF;														// Nine 1's in the LSB

		for (int i = 0; i < 16; i++)
		{
			if (((plusMask & cornerMask) == cornerMask) || ((minusMask & cornerMask) == cornerMask))
				return true;
			plusMask >>= 1;
			minusMask >>= 1;
		}
	}

	return false;
}

static inline bool checkForCornerAndGetStrength_SSE(unsigned char pixel, __m128i boundary, short threshold, short * strength)
{
	// Direct strength computation: for each of the 16 candidate 9-contiguous-pixel windows
	// of the FAST boundary, compute the min (for brighter corners) and max (for darker
	// corners). The corner exists at the largest threshold such that some window is
	// uniformly > pixel + threshold (brighter) or < pixel - threshold (darker). Using
	// _mm_alignr_epi8 to produce cyclically-shifted copies of the boundary gives 16
	// window mins/maxes with a handful of SIMD min/max ops, removing the ~8-iteration
	// binary search and per-iteration isCornerPlus_SSE / isCornerMinus_SSE that used to
	// dominate the inner loop.
	__m128i b = boundary;
	__m128i sh1 = _mm_alignr_epi8(b, b, 1);
	__m128i sh2 = _mm_alignr_epi8(b, b, 2);
	__m128i sh3 = _mm_alignr_epi8(b, b, 3);
	__m128i sh4 = _mm_alignr_epi8(b, b, 4);
	__m128i sh5 = _mm_alignr_epi8(b, b, 5);
	__m128i sh6 = _mm_alignr_epi8(b, b, 6);
	__m128i sh7 = _mm_alignr_epi8(b, b, 7);
	__m128i sh8 = _mm_alignr_epi8(b, b, 8);

	__m128i min_w = _mm_min_epu8(_mm_min_epu8(_mm_min_epu8(b, sh1), _mm_min_epu8(sh2, sh3)),
	                              _mm_min_epu8(_mm_min_epu8(sh4, sh5), _mm_min_epu8(_mm_min_epu8(sh6, sh7), sh8)));
	__m128i max_w = _mm_max_epu8(_mm_max_epu8(_mm_max_epu8(b, sh1), _mm_max_epu8(sh2, sh3)),
	                              _mm_max_epu8(_mm_max_epu8(sh4, sh5), _mm_max_epu8(_mm_max_epu8(sh6, sh7), sh8)));

	// Horizontal max of min_w (= brightest minimum across all 16 windows)
	__m128i a = min_w;
	a = _mm_max_epu8(a, _mm_srli_si128(a, 8));
	a = _mm_max_epu8(a, _mm_srli_si128(a, 4));
	a = _mm_max_epu8(a, _mm_srli_si128(a, 2));
	a = _mm_max_epu8(a, _mm_srli_si128(a, 1));
	int brighter_max = _mm_extract_epi8(a, 0) & 0xFF;

	// Horizontal min of max_w (= darkest maximum across all 16 windows)
	a = max_w;
	a = _mm_min_epu8(a, _mm_srli_si128(a, 8));
	a = _mm_min_epu8(a, _mm_srli_si128(a, 4));
	a = _mm_min_epu8(a, _mm_srli_si128(a, 2));
	a = _mm_min_epu8(a, _mm_srli_si128(a, 1));
	int darker_min = _mm_extract_epi8(a, 0) & 0xFF;

	int p = (int)pixel;
	int strength_pos = (brighter_max > p) ? (brighter_max - p - 1) : 0;
	int strength_neg = (darker_min  < p) ? (p - darker_min  - 1) : 0;

	bool is_brighter = (strength_pos >= (int)threshold);
	bool is_darker   = (strength_neg >= (int)threshold);

	if (is_brighter || is_darker)
	{
		int s_pos = is_brighter ? strength_pos : 0;
		int s_neg = is_darker   ? strength_neg : 0;
		*strength = (short)((s_pos > s_neg) ? s_pos : s_neg);
		return true;
	}
	return false;
}

int HafCpu_FastCorners_XY_U8_NoSupression
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32       srcWidth,
		vx_uint32       srcHeight,
		vx_uint8      * pSrcImage,
		vx_uint32       srcImageStrideInBytes,
		vx_float32      strength_threshold
	)
{
	unsigned char * pLocalSrc;
	int srcStride = (int)srcImageStrideInBytes;
	vx_uint32 cornerCount = 0;
	short t = (short)floorf(strength_threshold);
	
	pSrcImage += (srcStride * 3);														// Leave first three rows

	int alignedWidth = (int)srcWidth & ~7;
	int postfixWidth = (int)srcWidth & 7;

	// Generate offsets for C code if necessary
	int neighbor_offset[16] = { 0 };
	if (postfixWidth)
		generateOffset(srcStride, neighbor_offset);

	__m128i zeromask = _mm_setzero_si128();

	for (int height = 0; height < (int)(srcHeight - 6); height++)
	{
		pLocalSrc = (unsigned char *) pSrcImage;
		int width = 0;
		
		for (int x = 0; x < (alignedWidth >> 3); x++)
		{
			__m128i rowMinus3, rowMinus2, rowMinus1, row, rowPlus1, rowPlus2, rowPlus3;
			__m128i thresh = _mm_set1_epi16(t);

			// Check for early escape based on pixels 1 and 9 around the candidate
			rowMinus3 = _mm_loadu_si128((__m128i *)(pLocalSrc - 3 * srcStride - 1));
			rowMinus2 = _mm_srli_si128(rowMinus3, 1);									// row - 3: Pixels 0..7 in lower 7 bytes
			rowMinus2 = _mm_cvtepu8_epi16(rowMinus2);

			row = _mm_loadu_si128((__m128i *)(pLocalSrc - 3));
			rowMinus1 = _mm_srli_si128(row, 3);											// row: Pixels 0..7 in lower 7 bytes
			rowMinus1 = _mm_cvtepu8_epi16(rowMinus1);

			rowPlus3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 3 * srcStride - 1));
			rowPlus2 = _mm_srli_si128(rowPlus3, 1);										// row + 3: Pixels 0..7 in lower 7 bytes
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowPlus1 = _mm_loadu_si128((__m128i *)(pLocalSrc + srcStride - 3));

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip = _mm_movemask_epi8(rowMinus2);
			maskSkip &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel

			// Check for early escape based on pixels 12 and 4 around the candidate
			rowMinus2 = _mm_cvtepu8_epi16(row);
			rowPlus2 = _mm_srli_si128(row, 6);
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus1 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 3));

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip1 = _mm_movemask_epi8(rowMinus2);
			rowMinus2 = _mm_loadu_si128((__m128i *)(pLocalSrc - (srcStride + srcStride) - 2));

			maskSkip1 &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel
			rowPlus2 = _mm_loadu_si128((__m128i *)(pLocalSrc + (srcStride + srcStride) - 2));

			maskSkip |= maskSkip1;

			// Check for corners in the eight pixels.
			// Use precomputed position-dependent masks to avoid per-iteration row shifts.
			if (maskSkip != 0xFFFF)
			{
				for (int i = 0; i < 8; i++, maskSkip >>= 2)
				{
					if (maskSkip & 1) continue;

					__m128i * tbl = (__m128i *) (dataFastCornersPixelMaskAll + i * 7 * 16);
					__m128i boundary = _mm_shuffle_epi8(rowMinus3, _mm_load_si128(tbl + 0));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowMinus2, _mm_load_si128(tbl + 1)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowMinus1, _mm_load_si128(tbl + 2)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(row,       _mm_load_si128(tbl + 3)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus1,  _mm_load_si128(tbl + 4)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus2,  _mm_load_si128(tbl + 5)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus3,  _mm_load_si128(tbl + 6)));

					if (isCorner_SSE(pLocalSrc[i], boundary, thresh))
					{
						if (cornerCount < capacityOfDstCorner)
						{
							dstCorner[cornerCount].y = height + 3;
							dstCorner[cornerCount].x = width + i;
							dstCorner[cornerCount].strength = strength_threshold;			// Undefined as per the 1.0.1 spec
							dstCorner[cornerCount].scale = 0;
							dstCorner[cornerCount].orientation = 0;
							dstCorner[cornerCount].error = 0;
							dstCorner[cornerCount++].tracking_status = 1;
						}
						else
							cornerCount++;
					}
				}
			}

			width += 8;
			pLocalSrc += 8;
		}

		for (int x = 0; x < postfixWidth; x++)
		{
			int masks[2];
			generateMasks_C(pLocalSrc, srcStride, neighbor_offset, t, masks);
			if (isCorner(masks))
			{
				if (cornerCount < capacityOfDstCorner)
				{
					dstCorner[cornerCount].y = height + 3;
					dstCorner[cornerCount].x = width;
					dstCorner[cornerCount].strength = strength_threshold;			// Undefined as per the 1.0.1 spec
					dstCorner[cornerCount].scale = 0;
					dstCorner[cornerCount].orientation = 0;
					dstCorner[cornerCount].error = 0;
					dstCorner[cornerCount++].tracking_status = 1;
				}
				else
					cornerCount++;
			}
			width++;
			pLocalSrc++;
		}
		pSrcImage += srcStride;
	}

	*pDstCornerCount = cornerCount;
	return AGO_SUCCESS;
}

int HafCpu_FastCorners_XY_U8_Supression
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32       srcWidth,
		vx_uint32       srcHeight,
		vx_uint8      * pSrcImage,
		vx_uint32       srcImageStrideInBytes,
		vx_float32      strength_threshold,
		vx_uint8	  * pScratch
	)
{
	unsigned char * pLocalSrc;
	int srcStride = (int)srcImageStrideInBytes;
	vx_uint32 cornerCount = 0;
	short t = (short)floorf(strength_threshold);

	pSrcImage += (srcStride * 3) + 3;														// Leave first three rows and start from the third pixel

	int alignedWidth = (int)(srcWidth - 6) & ~7;
	int postfixWidth = (int)(srcWidth - 6) & 7;

	// Generate offsets for C code if necessary
	int neighbor_offset[16] = { 0 };
	if (postfixWidth)
		generateOffset(srcStride, neighbor_offset);

	memset(pScratch, 0, sizeof(vx_uint8) * srcWidth * srcHeight);

	for (int height = 0; height < (int)(srcHeight - 6); height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 3;

		for (int x = 0; x < (alignedWidth >> 3); x++)
		{
			__m128i rowMinus3, rowMinus2, rowMinus1, row, rowPlus1, rowPlus2, rowPlus3;
			__m128i thresh = _mm_set1_epi16(t);

			// Load all 7 rows in their final form for boundary shuffles.
			rowMinus3 = _mm_loadu_si128((__m128i *)(pLocalSrc - 3 * srcStride - 1));
			rowMinus2 = _mm_loadu_si128((__m128i *)(pLocalSrc - (srcStride + srcStride) - 2));
			rowMinus1 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 3));
			row       = _mm_loadu_si128((__m128i *)(pLocalSrc - 3));
			rowPlus1  = _mm_loadu_si128((__m128i *)(pLocalSrc + srcStride - 3));
			rowPlus2  = _mm_loadu_si128((__m128i *)(pLocalSrc + (srcStride + srcStride) - 2));
			rowPlus3  = _mm_loadu_si128((__m128i *)(pLocalSrc + 3 * srcStride - 1));

			// FAST-9 adjacent-cardinal-pair pre-filter. A 9-pixel contiguous arc
			// of brighter (or darker) pixels around the 16-pixel boundary must
			// always contain at least one pair of adjacent cardinal points
			// {(p1,p5),(p5,p9),(p9,p13),(p13,p1)} that are all brighter (or all
			// darker). Conversely, if no adjacent cardinal pair satisfies the
			// polarity-aware test, the candidate cannot be a FAST-9 corner. This
			// is strictly tighter than the original 2-of-2 axial-pair check
			// (it also rejects cases where a single bright/dark and an opposing
			// dark/bright neighbour would have slipped through).
			__m128i cand   = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(pLocalSrc)));
			__m128i cand_p = _mm_add_epi16(cand, thresh);
			__m128i cand_m = _mm_sub_epi16(cand, thresh);
			// Boundary pixel candidates for 8 candidates at columns [x..x+7]. The
			// rows were loaded with a -1 byte offset for rowMinus3/rowPlus3 and a
			// -3 byte offset for `row`, so the candidate column is at byte 1
			// (rowMinus3/rowPlus3) / byte 3 (row). Byte 0 of `row` is left-3,
			// byte 6 of `row` is right+3.
			__m128i p1  = _mm_cvtepu8_epi16(_mm_srli_si128(rowMinus3, 1));  // top:    (x, y-3)
			__m128i p9  = _mm_cvtepu8_epi16(_mm_srli_si128(rowPlus3, 1));   // bottom: (x, y+3)
			__m128i p5  = _mm_cvtepu8_epi16(_mm_srli_si128(row, 6));        // right:  (x+3, y)
			__m128i p13 = _mm_cvtepu8_epi16(row);                            // left:   (x-3, y)
			__m128i b1  = _mm_cmpgt_epi16(p1,  cand_p);
			__m128i b5  = _mm_cmpgt_epi16(p5,  cand_p);
			__m128i b9  = _mm_cmpgt_epi16(p9,  cand_p);
			__m128i b13 = _mm_cmpgt_epi16(p13, cand_p);
			__m128i d1  = _mm_cmpgt_epi16(cand_m, p1);
			__m128i d5  = _mm_cmpgt_epi16(cand_m, p5);
			__m128i d9  = _mm_cmpgt_epi16(cand_m, p9);
			__m128i d13 = _mm_cmpgt_epi16(cand_m, p13);
			__m128i pair_b = _mm_or_si128(
				_mm_or_si128(_mm_and_si128(b1,  b5),  _mm_and_si128(b5,  b9)),
				_mm_or_si128(_mm_and_si128(b9,  b13), _mm_and_si128(b13, b1)));
			__m128i pair_d = _mm_or_si128(
				_mm_or_si128(_mm_and_si128(d1,  d5),  _mm_and_si128(d5,  d9)),
				_mm_or_si128(_mm_and_si128(d9,  d13), _mm_and_si128(d13, d1)));
			__m128i pass = _mm_or_si128(pair_b, pair_d);
			// `pass` lanes are 0xFFFF if the candidate may be a corner, else 0x0000.
			// `_mm_movemask_epi8` yields a 16-bit mask with two bits per lane,
			// matching the original `maskSkip` packing (bit pair set => keep), so
			// invert to obtain the skip mask consumed by the per-pixel loop below.
			int passMask = _mm_movemask_epi8(pass);
			int maskSkip = (~passMask) & 0xFFFF;

			// Check for corners in the eight pixels.
			// Use precomputed position-dependent masks to avoid the per-iteration
			// row shifts that previously cost 7 srli_si128 ops × 8 candidates.
			if (maskSkip != 0xFFFF)
			{
				for (int i = 0; i < 8; i++, maskSkip >>= 2)
				{
					if (maskSkip & 1) continue;

					__m128i * tbl = (__m128i *) (dataFastCornersPixelMaskAll + i * 7 * 16);
					__m128i boundary = _mm_shuffle_epi8(rowMinus3, _mm_load_si128(tbl + 0));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowMinus2, _mm_load_si128(tbl + 1)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowMinus1, _mm_load_si128(tbl + 2)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(row,       _mm_load_si128(tbl + 3)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus1,  _mm_load_si128(tbl + 4)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus2,  _mm_load_si128(tbl + 5)));
					boundary = _mm_or_si128(boundary, _mm_shuffle_epi8(rowPlus3,  _mm_load_si128(tbl + 6)));

					short strength = 0;
					if (checkForCornerAndGetStrength_SSE(pLocalSrc[i], boundary, t, &strength))
						pScratch[(height + 3) * srcWidth + width + i] = (vx_uint8)strength;
				}
			}

			width += 8;
			pLocalSrc += 8;
		}

		for (int x = 0; x < postfixWidth; x++)
		{
			short strength = 0;
			if (checkForCornerAndGetStrength(pLocalSrc, neighbor_offset, t, &strength))
				pScratch[(height + 3) * srcWidth + width] = (vx_uint8)strength;

			width++;
			pLocalSrc++;
		}
		pSrcImage += srcStride;
	}

	// Non-max supression
	pScratch += (3 * srcWidth + 3);
	cornerCount = 0;
	const int nmsW = (int)srcWidth - 6;
	for (int height = 0; height < int(srcHeight - 6); height++)
	{
		int width = 0;
#if USE_AVX
		// SIMD prefilter: scratch is mostly 0 in FAST output, so reject blocks of
		// 16 columns at once when no candidate could be a local max.
		const __m128i zero128 = _mm_setzero_si128();
		for (; width + 16 <= nmsW; width += 16)
		{
			__m128i cand = _mm_loadu_si128((const __m128i *)pScratch);
			__m128i any_nz = _mm_cmpgt_epi8(cand, zero128); // unsigned compare via signed >0 for 0..127 works generally; refine below
			// For unsigned bytes, _mm_cmpgt_epi8 treats sign bit as MSB, but FAST strength is in 0..127 typically.
			// To be safe, use _mm_max_epu8 to keep semantics: any_nz mask: cand != 0
			__m128i is_nz = _mm_cmpeq_epi8(_mm_max_epu8(cand, zero128), zero128); // 0xFF if cand==0
			int zero_mask = _mm_movemask_epi8(is_nz);
			if (zero_mask == 0xFFFF) { pScratch += 16; continue; }
			// Compare against the 8 neighbors with the correct >=/> rules.
			__m128i ul = _mm_loadu_si128((const __m128i *)(pScratch - srcWidth - 1));
			__m128i uc = _mm_loadu_si128((const __m128i *)(pScratch - srcWidth));
			__m128i ur = _mm_loadu_si128((const __m128i *)(pScratch - srcWidth + 1));
			__m128i ll = _mm_loadu_si128((const __m128i *)(pScratch - 1));
			__m128i rr = _mm_loadu_si128((const __m128i *)(pScratch + 1));
			__m128i dl = _mm_loadu_si128((const __m128i *)(pScratch + srcWidth - 1));
			__m128i dc = _mm_loadu_si128((const __m128i *)(pScratch + srcWidth));
			__m128i dr = _mm_loadu_si128((const __m128i *)(pScratch + srcWidth + 1));
			// cand >= n iff cand == max(cand, n).
			__m128i geq_ul = _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, ul));
			__m128i geq_uc = _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, uc));
			__m128i geq_ur = _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, ur));
			__m128i geq_ll = _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, ll));
			// cand > n iff cand != n AND cand >= n.
			__m128i gt_rr = _mm_andnot_si128(_mm_cmpeq_epi8(cand, rr), _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, rr)));
			__m128i gt_dl = _mm_andnot_si128(_mm_cmpeq_epi8(cand, dl), _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, dl)));
			__m128i gt_dc = _mm_andnot_si128(_mm_cmpeq_epi8(cand, dc), _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, dc)));
			__m128i gt_dr = _mm_andnot_si128(_mm_cmpeq_epi8(cand, dr), _mm_cmpeq_epi8(cand, _mm_max_epu8(cand, dr)));
			__m128i m = _mm_and_si128(geq_ul, geq_uc);
			m = _mm_and_si128(m, geq_ur);
			m = _mm_and_si128(m, geq_ll);
			m = _mm_and_si128(m, gt_rr);
			m = _mm_and_si128(m, gt_dl);
			m = _mm_and_si128(m, gt_dc);
			m = _mm_and_si128(m, gt_dr);
			m = _mm_andnot_si128(is_nz, m); // cand must also be != 0
			unsigned int ok_mask = (unsigned int)_mm_movemask_epi8(m);
			alignas(16) vx_uint8 cand_arr[16];
			if (ok_mask)
				_mm_store_si128((__m128i *)cand_arr, cand);
			while (ok_mask)
			{
				int lane = agoCtz32(ok_mask);
				ok_mask &= ok_mask - 1u;
				if (cornerCount < capacityOfDstCorner)
				{
					dstCorner[cornerCount].x = (vx_int32)(width + lane + 3);
					dstCorner[cornerCount].y = (vx_int32)(height + 3);
					dstCorner[cornerCount].strength = (vx_float32)cand_arr[lane];
					dstCorner[cornerCount].scale = 0;
					dstCorner[cornerCount].orientation = 0;
					dstCorner[cornerCount].error = 0;
					dstCorner[cornerCount++].tracking_status = 1;
				}
				else
					cornerCount++;
			}
			pScratch += 16;
		}
#endif
		for (; width < nmsW; width++)
		{
			vx_uint8 * prev = pScratch - srcWidth;
			vx_uint8 * nxt = pScratch + srcWidth;
			vx_uint8 cand = *pScratch;
			if (cand && (cand >= *(prev - 1)) && (cand >= *prev) && (cand >= *(prev + 1))
				&& (cand >= *(pScratch - 1)) && (cand > *(pScratch + 1))
				&& (cand > *(nxt - 1)) && (cand > *nxt) && (cand > *(nxt + 1)))
			{
				if (cornerCount < capacityOfDstCorner)
				{
					dstCorner[cornerCount].x = (vx_int32)(width + 3);
					dstCorner[cornerCount].y = (vx_int32)(height + 3);
					dstCorner[cornerCount].strength = (vx_float32)cand;
					dstCorner[cornerCount].scale = 0;
					dstCorner[cornerCount].orientation = 0;
					dstCorner[cornerCount].error = 0;
					dstCorner[cornerCount++].tracking_status = 1;
				}
				else
					cornerCount++;
			}
			pScratch++;
		}
		pScratch += 6;
	}
	*pDstCornerCount = cornerCount;
	return AGO_SUCCESS;
}


int HafCpu_FastCornerMerge_XY_XY
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32		numSrcCornerBuffers,
		vx_keypoint_t * pSrcCorners[],
		vx_uint32       numSrcCorners[]
	)
{
	int dstCount = 0;
	int srcCount;
	vx_keypoint_t * srcList;

	for (int i = 0; i < (int)numSrcCornerBuffers; i++)
	{
		srcCount = numSrcCorners[i];
		srcList = pSrcCorners[i];

		while (srcCount)
		{
			*dstCorner++ = *srcList++;
			dstCount++;
			srcCount--;
			if (dstCount >(int) capacityOfDstCorner)
			{
				*pDstCornerCount = (vx_uint32)(dstCount - 1);
				return AGO_SUCCESS;
			}
		}
	}

	*pDstCornerCount = (vx_uint32)(dstCount - 1);
	return AGO_SUCCESS;
}
