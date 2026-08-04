/*
 * Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * AGO_HAF_CPU_LOGICAL_PARALLEL.CPP
 *
 * Row-based parallel implementations of logical/bitwise kernels.
 */

#include "ago_internal.h"
#include "ago_parallel.h"

// ============================================================================
// Parallel And U8 = U8 & U8
// ============================================================================

#if USE_AVX

typedef struct {
    vx_uint32 width;
    vx_uint32 height;
    vx_uint8* dst;
    vx_uint32 dst_stride;
    vx_uint8* src1;
    vx_uint32 src1_stride;
    vx_uint8* src2;
    vx_uint32 src2_stride;
} Logical_U8_Args_t;

static void And_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Logical_U8_Args_t* a = (Logical_U8_Args_t*)user_data;
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        vx_uint8* pSrc1 = a->src1 + y * a->src1_stride;
        vx_uint8* pSrc2 = a->src2 + y * a->src2_stride;
        vx_uint8* pDst = a->dst + y * a->dst_stride;
        
        vx_uint32 width = 0;
        // Process 128 bytes at a time
        for (; width + 128 <= a->width; width += 128) {
            __m256i a0 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i a1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 32));
            __m256i a2 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 64));
            __m256i a3 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 96));
            __m256i b0 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            __m256i b1 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 32));
            __m256i b2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 64));
            __m256i b3 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 96));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_and_si256(a0, b0));
            _mm256_storeu_si256((__m256i *)(pDst + width + 32), _mm256_and_si256(a1, b1));
            _mm256_storeu_si256((__m256i *)(pDst + width + 64), _mm256_and_si256(a2, b2));
            _mm256_storeu_si256((__m256i *)(pDst + width + 96), _mm256_and_si256(a3, b3));
        }
        // Process remaining 32-byte chunks
        for (; width + 32 <= a->width; width += 32) {
            __m256i pixels1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i pixels2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_and_si256(pixels1, pixels2));
        }
        // Scalar remainder
        for (; width < a->width; width++) {
            pDst[width] = pSrc1[width] & pSrc2[width];
        }
    }
}

#endif // USE_AVX

int HafCpu_And_U8_U8U8_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    vx_uint8    * pSrcImage2,
    vx_uint32     srcImage2StrideInBytes
) {
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        return HafCpu_And_U8_U8U8(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                   pSrcImage1, srcImage1StrideInBytes,
                                   pSrcImage2, srcImage2StrideInBytes);
    }
    
#if USE_AVX
    Logical_U8_Args_t args = {
        .width = dstWidth,
        .height = dstHeight,
        .dst = pDstImage,
        .dst_stride = dstImageStrideInBytes,
        .src1 = pSrcImage1,
        .src1_stride = srcImage1StrideInBytes,
        .src2 = pSrcImage2,
        .src2_stride = srcImage2StrideInBytes
    };
    
    AgoParallelForRows(dstHeight, And_U8_Row_AVX, &args);
#else
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < (int)dstHeight; y++) {
        vx_uint8* pSrc1 = pSrcImage1 + y * srcImage1StrideInBytes;
        vx_uint8* pSrc2 = pSrcImage2 + y * srcImage2StrideInBytes;
        vx_uint8* pDst = pDstImage + y * dstImageStrideInBytes;
        for (vx_uint32 x = 0; x < dstWidth; x++) {
            pDst[x] = pSrc1[x] & pSrc2[x];
        }
    }
#endif
    
    return AGO_SUCCESS;
}

// ============================================================================
// Parallel Or U8 = U8 | U8
// ============================================================================

#if USE_AVX

static void Or_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Logical_U8_Args_t* a = (Logical_U8_Args_t*)user_data;
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        vx_uint8* pSrc1 = a->src1 + y * a->src1_stride;
        vx_uint8* pSrc2 = a->src2 + y * a->src2_stride;
        vx_uint8* pDst = a->dst + y * a->dst_stride;
        
        vx_uint32 width = 0;
        for (; width + 128 <= a->width; width += 128) {
            __m256i a0 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i a1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 32));
            __m256i a2 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 64));
            __m256i a3 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 96));
            __m256i b0 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            __m256i b1 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 32));
            __m256i b2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 64));
            __m256i b3 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 96));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_or_si256(a0, b0));
            _mm256_storeu_si256((__m256i *)(pDst + width + 32), _mm256_or_si256(a1, b1));
            _mm256_storeu_si256((__m256i *)(pDst + width + 64), _mm256_or_si256(a2, b2));
            _mm256_storeu_si256((__m256i *)(pDst + width + 96), _mm256_or_si256(a3, b3));
        }
        for (; width + 32 <= a->width; width += 32) {
            __m256i pixels1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i pixels2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_or_si256(pixels1, pixels2));
        }
        for (; width < a->width; width++) {
            pDst[width] = pSrc1[width] | pSrc2[width];
        }
    }
}

#endif // USE_AVX

int HafCpu_Or_U8_U8U8_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    vx_uint8    * pSrcImage2,
    vx_uint32     srcImage2StrideInBytes
) {
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        return HafCpu_Or_U8_U8U8(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                  pSrcImage1, srcImage1StrideInBytes,
                                  pSrcImage2, srcImage2StrideInBytes);
    }
    
#if USE_AVX
    Logical_U8_Args_t args = {
        .width = dstWidth,
        .height = dstHeight,
        .dst = pDstImage,
        .dst_stride = dstImageStrideInBytes,
        .src1 = pSrcImage1,
        .src1_stride = srcImage1StrideInBytes,
        .src2 = pSrcImage2,
        .src2_stride = srcImage2StrideInBytes
    };
    
    AgoParallelForRows(dstHeight, Or_U8_Row_AVX, &args);
#else
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < (int)dstHeight; y++) {
        vx_uint8* pSrc1 = pSrcImage1 + y * srcImage1StrideInBytes;
        vx_uint8* pSrc2 = pSrcImage2 + y * srcImage2StrideInBytes;
        vx_uint8* pDst = pDstImage + y * dstImageStrideInBytes;
        for (vx_uint32 x = 0; x < dstWidth; x++) {
            pDst[x] = pSrc1[x] | pSrc2[x];
        }
    }
#endif
    
    return AGO_SUCCESS;
}

// ============================================================================
// Parallel Xor U8 = U8 ^ U8
// ============================================================================

#if USE_AVX

static void Xor_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Logical_U8_Args_t* a = (Logical_U8_Args_t*)user_data;
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        vx_uint8* pSrc1 = a->src1 + y * a->src1_stride;
        vx_uint8* pSrc2 = a->src2 + y * a->src2_stride;
        vx_uint8* pDst = a->dst + y * a->dst_stride;
        
        vx_uint32 width = 0;
        for (; width + 128 <= a->width; width += 128) {
            __m256i a0 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i a1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 32));
            __m256i a2 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 64));
            __m256i a3 = _mm256_loadu_si256((__m256i *)(pSrc1 + width + 96));
            __m256i b0 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            __m256i b1 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 32));
            __m256i b2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 64));
            __m256i b3 = _mm256_loadu_si256((__m256i *)(pSrc2 + width + 96));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_xor_si256(a0, b0));
            _mm256_storeu_si256((__m256i *)(pDst + width + 32), _mm256_xor_si256(a1, b1));
            _mm256_storeu_si256((__m256i *)(pDst + width + 64), _mm256_xor_si256(a2, b2));
            _mm256_storeu_si256((__m256i *)(pDst + width + 96), _mm256_xor_si256(a3, b3));
        }
        for (; width + 32 <= a->width; width += 32) {
            __m256i pixels1 = _mm256_loadu_si256((__m256i *)(pSrc1 + width));
            __m256i pixels2 = _mm256_loadu_si256((__m256i *)(pSrc2 + width));
            _mm256_storeu_si256((__m256i *)(pDst + width), _mm256_xor_si256(pixels1, pixels2));
        }
        for (; width < a->width; width++) {
            pDst[width] = pSrc1[width] ^ pSrc2[width];
        }
    }
}

#endif // USE_AVX

int HafCpu_Xor_U8_U8U8_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    vx_uint8    * pSrcImage2,
    vx_uint32     srcImage2StrideInBytes
) {
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        return HafCpu_Xor_U8_U8U8(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                   pSrcImage1, srcImage1StrideInBytes,
                                   pSrcImage2, srcImage2StrideInBytes);
    }
    
#if USE_AVX
    Logical_U8_Args_t args = {
        .width = dstWidth,
        .height = dstHeight,
        .dst = pDstImage,
        .dst_stride = dstImageStrideInBytes,
        .src1 = pSrcImage1,
        .src1_stride = srcImage1StrideInBytes,
        .src2 = pSrcImage2,
        .src2_stride = srcImage2StrideInBytes
    };
    
    AgoParallelForRows(dstHeight, Xor_U8_Row_AVX, &args);
#else
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < (int)dstHeight; y++) {
        vx_uint8* pSrc1 = pSrcImage1 + y * srcImage1StrideInBytes;
        vx_uint8* pSrc2 = pSrcImage2 + y * srcImage2StrideInBytes;
        vx_uint8* pDst = pDstImage + y * dstImageStrideInBytes;
        for (vx_uint32 x = 0; x < dstWidth; x++) {
            pDst[x] = pSrc1[x] ^ pSrc2[x];
        }
    }
#endif
    
    return AGO_SUCCESS;
}

// ============================================================================
// Parallel Not U8 = ~U8
// ============================================================================

#if USE_AVX

typedef struct {
    vx_uint32 width;
    vx_uint32 height;
    vx_uint8* dst;
    vx_uint32 dst_stride;
    vx_uint8* src;
    vx_uint32 src_stride;
} Not_U8_Args_t;

static void Not_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Not_U8_Args_t* a = (Not_U8_Args_t*)user_data;
    __m256i all_ones = _mm256_set1_epi8(0xFF);
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        vx_uint8* pSrc = a->src + y * a->src_stride;
        vx_uint8* pDst = a->dst + y * a->dst_stride;
        
        vx_uint32 width = 0;
        for (; width + 128 <= a->width; width += 128) {
            _mm256_storeu_si256((__m256i *)(pDst + width), 
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(pSrc + width)), all_ones));
            _mm256_storeu_si256((__m256i *)(pDst + width + 32), 
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(pSrc + width + 32)), all_ones));
            _mm256_storeu_si256((__m256i *)(pDst + width + 64), 
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(pSrc + width + 64)), all_ones));
            _mm256_storeu_si256((__m256i *)(pDst + width + 96), 
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(pSrc + width + 96)), all_ones));
        }
        for (; width + 32 <= a->width; width += 32) {
            _mm256_storeu_si256((__m256i *)(pDst + width), 
                _mm256_xor_si256(_mm256_loadu_si256((__m256i *)(pSrc + width)), all_ones));
        }
        for (; width < a->width; width++) {
            pDst[width] = ~pSrc[width];
        }
    }
}

#endif // USE_AVX

int HafCpu_Not_U8_U8_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage,
    vx_uint32     srcImageStrideInBytes
) {
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        return HafCpu_Not_U8_U8(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                  pSrcImage, srcImageStrideInBytes);
    }
    
#if USE_AVX
    Not_U8_Args_t args = {
        .width = dstWidth,
        .height = dstHeight,
        .dst = pDstImage,
        .dst_stride = dstImageStrideInBytes,
        .src = pSrcImage,
        .src_stride = srcImageStrideInBytes
    };
    
    AgoParallelForRows(dstHeight, Not_U8_Row_AVX, &args);
#else
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < (int)dstHeight; y++) {
        vx_uint8* pSrc = pSrcImage + y * srcImageStrideInBytes;
        vx_uint8* pDst = pDstImage + y * dstImageStrideInBytes;
        for (vx_uint32 x = 0; x < dstWidth; x++) {
            pDst[x] = ~pSrc[x];
        }
    }
#endif
    
    return AGO_SUCCESS;
}
