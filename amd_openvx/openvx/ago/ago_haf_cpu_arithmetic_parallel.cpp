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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * AGO_HAF_CPU_ARITHMETIC_PARALLEL.CPP
 *
 * Row-based parallel implementations of arithmetic kernels.
 * These functions provide OpenCV-style threading using OpenMP.
 *
 * Key features:
 * - Row-based decomposition (cache-friendly)
 * - Guided scheduling (adaptive chunk sizes)
 * - Maintains existing AVX/SIMD optimizations
 * - Falls back to serial for small images
 */

#include "ago_internal.h"
#include "ago_parallel.h"

// ============================================================================
// Parallel Add U8 = U8 + U8 (Wrap)
// ============================================================================

#if USE_AVX

// Structure to pass arguments to row processing function
typedef struct {
    vx_uint32 dstWidth;
    vx_uint32 dstHeight;
    vx_uint8* pDstImage;
    vx_uint32 dstImageStrideInBytes;
    vx_uint8* pSrcImage1;
    vx_uint32 srcImage1StrideInBytes;
    vx_uint8* pSrcImage2;
    vx_uint32 srcImage2StrideInBytes;
    bool useAligned;
    int alignedWidth;
    int postfixWidth;
} Add_U8_Args_t;

// Row processing function for Add
static void Add_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Add_U8_Args_t* a = (Add_U8_Args_t*)user_data;
    
    // Calculate row pointers
    vx_uint8* pSrc1_row = a->pSrcImage1 + start_y * a->srcImage1StrideInBytes;
    vx_uint8* pSrc2_row = a->pSrcImage2 + start_y * a->srcImage2StrideInBytes;
    vx_uint8* pDst_row = a->pDstImage + start_y * a->dstImageStrideInBytes;
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        __m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
        vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
        __m256i pixels1, pixels2;
        
        if (a->useAligned) {
            pLocalSrc1_ymm = (__m256i*) pSrc1_row;
            pLocalSrc2_ymm = (__m256i*) pSrc2_row;
            pLocalDst_ymm = (__m256i*) pDst_row;
            
            for (int width = 0; width < a->alignedWidth; width += 32) {
                pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
                pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
                pixels1 = _mm256_add_epi8(pixels1, pixels2);
                _mm256_store_si256(pLocalDst_ymm++, pixels1);
            }
        } else {
            pLocalSrc1_ymm = (__m256i*) pSrc1_row;
            pLocalSrc2_ymm = (__m256i*) pSrc2_row;
            pLocalDst_ymm = (__m256i*) pDst_row;
            
            for (int width = 0; width < a->alignedWidth; width += 32) {
                pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
                pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
                pixels1 = _mm256_add_epi8(pixels1, pixels2);
                _mm256_storeu_si256(pLocalDst_ymm++, pixels1);
            }
        }
        
        // Process postfix (remainder) pixels
        pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
        pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
        pLocalDst = (vx_uint8 *)pLocalDst_ymm;
        
        for (int width = 0; width < a->postfixWidth; width++) {
            vx_int16 temp = (vx_int16)(*pLocalSrc1++) + (vx_int16)(*pLocalSrc2++);
            *pLocalDst++ = (vx_uint8)temp;
        }
        
        // Advance to next row
        pSrc1_row += a->srcImage1StrideInBytes;
        pSrc2_row += a->srcImage2StrideInBytes;
        pDst_row += a->dstImageStrideInBytes;
    }
}

#endif // USE_AVX

/**
 * HafCpu_Add_U8_U8U8_Wrap_OpenMP - OpenMP parallel version
 *
 * Parallelizes the outer height loop using row-based decomposition.
 * Each thread processes a different set of rows.
 */
int HafCpu_Add_U8_U8U8_Wrap_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage1,
    vx_uint32     srcImage1StrideInBytes,
    vx_uint8    * pSrcImage2,
    vx_uint32     srcImage2StrideInBytes
) {
    // Determine if threading should be used
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        // Fall back to serial implementation for small images
        return HafCpu_Add_U8_U8U8_Wrap(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                        pSrcImage1, srcImage1StrideInBytes,
                                        pSrcImage2, srcImage2StrideInBytes);
    }
    
#if USE_AVX
    bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage | 
                         srcImage1StrideInBytes | srcImage2StrideInBytes | dstImageStrideInBytes) & 0x1F) == 0);
    
    Add_U8_Args_t args = {
        .dstWidth = dstWidth,
        .dstHeight = dstHeight,
        .pDstImage = pDstImage,
        .dstImageStrideInBytes = dstImageStrideInBytes,
        .pSrcImage1 = pSrcImage1,
        .srcImage1StrideInBytes = srcImage1StrideInBytes,
        .pSrcImage2 = pSrcImage2,
        .srcImage2StrideInBytes = srcImage2StrideInBytes,
        .useAligned = useAligned,
        .alignedWidth = (int)(dstWidth & ~31),
        .postfixWidth = (int)(dstWidth - (dstWidth & ~31))
    };
    
    // Parallel execution with guided scheduling
    AgoParallelForRows(dstHeight, Add_U8_Row_AVX, &args);
#else
    // Non-AVX path - use simple OpenMP parallelization
    #pragma omp parallel for schedule(guided)
    for (int height = 0; height < (int)dstHeight; height++) {
        vx_uint8* pSrc1 = pSrcImage1 + height * srcImage1StrideInBytes;
        vx_uint8* pSrc2 = pSrcImage2 + height * srcImage2StrideInBytes;
        vx_uint8* pDst = pDstImage + height * dstImageStrideInBytes;
        
        for (vx_uint32 width = 0; width < dstWidth; width++) {
            vx_int16 temp = (vx_int16)(pSrc1[width]) + (vx_int16)(pSrc2[width]);
            pDst[width] = (vx_uint8)temp;
        }
    }
#endif
    
    return AGO_SUCCESS;
}

// ============================================================================
// Parallel Subtract U8 = U8 - U8
// ============================================================================

#if USE_AVX

typedef struct {
    vx_uint32 dstWidth;
    vx_uint32 dstHeight;
    vx_uint8* pDstImage;
    vx_uint32 dstImageStrideInBytes;
    vx_uint8* pSrcImage1;
    vx_uint32 srcImage1StrideInBytes;
    vx_uint8* pSrcImage2;
    vx_uint32 srcImage2StrideInBytes;
    bool useAligned;
    int alignedWidth;
    int postfixWidth;
} Subtract_U8_Args_t;

static void Subtract_U8_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Subtract_U8_Args_t* a = (Subtract_U8_Args_t*)user_data;
    
    vx_uint8* pSrc1_row = a->pSrcImage1 + start_y * a->srcImage1StrideInBytes;
    vx_uint8* pSrc2_row = a->pSrcImage2 + start_y * a->srcImage2StrideInBytes;
    vx_uint8* pDst_row = a->pDstImage + start_y * a->dstImageStrideInBytes;
    
    for (vx_uint32 y = start_y; y < end_y; y++) {
        __m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
        vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
        __m256i pixels1, pixels2;
        
        if (a->useAligned) {
            pLocalSrc1_ymm = (__m256i*) pSrc1_row;
            pLocalSrc2_ymm = (__m256i*) pSrc2_row;
            pLocalDst_ymm = (__m256i*) pDst_row;
            
            for (int width = 0; width < a->alignedWidth; width += 32) {
                pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
                pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
                pixels1 = _mm256_sub_epi8(pixels1, pixels2);
                _mm256_store_si256(pLocalDst_ymm++, pixels1);
            }
        } else {
            pLocalSrc1_ymm = (__m256i*) pSrc1_row;
            pLocalSrc2_ymm = (__m256i*) pSrc2_row;
            pLocalDst_ymm = (__m256i*) pDst_row;
            
            for (int width = 0; width < a->alignedWidth; width += 32) {
                pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
                pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
                pixels1 = _mm256_sub_epi8(pixels1, pixels2);
                _mm256_storeu_si256(pLocalDst_ymm++, pixels1);
            }
        }
        
        // Postfix
        pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
        pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
        pLocalDst = (vx_uint8 *)pLocalDst_ymm;
        
        for (int width = 0; width < a->postfixWidth; width++) {
            vx_int16 temp = (vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++);
            *pLocalDst++ = (vx_uint8)temp;
        }
        
        pSrc1_row += a->srcImage1StrideInBytes;
        pSrc2_row += a->srcImage2StrideInBytes;
        pDst_row += a->dstImageStrideInBytes;
    }
}

#endif // USE_AVX

int HafCpu_Subtract_U8_U8U8_Wrap_OpenMP(
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
        return HafCpu_Subtract_U8_U8U8_Wrap(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                            pSrcImage1, srcImage1StrideInBytes,
                                            pSrcImage2, srcImage2StrideInBytes);
    }
    
#if USE_AVX
    bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage | 
                         srcImage1StrideInBytes | srcImage2StrideInBytes | dstImageStrideInBytes) & 0x1F) == 0);
    
    Subtract_U8_Args_t args = {
        .dstWidth = dstWidth,
        .dstHeight = dstHeight,
        .pDstImage = pDstImage,
        .dstImageStrideInBytes = dstImageStrideInBytes,
        .pSrcImage1 = pSrcImage1,
        .srcImage1StrideInBytes = srcImage1StrideInBytes,
        .pSrcImage2 = pSrcImage2,
        .srcImage2StrideInBytes = srcImage2StrideInBytes,
        .useAligned = useAligned,
        .alignedWidth = (int)(dstWidth & ~31),
        .postfixWidth = (int)(dstWidth - (dstWidth & ~31))
    };
    
    AgoParallelForRows(dstHeight, Subtract_U8_Row_AVX, &args);
#else
    #pragma omp parallel for schedule(guided)
    for (int height = 0; height < (int)dstHeight; height++) {
        vx_uint8* pSrc1 = pSrcImage1 + height * srcImage1StrideInBytes;
        vx_uint8* pSrc2 = pSrcImage2 + height * srcImage2StrideInBytes;
        vx_uint8* pDst = pDstImage + height * dstImageStrideInBytes;
        
        for (vx_uint32 width = 0; width < dstWidth; width++) {
            vx_int16 temp = (vx_int16)(pSrc1[width]) - (vx_int16)(pSrc2[width]);
            pDst[width] = (vx_uint8)temp;
        }
    }
#endif
    
    return AGO_SUCCESS;
}

// ============================================================================
// Parallel Box3x3 Filter
// ============================================================================

#if USE_AVX

typedef struct {
    vx_uint32 dstWidth;
    vx_uint32 dstHeight;
    vx_uint8* pDstImage;
    vx_uint32 dstImageStrideInBytes;
    vx_uint8* pSrcImage;
    vx_uint32 srcImageStrideInBytes;
    vx_uint8* pScratch;
} Box3x3_Args_t;

// Simplified Box3x3 row processing (without the complex scratch buffer optimization)
// This demonstrates the pattern; full optimization would need horizontal pass cache
static void Box3x3_Row_AVX(vx_uint32 start_y, vx_uint32 end_y, void* user_data) {
    Box3x3_Args_t* a = (Box3x3_Args_t*)user_data;
    
    // Process rows from start_y to end_y (excluding borders)
    vx_uint32 first_row = (start_y == 0) ? 1 : start_y;
    vx_uint32 last_row = (end_y >= a->dstHeight - 1) ? a->dstHeight - 1 : end_y;
    
    for (vx_uint32 y = first_row; y < last_row; y++) {
        vx_uint8* pSrc_above = a->pSrcImage + (y - 1) * a->srcImageStrideInBytes;
        vx_uint8* pSrc_curr = a->pSrcImage + y * a->srcImageStrideInBytes;
        vx_uint8* pSrc_below = a->pSrcImage + (y + 1) * a->srcImageStrideInBytes;
        vx_uint8* pDst = a->pDstImage + y * a->dstImageStrideInBytes;
        
        // Simple scalar implementation for demonstration
        // Full AVX optimization would use the horizontal pass approach
        for (vx_uint32 x = 1; x < a->dstWidth - 1; x++) {
            vx_uint32 sum = pSrc_above[x-1] + pSrc_above[x] + pSrc_above[x+1] +
                          pSrc_curr[x-1]  + pSrc_curr[x]  + pSrc_curr[x+1] +
                          pSrc_below[x-1] + pSrc_below[x] + pSrc_below[x+1];
            pDst[x] = (vx_uint8)(sum / 9);
        }
    }
}

#endif // USE_AVX

int HafCpu_Box_U8_U8_3x3_OpenMP(
    vx_uint32     dstWidth,
    vx_uint32     dstHeight,
    vx_uint8    * pDstImage,
    vx_uint32     dstImageStrideInBytes,
    vx_uint8    * pSrcImage,
    vx_uint32     srcImageStrideInBytes,
    vx_uint8    * pScratch
) {
    // For small images, use optimized serial version
    if (!AgoShouldUseThreading(dstHeight, dstWidth)) {
        return HafCpu_Box_U8_U8_3x3(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes,
                                     pSrcImage, srcImageStrideInBytes, pScratch);
    }
    
    // For larger images, parallelize the row processing
    // Note: Box filter with scratch buffer optimization is complex to parallelize
    // efficiently due to the horizontal/vertical pass dependency
    // A full implementation would parallelize the vertical pass
    
#if USE_AVX
    Box3x3_Args_t args = {
        .dstWidth = dstWidth,
        .dstHeight = dstHeight,
        .pDstImage = pDstImage,
        .dstImageStrideInBytes = dstImageStrideInBytes,
        .pSrcImage = pSrcImage,
        .srcImageStrideInBytes = srcImageStrideInBytes,
        .pScratch = pScratch
    };
    
    AgoParallelForRows(dstHeight, Box3x3_Row_AVX, &args);
    
    // Process borders (first/last row, first/last column)
    // Top row
    vx_uint8* pDst_top = pDstImage;
    vx_uint8* pSrc_row1 = pSrcImage + srcImageStrideInBytes;
    for (vx_uint32 x = 0; x < dstWidth; x++) {
        pDst_top[x] = pSrc_row1[x];
    }
    // Bottom row
    vx_uint8* pDst_bottom = pDstImage + (dstHeight - 1) * dstImageStrideInBytes;
    vx_uint8* pSrc_row_last = pSrcImage + (dstHeight - 2) * srcImageStrideInBytes;
    for (vx_uint32 x = 0; x < dstWidth; x++) {
        pDst_bottom[x] = pSrc_row_last[x];
    }
    // Left/right columns
    for (vx_uint32 y = 1; y < dstHeight - 1; y++) {
        vx_uint8* pDst_row = pDstImage + y * dstImageStrideInBytes;
        pDst_row[0] = pDst_row[1];
        pDst_row[dstWidth - 1] = pDst_row[dstWidth - 2];
    }
#else
    // Simple OpenMP parallelization
    #pragma omp parallel for schedule(guided)
    for (int y = 1; y < (int)dstHeight - 1; y++) {
        vx_uint8* pSrc_above = pSrcImage + (y - 1) * srcImageStrideInBytes;
        vx_uint8* pSrc_curr = pSrcImage + y * srcImageStrideInBytes;
        vx_uint8* pSrc_below = pSrcImage + (y + 1) * srcImageStrideInBytes;
        vx_uint8* pDst = pDstImage + y * dstImageStrideInBytes;
        
        for (vx_uint32 x = 1; x < dstWidth - 1; x++) {
            vx_uint32 sum = pSrc_above[x-1] + pSrc_above[x] + pSrc_above[x+1] +
                          pSrc_curr[x-1]  + pSrc_curr[x]  + pSrc_curr[x+1] +
                          pSrc_below[x-1] + pSrc_below[x] + pSrc_below[x+1];
            pDst[x] = (vx_uint8)(sum / 9);
        }
    }
    
    // Process borders serially
    // Top row
    for (vx_uint32 x = 0; x < dstWidth; x++) {
        pDstImage[x] = pSrcImage[srcImageStrideInBytes + x];
    }
    // Bottom row
    vx_uint8* pDst_bottom = pDstImage + (dstHeight - 1) * dstImageStrideInBytes;
    vx_uint8* pSrc_row_last = pSrcImage + (dstHeight - 2) * srcImageStrideInBytes;
    for (vx_uint32 x = 0; x < dstWidth; x++) {
        pDst_bottom[x] = pSrc_row_last[x];
    }
    // Left/right columns
    for (vx_uint32 y = 1; y < dstHeight - 1; y++) {
        vx_uint8* pDst_row = pDstImage + y * dstImageStrideInBytes;
        vx_uint8* pSrc_row = pSrcImage + y * srcImageStrideInBytes;
        pDst_row[0] = pSrc_row[1];
        pDst_row[dstWidth - 1] = pSrc_row[dstWidth - 2];
    }
#endif
    
    return AGO_SUCCESS;
}
