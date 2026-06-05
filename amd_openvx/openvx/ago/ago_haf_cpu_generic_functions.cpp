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

#include <VX/vx.h>
#include <VX/vxu.h>
#include "ago_internal.h"

/*! \brief The largest nonlinear filter matrix the specification requires support for is 9x9.
*/
#define C_MAX_NONLINEAR_DIM (9)

#if USE_AVX
static inline void HafCpu_StoreEightU8FromI32(vx_uint8 *dst, __m256i values)
{
    __m128i lo = _mm256_castsi256_si128(values);
    __m128i hi = _mm256_extracti128_si256(values, 1);
    __m128i packed16 = _mm_packus_epi32(lo, hi);
    __m128i packed8 = _mm_packus_epi16(packed16, packed16);
    _mm_storel_epi64((__m128i *)dst, packed8);
}
#endif

int HafCpu_WeightedAverage_U8_U8U8
    (
        vx_image img1, 
        vx_float32 alpha, 
        vx_image img2, 
        vx_image output
    )
{
    vx_uint32 y, x, width = 0, height = 0;
    void *dst_base = NULL;
    void *src_base[2] = { NULL, NULL };
    vx_imagepatch_addressing_t dst_addr, src_addr[2];
    vx_rectangle_t rect;
    vx_df_image img1_format = 0;
    vx_df_image img2_format = 0;
    vx_df_image out_format = 0;
    vx_status status = VX_SUCCESS;
    vx_map_id src_map_id[2];
    vx_map_id dst_map_id;
    vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    vxQueryImage(img1, VX_IMAGE_FORMAT, &img1_format, sizeof(img1_format));
    vxQueryImage(img2, VX_IMAGE_FORMAT, &img2_format, sizeof(img2_format));

    status = vxGetValidRegionImage(img1, &rect);
    status |= vxMapImagePatch(img1, &rect, 0, &src_map_id[0], &src_addr[0], (void **)&src_base[0],
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(img2, &rect, 0, &src_map_id[1], &src_addr[1], (void **)&src_base[1],
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(output, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_base,
                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    width = src_addr[0].dim_x;
    height = src_addr[0].dim_y;
    if (img1_format == VX_DF_IMAGE_U8 && img2_format == VX_DF_IMAGE_U8 && out_format == VX_DF_IMAGE_U8 &&
        src_addr[0].stride_x == 1 && src_addr[1].stride_x == 1 && dst_addr.stride_x == 1)
    {
        const vx_float32 beta = 1.0f - alpha;
#if USE_AVX
        const __m256 alpha_ps = _mm256_set1_ps(alpha);
        const __m256 beta_ps = _mm256_set1_ps(beta);
#endif
        for (y = 0; y < height; y++)
        {
            vx_uint8 *src0 = (vx_uint8 *)src_base[0] + y * src_addr[0].stride_y;
            vx_uint8 *src1 = (vx_uint8 *)src_base[1] + y * src_addr[1].stride_y;
            vx_uint8 *dst = (vx_uint8 *)dst_base + y * dst_addr.stride_y;
            x = 0;
#if USE_AVX
            for (; x + 8 <= width; x += 8)
            {
                __m128i s0_u8 = _mm_loadl_epi64((__m128i *)(src0 + x));
                __m128i s1_u8 = _mm_loadl_epi64((__m128i *)(src1 + x));
                __m256 s0_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(s0_u8));
                __m256 s1_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(s1_u8));
                __m256 result_ps = _mm256_add_ps(_mm256_mul_ps(alpha_ps, s0_ps), _mm256_mul_ps(beta_ps, s1_ps));
                HafCpu_StoreEightU8FromI32(dst + x, _mm256_cvttps_epi32(result_ps));
            }
#endif
            for (; x < width; x++)
            {
                vx_int32 src0Value = src0[x];
                vx_int32 src1Value = src1[x];
                vx_int32 result = (vx_int32)(beta * (vx_float32)src1Value + alpha * (vx_float32)src0Value);
                dst[x] = (vx_uint8)result;
            }
        }
    }
    else
    {
        for (y = 0; y < height; y++)
        {
            for (x = 0; x < width; x++)
            {
                void *src0p = vxFormatImagePatchAddress2d(src_base[0], x, y, &src_addr[0]);
                void *src1p = vxFormatImagePatchAddress2d(src_base[1], x, y, &src_addr[1]);
                void *dstp = vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                vx_int32 src0 = *(vx_uint8 *)src0p;
                vx_int32 src1 = *(vx_uint8 *)src1p;
                vx_int32 result = (vx_int32)((1 - alpha) * (vx_float32)src1 + alpha * (vx_float32)src0);
                *(vx_uint8 *)dstp = (vx_uint8)result;
            }
        }
    }
    status |= vxUnmapImagePatch(img1, src_map_id[0]);
    status |= vxUnmapImagePatch(img2, src_map_id[1]);
    status |= vxUnmapImagePatch(output, dst_map_id);
    return status;
}

// helpers
static int vx_uint8_compare(const void *p1, const void *p2)
{
    vx_uint8 a = *(vx_uint8 *)p1;
    vx_uint8 b = *(vx_uint8 *)p2;
    if (a > b)
        return 1;
    else if (a == b)
        return 0;
    else
        return -1;
}

static vx_uint32 readMaskedRectangle(const void *base,
    const vx_imagepatch_addressing_t *addr,
    const vx_border_t *borders,
    vx_df_image type,
    vx_uint32 center_x,
    vx_uint32 center_y,
    vx_uint32 left,
    vx_uint32 top,
    vx_uint32 right,
    vx_uint32 bottom,
    vx_uint8 *mask,
    vx_uint8 *destination,
    vx_uint32 border_x_start)
{
    vx_int32 width = (vx_int32)addr->dim_x, height = (vx_int32)addr->dim_y;
    vx_int32 stride_y = addr->stride_y;
    vx_int32 stride_x = addr->stride_x;
    vx_uint16 stride_x_bits = addr->stride_x_bits;
    const vx_uint8 *ptr = (const vx_uint8 *)base;
    vx_int32 ky, kx;
    vx_uint32 mask_index = 0;
    vx_uint32 dest_index = 0;

    // kx, ky - kernel x and y
    if (borders->mode == VX_BORDER_REPLICATE || borders->mode == VX_BORDER_UNDEFINED)
    {
        for (ky = -(int32_t)top; ky <= (int32_t)bottom; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            y = y < 0 ? 0 : y >= height ? height - 1 : y;

            for (kx = -(int32_t)left; kx <= (int32_t)right; ++kx, ++mask_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                x = x < (int32_t)border_x_start ? (int32_t)border_x_start : x >= width ? width - 1 : x;
                if (mask[mask_index])
                {
                    if (type == VX_DF_IMAGE_U1)
                        ((vx_uint8*)destination)[dest_index++] =
                            ( *(vx_uint8*)(ptr + y*stride_y + (x*stride_x_bits) / 8) & (1 << (x % 8)) ) >> (x % 8);
                    else    // VX_DF_IMAGE_U8
                        ((vx_uint8*)destination)[dest_index++] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                }
            }
        }
    }
    else if (borders->mode == VX_BORDER_CONSTANT)
    {
        vx_pixel_value_t cval = borders->constant_value;
        for (ky = -(int32_t)top; ky <= (int32_t)bottom; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            int ccase_y = y < 0 || y >= height;

            for (kx = -(int32_t)left; kx <= (int32_t)right; ++kx, ++mask_index)
            {
                if (mask[mask_index])
                {
                    vx_int32 x = (int32_t)(center_x + kx);
                    int ccase = ccase_y || x < (int32_t)border_x_start || x >= width;
                    if (type == VX_DF_IMAGE_U1)
                        ((vx_uint8*)destination)[dest_index++] = ccase ? ( (vx_uint8)cval.U1 ? 1 : 0 ) :
                            ( *(vx_uint8*)(ptr + y*stride_y + (x*stride_x_bits) / 8) & (1 << (x % 8)) ) >> (x % 8);
                    else    // VX_DF_IMAGE_U8
                        ((vx_uint8*)destination)[dest_index++] = ccase ? (vx_uint8)cval.U8 : *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                }
            }
        }
    }

    return dest_index;
}

vx_status vxAlterRectangle(vx_rectangle_t *rect,
                           vx_int32 dsx,
                           vx_int32 dsy,
                           vx_int32 dex,
                           vx_int32 dey)
{
    if (rect)
    {
        rect->start_x += dsx;
        rect->start_y += dsy;
        rect->end_x += dex;
        rect->end_y += dey;
        return VX_SUCCESS;
    }
    return VX_ERROR_INVALID_REFERENCE;
}

// nodeless version of NonLinearFilter kernel
int HafCpu_NonLinearFilter_DATA_DATADATA
    (
        vx_int32 function,     
        vx_image src, 
        vx_matrix mask, 
        vx_image dst, 
        vx_border_t *border
    )
{
    vx_uint32 y, x;
    void *src_base = NULL;
    void *dst_base = NULL;
    vx_df_image format = 0;
    vx_imagepatch_addressing_t src_addr, dst_addr;
    vx_rectangle_t rect;
    vx_uint32 low_x = 0, low_y = 0, high_x, high_y, shift_x_u1;

    vx_uint8 m[C_MAX_NONLINEAR_DIM * C_MAX_NONLINEAR_DIM];
    vx_uint8 v[C_MAX_NONLINEAR_DIM * C_MAX_NONLINEAR_DIM];
    vx_uint8 res_val = 0;

    vx_status status = vxGetValidRegionImage(src, &rect);
    status |= vxQueryImage(src, VX_IMAGE_FORMAT, &format, sizeof(format));
    status |= vxAccessImagePatch(src, &rect, 0, &src_addr, &src_base, VX_READ_ONLY);
    status |= vxAccessImagePatch(dst, &rect, 0, &dst_addr, &dst_base, VX_WRITE_ONLY);

    vx_enum func = function;

    vx_size mrows, mcols;
    vx_enum mtype = 0;
    status |= vxQueryMatrix(mask, VX_MATRIX_ROWS, &mrows, sizeof(mrows));
    status |= vxQueryMatrix(mask, VX_MATRIX_COLUMNS, &mcols, sizeof(mcols));
    status |= vxQueryMatrix(mask, VX_MATRIX_TYPE, &mtype, sizeof(mtype));

    vx_coordinates2d_t origin;
    status |= vxQueryMatrix(mask, VX_MATRIX_ORIGIN, &origin, sizeof(origin));

    if ((mtype != VX_TYPE_UINT8) || (sizeof(m) < mrows * mcols))
        status = VX_ERROR_INVALID_PARAMETERS;

    status |= vxCopyMatrix(mask, m, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (status == VX_SUCCESS)
    {
        vx_size rx0 = origin.x;
        vx_size ry0 = origin.y;
        vx_size rx1 = mcols - origin.x - 1;
        vx_size ry1 = mrows - origin.y - 1;

        shift_x_u1 = (format == VX_DF_IMAGE_U1) ? rect.start_x % 8 : 0;
        high_x = src_addr.dim_x - shift_x_u1;   // U1 addressing rounds down imagepatch start_x to nearest byte boundary
        high_y = src_addr.dim_y;

        if (border->mode == VX_BORDER_UNDEFINED)
        {
            low_x  += (vx_uint32)rx0;
            low_y  += (vx_uint32)ry0;
            high_x -= (vx_uint32)rx1;
            high_y -= (vx_uint32)ry1;
            vxAlterRectangle(&rect, (vx_int32)rx0, (vx_int32)ry0, -(vx_int32)rx1, -(vx_int32)ry1);
        }

        // SIMD fast path: 3x3 mask, U8 image, REPLICATE or UNDEFINED border.
        // Median over 5 elements (e.g., cross) is computed with 10 SIMD min/max ops;
        // median over 9 (3x3 box) uses the same sort network as HafCpu_Median_U8_U8_3x3.
        // Falls through to the scalar code path for borders or unsupported cases.
        vx_uint32 ix0 = low_x, ix1 = high_x, iy0 = low_y, iy1 = high_y;
        bool simd_did_interior = false;
        if (format == VX_DF_IMAGE_U8 && mtype == VX_TYPE_UINT8 &&
            mcols == 3 && mrows == 3 && origin.x == 1 && origin.y == 1 &&
            src_addr.stride_x == 1 && dst_addr.stride_x == 1 &&
            (border->mode == VX_BORDER_REPLICATE || border->mode == VX_BORDER_UNDEFINED) &&
            src_addr.dim_x >= 16 && src_addr.dim_y >= 2)
        {
            vx_int32 dxo[9], dyo[9];
            vx_int32 active_count = 0;
            for (vx_int32 my = 0; my < 3; my++)
            {
                for (vx_int32 mx = 0; mx < 3; mx++)
                {
                    if (m[my*3 + mx])
                    {
                        dxo[active_count] = mx - 1;
                        dyo[active_count] = my - 1;
                        active_count++;
                    }
                }
            }

            bool simd_supported = (active_count > 0) &&
                ((func == VX_NONLINEAR_FILTER_MIN) ||
                 (func == VX_NONLINEAR_FILTER_MAX) ||
                 (func == VX_NONLINEAR_FILTER_MEDIAN && (active_count == 5 || active_count == 9)));

            if (simd_supported)
            {
                // Interior = SIMD-safe region (no need to clamp to borders).
                ix0 = (border->mode == VX_BORDER_UNDEFINED) ? low_x  : ((low_x  > 1) ? low_x  : 1);
                iy0 = (border->mode == VX_BORDER_UNDEFINED) ? low_y  : ((low_y  > 1) ? low_y  : 1);
                ix1 = (border->mode == VX_BORDER_UNDEFINED) ? high_x : ((high_x < src_addr.dim_x - 1) ? high_x : src_addr.dim_x - 1);
                iy1 = (border->mode == VX_BORDER_UNDEFINED) ? high_y : ((high_y < src_addr.dim_y - 1) ? high_y : src_addr.dim_y - 1);

                const vx_uint8 *src_p = (const vx_uint8 *)src_base;
                vx_uint8 *dst_p = (vx_uint8 *)dst_base;
                vx_int32 sstride = src_addr.stride_y;
                vx_int32 dstride = dst_addr.stride_y;

                // Precompute, per active tap, the row-relative byte offset
                // (dyo[i]*sstride + dxo[i]). This is invariant across x, so the
                // inner SIMD/scalar loops can load from (src_p + y*sstride + tapOff[i] + x)
                // without the per-load row-stride multiply the original code paid.
                vx_int32 tapOff[9];
                for (vx_int32 i = 0; i < active_count; i++)
                    tapOff[i] = dyo[i] * sstride + dxo[i];

                for (y = iy0; y < iy1; y++)
                {
                    vx_uint8 *drow = dst_p + y * dstride;
                    const vx_uint8 *srow = src_p + (vx_int32)y * sstride;
                    // Hoist the per-tap base row pointers out of the x loop and into
                    // registers. The previous code stored every tap into a vals[]
                    // array indexed by a runtime trip count (active_count), which
                    // forced the array to the stack and serialized the loads; pinning
                    // the taps to named pointers lets the compiler keep them resident
                    // and is what makes the cross/box median match the dedicated
                    // Median3x3 kernel's throughput.
                    const vx_uint8 *t0 = srow + tapOff[0];
                    const vx_uint8 *t1 = (active_count > 1) ? srow + tapOff[1] : t0;
                    const vx_uint8 *t2 = (active_count > 2) ? srow + tapOff[2] : t0;
                    const vx_uint8 *t3 = (active_count > 3) ? srow + tapOff[3] : t0;
                    const vx_uint8 *t4 = (active_count > 4) ? srow + tapOff[4] : t0;
                    const vx_uint8 *t5 = (active_count > 5) ? srow + tapOff[5] : t0;
                    const vx_uint8 *t6 = (active_count > 6) ? srow + tapOff[6] : t0;
                    const vx_uint8 *t7 = (active_count > 7) ? srow + tapOff[7] : t0;
                    const vx_uint8 *t8 = (active_count > 8) ? srow + tapOff[8] : t0;
                    x = ix0;
#if USE_AVX
                    // AVX2 chunks of 32 bytes.
                    for (; x + 32 <= ix1; x += 32)
                    {
                        __m256i result;
                        if (func == VX_NONLINEAR_FILTER_MEDIAN && active_count == 5)
                        {
                            __m256i a = _mm256_loadu_si256((const __m256i *)(t0 + x));
                            __m256i b = _mm256_loadu_si256((const __m256i *)(t1 + x));
                            __m256i c = _mm256_loadu_si256((const __m256i *)(t2 + x));
                            __m256i d = _mm256_loadu_si256((const __m256i *)(t3 + x));
                            __m256i e = _mm256_loadu_si256((const __m256i *)(t4 + x));
                            __m256i ab_lo = _mm256_min_epu8(a, b);
                            __m256i ab_hi = _mm256_max_epu8(a, b);
                            __m256i cd_lo = _mm256_min_epu8(c, d);
                            __m256i cd_hi = _mm256_max_epu8(c, d);
                            __m256i rl = _mm256_max_epu8(ab_lo, cd_lo);
                            __m256i ru = _mm256_min_epu8(ab_hi, cd_hi);
                            __m256i lo = _mm256_min_epu8(rl, ru);
                            __m256i hi = _mm256_max_epu8(rl, ru);
                            result = _mm256_max_epu8(lo, _mm256_min_epu8(hi, e));
                        }
                        else if (func == VX_NONLINEAR_FILTER_MEDIAN) // active_count == 9, median 3x3 box
                        {
                            __m256i a0 = _mm256_loadu_si256((const __m256i *)(t0 + x));
                            __m256i a1 = _mm256_loadu_si256((const __m256i *)(t1 + x));
                            __m256i a2 = _mm256_loadu_si256((const __m256i *)(t2 + x));
                            __m256i a3 = _mm256_loadu_si256((const __m256i *)(t3 + x));
                            __m256i a4 = _mm256_loadu_si256((const __m256i *)(t4 + x));
                            __m256i a5 = _mm256_loadu_si256((const __m256i *)(t5 + x));
                            __m256i a6 = _mm256_loadu_si256((const __m256i *)(t6 + x));
                            __m256i a7 = _mm256_loadu_si256((const __m256i *)(t7 + x));
                            __m256i a8 = _mm256_loadu_si256((const __m256i *)(t8 + x));
                            #define MFCS256(p1, p2) { __m256i mn = _mm256_min_epu8((p1),(p2)); __m256i mx = _mm256_max_epu8((p1),(p2)); (p1) = mn; (p2) = mx; }
                            MFCS256(a1, a2); MFCS256(a4, a5); MFCS256(a7, a8);
                            MFCS256(a0, a1); MFCS256(a3, a4); MFCS256(a6, a7);
                            MFCS256(a1, a2); MFCS256(a4, a5); MFCS256(a7, a8);
                            MFCS256(a0, a3); MFCS256(a5, a8); MFCS256(a4, a7);
                            MFCS256(a3, a6); MFCS256(a1, a4); MFCS256(a2, a5);
                            MFCS256(a4, a7); MFCS256(a4, a2); MFCS256(a6, a4);
                            MFCS256(a4, a2);
                            #undef MFCS256
                            result = a4;
                        }
                        else // MIN / MAX: accumulate directly, no array spill.
                        {
                            result = _mm256_loadu_si256((const __m256i *)(t0 + x));
                            for (vx_int32 i = 1; i < active_count; i++)
                            {
                                __m256i v = _mm256_loadu_si256((const __m256i *)(srow + tapOff[i] + x));
                                result = (func == VX_NONLINEAR_FILTER_MIN) ? _mm256_min_epu8(result, v) : _mm256_max_epu8(result, v);
                            }
                        }
                        _mm256_storeu_si256((__m256i *)(drow + x), result);
                    }
#endif
                    // SSE chunks of 16 bytes for the remainder.
                    for (; x + 16 <= ix1; x += 16)
                    {
                        __m128i result;
                        if (func == VX_NONLINEAR_FILTER_MEDIAN && active_count == 5)
                        {
                            // Median-of-5 via 10 SIMD min/max ops.
                            __m128i a = _mm_loadu_si128((const __m128i *)(t0 + x));
                            __m128i b = _mm_loadu_si128((const __m128i *)(t1 + x));
                            __m128i c = _mm_loadu_si128((const __m128i *)(t2 + x));
                            __m128i d = _mm_loadu_si128((const __m128i *)(t3 + x));
                            __m128i e = _mm_loadu_si128((const __m128i *)(t4 + x));
                            __m128i ab_lo = _mm_min_epu8(a, b);
                            __m128i ab_hi = _mm_max_epu8(a, b);
                            __m128i cd_lo = _mm_min_epu8(c, d);
                            __m128i cd_hi = _mm_max_epu8(c, d);
                            __m128i rl = _mm_max_epu8(ab_lo, cd_lo);
                            __m128i ru = _mm_min_epu8(ab_hi, cd_hi);
                            __m128i lo = _mm_min_epu8(rl, ru);
                            __m128i hi = _mm_max_epu8(rl, ru);
                            result = _mm_max_epu8(lo, _mm_min_epu8(hi, e));
                        }
                        else if (func == VX_NONLINEAR_FILTER_MEDIAN) // active_count == 9, median 3x3 box
                        {
                            __m128i a0 = _mm_loadu_si128((const __m128i *)(t0 + x));
                            __m128i a1 = _mm_loadu_si128((const __m128i *)(t1 + x));
                            __m128i a2 = _mm_loadu_si128((const __m128i *)(t2 + x));
                            __m128i a3 = _mm_loadu_si128((const __m128i *)(t3 + x));
                            __m128i a4 = _mm_loadu_si128((const __m128i *)(t4 + x));
                            __m128i a5 = _mm_loadu_si128((const __m128i *)(t5 + x));
                            __m128i a6 = _mm_loadu_si128((const __m128i *)(t6 + x));
                            __m128i a7 = _mm_loadu_si128((const __m128i *)(t7 + x));
                            __m128i a8 = _mm_loadu_si128((const __m128i *)(t8 + x));
                            #define MFCS(p1, p2) { __m128i mn = _mm_min_epu8((p1),(p2)); __m128i mx = _mm_max_epu8((p1),(p2)); (p1) = mn; (p2) = mx; }
                            MFCS(a1, a2); MFCS(a4, a5); MFCS(a7, a8);
                            MFCS(a0, a1); MFCS(a3, a4); MFCS(a6, a7);
                            MFCS(a1, a2); MFCS(a4, a5); MFCS(a7, a8);
                            MFCS(a0, a3); MFCS(a5, a8); MFCS(a4, a7);
                            MFCS(a3, a6); MFCS(a1, a4); MFCS(a2, a5);
                            MFCS(a4, a7); MFCS(a4, a2); MFCS(a6, a4);
                            MFCS(a4, a2);
                            #undef MFCS
                            result = a4;
                        }
                        else // MIN / MAX
                        {
                            result = _mm_loadu_si128((const __m128i *)(t0 + x));
                            for (vx_int32 i = 1; i < active_count; i++)
                            {
                                __m128i v = _mm_loadu_si128((const __m128i *)(srow + tapOff[i] + x));
                                result = (func == VX_NONLINEAR_FILTER_MIN) ? _mm_min_epu8(result, v) : _mm_max_epu8(result, v);
                            }
                        }
                        _mm_storeu_si128((__m128i *)(drow + x), result);
                    }
                    // Scalar tail for interior columns past the SIMD chunks.
                    for (; x < ix1; x++)
                    {
                        vx_uint8 sv[9];
                        for (vx_int32 i = 0; i < active_count; i++)
                            sv[i] = srow[tapOff[i] + (vx_int32)x];
                        vx_uint8 r;
                        if (func == VX_NONLINEAR_FILTER_MIN)
                        {
                            r = sv[0];
                            for (vx_int32 i = 1; i < active_count; i++) if (sv[i] < r) r = sv[i];
                        }
                        else if (func == VX_NONLINEAR_FILTER_MAX)
                        {
                            r = sv[0];
                            for (vx_int32 i = 1; i < active_count; i++) if (sv[i] > r) r = sv[i];
                        }
                        else
                        {
                            for (vx_int32 i = 1; i < active_count; i++)
                            {
                                vx_uint8 key = sv[i]; vx_int32 j = i - 1;
                                while (j >= 0 && sv[j] > key) { sv[j+1] = sv[j]; j--; }
                                sv[j+1] = key;
                            }
                            r = sv[active_count >> 1];
                        }
                        drow[x] = r;
                    }
                }
                simd_did_interior = (iy0 < iy1) && (ix0 < ix1);
            }
        }

        for (y = low_y; y < high_y; y++)
        {
            bool y_in_simd = simd_did_interior && (y >= iy0 && y < iy1);
            vx_uint32 xs_lo = low_x, xs_hi = high_x;
            // If SIMD handled this row's interior, only process the side borders here.
            if (y_in_simd) xs_hi = ix0;  // first scalar pass: [low_x, ix0)
            for (vx_uint32 pass = 0; pass < 2u; pass++)
            {
                for (x = xs_lo; x < xs_hi; x++)
                {
                    vx_uint32 xShftd = x + shift_x_u1;      // Bit-shift for U1 valid region start
                    vx_uint8 *dst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(dst_base, xShftd, y, &dst_addr);
                    vx_int32 count = (vx_int32)readMaskedRectangle(src_base, &src_addr, border, format, xShftd, y, (vx_uint32)rx0, (vx_uint32)ry0, (vx_uint32)rx1, (vx_uint32)ry1, m, v, shift_x_u1);

                    // Avoid the qsort per pixel: linear scan for min/max, or a 256-bucket
                    // histogram for median (count <= mrows*mcols <= 81).
                    switch (func)
                    {
                    case VX_NONLINEAR_FILTER_MIN:
                    {
                        vx_uint8 mn = v[0];
                        for (vx_int32 i = 1; i < count; i++) if (v[i] < mn) mn = v[i];
                        res_val = mn;
                        break;
                    }
                    case VX_NONLINEAR_FILTER_MAX:
                    {
                        vx_uint8 mx = v[0];
                        for (vx_int32 i = 1; i < count; i++) if (v[i] > mx) mx = v[i];
                        res_val = mx;
                        break;
                    }
                    case VX_NONLINEAR_FILTER_MEDIAN:
                    {
                        // Insertion sort is fastest for the small counts we see (<= 81).
                        for (vx_int32 i = 1; i < count; i++)
                        {
                            vx_uint8 key = v[i];
                            vx_int32 j = i - 1;
                            while (j >= 0 && v[j] > key) { v[j+1] = v[j]; j--; }
                            v[j+1] = key;
                        }
                        res_val = v[count >> 1];
                        break;
                    }
                    }
                    if (format == VX_DF_IMAGE_U1)
                    {
                        *dst_ptr = (*dst_ptr & ~(1 << (xShftd % 8))) | (res_val << (xShftd % 8));
                    }
                    else
                        *dst_ptr = res_val;
                }
                if (!y_in_simd) break;
                // second pass: [ix1, high_x)
                xs_lo = ix1; xs_hi = high_x;
            }
        }
    }

    status |= vxCommitImagePatch(src, NULL, 0, &src_addr, src_base);
    status |= vxCommitImagePatch(dst, &rect, 0, &dst_addr, dst_base);
    
    return status;
}

#define C_MAX_CONVOLUTION_DIM (15)

static const vx_int16 gaussian5x5[5][5] =
{
    {1,  4,  6,  4, 1},
    {4, 16, 24, 16, 4},
    {6, 24, 36, 24, 6},
    {4, 16, 24, 16, 4},
    {1,  4,  6,  4, 1}
};

static const vx_uint32 gaussian5x5scale = 256;

static vx_convolution vxCreateGaussian5x5Convolution(vx_context context)
{
    vx_convolution conv = vxCreateConvolution(context, 5, 5);
    vx_status status = vxCopyConvolutionCoefficients(conv, (vx_int16 *)gaussian5x5, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }

    status = vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, (void *)&gaussian5x5scale, sizeof(vx_uint32));
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }
    return conv;
}

static vx_status ownCopyImage(vx_image input, vx_image output)
{
    vx_status status = VX_SUCCESS; // assume success until an error occurs.
    vx_uint32 p = 0;
    vx_uint32 y = 0, x = 0;
    vx_size planes = 0;

    void* src;
    void* dst;
    vx_imagepatch_addressing_t src_addr;
    vx_imagepatch_addressing_t dst_addr;
    vx_rectangle_t src_rect, dst_rect;
    vx_map_id map_id1;
    vx_map_id map_id2;
    vx_df_image src_format = 0;
    vx_df_image out_format = 0;

    status |= vxQueryImage(input, VX_IMAGE_PLANES, &planes, sizeof(planes));
    vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    vxQueryImage(input, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxGetValidRegionImage(input, &src_rect);
    status |= vxGetValidRegionImage(output, &dst_rect);
    for (p = 0; p < planes && status == VX_SUCCESS; p++)
    {
        status = VX_SUCCESS;
        src = NULL;
        dst = NULL;

        status |= vxMapImagePatch(input, &src_rect, p, &map_id1, &src_addr, &src, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        status |= vxMapImagePatch(output, &dst_rect, p, &map_id2, &dst_addr, &dst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
        for (y = 0; y < src_addr.dim_y && status == VX_SUCCESS; y += src_addr.step_y)
        {
            for (x = 0; x < src_addr.dim_x && status == VX_SUCCESS; x += src_addr.step_x)
            {
                void* srcp = vxFormatImagePatchAddress2d(src, x, y, &src_addr);
                void* dstp = vxFormatImagePatchAddress2d(dst, x, y, &dst_addr);
                vx_int32 out0 = src_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)srcp : *(vx_int16 *)srcp;

                if (out_format == VX_DF_IMAGE_U8)
                {
                    if (out0 > UINT8_MAX)
                        out0 = UINT8_MAX;
                    else if (out0 < 0)
                        out0 = 0;
                    *(vx_uint8 *)dstp = (vx_uint8)out0;
                }
                else
                {
                    if (out0 > INT16_MAX)
                        out0 = INT16_MAX;
                    else if (out0 < INT16_MIN)
                        out0 = INT16_MIN;
                    *(vx_int16 *)dstp = (vx_int16)out0;
                }
            }
        }

        if (status == VX_SUCCESS)
        {
            status |= vxUnmapImagePatch(input, map_id1);
            status |= vxUnmapImagePatch(output, map_id2);
        }
    }

    return status;
}

static void auxReadRect(const void *base, const vx_imagepatch_addressing_t *addr, const vx_border_t *borders, vx_df_image type,
    vx_uint32 center_x, vx_uint32 center_y, vx_uint32 radius_x, vx_uint32 radius_y, void *destination)
{
    vx_int32 width = (vx_int32)addr->dim_x, height = (vx_int32)addr->dim_y;
    vx_int32 stride_y = addr->stride_y;
    vx_int32 stride_x = addr->stride_x;
    const vx_uint8 *ptr = (const vx_uint8 *)base;
    vx_int32 ky, kx;
    vx_uint32 dest_index = 0;
    // kx, kx - kernel x and y
    if (borders->mode == VX_BORDER_REPLICATE || borders->mode == VX_BORDER_UNDEFINED)
    {
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            y = y < 0 ? 0 : (y >= height ? height - 1 : y);

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                x = x < 0 ? 0 : (x >= width ? width - 1 : x);

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else if (borders->mode == VX_BORDER_CONSTANT)
    {
        vx_pixel_value_t cval = borders->constant_value;
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            int ccase_y = y < 0 || y >= height;

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                int ccase = ccase_y || x < 0 || x >= width;

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    if (!ccase)
                        ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint8*)destination)[dest_index] = (vx_uint8)cval.U8;
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    if (!ccase)
                        ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint16*)destination)[dest_index] = (vx_uint16)cval.U16;
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    if (!ccase)
                        ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint32*)destination)[dest_index] = (vx_uint32)cval.U32;
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else
        abort();
}

#define CONV_DIM 5
#define CONV_DIM_HALF CONV_DIM / 2

#define INSERT_ZERO_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(1-y)+i] = 0;
#define INSERT_VALUES_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(high_y-y)+i+CONV_DIM_HALF*CONV_DIM] = slice[CONV_DIM*(high_y-y)+i];
#define INSERT_ZERO_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+1-x] = 0;
#define INSERT_VALUES_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+(high_x-x)+CONV_DIM_HALF] = slice[CONV_DIM*i+(high_x-x)];

static vx_status replicateConvolve(vx_image src, vx_convolution conv, vx_image dst, vx_border_t *bordermode)
{
    vx_int32 y, x, i;
    void *src_base = NULL;
    void *dst_base = NULL;
    vx_imagepatch_addressing_t src_addr, dst_addr;
    vx_rectangle_t rect;

    vx_size conv_width, conv_height;
    vx_int32 conv_radius_x, conv_radius_y;
    vx_int16 conv_mat[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };
    vx_int32 sum = 0, value = 0;
    vx_uint32 scale = 1;
    vx_df_image src_format = 0;
    vx_df_image dst_format = 0;
    vx_status status = VX_SUCCESS;
    vx_int32 low_x, low_y, high_x, high_y;

    status |= vxQueryImage(src, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxQueryImage(dst, VX_IMAGE_FORMAT, &dst_format, sizeof(dst_format));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_COLUMNS, &conv_width, sizeof(conv_width));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_ROWS, &conv_height, sizeof(conv_height));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
    conv_radius_x = (vx_int32)conv_width / 2;
    conv_radius_y = (vx_int32)conv_height / 2;
    status |= vxCopyConvolutionCoefficients(conv, conv_mat, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxGetValidRegionImage(src, &rect);
    status |= vxAccessImagePatch(src, &rect, 0, &src_addr, &src_base, VX_READ_ONLY);
    status |= vxAccessImagePatch(dst, &rect, 0, &dst_addr, &dst_base, VX_WRITE_ONLY);

    low_x = 0;
    high_x = src_addr.dim_x;
    low_y = 0;
    high_y = src_addr.dim_y;

    for (y = low_y; y < high_y; ++y)
    {
        for (x = low_x; x < high_x; ++x)
        {
            sum = 0;

            if (src_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                auxReadRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                // purpose of this section is to compensate extra terms caused by replicate border mode (it is the only one allowed)

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }
            else if (src_format == VX_DF_IMAGE_S16)
            {
                vx_int16 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                auxReadRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                // purpose of this section is to compensate extra terms caused by replicate border mode (it is the only one allowed)

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }

            value = sum / (vx_int32)scale;

            if (dst_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 *dstp = (vx_uint8 *)vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < 0) *dstp = 0;
                else if (value > UINT8_MAX) *dstp = UINT8_MAX;
                else *dstp = value;
            }
            else if (dst_format == VX_DF_IMAGE_S16)
            {
                vx_int16 *dstp = (vx_int16 *)vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < INT16_MIN) *dstp = INT16_MIN;
                else if (value > INT16_MAX) *dstp = INT16_MAX;
                else *dstp = value;
            }
        }
    }

    status |= vxCommitImagePatch(src, NULL, 0, &src_addr, src_base);
    status |= vxCommitImagePatch(dst, &rect, 0, &dst_addr, dst_base);

    return status;
}

// Fast pyramid-up with 5x5 Gaussian filter for U8 input -> U8/S16 output.
// Combines zero-stuffing, 5x5 separable Gaussian (kernel = {1,4,6,4,1}/16 x same) and *4 scaling
// into a single pass using two 1D passes on the original source, exploiting the fact that
// 4 of every 5 columns and rows in the zero-stuffed image contribute zero. This eliminates
// ~25x per-pixel work compared to the original scalar replicateConvolve path.
// Border behaviour matches a tmp-coordinate VX_BORDER_REPLICATE convolution: replicate at
// top/left edges, zero contribution where the kernel extends past the bottom/right edge.
static void HafCpu_PyramidUp_Gaussian5x5_U8(
    const vx_uint8 *src, vx_int32 srcStride, vx_int32 srcW, vx_int32 srcH,
    void *dst, vx_int32 dstStride, vx_int32 dstW, vx_int32 dstH,
    bool dst_is_s16)
{
    // Vertical-pass buffer: extra 1 slot on each side, indexed as V[1..srcW] = real cols.
    // V[0] := V[1] (left replicate); V[srcW+1] := 0 (right zero); V[srcW+2] := 0 (pad).
    // Add extra padding to allow safe AVX2 over-reads in the horizontal pass.
    std::vector<vx_int16> Vbuf((size_t)srcW + 18);
    vx_int16 *V = Vbuf.data();

    const __m128i zero128 = _mm_setzero_si128();
    const __m128i mul6_128 = _mm_set1_epi16(6);
#if USE_AVX
    const __m256i zero256 = _mm256_setzero_si256();
    const __m256i mul6_256 = _mm256_set1_epi16(6);
#endif

    for (vx_int32 y = 0; y < dstH; y++)
    {
        bool y_even = (y & 1) == 0;
        vx_int32 fy = y >> 1;

        vx_int32 fy_top = fy - 1;
        vx_int32 fy_mid = fy;
        vx_int32 fy_bot = fy + 1;
        // Source-level replicate at both ends (matches CTS reference convolve on
        // the zero-stuffed image, including INSERT_ZERO_Y/INSERT_VALUES_Y).
        if (fy_top < 0) fy_top = 0;
        if (fy_mid >= srcH) fy_mid = srcH - 1;
        if (fy_bot >= srcH) fy_bot = srcH - 1;

        const vx_uint8 *rm = y_even ? (src + (size_t)fy_top * srcStride) : nullptr;
        const vx_uint8 *r0 = (src + (size_t)fy_mid * srcStride);
        const vx_uint8 *rp = (src + (size_t)fy_bot * srcStride);

        // Vertical pass: V[fx] in [0, srcW) stored to V[fx + 1].
        vx_int32 fx = 0;
        if (y_even)
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i a = rm ? _mm256_loadu_si256((const __m256i *)(rm + fx)) : zero256;
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i a_lo = _mm256_unpacklo_epi8(a, zero256);
                __m256i a_hi = _mm256_unpackhi_epi8(a, zero256);
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_add_epi16(_mm256_add_epi16(a_lo, c_lo), _mm256_mullo_epi16(b_lo, mul6_256));
                __m256i vhi = _mm256_add_epi16(_mm256_add_epi16(a_hi, c_hi), _mm256_mullo_epi16(b_hi, mul6_256));
                // unpack interleaves 128-bit lanes, so de-interleave with permute
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i a = rm ? _mm_loadu_si128((const __m128i *)(rm + fx)) : zero128;
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i a_lo = _mm_unpacklo_epi8(a, zero128);
                __m128i a_hi = _mm_unpackhi_epi8(a, zero128);
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_add_epi16(_mm_add_epi16(a_lo, c_lo), _mm_mullo_epi16(b_lo, mul6_128));
                __m128i vhi = _mm_add_epi16(_mm_add_epi16(a_hi, c_hi), _mm_mullo_epi16(b_hi, mul6_128));
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 av = rm ? rm[fx] : 0;
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(av + 6 * bv + cv);
            }
        }
        else // y_odd: V = 4*(r0 + rp)
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_slli_epi16(_mm256_add_epi16(b_lo, c_lo), 2);
                __m256i vhi = _mm256_slli_epi16(_mm256_add_epi16(b_hi, c_hi), 2);
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_slli_epi16(_mm_add_epi16(b_lo, c_lo), 2);
                __m128i vhi = _mm_slli_epi16(_mm_add_epi16(b_hi, c_hi), 2);
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(4 * (bv + cv));
            }
        }
        V[0] = V[1];                       // left replicate
        // Right replicate: matches CTS reference convolve at the right edge
        // (INSERT_VALUES_X copies V[srcW] outward for x >= dstW - 2*kernelHalf).
        V[srcW + 1] = V[srcW];
        V[srcW + 2] = V[srcW];
        // pad the AVX2 over-read range with the same replicate so wide SIMD loads at
        // fx = srcW - 15 still see well-defined values
        for (int i = 3; i < 18; i++) V[srcW + i] = V[srcW];

        // Horizontal pass.
        if (!dst_is_s16)
        {
            vx_uint8 *drow = (vx_uint8 *)dst + (size_t)y * dstStride;
            fx = 0;
#if USE_AVX
            // Process 16 source cols -> 32 dst bytes.
            vx_int32 fx_max256 = (srcW >= 16) ? srcW - 16 : 0;
            for (; fx <= fx_max256 && 2*fx + 32 <= dstW; fx += 16)
            {
                __m256i v_left   = _mm256_loadu_si256((const __m256i *)(V + fx));
                __m256i v_center = _mm256_loadu_si256((const __m256i *)(V + fx + 1));
                __m256i v_right  = _mm256_loadu_si256((const __m256i *)(V + fx + 2));
                __m256i h_even = _mm256_add_epi16(_mm256_add_epi16(v_left, v_right), _mm256_mullo_epi16(v_center, mul6_256));
                __m256i h_odd  = _mm256_slli_epi16(_mm256_add_epi16(v_center, v_right), 2);
                // Spec computes (sum / 256) * 4, NOT sum / 64. The integer division
                // before the multiply discards the lower 8 bits, so use (h >> 8) << 2.
                __m256i out_even = _mm256_slli_epi16(_mm256_srai_epi16(h_even, 8), 2);
                __m256i out_odd  = _mm256_slli_epi16(_mm256_srai_epi16(h_odd, 8), 2);
                __m256i out_lo = _mm256_unpacklo_epi16(out_even, out_odd);  // 128-bit lanes interleave separately
                __m256i out_hi = _mm256_unpackhi_epi16(out_even, out_odd);
                __m256i packed = _mm256_packus_epi16(out_lo, out_hi);       // also lane-wise
                // packed layout: [lane0: lo0..7,hi0..7][lane1: lo8..15,hi8..15] -> 32 bytes
                // Need permute to fix: we want sequential output (idx 0..31)
                // After unpacklo: even-odd interleaved within each lane (8 pairs)
                // After packus: 16 bytes from out_lo then 16 from out_hi, per lane
                // Goal: positions 0..15 from lane0, 16..31 from lane1
                // packed already has lane0 first (16 bytes from lo+hi of lane0), lane1 next
                // But out_lo has unpacklo of [lo0..3, hi0..3, lo8..11, hi8..11]
                // Let's compute via simpler approach: write 128-bit halves
                __m128i pack_lo = _mm256_castsi256_si128(packed);
                __m128i pack_hi = _mm256_extracti128_si256(packed, 1);
                _mm_storeu_si128((__m128i *)(drow + 2*fx), pack_lo);
                _mm_storeu_si128((__m128i *)(drow + 2*fx + 16), pack_hi);
            }
#endif
            // Process 8 source cols -> 16 dst bytes.
            vx_int32 fx_max = (srcW >= 8) ? srcW - 8 : 0;
            for (; fx <= fx_max && 2*fx + 16 <= dstW; fx += 8)
            {
                __m128i v_left   = _mm_loadu_si128((const __m128i *)(V + fx));
                __m128i v_center = _mm_loadu_si128((const __m128i *)(V + fx + 1));
                __m128i v_right  = _mm_loadu_si128((const __m128i *)(V + fx + 2));
                __m128i h_even = _mm_add_epi16(_mm_add_epi16(v_left, v_right), _mm_mullo_epi16(v_center, mul6_128));
                __m128i h_odd  = _mm_slli_epi16(_mm_add_epi16(v_center, v_right), 2);
                __m128i out_even = _mm_slli_epi16(_mm_srai_epi16(h_even, 8), 2);
                __m128i out_odd  = _mm_slli_epi16(_mm_srai_epi16(h_odd, 8), 2);
                __m128i out_lo = _mm_unpacklo_epi16(out_even, out_odd);
                __m128i out_hi = _mm_unpackhi_epi16(out_even, out_odd);
                __m128i packed = _mm_packus_epi16(out_lo, out_hi);
                _mm_storeu_si128((__m128i *)(drow + 2*fx), packed);
            }
            for (vx_int32 x = 2*fx; x < dstW; x++)
            {
                vx_int32 fxx = x >> 1;
                vx_int32 H = ((x & 1) == 0)
                    ? (V[fxx] + 6 * V[fxx + 1] + V[fxx + 2])
                    : (4 * (V[fxx + 1] + V[fxx + 2]));
                vx_int32 dv = (H >> 8) << 2;
                if (dv > 255) dv = 255;
                else if (dv < 0) dv = 0;
                drow[x] = (vx_uint8)dv;
            }
        }
        else
        {
            vx_int16 *drow = (vx_int16 *)((vx_uint8 *)dst + (size_t)y * dstStride);
            fx = 0;
#if USE_AVX
            vx_int32 fx_max256 = (srcW >= 16) ? srcW - 16 : 0;
            for (; fx <= fx_max256 && 2*fx + 32 <= dstW; fx += 16)
            {
                __m256i v_left   = _mm256_loadu_si256((const __m256i *)(V + fx));
                __m256i v_center = _mm256_loadu_si256((const __m256i *)(V + fx + 1));
                __m256i v_right  = _mm256_loadu_si256((const __m256i *)(V + fx + 2));
                __m256i h_even = _mm256_add_epi16(_mm256_add_epi16(v_left, v_right), _mm256_mullo_epi16(v_center, mul6_256));
                __m256i h_odd  = _mm256_slli_epi16(_mm256_add_epi16(v_center, v_right), 2);
                // Spec computes (sum / 256) * 4, NOT sum / 64.
                __m256i out_even = _mm256_slli_epi16(_mm256_srai_epi16(h_even, 8), 2);
                __m256i out_odd  = _mm256_slli_epi16(_mm256_srai_epi16(h_odd, 8), 2);
                __m256i interlo = _mm256_unpacklo_epi16(out_even, out_odd); // lane-wise
                __m256i interhi = _mm256_unpackhi_epi16(out_even, out_odd);
                // Each lane has 8 sequential 16-bit values. Need to write in correct order:
                // First write lane0 of interlo, then lane0 of interhi, then lane1 of interlo, then lane1 of interhi.
                __m128i lo_lo = _mm256_castsi256_si128(interlo);
                __m128i hi_lo = _mm256_castsi256_si128(interhi);
                __m128i lo_hi = _mm256_extracti128_si256(interlo, 1);
                __m128i hi_hi = _mm256_extracti128_si256(interhi, 1);
                _mm_storeu_si128((__m128i *)(drow + 2*fx), lo_lo);
                _mm_storeu_si128((__m128i *)(drow + 2*fx + 8), hi_lo);
                _mm_storeu_si128((__m128i *)(drow + 2*fx + 16), lo_hi);
                _mm_storeu_si128((__m128i *)(drow + 2*fx + 24), hi_hi);
            }
#endif
            vx_int32 fx_max = (srcW >= 8) ? srcW - 8 : 0;
            for (; fx <= fx_max && 2*fx + 16 <= dstW; fx += 8)
            {
                __m128i v_left   = _mm_loadu_si128((const __m128i *)(V + fx));
                __m128i v_center = _mm_loadu_si128((const __m128i *)(V + fx + 1));
                __m128i v_right  = _mm_loadu_si128((const __m128i *)(V + fx + 2));
                __m128i h_even = _mm_add_epi16(_mm_add_epi16(v_left, v_right), _mm_mullo_epi16(v_center, mul6_128));
                __m128i h_odd  = _mm_slli_epi16(_mm_add_epi16(v_center, v_right), 2);
                __m128i out_even = _mm_slli_epi16(_mm_srai_epi16(h_even, 8), 2);
                __m128i out_odd  = _mm_slli_epi16(_mm_srai_epi16(h_odd, 8), 2);
                __m128i out_lo = _mm_unpacklo_epi16(out_even, out_odd);
                __m128i out_hi = _mm_unpackhi_epi16(out_even, out_odd);
                _mm_storeu_si128((__m128i *)(drow + 2*fx), out_lo);
                _mm_storeu_si128((__m128i *)(drow + 2*fx + 8), out_hi);
            }
            for (vx_int32 x = 2*fx; x < dstW; x++)
            {
                vx_int32 fxx = x >> 1;
                vx_int32 H = ((x & 1) == 0)
                    ? (V[fxx] + 6 * V[fxx + 1] + V[fxx + 2])
                    : (4 * (V[fxx + 1] + V[fxx + 2]));
                vx_int32 dv = (H >> 8) << 2;
                if (dv > INT16_MAX) dv = INT16_MAX;
                else if (dv < INT16_MIN) dv = INT16_MIN;
                drow[x] = (vx_int16)dv;
            }
        }
    }
}

// Fused PyramidUp (5x5 Gaussian, replicate border) + saturate subtract:
//   out_s16[y,x] = saturate_s16(src_u8[y,x] - upsampled_s16[y,x])
// where upsampled_s16 = HafCpu_PyramidUp_Gaussian5x5_U8(filling -> S16, dst size = src size).
// Eliminates the intermediate S16 image and the vxuSubtract graph overhead used by
// the legacy LaplacianPyramid flow.
static void HafCpu_PyramidUp_Gaussian5x5_Subtract_U8(
    const vx_uint8 *fill, vx_int32 fillStride, vx_int32 srcW, vx_int32 srcH,
    const vx_uint8 *src,  vx_int32 srcStride,
    vx_int16 *dst, vx_int32 dstStride, vx_int32 dstW, vx_int32 dstH)
{
    // Vertical buffer (same layout as HafCpu_PyramidUp_Gaussian5x5_U8).
    std::vector<vx_int16> Vbuf((size_t)srcW + 18);
    vx_int16 *V = Vbuf.data();

    const __m128i zero128 = _mm_setzero_si128();
    const __m128i mul6_128 = _mm_set1_epi16(6);
#if USE_AVX
    const __m256i zero256 = _mm256_setzero_si256();
    const __m256i mul6_256 = _mm256_set1_epi16(6);
#endif

    for (vx_int32 y = 0; y < dstH; y++)
    {
        bool y_even = (y & 1) == 0;
        vx_int32 fy = y >> 1;
        vx_int32 fy_top = fy - 1;
        vx_int32 fy_mid = fy;
        vx_int32 fy_bot = fy + 1;
        // Source-level replicate at both ends (matches CTS reference).
        if (fy_top < 0) fy_top = 0;
        if (fy_mid >= srcH) fy_mid = srcH - 1;
        if (fy_bot >= srcH) fy_bot = srcH - 1;
        const vx_uint8 *rm = y_even ? (fill + (size_t)fy_top * fillStride) : nullptr;
        const vx_uint8 *r0 = (fill + (size_t)fy_mid * fillStride);
        const vx_uint8 *rp = (fill + (size_t)fy_bot * fillStride);

        vx_int32 fx = 0;
        if (y_even)
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i a = rm ? _mm256_loadu_si256((const __m256i *)(rm + fx)) : zero256;
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i a_lo = _mm256_unpacklo_epi8(a, zero256);
                __m256i a_hi = _mm256_unpackhi_epi8(a, zero256);
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_add_epi16(_mm256_add_epi16(a_lo, c_lo), _mm256_mullo_epi16(b_lo, mul6_256));
                __m256i vhi = _mm256_add_epi16(_mm256_add_epi16(a_hi, c_hi), _mm256_mullo_epi16(b_hi, mul6_256));
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i a = rm ? _mm_loadu_si128((const __m128i *)(rm + fx)) : zero128;
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i a_lo = _mm_unpacklo_epi8(a, zero128);
                __m128i a_hi = _mm_unpackhi_epi8(a, zero128);
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_add_epi16(_mm_add_epi16(a_lo, c_lo), _mm_mullo_epi16(b_lo, mul6_128));
                __m128i vhi = _mm_add_epi16(_mm_add_epi16(a_hi, c_hi), _mm_mullo_epi16(b_hi, mul6_128));
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 av = rm ? rm[fx] : 0;
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(av + 6 * bv + cv);
            }
        }
        else
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_slli_epi16(_mm256_add_epi16(b_lo, c_lo), 2);
                __m256i vhi = _mm256_slli_epi16(_mm256_add_epi16(b_hi, c_hi), 2);
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_slli_epi16(_mm_add_epi16(b_lo, c_lo), 2);
                __m128i vhi = _mm_slli_epi16(_mm_add_epi16(b_hi, c_hi), 2);
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(4 * (bv + cv));
            }
        }
        V[0] = V[1];                       // left replicate
        V[srcW + 1] = V[srcW];             // right replicate (CTS reference INSERT_VALUES_X)
        V[srcW + 2] = V[srcW];
        for (int i = 3; i < 18; i++) V[srcW + i] = V[srcW];

        // Horizontal pass + subtract: produces 16 dst S16 values from 8 source cols at a time.
        const vx_uint8 *srow = src + (size_t)y * srcStride;
        vx_int16 *drow = dst + (size_t)y * (dstStride / (vx_int32)sizeof(vx_int16));
        fx = 0;
#if USE_AVX
        vx_int32 fx_max256 = (srcW >= 16) ? srcW - 16 : 0;
        for (; fx <= fx_max256 && 2*fx + 32 <= dstW; fx += 16)
        {
            __m256i v_left   = _mm256_loadu_si256((const __m256i *)(V + fx));
            __m256i v_center = _mm256_loadu_si256((const __m256i *)(V + fx + 1));
            __m256i v_right  = _mm256_loadu_si256((const __m256i *)(V + fx + 2));
            __m256i h_even = _mm256_add_epi16(_mm256_add_epi16(v_left, v_right), _mm256_mullo_epi16(v_center, mul6_256));
            __m256i h_odd  = _mm256_slli_epi16(_mm256_add_epi16(v_center, v_right), 2);
            // Spec computes (sum / 256) * 4, NOT sum / 64.
            __m256i out_even = _mm256_slli_epi16(_mm256_srai_epi16(h_even, 8), 2);
            __m256i out_odd  = _mm256_slli_epi16(_mm256_srai_epi16(h_odd, 8), 2);
            __m256i interlo = _mm256_unpacklo_epi16(out_even, out_odd);
            __m256i interhi = _mm256_unpackhi_epi16(out_even, out_odd);
            // Load 32 U8 source pixels and widen to two __m256i s16 vectors in the correct order.
            __m256i src32 = _mm256_loadu_si256((const __m256i *)(srow + 2*fx));
            __m128i s_lo = _mm256_castsi256_si128(src32);
            __m128i s_hi = _mm256_extracti128_si256(src32, 1);
            __m256i src_w0 = _mm256_cvtepu8_epi16(s_lo);   // 16 s16: src[0..15]
            __m256i src_w1 = _mm256_cvtepu8_epi16(s_hi);   // 16 s16: src[16..31]
            // 'interlo' lane0 holds out[0..7] interleaved (positions 0..15 of dst), lane1 holds out[16..23]
            // 'interhi' lane0 holds out[8..15] (positions 16..31 of dst), wait re-examine.
            // unpacklo/hi each works per 128-bit lane:
            //   interlo lane0: pairs from out_even[0..3], out_odd[0..3] => dst positions 0..7
            //   interhi lane0: pairs from out_even[4..7], out_odd[4..7] => dst positions 8..15
            //   interlo lane1: pairs from out_even[8..11], out_odd[8..11] => dst positions 16..23
            //   interhi lane1: pairs from out_even[12..15], out_odd[12..15] => dst positions 24..31
            // Sequential write order: lane0_lo, lane0_hi, lane1_lo, lane1_hi.
            __m128i d0 = _mm256_castsi256_si128(interlo);
            __m128i d1 = _mm256_castsi256_si128(interhi);
            __m128i d2 = _mm256_extracti128_si256(interlo, 1);
            __m128i d3 = _mm256_extracti128_si256(interhi, 1);
            __m256i up_lo = _mm256_setr_m128i(d0, d1); // dst positions 0..15
            __m256i up_hi = _mm256_setr_m128i(d2, d3); // dst positions 16..31
            // Saturating subtract: src(u8 widened to s16) - upsample(s16). Range fits in s16 since both inputs are in [-32768, 32767].
            __m256i out_lo = _mm256_subs_epi16(src_w0, up_lo);
            __m256i out_hi = _mm256_subs_epi16(src_w1, up_hi);
            _mm256_storeu_si256((__m256i *)(drow + 2*fx),      out_lo);
            _mm256_storeu_si256((__m256i *)(drow + 2*fx + 16), out_hi);
        }
#endif
        vx_int32 fx_max = (srcW >= 8) ? srcW - 8 : 0;
        for (; fx <= fx_max && 2*fx + 16 <= dstW; fx += 8)
        {
            __m128i v_left   = _mm_loadu_si128((const __m128i *)(V + fx));
            __m128i v_center = _mm_loadu_si128((const __m128i *)(V + fx + 1));
            __m128i v_right  = _mm_loadu_si128((const __m128i *)(V + fx + 2));
            __m128i h_even = _mm_add_epi16(_mm_add_epi16(v_left, v_right), _mm_mullo_epi16(v_center, mul6_128));
            __m128i h_odd  = _mm_slli_epi16(_mm_add_epi16(v_center, v_right), 2);
            __m128i out_even = _mm_slli_epi16(_mm_srai_epi16(h_even, 8), 2);
            __m128i out_odd  = _mm_slli_epi16(_mm_srai_epi16(h_odd, 8), 2);
            __m128i up_lo = _mm_unpacklo_epi16(out_even, out_odd);
            __m128i up_hi = _mm_unpackhi_epi16(out_even, out_odd);
            __m128i src16 = _mm_loadu_si128((const __m128i *)(srow + 2*fx));
            __m128i src_w_lo = _mm_cvtepu8_epi16(src16);
            __m128i src_w_hi = _mm_cvtepu8_epi16(_mm_srli_si128(src16, 8));
            __m128i out_lo = _mm_subs_epi16(src_w_lo, up_lo);
            __m128i out_hi = _mm_subs_epi16(src_w_hi, up_hi);
            _mm_storeu_si128((__m128i *)(drow + 2*fx),     out_lo);
            _mm_storeu_si128((__m128i *)(drow + 2*fx + 8), out_hi);
        }
        for (vx_int32 x = 2*fx; x < dstW; x++)
        {
            vx_int32 fxx = x >> 1;
            vx_int32 H = ((x & 1) == 0)
                ? (V[fxx] + 6 * V[fxx + 1] + V[fxx + 2])
                : (4 * (V[fxx + 1] + V[fxx + 2]));
            vx_int32 dv = (H >> 8) << 2;
            // Saturating subtract to s16
            vx_int32 sub = (vx_int32)srow[x] - dv;
            if (sub > INT16_MAX) sub = INT16_MAX;
            else if (sub < INT16_MIN) sub = INT16_MIN;
            drow[x] = (vx_int16)sub;
        }
    }
}

// Fused PyramidUp (5x5 Gaussian, replicate border) + saturating add:
//   recon[y,x] = saturate(upsampled_s16[y,x] + laplacian_s16[y,x])
// where upsampled_s16 = HafCpu_PyramidUp_Gaussian5x5(fill_u8 -> S16, dst size = laplacian size),
// using the exact (sum >> 8) << 2 spec formula the upsampleImage fast path already
// emits. This is the reconstruction counterpart of HafCpu_PyramidUp_Gaussian5x5_Subtract_U8
// and lets LaplacianReconstruct avoid the per-level immediate-mode vxuAdd graph
// (build/verify/execute/teardown) plus the scalar ownCopyImage that made it ~38x
// slower than OpenCV.
//
// `fill` is the (already U8-saturated) lower-resolution level. `lap` is the S16
// laplacian level at the destination resolution. When `out_is_s16` the result is
// written as S16 saturated to [INT16_MIN, INT16_MAX] (final output image is S16);
// otherwise it is written as U8 saturated to [0, 255] (intermediate levels feed the
// next upsample as U8, exactly as the legacy path saturated the S16 sum back to U8
// before the next upsampleImage call, and the final U8 output image).
static void HafCpu_PyramidUp_Gaussian5x5_Add(
    const vx_uint8 *fill, vx_int32 fillStride, vx_int32 srcW, vx_int32 srcH,
    const vx_int16 *lap, vx_int32 lapStride,
    void *dst, vx_int32 dstStride, vx_int32 dstW, vx_int32 dstH,
    bool out_is_s16)
{
    // Vertical buffer (same layout as HafCpu_PyramidUp_Gaussian5x5_U8).
    std::vector<vx_int16> Vbuf((size_t)srcW + 18);
    vx_int16 *V = Vbuf.data();

    const __m128i zero128 = _mm_setzero_si128();
    const __m128i mul6_128 = _mm_set1_epi16(6);
#if USE_AVX
    const __m256i zero256 = _mm256_setzero_si256();
    const __m256i mul6_256 = _mm256_set1_epi16(6);
#endif

    for (vx_int32 y = 0; y < dstH; y++)
    {
        bool y_even = (y & 1) == 0;
        vx_int32 fy = y >> 1;
        vx_int32 fy_top = fy - 1;
        vx_int32 fy_mid = fy;
        vx_int32 fy_bot = fy + 1;
        if (fy_top < 0) fy_top = 0;
        if (fy_mid >= srcH) fy_mid = srcH - 1;
        if (fy_bot >= srcH) fy_bot = srcH - 1;
        const vx_uint8 *rm = y_even ? (fill + (size_t)fy_top * fillStride) : nullptr;
        const vx_uint8 *r0 = (fill + (size_t)fy_mid * fillStride);
        const vx_uint8 *rp = (fill + (size_t)fy_bot * fillStride);

        vx_int32 fx = 0;
        if (y_even)
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i a = rm ? _mm256_loadu_si256((const __m256i *)(rm + fx)) : zero256;
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i a_lo = _mm256_unpacklo_epi8(a, zero256);
                __m256i a_hi = _mm256_unpackhi_epi8(a, zero256);
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_add_epi16(_mm256_add_epi16(a_lo, c_lo), _mm256_mullo_epi16(b_lo, mul6_256));
                __m256i vhi = _mm256_add_epi16(_mm256_add_epi16(a_hi, c_hi), _mm256_mullo_epi16(b_hi, mul6_256));
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i a = rm ? _mm_loadu_si128((const __m128i *)(rm + fx)) : zero128;
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i a_lo = _mm_unpacklo_epi8(a, zero128);
                __m128i a_hi = _mm_unpackhi_epi8(a, zero128);
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_add_epi16(_mm_add_epi16(a_lo, c_lo), _mm_mullo_epi16(b_lo, mul6_128));
                __m128i vhi = _mm_add_epi16(_mm_add_epi16(a_hi, c_hi), _mm_mullo_epi16(b_hi, mul6_128));
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 av = rm ? rm[fx] : 0;
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(av + 6 * bv + cv);
            }
        }
        else
        {
#if USE_AVX
            for (; fx + 32 <= srcW; fx += 32)
            {
                __m256i b = r0 ? _mm256_loadu_si256((const __m256i *)(r0 + fx)) : zero256;
                __m256i c = rp ? _mm256_loadu_si256((const __m256i *)(rp + fx)) : zero256;
                __m256i b_lo = _mm256_unpacklo_epi8(b, zero256);
                __m256i b_hi = _mm256_unpackhi_epi8(b, zero256);
                __m256i c_lo = _mm256_unpacklo_epi8(c, zero256);
                __m256i c_hi = _mm256_unpackhi_epi8(c, zero256);
                __m256i vlo = _mm256_slli_epi16(_mm256_add_epi16(b_lo, c_lo), 2);
                __m256i vhi = _mm256_slli_epi16(_mm256_add_epi16(b_hi, c_hi), 2);
                __m256i out_lo = _mm256_permute2x128_si256(vlo, vhi, 0x20);
                __m256i out_hi = _mm256_permute2x128_si256(vlo, vhi, 0x31);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx), out_lo);
                _mm256_storeu_si256((__m256i *)(V + 1 + fx + 16), out_hi);
            }
#endif
            for (; fx + 16 <= srcW; fx += 16)
            {
                __m128i b = r0 ? _mm_loadu_si128((const __m128i *)(r0 + fx)) : zero128;
                __m128i c = rp ? _mm_loadu_si128((const __m128i *)(rp + fx)) : zero128;
                __m128i b_lo = _mm_unpacklo_epi8(b, zero128);
                __m128i b_hi = _mm_unpackhi_epi8(b, zero128);
                __m128i c_lo = _mm_unpacklo_epi8(c, zero128);
                __m128i c_hi = _mm_unpackhi_epi8(c, zero128);
                __m128i vlo = _mm_slli_epi16(_mm_add_epi16(b_lo, c_lo), 2);
                __m128i vhi = _mm_slli_epi16(_mm_add_epi16(b_hi, c_hi), 2);
                _mm_storeu_si128((__m128i *)(V + 1 + fx), vlo);
                _mm_storeu_si128((__m128i *)(V + 1 + fx + 8), vhi);
            }
            for (; fx < srcW; fx++)
            {
                vx_int16 bv = r0 ? r0[fx] : 0;
                vx_int16 cv = rp ? rp[fx] : 0;
                V[1 + fx] = (vx_int16)(4 * (bv + cv));
            }
        }
        V[0] = V[1];                       // left replicate
        V[srcW + 1] = V[srcW];             // right replicate (CTS reference INSERT_VALUES_X)
        V[srcW + 2] = V[srcW];
        for (int i = 3; i < 18; i++) V[srcW + i] = V[srcW];

        // Horizontal pass + add laplacian, at 128-bit (8 source cols -> 16 dst) granularity.
        const vx_int16 *lrow = (const vx_int16 *)((const vx_uint8 *)lap + (size_t)y * lapStride);
        vx_uint8 *drow8 = (vx_uint8 *)dst + (size_t)y * dstStride;
        vx_int16 *drow16 = (vx_int16 *)((vx_uint8 *)dst + (size_t)y * dstStride);
        fx = 0;
        vx_int32 fx_max = (srcW >= 8) ? srcW - 8 : 0;
        for (; fx <= fx_max && 2*fx + 16 <= dstW; fx += 8)
        {
            __m128i v_left   = _mm_loadu_si128((const __m128i *)(V + fx));
            __m128i v_center = _mm_loadu_si128((const __m128i *)(V + fx + 1));
            __m128i v_right  = _mm_loadu_si128((const __m128i *)(V + fx + 2));
            __m128i h_even = _mm_add_epi16(_mm_add_epi16(v_left, v_right), _mm_mullo_epi16(v_center, mul6_128));
            __m128i h_odd  = _mm_slli_epi16(_mm_add_epi16(v_center, v_right), 2);
            __m128i out_even = _mm_slli_epi16(_mm_srai_epi16(h_even, 8), 2);
            __m128i out_odd  = _mm_slli_epi16(_mm_srai_epi16(h_odd, 8), 2);
            __m128i up_lo = _mm_unpacklo_epi16(out_even, out_odd);  // dst[2fx .. 2fx+7]
            __m128i up_hi = _mm_unpackhi_epi16(out_even, out_odd);  // dst[2fx+8 .. 2fx+15]
            __m128i lap_lo = _mm_loadu_si128((const __m128i *)(lrow + 2*fx));
            __m128i lap_hi = _mm_loadu_si128((const __m128i *)(lrow + 2*fx + 8));
            __m128i sum_lo = _mm_adds_epi16(up_lo, lap_lo);  // saturate to s16
            __m128i sum_hi = _mm_adds_epi16(up_hi, lap_hi);
            if (out_is_s16)
            {
                _mm_storeu_si128((__m128i *)(drow16 + 2*fx),     sum_lo);
                _mm_storeu_si128((__m128i *)(drow16 + 2*fx + 8), sum_hi);
            }
            else
            {
                __m128i packed = _mm_packus_epi16(sum_lo, sum_hi); // saturate to u8
                _mm_storeu_si128((__m128i *)(drow8 + 2*fx), packed);
            }
        }
        for (vx_int32 x = 2*fx; x < dstW; x++)
        {
            vx_int32 fxx = x >> 1;
            vx_int32 H = ((x & 1) == 0)
                ? (V[fxx] + 6 * V[fxx + 1] + V[fxx + 2])
                : (4 * (V[fxx + 1] + V[fxx + 2]));
            vx_int32 dv = (H >> 8) << 2;
            vx_int32 sum = dv + (vx_int32)lrow[x];
            // Saturate to s16 first (matches the SATURATE policy vxuAdd applied to the
            // S16 intermediate), then narrow to u8 when the destination is U8.
            if (sum > INT16_MAX) sum = INT16_MAX;
            else if (sum < INT16_MIN) sum = INT16_MIN;
            if (out_is_s16)
            {
                drow16[x] = (vx_int16)sum;
            }
            else
            {
                if (sum > 255) sum = 255;
                else if (sum < 0) sum = 0;
                drow8[x] = (vx_uint8)sum;
            }
        }
    }
}

static vx_status upsampleImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_image filling, vx_convolution conv, vx_image upsample, vx_border_t *border)
{
    vx_status status = VX_SUCCESS;
    vx_df_image format, filling_format;

    format = VX_DF_IMAGE_U8;
    status |= vxQueryImage(filling, VX_IMAGE_FORMAT, &filling_format, sizeof(filling_format));

    // Fast direct upsample path: 5x5 Gaussian with scale 256 is the only convolution used by
    // LaplacianPyramid/LaplacianReconstruct. Combining the zero-stuff + 5x5 conv + x4 scaling
    // into a separable two-pass kernel cuts the original ~25 mul-adds/pixel scalar work down to
    // ~6 SIMD adds/shifts/pixel and removes the temporary U8 image entirely.
    {
        vx_size conv_w = 0, conv_h = 0;
        vx_uint32 conv_scale = 1;
        vxQueryConvolution(conv, VX_CONVOLUTION_COLUMNS, &conv_w, sizeof(conv_w));
        vxQueryConvolution(conv, VX_CONVOLUTION_ROWS, &conv_h, sizeof(conv_h));
        vxQueryConvolution(conv, VX_CONVOLUTION_SCALE, &conv_scale, sizeof(conv_scale));

        vx_df_image upsample_format = 0;
        vxQueryImage(upsample, VX_IMAGE_FORMAT, &upsample_format, sizeof(upsample_format));

        if (conv_w == 5 && conv_h == 5 && conv_scale == 256 &&
            border && border->mode == VX_BORDER_REPLICATE &&
            (filling_format == VX_DF_IMAGE_U8 || filling_format == VX_DF_IMAGE_S16) &&
            (upsample_format == VX_DF_IMAGE_U8 || upsample_format == VX_DF_IMAGE_S16))
        {
            vx_int16 coef[25];
            if (vxCopyConvolutionCoefficients(conv, coef, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) == VX_SUCCESS)
            {
                static const vx_int16 gauss_ref[25] = {
                    1,  4,  6,  4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1,  4,  6,  4, 1
                };
                bool is_gauss = true;
                for (int i = 0; i < 25; i++) if (coef[i] != gauss_ref[i]) { is_gauss = false; break; }

                if (is_gauss)
                {
                    vx_rectangle_t f_rect, u_rect;
                    vx_imagepatch_addressing_t f_addr = VX_IMAGEPATCH_ADDR_INIT;
                    vx_imagepatch_addressing_t u_addr = VX_IMAGEPATCH_ADDR_INIT;
                    vx_map_id f_id, u_id;
                    void *f_base = NULL, *u_base = NULL;

                    status = vxGetValidRegionImage(filling, &f_rect);
                    status |= vxMapImagePatch(filling, &f_rect, 0, &f_id, &f_addr, &f_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
                    status |= vxGetValidRegionImage(upsample, &u_rect);
                    status |= vxMapImagePatch(upsample, &u_rect, 0, &u_id, &u_addr, &u_base, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

                    if (status == VX_SUCCESS && f_addr.stride_x == (filling_format == VX_DF_IMAGE_U8 ? 1 : 2) &&
                        u_addr.stride_x == (upsample_format == VX_DF_IMAGE_U8 ? 1 : 2))
                    {
                        vx_int32 srcW = (vx_int32)f_addr.dim_x;
                        vx_int32 srcH = (vx_int32)f_addr.dim_y;
                        vx_int32 dstW = (vx_int32)u_addr.dim_x;
                        vx_int32 dstH = (vx_int32)u_addr.dim_y;

                        // Saturate S16 filling to U8 into a small scratch row buffer when needed.
                        if (filling_format == VX_DF_IMAGE_U8)
                        {
                            HafCpu_PyramidUp_Gaussian5x5_U8(
                                (const vx_uint8 *)f_base, f_addr.stride_y, srcW, srcH,
                                u_base, u_addr.stride_y, dstW, dstH,
                                upsample_format == VX_DF_IMAGE_S16);
                        }
                        else // VX_DF_IMAGE_S16: original behaviour saturates filling to U8 first.
                        {
                            std::vector<vx_uint8> u8src((size_t)srcW * srcH);
                            for (vx_int32 ry = 0; ry < srcH; ry++)
                            {
                                const vx_int16 *srow = (const vx_int16 *)((const vx_uint8 *)f_base + (size_t)ry * f_addr.stride_y);
                                vx_uint8 *drow = u8src.data() + (size_t)ry * srcW;
                                for (vx_int32 rx = 0; rx < srcW; rx++)
                                {
                                    vx_int32 v = srow[rx];
                                    drow[rx] = (vx_uint8)((v < 0) ? 0 : ((v > 255) ? 255 : v));
                                }
                            }
                            HafCpu_PyramidUp_Gaussian5x5_U8(
                                u8src.data(), srcW, srcW, srcH,
                                u_base, u_addr.stride_y, dstW, dstH,
                                upsample_format == VX_DF_IMAGE_S16);
                        }
                    }
                    else
                    {
                        is_gauss = false;
                    }
                    vxUnmapImagePatch(filling, f_id);
                    vxUnmapImagePatch(upsample, u_id);
                    if (is_gauss)
                        return status;
                }
            }
        }
    }

    vx_image tmp = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

    vx_rectangle_t tmp_rect, filling_rect;
    vx_imagepatch_addressing_t tmp_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_imagepatch_addressing_t filling_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id tmp_map_id, filling_map_id;
    void *tmp_base = NULL;
    void *filling_base = NULL;

    status = vxGetValidRegionImage(tmp, &tmp_rect);
    status |= vxMapImagePatch(tmp, &tmp_rect, 0, &tmp_map_id, &tmp_addr, (void **)&tmp_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
    status = vxGetValidRegionImage(filling, &filling_rect);
    status |= vxMapImagePatch(filling, &filling_rect, 0, &filling_map_id, &filling_addr, (void **)&filling_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    // Zero-stuff upsample: write tmp[ix, iy] = (ix even && iy even) ? saturate(filling[ix/2, iy/2]) : 0
    // tmp is U8; filling is U8 or S16. Use row-major SIMD for high throughput
    // (the original column-major scalar loop was a major bottleneck).
    {
        const vx_int32 tmpStride = tmp_addr.stride_y;
        const vx_int32 fillStride = filling_addr.stride_y;
        const int wInt = (int)width;
        const int hInt = (int)height;

        for (int iy = 0; iy < hInt; iy++)
        {
            vx_uint8 *tmpRow = (vx_uint8 *)tmp_base + (size_t)iy * tmpStride;
            if ((iy & 1) != 0)
            {
                memset(tmpRow, 0, (size_t)wInt);
                continue;
            }
            // Even row: even-indexed pixels are saturated copies; odd indexed are 0.
            int hf = iy >> 1;
            if (filling_format == VX_DF_IMAGE_U8)
            {
                const vx_uint8 *fillRow = (const vx_uint8 *)filling_base + (size_t)hf * fillStride;
                int ix = 0;
#if USE_AVX
                __m256i zero256 = _mm256_setzero_si256();
                for (; ix + 32 <= wInt; ix += 32)
                {
                    __m128i src = _mm_loadu_si128((const __m128i *)(fillRow + (ix >> 1)));
                    // Interleave src bytes with zero bytes via unpack to get 32 bytes
                    __m128i lo = _mm_unpacklo_epi8(src, _mm_setzero_si128());
                    __m128i hi = _mm_unpackhi_epi8(src, _mm_setzero_si128());
                    __m256i out = _mm256_setr_m128i(lo, hi);
                    (void)zero256;
                    _mm256_storeu_si256((__m256i *)(tmpRow + ix), out);
                }
#endif
                for (; ix + 16 <= wInt; ix += 16)
                {
                    __m128i src = _mm_loadl_epi64((const __m128i *)(fillRow + (ix >> 1)));
                    __m128i out = _mm_unpacklo_epi8(src, _mm_setzero_si128());
                    _mm_storeu_si128((__m128i *)(tmpRow + ix), out);
                }
                for (; ix < wInt; ix++)
                {
                    tmpRow[ix] = ((ix & 1) == 0) ? fillRow[ix >> 1] : (vx_uint8)0;
                }
            }
            else // filling is S16; saturate to U8 (0..255)
            {
                const vx_int16 *fillRow = (const vx_int16 *)((const vx_uint8 *)filling_base + (size_t)hf * fillStride);
                int ix = 0;
                for (; ix + 16 <= wInt; ix += 16)
                {
                    __m128i lo = _mm_loadu_si128((const __m128i *)(fillRow + (ix >> 1)));     // 8 int16
                    __m128i hi = _mm_setzero_si128();
                    __m128i packed = _mm_packus_epi16(lo, hi);                                // 8 saturated U8 in lower 8
                    __m128i out = _mm_unpacklo_epi8(packed, _mm_setzero_si128());             // 16 U8 with zeros between
                    _mm_storeu_si128((__m128i *)(tmpRow + ix), out);
                }
                for (; ix < wInt; ix++)
                {
                    if ((ix & 1) != 0) { tmpRow[ix] = 0; continue; }
                    vx_int32 v = fillRow[ix >> 1];
                    v = v < 0 ? 0 : (v > 255 ? 255 : v);
                    tmpRow[ix] = (vx_uint8)v;
                }
            }
        }
        (void)format;
    }

    status |= vxUnmapImagePatch(tmp, tmp_map_id);
    status |= vxUnmapImagePatch(filling, filling_map_id);

    status |=replicateConvolve(tmp, conv, upsample, border);

    vx_rectangle_t upsample_rect;
    vx_imagepatch_addressing_t upsample_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id upsample_map_id;
    void * upsample_base = NULL;
    vx_df_image upsample_format;

    status |= vxQueryImage(upsample, VX_IMAGE_FORMAT, &upsample_format, sizeof(upsample_format));
    status = vxGetValidRegionImage(upsample, &upsample_rect);
    status |= vxMapImagePatch(upsample, &upsample_rect, 0, &upsample_map_id, &upsample_addr, (void **)&upsample_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    // Multiply by 4 with saturation, row-major SIMD (was column-major scalar bottleneck).
    {
        const vx_int32 upStride = upsample_addr.stride_y;
        const int wInt = (int)width;
        const int hInt = (int)height;

        if (upsample_format == VX_DF_IMAGE_U8)
        {
            for (int iy = 0; iy < hInt; iy++)
            {
                vx_uint8 *row = (vx_uint8 *)upsample_base + (size_t)iy * upStride;
                int ix = 0;
                for (; ix + 16 <= wInt; ix += 16)
                {
                    __m128i v = _mm_loadu_si128((const __m128i *)(row + ix));
                    __m128i lo = _mm_unpacklo_epi8(v, _mm_setzero_si128());
                    __m128i hi = _mm_unpackhi_epi8(v, _mm_setzero_si128());
                    lo = _mm_slli_epi16(lo, 2);
                    hi = _mm_slli_epi16(hi, 2);
                    __m128i out = _mm_packus_epi16(lo, hi);
                    _mm_storeu_si128((__m128i *)(row + ix), out);
                }
                for (; ix < wInt; ix++)
                {
                    vx_int32 v = row[ix] * 4;
                    if (v > 255) v = 255;
                    row[ix] = (vx_uint8)v;
                }
            }
        }
        else // S16 with saturation to [INT16_MIN, INT16_MAX]
        {
            for (int iy = 0; iy < hInt; iy++)
            {
                vx_int16 *row = (vx_int16 *)((vx_uint8 *)upsample_base + (size_t)iy * upStride);
                int ix = 0;
                for (; ix + 8 <= wInt; ix += 8)
                {
                    __m128i v = _mm_loadu_si128((const __m128i *)(row + ix));
                    // Sign-extend to 32-bit, shift, then signed-saturate-pack
                    __m128i lo = _mm_cvtepi16_epi32(v);
                    __m128i hi = _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));
                    lo = _mm_slli_epi32(lo, 2);
                    hi = _mm_slli_epi32(hi, 2);
                    __m128i out = _mm_packs_epi32(lo, hi);
                    _mm_storeu_si128((__m128i *)(row + ix), out);
                }
                for (; ix < wInt; ix++)
                {
                    vx_int32 v = row[ix] * 4;
                    if (v > INT16_MAX) v = INT16_MAX;
                    else if (v < INT16_MIN) v = INT16_MIN;
                    row[ix] = (vx_int16)v;
                }
            }
        }
    }
    status |= vxUnmapImagePatch(upsample, upsample_map_id);
    status |= vxReleaseImage(&tmp);
    return status;
}

// Legacy reference path retained for non-U8 input formats and as a safety
// fallback when the fast direct path's preconditions are not met. Avoid using
// for hot benchmark cases.
static int HafCpu_LaplacianPyramid_Legacy
    (
        vx_node node,
        vx_image input,
        vx_pyramid laplacian,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;
    vx_context context = vxGetContext((vx_reference)node);

    vx_size lev;
    vx_size levels = 1;
    vx_uint32 width = 0, height = 0;
    vx_uint32 level_width = 0, level_height = 0;
    vx_df_image format;
    vx_enum policy = VX_CONVERT_POLICY_SATURATE;
    vx_border_t border;
    vx_convolution conv = 0;
    vx_image pyr_gauss_curr_level_filtered = 0;
    vx_image pyr_laplacian_curr_level = 0;
    vx_pyramid gaussian = 0;
    vx_image gauss_cur = 0;
    vx_image gauss_next = 0;

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));
    status |= vxQueryImage(input, VX_IMAGE_FORMAT, &format, sizeof(format));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));
    status |= vxQueryNode(node, VX_NODE_BORDER, &border, sizeof(border));

    vx_border_t saved_border;
    vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER, &saved_border, sizeof(saved_border));
    border.mode = VX_BORDER_REPLICATE;
    vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border));

    gaussian = vxCreatePyramid(context, levels + 1, VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_U8);
    vxuGaussianPyramid(context, input, gaussian);
    conv = vxCreateGaussian5x5Convolution(context);

    level_width = width;
    level_height = height;
    gauss_cur = vxGetPyramidLevel(gaussian, 0);
    gauss_next = vxGetPyramidLevel(gaussian, 1);
    for (lev = 0; lev < levels; lev++)
    {
        pyr_laplacian_curr_level = vxGetPyramidLevel(laplacian, (vx_uint32)lev);

        pyr_gauss_curr_level_filtered = vxCreateImage(context, level_width, level_height, VX_DF_IMAGE_S16);
        upsampleImage(context, level_width, level_height, gauss_next, conv, pyr_gauss_curr_level_filtered, &border);
        status |= vxuSubtract(context, gauss_cur, pyr_gauss_curr_level_filtered, policy, pyr_laplacian_curr_level);
        status |= vxReleaseImage(&pyr_gauss_curr_level_filtered);

        if (lev == levels - 1)
        {
            vx_image tmp = vxGetPyramidLevel(gaussian, (vx_uint32)levels);
            ownCopyImage(tmp, output);
            vxReleaseImage(&tmp);
            vxReleaseImage(&gauss_next);
            vxReleaseImage(&gauss_cur);
        }
        else
        {
            level_width = (vx_uint32)ceilf(level_width * VX_SCALE_PYRAMID_HALF);
            level_height = (vx_uint32)ceilf(level_height * VX_SCALE_PYRAMID_HALF);
            vxReleaseImage(&gauss_next);
            vxReleaseImage(&gauss_cur);
            gauss_cur = vxGetPyramidLevel(gaussian, (vx_uint32)lev + 1);
            gauss_next = vxGetPyramidLevel(gaussian, (vx_uint32)lev + 2);
        }

        status |= vxReleaseImage(&pyr_laplacian_curr_level);
    }

    status |= vxReleasePyramid(&gaussian);
    status |= vxReleaseConvolution(&conv);
    vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &saved_border, sizeof(saved_border));

    return status;
}

int HafCpu_LaplacianPyramid_DATA_DATA_DATA
    (
        vx_node node,
        vx_image input,
        vx_pyramid laplacian,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;

    vx_size levels = 1;
    vx_uint32 width = 0, height = 0;
    vx_df_image format = 0, out_format = 0, lap_format = 0;

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));
    status |= vxQueryImage(input, VX_IMAGE_FORMAT, &format, sizeof(format));
    status |= vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_FORMAT, &lap_format, sizeof(lap_format));

    // Direct CPU path: U8 input, S16 laplacian, U8 output, HALF-scale pyramid.
    // Bypasses vxuGaussianPyramid (which creates/verifies/destroys an internal
    // graph each call) and routes straight to the SSE/AVX2 SIMD primitives,
    // keeping all intermediate gaussian levels in private aligned buffers.
    if (status != VX_SUCCESS || format != VX_DF_IMAGE_U8 ||
        lap_format != VX_DF_IMAGE_S16 || out_format != VX_DF_IMAGE_U8)
    {
        return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
    }
    vx_float32 lap_scale = 0.f;
    vxQueryPyramid(laplacian, VX_PYRAMID_SCALE, &lap_scale, sizeof(lap_scale));
    if (lap_scale != VX_SCALE_PYRAMID_HALF)
    {
        return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
    }

    // GPU-affinity fallback: when the context is configured for GPU execution
    // (OpenCL or HIP backend with GPU affinity), the CTS reference path's
    // vxuGaussianPyramid runs on the GPU and produces gaussian level values
    // that differ from the CPU ScaleGaussianHalf primitive by a few units at
    // the very edge pixels (different border-handling rounding between the
    // GPU kernel and the SIMD CPU kernel). Comparing CPU-computed gaussian
    // against a GPU-computed reference exceeds the 1-unit tolerance the CTS
    // LaplacianPyramid tests allow. Route through the legacy graph path so
    // both reference and implementation run gaussian on the same device.
    //
    // Mirrors vxuSetGraphAffinityDefault()/AGO_KERNEL_TARGET_DEFAULT: on
    // HIP/OpenCL builds the default graph target is GPU unless the user
    // explicitly forces CPU via AGO_DEFAULT_TARGET=CPU or via context
    // affinity. Also fall back whenever the input image already has a live
    // GPU mirror, which is a strong signal that earlier nodes ran on GPU.
    {
        AgoNode * agoNode = (AgoNode *)node;
        AgoData * inputData = (AgoData *)input;
        bool gpu_path_required = false;
#if ENABLE_OPENCL || ENABLE_HIP
        // Compile-time default-target on HIP/OCL builds is GPU.
        gpu_path_required = true;
        // Honour explicit CPU overrides (env var or context affinity) so that
        // a HIP/OCL build forced to CPU still benefits from the fast path.
        char envBuf[64];
        if (agoGetEnvironmentVariable("AGO_DEFAULT_TARGET", envBuf, sizeof(envBuf)) &&
            !strcmp(envBuf, "CPU"))
        {
            gpu_path_required = false;
        }
        if (agoNode && agoNode->ref.context &&
            agoNode->ref.context->attr_affinity.device_type == AGO_TARGET_AFFINITY_CPU)
        {
            gpu_path_required = false;
        }
#else
        // CPU-only build: vxuGaussianPyramid always runs on CPU; safe to use
        // the fast path unless the user has explicitly forced GPU affinity.
        if (agoNode && agoNode->ref.context &&
            agoNode->ref.context->attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        {
            gpu_path_required = true;
        }
#endif
#if ENABLE_OPENCL
        if (inputData && inputData->opencl_buffer) gpu_path_required = true;
#endif
#if ENABLE_HIP
        if (inputData && inputData->hip_memory) gpu_path_required = true;
#endif
        (void)inputData;
        (void)agoNode;
        if (gpu_path_required)
        {
            return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
        }
    }

    // Pre-compute per-level dimensions using the same rounding rule as
    // vxCreatePyramid (ceilf(prev * 0.5)).
    std::vector<vx_uint32> lw(levels + 1), lh(levels + 1);
    lw[0] = width;
    lh[0] = height;
    for (vx_size i = 1; i <= levels; i++)
    {
        lw[i] = (vx_uint32)ceilf(lw[i - 1] * VX_SCALE_PYRAMID_HALF);
        lh[i] = (vx_uint32)ceilf(lh[i - 1] * VX_SCALE_PYRAMID_HALF);
    }

    // Owned gaussian-pyramid buffers for levels 0..N. Level 0 is a copy of the
    // input (the byte-for-byte CHANNEL_COPY equivalent that vxuGaussianPyramid
    // does internally). Owning level 0 is required because the SIMD primitive
    // HafCpu_ScaleGaussianHalf_U8_U8_5x5 issues horizontal loads at
    // srcImage[-1] / srcImage[-2] for the leftmost sampled column (see
    // ago_haf_cpu_pyramid.cpp Horizontal5x5GaussianFilter_C / SSE variants).
    // The caller-supplied input map carries no guaranteed left/top padding, so
    // reading before it is undefined; routing those reads into our private
    // padded buffer keeps the behaviour identical to the legacy copied
    // pyramid path. Strides are ALIGN16(width) to match the layout that
    // vxCreatePyramid would produce, so the kernel's small out-of-row reads
    // (e.g. pLocalSrc[-2] at column 0 of dst row 1) land on exactly the same
    // bytes as the legacy graph path. Buffers are zero-initialised so the
    // un-touched border rows (which ScaleGaussianHalf/5x5 deliberately skips)
    // read back as zero, matching the legacy behaviour. Each buffer also has a
    // two-row zero pad above and below so that the fused upsample/subtract
    // kernel's vertical reads at the very top and bottom rows never alias
    // into another allocation or before the allocation start.
    std::vector<vx_uint8 *> gptr(levels + 1, nullptr);
    std::vector<vx_int32> gstride(levels + 1, 0);
    std::vector<void *> gowned(levels + 1, nullptr);

    auto release_owned = [&]() {
        for (vx_size i = 0; i <= levels; i++) {
            if (gowned[i]) { free(gowned[i]); gowned[i] = nullptr; }
        }
    };

    constexpr size_t kPadRows = 2;
    for (vx_size i = 0; i <= levels; i++)
    {
        vx_int32 stride = (vx_int32)((lw[i] + 15) & ~15);
        size_t bytes = (size_t)stride * (lh[i] + 2 * kPadRows);
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || !p)
        {
            release_owned();
            return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
        }
        memset(p, 0, bytes);
        gowned[i] = p;
        gptr[i] = (vx_uint8 *)p + kPadRows * (size_t)stride;
        gstride[i] = stride;
    }

    // Map the input read-only and copy it into the owned gauss[0] buffer.
    // The copy is per-row so any stride padding in either the input map or
    // our owned buffer is preserved as zero (matching the legacy CHANNEL_COPY
    // into a freshly-allocated, calloc'd pyramid level 0 buffer).
    {
        vx_rectangle_t in_rect;
        vxGetValidRegionImage(input, &in_rect);
        vx_imagepatch_addressing_t in_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id in_map = 0;
        void *in_base = nullptr;
        vx_status st = vxMapImagePatch(input, &in_rect, 0, &in_map, &in_addr,
                                       &in_base, VX_READ_ONLY,
                                       VX_MEMORY_TYPE_HOST, 0);
        if (st != VX_SUCCESS || in_addr.stride_x != 1)
        {
            if (st == VX_SUCCESS) vxUnmapImagePatch(input, in_map);
            release_owned();
            return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
        }
        const vx_uint8 *src_p = (const vx_uint8 *)in_base;
        vx_uint8 *dst_p = gptr[0];
        vx_uint32 copy_w = (in_addr.dim_x < lw[0]) ? in_addr.dim_x : lw[0];
        vx_uint32 copy_h = (in_addr.dim_y < lh[0]) ? in_addr.dim_y : lh[0];
        for (vx_uint32 r = 0; r < copy_h; r++)
        {
            memcpy(dst_p + (size_t)r * gstride[0],
                   src_p + (size_t)r * in_addr.stride_y, copy_w);
        }
        vxUnmapImagePatch(input, in_map);
    }

    // Scratch storage reused across all ScaleGaussianHalf invocations. Sized
    // for the largest destination stride (always level 1 with HALF scale).
    std::vector<vx_uint8> scratch;
    {
        int alignedDstStride = (gstride[1] + 15) & ~15;
        size_t scratch_needed = (size_t)5 * 4 * (size_t)alignedDstStride * sizeof(vx_int16);
        scratch.resize(scratch_needed);
    }

    // Build the gaussian pyramid levels in-place using the SIMD primitive.
    // The kernel only writes dst rows 1..(dstHeight - 2); the unwritten edge
    // rows stay at zero (the same border behaviour as the legacy graph path).
    for (vx_size i = 1; i <= levels; i++)
    {
        bool sampleFirstRow = (lh[i - 1] & 1) ? true : false;
        bool sampleFirstCol = (lw[i - 1] & 1) ? true : false;
        if (lh[i] >= 3 && lw[i] >= 3 && lw[i - 1] >= 5 && lh[i - 1] >= 5)
        {
            HafCpu_ScaleGaussianHalf_U8_U8_5x5(
                lw[i], lh[i] - 2,
                gptr[i] + (size_t)gstride[i],
                (vx_uint32)gstride[i],
                gptr[i - 1] + (size_t)2 * gstride[i - 1],
                (vx_uint32)gstride[i - 1],
                sampleFirstRow, sampleFirstCol,
                scratch.data());
        }
    }

    // Fused upsample(gauss[i+1]) - subtract from gauss[i] -> laplacian[i].
    // If any vxMapImagePatch fails (or returns an unexpected stride), bail out
    // to the legacy path instead of leaving the caller with a partially-
    // written pyramid and a misleading VX_SUCCESS status. The unmap is only
    // issued on successful maps to avoid passing invalid map ids to the API.
    for (vx_size lev = 0; lev < levels; lev++)
    {
        vx_image lap_img = vxGetPyramidLevel(laplacian, (vx_uint32)lev);
        vx_rectangle_t r_lap;
        vxGetValidRegionImage(lap_img, &r_lap);
        vx_imagepatch_addressing_t a_lap = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id m_lap = 0;
        void *b_lap = nullptr;
        vx_status st = vxMapImagePatch(lap_img, &r_lap, 0, &m_lap, &a_lap,
                                       &b_lap, VX_WRITE_ONLY,
                                       VX_MEMORY_TYPE_HOST, 0);
        if (st != VX_SUCCESS || a_lap.stride_x != 2)
        {
            if (st == VX_SUCCESS) vxUnmapImagePatch(lap_img, m_lap);
            vxReleaseImage(&lap_img);
            release_owned();
            return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
        }
        HafCpu_PyramidUp_Gaussian5x5_Subtract_U8(
            gptr[lev + 1], gstride[lev + 1],
            (vx_int32)lw[lev + 1], (vx_int32)lh[lev + 1],
            gptr[lev], gstride[lev],
            (vx_int16 *)b_lap, a_lap.stride_y,
            (vx_int32)a_lap.dim_x, (vx_int32)a_lap.dim_y);
        vxUnmapImagePatch(lap_img, m_lap);
        vxReleaseImage(&lap_img);
    }

    // Copy the deepest gaussian level into the output image (mirrors the
    // ownCopyImage(gaussian[levels], output) in the legacy implementation).
    // Same map-error handling: only unmap on success, and propagate a failure
    // through the legacy path so the caller never sees VX_SUCCESS with an
    // unwritten output.
    {
        vx_rectangle_t r_out;
        vxGetValidRegionImage(output, &r_out);
        vx_imagepatch_addressing_t a_out = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id m_out = 0;
        void *b_out = nullptr;
        vx_status st = vxMapImagePatch(output, &r_out, 0, &m_out, &a_out,
                                       &b_out, VX_WRITE_ONLY,
                                       VX_MEMORY_TYPE_HOST, 0);
        if (st != VX_SUCCESS || a_out.stride_x != 1)
        {
            if (st == VX_SUCCESS) vxUnmapImagePatch(output, m_out);
            release_owned();
            return HafCpu_LaplacianPyramid_Legacy(node, input, laplacian, output);
        }
        vx_uint8 *src_p = gptr[levels];
        vx_uint8 *dst_p = (vx_uint8 *)b_out;
        vx_uint32 copy_w = (a_out.dim_x < lw[levels]) ? a_out.dim_x : lw[levels];
        vx_uint32 copy_h = (a_out.dim_y < lh[levels]) ? a_out.dim_y : lh[levels];
        for (vx_uint32 r = 0; r < copy_h; r++)
        {
            memcpy(dst_p + (size_t)r * a_out.stride_y,
                   src_p + (size_t)r * gstride[levels], copy_w);
        }
        vxUnmapImagePatch(output, m_out);
    }

    release_owned();

    return status;
}

#define VX_SCALE_PYRAMID_DOUBLE (2.0f)

// Legacy reference path retained as a safety fallback when the fast direct
// path's preconditions are not met (unexpected formats, non-HALF pyramid
// scale, GPU affinity, map/stride failures). Avoid using for hot benchmark
// cases: each level builds/verifies/executes/tears-down an immediate-mode
// vxuAdd graph and runs the scalar ownCopyImage copies.
static int HafCpu_LaplacianReconstruct_Legacy
    (
        vx_node node,
        vx_pyramid laplacian,
        vx_image input,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;

    vx_context context = vxGetContext((vx_reference)node);

    vx_size lev;
    vx_size levels = 1;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_uint32 level_width = 0;
    vx_uint32 level_height = 0;
    vx_df_image format = VX_DF_IMAGE_S16;
    vx_enum policy = VX_CONVERT_POLICY_SATURATE;
    vx_border_t border;
    vx_image filling = 0;
    vx_image pyr_level = 0;
    vx_image filter = 0;
    vx_image out = 0;
    vx_convolution conv;

    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));
    
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));

    status |= vxQueryNode(node, VX_NODE_BORDER, &border, sizeof(border));
    border.mode = VX_BORDER_REPLICATE;
    conv = vxCreateGaussian5x5Convolution(context);

    level_width = (vx_uint32)ceilf(width  * VX_SCALE_PYRAMID_DOUBLE);
    level_height = (vx_uint32)ceilf(height * VX_SCALE_PYRAMID_DOUBLE);
    filling = vxCreateImage(context, width, height, format);
    for (lev = 0; lev < levels; lev++)
    {
        out = vxCreateImage(context, level_width, level_height, format);
        filter = vxCreateImage(context, level_width, level_height, format);

        pyr_level = vxGetPyramidLevel(laplacian, (vx_uint32)((levels - 1) - lev));

        if (lev == 0)
        {
            ownCopyImage(input, filling);
        }
        upsampleImage(context, level_width, level_height, filling, conv, filter, &border);
        vxuAdd(context, filter, pyr_level, policy, out);
        //vxAddition(filter, pyr_level, spolicy, out);

        status |= vxReleaseImage(&pyr_level);

        if ((levels - 1) - lev == 0)
        {
            ownCopyImage(out, output);
            status |= vxReleaseImage(&filling);
        }
        else
        {
            /* compute dimensions for the next level */
            status |= vxReleaseImage(&filling);
            filling = vxCreateImage(context, level_width, level_height, format);
            ownCopyImage(out, filling);

            level_width = (vx_uint32)ceilf(level_width  * VX_SCALE_PYRAMID_DOUBLE);
            level_height = (vx_uint32)ceilf(level_height * VX_SCALE_PYRAMID_DOUBLE);


        }
        status |= vxReleaseImage(&out);
        status |= vxReleaseImage(&filter);

    }
    status |= vxReleaseConvolution(&conv);
    status |= vxReleaseScalar(&spolicy);
    
    return status;
}

int HafCpu_LaplacianReconstruct_DATA_DATA_DATA
    (
        vx_node node,
        vx_pyramid laplacian,
        vx_image input,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;

    vx_size levels = 1;
    vx_uint32 in_w = 0, in_h = 0;
    vx_df_image in_format = 0, out_format = 0, lap_format = 0;
    vx_float32 lap_scale = 0.f;

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &in_w, sizeof(in_w));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &in_h, sizeof(in_h));
    status |= vxQueryImage(input, VX_IMAGE_FORMAT, &in_format, sizeof(in_format));
    status |= vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_FORMAT, &lap_format, sizeof(lap_format));
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_SCALE, &lap_scale, sizeof(lap_scale));

    // Direct CPU path preconditions: U8/S16 input, S16 laplacian, U8/S16 output,
    // HALF-scale pyramid (so each reconstruct step upsamples by 2x). Anything
    // else routes to the legacy immediate-mode graph path.
    if (status != VX_SUCCESS || levels < 1 ||
        (in_format != VX_DF_IMAGE_U8 && in_format != VX_DF_IMAGE_S16) ||
        lap_format != VX_DF_IMAGE_S16 ||
        (out_format != VX_DF_IMAGE_U8 && out_format != VX_DF_IMAGE_S16) ||
        lap_scale != VX_SCALE_PYRAMID_HALF)
    {
        return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
    }

    // GPU-affinity fallback: mirror HafCpu_LaplacianPyramid_DATA_DATA_DATA so the
    // reference path and implementation run gaussian/upsample on the same device
    // (the CTS reference's vxuAdd / upsample would otherwise execute on GPU on a
    // HIP/OCL build and disagree with the CPU SIMD primitive at edge pixels).
    {
        AgoNode * agoNode = (AgoNode *)node;
        AgoData * inputData = (AgoData *)input;
        bool gpu_path_required = false;
#if ENABLE_OPENCL || ENABLE_HIP
        gpu_path_required = true;
        char envBuf[64];
        if (agoGetEnvironmentVariable("AGO_DEFAULT_TARGET", envBuf, sizeof(envBuf)) &&
            !strcmp(envBuf, "CPU"))
        {
            gpu_path_required = false;
        }
        if (agoNode && agoNode->ref.context &&
            agoNode->ref.context->attr_affinity.device_type == AGO_TARGET_AFFINITY_CPU)
        {
            gpu_path_required = false;
        }
#else
        if (agoNode && agoNode->ref.context &&
            agoNode->ref.context->attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        {
            gpu_path_required = true;
        }
#endif
#if ENABLE_OPENCL
        if (inputData && inputData->opencl_buffer) gpu_path_required = true;
#endif
#if ENABLE_HIP
        if (inputData && inputData->hip_memory) gpu_path_required = true;
#endif
        (void)inputData;
        (void)agoNode;
        if (gpu_path_required)
        {
            return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
        }
    }

    // The reconstruct walks the laplacian pyramid from the deepest level
    // (index levels-1, smallest) up to level 0 (largest = output resolution).
    // Step `lev` upsamples the running level by 2x and adds laplacian level
    // (levels-1-lev). We drive each step's destination dimensions from the
    // actual laplacian level image so ceil-rounding always matches the stored
    // pyramid (and bail to legacy on any map/stride surprise).
    std::vector<vx_uint8 *> cptr(levels + 1, nullptr);   // U8 working buffers, index by "src" level
    std::vector<vx_int32> cstride(levels + 1, 0);
    std::vector<void *> cowned(levels + 1, nullptr);
    auto release_owned = [&]() {
        for (vx_size i = 0; i <= levels; i++) {
            if (cowned[i]) { free(cowned[i]); cowned[i] = nullptr; }
        }
    };

    // Allocate the deepest U8 working buffer (holds the saturated input) and one
    // U8 buffer per intermediate reconstructed level (levels 1..levels-1). The
    // final level writes straight into the output image, so it needs no owned
    // buffer. Index convention: buf[k] holds the reconstructed/input level whose
    // resolution feeds the upsample producing laplacian level (k-1).
    //   buf[levels] = saturate_u8(input)            (smallest)
    //   buf[k]      = reconstructed laplacian level k (1 <= k <= levels-1)
    // Sizes come from the laplacian level dims; buf[levels] uses the input dims.
    {
        vx_int32 stride = (vx_int32)((in_w + 15) & ~15);
        size_t bytes = (size_t)stride * (in_h ? in_h : 1);
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || !p)
        {
            release_owned();
            return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
        }
        memset(p, 0, bytes);
        cowned[levels] = p;
        cptr[levels] = (vx_uint8 *)p;
        cstride[levels] = stride;
    }

    // Saturate-copy the deepest input image into buf[levels] as U8 (matches the
    // legacy path saturating the S16 `filling` to U8 inside upsampleImage).
    {
        vx_rectangle_t r_in;
        vxGetValidRegionImage(input, &r_in);
        vx_imagepatch_addressing_t a_in = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id m_in = 0;
        void *b_in = nullptr;
        vx_status st = vxMapImagePatch(input, &r_in, 0, &m_in, &a_in, &b_in,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        // The U8 memcpy / S16 src_p[c] indexing below assumes tightly packed
        // pixels (stride_x == element size, step_x/step_y == 1). Mirror the
        // LaplacianPyramid fast path and bail to legacy on any non-standard
        // layout so a strided/sub-sampled input view never reads wrong pixels.
        vx_int32 expected_in_stride_x = (in_format == VX_DF_IMAGE_U8) ? 1 : 2;
        if (st != VX_SUCCESS || a_in.stride_x != expected_in_stride_x ||
            a_in.step_x != 1 || a_in.step_y != 1)
        {
            if (st == VX_SUCCESS) vxUnmapImagePatch(input, m_in);
            release_owned();
            return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
        }
        vx_uint32 cw = (a_in.dim_x < in_w) ? a_in.dim_x : in_w;
        vx_uint32 ch = (a_in.dim_y < in_h) ? a_in.dim_y : in_h;
        vx_uint8 *dst_p = cptr[levels];
        if (in_format == VX_DF_IMAGE_U8)
        {
            const vx_uint8 *src_p = (const vx_uint8 *)b_in;
            for (vx_uint32 r = 0; r < ch; r++)
                memcpy(dst_p + (size_t)r * cstride[levels],
                       src_p + (size_t)r * a_in.stride_y, cw);
        }
        else // S16 input -> saturate to U8
        {
            for (vx_uint32 r = 0; r < ch; r++)
            {
                const vx_int16 *src_p = (const vx_int16 *)((const vx_uint8 *)b_in + (size_t)r * a_in.stride_y);
                vx_uint8 *drow = dst_p + (size_t)r * cstride[levels];
                for (vx_uint32 c = 0; c < cw; c++)
                {
                    vx_int32 v = src_p[c];
                    drow[c] = (vx_uint8)((v < 0) ? 0 : ((v > 255) ? 255 : v));
                }
            }
        }
        vxUnmapImagePatch(input, m_in);
    }

    vx_int32 src_w = (vx_int32)in_w;
    vx_int32 src_h = (vx_int32)in_h;
    vx_size cur_idx = levels;

    for (vx_size lev = 0; lev < levels; lev++)
    {
        vx_uint32 idx = (vx_uint32)((levels - 1) - lev);   // laplacian level for this step
        bool is_final = (idx == 0);

        vx_image lap_img = vxGetPyramidLevel(laplacian, idx);
        vx_rectangle_t r_lap;
        vxGetValidRegionImage(lap_img, &r_lap);
        vx_imagepatch_addressing_t a_lap = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id m_lap = 0;
        void *b_lap = nullptr;
        vx_status st = vxMapImagePatch(lap_img, &r_lap, 0, &m_lap, &a_lap, &b_lap,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if (st != VX_SUCCESS || a_lap.stride_x != 2 ||
            a_lap.step_x != 1 || a_lap.step_y != 1)
        {
            if (st == VX_SUCCESS) vxUnmapImagePatch(lap_img, m_lap);
            vxReleaseImage(&lap_img);
            release_owned();
            return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
        }
        vx_int32 dst_w = (vx_int32)a_lap.dim_x;
        vx_int32 dst_h = (vx_int32)a_lap.dim_y;

        if (is_final)
        {
            // Write the reconstructed full-resolution level straight into output.
            vx_rectangle_t r_out;
            vxGetValidRegionImage(output, &r_out);
            vx_imagepatch_addressing_t a_out = VX_IMAGEPATCH_ADDR_INIT;
            vx_map_id m_out = 0;
            void *b_out = nullptr;
            vx_status sto = vxMapImagePatch(output, &r_out, 0, &m_out, &a_out, &b_out,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
            bool out_is_s16 = (out_format == VX_DF_IMAGE_S16);
            if (sto != VX_SUCCESS ||
                a_out.stride_x != (out_is_s16 ? 2 : 1) ||
                a_out.step_x != 1 || a_out.step_y != 1)
            {
                if (sto == VX_SUCCESS) vxUnmapImagePatch(output, m_out);
                vxUnmapImagePatch(lap_img, m_lap);
                vxReleaseImage(&lap_img);
                release_owned();
                return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
            }
            // Guard against the reconstruct node's floor-based output meta being
            // smaller than the (ceil-based) laplacian level 0: never write past
            // the mapped output region. Matches the legacy ownCopyImage, which
            // copies only the overlapping valid region.
            vx_int32 ow = (vx_int32)a_out.dim_x, oh = (vx_int32)a_out.dim_y;
            if (dst_w > ow) dst_w = ow;
            if (dst_h > oh) dst_h = oh;
            HafCpu_PyramidUp_Gaussian5x5_Add(
                cptr[cur_idx], cstride[cur_idx], src_w, src_h,
                (const vx_int16 *)b_lap, a_lap.stride_y,
                b_out, a_out.stride_y, dst_w, dst_h,
                out_is_s16);
            vxUnmapImagePatch(output, m_out);
        }
        else
        {
            // Allocate this level's U8 working buffer and reconstruct into it.
            vx_int32 stride = (vx_int32)((dst_w + 15) & ~15);
            size_t bytes = (size_t)stride * (dst_h ? dst_h : 1);
            void *p = nullptr;
            if (posix_memalign(&p, 64, bytes) != 0 || !p)
            {
                vxUnmapImagePatch(lap_img, m_lap);
                vxReleaseImage(&lap_img);
                release_owned();
                return HafCpu_LaplacianReconstruct_Legacy(node, laplacian, input, output);
            }
            memset(p, 0, bytes);
            cowned[idx] = p;
            cptr[idx] = (vx_uint8 *)p;
            cstride[idx] = stride;
            HafCpu_PyramidUp_Gaussian5x5_Add(
                cptr[cur_idx], cstride[cur_idx], src_w, src_h,
                (const vx_int16 *)b_lap, a_lap.stride_y,
                cptr[idx], cstride[idx], dst_w, dst_h,
                false);
            cur_idx = idx;
            src_w = dst_w;
            src_h = dst_h;
        }

        vxUnmapImagePatch(lap_img, m_lap);
        vxReleaseImage(&lap_img);
    }

    release_owned();
    return status;
}
