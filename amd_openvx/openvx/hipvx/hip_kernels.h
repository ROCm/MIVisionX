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


#ifndef MIVISIONX_HIP_KERNELS_H
#define MIVISIONX_HIP_KERNELS_H
#include <VX/vx.h>
#include "hip/hip_runtime.h"
#include "ago_haf_cpu.h"

typedef struct AgoConfigScaleMatrix ago_scale_matrix_t;
#define PIXELSATURATEU8(pixel)  (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
#define PIXELROUNDF32(value)    ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))
#define HIPSELECT(a,b,c)        (c ? b : a)

typedef struct d_uint6 {
  uint data[6];
} d_uint6;

typedef struct d_uint8 {
  uint data[8];
} d_uint8;

// common device kernels

__device__ __forceinline__ uint pack_(float4 src) {
    return ((uint)src.x  & 0xFF) |
           (((uint)src.y & 0xFF) << 8) |
           (((uint)src.z & 0xFF) << 16) |
           (((uint)src.w & 0xFF) << 24);
}

__device__ __forceinline__ float4 unpack_(uint src) {
    return make_float4((float)(src  & 0xFF),
                       (float)((src & 0xFF00)     >> 8),
                       (float)((src & 0xFF0000)   >> 16),
                       (float)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ float unpack0_(uint src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float unpack1_(uint src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float unpack2_(uint src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float unpack3_(uint src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 fabs4(float4 src) {
    return make_float4(fabs(src.x), fabs(src.y), fabs(src.z), fabs(src.w));
}

template<class T>
__device__ __forceinline__  constexpr const T& hip_clamp( const T& v, const T& lo, const T& hi ) {
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

__device__ __forceinline__ short hip_convert_short_rte(float a) {
    a = rint(a);
    return (short) a;
}

__device__ __forceinline__ short hip_convert_short_sat_rte(float a) {
    a = rint(a);
    return (short) hip_clamp(a, (float) INT16_MIN, (float) INT16_MAX);
}

__device__ __forceinline__ void hip_convert_U8_U1 (uint2 * p0, __u_char p1) {
    uint2 r;
    r.x  = (-(p1 &   1)) & 0x000000ff;
    r.x |= (-(p1 &   2)) & 0x0000ff00;
    r.x |= (-(p1 &   4)) & 0x00ff0000;
    r.x |= (-(p1 &   8)) & 0xff000000;
    r.y  = (-((p1 >> 4) & 1)) & 0x000000ff;
    r.y |= (-(p1 &  32)) & 0x0000ff00;
    r.y |= (-(p1 &  64)) & 0x00ff0000;
    r.y |= (-(p1 & 128)) & 0xff000000;
    *p0 = r;
}

__device__ __forceinline__ void hip_convert_U1_U8 (__u_char * p0, uint2 p1) {
    __u_char r;
    r  =  p1.x        &   1;
    r |= (p1.x >>  7) &   2;
    r |= (p1.x >> 14) &   4;
    r |= (p1.x >> 21) &   8;
    r |= (p1.y <<  4) &  16;
    r |= (p1.y >>  3) &  32;
    r |= (p1.y >> 10) &  64;
    r |= (p1.y >> 17) & 128;
    *p0 = r;
}

__device__ __forceinline__ float4 hip_select (float4 a, float4 b, int c) {
    return (c ? b : a);
}

// common device kernels - old ones, but still in use - can be removed once they aren't in use anywhere

__device__ __forceinline__ float4 uchars_to_float4(uint src) {
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint float4_to_uchars(float4 src) {
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

// common device kernels - to check usage later and delete accordingly

// #define atan2_p0    (0.273*0.3183098862f)
// #define atan2_p1    (0.9997878412794807f*57.29577951308232f)
// #define atan2_p3    (-0.3258083974640975f*57.29577951308232f)
// #define atan2_p5    (0.1555786518463281f*57.29577951308232f)
// #define atan2_p7    (-0.04432655554792128f*57.29577951308232f)
// #define DBL_EPSILON __DBL_EPSILON__

// __device__ __forceinline__ float4 s16s_to_float4_grouped(int src1, int src2) {
//     return make_float4((float)(src1&0xFFFF), (float)((src1&0xFFFF0000)>>16), (float)(src2&0xFFFF), (float)((src2&0xFFFF0000)>>16));
// }

// __device__ __forceinline__ float4 s16s_to_float4_ungrouped(const short int *src, unsigned int srcIdx) {
//     short4 srcs4 = *((short4 *)(&src[srcIdx]));
//     return make_float4((float)srcs4.x, (float)srcs4.y, (float)srcs4.z, (float)srcs4.w);
// }

// __device__ __forceinline__ void float4_to_s16s(short int *dst_s16s, unsigned int dstIdx, float4 dst_float4) {
//     *((short4 *)(&dst_s16s[dstIdx])) = make_short4(dst_float4.x, dst_float4.y, dst_float4.z, dst_float4.w);
// }

// __device__ __forceinline__ float4 generic_mod_float4(float4 src, int b) {
//     src.x = (float) ((int)src.x % b < 0 ? (int)src.x % b + b : (int)src.x % b);
//     src.y = (float) ((int)src.y % b < 0 ? (int)src.y % b + b : (int)src.y % b);
//     src.z = (float) ((int)src.z % b < 0 ? (int)src.z % b + b : (int)src.z % b);
//     src.w = (float) ((int)src.w % b < 0 ? (int)src.w % b + b : (int)src.w % b);
//     return src;
// }

// __device__ float Norm_Atan2_deg (float Gx, float Gy) {
//     float scale = (float)128 / 180.f;
//     vx_uint16 ax, ay;
//     ax = fabsf(Gx), ay = fabsf(Gy);
//     float a, c, c2;
//     if (ax >= ay) {
//         c = (float)ay / ((float)ax + (float)DBL_EPSILON);
//         c2 = c*c;
//         a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
//     }
//     else {
//         c = (float)ax / ((float)ay + (float)DBL_EPSILON);
//         c2 = c*c;
//         a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
//     }
//     if (Gx < 0)
//     a = 180.f - a;
//     if (Gy < 0)
//     a = 360.f - a;

//     // normalize and copy to dst
//     float arct_norm = (a*scale + 0.5);
//     return arct_norm;
// }


// arithmetic_kernels

int HipExec_AbsDiff_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_AbsDiff_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_U8_U8U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_U8_U8U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Add_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_U8_U8U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_U8_U8U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_U8S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Sub_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Mul_U8_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_U8_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Mul_S16_S16S16_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale
        );
int HipExec_Magnitude_S16_S16S16(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Phase_U8_S16S16(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_WeightedAverage_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 alpha
        );


// logical_kernels

int HipExec_And_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_And_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Or_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Xor_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_Not_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U8_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U1_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_Not_U1_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );

// statistical_kernels

int HipExec_Threshold_U8_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_Threshold_U8_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_Threshold_U1_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_Threshold_U1_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_ThresholdNot_U8_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_ThresholdNot_U8_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_ThresholdNot_U1_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdValue
        );
int HipExec_ThresholdNot_U1_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper
        );
int HipExec_IntegralImage_U32_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint32 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_MinMax_DATA_U8(
        hipStream_t stream, vx_int32    * pHipDstMinValue, vx_int32    * pHipDstMaxValue,
        vx_uint32     srcWidth,  vx_uint32     srcHeight,
        vx_uint8    * pHipSrcImage, vx_uint32     srcImageStrideInBytes
        );
int HipExec_MeanStdDev_DATA_U8(
        hipStream_t stream, vx_float32  * pHipSum, vx_float32  * pHipSumOfSquared,
        vx_uint32  srcWidth, vx_uint32  srcHeight,
        vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
	    );
// int HipExec_HistogramFixedBins_DATA_U8(
// 		vx_uint32     dstHist[],
// 		vx_uint32     distBinCount,
// 		vx_uint32     distOffset,
// 		vx_uint32     distRange,
// 		vx_uint32     distWindow,
// 		vx_uint32     srcWidth,
// 		vx_uint32     srcHeight,
// 		vx_uint8    * pSrcImage,
// 		vx_uint32     srcImageStrideInBytes
// 	);


// color_kernels

int HipExec_Lut_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_uint8 *lut
        );

int HipExec_ColorDepth_U8_S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift
        );
int HipExec_ColorDepth_U8_S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift
        );
int HipExec_ColorDepth_S16_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift
        );

int HipExec_ChannelExtract_U8_U16_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U16_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U24_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U24_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U24_Pos2(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U32_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U32_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U32_Pos2(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8_U32_Pos3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ChannelExtract_U8U8U8_U24(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ChannelExtract_U8U8U8_U32(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ChannelExtract_U8U8U8U8_U32(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ChannelCombine_U16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes
        );
int HipExec_ChannelCombine_U24_U8U8U8_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
        );
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
        );
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes
        );
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
        const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes
        );

int HipExec_ColorConvert_RGBX_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGBX_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_RGB_IYUV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
        );
int HipExec_ColorConvert_RGB_NV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
        );
int HipExec_ColorConvert_RGB_NV21(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
        );
int HipExec_ColorConvert_RGBX_IYUV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
        );
int HipExec_ColorConvert_RGBX_NV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
        );
int HipExec_ColorConvert_RGBX_NV21(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
        );
int HipExec_ColorConvert_NV12_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
        vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_NV12_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
        vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );
int HipExec_ColorConvert_IYUV_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ColorConvert_IYUV_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_NV12_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
        vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
        const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_NV12_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
        vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
        const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_IYUV_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_IYUV_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ColorConvert_YUV4_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_IUV_UV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes
        );
int HipExec_FormatConvert_UV12_IUV(
       hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstChromaImage, vx_uint32 DstChromaImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes
        );
int HipExec_ColorConvert_YUV4_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_FormatConvert_UV_UV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ScaleUp2x2_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
// filter_kernels

int HipExec_Box_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Dilate_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Erode_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Median_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Gaussian_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Convolve_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
        );
int HipExec_Convolve_S16_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int16 *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight
        );
int HipExec_Sobel_S16S16_U8_3x3_GXY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
        vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Sobel_S16_U8_3x3_GX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_Sobel_S16_U8_3x3_GY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );

// geometric_kernels

int HipExec_ScaleImage_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const ago_scale_matrix_t *matrix
        );
int HipExec_ScaleImage_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const ago_scale_matrix_t *matrix
        );
int HipExec_ScaleImage_U8_U8_Bilinear_Replicate(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const ago_scale_matrix_t *matrix
        );
int HipExec_ScaleImage_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const ago_scale_matrix_t *matrix,
        const vx_uint8 border
        );
int HipExec_ScaleImage_U8_U8_Area(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const ago_scale_matrix_t *matrix
        );
int HipExec_ScaleGaussianHalf_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_ScaleGaussianHalf_U8_U8_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );

int HipExec_WarpAffine_U8_U8_Nearest(
      hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix
        );
int HipExec_WarpAffine_U8_U8_Nearest_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix,
        vx_uint8 border
        );
int HipExec_WarpAffine_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix
        );
int HipExec_WarpAffine_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix,
        vx_uint8 border
        );

int HipExec_WarpPerspective_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix
        );
int HipExec_WarpPerspective_U8_U8_Nearest_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix,
        vx_uint8 border
        );
int HipExec_WarpPerspective_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix
        );
int HipExec_WarpPerspective_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix,
        vx_uint8 border
        );

int HipExec_Remap_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
        );
int HipExec_Remap_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_coord2d_ushort_t *map, vx_uint32 mapStrideInBytes
        );

// vision_kernels
int HipExec_HarrisSobel_HG3_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_HarrisSobel_HG3_U8_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_HarrisSobel_HG3_U8_7x7(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 * pDstGxy_, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 * pSrcImage, vx_uint32 srcImageStrideInBytes
        );

int HipExec_HarrisScore_HVC_HG3_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_float32 normalization_factor
        );
int HipExec_HarrisScore_HVC_HG3_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_float32 normalization_factor
        );
int HipExec_HarrisScore_HVC_HG3_7x7(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pSrcGxy_, vx_uint32 srcGxyStrideInBytes,
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_float32 normalization_factor
        );

int HipExec_FastCorners_XY_U8_NoSupression(
        hipStream_t stream,
        vx_uint32  capacityOfDstCorner,
        vx_keypoint_t   pHipDstCorner[],
        vx_uint32  *pHipDstCornerCount,
        vx_uint32  srcWidth, vx_uint32 srcHeight,
        vx_uint8   *pHipSrcImage,
        vx_uint32   srcImageStrideInBytes,
        vx_float32  strength_threshold
        );
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
        );

int HipExec_CannySobel_U16_U8_3x3_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySobel_U16_U8_3x3_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySobel_U16_U8_5x5_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySobel_U16_U8_5x5_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySobel_U16_U8_7x7_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySobel_U16_U8_7x7_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes
        );
int HipExec_CannySuppThreshold_U8XY_U16_3x3(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySuppThreshold_U8XY_U16_7x7(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(
        hipStream_t stream, vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 *pxyStackTop,
        vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint16 hyst_lower, vx_uint16 hyst_upper
        );
int HipExec_CannyEdgeTrace_U8_U8XY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 capacityOfXY, ago_coord2d_ushort_t xyStack[], vx_uint32 xyStackTop
        );

// miscellaneous_kernels

int HipExec_ChannelCopy_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes
        );


int HipExec_ChannelCopy(
        hipStream_t  stream,
        vx_uint32     dstWidth,
        vx_uint32     dstHeight,
        vx_uint8     * pHipDstImage,
        vx_uint32     dstImageStrideInBytes,
        const vx_uint8    * pHipSrcImage,
        vx_uint32     srcImageStrideInBytes
        );

#endif //MIVISIONX_HIP_KERNELS_H
