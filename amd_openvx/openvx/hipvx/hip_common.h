
#ifndef MIVISIONX_HIP_COMMON_H
#define MIVISIONX_HIP_COMMON_H

#include "hip/hip_runtime.h"

#define PIXELSATURATEU8(pixel)  (pixel < 0) ? 0 : ((pixel < UINT8_MAX) ? pixel : UINT8_MAX)
#define PIXELSATURATES16(pixel) (pixel < INT16_MIN) ? INT16_MIN : ((pixel < INT16_MAX) ? pixel : INT16_MAX)
#define PIXELROUNDF32(value)    ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))
#define PIXELROUNDU8(value)     ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))
#define HIPSELECT(a, b, c)  (c ? b : a)
#define HIPVXMAX3(a, b, c)  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define HIPVXMIN3(a, b, c)  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define PIXELTHRESHOLDBINARY(pixel, thresholdValue) ((pixel > thresholdValue) ? 255 : 0)
#define PIXELTHRESHOLDRANGE(pixel, thresholdLower, thresholdUpper)  ((pixel > thresholdUpper) ? 0 : ((pixel < thresholdLower) ? 0 : 255))
#define PIXELBITCHECKU1(pixel) ((pixel == (vx_int32)0) ? ((vx_uint32)0) : ((vx_uint32)1))
#define CHECKMAX(a, b)  (a > b ? 1 : 0)
#define CHECKMIN(a, b)  (a < b ? 1 : 0)

const float pi = 0x1.921fb6p+1f;
const float inv_pi = 1 / pi;

typedef struct d_uint6 {
  uint data[6];
} d_uint6;

typedef struct d_uint8 {
  uint data[8];
} d_uint8;

typedef struct d_float8 {
  float data[8];
} d_float8;

// common device kernels

__device__ __forceinline__ uint pack_(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
               __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
                   __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
                       __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
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

__device__ __forceinline__ float4 unpack_(uint src) {
    return make_float4(unpack0_(src), unpack1_(src), unpack2_(src), unpack3_(src));
}

__device__ __forceinline__ float dot2_(float2 src0, float2 src1) {
    return fmaf(src0.y, src1.y, src0.x * src1.x);
}

__device__ __forceinline__ float dot3_(float3 src0, float3 src1) {
    return fmaf(src0.z, src1.z, fmaf(src0.y, src1.y, src0.x * src1.x));
}

__device__ __forceinline__ float dot4_(float4 src0, float4 src1) {
    return fmaf(src0.w, src1.w, fmaf(src0.z, src1.z, fmaf(src0.y, src1.y, src0.x * src1.x)));
}

__device__ __forceinline__ uint max3_(uint src0, uint src1, uint src2) {
    return max(src0, max(src1, src2));
}

__device__ __forceinline__ uint min3_(uint src0, uint src1, uint src2) {
    return min(src0, min(src1, src2));
}

__device__ __forceinline__ uint median3_(uint src0, uint src1, uint src2) {
    return max(min(src0, src1), min(max(src0, src1), src2));
}

__device__ __forceinline__ uint lerp_(uint src0, uint src1, uint src2) {
    uint dst = (((((src0 >>  0) & 0xff) + ((src1 >>  0) & 0xff) + ((src2 >>  0) & 1)) >> 1) <<  0) +
                (((((src0 >>  8) & 0xff) + ((src1 >>  8) & 0xff) + ((src2 >>  8) & 1)) >> 1) <<  8) +
                (((((src0 >> 16) & 0xff) + ((src1 >> 16) & 0xff) + ((src2 >> 16) & 1)) >> 1) << 16) +
                (((((src0 >> 24) & 0xff) + ((src1 >> 24) & 0xff) + ((src2 >> 24) & 1)) >> 1) << 24);
    return dst;
}

__device__ __forceinline__ float4 fabs4(float4 src) {
    return make_float4(fabsf(src.x), fabsf(src.y), fabsf(src.z), fabsf(src.w));
}

template<class T>
__device__ __forceinline__ T hip_clamp(T v, T lo, T hi) {
    return min(max(v, lo), hi);
}

__device__ __forceinline__ short hip_convert_short_rte(float a) {
    a = rint(a);
    return (short)a;
}

__device__ __forceinline__ short hip_convert_short_sat_rte(float a) {
    a = rint(a);
    return (short)hip_clamp(a, (float)INT16_MIN, (float)INT16_MAX);
}

__device__ __forceinline__ void hip_convert_U8_U1 (uint2 * p0, unsigned char p1) {
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

__device__ __forceinline__ void hip_convert_U1_U8 (unsigned char * p0, uint2 p1) {
    unsigned char r;
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


// common device kernels - old ones, but still in use - can be removed once they aren't in use anywhere
__device__ __forceinline__ float4 uchars_to_float4(uint src) {
    return make_float4((float)(src&0xFF), (float)((src&0xFF00)>>8), (float)((src&0xFF0000)>>16), (float)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint float4_to_uchars(float4 src) {
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ uint float4ToUint(float4 src) {
    return ((int)src.x&0xFF) | (((int)src.y&0xFF)<<8) | (((int)src.z&0xFF)<<16)| (((int)src.w&0xFF) << 24);
}

__device__ __forceinline__ int4 uchars_to_int4(uint src) {
    return make_int4((int)(src&0xFF), (int)((src&0xFF00)>>8), (int)((src&0xFF0000)>>16), (int)((src&0xFF000000)>>24));
}

__device__ __forceinline__ uint int4_to_uchars(int4 src) {
    return ((uint)src.x&0xFF) | (((uint)src.y&0xFF)<<8) | (((uint)src.z&0xFF)<<16)| (((uint)src.w&0xFF) << 24);
}

__device__ __forceinline__ uchar4 uchars_to_uchar4(unsigned int src) {
    return make_uchar4((unsigned char)(src & 0xFF), (unsigned char)((src & 0xFF00) >> 8), (unsigned char)((src & 0xFF0000) >> 16), (unsigned char)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ unsigned int uchar4_to_uchars(uchar4 src) {
    return ((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8) | (((unsigned char)src.z & 0xFF) << 16) | (((unsigned char)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint4 uchars_to_uint4 (unsigned int src) {
    return make_uint4((unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8), (unsigned int)((src & 0xFF0000) >> 16), (unsigned int)((src & 0xFF000000) >> 24));
}

__device__ __forceinline__ unsigned int uint4_to_uchars (uint4 src) {
    return ((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8) | (((unsigned char)src.z & 0xFF) << 16) | (((unsigned char)src.w & 0xFF) << 24);
}

__device__ __forceinline__ uint2 uchars_to_uint2 (unsigned int src) {
    return make_uint2((unsigned int)(src & 0xFF), (unsigned int)((src & 0xFF00) >> 8));
}

__device__ __forceinline__ unsigned int uint2_to_uchars (uint2 src) {
    return (((unsigned char)src.x & 0xFF) | (((unsigned char)src.y & 0xFF) << 8));
}

__device__ __forceinline__ void prefixSum(unsigned int* output, unsigned int* input, int w, int nextpow2) {
    extern __shared__ int temp[];
    const int tdx = threadIdx.x;
    int offset = 1;
    const int tdx2 = 2*tdx;
    const int tdx2p = tdx2 + 1;
    temp[tdx2] =  tdx2 < w ? input[tdx2] : 0;
    temp[tdx2p] = tdx2p < w ? input[tdx2p] : 0;
    for(int d = nextpow2>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(tdx < d) {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    int last = temp[nextpow2 - 1];
    if(tdx == 0) temp[nextpow2 - 1] = 0;
    for(int d = 1; d < nextpow2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(tdx < d ) {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            int t  = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if(tdx2 < w)  output[tdx2 - 1] = temp[tdx2];
    if(tdx2p < w) output[tdx2p - 1] = temp[tdx2p];
    if(tdx2p < w) output[w - 1] = last;
}

#endif //MIVISIONX_HIP_COMMON_H