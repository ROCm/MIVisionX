/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <half.hpp>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "rali_hip_kernels.h"

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp32
(
    const unsigned char*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    uint4 nchw,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels
)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w; const int H = nchw.z; const int C = nchw.y;
    const int img_offset = C * W * H;

    if ((x >= W) || (y >= H)) return;
    for (unsigned int n=0; n < nchw.x; n++) {
        unsigned int srcIdx =  (y*W + x) * C;     // src is RGB
        unsigned int dstIdx =  y*W + x;
        // copy float3  pixels to dst
        if (C == 3){
            const uchar *inp_img = &inp_image_u8[n*img_offset + dst_buf_offset];
            float3 *out_tensor = (float3 *)((float*)output_tensor + dst_buf_offset + n*img_offset);
            if (reverse_channels)
                out_tensor[dstIdx] = make_float3((float)inp_img[srcIdx+2], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx])*multiplier + offset;
            else
                out_tensor[dstIdx] = make_float3((float)inp_img[srcIdx], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx+2])*multiplier + offset;
        } else{
            const uchar *inp_img = &inp_image_u8[n*img_offset + dst_buf_offset];
            float *out_tensor = (float *)output_tensor + dst_buf_offset + n*img_offset;
            out_tensor[dstIdx] = (float)inp_img[srcIdx]*multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp16
(
    const unsigned char*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    uint4 nchw,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels
)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w; const int H = nchw.z; const int C = nchw.y;
    const int img_offset = C * W * H;

    if ((x >= W) || (y >= H)) return;
    for (unsigned int n=0; n < nchw.x; n++) {
        unsigned short *out_tensor = (unsigned short *)output_tensor + dst_buf_offset + n*img_offset;
        unsigned int srcIdx =  (y*W + x) * C;
        // copy float3  pixels to dst
        if (C == 3){
            unsigned int dstIdx =  y*W + x*3;
            const uchar *inp_img = &inp_image_u8[n*img_offset + dst_buf_offset];
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[srcIdx+2], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx])*multiplier + offset;
            else
                dst = make_float3((float)inp_img[srcIdx], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx+2])*multiplier + offset;
            out_tensor[dstIdx] = __float2half(dst.x);
            out_tensor[dstIdx+1] = __float2half(dst.y);
            out_tensor[dstIdx+2] = __float2half(dst.z);
        } else{
            unsigned int dstIdx =  y*W + x;
            const uchar *inp_img = &inp_image_u8[n*img_offset];
            float *out_tensor = (float *)output_tensor + n*img_offset;
            out_tensor[dstIdx] = __float2half((float)inp_img[srcIdx]*multiplier.x + offset.x);
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp32
(
    const uchar*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    uint4 nchw,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels
)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w; const int H = nchw.z; const int C = nchw.y;
    const int img_offset = C * W * H;

    if ((x >= W) || (y >= H)) return;
    for (unsigned int n=0; n < nchw.x; n++) {
        unsigned int srcIdx =  (y*W + x)*C;
        unsigned int dstIdx =  y*W + x;
        // copy float3  pixels to dst
        const uchar *inp_img = &inp_image_u8[n*img_offset+dst_buf_offset];
        float *out_tensor = (float *)output_tensor + n*img_offset + dst_buf_offset;
        if (C == 3){
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[srcIdx+2], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx])*multiplier + offset;
            else
                dst = make_float3((float)inp_img[srcIdx], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx+2])*multiplier + offset;
            out_tensor[dstIdx] = dst.x;
            out_tensor[dstIdx+W*H] = dst.y;
            out_tensor[dstIdx+W*H*2] = dst.z;
        } else{
            out_tensor[dstIdx] = (float)inp_img[srcIdx]*multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp16
(
    const uchar*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    uint4 nchw,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels
)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w; const int H = nchw.z; const int C = nchw.y;
    const int img_offset = C * W * H;

    if ((x >= W) || (y >= H)) return;
    for (unsigned int n=0; n < nchw.x; n++) {
        unsigned short *out_tensor = (unsigned short *)output_tensor + n*img_offset+dst_buf_offset;
        const uchar *inp_img = &inp_image_u8[n*img_offset+dst_buf_offset];
        unsigned int srcIdx =  (y*W + x)*C;
        // copy float3  pixels to dst
        unsigned int dstIdx =  y*W + x;
        if (C == 3){
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[srcIdx+2], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx])*multiplier + offset;
            else
                dst = make_float3((float)inp_img[srcIdx], (float) inp_img[srcIdx+1], (float)inp_img[srcIdx+2])*multiplier + offset;
            out_tensor[dstIdx] = __float2half(dst.x);
            out_tensor[dstIdx+W*H] = __float2half(dst.y);
            out_tensor[dstIdx+W*H*2] = __float2half(dst.z);
        } else{
            out_tensor[dstIdx] = __float2half((float)inp_img[srcIdx]*multiplier.x + offset.x);
        }
    }
}

int HipExecCopyInt8ToNHWC
(
    hipStream_t stream,
    const void*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    const unsigned int     n,
    const unsigned int     c,
    const unsigned int     h,
    const unsigned int     w,
    float     multiplier0,
    float     multiplier1,
    float     multiplier2,
    float     offset0,
    float     offset1,
    float     offset2,
    unsigned int reverse_channels,
    unsigned int fp16
)
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = w, globalThreads_y = h;
    if (!fp16){
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp32,
                        dim3(ceil((float) globalThreads_x / localThreads_x), ceil((float) globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y),
                        0, stream, (const uchar*)inp_image_u8, output_tensor, dst_buf_offset,
                        make_uint4(n, c, h, w),
                        make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                        reverse_channels);
    }else{
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp16,
                        dim3(ceil((float) globalThreads_x / localThreads_x), ceil((float) globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y),
                        0, stream, (const uchar*)inp_image_u8, output_tensor, dst_buf_offset,
                        make_uint4(n, c, h, w),
                        make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                        reverse_channels);
    }
    return 0;
}

int HipExecCopyInt8ToNCHW
(
    hipStream_t stream,
    const void*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    const unsigned int     n,
    const unsigned int     c,
    const unsigned int     h,
    const unsigned int     w,
    float     multiplier0,
    float     multiplier1,
    float     multiplier2,
    float     offset0,
    float     offset1,
    float     offset2,
    unsigned int reverse_channels,
    unsigned int fp16
)
{
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = w, globalThreads_y = h;
    if (!fp16){
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp32,
                        dim3(ceil((float) globalThreads_x / localThreads_x), ceil((float) globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y),
                        0, stream, (const uchar*)inp_image_u8, output_tensor, dst_buf_offset,
                        make_uint4(n, c, h, w),
                        make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                        reverse_channels);
    }else{
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp16,
                        dim3(ceil((float) globalThreads_x / localThreads_x), ceil((float) globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y),
                        0, stream, (const uchar*)inp_image_u8, output_tensor, dst_buf_offset,
                        make_uint4(n, c, h, w),
                        make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                        reverse_channels);
    }
    return 0;
}

