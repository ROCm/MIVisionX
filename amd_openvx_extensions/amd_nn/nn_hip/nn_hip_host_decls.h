/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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


#ifndef NN_HIP_HOST_DECLS_H
#define NN_HIP_HOST_DECLS_H
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <VX/vx.h>

int HipExec_Gather_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in, uint in_offset,
    uint4 in_stride, unsigned char* ind, uint ind_offset, uint4 ind_stride, unsigned char* out, uint out_offset,
    uint4 out_stride, uint axis);

int HipExec_Tile_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in,
    uint in_offset, uint4 in_stride, uint4 in_dims, unsigned char* rep, uint rep_offset, uint4 rep_stride, unsigned char* out,
    uint out_offset, uint4 out_stride);

int HipExec_Cast_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum input_type, vx_enum output_type, unsigned char* in,
    uint in_offset, uint4 in_stride, unsigned char* out, uint out_offset, uint4 out_stride);

int HipExec_image_to_tensor_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, unsigned char* in,
    uint in_offset, uint in_stride, unsigned char* out, uint out_offset, uint4 out_stride, float ka, float kb, uint reverse_channel_order);

int HipExec_tensor_to_image_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, unsigned char* in,
    uint in_offset, uint4 in_stride, unsigned char* out, uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order);

int HipExec_copy(hipStream_t stream, vx_enum type, unsigned char* inp, unsigned char* out, uint width, uint height, uint ldi, uint i_offset,
    uint ldc, uint c_offset, bool tI);

#endif //NN_HIP_HOST_DECLS_H