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

#include "../../../amd_openvx/openvx/hipvx/hip_common_funcs.h"
#include "nn_hip_host_decls.h"

// ----------------------------------------------------------------------------
// Neural Network kernels for hip backend
// ----------------------------------------------------------------------------

typedef struct d_half4 {
  __half data[4];
} d_half4;

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Gather_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* ind, uint ind_offset, uint4 ind_stride,
 uchar* out, uint out_offset, uint4 out_stride, uint axis) {

   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint c = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

   int indices = *(int*)&ind[ind_offset + y * ind_stride.x];
   T value;
   uint offset;
   if (axis == 0) {
       value = *(T*)&in[in_offset + x * in_stride.x + indices * in_stride.y + c * in_stride.z];
       offset = out_offset + x * out_stride.x + y * out_stride.y + c * out_stride.z;
   } else if (axis == 1) {
       value = *(T*)&in[in_offset + indices * in_stride.x + c * in_stride.y];
       offset = out_offset + y * out_stride.x + c * out_stride.y;
   } else if (axis == 2) {
       value = *(T*)&in[in_offset + c * in_stride.x];
       offset = out_offset + c * out_stride.x;
   }
   out += offset;
   *(T *)&out[0] = value;
}

int HipExec_Gather_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uchar* ind, uint ind_offset, uint4 ind_stride, uchar* out, uint out_offset,
    uint4 out_stride, uint axis) {

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_Gather_layer<float>, dim3(ceil((float)globalThreads.x/localThreads.x),
            ceil((float)globalThreads.y/localThreads.y), ceil((float)globalThreads.z/localThreads.z)),
            dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            ind, ind_offset, ind_stride, out, out_offset, out_stride, axis);
    } else {
        hipLaunchKernelGGL(Hip_Gather_layer<__half>, dim3(ceil((float)globalThreads.x/localThreads.x),
            ceil((float)globalThreads.y/localThreads.y), ceil((float)globalThreads.z/localThreads.z)),
            dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            ind, ind_offset, ind_stride, out, out_offset, out_stride, axis);
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Tile_layer(uchar* in, uint in_offset, uint4 in_stride, uint4 in_dims, uchar* rep, uint rep_offset,
    uint4 rep_stride, uchar* out, uint out_offset, uint4 out_stride) {

   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

   uint nx = x % in_dims.x;
   uint ny = y % in_dims.y;
   uint nz = z % in_dims.z;

   T value = *(T*)&in[in_offset + nx * in_stride.x + ny * in_stride.y + nz * in_stride.z];
   uint offset = out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   out += offset;
   *(T *)&out[0] = value;
}

int HipExec_Tile_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uint4 in_dims, uchar* rep, uint rep_offset, uint4 rep_stride, uchar* out,
    uint out_offset, uint4 out_stride) {

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_Tile_layer<float>, dim3(ceil((float)globalThreads.x/localThreads.x),
            ceil((float)globalThreads.y/localThreads.y), ceil((float)globalThreads.z/localThreads.z)),
            dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            in_dims, rep, rep_offset, rep_stride, out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_Tile_layer<__half>, dim3(ceil((float)globalThreads.x/localThreads.x),
            ceil((float)globalThreads.y/localThreads.y), ceil((float)globalThreads.z/localThreads.z)),
            dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            in_dims, rep, rep_offset, rep_stride, out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   float4 f = *(float4 *)in;
   *(int4 *)&out[0] = make_int4(__float2int_rd(f.x), __float2int_rd(f.y), __float2int_rd(f.z), __float2int_rd(f.w));
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   float4 f = *(float4 *)in;
   *(longlong4 *)&out[0] = make_longlong4(__float2ll_rd(f.x), __float2ll_rd(f.y), __float2ll_rd(f.z), __float2ll_rd(f.w));
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_float_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(float4 *)&out[0] = *(float4 *)in;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_int32_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   int4 f = *(int4 *)in;
   *(longlong4 *)&out[0] = make_longlong4(f.x, f.y, f.z, f.w);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_int64_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   longlong4 f = *(longlong4 *)in;
   *(int4 *)&out[0] = make_int4(f.x, f.y, f.z, f.w);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(int *)&out[0] = __float2int_rd(*(float *)in);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(long long int*)&out[0] = __float2ll_rd(*(float *)in);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_float_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(float *)&out[0] = *(float *)in;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_int32(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   int f = *(int *)in;
   *(long long int*)&out[0] = (long long int)f;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_int64(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   long long int f = *(long long int*)in;
   *(int *)&out[0] = (int)f;
}

int HipExec_Cast_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum input_type, vx_enum output_type, uchar* in, uint in_offset,
    uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {

    bool input_element_count_multiple_of_4 = ((globalThreads.x * globalThreads.y) & 3) ? false : true;
    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));
    dim3 blockDim = dim3(localThreads.x, localThreads.y, localThreads.z);
    if(input_element_count_multiple_of_4) {
        if(input_type == VX_TYPE_FLOAT32) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_float_v, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_float_v, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_FLOAT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_float_float_v, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT32) {
            if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_int32_v, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT64) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_int64_v, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        }
    } else {
        if(input_type == VX_TYPE_FLOAT32) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_float, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_float, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_FLOAT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_float_float, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT32) {
            if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_int32, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT64) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_int64, gridDim, blockDim, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        }
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_RGBimage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < srcWidth && y < srcHeight) {
        uint ioffset = in_offset + (y + z * srcHeight) * in_stride + x * 3;
        uint2 rgb2 = *(uint2 *)&in[ioffset & ~3];
        uint rgb = hip_bytealign(rgb2.y, rgb2.x, ioffset & 3);
        T r = (T)sc1 * (T)hip_unpack0(rgb) + (T)sc2;
        T g = (T)sc1 * (T)hip_unpack1(rgb) + (T)sc2;
        T b = (T)sc1 * (T)hip_unpack2(rgb) + (T)sc2;
        out += out_offset + z * out_stride.w + y * out_stride.y + x * out_stride.x;
        *(T *)&out[0] = reverse_channel_order ? b : r;
        *(T *)&out[out_stride.z] = g;
        *(T *)&out[2 * out_stride.z] = reverse_channel_order ? r : b;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_U8floatImage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    if(x < srcWidth && y < srcHeight) {
        uint u4 = *(uint *)&in[in_offset + (y + z * srcHeight) * in_stride + x];
        float p0 = sc1 * hip_unpack0(u4) + sc2;
        float p1 = sc1 * hip_unpack1(u4) + sc2;
        float p2 = sc1 * hip_unpack2(u4) + sc2;
        float p3 = sc1 * hip_unpack3(u4) + sc2;
        *(float4 *)&out[out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x] = make_float4(p0, p1, p2, p3);
    }
}

__global__ void __attribute__((visibility("default")))
Hip_U8halfImage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    if(x < srcWidth && y < srcHeight) {
        uint u4 = *(uint *)&in[in_offset + (y + z * srcHeight) * in_stride + x];

        __half hsc1 = __float2half(sc1);
        __half hsc2 = __float2half(sc2);

        __half p0 = hsc1 * __float2half(hip_unpack0(u4)) + hsc2;
        __half p1 = hsc1 * __float2half(hip_unpack1(u4)) + hsc2;
        __half p2 = hsc1 * __float2half(hip_unpack2(u4)) + hsc2;
        __half p3 = hsc1 * __float2half(hip_unpack3(u4)) + hsc2;

        d_half4 p;
        p.data[0] = p0;
        p.data[1] = p1;
        p.data[2] = p2;
        p.data[3] = p3;
        *(d_half4 *)&out[out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x] = p;
    }
}

int HipExec_image_to_tensor_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, uchar* in,
    uint in_offset, uint in_stride, uchar* out, uint out_offset, uint4 out_stride, float sc1, float sc2, uint reverse_channel_order) {

    dim3 blockDim(8, 8, 1);
    uint newWidth = (format == VX_DF_IMAGE_RGB) ? width : (width + 3) / 4;
    dim3 gridDim = dim3(ceil((float)newWidth / blockDim.x), ceil((float)height / blockDim.y), ceil((float)N / blockDim.z));

    if (format == VX_DF_IMAGE_RGB) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_RGBimage_to_tensor_layer<float>, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_RGBimage_to_tensor_layer<__half>, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else if (format == VX_DF_IMAGE_U8) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_U8floatImage_to_tensor_layer, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2);
        } else {
            hipLaunchKernelGGL(Hip_U8halfImage_to_tensor_layer, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2);
        }
    } else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_RGBfloatImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out, uint out_offset,
    uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        float4 r = *(float4 *)&in[reverse_channel_order ? 2 * in_stride.z : 0];
        float4 g = *(float4 *)&in[in_stride.z];
        float4 b = *(float4 *)&in[reverse_channel_order ? 0 : 2 * in_stride.z];
        r = r * (float4)sc1 + (float4)sc2;
        g = g * (float4)sc1 + (float4)sc2;
        b = b * (float4)sc1 + (float4)sc2;
        uint3 u3;
        u3.x = hip_pack(make_float4(r.x, g.x, b.x, r.y));
        u3.y = hip_pack(make_float4(g.y, b.y, r.z, g.z));
        u3.z = hip_pack(make_float4(b.z, r.w, g.w, b.w));
        *(uint3 *)&out[out_offset + (y + z * outHeight) * out_stride + x * 3] = u3;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_RGBhalfImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        d_half4 r = *(d_half4 *)&in[reverse_channel_order ? 2 * in_stride.z : 0];
        d_half4 g = *(d_half4 *)&in[in_stride.z];
        d_half4 b = *(d_half4 *)&in[reverse_channel_order ? 0 : 2 * in_stride.z];

        __half hsc1 = __float2half(sc1);
        __half hsc2 = __float2half(sc2);

        r.data[0] = r.data[0] * hsc1 + hsc2;
        r.data[1] = r.data[1] * hsc1 + hsc2;
        r.data[2] = r.data[2] * hsc1 + hsc2;
        r.data[3] = r.data[3] * hsc1 + hsc2;

        g.data[0] = g.data[0] * hsc1 + hsc2;
        g.data[1] = g.data[1] * hsc1 + hsc2;
        g.data[2] = g.data[2] * hsc1 + hsc2;
        g.data[3] = g.data[3] * hsc1 + hsc2;

        b.data[0] = b.data[0] * hsc1 + hsc2;
        b.data[1] = b.data[1] * hsc1 + hsc2;
        b.data[2] = b.data[2] * hsc1 + hsc2;
        b.data[3] = b.data[3] * hsc1 + hsc2;

        uint3 u3;
        u3.x = hip_pack(make_float4(__half2float(r.data[0]), __half2float(g.data[0]), __half2float(b.data[0]), __half2float(r.data[1])));
        u3.y = hip_pack(make_float4(__half2float(g.data[1]), __half2float(b.data[1]), __half2float(r.data[2]), __half2float(g.data[2])));
        u3.z = hip_pack(make_float4(__half2float(b.data[2]), __half2float(r.data[3]), __half2float(g.data[3]), __half2float(b.data[3])));
        *(uint3 *)&out[out_offset + (y + z * outHeight) * out_stride + x * 3] = u3;
    }
}


__global__ void __attribute__((visibility("default")))
Hip_tensor_to_U8floatImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        float4 f = *(float4 *)in;
        f = f * (float4)sc1 + (float4)sc2;
        *(uint *)&out[out_offset + (y + z * outHeight) * out_stride + x] = hip_pack(f);
    }
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_U8halfImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        d_half4 h = *(d_half4 *)in;

        float4 f;
        f.x = __half2float(h.data[0]);
        f.y = __half2float(h.data[1]);
        f.z = __half2float(h.data[2]);
        f.w = __half2float(h.data[3]);

        f = f * (float4)sc1 + (float4)sc2;
        *(uint *)&out[out_offset + (y + z * outHeight) * out_stride + x] = hip_pack(f);
    }
}

int HipExec_tensor_to_image_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, uchar* in,
    uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    dim3 blockDim(8, 8, 1);
    dim3 gridDim = dim3(ceil((float)((width + 3) / 4) / blockDim.x), ceil((float)height / blockDim.y), ceil((float)N / blockDim.z));

    if (format == VX_DF_IMAGE_RGB) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_tensor_to_RGBfloatImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_tensor_to_RGBhalfImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else if (format == VX_DF_IMAGE_U8) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_tensor_to_U8floatImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_tensor_to_U8halfImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
copy_v1(const T* inp, T* out, uint width, uint height, uint BLKW, uint ldi, uint i_offset, uint ldc, uint c_offset) {
    __shared__ float lbuf[256];
    uint gx = blockIdx.x;
    uint gy = blockIdx.y;
    uint lx = threadIdx.x;
    uint ly = threadIdx.y;
    uint ix = hip_mad24(gx, BLKW, lx);
    uint iy = hip_mad24(gy, BLKW, ly);
    if (ix < width && iy < height) {
        uint iloc = iy * ldi + ix + i_offset;
        lbuf[hip_mad24(ly, BLKW + 1, lx)] = inp[iloc];
    }
    __syncthreads();
    uint ox = hip_mad24(gy, BLKW, lx);
    uint oy = hip_mad24(gx, BLKW, ly);
    if(oy < width && ox < height) {
        uint oloc = oy * ldc + ox + c_offset;
        out[oloc] = lbuf[hip_mad24(lx, BLKW + 1, ly)];
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
copy_v2(const T* inp, T* out, uint width, uint height, uint ldi, uint i_offset, uint ldc, uint c_offset) {
    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if(x < width && y < height) {
        uint i = y * ldi + x + i_offset;
        uint o = y * ldc + x + c_offset;
        out[o] = inp[i];
    }
}

int HipExec_copy(hipStream_t stream, vx_enum type, uchar* inp, uchar* out, uint width, uint height, uint ldi, uint i_offset,
    uint ldc, uint c_offset, bool tI) {
    if(tI) {
        dim3 blockDim(16, 16, 1);
        dim3 gridDim = dim3(ceil((float)width / blockDim.x), ceil((float)height / blockDim.y), 1);
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(copy_v1<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, blockDim.x, ldi,
                i_offset, ldc, c_offset);
        } else {
            hipLaunchKernelGGL(copy_v1<__half>, gridDim, blockDim, 0, stream, (__half*)inp, (__half*)out, width, height, blockDim.x, ldi,
                i_offset, ldc, c_offset);
        }
    } else {
        dim3 blockDim(64, 1, 1);
        dim3 gridDim = dim3(ceil((float)width / blockDim.x), height, 1);
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(copy_v2<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, ldi, i_offset, ldc, c_offset);
        } else {
            hipLaunchKernelGGL(copy_v2<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, ldi, i_offset, ldc, c_offset);
        }
    }
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_permute_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* order_buf, uint order_offset, uint order_cap, uchar* out, uint out_offset,
    uint4 out_stride) {
    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    int num_axis = order_cap;
    int i = x * out_stride.x + y * out_stride.y + z * out_stride.z;
    int old_idx = 0;
    int idx = i;
    for(int k = num_axis - 1, j = 0; k >= 0; k--, j++) {
        int order = 3 - ((int *)(order_buf + order_offset))[j];
        old_idx += (idx / out_stride.data[k]) * (in_stride.data[order]);
        idx %= (out_stride.data[k]);
    }
    out += out_offset + i;
    in += in_offset + old_idx;
    *(float4 *)&out[0] = *(float4 *)&in[0];
}

int HipExec_permute_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uchar* in, uint in_offset, uint4 in_stride, uchar* order_buf,
    uint order_offset, uint order_cap, uchar* out, uint out_offset, uint4 out_stride) {

    hipLaunchKernelGGL(Hip_permute_layer, dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y),
        ceil((float)globalThreads.z/localThreads.z)), dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
        order_buf, order_offset, order_cap, out, out_offset, out_stride);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_log_layer(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     float4 value = *(float4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
     *(float4 *)&out[0] = make_float4(log(value.x), log(value.y), log(value.z), log(value.w));
 }

__global__ void __attribute__((visibility("default")))
Hip_tensor_log_layer_half(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     d_half4 value = *(d_half4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
     d_half4 p;
     p.data[0] = hlog(value.data[0]);
     p.data[1] = hlog(value.data[1]);
     p.data[2] = hlog(value.data[2]);
     p.data[3] = hlog(value.data[3]);
     *(d_half4 *)&out[0] = p;
 }

int HipExec_tensor_log_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar *in, uint in_offset, uint4 in_stride, uchar *out,
    uint out_offset, uint4 out_stride) {

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_tensor_log_layer, dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y),
            ceil((float)globalThreads.z/localThreads.z)), dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_tensor_log_layer_half, dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y),
            ceil((float)globalThreads.z/localThreads.z)), dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_exp_layer(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     float4 value = *(float4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x  * out_stride.x + y * out_stride.y + z * out_stride.z;
     *(float4 *)&out[0] = make_float4(exp(value.x), exp(value.y), exp(value.z), exp(value.w));
 }

__global__ void __attribute__((visibility("default")))
Hip_tensor_exp_layer_half(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    d_half4 value = *(d_half4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;

     d_half4 p;
     p.data[0] = hexp(value.data[0]);
     p.data[1] = hexp(value.data[1]);
     p.data[2] = hexp(value.data[2]);
     p.data[3] = hexp(value.data[3]);
     *(d_half4 *)&out[0] = p;
 }

int HipExec_tensor_exp_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar *in, uint in_offset, uint4 in_stride, uchar *out,
    uint out_offset, uint4 out_stride) {

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_tensor_exp_layer, dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y),
            ceil((float)globalThreads.z/localThreads.z)), dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_tensor_exp_layer_half, dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y),
            ceil((float)globalThreads.z/localThreads.z)), dim3(localThreads.x, localThreads.y, localThreads.z), 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}
