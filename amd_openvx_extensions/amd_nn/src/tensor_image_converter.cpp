/*
Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include "kernels.h"

static vx_status VX_CALLBACK validateTensorToImageKernel(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input configuration
    vx_enum type;
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: tensor2img: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: tensor2img: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims[0])*num_dims));
    if ((input_dims[2] != 3 && input_dims[2] != 1) || ((input_dims[0] & 3) != 0))
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: tensor2img: input_dims[%ldx%ldx%ldx%ld]\n", input_dims[3], input_dims[2], input_dims[1], input_dims[0]);
    vx_enum scalar_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: tensor2img: #2 type=%d (must be float)\n", scalar_type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: tensor2img: #3 type=%d (must be float)\n", scalar_type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_BOOL) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: tensor2img: #4 type=%d (must be bool)\n", scalar_type);

    // set output image configuration
    vx_uint32 width = (vx_uint32)input_dims[0];
    vx_uint32 height = (vx_uint32)(input_dims[1]*input_dims[3]);
    vx_df_image format = (input_dims[2] == 3) ? VX_DF_IMAGE_RGB : VX_DF_IMAGE_U8;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format)));

    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
    )
{
    // get configuration
    vx_df_image format;
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 };
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims[0])*num_dims));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    vx_uint32 width = (vx_uint32)input_dims[0];
    vx_uint32 height = (vx_uint32)input_dims[1];
    vx_uint32 N = (vx_uint32)input_dims[3];

    // compute global work
    vx_uint32 width_div_4 = (width + 3) / 4;
    opencl_work_dim = 3;
    opencl_local_work[0] = 8;
    opencl_local_work[1] = 8;
    opencl_local_work[2] = 1;
    opencl_global_work[0] = (width_div_4  + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
    opencl_global_work[1] = (height + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);
    opencl_global_work[2] = N;

    // generate OpenCL C code
    strcpy(opencl_kernel_function_name, "tensor_to_image");
    if(format == VX_DF_IMAGE_RGB) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32){
        sprintf(item,
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
            "void %s(__global uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset, float ka, float kb, uint reverse_channel_order)\n"
            "{\n"
            "    uint x = get_global_id(0) * 4;\n"
            "    uint y = get_global_id(1);\n"
            "    uint n = get_global_id(2);\n"
            "    if(x < %d && y < %d) {\n"
            "        i0_buf += i0_offset + n * i0_stride.s3 + y * i0_stride.s1 + x * i0_stride.s0;\n"
            "        float4 r = *(__global float4 *)&i0_buf[reverse_channel_order ? 2 * i0_stride.s2 : 0];\n"
            "        float4 g = *(__global float4 *)&i0_buf[                              i0_stride.s2  ];\n"
            "        float4 b = *(__global float4 *)&i0_buf[reverse_channel_order ? 0 : 2 * i0_stride.s2];\n"
            "        r = r * (float4)ka + (float4)kb;\n"
            "        g = g * (float4)ka + (float4)kb;\n"
            "        b = b * (float4)ka + (float4)kb;\n"
            "        uint3 u3;\n"
            "        u3.s0 = amd_pack((float4)(r.s0, g.s0, b.s0, r.s1));\n"
            "        u3.s1 = amd_pack((float4)(g.s1, b.s1, r.s2, g.s2));\n"
            "        u3.s2 = amd_pack((float4)(b.s2, r.s3, g.s3, b.s3));\n"
            "        vstore3(u3, 0, (__global uint *)&o0_buf[o0_offset + (y + n * %d) * o0_stride + x * 3]);\n"
            "    }\n"
            "}\n"
            , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        }else
        {
            sprintf(item,
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
            "void %s(__global uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset, float ka, float kb, uint reverse_channel_order)\n"
            "{\n"
            "    uint x = get_global_id(0) * 4;\n"
            "    uint y = get_global_id(1);\n"
            "    uint n = get_global_id(2);\n"
            "    if(x < %d && y < %d) {\n"
            "        i0_buf += i0_offset + n * i0_stride.s3 + y * i0_stride.s1 + x * i0_stride.s0;\n"
            "        half4 r = *(__global half4 *)&i0_buf[reverse_channel_order ? 2 * i0_stride.s2 : 0];\n"
            "        half4 g = *(__global half4 *)&i0_buf[                              i0_stride.s2  ];\n"
            "        half4 b = *(__global half4 *)&i0_buf[reverse_channel_order ? 0 : 2 * i0_stride.s2];\n"
            "        r = r * (half4)ka + (half4)kb;\n"
            "        g = g * (half4)ka + (half4)kb;\n"
            "        b = b * (half4)ka + (half4)kb;\n"
            "        uint3 u3;\n"
            "        u3.s0 = amd_pack(convert_float4(r.s0, g.s0, b.s0, r.s1));\n"
            "        u3.s1 = amd_pack(convert_float4(g.s1, b.s1, r.s2, g.s2));\n"
            "        u3.s2 = amd_pack(convert_float4(b.s2, r.s3, g.s3, b.s3));\n"
            "        vstore3(u3, 0, (__global uint *)&o0_buf[o0_offset + (y + n * %d) * o0_stride + x * 3]);\n"
            "    }\n"
            "}\n"
            , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        }
        opencl_kernel_code = item;
    }
    else {
        char item[8192];
        if (type == VX_TYPE_FLOAT32){
        sprintf(item,
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
            "void %s(__global uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset, float ka, float kb, uint reverse_channel_order)\n"
            "{\n"
            "    uint x = get_global_id(0) * 4;\n"
            "    uint y = get_global_id(1);\n"
            "    uint n = get_global_id(2);\n"
            "    if(x < %d && y < %d) {\n"
            "        i0_buf += i0_offset + n * i0_stride.s3 + y * i0_stride.s1 + x * i0_stride.s0;\n"
            "        float4 i = *(__global float4 *)i0_buf;\n"
            "        i = i * (float4)ka + (float4)kb;\n"
            "        *(__global uint *)&o0_buf[o0_offset + (y + n * %d) * o0_stride + x] = amd_pack((float4)(i.s0, i.s1, i.s2, i.s3));\n"
            "    }\n"
            "}\n"
            , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        }else
        {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
                "void %s(__global uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset, float ka, float kb, uint reverse_channel_order)\n"
                "{\n"
                "    uint x = get_global_id(0) * 4;\n"
                "    uint y = get_global_id(1);\n"
                "    uint n = get_global_id(2);\n"
                "    if(x < %d && y < %d) {\n"
                "        i0_buf += i0_offset + n * i0_stride.s3 + y * i0_stride.s1 + x * i0_stride.s0;\n"
                "        half4 i = *(__global half4 *)i0_buf;\n"
                "        float4 o = convert_float4(i) * (float4)ka + (float4)kb;\n"
                "        *(__global uint *)&o0_buf[o0_offset + (y + n * %d) * o0_stride + x] = amd_pack((float4)(o.s0, o.s1, o.s2, o.s3));\n"
                "    }\n"
                "}\n"
                , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        }
        opencl_kernel_code = item;
    }

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "KERNEL tensor_to_image output " << width << "x" << height << " " << N << std::endl;
#endif

    return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishTensorToImageConvert(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.convert_tensor_to_image", VX_KERNEL_CONVERT_TENSOR_TO_IMAGE_AMD, host_kernel, 5, validateTensorToImageKernel, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxConvertTensorToImageNode(vx_graph graph, vx_tensor input, vx_image output, vx_float32 a, vx_float32 b, vx_bool reverse_channel_order)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_a = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &a, sizeof(a));
        vx_scalar s_b = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &b, sizeof(b));
        vx_scalar s_order = vxCreateScalarWithSize(context, VX_TYPE_BOOL, &reverse_channel_order, sizeof(reverse_channel_order));
        if(vxGetStatus((vx_reference)s_order) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)input,
                (vx_reference)output,
                (vx_reference)s_a,
                (vx_reference)s_b,
                (vx_reference)s_order
            };
            node = createNode(graph, VX_KERNEL_CONVERT_TENSOR_TO_IMAGE_AMD, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_a);
            vxReleaseScalar(&s_b);
            vxReleaseScalar(&s_order);
        }
    }
    return node;
}
