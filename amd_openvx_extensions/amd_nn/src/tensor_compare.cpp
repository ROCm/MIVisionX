#include <kernels.h>
/*
Returns the tensor resulted from performing the comparison operation elementwise on the input tensors A and B.
Supports the following mode values:
0 - Less than (<)
1 - Greater than (>)
2 - Less than or equal to (<=)
3 - Greater than or equal to (>=)
4 - Equal to (==)
5 - Not equal to (!=)
*/
static vx_status VX_CALLBACK validateTensorCompare(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) 
{
    // check tensor dims    
    vx_enum out_type;
    vx_size num_dims;
    vx_size input_dims[4], input2_dims[4], output_dims[4];
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input2_dims, sizeof(input2_dims)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (out_type != VX_TYPE_BOOL) return VX_ERROR_INVALID_TYPE;

    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2] || output_dims[1] != input_dims[1] || output_dims[0] != input_dims[0] ||
        output_dims[3] != input2_dims[3] || output_dims[2] != input2_dims[2] || output_dims[1] != input2_dims[1] || output_dims[0] != input2_dims[0]
    )
    {
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: tensor_compare: dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    vx_uint32 mode;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (mode < 0 || mode > 5) {
        return ERRMSG(VX_ERROR_INVALID_PARAMETERS, "validate: tensor_compare: mode value should be within 0-5(mode = %d)\n", mode);
    }
    
    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));
    return VX_SUCCESS;

}

static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,
    vx_uint32& supported_target_affinity
)
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

#if ENABLE_OPENCL
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
    //get tensor dimensions
    vx_size input_dims[4];
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    strcpy(opencl_kernel_function_name, "tensor_compare");

    opencl_work_dim = 3;
    opencl_global_work[0] = input_dims[0];
    opencl_global_work[1] = input_dims[1];
    opencl_global_work[2] = input_dims[2] * input_dims[3];

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims == 4) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32) {
        sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * in2, uint in2_offset, uint4 in2_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint mode) \n"
                "{ \n"
                "     uint x = get_global_id(0);\n"
                "     uint y = get_global_id(1);\n"
                "     uint c = get_global_id(2);\n"
                "     float value = *(__global float *)&in[in_offset + x * in_stride.s0 + y * in_stride.s1 + c * in_stride.s2];\n"
                "     float value2 = *(__global float *)&in2[in2_offset + x * in2_stride.s0 + y * in2_stride.s1 + c * in2_stride.s2];\n"
                "     out += out_offset + x  * out_stride.s0 + y * out_stride.s1 + c * out_stride.s2;\n"
                "     // compare the values and write to the output\n"
                "     bool result;\n"
                "     if (mode == 0)\n"
                "       result = (value < value2);\n"
                "     else if (mode == 1)\n"
                "       result = (value > value2);\n"
                "     else if (mode == 2)\n"
                "       result = (value <= value2);\n"
                "     else if (mode == 3)\n"
                "       result = (value >= value2);\n"
                "     else if (mode == 4)\n"
                "       result = (value == value2);\n"
                "     else if (mode == 5)\n"
                "       result = (value != value2);\n"
                "     *(__global bool *)&out[0] = result;\n"
                " }\n", opencl_kernel_function_name);
        } else {
            sprintf(item,
                  "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * in2, uint in2_offset, uint4 in2_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint mode) \n"
                "{ \n"
                "     uint x = get_global_id(0);\n"
                "     uint y = get_global_id(1);\n"
                "     uint c = get_global_id(2);\n"
                "     half value = *(__global half *)&in[in_offset + x * in_stride.s0 + y * in_stride.s1 + c * in_stride.s2];\n"
                "     half value2 = *(__global half *)&in2[in2_offset + x * in2_stride.s0 + y * in2_stride.s1 + c * in2_stride.s2];\n"
                "     out += out_offset + x  * out_stride.s0 + y * out_stride.s1 + c * out_stride.s2;\n"
                "     // compare the values and write to the output\n"
                "     bool result;\n"
                "     if (mode == 0)\n"
                "       result = (value < value2);\n"
                "     else if (mode == 1)\n"
                "       result = (value > value2);\n"
                "     else if (mode == 2)\n"
                "       result = (value <= value2);\n"
                "     else if (mode == 3)\n"
                "       result = (value >= value2);\n"
                "     else if (mode == 4)\n"
                "       result = (value == value2);\n"
                "     else if (mode == 5)\n"
                "       result = (value != value2);\n"
                "     *(__global bool *)&out[0] = result;\n"
                    " }\n", opencl_kernel_function_name);
        }
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}
#endif

static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    return VX_ERROR_NOT_IMPLEMENTED;
}

vx_status publishTensorCompare(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.tensor_compare", VX_KERNEL_TENSOR_COMPARE_AMD, host_kernel, 4, validateTensorCompare, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorCompareNode(vx_graph graph, vx_tensor input, vx_tensor input2, vx_tensor output, vx_int32 mode)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar s_mode = vxCreateScalarWithSize(context, VX_TYPE_INT32, &mode, sizeof(mode));
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)input2,
            (vx_reference)output,
            (vx_reference)s_mode
        };
        node = createNode(graph, VX_KERNEL_TENSOR_COMPARE_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}

