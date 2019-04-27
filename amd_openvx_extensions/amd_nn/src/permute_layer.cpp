#include "kernels.h"

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_enum scalar_type;
    vx_int32 scalar_value;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &scalar_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(scalar_value > 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: permute: #2 scalar type=%d (not supported yet)\n", scalar_value);


    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
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
    vx_size input_dims[4], output_dims[4];
    vx_size num_of_dims;
    vx_int32 scalar_value;
    vx_enum type;

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &scalar_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));


    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    strcpy(opencl_kernel_function_name, "permute_layer");
    //vx_uint32 input_dim_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];

    opencl_work_dim = 3;
    opencl_global_work[0] = output_dims[1];
    opencl_global_work[1] = output_dims[2];
    opencl_global_work[2] = output_dims[3];
    
    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims == 4) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32) {
        sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, uint s_value,"\
                "                 __global uchar * out, uint out_offset, uint4 out_stride)\n"
                "{   \n"
                "   uint x = get_global_id(0); \n "
                "   uint y = get_global_id(1); \n "
                "   uint c = get_global_id(2); \n "
                "   int scalar_value = %d; \n"
                "   if(scalar_value == 0) \n"
                "   { \n"
                "      float value = *(__global float *)&in[c*in_stride.s3 + y*in_stride.s2 + x*in_stride.s1]; \n"
                "      out += c*out_stride.s3 + y*out_stride.s2 + x*out_stride.s1; \n"
                "      *(__global float *)&out[0] = value; \n"
                "   } \n"
                "   if(scalar_value == 1)\n"
                "   {  \n"
                "      float value = *(__global float *)&in[c*get_global_size(0)*get_global_size(1)*sizeof(float) + y*get_global_size(0)*sizeof(float) + x*sizeof(float)]; \n"
                "      out += y*get_global_size(0)*get_global_size(2)*sizeof(float) + x*get_global_size(2)*sizeof(float) + c*sizeof(float); \n"
                "      *(__global float *)&out[0] = value; \n"
                "   } \n"
                "}\n", opencl_kernel_function_name, scalar_value);
        } else {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, uint s_value,"\
                "                 __global uchar * out, uint out_offset, uint4 out_stride)\n"
                "{   \n"
                "   uint x = get_globaloutid(0); \n "
                "   uint y = get_global_id(1); \n "
                "   uint c = get_global_id(2); \n "
                "   int scalar_value = %d; \n"
                "   if(scalar_value == 0) \n"
                "   { \n"
                "      half value = *(__global half *)&in[c*in_stride.s3 + y*in_stride.s2 + x*in_stride.s1]; \n"
                "      out += c*out_stride.s3 + y*out_stride.s2 + x*out_stride.s1; \n"
                "      *(__global half *)&out[0] = value; \n"
                "   } \n"
                "   if(scalar_value == 1)\n"
                "   {  \n"
                "      half value = *(__global half *)&in[c*get_global_size(0)*get_global_size(1)*sizeof(float) + y*get_global_size(0)*sizeof(float) + x*sizeof(float)]; \n"
                "      out += y*get_global_size(0)*get_global_size(2)*sizeof(float) + x*get_global_size(2)*sizeof(float) + c*sizeof(float); \n"
                "      *(__global half *)&out[0] = value; \n"
                "   } \n"
                "}\n", opencl_kernel_function_name, scalar_value);
        }
        opencl_kernel_code = item;
    }

    return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishPermuteLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.permute_layer", VX_KERNEL_PERMUTE_LAYER_AMD, host_kernel, 3, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}


/*order value  |     function
*     0        |  order: 0,1,2,3 (doesn't permute)
*     1        |  order: 0,2,3,1 ([n,c,h,w] --> [n,h,w,c])
*/
VX_API_ENTRY vx_node VX_API_CALL vxPermuteLayer(vx_graph graph, vx_tensor input, vx_int32 order, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_order = vxCreateScalarWithSize(context, VX_TYPE_INT32, &order, sizeof(order));

        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)s_order,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_PERMUTE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
