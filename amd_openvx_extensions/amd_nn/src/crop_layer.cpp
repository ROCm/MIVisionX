#include <kernels.h>

static vx_status VX_CALLBACK validateCropLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, out_type;
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type2, sizeof(type2)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type2 != VX_TYPE_FLOAT32) && (type2 != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, ref_dims, sizeof(ref_dims)));


    vx_int32 axis;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (axis > 4 || axis < 0) {
        printf("initialize: crop: Wrong axis value\n");
        return VX_ERROR_INVALID_PARAMETERS;
    } 
    
    vx_size offset[4];
    for (int i = 0; i < 4; i++) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4+i], &offset[i], VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if ((int)(offset[i]) < 0) {
            printf("initialize: crop: Offset should be larger than 0\n");
            return VX_ERROR_INVALID_PARAMETERS;
        } 
        else if (((int)(offset[i] + ref_dims[i]) > (int)input_dims[i])) {
            printf("initialize: crop: Offset out of bound\n");
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }

    for (int i = 0; i < 4; i++) {
        if (output_dims[i] != (i < axis) ? input_dims[i] : ref_dims[i]) {
            printf("initialize: crop: Wrong output dimension. Please check the output dimension\n");
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));
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
    vx_size input_dims[4], ref_dims[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, ref_dims, sizeof(ref_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    strcpy(opencl_kernel_function_name, "crop_layer");

    vx_uint32 input_dim_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];

    opencl_work_dim = 3;
    opencl_global_work[0] = ref_dims[0];
    opencl_global_work[1] = ref_dims[1];
    opencl_global_work[2] = ref_dims[2];

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    vx_size offset[4];
    for (int i = 0; i < 4; i++) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4+i], &offset[i], VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    if (num_of_dims == 4) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32) {
        sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ref, uint ref_offset, uint4 ref_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint offset1, uint offset2, uint offset3, uint offset4) \n"
                "{ \n"
                "   for (uint n = %d; n < %d; n++) {\n"
                "       uint x = get_global_id(0) + %d;\n"
                "       uint y = get_global_id(1) + %d;\n"
                "       uint c = get_global_id(2) + %d;\n"
                "       float value = *(__global float*)&in[in_offset + x*in_stride.s0 + y*in_stride.s1 + c*in_stride.s2 + n*in_stride.s3];\n"
                "       out += out_offset + get_global_id(0)*out_stride.s0 + get_global_id(1)*out_stride.s1 + get_global_id(2)*out_stride.s2 + (n-%d)*out_stride.s3;\n"
                "       *(__global float *)&out[0] = value;\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, (int)offset[3], (int)ref_dims[3], (int)offset[0], (int)offset[1], (int)offset[2], (int)offset[3]);
        }
        else {
            sprintf(item,
               "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ref, uint ref_offset, uint4 ref_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint offset1, uint offset2, uint offset3, uint offset4) \n"
                "{ \n"
                "   for (uint n = %d; n < %d; n++) {\n"
                "       uint x = get_global_id(0) + %d;\n"
                "       uint y = get_global_id(1) + %d;\n"
                "       uint c = get_global_id(2) + %d;\n"
                "       half value = *(__global half*)&in[x*in_stride.s0 + y*in_stride.s1 + c*in_stride.s2 + n*in_stride.s3];\n"
                "       out += get_global_id(0)*out_stride.s0 + get_global_id(1)*out_stride.s1 + get_global_id(2)*out_stride.s2 + (n-%d)*out_stride.s3;\n"
                "       *(__global half *)&out[0] = value;\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, (int)offset[3], (int)ref_dims[3], (int)offset[0], (int)offset[1], (int)offset[2], (int)offset[3]);
        }
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}


static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    return VX_ERROR_NOT_IMPLEMENTED;
}

vx_status publishCropLayer(vx_context context) {
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.crop_layer", VX_KERNEL_CROP_LAYER_AMD, host_kernel, 6, validateCropLayer, NULL, NULL);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxCropLayer(vx_graph graph, vx_tensor input, vx_tensor output, vx_scalar x_coord, vx_scalar y_coord, vx_scalar width, vx_scalar height) 
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) output,
            (vx_reference) x_coord,
            (vx_reference) y_coord,
            (vx_reference) width,
            (vx_reference) height,
        };
        node = createNode(graph, VX_KERNEL_CROP_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
