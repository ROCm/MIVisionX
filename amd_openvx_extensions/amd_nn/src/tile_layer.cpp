#include <kernels.h>

static vx_status VX_CALLBACK validateTileLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, type2, out_type;
    vx_size num_dims, num_dims2, out_num_dims;
    vx_size input_dims[4], input_dims2[4], output_dims[4];
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims2, sizeof(num_dims2)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type2, sizeof(type2)));
    if ((type2 != VX_TYPE_INT32) && (type2 != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims2, sizeof(input_dims2)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if ((num_dims != out_num_dims) || (num_dims != input_dims2[0])) {
        printf("validate: tile: Ranks of input, repeat, and output tensors should be equal\n");
        return VX_ERROR_INVALID_DIMENSION;        
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(num_dims)));
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
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    for (int i=0; i<4; i++) {  //convert input tensor to 4-D
        if(i >= num_of_dims) {
            input_dims[i] = 1;
            output_dims[i] = 1;
        }
    }

    strcpy(opencl_kernel_function_name, "tile_layer");

    opencl_work_dim = 3;    
    opencl_global_work[0] = output_dims[0];
    opencl_global_work[1] = output_dims[1];
    opencl_global_work[2] = output_dims[2] * output_dims[3];

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32) {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * rep, uint rep_offset, uint4 rep_stride, __global uchar * out, uint out_offset, uint4 out_stride) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   uint nx = x %% %d;\n"
                "   uint ny = y %% %d;\n"
                "   uint nc = c %% %d;\n"
                "   float value = *(__global float *)&in[in_offset + nx*in_stride.s0 + ny*in_stride.s1 + nc*in_stride.s2];\n"
                "   uint offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   out += offset;\n"
                "   *(__global float *)&out[0] = value;\n"
                "}\n", opencl_kernel_function_name, (int)input_dims[0], (int)input_dims[1], (int)input_dims[2]);
        }
        else {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * rep, uint rep_offset, uint4 rep_stride, __global uchar * out, uint out_offset, uint4 out_stride) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   uint nx = x %% %d;\n"
                "   uint ny = y %% %d;\n"
                "   uint nc = c %% %d;\n"
                "   half value = *(__global half *)&in[in_offset + nx*in_stride.s0 + ny*in_stride.s1 + nc*in_stride.s2];\n"
                "   uint offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   out += offset;\n"
                "   *(__global half *)&out[0] = value;\n"
                "}\n", opencl_kernel_function_name, (int)input_dims[0], (int)input_dims[1], (int)input_dims[2]);
        }
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) 
{
#if ENABLE_HIP
    //get tensor dimensions
    vx_size input_dims[4] = {1, 1, 1, 1};
    vx_size output_dims[4] = {1, 1, 1, 1};
    vx_size num_of_dims;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    dim3 globalThreads = dim3(1);
    globalThreads.x = output_dims[0];
    globalThreads.y = output_dims[1];
    globalThreads.z = output_dims[2] * output_dims[3];

    AgoData *input  = reinterpret_cast<AgoData *>(parameters[0]);
    AgoData *repeat = reinterpret_cast<AgoData *>(parameters[1]);
    AgoData *output = reinterpret_cast<AgoData *>(parameters[2]);

    uint4 input_stride = make_uint4((uint)input->u.tensor.stride[0], (uint)input->u.tensor.stride[1],
                                    (uint)input->u.tensor.stride[2], (uint)input->u.tensor.stride[3]);

    uint4 input_dimensions = make_uint4((uint)input_dims[0], (uint)input_dims[1], (uint)input_dims[2], (uint)input_dims[3]);

    uint4 repeat_stride = make_uint4((uint)repeat->u.tensor.stride[0], (uint)repeat->u.tensor.stride[1],
                                     (uint)repeat->u.tensor.stride[2], (uint)repeat->u.tensor.stride[3]);

    uint4 output_stride = make_uint4((uint)output->u.tensor.stride[0], (uint)output->u.tensor.stride[1],
                                     (uint)output->u.tensor.stride[2], (uint)output->u.tensor.stride[3]);

    if (HipExec_Tile_layer(node->hip_stream0, globalThreads, dim3(1), type, input->hip_memory, input->u.tensor.offset, input_stride,
        input_dimensions, repeat->hip_memory, repeat->u.tensor.offset, repeat_stride, output->hip_memory, output->u.tensor.offset, output_stride)) {
        return VX_FAILURE;
    }

    return VX_SUCCESS;

#elif ENABLE_OPENCL
    return VX_ERROR_NOT_IMPLEMENTED;
#endif
}

//! \brief The kernel publisher.
vx_status publishTileLayer(vx_context context) {
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.tile_layer", VX_KERNEL_TILE_LAYER_AMD, host_kernel, 3, validateTileLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxTileLayer(vx_graph graph, vx_tensor input, vx_tensor repeats, vx_tensor output) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) repeats,
            (vx_reference) output,
        };
        node = createNode(graph, VX_KERNEL_TILE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }

    return node;
}

