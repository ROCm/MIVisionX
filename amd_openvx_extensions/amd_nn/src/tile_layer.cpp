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

    // the output num_dim should equal to (input_dim + input_dim2 - 1)
    // if (num_dims + num_dims2 - 1 != out_num_dims) {
    //     printf("validate: gather: The [rank(output tensor)] should equal to [rank(input tensor) + rank(indices tensor) - 1)]\n");
    //     printf("validate: gather: %d != %d + %d - 1\n", (int)out_num_dims, (int)num_dims, (int)num_dims2);
    //     return VX_ERROR_INVALID_DIMENSION;
    // }
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
    vx_size input_dims[4], rep_dims[4], output_dims[4];
    vx_size num_of_dims, rep_num_dims;
    vx_enum type;
    vx_map_id map_id;
    vx_size stride[4];
    vx_status status;
    std::vector<int> repeats;
    int * ptr;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &rep_num_dims, sizeof(rep_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, rep_dims, sizeof(rep_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    // copy repeat tensor
    status = vxMapTensorPatch((vx_tensor)parameters[1], rep_num_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for repeat tensor (" << status << ")" << std::endl;
        return -1;
    }

    for(int i=0; i<rep_dims[0]; i++) {
        repeats.push_back((int)ptr[i]);
    }
    vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);

    while(repeats.size() < 4)
        repeats.push_back(1);

    strcpy(opencl_kernel_function_name, "tile_layer");

    opencl_work_dim = 3;    
    opencl_global_work[0] = input_dims[0];
    opencl_global_work[1] = input_dims[1];
    opencl_global_work[2] = input_dims[2] * input_dims[3];

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
                "   float value;\n"
                "   uint offset;\n"
                "   value = *(__global float*)&in[in_offset + x*in_stride.s0 + indices*in_stride.s1 + c*in_stride.s2];\n"
                "   offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   out += offset;\n"
                "   for (int k=0; k<%d; k++) {\n"
                "       for (int j=0; j<%d; j++) {\n"
                "           for (int i=0; i<%d; i++) {\n"
                "               int stride = i*out_stride.s0 + j*out_stride.s1 + k*out_stride.s2;\n"
                "               *(__global float *)&out[stride] = value;\n"
                "           }\n"
                "       }\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, repeats[0], repeats[1], repeats[2]);
        }
        else {
            sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ind, uint ind_offset, uint4 ind_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint axis) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   int indices = *(__global int*)&ind[ind_offset + y*ind_stride.s0];\n"
                "   half value;\n"
                "   uint offset;\n"
                "   if (axis == 0) {\n"
                "       value = *(__global half*)&in[in_offset + x*in_stride.s0 + indices*in_stride.s1 + c*in_stride.s2];\n"
                "       offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   }\n"
                "   else if (axis == 1) {\n"
                "       value = *(__global half*)&in[in_offset + indices*in_stride.s0 + c*in_stride.s1];\n"
                "       offset = out_offset + y*out_stride.s0 + c*out_stride.s1;\n"
                "   }\n"
                "   else if (axis == 2) {\n"
                "       value = *(__global half*)&in[in_offset + c*in_stride.s0];\n"
                "       offset = out_offset + c*out_stride.s0;\n"
                "   }\n"
                "   out += offset;\n"
                "   *(__global half *)&out[0] = value;\n"
                "}\n", opencl_kernel_function_name);
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
vx_status publishTileLayer(vx_context context) {
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.tile_layer", VX_KERNEL_TILE_LAYER_AMD, host_kernel, 3, validateTileLayer, NULL, NULL);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

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

