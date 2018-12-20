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

void concat_codegen_batchsz1(std::string& opencl_code, vx_size work_items, vx_size output_dims[4], int num_inputs, vx_size ip_size_per_batch[8])
{
    vx_size ip_buffer_offset[8];   // index 0 is unused
    for(int i = 0; i < num_inputs; i++) {
        ip_buffer_offset[i] = 0;
        for(int j = 0; j < i; j++) {
            ip_buffer_offset[i] += ip_size_per_batch[j];
        }
    }

    char item[8192];
    sprintf(item,
        "{\n"
        "  size_t id = get_global_id(0);\n"
        "  if(id < %ld)\n"
        "  {\n"
        "    out += out_offset >> 2;\n\n"
        , work_items);
    opencl_code += item;

    sprintf(item,
        "    if(id < %ld)\n"   // ip_size_per_batch[0]
        "    {\n"
        "      in0 = in0 + (in0_offset >> 2);\n"
        "      out[id] = in0[id];\n"
        "    }\n"
        , ip_size_per_batch[0]);
    opencl_code += item;

    for(int i = 1; i < num_inputs; i++) {
        sprintf(item,
            "    else if((id >= %ld) && (id < %ld))\n"  // ip_buffer_offset[i], ip_buffer_offset[i] + ip_size_per_batch[i]
            "    {\n"
            "      in%d = in%d + (in%d_offset >> 2);\n"    // i, i, i
            "      out[id] = in%d[id - %ld];\n"    // i, ip_buffer_offset[i]
            "    }\n"
            , ip_buffer_offset[i], ip_buffer_offset[i] + ip_size_per_batch[i], i, i, i, i, ip_buffer_offset[i]);
        opencl_code += item;
    }
    opencl_code +=
            "  }\n"
            "}\n";

}

void concat_codegen_batchszN(std::string& opencl_code, vx_size work_items, vx_size output_dims[4], int num_inputs, vx_size ip_size_per_batch[8])
{
    vx_size ip_buffer_offset[8];   // index 0 is unused
    for(int i = 0; i < num_inputs; i++) {
        ip_buffer_offset[i] = 0;
        for(int j = 0; j < i; j++) {
            ip_buffer_offset[i] += ip_size_per_batch[j];
        }
    }

    char item[8192];
    sprintf(item,
        "{\n"
        "  size_t id = get_global_id(0);\n"
        "  if(id < %ld)\n"
        "  {\n"
        "    size_t batch_id = id / %ld;     // out_c*out_h*out_w\n"  // output_dims[2] * output_dims[1] * output_dims[0]
        "    size_t id_within_batch = id - batch_id * %ld;\n\n"    // output_dims[2] * output_dims[1] * output_dims[0]
        "    out += out_offset >> 2;\n\n"
        , work_items, output_dims[2] * output_dims[1] * output_dims[0], output_dims[2] * output_dims[1] * output_dims[0]);
    opencl_code += item;

    sprintf(item,
        "    if(id_within_batch < %ld)\n"   // ip_size_per_batch[0]
        "    {\n"
        "      in0 = in0 + (in0_offset >> 2) + (batch_id * %ld);\n"   // ip_size_per_batch[0]
        "      out[id] = in0[id_within_batch];\n"
        "    }\n"
        , ip_size_per_batch[0], ip_size_per_batch[0]);
    opencl_code += item;

    for(int i = 1; i < num_inputs; i++) {
        sprintf(item,
            "    else if((id_within_batch >= %ld) && (id_within_batch < %ld))\n"  // ip_buffer_offset[i], ip_buffer_offset[i] + ip_size_per_batch[i]
            "    {\n"
            "      in%d = in%d + (in%d_offset >> 2) + (batch_id * %ld);\n"    // i, i, i, ip_size_per_batch[i]
            "      out[id] = in%d[id_within_batch - %ld];\n"    // i, ip_buffer_offset[i]
            "    }\n"
            , ip_buffer_offset[i], ip_buffer_offset[i] + ip_size_per_batch[i], i, i, i, ip_size_per_batch[i], i, ip_buffer_offset[i]);
        opencl_code += item;
    }
    opencl_code +=
            "  }\n"
            "}\n";
}

static vx_status VX_CALLBACK validateConcatLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    //check tensor dims and type for input
    vx_enum type;
    vx_size num_dims;
    vx_size input1_dims[4], input2_dims[4], output_dims[4], num_channels = 0;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #1 num_dims=%ld (must be 4)\n", num_dims);
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: concat: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));
    num_channels = input1_dims[2];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #2 num_dims=%ld (must be 4)\n", num_dims);
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: concat: #2 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input2_dims, sizeof(input2_dims)));
    if (input1_dims[3] != input2_dims[3] || input1_dims[1] != input2_dims[1] || input1_dims[0] != input2_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #2 dims input1[%ld,%ld,%ld,%ld] != dims_input2[%ld,%ld,%ld,%ld]\n",
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3]);
    num_channels += input2_dims[2];
    int i = 3;
    while(parameters[i] && (i < 9)) {
        vx_size inputn_dims[4];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #%d num_dims=%ld (must be 4)\n", i, num_dims);
        if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: concat: #%d type=%d (must be float)\n", i, type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, inputn_dims, sizeof(inputn_dims)));
        if (input1_dims[3] != inputn_dims[3] || input1_dims[1] != inputn_dims[1] || input1_dims[0] != inputn_dims[0])
            return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #%d dims input1[%ld,%ld,%ld,%ld] != dims_input%d[%ld,%ld,%ld,%ld]\n", i,
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3], i,
                    inputn_dims[0], inputn_dims[1], inputn_dims[2], inputn_dims[3]);
        num_channels += inputn_dims[2];

        i++;
    }

    // Check tensor dims and type for output
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #0 num_dims=%ld (must be 4)\n", num_dims);
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: concat: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if((input1_dims[0] != output_dims[0]) || (input1_dims[1] != output_dims[1]) || (num_channels != output_dims[2]) || (input1_dims[3] != output_dims[3]))
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: concat: #all dims total input[%ld,%ld,%ld,%ld] != dims_output[%ld,%ld,%ld,%ld]\n",
                    input1_dims[0], input1_dims[1], num_channels, input1_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    //output tensor configuration.
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

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
    //get tensor dimensions
    vx_size output_dims[4];
    vx_size ip_size_per_batch[8], batch_size = 0, op_batch_stride = 0;
    int num_inputs = 0;

    int i = 1;
    while(parameters[i] && (i < 9)) {
        vx_size input_dims[4];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
        ip_size_per_batch[i - 1] = input_dims[2] * input_dims[1] * input_dims[0];
#if ENABLE_DEBUG_PRINT_DIMS
        std::cout << "concat input" << i << " " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << std::endl;
#endif

        i++;
    }
    num_inputs = i-1;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    batch_size = output_dims[3];
    op_batch_stride = output_dims[2] * output_dims[1] * output_dims[0];
#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "concat output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    strcpy(opencl_kernel_function_name, "concat_layer");
    vx_size work_items = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    opencl_work_dim = 1;
    opencl_local_work[0] = 128;
    opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    char item[8192];
    sprintf(item,
        "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
        "void %s(__global float * out, uint out_offset, uint4 out_stride" // opencl_kernel_function_name
        , (int)opencl_local_work[0], opencl_kernel_function_name);
    opencl_kernel_code = item;

    for(int i = 0; i < num_inputs; i++) {
        sprintf(item,
            ",\n"
            "                  __global float * in%d, uint in%d_offset, uint4 in%d_stride"  // i, i, i
            , i, i, i);
        opencl_kernel_code += item;
    }
    opencl_kernel_code += ")\n";

    if(output_dims[3] == 1) {
        concat_codegen_batchsz1(opencl_kernel_code, work_items, output_dims, num_inputs, ip_size_per_batch);
    }
    else {
        concat_codegen_batchszN(opencl_kernel_code, work_items, output_dims, num_inputs, ip_size_per_batch);
    }

    return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishConcatLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.concat_layer", VX_KERNEL_CONCAT_LAYER_AMD, host_kernel, 9, validateConcatLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));


    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxConcatLayer(vx_graph graph, vx_tensor output, vx_tensor input1, vx_tensor input2, vx_tensor input3, vx_tensor input4, vx_tensor input5, vx_tensor input6, vx_tensor input7, vx_tensor input8)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)output,
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)input3,
            (vx_reference)input4,
            (vx_reference)input5,
            (vx_reference)input6,
            (vx_reference)input7,
            (vx_reference)input8
        };
        node = createNode(graph, VX_KERNEL_CONCAT_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
