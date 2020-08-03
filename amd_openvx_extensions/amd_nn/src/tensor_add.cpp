/*
Copyright (c) 2017 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

struct TensorAddLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenTensorOp_t operation;
    float alpha1;
    float alpha2;
    float beta;
    miopenTensorDescriptor_t input1;
    cl_mem input1_mem;
    miopenTensorDescriptor_t input2;
    cl_mem input2_mem;
    miopenTensorDescriptor_t output;
    cl_mem output_mem;
};

static vx_status VX_CALLBACK validateTensorAddition(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: add: #2 type=%d (must be enum)\n", type);

    // check tensor dimensions
    vx_size num_dims;
    vx_size input1_dims[4],input2_dims[4] = { 1, 1, 0, 0 }, output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: add: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: add: #0 tensor type=%d (not float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: add: #1 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: add: #1 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &input2_dims[4-num_dims], num_dims * sizeof(vx_size)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: add: #3 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: add: #3 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if (output_dims[3] != input1_dims[3] || output_dims[2] != input1_dims[2] ||
        output_dims[1] != input1_dims[1] || output_dims[0] != input1_dims[0] ||
        output_dims[2] != input2_dims[2] || type != out_type ||
        !((             1 == input2_dims[3] &&              1 == input2_dims[1] &&              1 == input2_dims[0]) ||
          (output_dims[3] == input2_dims[3] && output_dims[1] == input2_dims[1] && output_dims[0] == input2_dims[0])))
    {
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: add: dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    // output tensor configuration
    out_type = type;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processTensorAddition(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Tensor_Add_Layer)
    TensorAddLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));

    //miopen elementwise addition call.
    ERROR_CHECK_MIOPEN_STATUS(miopenOpTensor(miopenHandle, data->operation, &data->alpha1, data->input1, data->input1_mem, &data->alpha2, data->input2, data->input2_mem, &data->beta, data->output, data->output_mem));


    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("tensor_add_%04d.bin", (vx_tensor)parameters[3]);
    #endif

PROFILER_STOP(VX_NN, Tensor_Add_Layer)

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeTensorAddition(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddLocalData * data = new TensorAddLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //initialize input and output tensor descriptors.
    vx_enum type;
    miopenDataType_t data_type;          // data_type for the kernel
    vx_size input1_dims[4], num_dims, input2_dims[4] = { 1, 1, 0, 0 }, output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &input2_dims[4-num_dims], num_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    data_type = (type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input1));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input2));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input1, data_type, input1_dims[3], input1_dims[2], input1_dims[1], input1_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input2, data_type, input2_dims[3], input2_dims[2], input2_dims[1], input2_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output, data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));

    //scaling parameters.
    data->alpha1 = 1;
    data->alpha2 = 1;
    data->beta = 0;
    data->operation = miopenTensorOpAdd;

    //input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "tensor_add input1 " << input1_dims[3] << " " << input1_dims[2] << " " << input1_dims[1] << " " << input1_dims[0] << " ";
    std::cout << "tensor_add input2 " << input2_dims[3] << " " << input2_dims[2] << " " << input2_dims[1] << " " << input2_dims[0] << " ";
    std::cout << "tensor_add output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorAddition(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input1));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input2));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishTensorAdd(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.openvx.tensor_add", VX_KERNEL_TENSOR_ADD, processTensorAddition, 4, validateTensorAddition, initializeTensorAddition, uninitializeTensorAddition);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorAddNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_policy = vxCreateScalarWithSize(context, VX_TYPE_ENUM, &policy, sizeof(policy));
        if (vxGetStatus((vx_reference)s_policy) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)input1,
                (vx_reference)input2,
                (vx_reference)s_policy,
                (vx_reference)output
            };
            node = createNode(graph, VX_KERNEL_TENSOR_ADD, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_policy);
        }
    }
    return node;
}

