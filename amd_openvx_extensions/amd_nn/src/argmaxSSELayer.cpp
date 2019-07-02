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

#include <smmintrin.h>
#include "kernels.h"

static vx_status VX_CALLBACK validateArgmaxSSELayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_enum scalar_type;
    vx_int32 axis;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(axis < 0 || axis > 3) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: argmax_SSE: #2 scalar type=%d (axis must be greater than 0 and lesser than 3)\n", axis);

    vx_int32 keepdims;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &keepdims, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(keepdims != 0 && keepdims != 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: argmax_SSE: #3 scalar type=%d (keepdims must be 0/1)\n", keepdims);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
}

static vx_status VX_CALLBACK processArgmaxSSELayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_size input_dims[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum input_type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));


    vx_int32 axis, keepdims;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &keepdims, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if(input_type == VX_TYPE_FLOAT32)
    {
        
    }
    else if(input_type == VX_TYPE_FLOAT16)
    {
        
    }


}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}


//! \brief The kernel publisher.
vx_status publishArgmaxSSELayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.argmax_sse_layer", VX_KERNEL_ARGMAX_SSE_LAYER_AMD, processArgmaxSSELayer, 4, validateArgmaxSSELayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxArgmaxSSELayer(vx_graph graph, vx_tensor input, vx_scalar axis, vx_scalar keepdims, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_axis = vxCreateScalarWithSize(context, VX_TYPE_INT32, &axis, sizeof(axis));
        vx_scalar s_keepdims = vxCreateScalarWithSize(context, VX_TYPE_INT32, &keepdims, sizeof(keepdims));
        
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)s_axis,
            (vx_reference)s_keepdims,
            (vx_reference)output
        };
        node = createNode(graph, VX_KERNEL_ARGMAX_SSE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}