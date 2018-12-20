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

////////////////////////////////////////////////////////////////////////////
// utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        return NULL;
    }
    vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
    if(vxGetStatus((vx_reference)kernel) == VX_SUCCESS) {
        node = vxCreateGenericNode(graph, kernel);
        if (node) {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++) {
                if (params[p]) {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS) {
                        char kernelName[VX_MAX_KERNEL_NAME];
                        vxQueryKernel(kernel, VX_KERNEL_NAME, kernelName, VX_MAX_KERNEL_NAME);
                        vxAddLogEntry((vx_reference)graph, status, "createNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to create node with kernel enum %d\n", kernelEnum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to retrieve kernel enum %d\n", kernelEnum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

vx_reference getNodeParameterByIndex(vx_node node, vx_uint32 index)
{
    vx_reference ref = NULL;
    vx_parameter param = vxGetParameterByIndex(node, index);
    if(vxGetStatus((vx_reference)param) == VX_SUCCESS) {
        vxQueryParameter(param, VX_PARAMETER_REF, &ref, sizeof(ref));
        vxReleaseParameter(&param);
    }
    return ref;
}

int getEnvironmentVariable(const char * name)
{
#if _WIN32
    char text[64] = { 0 };
    if (GetEnvironmentVariableA(name, text, (DWORD)sizeof(text)) > 0) {
        return atoi(text);
    }
#else
    const char * text = getenv(name);
    if (text) {
        return atoi(text);
    }
#endif
    return -1;
}

vx_status createGraphHandle(vx_node node, NeuralNetworkCommonHandle ** pHandle)
{
    NeuralNetworkCommonHandle * handle = NULL;
    ERROR_CHECK_STATUS(vxGetModuleHandle(node, OPENVX_KHR_NN, (void **)&handle));
    if(handle) {
        handle->count++;
    }
    else {
        handle = new NeuralNetworkCommonHandle;
        memset(handle, 0, sizeof(*handle));
        const char * searchEnvName = "NN_MIOPEN_SEARCH";
        int isEnvSet = getEnvironmentVariable(searchEnvName);
        if (isEnvSet > 0)
            handle->exhaustiveSearch = true;

        handle->count = 1;
        ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &handle->cmdq, sizeof(handle->cmdq)));
        
        //create miopen_handle from cmdq
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateWithStream(&handle->miopen_handle, handle->cmdq));
        ERROR_CHECK_STATUS(vxSetModuleHandle(node, OPENVX_KHR_NN, handle));
    }
    *pHandle = handle;
    return VX_SUCCESS;
}

vx_status releaseGraphHandle(vx_node node, NeuralNetworkCommonHandle * handle)
{
    handle->count--;
    if(handle->count == 0) {
        //TBD: release miopen_handle
        delete handle;
        ERROR_CHECK_STATUS(vxSetModuleHandle(node, OPENVX_KHR_NN, NULL));
    }
    return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    // set command-queue properties to be CL_QUEUE_PROFILING_ENABLE needed by MIOpen (default)
    const char * searchEnvName = "NN_MIOPEN_CL_QUEUE_PROPERTIES";
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
#if _WIN32
    char text[64] = { 0 };
    if (GetEnvironmentVariableA(searchEnvName, text, (DWORD)sizeof(text)) > 0) {
        properties = atoi(text);
    }
#else
    const char * text = getenv(searchEnvName);
    if (text) {
        properties = atoi(text);
    }
#endif
    ERROR_CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_CL_QUEUE_PROPERTIES, &properties, sizeof(properties)));

    // register kernels
    ERROR_CHECK_STATUS(publishConvolutionLayer(context));
    ERROR_CHECK_STATUS(publishFullyConnectedLayer(context));
    ERROR_CHECK_STATUS(publishPoolingLayer(context));
    ERROR_CHECK_STATUS(publishSoftmaxLayer(context));
    ERROR_CHECK_STATUS(publishNormalizationLayer(context));
    ERROR_CHECK_STATUS(publishActivationLayer(context));
    ERROR_CHECK_STATUS(publishROIPoolingLayer(context));
    ERROR_CHECK_STATUS(publishDeconvolutionLayer(context));
    ERROR_CHECK_STATUS(publishBatchNormalizationLayer(context));
    ERROR_CHECK_STATUS(publishArgmaxLayer(context));
    ERROR_CHECK_STATUS(publishConcatLayer(context));
    ERROR_CHECK_STATUS(publishSliceLayer(context));
    ERROR_CHECK_STATUS(publishImageToTensorConvert(context));
    ERROR_CHECK_STATUS(publishTensorToImageConvert(context));
    ERROR_CHECK_STATUS(publishTensorAdd(context));
    ERROR_CHECK_STATUS(publishTensorSubtraction(context));
    ERROR_CHECK_STATUS(publishTensorMultiply(context));
    ERROR_CHECK_STATUS(publishScaleLayer(context));
    ERROR_CHECK_STATUS(publishUpsampleNearest(context));
    ERROR_CHECK_STATUS(publishTensorTableLookup(context));
    ERROR_CHECK_STATUS(publishTensorMatrixMultiply(context));
    ERROR_CHECK_STATUS(publishReshapeLayer(context));

    // register drama rules
    AgoNodeMergeRule softmax_rule = {
        {
            { VX_KERNEL_SOFTMAX_LAYER, { 1, 2 | AGO_MERGE_RULE_SOLITARY_FLAG } },
            { VX_KERNEL_ARGMAX_LAYER_AMD, { 2 | AGO_MERGE_RULE_SOLITARY_FLAG, 3 } },
        },
        {
            { VX_KERNEL_ARGMAX_LAYER_AMD, { 1, 3 } },
        }
    };
    ERROR_CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE, &softmax_rule, sizeof(softmax_rule)));

    return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for unpublishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxUnpublishKernels(vx_context context)
{
    return VX_SUCCESS;
}
