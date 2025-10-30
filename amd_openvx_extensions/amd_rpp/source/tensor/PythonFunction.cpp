/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"

// Local copy of rocAL Python bridge C ABI (keep in sync with rocAL)
#ifndef ROCAL_PY_MAX_TENSOR_DIMS
#define ROCAL_PY_MAX_TENSOR_DIMS 5
#endif

typedef struct {
    size_t num_dims;                          /* e.g., 4 for [N,H,W,C] */
    size_t shape[ROCAL_PY_MAX_TENSOR_DIMS];   /* lengths per dimension */
    size_t strides[ROCAL_PY_MAX_TENSOR_DIMS]; /* strides in elements */
    vx_enum dtype;                            /* OpenVX scalar type enum */
    int layout;                               /* matches rocAL/vx tensor layout enums */
} RocalPyTensorDesc;

typedef struct {
    uint64_t function_id;        /* CPython id(function), provided by python front-end */
    RocalPyTensorDesc in_desc;   /* Input tensor description */
    RocalPyTensorDesc out_desc;  /* Output tensor description */
    uint32_t device_type;        /* AGO_TARGET_AFFINITY_{CPU,GPU}; currently CPU-only */
} RocalPyExecParams;

// Function pointer type for the rocAL Python bridge function.
typedef vx_status (*rocal_process_python_function_fn)(void *src_ptr, void *dst_ptr, const RocalPyExecParams *params);

// Map RpptDataType -> OpenVX type enum
vx_enum getVxDataType(RpptDataType dataType) {
    switch (dataType) {
        case RpptDataType::F32:
            return VX_TYPE_FLOAT32;
        case RpptDataType::F16:
            return VX_TYPE_FLOAT16;
        case RpptDataType::U8:
            return VX_TYPE_UINT8;
        case RpptDataType::I8:
            return VX_TYPE_INT8;
        case RpptDataType::I16:
            return VX_TYPE_INT16;
        default:
            throw std::runtime_error("Unsupported RpptDataType");
    }
}

size_t getItemSize(RpptDataType dataType) {
    switch (dataType) {
        case RpptDataType::F32:
            return sizeof(float);
        case RpptDataType::F16:
            return sizeof(vx_float16);
        case RpptDataType::U8:
            return sizeof(uint8_t);
        case RpptDataType::I8:
            return sizeof(int8_t);
        case RpptDataType::I16:
            return sizeof(int16_t);
        default:
            return 0;
    }
}

struct PythonFunctionLocalData {
    vx_uint32 deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    RpptGenericDescPtr pSrcGenericDesc;
    RpptGenericDescPtr pDstGenericDesc;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];

    // Bridge info
    uint64_t function_id;
    rocal_process_python_function_fn bridge_fn;
};

static vx_status VX_CALLBACK refreshPythonFunction(vx_node node, const vx_reference *parameters, vx_uint32 num, PythonFunctionLocalData *data) {
    // CPU-only initial implementation: GPU not supported yet
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));

    return status;
}

static vx_status VX_CALLBACK validatePythonFunction(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT64)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "PythonFunction validate: Parameter #2 (bridgeFnPtr) type=%d, must be UINT64\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT64)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "PythonFunction validate: Parameter #3 (functionId) type=%d, must be UINT64\n", scalar_type);
    for (int idx : {4, 5}) {
        STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[idx], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if (scalar_type != VX_TYPE_INT32)
            return ERRMSG(VX_ERROR_INVALID_TYPE, "PythonFunction validate: Parameter #%d must be INT32\n", idx);
    }
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "PythonFunction validate: Parameter #6 (deviceType) type=%d, must be UINT32\n", scalar_type);

    // Mirror output meta from provided output tensor (created by API with proper dims/dtype)
    size_t num_dims = 0;
    vx_uint8 fixed_point = 0;
    size_t dims[RPP_MAX_TENSOR_DIMS] = {0};
    vx_enum dtype = 0;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &dims, sizeof(dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &dtype, sizeof(dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point, sizeof(fixed_point)));

    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &dims, sizeof(dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &dtype, sizeof(dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point, sizeof(fixed_point)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processPythonFunction(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    PythonFunctionLocalData *data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(refreshPythonFunction(node, parameters, num, data));

    if (!data->pSrc || !data->pDst)
        return ERRMSG(VX_ERROR_INVALID_REFERENCE, "PythonFunction process: null tensor buffers\n");

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        return VX_ERROR_NOT_IMPLEMENTED;

    if (!data->bridge_fn) {
        vxAddLogEntry((vx_reference)node, VX_ERROR_NOT_IMPLEMENTED, "PythonFunction callback function is null.\n");
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    RocalPyExecParams params{};
    params.function_id = data->function_id;
    params.device_type = data->deviceType;

    // in_desc
    params.in_desc.num_dims = data->pSrcGenericDesc->numDims;
    params.in_desc.dtype = getVxDataType(data->pSrcGenericDesc->dataType);
    params.in_desc.layout = static_cast<int>(data->inputLayout);
    size_t in_itemsize = getItemSize(data->pSrcGenericDesc->dataType);
    if (in_itemsize == 0) return VX_ERROR_INVALID_TYPE;
    for (size_t i = 0; i < params.in_desc.num_dims; ++i) {
        params.in_desc.shape[i] = data->inputTensorDims[i];
        params.in_desc.strides[i] = static_cast<size_t>(data->pSrcGenericDesc->strides[i]) / in_itemsize;
    }

    // out_desc
    params.out_desc.num_dims = data->pDstGenericDesc->numDims;
    params.out_desc.dtype = getVxDataType(data->pDstGenericDesc->dataType);
    params.out_desc.layout = static_cast<int>(data->outputLayout);
    size_t out_itemsize = getItemSize(data->pDstGenericDesc->dataType);
    if (out_itemsize == 0) return VX_ERROR_INVALID_TYPE;
    for (size_t i = 0; i < params.out_desc.num_dims; ++i) {
        params.out_desc.shape[i] = data->outputTensorDims[i];
        params.out_desc.strides[i] = static_cast<size_t>(data->pDstGenericDesc->strides[i]) / out_itemsize;
    }

    vx_status st = data->bridge_fn(data->pSrc, data->pDst, &params);
    if (st != VX_SUCCESS) {
        vxAddLogEntry((vx_reference)node, st, "PythonFunction bridge returned error: %d\n", st);
        return st;
    }
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializePythonFunction(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    (void)num;
    auto *data = new PythonFunctionLocalData;
    vx_int32 input_layout = 0, output_layout = 0;
    vx_enum input_tensor_dtype = 0, output_tensor_dtype = 0;

    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    // Allocate descriptors (host)
    data->pSrcGenericDesc = new RpptGenericDesc;
    data->pDstGenericDesc = new RpptGenericDesc;

    // Input tensor info
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcGenericDesc->numDims, sizeof(data->pSrcGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcGenericDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pSrcGenericDesc, data->inputLayout, data->inputTensorDims);

    // Output tensor info
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstGenericDesc->numDims, sizeof(data->pDstGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstGenericDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pDstGenericDesc, data->outputLayout, data->outputTensorDims);

    // Get bridge function pointer from scalar
    uint64_t bridge_fn_ptr = 0;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[2], &bridge_fn_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->bridge_fn = reinterpret_cast<rocal_process_python_function_fn>(static_cast<uintptr_t>(bridge_fn_ptr));

    if (!data->bridge_fn) {
        vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_REFERENCE, "PythonFunction bridge function pointer is null.\n");
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Python function id to be forwarded to rocAL bridge
    vx_int64 function_id = 0;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &function_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->function_id = static_cast<uint64_t>(function_id);

    // Call refreshPythonFunction
    STATUS_ERROR_CHECK(refreshPythonFunction(node, parameters, num, data));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializePythonFunction(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    PythonFunctionLocalData *data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (!data) return VX_SUCCESS;

    if (data->pSrcGenericDesc) delete data->pSrcGenericDesc;
    if (data->pDstGenericDesc) delete data->pDstGenericDesc;

    delete data;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32 &supported_target_affinity) {
    AgoTargetAffinityInfo affinity;
    vxQueryContext(vxGetContext((vx_reference)graph), VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    supported_target_affinity = (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
                                    ? AGO_TARGET_AFFINITY_GPU
                                    : AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}

vx_status PythonFunction_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.PythonFunction",
                                       VX_KERNEL_PYTHONFUNCTION,
                                       processPythonFunction,
                                       7,
                                       validatePythonFunction,
                                       initializePythonFunction,
                                       uninitializePythonFunction);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    // GPU not implemented yet, but keep buffer access attribute for future
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#endif
    amd_kernel_query_target_support_f query_f = query_target_support;
    STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_f, sizeof(query_f)));

    // Parameters: pSrc, pDst, functionPtr, functionId, inputLayout, outputLayout, deviceType
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    STATUS_ERROR_CHECK(vxFinalizeKernel(kernel));
    if (status != VX_SUCCESS) {
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
