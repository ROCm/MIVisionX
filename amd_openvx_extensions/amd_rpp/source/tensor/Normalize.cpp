/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

struct NormalizeLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32u axis_mask;
    Rpp32f *pMean;
    Rpp32f *pStddev;
    Rpp32u computeMean;
    Rpp32u computeStddev;
    Rpp32f scale;
    Rpp32f shift;
    RpptGenericDescPtr pSrcGenericDesc;
    RpptGenericDescPtr pDstGenericDesc;
    Rpp32u *pSrcRoi;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

static vx_status VX_CALLBACK refreshNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num, NormalizeLocalData *data) {
    vx_status status = VX_SUCCESS;
    void *roi_tensor_ptr;
    int mean_stddev_array_size = 1;
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[3], &data->axis_mask));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }

    data->pSrcRoi = static_cast<unsigned *>(roi_tensor_ptr);
    int nDim = data->pSrcGenericDesc->numDims - 1;
    Rpp32u axis[nDim];
    for (unsigned i = 0; i < data->inputTensorDims[0]; i++) {
        unsigned index = i * nDim * 2;
        int totalElements = 1;
        for(Rpp32u j = 0; j < nDim; j++)
        {
            axis[j] = ((data->axis_mask & (int)(pow(2,j))) >= 1) ? 1 : 0;
            totalElements *= !axis[j] ? data->pSrcRoi[index + j + nDim] : 1;
        }
        mean_stddev_array_size = std::max(mean_stddev_array_size, totalElements);
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        if (!data->pMean) {
            hipError_t err = hipHostMalloc(&data->pMean, data->inputTensorDims[0] * mean_stddev_array_size * sizeof(float), hipHostMallocDefault);
            if (err != hipSuccess)
                return ERRMSG(VX_ERROR_NOT_ALLOCATED, "refresh: hipHostMalloc of size %ld failed \n", data->inputTensorDims[0] * mean_stddev_array_size * sizeof(float));
        }
        if (!data->pStddev) {
            hipError_t err = hipHostMalloc(&data->pStddev, data->inputTensorDims[0] * mean_stddev_array_size * sizeof(float), hipHostMallocDefault);
            if (err != hipSuccess)
                return ERRMSG(VX_ERROR_NOT_ALLOCATED, "refresh: hipHostMalloc of size %ld failed \n", data->inputTensorDims[0] * mean_stddev_array_size * sizeof(float));
        }
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->inputTensorDims[0] * mean_stddev_array_size, sizeof(float), data->pMean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->inputTensorDims[0] * mean_stddev_array_size, sizeof(float), data->pStddev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        if (!data->pMean) data->pMean = new float[data->inputTensorDims[0] * mean_stddev_array_size];
        if (!data->pStddev) data->pStddev = new float[data->inputTensorDims[0] * mean_stddev_array_size];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->inputTensorDims[0] * mean_stddev_array_size, sizeof(float), data->pMean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->inputTensorDims[0] * mean_stddev_array_size, sizeof(float), data->pStddev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    return status;
}

static vx_status VX_CALLBACK validateNormalize(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;

    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[11], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[12], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[13], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Normalize: tensor: #0 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    // Check for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Normalize: tensor: #2 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    NormalizeLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshNormalize(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        rpp_status = rppt_normalize_generic_gpu(data->pSrc, data->pSrcGenericDesc, data->pDst, data->pDstGenericDesc, data->axis_mask, data->pMean, data->pStddev, data->computeMean, data->computeStddev, data->scale, data->shift, data->pSrcRoi, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        rpp_status = rppt_normalize_generic_host(data->pSrc, data->pSrcGenericDesc, data->pDst, data->pDstGenericDesc, data->axis_mask, data->pMean, data->pStddev, data->computeMean, data->computeStddev, data->scale, data->shift, data->pSrcRoi, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    NormalizeLocalData *data = new NormalizeLocalData;
    memset(data, 0, sizeof(*data));

    vx_enum input_tensor_dtype, output_tensor_dtype;
    vx_int32 roi_type, input_layout, output_layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->computeMean, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->computeStddev, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &data->scale, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &data->shift, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[10], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[11], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[12], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[13], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = static_cast<RpptRoiType>(roi_type);
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    // Querying for input tensor
    data->pSrcGenericDesc = new RpptGenericDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcGenericDesc->numDims, sizeof(data->pSrcGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcGenericDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pSrcGenericDesc, data->inputLayout, data->inputTensorDims);

    // Querying for output tensor
    data->pDstGenericDesc = new RpptGenericDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstGenericDesc->numDims, sizeof(data->pDstGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstGenericDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pDstGenericDesc, data->outputLayout, data->outputTensorDims); 
    
    // refreshNormalize(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->inputTensorDims[0], data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeNormalize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    NormalizeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(data->pMean);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        err = hipHostFree(data->pStddev);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (data->pMean) delete[] data->pMean;
        if (data->pStddev) delete[] data->pStddev;
    }
    delete data->pSrcGenericDesc;
    delete data->pDstGenericDesc;
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status Normalize_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Normalize",
                                       VX_KERNEL_RPP_NORMALIZE,
                                       processNormalize,
                                       14,
                                       validateNormalize,
                                       initializeNormalize,
                                       uninitializeNormalize);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0,  VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1,  VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2,  VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3,  VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4,  VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5,  VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6,  VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7,  VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8,  VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9,  VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 13, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}