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

struct RicapLocalData {
    vxRppHandle* handle = nullptr;
    vx_uint32 deviceType = AGO_TARGET_AFFINITY_CPU;

    // IO buffers
    RppPtr_t pSrc = nullptr;
    RppPtr_t pDst = nullptr;

    // Param buffers
    Rpp32u* pPermutation = nullptr;     // size = batchSize * 4
    vx_size permutationLength = 0;      // number of u32 items copied in pPermutation
    RpptROI* pInputCropRoi = nullptr;   // array of 4 ROIs

    // Tensor descriptions
    RpptDescPtr pSrcDesc = nullptr;
    RpptDescPtr pDstDesc = nullptr;

    // Layouts and roi
    vxTensorLayout inputLayout = vxTensorLayout::VX_NHWC;
    vxTensorLayout outputLayout = vxTensorLayout::VX_NHWC;
    RpptRoiType roiType = RpptRoiType::XYWH;

    // Cached dims
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS] = {};
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS] = {};
};

static vx_status VX_CALLBACK refreshRicap(vx_node node, const vx_reference* parameters, vx_uint32 num, RicapLocalData* data)
{
    vx_status status = VX_SUCCESS;

    // Query IO buffers
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
        void* cropRoiTensorPtr = nullptr;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &cropRoiTensorPtr, sizeof(cropRoiTensorPtr)));
        data->pInputCropRoi = reinterpret_cast<RpptROI*>(cropRoiTensorPtr);
#endif
    } else {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        void* cropRoiTensorPtr = nullptr;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &cropRoiTensorPtr, sizeof(cropRoiTensorPtr)));
        data->pInputCropRoi = reinterpret_cast<RpptROI*>(cropRoiTensorPtr);
    }

    // Copy permutation array (vx_array of VX_TYPE_UINT32)
    vx_size numItems = 0;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_NUMITEMS, &numItems, sizeof(numItems)));
    if (numItems != data->permutationLength) {
        // (Re)allocate if length changed
        delete[] data->pPermutation;
        data->pPermutation = nullptr;
        if (numItems > 0) {
            data->pPermutation = new Rpp32u[numItems];
        }
        data->permutationLength = numItems;
    }
    if (data->pPermutation && data->permutationLength) {
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2],
                                            0,
                                            data->permutationLength,
                                            sizeof(Rpp32u),
                                            data->pPermutation,
                                            VX_READ_ONLY,
                                            VX_MEMORY_TYPE_HOST));
    }

    return status;
}

static vx_status VX_CALLBACK validateRicap(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;

    // inputLayout (idx: 4)
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Ricap: Parameter: #4 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // outputLayout (idx: 5)
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Ricap: Parameter: #5 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // roiType (idx: 6)
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Ricap: Parameter: #6 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // deviceType (idx: 7)
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Ricap: Parameter: #7 type=%d (must be VX_TYPE_UINT32)\n", scalar_type);

    // Validate output tensor meta (index 1)
    vx_uint8 tensor_fixed_point_position = 0;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS] = {0};
    size_t num_tensor_dims = 0;
    vx_enum tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 4)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Ricap: tensor: #1 dimensions=%lu (must be >= 4)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));

    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));

    return status;
}

static vx_status VX_CALLBACK processRicap(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    vx_status return_status = VX_SUCCESS;
    RicapLocalData* data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(refreshRicap(node, parameters, num, data));

    // The permutation tensor must be batchSize * 4
    const vx_uint32 batchSize = data->pSrcDesc ? data->pSrcDesc->n : 0;
    const vx_size expected = static_cast<vx_size>(batchSize) * 4;
    if (data->permutationLength != expected) {
        // Basic guard - not fatal error but return invalid if lengths do not match expectations
        return ERRMSG(VX_ERROR_INVALID_VALUE, "process: Ricap: permutation length=%lu (expected %u*4=%lu)\n",
                      data->permutationLength, batchSize, expected);
    }

    RppStatus rpp_status = RPP_SUCCESS;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return_status = VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        rpp_status = rppt_ricap_gpu(data->pSrc,
                                    data->pSrcDesc,
                                    data->pDst,
                                    data->pDstDesc,
                                    data->pPermutation,
                                    data->pInputCropRoi,
                                    data->roiType,
                                    data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else {
        rpp_status = rppt_ricap_host(data->pSrc,
                                     data->pSrcDesc,
                                     data->pDst,
                                     data->pDstDesc,
                                     data->pPermutation,
                                     data->pInputCropRoi,
                                     data->roiType,
                                     data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }

    return return_status;
}

static vx_status VX_CALLBACK initializeRicap(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    auto* data = new RicapLocalData();
    memset(data, 0, sizeof(RicapLocalData));

    // Read scalars
    vx_int32 input_layout, output_layout, roi_type;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);
    data->roiType = static_cast<RpptRoiType>(roi_type);

    // Build RpptDesc for src
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    vx_enum input_tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pSrcDesc, data->inputLayout, data->inputTensorDims);

    // Build RpptDesc for dst
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    vx_enum output_tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pDstDesc, data->outputLayout, data->outputTensorDims);

    // Initial refresh and RPP handle
    STATUS_ERROR_CHECK(refreshRicap(node, parameters, num, data));
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeRicap(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    RicapLocalData* data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (!data) return VX_SUCCESS;

    delete[] data->pPermutation;
    data->pPermutation = nullptr;

    delete data->pSrcDesc;
    delete data->pDstDesc;

    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete data;

    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32& supported_target_affinity)
{
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}

vx_status Ricap_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;

    // Add kernel with callbacks
    vx_kernel kernel = vxAddUserKernel(context,
                                       VX_KERNEL_RPP_RICAP_NAME,
                                       VX_KERNEL_RPP_RICAP,
                                       processRicap,
                                       8,
                                       validateRicap,
                                       initializeRicap,
                                       uninitializeRicap);
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
    STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

    // Parameters:
    // 0: pSrc tensor (input)
    // 1: pDst tensor (output)
    // 2: pPermutation (vx_array, uint32) (input)
    // 3: pInputCropRoi (vx_tensor, array of 4 RpptROI) (input)
    // 4: inputLayout (scalar int32) (input)
    // 5: outputLayout (scalar int32) (input)
    // 6: roiType (scalar int32) (input)
    // 7: deviceType (scalar uint32) (input)
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pSrc
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pDst
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // pPermutation
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pInputCropRoi (4 ROIs)
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // inputLayout
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // outputLayout
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // roiType
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // deviceType

    PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));

    return status;

exit:
    vxRemoveKernel(kernel);
    return VX_FAILURE;
}
