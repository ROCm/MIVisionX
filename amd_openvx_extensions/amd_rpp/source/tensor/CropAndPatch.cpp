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

struct CropAndPatchLocalData {
    vxRppHandle *handle;
    vx_uint32 deviceType;
    RppPtr_t pSrc1;
    RppPtr_t pSrc2;
    RppPtr_t pDst;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    RpptROI *pDstRoi;
    RpptROI *pCropRoi;
    RpptROI *pPatchRoi;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

static vx_status VX_CALLBACK refreshCropAndPatch(vx_node node, const vx_reference *parameters, vx_uint32 num, CropAndPatchLocalData *data) {
    vx_status status = VX_SUCCESS;

    void *dst_roi_tensor_ptr = nullptr;
    void *crop_roi_tensor_ptr = nullptr;
    void *patch_roi_tensor_ptr = nullptr;

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc1, sizeof(data->pSrc1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pSrc2, sizeof(data->pSrc2)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &dst_roi_tensor_ptr, sizeof(dst_roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &crop_roi_tensor_ptr, sizeof(crop_roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HIP, &patch_roi_tensor_ptr, sizeof(patch_roi_tensor_ptr)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc1, sizeof(data->pSrc1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pSrc2, sizeof(data->pSrc2)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &dst_roi_tensor_ptr, sizeof(dst_roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &crop_roi_tensor_ptr, sizeof(crop_roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HOST, &patch_roi_tensor_ptr, sizeof(patch_roi_tensor_ptr)));
    }

    data->pDstRoi = reinterpret_cast<RpptROI *>(dst_roi_tensor_ptr);
    data->pCropRoi = reinterpret_cast<RpptROI *>(crop_roi_tensor_ptr);
    data->pPatchRoi = reinterpret_cast<RpptROI *>(patch_roi_tensor_ptr);

    // If input is NFHWC/NFCHW, expand per-sample ROIs over frames
    if (data->inputLayout == vxTensorLayout::VX_NFHWC || data->inputLayout == vxTensorLayout::VX_NFCHW) {
        unsigned num_of_frames = data->inputTensorDims[1]; // Num of frames 'F'
        for (int n = static_cast<int>(data->inputTensorDims[0]) - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for (unsigned f = 0; f < num_of_frames; f++) {
                data->pDstRoi[index + f].xywhROI = data->pDstRoi[n].xywhROI;
                data->pCropRoi[index + f].xywhROI = data->pCropRoi[n].xywhROI;
                data->pPatchRoi[index + f].xywhROI = data->pPatchRoi[n].xywhROI;
            }
        }
    }

    return status;
}

static vx_status VX_CALLBACK validateCropAndPatch(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;

    // inputLayout
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: CropAndPatch: Parameter: #6 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // outputLayout
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: CropAndPatch: Parameter: #7 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // roiType
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: CropAndPatch: Parameter: #8 type=%d (must be VX_TYPE_INT32)\n", scalar_type);

    // deviceType
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: CropAndPatch: Parameter: #9 type=%d (must be VX_TYPE_UINT32)\n", scalar_type);

    // Check for input tensor dims (src1)
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 4)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: CropAndPatch: tensor: #0 dimensions=%lu (must be >= 4)\n", num_tensor_dims);

    // Check and propagate output tensor meta (index 2)
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 4)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: CropAndPatch: tensor: #2 dimensions=%lu (must be >= 4)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));

    return status;
}

static vx_status VX_CALLBACK processCropAndPatch(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;

    CropAndPatchLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(refreshCropAndPatch(node, parameters, num, data));

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        rpp_status = rppt_crop_and_patch_gpu(data->pSrc1, data->pSrc2, data->pSrcDesc,
                                             data->pDst, data->pDstDesc,
                                             data->pDstRoi, data->pCropRoi, data->pPatchRoi,
                                             data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        rpp_status = rppt_crop_and_patch_host(data->pSrc1, data->pSrc2, data->pSrcDesc,
                                              data->pDst, data->pDstDesc,
                                              data->pDstRoi, data->pCropRoi, data->pPatchRoi,
                                              data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }

    return return_status;
}

static vx_status VX_CALLBACK initializeCropAndPatch(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    CropAndPatchLocalData *data = new CropAndPatchLocalData;
    memset(data, 0, sizeof(CropAndPatchLocalData));

    vx_enum input_tensor_dtype, output_tensor_dtype;
    vx_int32 roi_type, input_layout, output_layout;

    // Read scalar args
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    data->roiType = static_cast<RpptRoiType>(roi_type);
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    // Input tensor desc (derive from src1)
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pSrcDesc, data->inputLayout, data->inputTensorDims);

    // Output tensor desc
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pDstDesc, data->outputLayout, data->outputTensorDims);

    // Initial refresh and create RPP handle
    STATUS_ERROR_CHECK(refreshCropAndPatch(node, parameters, num, data));
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeCropAndPatch(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    CropAndPatchLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
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
                                                  vx_uint32 &supported_target_affinity) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status CropAndPatch_Register(vx_context context) {
    vx_status status = VX_SUCCESS;

    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, VX_KERNEL_RPP_CROP_AND_PATCH_NAME,
                                       VX_KERNEL_RPP_CROP_AND_PATCH,
                                       processCropAndPatch,
                                       10,
                                       validateCropAndPatch,
                                       initializeCropAndPatch,
                                       uninitializeCropAndPatch);
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

        // Parameters: src1 tensor, src2 tensor, dst tensor, dstRoi tensor, cropRoi tensor, patchRoi tensor, inputLayout, outputLayout, roiType, deviceType
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pSrc1
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pSrc2
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pDst
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pDstRoi
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pCropRoi
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // pPatchRoi
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // inputLayout
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // outputLayout
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // roiType
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // deviceType

        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
