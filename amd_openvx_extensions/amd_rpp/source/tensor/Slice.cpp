/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

struct SliceLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32s *pAnchor;
    Rpp32s *pShape;
    Rpp32f *pFillValues;
    Rpp32u policy;
    Rpp32u *pSrcDims;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    RpptGenericDescPtr pSrcGenericDesc;
    RpptGenericDescPtr pDstGenericDesc;
    Rpp32u *pSrcRoi3D;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

void updateDestinationRoi(SliceLocalData *data, unsigned *src_roi, unsigned *dst_roi, const vx_reference parameters[]) {
    // Query the Anchor / Shape Tensor Dims to determine the per-sample dimensionality
    RppSize_t num_dims;
    size_t anchorTensorDims[RPP_MAX_TENSOR_DIMS];
    vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims));
    vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, &anchorTensorDims, sizeof(vx_size) * num_dims);
    uint dims_per_sample = 1;
    for (uint n = 0; n < num_dims; n++)
        dims_per_sample *= anchorTensorDims[n];
    dims_per_sample /= anchorTensorDims[0];

    unsigned batchSize = data->inputTensorDims[0];
    bool enablePadding = (data->policy == 0);  // PAD = 0

    if (dims_per_sample == 1) {
        // 1D input: dst_roi layout is [begin_x, begin_y, end_x, end_y] per sample (2D ROI with y=0, h=1)
        for (unsigned i = 0, j = 0; i < batchSize; i++, j += 4) {
            dst_roi[j] = 0;      // begin_x
            dst_roi[j + 1] = 0;  // begin_y
            Rpp32s shape_val = data->pShape[i];
            if (!enablePadding) {
                // TRIMTOSHAPE: clamp shape to available data from anchor within the source extent
                Rpp32s anchor_val = data->pAnchor[i];
                Rpp32s src_length = static_cast<Rpp32s>(src_roi[i * 4 + 2]);  // src end_x (roiWidth)
                Rpp32s available = src_length - anchor_val;
                if (available < 0) available = 0;
                if (shape_val > available) shape_val = available;
            }
            dst_roi[j + 2] = static_cast<unsigned>(shape_val);  // end_x (width)
            dst_roi[j + 3] = 1;                                 // end_y (height)
        }
    } else {
        // nD input: dst_roi layout is [begin_0..begin_(D-1), end_0..end_(D-1)] per sample
        for (unsigned i = 0; i < batchSize; i++) {
            unsigned dst_offset = i * dims_per_sample * 2;
            unsigned src_offset = i * dims_per_sample * 2;
            unsigned shape_offset = i * dims_per_sample;
            // Set begin to zeros
            for (unsigned d = 0; d < dims_per_sample; d++) {
                dst_roi[dst_offset + d] = 0;
            }
            // Set end to the effective output shape
            for (unsigned d = 0; d < dims_per_sample; d++) {
                Rpp32s shape_val = data->pShape[shape_offset + d];
                if (!enablePadding) {
                    // TRIMTOSHAPE: clamp shape to available data from anchor within source ROI
                    Rpp32s anchor_val = data->pAnchor[shape_offset + d];
                    Rpp32s src_end = static_cast<Rpp32s>(src_roi[src_offset + dims_per_sample + d]);
                    Rpp32s available = src_end - anchor_val;
                    if (available < 0) available = 0;
                    if (shape_val > available) shape_val = available;
                }
                dst_roi[dst_offset + dims_per_sample + d] = static_cast<unsigned>(shape_val);
            }
        }
    }
}

static vx_status VX_CALLBACK refreshSlice(vx_node node, const vx_reference *parameters, SliceLocalData *data) {
    vx_status status = VX_SUCCESS;
    void *roi_tensor_ptr, *roi_tensor_ptr_dst;

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->pAnchor, sizeof(data->pAnchor)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HIP, &data->pShape, sizeof(data->pShape)));
        if (!data->pFillValues) {
            CHECK_HIP_RETURN_STATUS(hipHostMalloc(&data->pFillValues, data->inputTensorDims[0] * sizeof(Rpp32f), hipHostMallocDefault));
        }
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->inputTensorDims[0], sizeof(float), data->pFillValues, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->pAnchor, sizeof(data->pAnchor)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HOST, &data->pShape, sizeof(data->pShape)));
        if (!data->pFillValues) data->pFillValues = new Rpp32f[data->inputTensorDims[0]];
        STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[6], 0, data->inputTensorDims[0], sizeof(float), data->pFillValues, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    RppSize_t numDims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &numDims, sizeof(numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * numDims));

    // For Identifying if the input Tensor is 1D (excluding the Nth dimension) [ even if 2nd dim is 1 - The tensor is considered 1D ]
    if ((numDims == 3) && (data->inputTensorDims[2] == 1)) {
        RpptROI *src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr);
        for (unsigned i = 0, j = 0; i < data->inputTensorDims[0]; i++, j += 2) {
            data->pSrcDims[j] = src_roi[i].xywhROI.xy.x;
            data->pSrcDims[j + 1] = src_roi[i].xywhROI.roiWidth;
        }
        data->pSrcRoi3D = static_cast<unsigned *>(data->pSrcDims);
    } else {
        data->pSrcRoi3D = static_cast<unsigned *>(roi_tensor_ptr);
    }
    unsigned *src_roi = static_cast<unsigned *>(roi_tensor_ptr);
    unsigned *dst_roi = static_cast<unsigned *>(roi_tensor_ptr_dst);
    updateDestinationRoi(data, src_roi, dst_roi, parameters);
    return status;
}

static vx_status VX_CALLBACK validateSlice(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[10], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #10 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Slice: tensor: #0 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    // Check for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Slice: tensor: #2 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_status status = refreshSlice(node, parameters, data);
    if (status != VX_SUCCESS) return status;
#if RPP_AUDIO
#if ENABLE_HIP
    RppBackend backend = (data->deviceType == AGO_TARGET_AFFINITY_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;
#else
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        return VX_ERROR_NOT_IMPLEMENTED;
    RppBackend backend = RPP_HOST_BACKEND;
#endif
    RppStatus rpp_status = rppt_slice(data->pSrc, data->pSrcGenericDesc, data->pDst, data->pDstGenericDesc, data->pAnchor, data->pShape, data->pFillValues, data->policy, data->pSrcRoi3D, data->handle->rppHandle, backend);
    return (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data = new SliceLocalData;
    if (data) {
        memset(data, 0, sizeof(SliceLocalData));

        vx_enum input_tensor_dtype, output_tensor_dtype;
        vx_int32 roi_type, input_layout;
        STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[7], &data->policy));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[10], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->roiType = static_cast<RpptRoiType>(roi_type);
        data->inputLayout = static_cast<vxTensorLayout>(input_layout);

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
        fillGenericDescriptionPtrfromDims(data->pDstGenericDesc, data->inputLayout, data->outputTensorDims);

        data->pSrcDims = new uint[data->inputTensorDims[0] * 2];

        refreshSlice(node, parameters, data);
        STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->inputTensorDims[0], data->deviceType));
        STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
        return VX_SUCCESS;
    } else {
        return VX_FAILURE;
    }
}

static vx_status VX_CALLBACK uninitializeSlice(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SliceLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->pSrcDims) delete[] data->pSrcDims;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        if (data->pFillValues) CHECK_HIP_RETURN_STATUS(hipHostFree(data->pFillValues));
#endif
    } else {
        if (data->pFillValues) delete[] data->pFillValues;
    }
    if (data->pSrcDesc) delete data->pSrcDesc;
    if (data->pDstDesc) delete data->pDstDesc;
    if (data->pSrcGenericDesc) delete data->pSrcGenericDesc;
    if (data->pDstGenericDesc) delete data->pDstGenericDesc;
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,               // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity  // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
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

vx_status Slice_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Slice",
                                       VX_KERNEL_RPP_SLICE,
                                       processSlice,
                                       11,
                                       validateSlice,
                                       initializeSlice,
                                       uninitializeSlice);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
