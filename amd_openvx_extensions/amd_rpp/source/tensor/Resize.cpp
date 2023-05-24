/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

struct ResizeLocalData {
    vxRppHandle * handle;
    Rpp32u device_type;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 *resize_h;
    vx_uint32 *resize_w;
    RpptDescPtr src_desc_ptr;
    RpptDesc srcDesc;
    RpptDesc dstDesc;
    RpptDescPtr dst_desc_ptr;
    void *roi_tensor_ptr;
    RpptROI * roi_ptr;
    RpptRoiType roiType;
    Rpp32s input_layout;
    Rpp32s output_layout;
    size_t in_tensor_dims[VX_TENSOR_DIMS];
    size_t out_tensor_dims[VX_TENSOR_DIMS];
    vx_enum in_tensor_type;
    vx_enum out_tensor_type;
    RpptImagePatch *dstImgSize;
    Rpp32s interpolation_type; 
};

static vx_status VX_CALLBACK refreshResize(vx_node node, const vx_reference *parameters, vx_uint32 num, ResizeLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[3], 0, data->src_desc_ptr->n, sizeof(vx_uint32), data->resize_w, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, data->src_desc_ptr->n, sizeof(vx_uint32), data->resize_h, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    for (int i = 0; i < data->src_desc_ptr->n; i++) {
        data->dstImgSize[i].width = data->resize_w[i];
        data->dstImgSize[i].height = data->resize_h[i];
    }
    if (data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->roi_tensor_ptr, sizeof(data->roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else if (data->device_type == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->roi_tensor_ptr, sizeof(data->roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    data->roi_ptr = (RpptROI *)data->roi_tensor_ptr;
    if(data->input_layout == 2 || data->input_layout == 3) {
        unsigned num_of_frames = data->in_tensor_dims[1]; // Num of frames 'F'
        for(int n = data->src_desc_ptr->n - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for(int f = 0; f < num_of_frames; f++) {
                data->resize_w[index + f] = data->resize_w[n];
                data->resize_h[index + f] = data->resize_h[n];
                data->roi_ptr[index + f].xywhROI = data->roi_ptr[n].xywhROI;
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK validateResize(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be a boolean size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #8 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #9 type=%d (must be size)\n", scalar_type);

    // Check for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Resize: tensor: #0 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);

    // Check for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[VX_TENSOR_DIMS];
    vx_enum tensor_type;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if(num_tensor_dims < 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: Resize: tensor: #2 dimensions=%lu (must be greater than or equal to 4)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ResizeLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        refreshResize(node, parameters, num, data);
        rpp_status = rppt_resize_gpu((void *)data->pSrc, data->src_desc_ptr, (void *)data->pDst, data->dst_desc_ptr, data->dstImgSize, (RpptInterpolationType)data->interpolation_type, data->roi_ptr, data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU) {
        refreshResize(node, parameters, num, data);
        rpp_status = rppt_resize_host(data->pSrc, data->src_desc_ptr, data->pDst, data->dst_desc_ptr, data->dstImgSize, (RpptInterpolationType)data->interpolation_type, data->roi_ptr, data->roiType, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResizeLocalData *data = new ResizeLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        THROW("initialize : Resize, OpenCL backend is not supported")
#endif
    int roiType;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->interpolation_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &roiType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[9], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = (roiType == 0) ? RpptRoiType::XYWH : RpptRoiType::LTRB;

    // Querying for input tensor
    data->src_desc_ptr = &data->srcDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->src_desc_ptr->numDims, sizeof(data->src_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->in_tensor_dims, sizeof(vx_size) * data->src_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->in_tensor_type, sizeof(data->in_tensor_type)));
    data->src_desc_ptr->dataType = getRpptDataType(data->in_tensor_type);
    data->src_desc_ptr->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->src_desc_ptr, data->input_layout, data->in_tensor_dims);

    // Querying for output tensor
    data->dst_desc_ptr = &data->dstDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->dst_desc_ptr->numDims, sizeof(data->dst_desc_ptr->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->out_tensor_dims, sizeof(vx_size) * data->dst_desc_ptr->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2],VX_TENSOR_DATA_TYPE, &data->out_tensor_type, sizeof(data->out_tensor_type)));
    data->dst_desc_ptr->dataType = getRpptDataType(data->out_tensor_type);
    data->dst_desc_ptr->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->dst_desc_ptr, data->output_layout, data->out_tensor_dims);

#if ENABLE_HIP
    hipHostMalloc(&data->dstImgSize, data->src_desc_ptr->n * sizeof(RpptImagePatch));
#else
    data->dstImgSize = (RpptImagePatch *)calloc(data->src_desc_ptr->n, sizeof(RpptImagePatch));
#endif    
    data->resize_w = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);
    data->resize_h = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->src_desc_ptr->n);
    refreshResize(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->src_desc_ptr->n, data->device_type));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeResize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResizeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->device_type));
    free(data->resize_w);
    free(data->resize_h);
#if ENABLE_HIP
    hipHostFree(data->dstImgSize);
#else
    free(data->dstImgSize);
#endif
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

vx_status Resize_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Resize",
                                       VX_KERNEL_RPP_RESIZE,
                                       processResize,
                                       10,
                                       validateResize,
                                       initializeResize,
                                       uninitializeResize);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
