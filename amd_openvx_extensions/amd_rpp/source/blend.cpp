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


#include <kernels_rpp.h>
#include <vx_ext_rpp.h>
#include <stdio.h>
#include <iostream>
#include "internal_rpp.h"

#include "internal_publishKernels.h"

#include </opt/rocm/rpp/include/rpp.h>
#include </opt/rocm/rpp/include/rppdefs.h>
#include </opt/rocm/rpp/include/rppi.h>

struct BlendLocalData {

#if ENABLE_OPENCL
    RPPCommonHandle handle;
#endif
    RppiSize dimensions;
    RppPtr_t pSrc1;
    RppPtr_t pSrc2;
    RppPtr_t pDst;
    Rpp32f alpha;
    Rpp32u device_type;

#if ENABLE_OPENCL
    cl_mem cl_pSrc1;
    cl_mem cl_pSrc2;
    cl_mem cl_pDst;
#endif

#if ENABLE_HIP
    void *hip_pSrc2;
    void *hip_pSrc1;
    void *hip_pDst;
#endif

};

static vx_status VX_CALLBACK validateBlend(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {

    vx_status status = VX_SUCCESS;
    vx_parameter param[2];
    vx_image image[2];
    vx_df_image df_image[2] = {VX_DF_IMAGE_VIRT};
    vx_enum type[2];
    for (int i = 0; i < 2; i++)
    {
        param[i] = vxGetParameterByIndex(node, i);
        STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar) parameters[3], VX_SCALAR_TYPE, &type[i], sizeof(type[i])));
        if (type[i] != VX_TYPE_FLOAT32)
            return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", type[i]);

        STATUS_ERROR_CHECK(vxQueryParameter(param[i], VX_PARAMETER_ATTRIBUTE_REF, &image[i], sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image[i], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image[i], sizeof(df_image[i])));
        if (df_image[i] != VX_DF_IMAGE_U8 && df_image[i] != VX_DF_IMAGE_RGB)
            status = VX_ERROR_INVALID_VALUE;
        vxReleaseImage(&image[i]);
    }
    // Assuming both images will be of same type.
    if(df_image[1] != df_image[0])
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Input images to blend node must be same color format: # format %d vs %d\n", df_image[0], df_image[1]);
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image[0], sizeof(df_image[0])));
    vx_image output; vx_uint32 width = 0, height = 0; //vx_df_image format = VX_DF_IMAGE_VIRT;
    vx_parameter output_param = vxGetParameterByIndex(node, 2);

    STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
    //STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

    vxReleaseImage(&output);
    vxReleaseParameter(&output_param);

    return status;

}

static vx_status VX_CALLBACK processBlend(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    BlendLocalData * data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    vx_df_image df_image2 = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image2, sizeof(df_image2)));

    if(data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
    cl_command_queue handle = data->handle.cmdq;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[3], &data->alpha));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc1, sizeof(data->cl_pSrc1)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc2, sizeof(data->cl_pSrc2)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[2], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pDst, sizeof(data->cl_pDst)));
    if (df_image == VX_DF_IMAGE_U8 ){
        rppi_blend_u8_pln1_gpu((void *)data->cl_pSrc1, (void *)data->cl_pSrc2, data->dimensions, (void*)data->cl_pDst, data->alpha, (void *)handle);
    }
    else if(df_image == VX_DF_IMAGE_RGB) {
        rppi_blend_u8_pkd3_gpu((void *)data->cl_pSrc1, (void *)data->cl_pSrc2, data->dimensions, (void*)data->cl_pDst, data->alpha, (void *)handle);
    }
    return VX_SUCCESS;

#endif
    } else if(data->device_type == AGO_TARGET_AFFINITY_CPU) {
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[3], &data->alpha));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc1, sizeof(vx_uint8)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc2, sizeof(vx_uint8)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[2], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pDst, sizeof(vx_uint8)));
    if (df_image == VX_DF_IMAGE_U8 ){
        rppi_blend_u8_pln1_host((void *)data->pSrc1, (void *)data->pSrc2, data->dimensions, (void*)data->pDst, data->alpha);
    }
    else if(df_image == VX_DF_IMAGE_RGB) {
        rppi_blend_u8_pkd3_host((void *)data->pSrc1, (void *)data->pSrc2, data->dimensions, (void*)data->pDst, data->alpha);
    }
    return VX_SUCCESS;
    }

  return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeBlend(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    BlendLocalData * data = new BlendLocalData;
    memset(data, 0, sizeof(*data));

#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#endif
    // Assuming both images are of same height and width
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &data->alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc1, sizeof(data->cl_pSrc1)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc2, sizeof(data->cl_pSrc2)));
#else
    //STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, data->pSrc, sizeof(data->pSrc)));
#endif

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeBlend(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}

vx_status Blend_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
// add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Blend",
            VX_KERNEL_RPP_BLEND,
            processBlend,
            5,
            validateBlend,
            initializeBlend,
            uninitializeBlend);

    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,&affinity, sizeof(affinity));
    #if ENABLE_OPENCL
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if(affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
    #else
    vx_bool enableBufferAccess = vx_false_e;
    #endif
    if (kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:   vxRemoveKernel(kernel); return VX_FAILURE;
    }

    return status;
}
