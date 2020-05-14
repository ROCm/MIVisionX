/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

struct PhaseLocalData { 
	RPPCommonHandle handle;
	rppHandle_t rppHandle; 
	RppiSize srcDimensions; 
	Rpp32u device_type;
	RppPtr_t pSrc1;
	RppPtr_t pSrc2;
	RppPtr_t pDst;
#if ENABLE_OPENCL
	cl_mem cl_pSrc1;
	cl_mem cl_pSrc2;
	cl_mem cl_pDst;
#endif 
};

static vx_status VX_CALLBACK refreshPhase(vx_node node, const vx_reference *parameters, vx_uint32 num, PhaseLocalData *data)
{
	vx_status status = VX_SUCCESS;
 	STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->srcDimensions.height, sizeof(data->srcDimensions.height)));
	STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->srcDimensions.width, sizeof(data->srcDimensions.width)));
	if(data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc1, sizeof(data->cl_pSrc1)));
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc2, sizeof(data->cl_pSrc2)));
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[2], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pDst, sizeof(data->cl_pDst)));
#endif
	}
	if(data->device_type == AGO_TARGET_AFFINITY_CPU) {
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc1, sizeof(vx_uint8)));
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc2, sizeof(vx_uint8)));
		STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[2], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pDst, sizeof(vx_uint8)));
	}
	return status; 
}

static vx_status VX_CALLBACK validatePhase(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	vx_enum scalar_type;
	STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
 	if(scalar_type != VX_TYPE_UINT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);
	// Check for input parameters 
	vx_parameter input_param; 
	vx_image input; 
	vx_df_image df_image;
	input_param = vxGetParameterByIndex(node,0);
	STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_image)));
	STATUS_ERROR_CHECK(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image))); 
	if(df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB) 
	{
		return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: Phase: image: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&df_image);
	}

	input_param = vxGetParameterByIndex(node,1);
	STATUS_ERROR_CHECK(vxQueryParameter(input_param, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_image)));
	STATUS_ERROR_CHECK(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image))); 
	if(df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB) 
	{
		return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: Phase: image: #1 format=%4.4s (must be RGB2 or U008)\n", (char *)&df_image);
	}

	// Check for output parameters 
	vx_image output; 
	vx_df_image format; 
	vx_parameter output_param; 
	vx_uint32  height, width; 
	output_param = vxGetParameterByIndex(node,2);
	STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image))); 
	STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width))); 
	STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height))); 
	STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
	vxReleaseImage(&input);
	vxReleaseImage(&output);
	vxReleaseParameter(&output_param);
	vxReleaseParameter(&input_param);
	return status;
}

static vx_status VX_CALLBACK processPhase(vx_node node, const vx_reference * parameters, vx_uint32 num) 
{ 
	RppStatus status = RPP_SUCCESS;
	PhaseLocalData * data = NULL;
	STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
	vx_df_image df_image = VX_DF_IMAGE_VIRT;
	STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
	if(data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
		cl_command_queue handle = data->handle.cmdq;
		refreshPhase(node, parameters, num, data);
		if (df_image == VX_DF_IMAGE_U8 ){ 
 			status = rppi_phase_u8_pln1_gpu((void *)data->cl_pSrc1,(void *)data->cl_pSrc2,data->srcDimensions,(void *)data->cl_pDst,data->rppHandle);
		}
		else if(df_image == VX_DF_IMAGE_RGB) {
			status = rppi_phase_u8_pkd3_gpu((void *)data->cl_pSrc1,(void *)data->cl_pSrc2,data->srcDimensions,(void *)data->cl_pDst,data->rppHandle);
		}
		return status;
#endif
	}
	if(data->device_type == AGO_TARGET_AFFINITY_CPU) {
		refreshPhase(node, parameters, num, data);
		if (df_image == VX_DF_IMAGE_U8 ){
			status = rppi_phase_u8_pln1_host(data->pSrc1,data->pSrc2,data->srcDimensions,data->pDst,data->rppHandle);
		}
		else if(df_image == VX_DF_IMAGE_RGB) {
			status = rppi_phase_u8_pkd3_host(data->pSrc1,data->pSrc2,data->srcDimensions,data->pDst,data->rppHandle);
		}
		return status;
	}
}

static vx_status VX_CALLBACK initializePhase(vx_node node, const vx_reference *parameters, vx_uint32 num) 
{
	PhaseLocalData * data = new PhaseLocalData;
	memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
	STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
#endif
	STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	refreshPhase(node, parameters, num, data);
#if ENABLE_OPENCL
	if(data->device_type == AGO_TARGET_AFFINITY_GPU)
		rppCreateWithStream(&data->rppHandle, data->handle.cmdq);
#endif
	if(data->device_type == AGO_TARGET_AFFINITY_CPU)
	rppCreateWithBatchSize(&data->rppHandle, 1);
	STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
	return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializePhase(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	PhaseLocalData * data; 
	STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL
	if(data->device_type == AGO_TARGET_AFFINITY_GPU)
		rppDestroyGPU(data->rppHandle);
#endif
	if(data->device_type == AGO_TARGET_AFFINITY_CPU)
		rppDestroyHost(data->rppHandle);
	delete(data);
	return VX_SUCCESS; 
}

vx_status Phase_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
	// Add kernel to the context with callbacks
	vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Phase",
		VX_KERNEL_RPP_PHASE,
		processPhase,
		4,
		validatePhase,
		initializePhase,
		uninitializePhase);
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
		PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
	}
	if (status != VX_SUCCESS)
	{
	exit:	vxRemoveKernel(kernel);	return VX_FAILURE; 
 	}
	return status;
}
