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

struct TensorMatrixMultiplyLocalData { 
	RPPCommonHandle handle; 
	rppHandle_t rppHandle; 
	Rpp32u device_type;
	Rpp8u *pSrc1;
    Rpp8u *pSrc2;
	Rpp8u *pDst;
	Rpp32u *tensorDimensionsValue1;
	Rpp32u *tensorDimensionsValue2;
#if ENABLE_OPENCL
	cl_mem cl_pSrc1;
    cl_mem cl_pSrc2;
	cl_mem cl_pDst;
#endif 
};

static vx_status VX_CALLBACK refreshTensorMatrixMultiply(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorMatrixMultiplyLocalData *data)
{
	vx_status status = VX_SUCCESS;
	size_t arr_size;
	vx_status copy_status;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
	data->pSrc1 = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
	copy_status = vxCopyArrayRange((vx_array)parameters[0], 0, arr_size, sizeof(Rpp8u),data->pSrc1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
	data->pSrc2 = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
	copy_status = vxCopyArrayRange((vx_array)parameters[1], 0, arr_size, sizeof(Rpp8u),data->pSrc2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->pDst = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->tensorDimensionsValue1 = (Rpp32u *)malloc(sizeof(Rpp32u) * arr_size);
	copy_status = vxCopyArrayRange((vx_array)parameters[3], 0, arr_size, sizeof(Rpp32u),data->tensorDimensionsValue1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[4], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->tensorDimensionsValue2 = (Rpp32u *)malloc(sizeof(Rpp32u) * arr_size);
	copy_status = vxCopyArrayRange((vx_array)parameters[4], 0, arr_size, sizeof(Rpp32u),data->tensorDimensionsValue2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	if(data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
    cl_context theContext;
    cl_command_queue theQueue;
    theQueue = data->handle.cmdq;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
    cl_int err;
    size_t bytes;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    bytes = arr_size * sizeof(Rpp8u);
    err = clEnqueueWriteBuffer(theQueue, data->cl_pSrc1, CL_TRUE, 0,
                                   bytes, data->pSrc1, 0, NULL, NULL);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    bytes = arr_size * sizeof(Rpp8u);
    err = clEnqueueWriteBuffer(theQueue, data->cl_pSrc2, CL_TRUE, 0,
                                   bytes, data->pSrc2, 0, NULL, NULL);
#endif
	}
	
	return status; 
}

static vx_status VX_CALLBACK validateTensorMatrixMultiply(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	vx_enum scalar_type;
    size_t arr_size;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &scalar_type, sizeof(scalar_type)));
	STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE, &scalar_type, sizeof(scalar_type)));
	return status;
}

static vx_status VX_CALLBACK processTensorMatrixMultiply(vx_node node, const vx_reference * parameters, vx_uint32 num) 
{
	RppStatus rpp_status = RPP_SUCCESS;
	vx_status return_status = VX_SUCCESS;
	TensorMatrixMultiplyLocalData * data = NULL;
	STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    size_t arr_size;
	if(data->device_type == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
		cl_command_queue handle = data->handle.cmdq;
		refreshTensorMatrixMultiply(node, parameters, num, data);
		rpp_status = rppi_tensor_matrix_multiply_u8_gpu((void *)data->cl_pSrc1, (void *)data->cl_pSrc2, (void *)data->cl_pDst,  data->tensorDimensionsValue1, data->tensorDimensionsValue2,data->rppHandle);
        cl_command_queue theQueue;
        theQueue = data->handle.cmdq;
        cl_int err;
        STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
        size_t bytes = arr_size * sizeof(Rpp8u);
        clEnqueueReadBuffer(theQueue, data->cl_pDst, CL_TRUE, 0, bytes, data->pDst, 0, NULL, NULL );
		return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
	}
	if(data->device_type == AGO_TARGET_AFFINITY_CPU) {
		refreshTensorMatrixMultiply(node, parameters, num, data);
		rpp_status = rppi_tensor_matrix_multiply_u8_host(data->pSrc1, data->pSrc2, data->pDst,  data->tensorDimensionsValue1, data->tensorDimensionsValue2,data->rppHandle);
		return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
	}
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    vx_status copy_status = vxCopyArrayRange((vx_array)parameters[2], 0, arr_size, sizeof(Rpp8u),data->pDst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    return return_status;
}

static vx_status VX_CALLBACK initializeTensorMatrixMultiply(vx_node node, const vx_reference *parameters, vx_uint32 num) 
{
	TensorMatrixMultiplyLocalData * data = new TensorMatrixMultiplyLocalData;
	memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
	STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &data->handle.cmdq, sizeof(data->handle.cmdq)));
    cl_context theContext;               // theContext
    cl_command_queue theQueue;           // command theQueue
    theQueue = data->handle.cmdq;
    clGetCommandQueueInfo(  theQueue,
                            CL_QUEUE_CONTEXT,
                            sizeof(cl_context), &theContext, NULL);
	size_t arr_size;
    size_t bytes;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    bytes = arr_size * sizeof(Rpp8u);
    data->cl_pSrc1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    bytes = arr_size * sizeof(Rpp8u);
    data->cl_pSrc2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    bytes = arr_size * sizeof(Rpp8u);
    data->cl_pDst = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
#endif
	STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    refreshTensorMatrixMultiply(node, parameters, num, data);
#if ENABLE_OPENCL
	if(data->device_type == AGO_TARGET_AFFINITY_GPU)
		rppCreateWithStream(&data->rppHandle, data->handle.cmdq);
#endif
	if(data->device_type == AGO_TARGET_AFFINITY_CPU)
	    rppCreateWithBatchSize(&data->rppHandle, 1);

    
	STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
	return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorMatrixMultiply(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	TensorMatrixMultiplyLocalData * data; 
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

vx_status TensorMatrixMultiply_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
	// Add kernel to the context with callbacks
	vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorMatrixMultiply",
		VX_KERNEL_RPP_TENSORMATRIXMULTIPLY,
		processTensorMatrixMultiply,
		6,
		validateTensorMatrixMultiply,
		initializeTensorMatrixMultiply,
		uninitializeTensorMatrixMultiply);
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
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT , VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
	}
	if (status != VX_SUCCESS)
	{
	exit:	vxRemoveKernel(kernel);	return VX_FAILURE; 
 	}
	return status;
}
