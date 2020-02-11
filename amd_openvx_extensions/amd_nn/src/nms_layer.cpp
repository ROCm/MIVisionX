#include "kernels.h"

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims_1[4], input_dims_2[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #1 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #2 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

    vx_int32 center_point_box;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &center_point_box, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(center_point_box != 0 && center_point_box != 1) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: NMS: #3 scalar type=%d ('center_point_box' must be between 0/1)\n", center_point_box);

    vx_size max_output_boxes_per_class_dims;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #4 input tensor data type=%d (must be int64)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, max_output_boxes_per_class_dims, sizeof(max_output_boxes_per_class_dims)));

    vx_size iou_threshold_dims;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #5 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, iou_threshold_dims, sizeof(iou_threshold_dims)));

    vx_size score_threshold_dims;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #6 input tensor data type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, score_threshold_dims, sizeof(score_threshold_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: NMS: #7 output tensor data type=%d (must be int64)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{

    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
)
{
	vx_enum input_type, output_type;
    vx_size num_dims;
    vx_size input_dims_1[4], input_dims_2[4],  output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

    vx_int32 center_point_box;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &center_point_box, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

#if ENABLE_DEBUG_PRINT_DIMS
    if (num_dims == 4) { 
       	std::cout << "nms input 1" << input_dims_1[3] << " " << input_dims_1[2] << " " << input_dims_1[1] << " " << input_dims_1[0] << " ";
        std::cout << "nms input 2" << input_dims_2[3] << " " << input_dims_2[2] << " " << input_dims_2[1] << " " << input_dims_2[0] << " ";
    	std::cout << "nms output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
    }/*
    else if(num_dims == 2) {
    	std::cout << "nms input " << " " << input_dims[1] << " " << input_dims[0] << " ";
    	std::cout << "nms output " << " " << output_dims[1] << " " << output_dims[0] << std::endl;
    }*/
    
#endif

	strcpy(opencl_kernel_function_name, "nms_layer");

    
    opencl_work_dim = 3;
    opencl_global_work[0] = input_dims[0];
    opencl_global_work[1] = input_dims[1];
    opencl_global_work[2] = input_dims[2] * input_dims[3];
   
    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_dims == 4) {
        char item[8192];
        sprintf(item,
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in_1, uint in_1_offset, uint4 in_1_stride, __global uchar * in_2, uint in_2_offset, uint4 in_2_stride, const int center_point_box, 
                                    __global uchar * max_output_boxes_per_class, uint max_output_boxes_per_class_offset, uint4 max_output_boxes_per_class_stride, 
                                    __global uchar * iou_threshold, uint iou_threshold_offset, uint4 iou_threshold_stride,
                                    __global uchar * score_threshold, uint score_threshold_offset, uint4 score_threshold_stride,
                                    __global uchar * out, uint out_offset, uint4 out_stride) \n"
                "{ \n"
                , opencl_kernel_function_name);
        opencl_kernel_code = item;
        sprintf(item,
        "    uint x = get_global_id(0) * %d;\n"
        "    uint y = get_global_id(1);\n"
        "    in += in_offset + y * in_stride.s1 + x * in_stride.s0;\n"
        "    out += out_offset + y * out_stride.s1 + x * out_stride.s0;\n"
        , input_element_count_multiple_of_4 ? 4 : 1);
    opencl_kernel_code += item;
		opencl_kernel_code +=
        "}\n";
	}
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishNMSLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.nms_layer", VX_KERNEL_CAST_LAYER_AMD, host_kernel, 7, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));    
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxNMSLayer(vx_graph graph, vx_tensor boxes, vx_tensor scores, vx_int32 center_point_box, vx_tensor max_output_boxes_per_class,
                                             vx_tensor iou_threshold, vx_tensor score_threshold, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
    	vx_scalar s_center_point_box = vxCreateScalarWithSize(context, VX_TYPE_INT32, &center_point_box, sizeof(center_point_box));
        vx_reference params[] = {
            (vx_reference)boxes,
            (vx_reference)scores,
            (vx_reference)s_center_point_box,
            (vx_reference)max_output_boxes_per_class,
            (vx_reference)iou_threshold,
            (vx_reference)score_threshold,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_NMS_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}