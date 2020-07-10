#include "kernels.h"


static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims_1[4], input_dims_2[4], output_dims_1[4], output_dims_2[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #1 input tensor data type=%d (must be float32) (other types not supported yet)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #2 input tensor data type=%d (must be int64)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

    vx_int32 axis;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(axis < -4 || axis > 3) 
        return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #3 scalar type=%d ('axis' must be between [-r,r-1])(r = rank of input)\n", axis);

    vx_int32 largest;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &largest, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(largest != 0 && largest != 1) 
        return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #4 scalar type=%d ('largest' must be either 0/1)\n", largest);

    vx_int32 sorted;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &sorted, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(sorted != 0 && sorted != 1) 
        return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #5 scalar type=%d ('sorted' must be either 0/1)\n", sorted);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #6 output tensor data 'values' type=%d (must be float32)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims_1, sizeof(output_dims_1)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DIMS, output_dims_1, sizeof(output_dims_1)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 /*&& num_dims != 3*/) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #7 output tensor data 'indices' type=%d (must be int64)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processTopKLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    //get scalar attributes
    vx_int32 axis, largest, sorted;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &largest, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &sorted, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    //get tensor dimensions
    vx_size input_dims_0[4], input_dims_1[4], output_dims_0[4], output_dims_1[4];
    vx_size num_of_dims;

    vx_enum usage = VX_READ_ONLY;
    vx_status status;
    vx_map_id map_id;
    vx_size stride[4]

    //query and copy inputs
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));

    //accessing the tensor from last dimension
    if(axis < 0)
        axis = num_of_dims + axis;
    //opposite of onnx..check if needed???
    vx_int32 openvx_axis = 3 - axis;

    float * ptr;
    vx_size count_input_dims_0 = input_dims_0[0]*input_dims_0[1]*input_dims_0[2]*input_dims_0[3];
    float *x_tensor = new float[count_input_dims_0]; 

    status = vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }

    memcpy(x_tensor, ptr, (count_input_dims_0*sizeof(float)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);

    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
        return -1;
    }

    for (int i = 0; i < count_input_dims_0; i++)
        printf("%f\n", x_tensor[i]);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));

    
    int64_t * ptr;
    vx_size count_input_dims_1 = input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*input_dims_1[3];
    int64_t *k_tensor = new int64_t[count_input_dims_1];

    status = vxMapTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#2 (" << status << ")" << std::endl;
        return -1;
    }

    memcpy(k_tensor, ptr, (count_input_dims_1*sizeof(int64_t)));

    status = vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
        return -1;
    }

    for (int i = 0; i < count_input_dims_1; i++)
        printf("%f\n", k_tensor[i]);

    vx_int32 axis;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    printf("axis = %d\n", axis);

    vx_int32 largest;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &largest, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    printf("largest = %d\n", largest);

    vx_int32 sorted;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &sorted, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    printf("sorted = %d\n", sorted);

    //output vectors
    std::vector<float> values;
    std::vector<int64_t> indices;

    //vector to sort indices
    std::vector<size_t> idx(input_dims_0[axis]);
    
    for(int i = 0; i < count_input_dims_0; i += input_dims_0[axis])
    {
        std::iota(idx.begin(), idx.end(), 0); 
        if (largest)
        {   if (sorted)
            {   
                std::sort(idx.begin(), idx.end(), [&x_tensor](const size_t &a, const size_t &b)
                                               { return x_tensor[a] > x_tensor[b];});
            }
            else
            {
                std::stable_sort(idx.begin(), idx.end(), [&x_tensor](const size_t &a, const size_t &b)
                                               { return x_tensor[a] > x_tensor[b];});
            }

        }
        else
        {
            if(sorted)
            {
                std::sort(idx.begin(), idx.end(), [&x_tensor](const size_t &a, const size_t &b)
                                               { return x_tensor[a] < x_tensor[b];}); 
            }
            else
            {
                std::stable_sort(idx.begin(), idx.end(), [&x_tensor](const size_t &a, const size_t &b)
                                               { return x_tensor[a] < x_tensor[b];}); 
            }
        }
        
        //keep only top k elements
        int keep_elements = (input_dims_0[axis] < k) ? input_dims_0[axis]:k;
        for (int j = 0; j < keep_elements; j++)
        {
            values.push_back(x_tensor[idx[j]]);
            indices.push_back(idx[j]);
        }

        //printf("after sorting = %f %f %f %f\n", x_tensor[idx[0]], x_tensor[idx[1]], x_tensor[idx[2]], x_tensor[idx[3]]);
        //printf("after sorting = %lu %lu %lu %lu\n", idx[0], idx[1], idx[2], idx[3]);
        x_tensor += input_dims_0[axis];
    }

    for (int i = 0; i < values.size(); i++)
        printf("idx = %lu value = %f \n", indices[i], values[i]);

    float* values_ptr = &values[0];
    int64_t indices_ptr = &indices[0];

    //finding size of topk output and assigning stride
    output_dims_0[3] = 1; 
    output_dims_0[2] = 1;
    output_dims_0[1] = 1; 
    output_dims_0[0] = values.size(); //total count of values. Acts as a 1D array

    output_dims_1[3] = 1; 
    output_dims_1[2] = 1;
    output_dims_1[1] = 1; 
    output_dims_1[0] = indices.size(); //total count of indices. Acts as a 1D array    

    vx_size stride_output_0[4] = {sizeof(float), output_dims_0[0]*sizeof(float), output_dims_0[0]*output_dims_0[1]*sizeof(float), output_dims_0[0]*output_dims_0[1]*output_dims_0[2]*sizeof(float) };
    vx_size stride_output_1[4] = {sizeof(int64_t), output_dims_1[0]*sizeof(int64_t), output_dims_1[0]*output_dims_1[1]*sizeof(int64_t), output_dims_1[0]*output_dims_1[1]*output_dims_1[2]*sizeof(int64_t) };
    
    status =  vxCopyTensorPatch((vx_tensor)parameters[5], 4, nullptr, nullptr, stride_output_0, values_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output tensor"  << std::endl;
        return -1;
    }

    status =  vxCopyTensorPatch((vx_tensor)parameters[6], 4, nullptr, nullptr, stride_output_1, indices_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output tensor"  << std::endl;
        return -1;
    }

    return VX_SUCCESS;

}


//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{

    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status publishTopKLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.topk_layer", VX_KERNEL_NMS_LAYER_AMD, processTopKLayer, 7, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));    
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTopKLayer(vx_graph graph, vx_tensor x_tensor, vx_tensor k_tensor, vx_int32 axis, vx_int32 largest, vx_int32 sorted, 
                                            vx_tensor values, vx_tensor indices)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_axis = vxCreateScalarWithSize(context, VX_TYPE_INT32, &axis, sizeof(axis));
        vx_scalar s_largest = vxCreateScalarWithSize(context, VX_TYPE_INT32, &largest, sizeof(largest));
        vx_scalar s_sorted = vxCreateScalarWithSize(context, VX_TYPE_INT32, &sorted, sizeof(sorted));
        vx_reference params[] = {
            (vx_reference)x_tensor,
            (vx_reference)k_tensor,
            (vx_reference)s_axis,
            (vx_reference)s_largest,
            (vx_reference)s_sorted,
            (vx_reference)values,
            (vx_reference)indices,
        };
        node = createNode(graph, VX_KERNEL_TOPK_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
} 