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
    //float *x_tensor = new float[count_input_dims_0];
    std::vector<float> x_tensor;

    status = vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input#1 (" << status << ")" << std::endl;
        return -1;
    }

    //memcpy(x_tensor, ptr, (count_input_dims_0*sizeof(float)));
    
    status = vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for input#5 (" << status << ")" << std::endl;
        return -1;
    }

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

    //create memory to store final output
    vx_enum type_output_0, type_output_1;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type_output_0, sizeof(type_output_0)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type_output_1, sizeof(type_output_1)));

    /*
    std::vector<std::vector<int64_t>> final_selected_indices;

    //get top_k scores with indices per batch per class. Common for both center point types.
    for (int b = 0; b < num_batches; ++b)
    {
        for (int c = 0; c < num_classes; ++c)
        {
            std::vector<std::pair<float, int>> score_index_vec;
            getMaxScoreIndex(scores[b][c], score_thresh[0], max_output_boxes_per_class[0], &score_index_vec);

            std::vector<int> selected_indices;
            selected_indices.clear();

            for(int i = 0; i < score_index_vec.size(); ++i)
            {
                const int idx = score_index_vec[i].second;
                bool keep = true;
                for(int k = 0; k < (int)selected_indices.size() && keep; ++k)
                {
                    const int prev_idx = selected_indices[k];
                    float overlap;
                    if (center_point_box == 0) /*indicates box data = [y1,x1,y2,x2] - mostly TF models */
/*                    {
                        overlap = computeOverlapCoordinates(boxes[idx], boxes[prev_idx]);
                    }
                    else if(center_point_box == 1) /*indicates box data = [x_center,y_center,width,height] - mostly PyTorch models*/
/*                    {
                        overlap = computeOverlapCenter(boxes[idx], boxes[prev_idx]);
                    }
                    if(overlap <= iou_thresh[0])
                        keep = true;
                    else
                        keep = false;
                    //keep = overlap <= iou_thresh[0];
                }
                if(keep)
                    selected_indices.push_back(idx);
            }
            if(max_output_boxes_per_class[0] < selected_indices.size())
                selected_indices.resize(max_output_boxes_per_class[0]);

            std::vector<int64_t> temp;

            for(int f = 0; f < selected_indices.size(); f++)
            {
                temp.clear();
                temp.push_back((int64_t)b);
                temp.push_back((int64_t)c);
                temp.push_back((int64_t)selected_indices[f]);
                final_selected_indices.push_back(temp);
            }
        }
    }    
    //for (int ab = 0; ab < final_selected_indices.size(); ab++)
    //    printf("final_selected_indices = %ld %ld %ld\n", final_selected_indices[ab][0],final_selected_indices[ab][1],final_selected_indices[ab][2]);
    int64_t *final_selected_indices_ptr = &final_selected_indices[0][0];

    //finding size of nms output and assigning stride
    output_dims[3] = 1; 
    output_dims[2] = 1;
    output_dims[1] = final_selected_indices.size(); //number of boxes found;
    output_dims[0] = 3; //3 values per index

    vx_size stride_output_final[4] = {sizeof(int64_t), output_dims[0]*sizeof(int64_t), output_dims[0]*output_dims[1]*sizeof(int64_t), output_dims[0]*output_dims[1]*output_dims[2]*sizeof(int64_t) };

    status =  vxCopyTensorPatch((vx_tensor)parameters[3], 4, nullptr, nullptr, stride_output_final, final_selected_indices_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxCopyTensorPatch() failed for output tensor"  << std::endl;
        return -1;
    }

    delete iou_thresh;
    delete score_thresh;
    delete max_output_boxes_per_class;
*/
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