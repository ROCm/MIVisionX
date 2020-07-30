#include <vx_amd_nn.h>
#include <kernels.h>

static vx_status VX_CALLBACK validateSliceLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, starts_type, ends_type, axes_type, steps_type, out_type;
    vx_size num_dims, starts_dims, ends_dims, axes_dims, steps_dims, out_num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &starts_dims, sizeof(starts_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &starts_type, sizeof(starts_type)));
    if ((starts_type != VX_TYPE_INT32) && (starts_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, &ends_dims, sizeof(ends_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &ends_type, sizeof(ends_type)));
    if ((ends_type != VX_TYPE_INT32) && (ends_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, &axes_dims, sizeof(axes_dims)));   
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &axes_type, sizeof(axes_type)));
        if ((axes_type != VX_TYPE_INT32) && (axes_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
        if (starts_dims != axes_dims) {
            printf("validate:slice: The dimension length of starts, ends, axes, and steps must be the same.\n");
            return VX_ERROR_INVALID_DIMENSION;
        }
    }
    
    if (parameters[5]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, &steps_dims, sizeof(steps_dims)));   
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &steps_type, sizeof(steps_type)));
        if ((steps_type != VX_TYPE_INT32) && (steps_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
        if (starts_dims != steps_dims) {
            printf("validate:slice: The dimension length of starts, ends, axes, and steps must be the same.\n");
            return VX_ERROR_INVALID_DIMENSION;
        }
    }
    
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));
    return VX_SUCCESS;

}

static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,
    vx_uint32& supported_target_affinity
)
{
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processSliceLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    //get input params
    vx_status status;
    vx_enum input_type, type;
    vx_map_id map_id;
    vx_size stride[4];
    vx_size icount = 1, ocount = 1;
    std::vector<float> input, output;
    std::vector<int> starts, ends, axes, steps;
    float * fptr;
    int * ptr;

    vx_size input_num_dims, input_dims[4], param_dims, output_num_dims, output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &input_num_dims, sizeof(input_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &output_num_dims, sizeof(output_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    for(int i=0; i<input_num_dims; i++) {
        icount *= input_dims[i];
    }

    for(int i=0; i<output_num_dims; i++) {
        ocount *= output_dims[i];
    }

    // check the number of param dimensions and type
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &param_dims, sizeof(param_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    // create array of input/indices/output tensors
    vx_tensor input_tensor[param_dims], indices_tensor[param_dims], output_tensor[param_dims];
    
    // create context & graph
    vx_context context = vxCreateContext();
    status = vxGetStatus((vx_reference)context);
    if(status) {
        std::cerr << "ERROR: vxCreateContext() failed (" << status << ")" << std::endl;
        return -1;
    }

    vxLoadKernels(context, "vx_nn"); // load vx_nn kernels

    vx_graph graph = vxCreateGraph(context);
    status = vxGetStatus((vx_reference)graph);
    if(status) {
        std::cerr << "ERROR: vxCreateGraph() failed (" << status << ")" << std::endl;
        return -1;
    }

    // copy input tensor
    status = vxMapTensorPatch((vx_tensor)parameters[0], 2, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input tensor (" << status << ")" << std::endl;
        return -1;
    }
    for(int i=0; i<8; i++) {
        input.push_back((float)fptr[i]);
    }
    vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);

    // create tmp_input and copy input tensor
    vx_tensor tmpi_tensor = vxCreateTensor(context, input_num_dims, input_dims, input_type, 0);
    status = vxMapTensorPatch(tmpi_tensor, input_num_dims, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for tmpi tensor (" << status << ")" << std::endl;
        return -1;
    }
    for(int i=0; i<icount; i++) {
        fptr[i] = input[i];
    }
    vxUnmapTensorPatch(tmpi_tensor, map_id);

    input_tensor[0]  = tmpi_tensor;

    // copy starts index
    status = vxMapTensorPatch((vx_tensor)parameters[2], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
        return -1;
    }

    for(int i=0; i<param_dims; i++) {
        int num_element = input_dims[input_num_dims-1-i];
        int value = (int)ptr[i];
        if (value >= num_element) // If the value passed to starts is larger than the n(the number of elements in this dimension), it represents n
            starts.push_back(num_element);
        else if (value < 0) {
            starts.push_back(num_element+value); // If a negative value is passed for any of the starts indices, it represents number of elements before the end of that dimension
        }
        else
            starts.push_back(value);
    }

    vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
        return -1;
    }

    // copy ends index
    status = vxMapTensorPatch((vx_tensor)parameters[3], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for ends tensor (" << status << ")" << std::endl;
        return -1;
    }

    
    for(int i=0; i<param_dims; i++) {
        int num_element = input_dims[input_num_dims-1-i];
        int value = (int)ptr[i];
        if (value >= num_element) // If the value passed to ends is larger than the n(the number of elements in this dimension), it represents n
            ends.push_back(num_element);
        else if (value < 0) {
            ends.push_back(num_element+value); // If a negative value is passed for any of the ends indices, it represents number of elements before the end of that dimension
        }
        else
            ends.push_back(value);
    }


    vxUnmapTensorPatch((vx_tensor)parameters[3], map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for ends tensor (" << status << ")" << std::endl;
        return -1;
    }

    // copy axes index        
    if(parameters[4]) {
        status = vxMapTensorPatch((vx_tensor)parameters[4], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for axes tensor (" << status << ")" << std::endl;
            return -1;
        }

        for(int i=0; i<param_dims; i++) {
            axes.push_back((int)ptr[i]);
        }

        vxUnmapTensorPatch((vx_tensor)parameters[4], map_id);
    }
    else {
        for(int i=0; i<param_dims; i++)
            axes.push_back(i);
    }

    // copy steps index
    if(parameters[5]) {
        status = vxMapTensorPatch((vx_tensor)parameters[5], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
            return -1;
        }

        for(int i=0; i<param_dims; i++) {
            steps.push_back((int)ptr[i]);
        }

        vxUnmapTensorPatch((vx_tensor)parameters[5], map_id);
    }
    else {
        for(int i=0; i<param_dims; i++)
            steps.push_back(1);
    }

    // calculate and create indices tensor
    std::vector<std::vector<int>> indices(param_dims);
    for(int i=0; i<param_dims; i++) {
        int index = starts[i];
        while (index < ends[i]) {
            indices[i].push_back(index);
            index += steps[i];
        }

        vx_size dim[1] = {indices[i].size()};
        
        vx_tensor tmp_tensor = vxCreateTensor(context, 1, dim, type, 0);
        status = vxMapTensorPatch(tmp_tensor, 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for indices tensor(" << status << ")" << std::endl;
            return -1;
        }
        
        for(auto itr = indices[i].begin(); itr != indices[i].end(); itr++) {
            *ptr++ = *itr;
        }
        vxUnmapTensorPatch(tmp_tensor, map_id);
        indices_tensor[i] = tmp_tensor;
    }

    // calculate output dims and create output tensor
    vx_size tmp_num_dims = input_num_dims;
    vx_size *tmp_dims = input_dims;

    for(int i=0; i<(param_dims-1); i++) {
        //reverse input dims w,h,c,n- > n,c,h,w
        int start = 0, end = tmp_num_dims-1;
        while (start < end) {
            int temp = tmp_dims[start];
            tmp_dims[start] = tmp_dims[end];
            tmp_dims[end] = temp;
            start++;
            end--;
        }

        int out_dim_rank = tmp_num_dims;    
        vx_size out_dims[out_dim_rank];
        
        for (int j=0; j<out_dim_rank; j++) {
            if (j == axes[i]) {
                out_dims[j] = indices[i].size();
            }
            else {
                out_dims[j] = tmp_dims[j];
            }
        }

        //reverse output dims n,c,h,w -> w,h,c,n
        start = 0, end = out_dim_rank-1;
        while (start < end) {
            int temp = out_dims[start];
            out_dims[start] = out_dims[end];
            out_dims[end] = temp;
            start++;
            end--;
        }
        
        //vx_tensor tmp_tensor = vxCreateTensor(context, out_dim_rank, out_dims, input_type, 0);
        vx_tensor tmp_tensor = vxCreateVirtualTensor(graph, out_dim_rank, out_dims, input_type, 0);
        input_tensor[i+1] = tmp_tensor;
        output_tensor[i] = tmp_tensor;
        
        tmp_num_dims = out_dim_rank;
        tmp_dims = out_dims;
    }
    
    // create temp output tensor which will hold the final output result
    vx_tensor tmpo_tensor = vxCreateTensor(context, output_num_dims, output_dims, input_type, 0);
    output_tensor[param_dims-1] = tmpo_tensor;

    for (int i=0; i<param_dims; i++) {
        vxGatherLayer(graph, (vx_tensor)input_tensor[i], (vx_tensor)indices_tensor[i], (vx_tensor)output_tensor[i], axes[i]);
    }
    
    status = vxVerifyGraph(graph);
    
    status = vxProcessGraph(graph);
    if(status) {
        std::cerr << "ERROR: vxProcessGraph() failed (" << status << ")" << std::endl;
        return -1;
    }

    // copy final output
    status = vxMapTensorPatch(tmpo_tensor, output_num_dims, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for temp output tensor (" << status << ")" << std::endl;
        return -1;
    }
    
    for(int i=0; i<ocount; i++) {
        output.push_back((float)fptr[i]);
    }
    vxUnmapTensorPatch(tmpo_tensor, map_id);
 
    // write final output
    status = vxMapTensorPatch((vx_tensor)parameters[1], 2, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for input3 tensor (" << status << ")" << std::endl;
        return -1;
    }

    for(int i=0; i<ocount; i++) {
        fptr[i] = output[i];
    }
    vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status publishSliceLayer(vx_context context) 
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.slice_layer", VX_KERNEL_SLICE_LAYER_AMD, processSliceLayer, 6, validateSliceLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    
    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxSliceLayer(vx_graph graph, vx_tensor input, vx_tensor output, vx_tensor starts, vx_tensor ends, vx_tensor axes, vx_tensor steps) 
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) output,
            (vx_reference) starts,
            (vx_reference) ends,
            (vx_reference) axes,
            (vx_reference) steps
        };
        node = createNode(graph, VX_KERNEL_SLICE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}

