#include <vx_amd_nn.h>
#include <kernels.h>

static vx_status VX_CALLBACK validateSliceLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, out_type;
    vx_size num_dims, out_num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    printf("validate slice %d\n", num);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    // ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[num], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(out_num_dims)));
    // ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[num], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    // if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    // ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[num], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // vx_uint32 starts, ends, axes, steps;
    // ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &starts, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    // ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &ends, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    // ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &axes, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    // if (starts < 0) {
    //     return ERRMSG(VX_ERROR_INVALID_PARAMETERS, "validate: slice: starts = %d (should be greater than 0)\n", starts);
    // }
    // else if (starts < 0) {
    //     return ERRMSG(VX_ERROR_INVALID_PARAMETERS, "validate: slice: starts = %d (should be greater than 0)\n", starts);
    // }
    // else if (end >= starts) {
    //     return ERRMSG(VX_ERROR_INVALID_PARAMETERS, "validate: slice: starts = %d ends = %d (starts value is greater than ends value)\n", starts, ends);
    // }
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));
    printf("validate slice done\n");
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
    std::vector<int> starts, ends, axes, steps;
    int * ptr;

    vx_size input_num_dims, input_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &input_num_dims, sizeof(input_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));

    // check the number of dimensions and type
    vx_size num_dims;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &num_dims, sizeof(num_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    // copy starts index
    status = vxMapTensorPatch((vx_tensor)parameters[2], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
        return -1;
    }

    for(int i=0; i<num_dims; i++) {
        starts.push_back((int)ptr[i]);
    }

    status = vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);
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

    for(int i=0; i<num_dims; i++) {
        ends.push_back((int)ptr[i]);
    }

    status = vxUnmapTensorPatch((vx_tensor)parameters[3], map_id);
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

        for(int i=0; i<num_dims; i++) {
            axes.push_back((int)ptr[i]);
        }

        status = vxUnmapTensorPatch((vx_tensor)parameters[4], map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for axes tensor (" << status << ")" << std::endl;
            return -1;
        }
    }
    else {
        for(int i=0; i<num_dims; i++)
            axes.push_back(i);
    }

    // copy steps index
    if(parameters[5]) {
        status = vxMapTensorPatch((vx_tensor)parameters[5], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
            return -1;
        }

        for(int i=0; i<num_dims; i++) {
            steps.push_back((int)ptr[i]);
        }

        status = vxUnmapTensorPatch((vx_tensor)parameters[5], map_id);
        if(status) {
            std::cerr << "ERROR: vxUnmapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
            return -1;
        }
    }
    else {
        for(int i=0; i<num_dims; i++)
            steps.push_back(1);
    }

    // for (auto itr = starts.begin(); itr!=starts.end(); itr++) {
    //     printf ("the starts value is %d\n", *itr);
    // }
    // for (auto itr = ends.begin(); itr!=ends.end(); itr++) {
    //     printf ("the ends value is %d\n", *itr);
    // }
    // for (auto itr = axes.begin(); itr!=axes.end(); itr++) {
    //     printf ("the axes value is %d\n", *itr);
    // }
    // for (auto itr = steps.begin(); itr!=steps.end(); itr++) {
    //     printf ("the steps value is %d\n", *itr);
    // }
    //std::cout << "calculating to " << num_dims << std::endl;
    
    vx_context context = vxCreateContext();
    status = vxGetStatus((vx_reference)context);
    if(status) {
        std::cerr << "ERROR: vxCreateContext() failed (" << status << ")" << std::endl;
        return -1;
    }

    vx_graph graph = vxCreateGraph(context);
    status = vxGetStatus((vx_reference)graph);
    if(status) {
        std::cerr << "ERROR: vxCreateGraph() failed (" << status << ")" << std::endl;
        return -1;
    }

    std::vector<vx_tensor> input_tensor(num_dims);
    std::vector<vx_tensor> indices_tensor(num_dims);
    std::vector<vx_tensor> output_tensor(num_dims);

    input_tensor.push_back((vx_tensor)parameters[0]);

    // calculate and create indices tensor
    std::vector<std::vector<int>> indices(num_dims);
    for(int i=0; i<num_dims; i++) {
        int index = starts[i];
        std::cout << "iteration" << i << std::endl;
        while (index < ends[i]) {
            std::cout << "inserting index " << index << std::endl;
            indices[i].push_back(index);
            index += steps[i];
        }

        vx_size dim[1] = {indices[i].size()};
        indices_tensor[i] = vxCreateTensor(context, 1, dim, type, 0);
        status = vxMapTensorPatch(indices_tensor[i], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for indices tensor(" << status << ")" << std::endl;
            return -1;
        }
        for(auto itr = indices[i].begin(); itr != indices[i].end(); itr++) {
            std::cout << "copying value: " << *itr << std::endl;
            *ptr++ = *itr;
        }
        //status = vxUnmapTensorPatch(indices_tensor[i], map_id);
    }

    // calculate output dims and create output tensor
    vx_size tmp_num_dims = input_num_dims;
    vx_size *tmp_dims = input_dims;

    for(int i=0; i<(num_dims-1); i++) {
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

        for (int k=0; k<out_dim_rank; k++) {
            std::cout << "out dim " << k << ": " << out_dims[k] << std::endl;
        }

        input_tensor[i+1] = vxCreateVirtualTensor(graph, out_dim_rank, out_dims, input_type, 0); 
        output_tensor[i] = vxCreateVirtualTensor(graph, out_dim_rank, out_dims, input_type, 0);

        tmp_num_dims = out_dim_rank;
        tmp_dims = out_dims;
    }
    
    output_tensor.push_back((vx_tensor)parameters[1]);

    for (int i=0; i<num_dims; i++) {
        vxGatherLayer(graph, input_tensor[i], indices_tensor[i], output_tensor[i], axes[i]);
    }
    
    status = vxVerifyGraph(graph);
    if(status) {
        std::cerr << "ERROR: vxVerifyGraph() failed (" << status << ")" << std::endl;
        return -1;
    }
    
    status = vxProcessGraph(graph);
    if(status) {
        std::cerr << "ERROR: vxProcessGraph() failed (" << status << ")" << std::endl;
        return -1;
    }

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

