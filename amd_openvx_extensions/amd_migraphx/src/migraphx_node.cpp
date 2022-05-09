#include "vx_amd_migraphx.h"
#include "kernels.h"

struct migraphXLocalData {
    unsigned char *prog;
    migraphx::program_parameters prog_params;
};

//! \brief The kernel execution.
static vx_status VX_CALLBACK amd_migraphx_node_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
     migraphXLocalData *data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    auto outputs = reinterpret_cast<migraphx::program *>(data->prog)->eval(data->prog_params);

    return VX_SUCCESS;
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK amd_migraphx_node_initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    migraphXLocalData *data = new migraphXLocalData;

    unsigned char *input_mem = NULL;
    unsigned char *output_mem = NULL;

    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[0], VX_SCALAR_BUFFER, &data->prog, sizeof(data->prog)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));

    auto param_shapes = reinterpret_cast<migraphx::program *>(data->prog)->get_parameter_shapes();
    auto input = param_shapes.names().back();
    auto output = param_shapes.names().front();
    data->prog_params.add(input, migraphx::argument(param_shapes[input], input_mem));
    data->prog_params.add(output, migraphx::argument(param_shapes[output], output_mem));

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK amd_migraphx_node_deinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    migraphXLocalData *data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        delete data;
    }

    return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK amd_migraphx_node_validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_enum type, out_type;
    vx_size in_num_dims, out_num_dims;
    vx_size output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &in_num_dims, sizeof(in_num_dims)));
    if (in_num_dims > 4) {
        return VX_ERROR_INVALID_TYPE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16) && (type != VX_TYPE_INT8)) {
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    if (out_num_dims > 4) {
        return VX_ERROR_INVALID_TYPE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16) && (type != VX_TYPE_INT8)) {
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status amd_vx_migraphx_node_publish(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.amd_migraphx_node", AMDOVX_KERNEL_AMD_MIGRAPHX,
                            amd_migraphx_node_kernel, 3, amd_migraphx_node_validate,
                            amd_migraphx_node_initialize, amd_migraphx_node_deinitialize);
    ERROR_CHECK_OBJECT(kernel);

    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // mivisionx program
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // input tensor
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // output tensor

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL amdMIGraphXnode(vx_graph graph, migraphx::program *prog, vx_tensor input, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_enum migraphx_prog_e = vxRegisterUserStruct(context, sizeof(*prog));
        vx_scalar migraphx_prog = vxCreateScalarWithSize(context, migraphx_prog_e, prog, sizeof(*prog));
        if (vxGetStatus((vx_reference)migraphx_prog) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)migraphx_prog,
                (vx_reference)input,
                (vx_reference)output,
            };
            node = createMIGraphXNode(graph, "com.amd.amd_migraphx_node", params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&migraphx_prog);
        }
    }
    return node;
}

VX_API_ENTRY vx_status VX_API_CALL amdMIGraphXcompile(const char *path, migraphx::program *prog,
    vx_size *input_num_of_dims, vx_size *input_dims, vx_enum *input_data_format,
    vx_size *output_num_of_dims, vx_size *output_dims, vx_enum *output_data_format,
    vx_bool fp16q, vx_bool int8q) {

    if ((prog == NULL) || (path == NULL) ||
       (input_num_of_dims == NULL) || (input_dims == NULL) || (input_data_format == NULL) ||
       (output_num_of_dims == NULL) || (output_dims == NULL) || (output_data_format == NULL)) {
           return VX_ERROR_INVALID_VALUE;
       }

    if (fp16q == true && int8q == true) {
        return VX_ERROR_INVALID_VALUE;
    }

    migraphx::onnx_options onnx_opts;
    *prog = parse_onnx(path, onnx_opts);
    migraphx::target targ = migraphx::target("gpu");

    if (fp16q) {
        migraphx::quantize_fp16(*prog);
    } else if (int8q) {
        migraphx::quantize_int8(*prog, targ, migraphx::quantize_int8_options());
    }

    migraphx::compile_options comp_opts;
    comp_opts.set_fast_math();
    prog->compile(targ, comp_opts);

    auto param_shapes = prog->get_parameter_shapes();
    auto input = param_shapes.names().back();
    auto output = param_shapes.names().front();

    std::vector<size_t> input_length = param_shapes[input].lengths();
    std::vector<size_t> output_length = param_shapes[output].lengths();

    *input_num_of_dims = input_length.size();
    for(auto i = 0; i < *input_num_of_dims; i++) {
        input_dims[i] = input_length[i];
    }

    *output_num_of_dims = output_length.size();
    for(auto i = 0; i < *output_num_of_dims; i++) {
        output_dims[i] = output_length[i];
    }

    *input_data_format = get_vx_type(param_shapes[input].type());
    *output_data_format = get_vx_type(param_shapes[output].type());

    return VX_SUCCESS;
    }
