#ifndef __VX_AMD_MIGRAPHX_H__
#define __VX_AMD_MIGRAPHX_H__

#include <VX/vx.h>
#include <migraphx/migraphx.hpp>

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief [Graph] Creates a MIGrpahX Node.
 * \param [in] graph The handle to the graph.
 * \param [in] prog The MIGraphX program
 * \param [in] input the input tensor
 * \param [out] output the output tensor
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL amdMIGraphXnode(vx_graph graph, migraphx::program *prog, vx_tensor input, vx_tensor output);

/*! \brief [Graph] loads and compiles onnx models using the MIGraphX
 * \param [in] path The path to the onnx model for compilation
 * \param [out] prog The MIGraphX program
 * \param [out] input_num_of_dims the number of dimensions for the input tensor
 * \param [out] input_dims the dimension of the input tensor
 * \param [out] input_data_format the data type of the input tensor
 * \param [out] output_num_of_dims the number of dimensions for the output tensor
 * \param [out] output_dims the dimension of the output tensor
 * \param [out] output_num_of_dims the data type of the output tensor
 * \param [in] fp16q if true, compie the model in fp16
 * \param [in] int8q if true, compie the model in int8
 * \return <tt> vx_status</tt>.
 * \returns Any possible errors preventing a successful compilation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL amdMIGraphXcompile(const char *path, migraphx::program *prog,
    vx_size *input_num_of_dims, vx_size *input_dims, vx_enum *input_data_format,
    vx_size *output_num_of_dims, vx_size *output_dims, vx_enum *output_data_format,
    vx_bool fp16q, vx_bool int8q);

#ifdef  __cplusplus
}
#endif

#endif
