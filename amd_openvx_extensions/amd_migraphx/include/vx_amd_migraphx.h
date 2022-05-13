#ifndef __VX_AMD_MIGRAPHX_H__
#define __VX_AMD_MIGRAPHX_H__

#include <VX/vx.h>
#include <migraphx/migraphx.hpp>

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief [Graph] Creates a MIGrpahX Node.
 * \param [in] graph The handle to the graph.
 * \param [in] path The path to the onnx file
 * \param [in] input the input tensor
 * \param [out] output the output tensor
 * \param [out] fp16q if true then the fp16 quantization will be appiled to the model
 * \param [out] int8q if true then the int8 quantization will be appiled to the model
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL amdMIGraphXnode(vx_graph graph, const vx_char *path, vx_tensor input, vx_tensor output, vx_bool fp16q = false, vx_bool int8q = false);

#ifdef  __cplusplus
}
#endif

#endif
