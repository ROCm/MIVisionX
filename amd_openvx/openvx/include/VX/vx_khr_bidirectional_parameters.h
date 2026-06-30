/*
 * Copyright (c) 2023-2026 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPENVX_BIDIRECTIONAL_H
#define OPENVX_BIDIRECTIONAL_H

/*!
 * \file
 * \brief The OpenVX Bidirectional Parameters extension API.
 */

#define OPENVX_KHR_BIDIRECTIONAL_PARAMETERS  "vx_khr_bidirectional_parameters"

#define OPENVX_KHR_BIDIRECTIONAL_OPTIONAL_KERNELS /* Remove if optional kernels are not implemented */

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief Extra enums.
 *
 * \ingroup group_parameter
 */

enum vx_bidirectional_enum_e
{
    VX_BIDIRECTIONAL = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_DIRECTION) + 0x2 /* Additional parameter direction enumeration */
};

#ifdef OPENVX_KHR_BIDIRECTIONAL_OPTIONAL_KERNELS

#ifndef VX_VERSION_1_1
/*! \brief [Graph] Creates an accumulate node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in,out] accum The accumulation image in <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \ingroup group_vision_function_accumulate
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>  
 */
VX_API_ENTRY vx_node VX_API_CALL vxAccumulateImageNode(vx_graph graph, vx_image input, vx_image accum);
#endif

/*! \brief [Graph] Creates a weighted accumulate node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] alpha The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar value with a value in the range of \f$ 0.0 \le \alpha \le 1.0 \f$.
 * \param [in,out] accum The <tt>\ref VX_DF_IMAGE_U8</tt> accumulation image.
 * \ingroup group_vision_function_accumulate_weighted
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>  
 */
VX_API_ENTRY vx_node VX_API_CALL vxAccumulateWeightedImageNodeX(vx_graph graph, vx_image input, vx_float32 alpha, vx_image accum);

/*! \brief [Graph] Creates an accumulate square node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] shift The input <tt>\ref VX_TYPE_UINT32</tt> with a value in the range of \f$ 0 \le shift \le 15 \f$.
 * \param [in,out] accum The accumulation image in <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \ingroup group_vision_function_accumulate_square
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>  
 */
VX_API_ENTRY vx_node VX_API_CALL vxAccumulateSquareImageNodeX(vx_graph graph, vx_image input, vx_uint32 shift, vx_image accum);
#endif

#ifdef  __cplusplus
}
#endif

#endif
