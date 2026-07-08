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


#ifndef OPENVX_SWAP_MOVE_H
#define OPENVX_SWAP_MOVE_H

#include <VX/vx.h>

/* NOTE: The bidirectional parameters extension is required for the swap_move extension */
#define OPENVX_KHR_SWAP_MOVE  "vx_khr_swap_move"

#ifdef  __cplusplus
extern "C" {
#endif


/* NOTE: The list of kernels supported is added in vx_kernels.h
 */

/*!
 * \brief The name of the swap kernel
 */
#define VX_KERNEL_SWAP_NAME "org.khronos.openvx.swap"

/*!
 * \brief The name of the move kernel
 */

#define VX_KERNEL_MOVE_NAME "org.khronos.openvx.move"

/*! \brief Swap data from one object to another.
 * \note An implementation may optimize away the swap when virtual data objects are used.
 * \param [in] graph The reference to the graph.
 * \param [in, out] first The first data object.
 * \param [in, out] second The second data object with meta-data identical to the input data object.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation
 * should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_swap
 */
VX_API_ENTRY vx_node VX_API_CALL vxSwapNode(vx_graph graph, vx_reference first, vx_reference second);

/*! \brief Move data from one object to another. Same as Swap but second parameter is an output
 * \note An implementation may optimize away the move when virtual data objects are used.
 * \param [in] graph The reference to the graph.
 * \param [in, out] first The first data object.
 * \param [out] second The second data object with meta-data identical to the input data object.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation
 * should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_move
 */
VX_API_ENTRY vx_node VX_API_CALL vxMoveNode(vx_graph graph, vx_reference first, vx_reference second);

/*! \brief Swap data from one object to another.
 * \param [in] context The OpenVX context.
 * \param [in, out] first The first data object.
 * \param [in, out] second The second data object with meta-data identical to the input data object.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 * \ingroup group_vision_function_swap
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSwap(vx_context context, vx_reference first, vx_reference second);

/*! \brief Move data from one object to another.
 * \note In immediate mode identical to Swap.
 * \param [in]  context The OpenVX context.
 * \param [in, out] first The first data object.
 * \param [out] second The second data object with meta-data identical to the input data object.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 * \ingroup group_vision_function_move
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMove(vx_context context, vx_reference first, vx_reference second);


#ifdef  __cplusplus
}
#endif

#endif
