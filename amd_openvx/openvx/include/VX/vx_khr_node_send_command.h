/*
 * Copyright (c) 2012-2026 The Khronos Group Inc.
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

#ifndef OPENVX_NODE_SEND_COMMAND_H
#define OPENVX_NODE_SEND_COMMAND_H

/*!
 * \file
 * \brief The OpenVX Node Send Command extension API.
 */

#define OPENVX_KHR_NODE_SEND_COMMAND  "vx_khr_node_send_command"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief Node send command enumeration extension to the <tt>\ref vx_node_attribute_e</tt> enumeration type.
 * \ingroup group_node_send_command
 */
/*!
 * \brief Attribute which queries if a node has been optimized away, is not
 *        replicated, or the number of instances that exist if it is replicated
 */
#define VX_NODE_NUM_WITH_REPLICAS (VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0xA)

/*!
 * \brief Constant to indicate that a timeout parameter to an
 *        api should check and not wait for the call to complete
 *
 * \ingroup group_node_send_command
 */
#define VX_TIMEOUT_NO_WAIT          (0u)

/*!
 * \brief Constant to indicate that a timeout parameter to an
 *        api should wait forever for the call to complete
 *
 * \ingroup group_node_send_command
 */
#define VX_TIMEOUT_WAIT_FOREVER     (0xFFFFFFFFu)

/*!
 * \brief When sending a control command to a replicated node,
 *        this can be used to send control command to all replicated node.
 *
 * \ingroup group_node_send_command
 */
#define VX_CONTROL_CMD_SEND_TO_ALL_REPLICATED_NODES (0xFFFFFFFFu)

/*!
 * \brief The pointer to the control command function for the kernel.
 *        It is the responsibility of this callback to verify that the command passed and type and size of the references
 *        that are passed are valid prior to processing.
 *
 * \param [in]     node         The handle to the node that contains this kernel.
 * \param [in]     node_cmd_id  The kernel-specific command identifier for the node.
 * \param [in,out] ref          The array of references.
 * \param [in]     num_refs     The number of references in the refs array.
 *
 * \ingroup group_node_send_command
 */
typedef vx_status(VX_CALLBACK *vx_kernel_command_f)(vx_node node, vx_uint32 node_cmd_id, const vx_reference *ref, vx_uint32 num_refs);
    

/*! \brief vxAddCommandToKernel Allows users to add a control command callback to the kernel.
*          
 *
 * \details This function can be optionally called when creating a user kernel to register a command callback function with the kernel
 *
 * \param [in]   kernel            The reference to the kernel added with <tt>\ref vxAddUserKernel</tt>.
 * \param [in]   command_func_ptr  The process-local function pointer to be invoked when application calls <tt>\ref vxNodeSendCommand</tt>.
 *
 * \return A <tt>\ref vx_status_e</tt> enumerated value.
 * \retval VX_SUCCESS
 *  - Command callback is successfully set on kernel, any other value indicates failure
 * \retval VX_ERROR_INVALID_REFERENCE kernel is not a valid <tt>\ref vx_kernel</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS If the parameter is not valid for any reason.
 * \pre <tt>\ref vxAddUserKernel</tt>
 * 
 * \ingroup group_node_send_command
 */
VX_API_ENTRY vx_status VX_API_CALL vxAddCommandToKernel(vx_kernel kernel, vx_kernel_command_f command_func_ptr);

/*!
 * \brief vxNodeSendCommand Send node-specific control command
 *
 * \details This is used to send a specific control command to the specified node asynchronously.
 *          Not all nodes support commands, so refer to the node documentation for the specific control command.
 *          - Note that this blocks for at most 'timeout' milli-seconds; i.e. this returns either when the command
 *          execution finishes or when the timeout occurs, whichever occurs first.
 *          - Multiple commands can be sent to same or different nodes from different threads.
 *
 * \param [in]     node                 Reference of the node to which this command is to be sent.
 * \param [in]     replicate_nodex_idx  The index of the node replica to which the command is targeted.
 *                                      - In case of a non-replicated node this should be 0. For a replicated node this is the index
 *                                        of the node replica to which the command is targeted.
 *                                      - To send same command to all replicated nodes use VX_CONTROL_CMD_SEND_TO_ALL_REPLICATED_NODES
 * \param [in]     node_cmd_id          Node-specific control command id, refer to node-specific documentation
 * \param [in,out] ref[]                List of references, they can be any OpenVX object, created using the create API
 *                                      - This is a list of references, required as parameters for this control command on the given node.
 *                                        They are bidirectional parameters, can be used for INPUT, OUTPUT or both.
 *                                        Refer to node documentation to get details about the parameters required for given control command.
 * \param [in]     num_refs             Number of valid entries/references in ref[] array
 *                                      - Number of valid entries/references in ref[] array shall valid depending on the given control command
 * \param [in]     timeout              Timeout in units of msecs
 *                                      - Timeout in units of msecs, use VX_TIMEOUT_WAIT_FOREVER to wait forever
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors, any other value indicates failure
 * \retval VX_ERROR_INVALID_NODE if the node parameter is not a valid node.
 * \retval VX_ERROR_GRAPH_NOT_VERIFIED if the graph that the node is a part of is not yet verified
 * \retval VX_ERROR_INVALID_REFERENCE if one or more of the ref parameter references is not a valid reference.
 * \retval VX_ERROR_TIMEOUT if the operation exceeds the timeout value.
 * \retval VX_ERROR_INVALID_PARAMETERS if any of the other parameters are not valid or out of bounds
 * \pre <tt>\ref vxVerifyGraph</tt>
 *
 * \ingroup group_node_send_command
 */
VX_API_ENTRY vx_status VX_API_CALL vxNodeSendCommand(vx_node node,
    vx_uint32 replicate_nodex_idx, vx_uint32 node_cmd_id,
    vx_reference ref[], vx_uint32 num_refs, vx_uint32 timeout);

#ifdef  __cplusplus
}
#endif

#endif
//[*CNT-NC14*]
