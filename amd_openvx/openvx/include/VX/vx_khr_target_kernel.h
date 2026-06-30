/*
 * Copyright (c) 2025-2026 The Khronos Group Inc.
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


#ifndef VX_KHR_TARGET_KERNEL_H
#define VX_KHR_TARGET_KERNEL_H

#include <VX/vx.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief Interface to kernel APIs on target
 */

/*! \brief Extra enums.
 *
 * \ingroup group_vx_target_kernel
 */
enum vx_target_kernel_mem_pool_enum_e
{
    VX_ENUM_MEM_POOL     = 0x27, /*!< \brief Memory pool type enumeration. */
};

/*! \brief Type of memory pool
 *
 * See <tt>\ref vxMemTargetAlloc</tt> and <tt>\ref vxMemTargetFree</tt>
 *
 * \ingroup group_vx_target_kernel
 */
enum vx_target_kernel_mem_pool_type_e {

    /*! \brief Allocate memory in any memory pool
    \req [REQ-USERKERNEL-01]: VX_MEM_POOL_ANY
      */
    VX_MEM_POOL_ANY = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_MEM_POOL) + 0x0,
};

/*! \brief A generic opaque reference that encapsulates all data necessary for node execution 
 *         on a target hardware
 *         
 * \details 
 *          the object must be shared between the host and (possibly multiple) remote cores 
 *          in a manner that prevents concurrent read/write access
 * 
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-02]: vx_object_desc
 */
typedef struct _vx_object_desc *vx_object_desc;


/*! \brief Handle to kernel on a target
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-03]: vx_target_kernel
 */
typedef struct _vx_target_kernel *vx_target_kernel;

/*! \brief Handle to instance of a kernel on a target
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-04]: vx_target_kernel_instance
 */
typedef struct _vx_target_kernel_instance *vx_target_kernel_instance;

/*!
 * \brief Allocates memory of given size in the specified memory pool on a target
 *
 * \param [in] size      size of the memory to be allocated
 * \param [in] mem_pool  dedicated memory pool to allocate from \see vx_target_kernel_mem_pool_type_e
 * 
 * \details memory allocator function, which is intended 
 * for exclusive use by the user node. It allows the allocation of memory blocks of 
 * specific sizes from designated memory pools. The memory allocation should happen
 * during the node create phase
 *
 * \return Pointer to the allocated memory
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-05]: vxMemTargetAlloc
 */
VX_API_ENTRY void* VX_API_CALL vxMemTargetAlloc(vx_uint32 size, vx_enum mem_pool);


/*!
 * \brief Frees already allocated memory
 *
 * \param [in] ptr  Pointer to the memory
 * \param [in] size size of the memory to be freed
 * \param [in] mem_pool Memory pool from which the memory was allocated  \see vx_target_kernel_mem_pool_type_e
 * 
 * \details During release graph, the memory allocated on each remote 
 * core must be freed by the corresponding node during the node delete phase.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-06]: vxMemTargetFree
 */
VX_API_ENTRY void VX_API_CALL vxMemTargetFree(void *ptr, vx_uint32 size, vx_enum mem_pool);

/*!
 * \brief The target kernel callbacks prototype
 *
 * \details
 * For create_func, delete_func, and process_func callbacks
 * 'obj_desc' points to array of data object descriptor parameters
 *
 * \param [in] kernel The kernel for which the callback is called
 * \param [in] obj_desc Object descriptor passed as input to this callback
 * \param [in] num_params valid entries in object descriptor (obj_desc) array
 * \param [in] priv_arg additional private argument passed to the callback
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-07]: vx_target_kernel_f
 */
typedef vx_status(VX_CALLBACK *vx_target_kernel_f)(vx_target_kernel_instance kernel, 
                  vx_object_desc obj_desc[], vx_uint16 num_params, 
                  void *priv_arg);

/*!
 * \brief The target kernel callback for control command
 *
 * \details
 *        Used for control_func,
 *        'obj_desc' points to array of objects descriptors
 *        for control parameter. It could be any vx_(object)
 *
 * \param [in] kernel The kernel for which the callback is called
 * \param [in] node_cmd_id Command ID to be processed in the given node
 * \param [in] obj_desc Object descriptor passed as input to this callback
 * \param [in] num_params valid entries in object descriptor (obj_desc) array
 * \param [in] priv_arg additional private argument passed to the callback
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-08]: vx_target_kernel_control_f
 */
typedef vx_status(VX_CALLBACK *vx_target_kernel_control_f)(
                  vx_target_kernel_instance kernel, vx_uint32 node_cmd_id,
                  vx_object_desc obj_desc[], vx_uint16 num_params, void *priv_arg);

/*!
 * \brief Allows users to add native kernels implementation to specific targets
 *
 * \details This is different from vxAddUserKernel() in that this is called
 *          on the target CPU and it allows users to implement plugin specific kernels
 *          An equivalent vxAddUserKernel is typically called to pair the target
 *          kernel with OpenVX user kernel.
 *
 *          Same as vxAddTargetKernelByName except that it take a kernel_id as input
 *          instead of a string name
 *
 * \param [in] kernel_id      Unique identifier for the kernel, based on the vx_kernel_e enumeration
 * \param [in] target_name    Name of the target
 * \param [in] process_func   Function pointer for the kernel processing function
 * \param [in] create_func    Function pointer for the kernel creation function
 * \param [in] delete_func    Function pointer for the kernel deletion function
 * \param [in] control_func   Function pointer for the kernel control function
 * \param [in] priv_arg       Private argument passed to the kernel
 *
 * \return A target kernel reference.
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-09]: vxAddTargetKernel
 *
 */
VX_API_ENTRY vx_target_kernel VX_API_CALL vxAddTargetKernel(
                              vx_enum kernel_id,
                              const vx_char *target_name,
                              vx_target_kernel_f process_func,
                              vx_target_kernel_f create_func,
                              vx_target_kernel_f delete_func,
                              vx_target_kernel_control_f control_func,
                              void *priv_arg);

/*!
 * \brief Allows users to add native kernels implementation to specific targets
 *
 * \details This is different from vxAddUserKernel() in that this is called
 *          on the target CPU and it allows users to implement plugin specific kernels
 *          An equivalent vxAddUserKernel is typically called to pair the target
 *          kernel with OpenVX user kernel.
 *
 *          Same as vxAddTargetKernel except that it take a string name as input
 *          instead of kernel_id
 *
 *          Important Note: The user must ensure all kernel names are unique on a given core.
 *
 * \param [in] kernel_name    Name of the target kernel
 * \param [in] target_name    Name of the target
 * \param [in] process_func   Function pointer for the kernel processing function
 * \param [in] create_func    Function pointer for the kernel creation function
 * \param [in] delete_func    Function pointer for the kernel deletion function
 * \param [in] control_func   Function pointer for the kernel control function
 * \param [in] priv_arg       Private argument passed to the kernel
 *
 * \return A target kernel reference.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-10]: vxAddTargetKernelByName
 *
 */
VX_API_ENTRY vx_target_kernel VX_API_CALL vxAddTargetKernelByName(
                              const vx_char *kernel_name,
                              const vx_char *target_name,
                              vx_target_kernel_f process_func,
                              vx_target_kernel_f create_func,
                              vx_target_kernel_f delete_func,
                              vx_target_kernel_control_f control_func,
                              void *priv_arg);

/*! 
 * \brief Allows users to remove a user kernel
 *
 * \param [in] target_kernel  Handle to the target kernel to be removed
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-11]: vxRemoveTargetKernel
 *
 */
VX_API_ENTRY vx_status VX_API_CALL vxRemoveTargetKernel(
                       vx_target_kernel target_kernel);

/*! \brief Allows users to remove a user kernel from a specific target
 *         by providing a kernel name and a target name
 *
 * \param [in] kernel_name    Name of the target kernel
 * \param [in] target_name    Name of the target
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-12]: vxRemoveTargetKernelByName
 *
 */
VX_API_ENTRY vx_status VX_API_CALL vxRemoveTargetKernelByName(
                       const vx_char *kernel_name,
                       const vx_char *target_name);

/*!
 * \brief Associate a kernel context or handle with a target kernel instance
 *        Typically set by the kernel function during the node create phase
 *
 *        The kernel context is typically a buffer containing a kernel
 *        specific data structure which may include pointers to locally
 *        allocated memory and/or parameters that need to be shared between
 *        kernel callbacks.
 * 
 *  \param [in] target_kernel_instance    Target Kernel Instance
 *  \param [in] kernel_context            Pointer to the kernel context to be set
 *  \param [in] kernel_context_size       Size of the kernel context
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-13]: vxSetTargetKernelInstanceContext
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetTargetKernelInstanceContext(
                       vx_target_kernel_instance target_kernel_instance,
                       void *kernel_context, vx_uint32 kernel_context_size);

/*!
 * \brief Get a kernel context or handle with a target kernel instance
 *        Typically used by the kernel function during run, control, delete phases
 * 
 *        The kernel context is typically a buffer containing a kernel
 *        specific data structure which may include pointers to locally
 *        allocated memory and/or parameters that need to be shared between
 *        kernel callbacks.
 *
 * \param [in] target_kernel_instance    Handle to the target kernel instance from which to retrieve the context
 * \param [out] kernel_context           Pointer to the kernel context to be retrieved
 * \param [out] kernel_context_size      Size of the kernel context
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_vx_target_kernel
 * \req [REQ-USERKERNEL-14]: vxGetTargetKernelInstanceContext
 */
VX_API_ENTRY vx_status VX_API_CALL vxGetTargetKernelInstanceContext(
                       vx_target_kernel_instance target_kernel_instance,
                       void **kernel_context, vx_uint32 *kernel_context_size);

#ifdef __cplusplus
}
#endif

#endif
