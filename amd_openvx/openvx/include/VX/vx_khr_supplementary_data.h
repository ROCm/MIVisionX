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

#ifndef OPENVX_SUPPLEMENTARY_DATA_H
#define OPENVX_SUPPLEMENTARY_DATA_H

#include <VX/vx_khr_user_data_object.h>
#ifdef  __cplusplus
extern "C" {
#endif

/*!
 * \brief Set supplementary user data object for a reference using another object as the source
 *
 * \param [in] destination             The reference for which supplementary data is to be set
 * \param [in] source                  The source of the supplementary data
 * 
 * \return a vx_status value
 * \retval VX_SUCCESS                  The data was successfully set
 * \retval VX_ERROR_INVALID_REFERENCE  Either destination or source was not a valid reference
 * \retval VX_ERROR_NOT_SUPPORTED      The destination or source was of an unsupported type
 * \retval VX_ERROR_INVALID_TYPE       The function had previously been called for this destination with a source of a different type name
 * \retval VX_ERROR_OPTIMIZED_AWAY     Indicates that the destination or source has been optimized out of existence or is inaccessible
 * \retval VX_ERROR_INVALID_PARAMETERS Indicates that the combination of references supplied is not valid
 * \retval VX_ERROR_NO_MEMORY          Indicates that an internal or implicit allocation failed.
 * \retval VX_FAILURE                  Indicates a generic error code, used when no other describes the error
 * 
 * When the function is called for the first time for a given destination, a new reference of the type specified by the source is
 * allocated to hold the supplementary data and the data is copied from the source to this new object.
 * When the function is called subsequently for the same destination, the type of the source is checked against the type of the
 * existing supplementary data, and if it matches, data is copied from the source to the supplementary data of the destination.
 * The number of bytes copied is given by the valid size of the source and the valid size of the destination is set to this value.
 * If it does not match, the function returns the status VX_ERROR_INVALID_TYPE.
*/
VX_API_ENTRY vx_status VX_API_CALL vxSetSupplementaryUserDataObject(vx_reference destination, const vx_user_data_object source);

/*!
 * \brief Set supplementary and extend supplementary data object for a reference using another object and local data as the source
 * If no supplementary data exists in destination first create it using source as exemplar
 * If user_bytes >= source bytes:
 *   copy source_bytes constrained by valid_size from source to new supplementary data unless source is NULL
 *   copy (num_bytes - source_bytes) from (user_data + source_bytes) to (destination + source bytes) unless user_data is NULL
 *   set destination valid size equal to user_bytes
 * Else
 *   copy user_bytes from user_data to new supplementary data unless user_data is NULL
 *   copy (source_bytes - user_bytes) from (source + user_bytes) to (destination + user_bytes) unless source is NULL
 *   set destination valid size equal to source_bytes
 * \param [in] destination             The reference for which supplementary data is to be set
 * \param [in] source                  The source of the supplementary data or NULL
 * \param [in] user_data               Pointer to local data to copy or NULL
 * \param [in] source_bytes            Number of bytes to copy from the source object
 * \param [in] user_bytes              Number of bytes to copy from user_data
 * 
 * \return a vx_status value
 * \retval VX_SUCCESS                  The data was successfully set
 * \retval VX_ERROR_INVALID_REFERENCE  Either destination or source was not a valid reference
 * \retval VX_ERROR_INVALID_VALUE      user_bytes or source_bytes larger than the memory size of the vx_user_data_object type
 * \retval VX_ERROR_NOT_SUPPORTED      The destination or source was of an unsupported type
 * \retval VX_ERROR_INVALID_TYPE       The function had previously been called for this destination with a source of a different type name
 * \retval VX_ERROR_OPTIMIZED_AWAY     Indicates that the destination or source has been optimized out of existence or is inaccessible
 * \retval VX_ERROR_INVALID_PARAMETERS Indicates that the combination of references supplied is not valid
 * \retval VX_ERROR_NO_MEMORY          Indicates that an internal or implicit allocation failed.
 * \retval VX_FAILURE                  Indicates a generic error code, used when no other describes the error
 * 
*/
VX_API_ENTRY vx_status VX_API_CALL vxExtendSupplementaryUserDataObject(vx_reference destination, const vx_user_data_object source, const void *user_data, vx_uint32 source_bytes, vx_uint32 user_bytes);

/*!
 * \brief Return the supplementary user data object associated with a given reference, if it exists.
 *
 * \param [in]  ref             The reference to be queried, this may be of any reference type.
 * \param [in]  type_name       Pointer to the expected type name of the supplementary data
 * \param [out] status          Pointer to a status code indicating whether the function was successful or not, or NULL
 *
 * \return a vx_reference. Check vxGetStatus() to check for possible errors in creating the reference: 
 *  VX_ERROR_INVALID_TYPE      Indicates that the supplied type parameter(s) do not match.
 *  VX_ERROR_INVALID_REFERENCE Indicates that the reference provided is not valid.
 *  VX_ERROR_OPTIMIZED_AWAY    Indicates that the object referred to has been optimized out of existence or is inaccessible
 *  VX_ERROR_NO_MEMORY         Indicates that an internal or implicit allocation failed.
 *  VX_ERROR_NO_RESOURCES      Indicates that an internal or implicit resource can not be acquired
 *  VX_FAILURE                 Indicates a generic error code, used when no other describes the error
 *  VX_SUCCESS                 Reference is valid
 */
VX_API_ENTRY vx_user_data_object VX_API_CALL vxGetSupplementaryUserDataObject(vx_reference ref, const vx_char * type_name, vx_status *status);

#ifdef  __cplusplus
}
#endif

#endif
