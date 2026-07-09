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

#ifndef OPENVX_SAFE_CASTS_H
#define OPENVX_SAFE_CASTS_H
#define OPENVX_KHR_SAFE_CASTS  "vx_khr_safe_casts"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif
#define __MAKE_SAFE_DOWNCASTS__(typename, Name)\
/*! \brief safely get a new vx_reference for the given vx_##typename variable*/\
VX_API_ENTRY vx_reference VX_API_CALL vxGetRefFrom##Name(const vx_##typename *typename);\
\
/*! \brief safe cast a vx##typename to a generic vx_reference*/\
VX_API_ENTRY vx_reference VX_API_CALL vxCastRefFrom##Name(vx_##typename typename);\
\
/*! \brief safe cast a pointer to vx##typename to a pointer to vx_reference */\
VX_API_ENTRY vx_reference * VX_API_CALL vxCastRefFrom##Name##P(vx_##typename *p_##typename);\
\
/*! \brief safe cast a const pointer to vx##typename to a const pointer to vx_reference */\
VX_API_ENTRY const vx_reference * VX_API_CALL vxCastRefFrom##Name##ConstP(const vx_##typename *p_##typename);

#define __MAKE_SAFE_CASTS__(typename, Name) \
/*! \brief safely get a new vx_##typename or an error object from a vx_reference*/\
VX_API_ENTRY vx_##typename VX_API_CALL vxGetRefAs##Name(const vx_reference *ref, vx_status *status); \
\
/*! \brief safely upcast a vx_reference to a vx_##typename or an error object */\
VX_API_ENTRY vx_##typename VX_API_CALL vxCastRefAs##Name(vx_reference ref, vx_status *status);\
\
__MAKE_SAFE_DOWNCASTS__(typename, Name)

__MAKE_SAFE_CASTS__(array, Array)
__MAKE_SAFE_CASTS__(convolution, Convolution)
__MAKE_SAFE_DOWNCASTS__(context, Context)
__MAKE_SAFE_CASTS__(delay, Delay)
__MAKE_SAFE_CASTS__(distribution, Distribution)
__MAKE_SAFE_CASTS__(graph, Graph)
__MAKE_SAFE_CASTS__(image, Image)
#ifdef VX_TYPE_IMPORT
__MAKE_SAFE_DOWNCASTS__(import, Import)
#endif
__MAKE_SAFE_CASTS__(kernel, Kernel)
__MAKE_SAFE_CASTS__(lut, LUT)
__MAKE_SAFE_CASTS__(matrix, Matrix)
__MAKE_SAFE_DOWNCASTS__(meta_format, MetaFormat)
__MAKE_SAFE_CASTS__(node, Node)
__MAKE_SAFE_CASTS__(object_array, ObjectArray)
__MAKE_SAFE_CASTS__(parameter, Parameter)
__MAKE_SAFE_CASTS__(pyramid, Pyramid)
__MAKE_SAFE_CASTS__(remap, Remap)
__MAKE_SAFE_CASTS__(scalar, Scalar)
__MAKE_SAFE_CASTS__(tensor, Tensor)
__MAKE_SAFE_CASTS__(threshold, Threshold)
#ifdef VX_TYPE_USER_DATA_OBJECT
__MAKE_SAFE_CASTS__(user_data_object, UserDataObject)
#endif

#ifdef  __cplusplus
}
#endif

#endif
