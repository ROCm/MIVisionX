/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*!
 * \file
 * \brief #1790: thin stubs for Khronos OpenVX extension APIs this
 * implementation does not support (neural network layers, user data
 * objects, XML/generic import-export, classifier models, buffer aliasing,
 * import-kernel-from-URL, custom tiling kernels, ICD platform enumeration,
 * and OpenCL context interop). MIVisionX targets the Vision conformance
 * feature set; these optional extensions have no backing implementation
 * anywhere in this codebase. Every function here is declared in an
 * installed header, so leaving it undefined causes a link error for any
 * caller (see https://github.com/ROCm/MIVisionX/issues/1790); each one is
 * defined here to compile, link, and fail cleanly at runtime instead
 * (VX_ERROR_NOT_SUPPORTED / NULL / vx_false_e, depending on its declared
 * return type), rather than silently claiming support it doesn't have.
 */

#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_khr_user_data_object.h>
#include <VX/vx_khr_xml.h>
#include <VX/vx_import.h>
#include <VX/vx_khr_class.h>
#include <VX/vx_khr_buffer_aliasing.h>
#include <VX/vx_khr_ix.h>
#include <VX/vx_khr_import_kernel.h>
#include <VX/vx_khr_tiling.h>
#include <VX/vx_khr_icd.h>
#include <VX/vx_khr_opencl_interop.h>

// ---------------------------------------------------------------------------
// vx_khr_nn.h -- Neural Network extension (8 functions). All vx_node-
// returning; no VX_KERNEL_* enum exists for any of these in vx_kernels.h, so
// a real node can never be created for them regardless of implementation.
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_node VX_API_CALL vxActivationLayer(vx_graph graph, vx_tensor inputs, vx_enum function, vx_float32 a, vx_float32 b, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)function; (void)a; (void)b; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxConvolutionLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases,
    const vx_nn_convolution_params_t *convolution_params, vx_size size_of_convolution_params, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)weights; (void)biases; (void)convolution_params; (void)size_of_convolution_params; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxDeconvolutionLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases,
    const vx_nn_deconvolution_params_t *deconvolution_params, vx_size size_of_deconv_params, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)weights; (void)biases; (void)deconvolution_params; (void)size_of_deconv_params; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxFullyConnectedLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases,
    vx_enum overflow_policy, vx_enum rounding_policy, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)weights; (void)biases; (void)overflow_policy; (void)rounding_policy; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxLocalResponseNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_enum type,
    vx_size normalization_size, vx_float32 alpha, vx_float32 beta, vx_float32 bias, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)type; (void)normalization_size; (void)alpha; (void)beta; (void)bias; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxPoolingLayer(vx_graph graph, vx_tensor inputs, vx_enum pooling_type,
    vx_size pooling_size_x, vx_size pooling_size_y, vx_size pooling_padding_x, vx_size pooling_padding_y,
    vx_enum rounding, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)pooling_type; (void)pooling_size_x; (void)pooling_size_y;
    (void)pooling_padding_x; (void)pooling_padding_y; (void)rounding; (void)outputs;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxROIPoolingLayer(vx_graph graph, vx_tensor input_data, vx_tensor input_rois,
    const vx_nn_roi_pool_params_t *roi_pool_params, vx_size size_of_roi_params, vx_tensor output_arr)
{
    (void)graph; (void)input_data; (void)input_rois; (void)roi_pool_params; (void)size_of_roi_params; (void)output_arr;
    return NULL;
}

VX_API_ENTRY vx_node VX_API_CALL vxSoftmaxLayer(vx_graph graph, vx_tensor inputs, vx_tensor outputs)
{
    (void)graph; (void)inputs; (void)outputs;
    return NULL;
}

// ---------------------------------------------------------------------------
// vx_khr_user_data_object.h -- User Data Object extension (8 functions).
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_user_data_object VX_API_CALL vxCreateUserDataObject(vx_context context, const vx_char *type_name, vx_size size, const void *ptr)
{
    (void)context; (void)type_name; (void)size; (void)ptr;
    return NULL;
}

VX_API_ENTRY vx_user_data_object VX_API_CALL vxCreateVirtualUserDataObject(vx_graph graph, const vx_char *type_name, vx_size size)
{
    (void)graph; (void)type_name; (void)size;
    return NULL;
}

VX_API_ENTRY vx_status VX_API_CALL vxReleaseUserDataObject(vx_user_data_object *user_data_object)
{
    (void)user_data_object;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxQueryUserDataObject(vx_user_data_object user_data_object, vx_enum attribute, void *ptr, vx_size size)
{
    (void)user_data_object; (void)attribute; (void)ptr; (void)size;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxSetUserDataObjectAttribute(vx_user_data_object user_data_object, vx_enum attribute, const void *ptr, vx_size size)
{
    (void)user_data_object; (void)attribute; (void)ptr; (void)size;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxCopyUserDataObject(vx_user_data_object user_data_object, vx_size offset, vx_size size, void *user_ptr, vx_enum usage, vx_enum user_mem_type)
{
    (void)user_data_object; (void)offset; (void)size; (void)user_ptr; (void)usage; (void)user_mem_type;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxMapUserDataObject(vx_user_data_object user_data_object, vx_size offset, vx_size size, vx_map_id *map_id, void **ptr, vx_enum usage, vx_enum mem_type, vx_uint32 flags)
{
    (void)user_data_object; (void)offset; (void)size; (void)map_id; (void)ptr; (void)usage; (void)mem_type; (void)flags;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxUnmapUserDataObject(vx_user_data_object user_data_object, vx_map_id map_id)
{
    (void)user_data_object; (void)map_id;
    return VX_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// vx_khr_xml.h (4) + vx_import.h (3) -- XML and generic graph import/export.
// vx_khr_xml.h's vxImportFromXML/vxGetImportReferenceByIndex/vxQueryImport
// share the same vx_import object model declared in vx_import.h.
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_status VX_API_CALL vxExportToXML(vx_context context, vx_char xmlfile[])
{
    (void)context; (void)xmlfile;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_import VX_API_CALL vxImportFromXML(vx_context context, vx_char xmlfile[])
{
    (void)context; (void)xmlfile;
    return NULL;
}

VX_API_ENTRY vx_reference VX_API_CALL vxGetImportReferenceByIndex(vx_import import, vx_uint32 index)
{
    (void)import; (void)index;
    return NULL;
}

VX_API_ENTRY vx_status VX_API_CALL vxQueryImport(vx_import import, vx_enum attribute, void *ptr, vx_size size)
{
    (void)import; (void)attribute; (void)ptr; (void)size;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_reference VX_API_CALL vxGetImportReferenceByName(vx_import import, const vx_char *name)
{
    (void)import; (void)name;
    return NULL;
}

VX_API_ENTRY vx_import VX_API_CALL vxImportObjectsFromMemory(vx_context context, vx_size numrefs, vx_reference *refs,
    const vx_enum *uses, const vx_uint8 *ptr, vx_size length)
{
    (void)context; (void)numrefs; (void)refs; (void)uses; (void)ptr; (void)length;
    return NULL;
}

VX_API_ENTRY vx_status VX_API_CALL vxReleaseImport(vx_import *import)
{
    (void)import;
    return VX_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// vx_khr_class.h -- Classifier extension (3 functions).
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_classifier_model VX_API_CALL vxImportClassifierModel(vx_context context, vx_enum format, const vx_uint8 *ptr, vx_size length)
{
    (void)context; (void)format; (void)ptr; (void)length;
    return NULL;
}

VX_API_ENTRY vx_status VX_API_CALL vxReleaseClassifierModel(vx_classifier_model *model)
{
    (void)model;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_node VX_API_CALL vxScanClassifierNode(vx_graph graph, vx_tensor input_feature_map, vx_classifier_model model,
    vx_int32 scanwindow_width, vx_int32 scanwindow_height, vx_int32 step_x, vx_int32 step_y,
    vx_array object_confidences, vx_array object_rectangles, vx_scalar num_objects)
{
    (void)graph; (void)input_feature_map; (void)model; (void)scanwindow_width; (void)scanwindow_height;
    (void)step_x; (void)step_y; (void)object_confidences; (void)object_rectangles; (void)num_objects;
    return NULL;
}

// ---------------------------------------------------------------------------
// vx_khr_buffer_aliasing.h -- Buffer Aliasing extension (2 functions).
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_status VX_API_CALL vxAliasParameterIndexHint(vx_kernel kernel, vx_uint32 parameter_index_a, vx_uint32 parameter_index_b, vx_enum processing_type)
{
    (void)kernel; (void)parameter_index_a; (void)parameter_index_b; (void)processing_type;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_bool VX_API_CALL vxIsParameterAliased(vx_node node, vx_uint32 parameter_index_a, vx_uint32 parameter_index_b)
{
    (void)node; (void)parameter_index_a; (void)parameter_index_b;
    return vx_false_e;
}

// ---------------------------------------------------------------------------
// vx_khr_ix.h -- generic memory export extension (2 functions).
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_status VX_API_CALL vxExportObjectsToMemory(vx_context context, vx_size numrefs, const vx_reference *refs,
    const vx_enum *uses, const vx_uint8 **ptr, vx_size *length)
{
    (void)context; (void)numrefs; (void)refs; (void)uses; (void)ptr; (void)length;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxReleaseExportedMemory(vx_context context, const vx_uint8 **ptr)
{
    (void)context; (void)ptr;
    return VX_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// vx_khr_import_kernel.h (1) and vx_khr_tiling.h (1) -- dynamic/tiling
// kernel registration extensions.
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_kernel VX_API_CALL vxImportKernelFromURL(vx_context context, const vx_char *type, const vx_char *url)
{
    (void)context; (void)type; (void)url;
    return NULL;
}

VX_API_ENTRY vx_kernel VX_API_CALL vxAddTilingKernel(vx_context context, vx_char name[VX_MAX_KERNEL_NAME], vx_enum enumeration,
    vx_tiling_kernel_f flexible_func_ptr, vx_tiling_kernel_f fast_func_ptr, vx_uint32 num_params,
    vx_kernel_input_validate_f input, vx_kernel_output_validate_f output)
{
    (void)context; (void)name; (void)enumeration; (void)flexible_func_ptr; (void)fast_func_ptr;
    (void)num_params; (void)input; (void)output;
    return NULL;
}

// ---------------------------------------------------------------------------
// vx_khr_icd.h -- ICD platform enumeration (2 of its 3 functions; the 3rd,
// vxCreateContextFromPlatform, is already implemented in vx_api.cpp). An ICD
// loader concept has no meaning for a HIP-only implementation.
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_status VX_API_CALL vxIcdGetPlatforms(vx_size capacity, vx_platform platform[], vx_size *pNumItems)
{
    (void)capacity; (void)platform;
    if (pNumItems)
        *pNumItems = 0;
    return VX_ERROR_NOT_SUPPORTED;
}

VX_API_ENTRY vx_status VX_API_CALL vxQueryPlatform(vx_platform platform, vx_enum attribute, void *ptr, vx_size size)
{
    (void)platform; (void)attribute; (void)ptr; (void)size;
    return VX_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// vx_khr_opencl_interop.h -- OpenCL context interop (1 function). This
// implementation is HIP-based, not OpenCL-based, so there is no OpenCL
// context to interoperate with.
// ---------------------------------------------------------------------------

VX_API_ENTRY vx_context VX_API_CALL vxCreateContextFromCL(cl_context opencl_context, cl_command_queue opencl_command_queue)
{
    (void)opencl_context; (void)opencl_command_queue;
    return NULL;
}
