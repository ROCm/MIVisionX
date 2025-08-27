/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// No global interpreter guard; assume embedding context elsewhere

vx_enum getVxDataType(RpptDataType dataType) {
    switch (dataType) {
        case RpptDataType::F32: return VX_TYPE_FLOAT32;
        case RpptDataType::F16: return VX_TYPE_FLOAT16;
        case RpptDataType::U8:  return VX_TYPE_UINT8;
        case RpptDataType::I8:  return VX_TYPE_INT8;
        default: throw std::runtime_error("Unsupported RpptDataType");
    }
}

std::pair<std::string, size_t> get_numpy_type(vx_enum type) {
    switch (type) {
        case VX_TYPE_FLOAT32:
            return {py::format_descriptor<float>::format(), sizeof(float)};
        case VX_TYPE_FLOAT16:
            return {"e", sizeof(vx_float16)};
        case VX_TYPE_UINT8:
            return {py::format_descriptor<uint8_t>::format(), sizeof(uint8_t)};
        case VX_TYPE_INT8:
            return {py::format_descriptor<int8_t>::format(), sizeof(int8_t)};
        case VX_TYPE_UINT32:
            return {py::format_descriptor<uint32_t>::format(), sizeof(uint32_t)};
        case VX_TYPE_INT32:
            return {py::format_descriptor<int32_t>::format(), sizeof(int32_t)};
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

struct ExternalSourceLocalData {
    vxRppHandle *handle;
    vx_uint32 deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    RpptGenericDescPtr pSrcGenericDesc;
    RpptGenericDescPtr pDstGenericDesc;
    RpptROI *pSrcRoi;
    RpptRoiType roiType;
    vxTensorLayout inputLayout;
    vxTensorLayout outputLayout;
    vx_uint32 dtype;
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
    py::object python_function;
};

static vx_status VX_CALLBACK refreshExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num, ExternalSourceLocalData *data) {
    vx_status status = VX_SUCCESS;
    void *roi_tensor_ptr;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        return VX_ERROR_NOT_IMPLEMENTED;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr, sizeof(roi_tensor_ptr)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    data->pSrcRoi = reinterpret_cast<RpptROI *>(roi_tensor_ptr);
    if (data->inputLayout == VX_NFHWC || data->inputLayout == VX_NFCHW) {
        unsigned num_of_frames = data->inputTensorDims[1];
        for (int n = data->inputTensorDims[0] - 1; n >= 0; n--) {
            unsigned index = n * num_of_frames;
            for (unsigned f = 0; f < num_of_frames; f++) {
                data->pSrcRoi[index + f].xywhROI = data->pSrcRoi[n].xywhROI;
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK validateExternalSource(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    for (int idx : {4,5,6}) {
        STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[idx], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
        if (scalar_type != VX_TYPE_INT32)
            return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter #%d must be INT32\n", idx+1);
    }
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter #8 must be UINT32\n");

    size_t num_dims = 0;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims < 1) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "ExternalSource input tensor dims < 1\n");

    vx_uint8 fixed_point = 0;
    size_t dims[RPP_MAX_TENSOR_DIMS] = {0};
    vx_enum dtype = 0;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &dims, sizeof(dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &dtype, sizeof(dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point, sizeof(fixed_point)));

    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &dims, sizeof(dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &dtype, sizeof(dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point, sizeof(fixed_point)));
    return status;
}

static vx_status VX_CALLBACK processExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    vx_status return_status = VX_SUCCESS;
    ExternalSourceLocalData *data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshExternalSource(node, parameters, num, data);
    if (!data->pSrc)
        return VX_FAILURE;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU)
        return VX_ERROR_NOT_IMPLEMENTED;

    std::cerr << "calling python function\n";
    py::gil_scoped_acquire acquire;
    try {
        // Determine input numpy dtype and itemsize from RpptDesc
        auto [in_format, in_itemsize] = get_numpy_type(getVxDataType(data->pSrcGenericDesc->dataType));

        // Build batched shape: [batch, D1, D2, ...]
        size_t batch = data->inputTensorDims[0];
        std::vector<size_t> sample_shape;
        size_t sample_count = 1;
        for (size_t j = 1; j < data->pSrcGenericDesc->numDims; ++j) {
            sample_shape.push_back(data->inputTensorDims[j]);
            sample_count *= data->inputTensorDims[j];
        }

        std::vector<size_t> batched_shape;
        batched_shape.reserve(1 + sample_shape.size());
        batched_shape.push_back(batch);
        batched_shape.insert(batched_shape.end(), sample_shape.begin(), sample_shape.end());

        // Compute strides in bytes for batched array
        std::vector<size_t> batched_strides(batched_shape.size());
        if (!batched_strides.empty()) {
            batched_strides.back() = in_itemsize;
            for (int j = int(batched_strides.size()) - 2; j >= 0; --j) {
                batched_strides[j] = batched_strides[j+1] * batched_shape[j+1];
            }
        }

        // Create a numpy view for the entire batch without copying
        // Use capsule with no destructor since rocAL owns the memory
        py::capsule owner((void*)data->pSrc, [](void*){ /* no-op */ });
        py::array numpy_batch(
            py::dtype(in_format),
            batched_shape,
            batched_strides,
            data->pSrc,
            owner
        );

        // Call user python function with single batched array (the user expects batched arrays)
        py::object result_obj = data->python_function(numpy_batch);

        // Cast to numpy array and make contiguous to simplify copying
        py::array result_array = py::cast<py::array>(result_obj);
        py::object np = py::module::import("numpy");
        py::array result_contig = np.attr("ascontiguousarray")(result_array);
        py::buffer_info buf_info = result_contig.request();

        // Basic validations:
        if (buf_info.ndim != static_cast<int>(data->pDstGenericDesc->numDims)) {
            std::cerr << "Python function returned array with wrong number of dims\n";
            return VX_FAILURE;
        }
        if (static_cast<size_t>(buf_info.shape[0]) != batch) {
            std::cerr << "Python function returned array with wrong batch size\n";
            return VX_FAILURE;
        }
        // Check remaining dims against expected output tensor dims
        for (size_t j = 1; j < data->pDstGenericDesc->numDims; ++j) {
            size_t expected = data->outputTensorDims[j];
            size_t got = static_cast<size_t>(buf_info.shape[j]);
            if (expected != got) {
                std::cerr << "Python function returned mismatched output shape at dim " << j << " expected " << expected << " got " << got << "\n";
                return VX_FAILURE;
            }
        }

        // Verify dtype / itemsize with expected output
        auto [out_format, out_itemsize] = get_numpy_type(getVxDataType(data->pDstGenericDesc->dataType));
        if (static_cast<size_t>(buf_info.itemsize) != out_itemsize) {
            std::cerr << "Python function returned array with wrong dtype itemsize (expected " << out_itemsize << ", got " << buf_info.itemsize << ")\n";
            return VX_FAILURE;
        }

        // Calculate total bytes and copy into output buffer
        size_t total_elems = 1;
        for (auto &d : buf_info.shape) total_elems *= static_cast<size_t>(d);
        size_t total_bytes = total_elems * static_cast<size_t>(buf_info.itemsize);

        if (!data->pDst)
            return VX_FAILURE;

        // Copy contiguous bytes
        memcpy(data->pDst, buf_info.ptr, total_bytes);

    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return_status = VX_FAILURE;
    }
    std::cerr << "Finished calling python function\n";
    return return_status;
}

static vx_status VX_CALLBACK initializeExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ExternalSourceLocalData *data = new ExternalSourceLocalData;
    vx_int32 roi_type = 0, input_layout = 0, output_layout = 0;
    vx_enum input_tensor_dtype, output_tensor_dtype;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &input_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &output_layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &roi_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->roiType = static_cast<RpptRoiType>(roi_type);
    data->inputLayout = static_cast<vxTensorLayout>(input_layout);
    data->outputLayout = static_cast<vxTensorLayout>(output_layout);

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        hipError_t err = hipHostMalloc(&data->pSrcGenericDesc, sizeof(RpptGenericDesc), hipHostMallocDefault);
        if (err != hipSuccess)
            return ERRMSG(VX_ERROR_NOT_ALLOCATED, "refresh: hipHostMalloc of size %ld failed \n", sizeof(RpptGenericDesc));
        err = hipHostMalloc(&data->pDstGenericDesc, sizeof(RpptGenericDesc), hipHostMallocDefault);
        if (err != hipSuccess)
            return ERRMSG(VX_ERROR_NOT_ALLOCATED, "refresh: hipHostMalloc of size %ld failed \n", sizeof(RpptGenericDesc));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        data->pSrcGenericDesc = new RpptGenericDesc;
        data->pDstGenericDesc = new RpptGenericDesc;
    }

    // Querying for input tensor
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcGenericDesc->numDims, sizeof(data->pSrcGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
    data->pSrcGenericDesc->dataType = getRpptDataType(input_tensor_dtype);
    data->pSrcGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pSrcGenericDesc, data->inputLayout, data->inputTensorDims);

    // Querying for output tensor
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstGenericDesc->numDims, sizeof(data->pDstGenericDesc->numDims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstGenericDesc->numDims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
    data->pDstGenericDesc->dataType = getRpptDataType(output_tensor_dtype);
    data->pDstGenericDesc->offsetInBytes = 0;
    fillGenericDescriptionPtrfromDims(data->pDstGenericDesc, data->outputLayout, data->outputTensorDims);

    vx_int64 function_id = 0;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &function_id, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    {
        py::gil_scoped_acquire acquire;
        py::handle h(reinterpret_cast<PyObject*>(function_id)); h.inc_ref();
        data->python_function = py::reinterpret_steal<py::object>(h.ptr());
    }

    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[8], &data->dtype, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    refreshExternalSource(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->inputTensorDims[0], data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    std::cerr << "Initialized ExternalSource node\n";
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeExternalSource(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ExternalSourceLocalData *data = nullptr;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    {
        py::gil_scoped_acquire acquire;
        data->python_function.release();
    }
        if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(data->pSrcGenericDesc);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
        err = hipHostFree(data->pDstGenericDesc);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (data->pSrcGenericDesc) delete data->pSrcGenericDesc;
        if (data->pDstGenericDesc) delete data->pDstGenericDesc;
    }
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete data;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32 &supported_target_affinity) {
    AgoTargetAffinityInfo affinity;
    vxQueryContext(vxGetContext((vx_reference)graph), VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    supported_target_affinity = (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        ? AGO_TARGET_AFFINITY_GPU : AGO_TARGET_AFFINITY_CPU;
    return VX_SUCCESS;
}

vx_status ExternalSource_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.ExternalSource",
                                       VX_KERNEL_EXTERNALSOURCE,
                                       processExternalSource,
                                       9,
                                       validateExternalSource,
                                       initializeExternalSource,
                                       uninitializeExternalSource);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#endif
    amd_kernel_query_target_support_f query_f = query_target_support;
    STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_f, sizeof(query_f)));

    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    if (status != VX_SUCCESS) { vxRemoveKernel(kernel); return VX_FAILURE; }
    return status;
}
