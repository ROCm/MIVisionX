/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "rocal_api_types.h"
#include "rocal_api.h"
#include "rocal_api_parameters.h"
#include "rocal_api_data_loaders.h"
#include "rocal_api_augmentation.h"
#include "rocal_api_data_transfer.h"
#include "rocal_api_info.h"
namespace py = pybind11;

using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");
namespace pybind11
{
    namespace detail
    {
        constexpr int NPY_FLOAT16 = 23;
        // Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
        template <> struct npy_format_descriptor<float16>
        {
            static constexpr auto name = _("float16");
            static pybind11::dtype dtype()
            {
                handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
                return reinterpret_borrow<pybind11::dtype>(ptr);
            }
        };
    }
}  // namespace pybind11::detail
namespace rocal{
    using namespace pybind11::literals; // NOLINT
    // PYBIND11_MODULE(rocal_backend_impl, m) {
    static void *ctypes_void_ptr(const py::object &object)
    {
        auto ptr_as_int = getattr(object, "value", py::none());
        if (ptr_as_int.is_none())
        {
            return nullptr;
        }
        void *ptr = PyLong_AsVoidPtr(ptr_as_int.ptr());
        return ptr;
    }

    py::object wrapper(RocalContext context, py::array_t<unsigned char> array)
    {
        auto buf = array.request();
        unsigned char* ptr = (unsigned char*) buf.ptr;
        // call pure C++ function
        int status = rocalCopyToOutput(context,ptr,buf.size);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_image_name_length(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        int length =rocalGetImageNameLen(context,ptr);
        return py::cast(length);
    }

    py::object wrapper_image_name(RocalContext context,  int array_len)
    {
        py::array_t<char> array;
        auto buf = array.request();
        char* ptr = (char*) buf.ptr;
        ptr = (char *)calloc(array_len, sizeof(char));
        // call pure C++ function
        rocalGetImageName(context,ptr);
        std::string s(ptr);
        free(ptr);
        return py::bytes(s);
    }

    py::object wrapper_tensor(RocalContext context, py::object p,
                                RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto ptr = ctypes_void_ptr(p);
        // call pure C++ function

        int status = rocalCopyToOutputTensor(context, ptr, tensor_format, tensor_output_type, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_tensor32(RocalContext context, py::array_t<float> array,
                                RocalTensorLayout tensor_format, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto buf = array.request();
        float* ptr = (float*) buf.ptr;
        // call pure C++ function
        int status = rocalCopyToOutputTensor32(context, ptr, tensor_format, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_tensor16(RocalContext context, py::array_t<float16> array,
                                RocalTensorLayout tensor_format, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto buf = array.request();
        float16* ptr = (float16*) buf.ptr;
        // call pure C++ function
        int status = rocalCopyToOutputTensor16(context, ptr, tensor_format, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_label_copy(RocalContext context, py::object p)
    {
        auto ptr = ctypes_void_ptr(p);
        // call pure C++ function
        rocalGetImageLabels(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_image_id(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        rocalGetImageId(context,ptr);
        return py::cast<py::none>(Py_None);
    }
    py::object wrapper_labels_BB_count_copy(RocalContext context, py::array_t<int> array)

    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        int count =rocalGetBoundingBoxCount(context,ptr);
        return py::cast(count);
    }


    py::object wrapper_BB_label_copy(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        rocalGetBoundingBoxLabel(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_encoded_bbox_label(RocalContext context, py::array_t<float>bboxes_array, py::array_t<int>labels_array)
    {
        auto bboxes_buf = bboxes_array.request();
        float* bboxes_ptr = (float*) bboxes_buf.ptr;
        auto labels_buf = labels_array.request();
        int* labels_ptr = (int*) labels_buf.ptr;
        // call pure C++ function
        rocalCopyEncodedBoxesAndLables(context, bboxes_ptr , labels_ptr);
        return py::cast<py::none>(Py_None);
    }

    std::pair<py::array_t<float>, py::array_t<int>>  wrapper_get_encoded_bbox_label(RocalContext context, int batch_size, int num_anchors)
    {
        float* bboxes_buf_ptr; int* labels_buf_ptr;
        // call pure C++ function
        rocalGetEncodedBoxesAndLables(context, &bboxes_buf_ptr, &labels_buf_ptr, num_anchors*batch_size);
        // create numpy arrays for boxes and labels tensor from the returned ptr
        // no need to free the memory as this is freed by c++ lib
        py::array_t<float> bboxes_array = py::array_t<float>(
                                                          {batch_size, num_anchors, 4},
                                                          {4*sizeof(float)*num_anchors, 4*sizeof(float), sizeof(float)},
                                                          bboxes_buf_ptr,
                                                          py::cast<py::none>(Py_None));
        py::array_t<int> labels_array = py::array_t<int>(
                                                          {batch_size, num_anchors},
                                                          {num_anchors*sizeof(int), sizeof(int)},
                                                          labels_buf_ptr,
                                                          py::cast<py::none>(Py_None));

        return std::make_pair(bboxes_array, labels_array);
    }


    py::object wrapper_BB_cord_copy(RocalContext context, py::array_t<float> array)
    {
        auto buf = array.request();
        float* ptr = (float*) buf.ptr;
        // call pure C++ function
        rocalGetBoundingBoxCords(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_img_sizes_copy(RocalContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        rocalGetImageSizes(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_one_hot_label_copy(RocalContext context, py::object p , unsigned numOfClasses, int dest)
    {
        auto ptr = ctypes_void_ptr(p);
        // call pure C++ function
        rocalGetOneHotImageLabels(context, ptr, numOfClasses, dest);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_random_bbox_crop(RocalContext context, bool all_boxes_overlap, bool no_crop, RocalFloatParam p_aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, RocalFloatParam p_scaling, int total_num_attempts )
    {
        // call pure C++ function
        rocalRandomBBoxCrop(context, all_boxes_overlap, no_crop, p_aspect_ratio, has_shape, crop_width, crop_height, num_attempts, p_scaling, total_num_attempts);
        return py::cast<py::none>(Py_None);
    }


    PYBIND11_MODULE(rocal_pybind, m) {
        m.doc() = "Python bindings for the C++ portions of ROCAL";
        // rocal_api.h
        m.def("rocalCreate",&rocalCreate,"Creates context with the arguments sent and returns it",
                py::return_value_policy::reference,
                py::arg("batch_size"),
                py::arg("affinity"),
                py::arg("gpu_id") = 0,
                py::arg("cpu_thread_count") = 1,
                py::arg("prefetch_queue_depth") = 3,
                py::arg("output_data_type") = 0);
        m.def("rocalVerify",&rocalVerify);
        m.def("rocalRun",&rocalRun);
        m.def("rocalRelease",&rocalRelease);
        // rocal_api_types.h
        py::class_<TimingInfo>(m, "TimingInfo")
            .def_readwrite("load_time",&TimingInfo::load_time)
            .def_readwrite("decode_time",&TimingInfo::decode_time)
            .def_readwrite("process_time",&TimingInfo::process_time)
            .def_readwrite("transfer_time",&TimingInfo::transfer_time);
        py::module types_m = m.def_submodule("types");
        types_m.doc() = "Datatypes and options used by ROCAL";
        py::enum_<RocalStatus>(types_m, "RocalStatus", "Status info")
            .value("OK",ROCAL_OK)
            .value("CONTEXT_INVALID",ROCAL_CONTEXT_INVALID)
            .value("RUNTIME_ERROR",ROCAL_RUNTIME_ERROR)
            .value("UPDATE_PARAMETER_FAILED",ROCAL_UPDATE_PARAMETER_FAILED)
            .value("INVALID_PARAMETER_TYPE",ROCAL_INVALID_PARAMETER_TYPE)
            .export_values();
        py::enum_<RocalProcessMode>(types_m,"RocalProcessMode","Processing mode")
            .value("GPU",ROCAL_PROCESS_GPU)
            .value("CPU",ROCAL_PROCESS_CPU)
            .export_values();
        py::enum_<RocalTensorOutputType>(types_m,"RocalTensorOutputType","Tensor types")
            .value("FLOAT",ROCAL_FP32)
            .value("FLOAT16",ROCAL_FP16)
            .export_values();
        py::enum_<RocalResizeScalingMode>(types_m,"RocalResizeScalingMode","Decode size policies")
            .value("SCALING_MODE_DEFAULT",ROCAL_SCALING_MODE_DEFAULT)
            .value("SCALING_MODE_STRETCH",ROCAL_SCALING_MODE_STRETCH)
            .value("SCALING_MODE_NOT_SMALLER",ROCAL_SCALING_MODE_NOT_SMALLER)
            .value("SCALING_MODE_NOT_LARGER",ROCAL_SCALING_MODE_NOT_LARGER)
            .export_values();
        py::enum_<RocalResizeInterpolationType>(types_m,"RocalResizeInterpolationType","Decode size policies")
            .value("NEAREST_NEIGHBOR_INTERPOLATION",ROCAL_NEAREST_NEIGHBOR_INTERPOLATION)
            .value("LINEAR_INTERPOLATION",ROCAL_LINEAR_INTERPOLATION)
            .value("CUBIC_INTERPOLATION",ROCAL_CUBIC_INTERPOLATION)
            .value("LANCZOS_INTERPOLATION",ROCAL_LANCZOS_INTERPOLATION)
            .value("GAUSSIAN_INTERPOLATION",ROCAL_GAUSSIAN_INTERPOLATION)
            .value("TRIANGULAR_INTERPOLATION",ROCAL_TRIANGULAR_INTERPOLATION)
            .export_values();
        py::enum_<RocalImageSizeEvaluationPolicy>(types_m,"RocalImageSizeEvaluationPolicy","Decode size policies")
            .value("MAX_SIZE",ROCAL_USE_MAX_SIZE)
            .value("USER_GIVEN_SIZE",ROCAL_USE_USER_GIVEN_SIZE)
            .value("MOST_FREQUENT_SIZE",ROCAL_USE_MOST_FREQUENT_SIZE)
            .value("MAX_SIZE_ORIG",ROCAL_USE_MAX_SIZE_RESTRICTED)
            .value("USER_GIVEN_SIZE_ORIG",ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED)
            .export_values();
        py::enum_<RocalImageColor>(types_m,"RocalImageColor","Image type")
            .value("RGB",ROCAL_COLOR_RGB24)
            .value("BGR",ROCAL_COLOR_BGR24)
            .value("GRAY",ROCAL_COLOR_U8)
            .value("RGB_PLANAR", ROCAL_COLOR_RGB_PLANAR)
            .export_values();
        py::enum_<RocalTensorLayout>(types_m,"RocalTensorLayout","Tensor layout type")
            .value("NHWC",ROCAL_NHWC)
            .value("NCHW",ROCAL_NCHW)
            .export_values();
        py::enum_<RocalDecodeDevice>(types_m,"RocalDecodeDevice","Decode device type")
            .value("HARDWARE_DECODE",ROCAL_HW_DECODE)
            .value("SOFTWARE_DECODE",ROCAL_SW_DECODE)
            .export_values();
        py::enum_<RocalDecoderType>(types_m,"RocalDecoderType", "Rocal Decoder Type")
            .value("DECODER_TJPEG",ROCAL_DECODER_TJPEG)
            .value("DECODER_OPENCV",ROCAL_DECODER_OPENCV)
            .value("DECODER_HW_JEPG",ROCAL_DECODER_HW_JEPG)
            .value("DECODER_VIDEO_FFMPEG_SW",ROCAL_DECODER_VIDEO_FFMPEG_SW)
            .value("DECODER_VIDEO_FFMPEG_HW",ROCAL_DECODER_VIDEO_FFMPEG_HW)
            .export_values();
        // rocal_api_info.h
        m.def("getOutputWidth",&rocalGetOutputWidth);
        m.def("getOutputHeight",&rocalGetOutputHeight);
        m.def("getOutputColorFormat",&rocalGetOutputColorFormat);
        m.def("getRemainingImages",&rocalGetRemainingImages);
        m.def("getOutputImageCount",&rocalGetAugmentationBranchCount);
        m.def("getImageWidth",&rocalGetImageWidth);
        m.def("getImageHeight",&rocalGetImageHeight);
        m.def("getImagePlanes",&rocalGetImagePlanes);
        m.def("getImageName",&wrapper_image_name);
        m.def("getImageId", &wrapper_image_id);
        m.def("getImageNameLen",&wrapper_image_name_length);
        m.def("getStatus",&rocalGetStatus);
        m.def("setOutputImages",&rocalSetOutputs);
        m.def("labelReader",&rocalCreateLabelReader);
        m.def("TFReader",&rocalCreateTFReader);
        m.def("TFReaderDetection",&rocalCreateTFReaderDetection);
        m.def("CaffeReader",&rocalCreateCaffeLMDBLabelReader);
        m.def("Caffe2Reader",&rocalCreateCaffe2LMDBLabelReader);
        m.def("CaffeReaderDetection",&rocalCreateCaffeLMDBReaderDetection);
        m.def("Caffe2ReaderDetection",&rocalCreateCaffe2LMDBReaderDetection);
        m.def("Cifar10LabelReader",&rocalCreateTextCifar10LabelReader);
        m.def("RandomBBoxCrop",&wrapper_random_bbox_crop);
        m.def("COCOReader",&rocalCreateCOCOReader);
        m.def("VideoMetaDataReader",&rocalCreateVideoLabelReader);
        m.def("getImageLabels",&wrapper_label_copy);
        m.def("getBBLabels",&wrapper_BB_label_copy);
        m.def("getBBCords",&wrapper_BB_cord_copy);
        m.def("rocalCopyEncodedBoxesAndLables",&wrapper_encoded_bbox_label);
        m.def("rocalGetEncodedBoxesAndLables",&wrapper_get_encoded_bbox_label);
        m.def("getImgSizes",&wrapper_img_sizes_copy);
        m.def("getBoundingBoxCount",&wrapper_labels_BB_count_copy);
        m.def("getOneHotEncodedLabels",&wrapper_one_hot_label_copy);
        m.def("isEmpty",&rocalIsEmpty);
        m.def("BoxEncoder",&rocalBoxEncoder);
        m.def("getTimingInfo",rocalGetTimingInfo);
        // rocal_api_parameter.h
        m.def("setSeed",&rocalSetSeed);
        m.def("getSeed",&rocalGetSeed);
        m.def("CreateIntUniformRand",&rocalCreateIntUniformRand);
        m.def("CreateFloatUniformRand",&rocalCreateFloatUniformRand);
        m.def("CreateIntRand",[](std::vector<int> values, std::vector<double> frequencies){
            return rocalCreateIntRand(values.data(), frequencies.data(), values.size());
        });
        m.def("CreateFloatRand",&rocalCreateFloatRand);
        m.def("CreateIntParameter",&rocalCreateIntParameter);
        m.def("CreateFloatParameter",&rocalCreateFloatParameter);
        m.def("UpdateIntRand", &rocalUpdateIntUniformRand);
        m.def("UpdateFloatRand", &rocalUpdateFloatUniformRand);
        m.def("UpdateIntParameter", &rocalUpdateIntParameter);
        m.def("UpdateFloatParameter", &rocalUpdateFloatParameter);
        m.def("GetIntValue",&rocalGetIntValue);
        m.def("GetFloatValue",&rocalGetFloatValue);
        // rocal_api_data_transfer.h
        m.def("rocalCopyToOutput",&wrapper);
        m.def("rocalCopyToOutputTensor",&wrapper_tensor);
        m.def("rocalCopyToOutputTensor32",&wrapper_tensor32);
        m.def("rocalCopyToOutputTensor16",&wrapper_tensor16);
        // rocal_api_data_loaders.h
        m.def("COCO_ImageDecoderSlice",&rocalJpegCOCOFileSourcePartial,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
         m.def("COCO_ImageDecoderSliceShard",&rocalJpegCOCOFileSourcePartialSingleShard,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        m.def("ImageDecoder",&rocalJpegFileSource,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference);
        m.def("ImageDecoderShard",&rocalJpegFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference);
        m.def("COCO_ImageDecoder",&rocalJpegCOCOFileSource,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("json_path"),
            py::arg("color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("COCO_ImageDecoderShard",&rocalJpegCOCOFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
	        py::arg("json_path"),
            py::arg("color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("TF_ImageDecoder",&rocalJpegTFRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("internal_shard_count"),
            py::arg("is_output"),
            py::arg("user_key_for_encoded"),
            py::arg("user_key_for_filename"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe_ImageDecoder",&rocalJpegCaffeLMDBRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe_ImageDecoderShard",&rocalJpegCaffeLMDBRecordSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe_ImageDecoderPartialShard",&rocalJpegCaffeLMDBRecordSourcePartialSingleShard);
        m.def("Caffe2_ImageDecoder",&rocalJpegCaffe2LMDBRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe2_ImageDecoderShard",&rocalJpegCaffe2LMDBRecordSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rocal_color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = ROCAL_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe2_ImageDecoderPartialShard",&rocalJpegCaffe2LMDBRecordSourcePartialSingleShard);
        m.def("FusedDecoderCrop",&rocalFusedJpegCrop,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference);
        m.def("FusedDecoderCropShard",&rocalFusedJpegCropSingleShard,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference);
        m.def("TF_ImageDecoderRaw",&rocalRawTFRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
              py::return_value_policy::reference,
              py::arg("p_context"),
              py::arg("source_path"),
              py::arg("user_key_for_encoded"),
              py::arg("user_key_for_filename"),
              py::arg("rocal_color_format"),
              py::arg("is_output"),
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("out_width") = 0,
              py::arg("out_height") = 0,
              py::arg("record_name_prefix") = "");
        m.def("Cifar10Decoder",&rocalRawCIFAR10Source,"Reads file from the source given and decodes it according to the policy only for TFRecords",
              py::return_value_policy::reference,
              py::arg("p_context"),
              py::arg("source_path"),
              py::arg("rocal_color_format"),
              py::arg("is_output"),
              py::arg("out_width") = 0,
              py::arg("out_height") = 0,
              py::arg("file_name_prefix") = "",
              py::arg("loop") = false);
        m.def("VideoDecoder",&rocalVideoFileSource,"Reads videos from the source given and decodes it according to the policy only for Videos as inputs",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("color_format"),
            py::arg("decoder_mode"),
            py::arg("shard_count"),
            py::arg("sequence_length"),
            py::arg("shuffle") = false,
            py::arg("is_output"),
            py::arg("loop") = false,
            py::arg("frame_step"),
            py::arg("frame_stride"),
            py::arg("file_list_frame_num") = false);
        m.def("VideoDecoderResize",&rocalVideoFileResize,"Reads videos from the source given and decodes it according to the policy only for Videos as inputs. Resizes the decoded frames to the dest width and height.",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("color_format"),
            py::arg("decoder_mode"),
            py::arg("shard_count"),
            py::arg("sequence_length"),
            py::arg("dest_width"),
            py::arg("dest_height"),
            py::arg("shuffle") = false,
            py::arg("is_output"),
            py::arg("loop") = false,
            py::arg("frame_step"),
            py::arg("frame_stride"),
            py::arg("file_list_frame_num") = false);
        m.def("SequenceReader",&rocalSequenceReader,"Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams.",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("color_format"),
            py::arg("shard_count"),
            py::arg("sequence_length"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("frame_step"),
            py::arg("frame_stride"));

        m.def("rocalResetLoaders",&rocalResetLoaders);
        // rocal_api_augmentation.h
        m.def("SSDRandomCrop",&rocalSSDRandomCrop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
	        py::arg("p_threshold"),
            py::arg("crop_area_factor") = NULL,
            py::arg("crop_aspect_ratio") = NULL,
            py::arg("crop_pos_x") = NULL,
            py::arg("crop_pos_y") = NULL,
            py::arg("num_of_attempts") = 20);
        m.def("Resize",&rocalResize, py::return_value_policy::reference);
        m.def("CropResize",&rocalCropResize,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("dest_width"),
            py::arg("dest_height"),
            py::arg("is_output"),
            py::arg("area") = NULL,
            py::arg("aspect_ratio") = NULL,
            py::arg("x_center_drift") = NULL,
            py::arg("y_center_drift") = NULL);
        m.def("rocalCopy",&rocalCopy,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("rocalNop",&rocalNop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("ColorTwist",&rocalColorTwist,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL,
            py::arg("beta") = NULL,
            py::arg("hue") = NULL,
            py::arg("sat") = NULL);
        m.def("ColorTwistFixed",&rocalColorTwistFixed);
        m.def("CropMirrorNormalize",&rocalCropMirrorNormalize,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("crop_depth"),
            py::arg("crop_height"),
            py::arg("crop_width"),
            py::arg("start_x"),
            py::arg("start_y"),
            py::arg("start_z"),
            py::arg("mean"),
            py::arg("std_dev"),
            py::arg("is_output"),
            py::arg("mirror") = NULL);
        m.def("Crop",&rocalCrop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("crop_width") = NULL,
            py::arg("crop_height") = NULL,
            py::arg("crop_depth") = NULL,
            py::arg("crop_pox_x") = NULL,
            py::arg("crop_pos_y") = NULL,
            py::arg("crop_pos_z") = NULL);
        m.def("CropFixed",&rocalCropFixed,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("crop_width"),
            py::arg("crop_height"),
            py::arg("crop_depth"),
            py::arg("is_output"),
            py::arg("crop_pox_x"),
            py::arg("crop_pos_y"),
            py::arg("crop_pos_z"));
        m.def("CenterCropFixed",&rocalCropCenterFixed,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("crop_width"),
            py::arg("crop_height"),
            py::arg("crop_depth"),
            py::arg("is_output"));
        m.def("Brightness",&rocalBrightness,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL,
            py::arg("beta") = NULL);
        m.def("Brightness",&rocalBrightness);
        m.def("GammaCorrection",&rocalGamma,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL);
        m.def("Rain",&rocalRain,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("rain_value") = NULL,
            py::arg("rain_width") = NULL,
            py::arg("rain_height") = NULL,
            py::arg("rain_transparency") = NULL);
        m.def("Snow",&rocalSnow,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("shift") = NULL);
        m.def("Blur",&rocalBlur,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL);
        m.def("Contrast",&rocalContrast,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("min") = NULL,
            py::arg("max") = NULL);
        m.def("Flip",&rocalFlip);
        m.def("Jitter",&rocalJitter,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("kernel_size") = NULL);
        m.def("Rotate",&rocalRotate,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("angle") = NULL,
            py::arg("dest_width") = 0,
            py::arg("dest_height") = 0);
        m.def("Hue",&rocalHue,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("hue") = NULL);
        m.def("Saturation",&rocalSaturation,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sat") = NULL);
        m.def("WarpAffine",&rocalWarpAffine,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("dest_width") = 0,
            py::arg("dest_height") = 0,
            py::arg("x0") = NULL,
            py::arg("x1") = NULL,
            py::arg("y0") = NULL,
            py::arg("y1") = NULL,
            py::arg("o0") = NULL,
            py::arg("o1") = NULL);
        m.def("Fog",&rocalFog,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("fog_value") = NULL);
        m.def("FishEye",&rocalFishEye,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("Vignette",&rocalVignette,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL);
        m.def("SnPNoise",&rocalSnPNoise,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL);
        m.def("Exposure",&rocalExposure,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("shift") = NULL);
        m.def("Pixelate",&rocalPixelate,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("Blend",&rocalBlend);
        m.def("Flip",&rocalFlip,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("flip_axis") = NULL);
        m.def("RandomCrop",&rocalRandomCrop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("crop_area_factor") = NULL,
            py::arg("crop_aspect_ratio") = NULL,
            py::arg("crop_pos_x") = NULL,
            py::arg("crop_pos_y") = NULL,
            py::arg("num_of_attempts") = 20);
        m.def("ColorTemp",&rocalColorTemp,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("adj_value_param") = NULL);
    }
}
