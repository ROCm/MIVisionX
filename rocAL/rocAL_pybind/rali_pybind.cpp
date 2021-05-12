#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>	
#include <pybind11/embed.h>	
#include <pybind11/eval.h>
#include "rali_api_types.h"
#include "rali_api.h"
#include "rali_api_parameters.h"
#include "rali_api_data_loaders.h"
#include "rali_api_augmentation.h"
#include "rali_api_data_transfer.h"
#include "rali_api_info.h"
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
namespace rali{
    using namespace pybind11::literals; // NOLINT
    // PYBIND11_MODULE(rali_backend_impl, m) {
    py::object wrapper(RaliContext context, py::array_t<unsigned char> array)
    {
        auto buf = array.request();
        unsigned char* ptr = (unsigned char*) buf.ptr;
        // call pure C++ function
        int status = raliCopyToOutput(context,ptr,buf.size);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_image_name_length(RaliContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        int length =raliGetImageNameLen(context,ptr);
        
        return py::cast(length);
    }

    py::object wrapper_image_name(RaliContext context,  int array_len)
    {
        py::array_t<char> array;
        auto buf = array.request();
        char* ptr = (char*) buf.ptr;
        ptr = (char *)malloc(array_len * sizeof(char));
        // call pure C++ function
        raliGetImageName(context,ptr);
        std::string s(ptr); 
        free(ptr);
        return py::bytes(s);
    }

    py::object wrapper_tensor32(RaliContext context, py::array_t<float> array,
                                RaliTensorLayout tensor_format, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto buf = array.request();
        float* ptr = (float*) buf.ptr;
        // call pure C++ function
        int status = raliCopyToOutputTensor32(context, ptr, tensor_format, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_tensor16(RaliContext context, py::array_t<float16> array,
                                RaliTensorLayout tensor_format, float multiplier0,
                                float multiplier1, float multiplier2, float offset0,
                                float offset1, float offset2,
                                bool reverse_channels)
    {
        auto buf = array.request();
        float16* ptr = (float16*) buf.ptr;
        // call pure C++ function
        int status = raliCopyToOutputTensor16(context, ptr, tensor_format, multiplier0,
                                              multiplier1, multiplier2, offset0,
                                              offset1, offset2, reverse_channels);
        // std::cerr<<"\n Copy failed with status :: "<<status;
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_label_copy(RaliContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        raliGetImageLabels(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_labels_BB_count_copy(RaliContext context, py::array_t<int> array)
    
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        int count =raliGetBoundingBoxCount(context,ptr);
        
        return py::cast(count);
    }


    py::object wrapper_BB_label_copy(RaliContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        raliGetBoundingBoxLabel(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_BB_cord_copy(RaliContext context, py::array_t<float> array)
    {
        auto buf = array.request();
        float* ptr = (float*) buf.ptr;
        // call pure C++ function
        raliGetBoundingBoxCords(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_img_sizes_copy(RaliContext context, py::array_t<int> array)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        raliGetImageSizes(context,ptr);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_one_hot_label_copy(RaliContext context, py::array_t<int> array , unsigned numOfClasses)
    {
        auto buf = array.request();
        int* ptr = (int*) buf.ptr;
        // call pure C++ function
        raliGetOneHotImageLabels(context, ptr, numOfClasses);
        return py::cast<py::none>(Py_None);
    }

    py::object wrapper_random_bbox_crop(RaliContext context, bool all_boxes_overlap, bool no_crop, RaliFloatParam p_aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attemps, RaliFloatParam p_scaling, int total_num_attempts )
    {
        // auto buf = array.request();
        // int* ptr = (int*) buf.ptr;
        // call pure C++ function
        raliRandomBBoxCrop(context, all_boxes_overlap, no_crop, p_aspect_ratio, has_shape, crop_width, crop_height, num_attemps, p_scaling, total_num_attempts);
        return py::cast<py::none>(Py_None);
    }


    PYBIND11_MODULE(rali_pybind, m) {
        m.doc() = "Python bindings for the C++ portions of RALI";
        // rali_api.h
        m.def("raliCreate",&raliCreate,"Creates context with the arguments sent and returns it",
                py::return_value_policy::reference,
                py::arg("batch_size"),
                py::arg("affinity"),
                py::arg("gpu_id") = 0, 
                py::arg("cpu_thread_count") = 1);
        m.def("raliVerify",&raliVerify);
        m.def("raliRun",&raliRun);
        m.def("raliRelease",&raliRelease);
        // rali_api_types.h
        py::class_<TimingInfo>(m, "TimingInfo")
            .def_readwrite("load_time",&TimingInfo::load_time)
            .def_readwrite("decode_time",&TimingInfo::decode_time)
            .def_readwrite("process_time",&TimingInfo::process_time)
            .def_readwrite("transfer_time",&TimingInfo::transfer_time);
        py::module types_m = m.def_submodule("types");
        types_m.doc() = "Datatypes and options used by RALI";
        py::enum_<RaliStatus>(types_m, "RaliStatus", "Status info")
            .value("OK",RALI_OK)
            .value("CONTEXT_INVALID",RALI_CONTEXT_INVALID)
            .value("RUNTIME_ERROR",RALI_RUNTIME_ERROR)
            .value("UPDATE_PARAMETER_FAILED",RALI_UPDATE_PARAMETER_FAILED)
            .value("INVALID_PARAMETER_TYPE",RALI_INVALID_PARAMETER_TYPE)
            .export_values();
        py::enum_<RaliProcessMode>(types_m,"RaliProcessMode","Processing mode")
            .value("GPU",RALI_PROCESS_GPU)
            .value("CPU",RALI_PROCESS_CPU)
            .export_values();
        py::enum_<RaliTensorOutputType>(types_m,"RaliTensorOutputType","Tensor types")
            .value("FLOAT",RALI_FP32)
            .value("FLOAT16",RALI_FP16)
            .export_values();
        py::enum_<RaliImageSizeEvaluationPolicy>(types_m,"RaliImageSizeEvaluationPolicy","Decode size policies")
            .value("MAX_SIZE",RALI_USE_MAX_SIZE)
            .value("USER_GIVEN_SIZE",RALI_USE_USER_GIVEN_SIZE)
            .value("MOST_FREQUENT_SIZE",RALI_USE_MOST_FREQUENT_SIZE)
            .value("MAX_SIZE_ORIG",RALI_USE_MAX_SIZE_RESTRICTED)
            .value("USER_GIVEN_SIZE_ORIG",RALI_USE_USER_GIVEN_SIZE_RESTRICTED)
            .export_values();
        py::enum_<RaliImageColor>(types_m,"RaliImageColor","Image type")
            .value("RGB",RALI_COLOR_RGB24)
            .value("BGR",RALI_COLOR_BGR24)
            .value("GRAY",RALI_COLOR_U8)
            .value("RGB_PLANAR", RALI_COLOR_RGB_PLANAR)
            .export_values();
        py::enum_<RaliTensorLayout>(types_m,"RaliTensorLayout","Tensor layout type")
            .value("NHWC",RALI_NHWC)
            .value("NCHW",RALI_NCHW)
            .export_values();
        // rali_api_info.h
        m.def("getOutputWidth",&raliGetOutputWidth);
        m.def("getOutputHeight",&raliGetOutputHeight);
        m.def("getOutputColorFormat",&raliGetOutputColorFormat);
        m.def("getRemainingImages",&raliGetRemainingImages);
        m.def("getOutputImageCount",&raliGetAugmentationBranchCount);
        m.def("getImageWidth",&raliGetImageWidth);
        m.def("getImageHeight",&raliGetImageHeight);
        m.def("getImagePlanes",&raliGetImagePlanes);
        m.def("getImageName",&wrapper_image_name);
        m.def("getImageNameLen",&wrapper_image_name_length);
        m.def("getStatus",&raliGetStatus);
        m.def("labelReader",&raliCreateLabelReader);
        m.def("TFReader",&raliCreateTFReader);
        m.def("TFReaderDetection",&raliCreateTFReaderDetection);
        m.def("CaffeReader",&raliCreateCaffeLMDBLabelReader);
        m.def("Caffe2Reader",&raliCreateCaffe2LMDBLabelReader);
        m.def("CaffeReaderDetection",&raliCreateCaffeLMDBReaderDetection);
        m.def("Caffe2ReaderDetection",&raliCreateCaffe2LMDBReaderDetection);
        m.def("Cifar10LabelReader",&raliCreateTextCifar10LabelReader);
        m.def("RandomBBoxCrop",&wrapper_random_bbox_crop);
        m.def("COCOReader",&raliCreateCOCOReader);
        m.def("getImageLabels",&wrapper_label_copy);
        m.def("getBBLabels",&wrapper_BB_label_copy);
        m.def("getBBCords",&wrapper_BB_cord_copy);
        m.def("getImgSizes",&wrapper_img_sizes_copy);
        m.def("getBoundingBoxCount",&wrapper_labels_BB_count_copy);
        m.def("getOneHotEncodedLabels",&wrapper_one_hot_label_copy );
        m.def("isEmpty",&raliIsEmpty);
        m.def("getTimingInfo",raliGetTimingInfo);
        // rali_api_parameter.h
        m.def("setSeed",&raliSetSeed);
        m.def("getSeed",&raliGetSeed);
        m.def("CreateIntUniformRand",&raliCreateIntUniformRand);
        m.def("CreateFloatUniformRand",&raliCreateFloatUniformRand);
        m.def("CreateIntRand",[](std::vector<int> values, std::vector<double> frequencies){
            return raliCreateIntRand(values.data(), frequencies.data(), values.size());
        });
        m.def("CreateFloatRand",&raliCreateFloatRand);
        m.def("CreateIntParameter",&raliCreateIntParameter);
        m.def("CreateFloatParameter",&raliCreateFloatParameter);
        m.def("UpdateIntRand", &raliUpdateIntUniformRand);
        m.def("UpdateFloatRand", &raliUpdateFloatUniformRand);
        m.def("UpdateIntParameter", &raliUpdateIntParameter);
        m.def("UpdateFloatParameter", &raliUpdateFloatParameter);
        // rali_api_data_transfer.h 
        m.def("raliCopyToOutput",&wrapper);
        m.def("raliCopyToOutputTensor32",&wrapper_tensor32);
        m.def("raliCopyToOutputTensor16",&wrapper_tensor16);
        // rali_api_data_loaders.h
         m.def("COCO_ImageDecoderSlice",&raliJpegCOCOFileSourcePartial,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("json_path"),
            py::arg("color_format"),	
            py::arg("num_threads"),	
            py::arg("is_output"),	
            py::arg("shuffle") = false,	
            py::arg("loop") = false,	
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,	
            py::arg("max_width") = 0,	
            py::arg("max_height") = 0,
            py::arg("area_factor") = NULL,	
            py::arg("aspect_ratio") = NULL,
            py::arg("x_drift_factor") = NULL,	
            py::arg("y_drift_factor") = NULL
            );
         m.def("COCO_ImageDecoderSliceShard",&raliJpegCOCOFileSourcePartialSingleShard,"Reads file from the source given and decodes it according to the policy",
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
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,	
            py::arg("max_width") = 0,	
            py::arg("max_height") = 0,
            py::arg("area_factor") = NULL,	
            py::arg("aspect_ratio") = NULL,
            py::arg("x_drift_factor") = NULL,	
            py::arg("y_drift_factor") = NULL
            );
        m.def("ImageDecoder",&raliJpegFileSource,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("color_format"),	
            py::arg("num_threads"),	
            py::arg("is_output"),	
            py::arg("shuffle") = false,	
            py::arg("loop") = false,	
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,	
            py::arg("max_width") = 0,	
            py::arg("max_height") = 0,
            py::arg("dec_type") = 0);	
        m.def("ImageDecoderShard",&raliJpegFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("COCO_ImageDecoder",&raliJpegCOCOFileSource,"Reads file from the source given and decodes it according to the policy",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("json_path"),
            py::arg("color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("COCO_ImageDecoderShard",&raliJpegCOCOFileSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
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
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("TF_ImageDecoder",&raliJpegTFRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",	
            py::return_value_policy::reference,	
            py::arg("p_context"),	
            py::arg("source_path"),	
            py::arg("rali_color_format"),
            py::arg("internal_shard_count"),
            py::arg("is_output"),
            py::arg("user_key_for_encoded"),
            py::arg("user_key_for_filename"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe_ImageDecoder",&raliJpegCaffeLMDBRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rali_color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe_ImageDecoderShard",&raliJpegCaffeLMDBRecordSourceSingleShard, "Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rali_color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe2_ImageDecoder",&raliJpegCaffe2LMDBRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rali_color_format"),
            py::arg("num_threads"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("Caffe2_ImageDecoderShard",&raliJpegCaffe2LMDBRecordSourceSingleShard,"Reads file from the source given and decodes it according to the shard id and number of shards",
            py::return_value_policy::reference,
            py::arg("p_context"),
            py::arg("source_path"),
            py::arg("rali_color_format"),
            py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0);
        m.def("FusedDecoderCrop",&raliFusedJpegCrop,"Reads file from the source and decodes them partially to output random crops",	
            py::return_value_policy::reference,	
            py::arg("context"),	
            py::arg("source_path"),	
            py::arg("color_format"),	
            py::arg("num_threads"),	
            py::arg("is_output"),	
            py::arg("shuffle") = false,	
            py::arg("loop") = false,	
            py::arg("decode_size_policy") = RALI_USE_MOST_FREQUENT_SIZE,	
            py::arg("max_width") = 0,	
            py::arg("max_height") = 0,	
            py::arg("area_factor") = NULL,	
            py::arg("aspect_ratio") = NULL,	
            py::arg("y_drift_factor") = NULL,	
            py::arg("x_drift_factor") = NULL);
        m.def("FusedDecoderCropShard",&raliFusedJpegCropSingleShard,"Reads file from the source and decodes them partially to output random crops",
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("source_path"),
            py::arg("color_format"),
	        py::arg("shard_id"),
            py::arg("shard_count"),
            py::arg("is_output"),
            py::arg("shuffle") = false,
            py::arg("loop") = false,
            py::arg("decode_size_policy") = RALI_USE_MAX_SIZE,
            py::arg("max_width") = 0,
            py::arg("max_height") = 0,
            py::arg("area_factor") = NULL,
            py::arg("aspect_ratio") = NULL,
            py::arg("y_drift_factor") = NULL,
            py::arg("x_drift_factor") = NULL);
        m.def("TF_ImageDecoderRaw",&raliRawTFRecordSource,"Reads file from the source given and decodes it according to the policy only for TFRecords",
              py::return_value_policy::reference,
              py::arg("p_context"),
              py::arg("source_path"),
              py::arg("user_key_for_encoded"),
              py::arg("user_key_for_filename"),
              py::arg("rali_color_format"),
              py::arg("is_output"),
              py::arg("shuffle") = false,
              py::arg("loop") = false,
              py::arg("out_width") = 0,
              py::arg("out_height") = 0,
              py::arg("record_name_prefix") = "");
        m.def("Cifar10Decoder",&raliRawCIFAR10Source,"Reads file from the source given and decodes it according to the policy only for TFRecords",
              py::return_value_policy::reference,
              py::arg("p_context"),
              py::arg("source_path"),
              py::arg("rali_color_format"),
              py::arg("is_output"),
              py::arg("out_width") = 0,
              py::arg("out_height") = 0,
              py::arg("file_name_prefix") = "",
              py::arg("loop") = false);

        m.def("raliResetLoaders",&raliResetLoaders);
        // rali_api_augmentation.h
        m.def("SSDRandomCrop",&raliSSDRandomCrop,
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
        m.def("Resize",&raliResize,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("dest_width"),
            py::arg("dest_height"),
            py::arg("is_output"));
        m.def("CropResize",&raliCropResize,
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
        m.def("raliCopy",&raliCopy,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("raliNop",&raliNop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("ColorTwist",&raliColorTwist,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL,
            py::arg("beta") = NULL,
            py::arg("hue") = NULL,
            py::arg("sat") = NULL);
        m.def("ColorTwistFixed",&raliColorTwistFixed);
        m.def("CropMirrorNormalize",&raliCropMirrorNormalize,
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
        m.def("Crop",&raliCrop,
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
        m.def("CropFixed",&raliCropFixed,
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
        m.def("CenterCropFixed",&raliCropCenterFixed,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("crop_width"),
            py::arg("crop_height"),
            py::arg("crop_depth"),
            py::arg("is_output"));
        m.def("Brightness",&raliBrightness,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL,
            py::arg("beta") = NULL);
        m.def("GammaCorrection",&raliGamma,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("alpha") = NULL);
        m.def("Rain",&raliRain,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("rain_value") = NULL,
            py::arg("rain_width") = NULL,
            py::arg("rain_height") = NULL,
            py::arg("rain_transparency") = NULL);
        m.def("Snow",&raliSnow,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("shift") = NULL);
        m.def("Blur",&raliBlur,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL);
        m.def("Contrast",&raliContrast,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("min") = NULL,
            py::arg("max") = NULL);
        m.def("Flip",&raliFlip);
        m.def("Jitter",&raliJitter,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("kernel_size") = NULL);
        m.def("Rotate",&raliRotate,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("angle") = NULL,
            py::arg("dest_width") = 0,
            py::arg("dest_height") = 0);
        m.def("Hue",&raliHue,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("hue") = NULL);
        m.def("Saturation",&raliSaturation,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sat") = NULL);     
        m.def("WarpAffine",&raliWarpAffine,
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
        m.def("Fog",&raliFog,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("fog_value") = NULL);
        m.def("FishEye",&raliFishEye,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("Vignette",&raliVignette,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL); 
        m.def("SnPNoise",&raliSnPNoise,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("sdev") = NULL);
        m.def("Exposure",&raliExposure,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("shift") = NULL);
        m.def("Pixelate",&raliPixelate,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"));
        m.def("Blend",&raliBlend,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input1"),
            py::arg("input2"),
            py::arg("is_output"),
            py::arg("ratio") = NULL);
        m.def("Flip",&raliFlip,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("flip_axis") = NULL);
        m.def("RandomCrop",&raliRandomCrop,
            py::return_value_policy::reference,
            py::arg("context"),
            py::arg("input"),
            py::arg("is_output"),
            py::arg("crop_area_factor") = NULL,
            py::arg("crop_aspect_ratio") = NULL,
            py::arg("crop_pos_x") = NULL,
            py::arg("crop_pos_y") = NULL,
            py::arg("num_of_attempts") = 20);
    }
}
