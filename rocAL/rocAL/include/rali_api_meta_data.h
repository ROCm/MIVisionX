/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MIVISIONX_RALI_API_META_DATA_H
#define MIVISIONX_RALI_API_META_DATA_H
#include "rali_api_types.h"
///
/// \param rali_context
/// \param source_path path to the folder that contains the dataset or metadata file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateLabelReader(RaliContext rali_context, const char* source_path);

///
/// \param rali_context
/// \param source_path path to the coco json file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTFReader(RaliContext rali_context, const char* source_path, bool is_output,
    const char* user_key_for_label, const char* user_key_for_filename);


///
/// \param rali_context
/// \param source_path path to the coco json file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTFReaderDetection(RaliContext rali_context, const char* source_path, bool is_output,
    const char* user_key_for_label, const char* user_key_for_text, 
    const char* user_key_for_xmin, const char* user_key_for_ymin, const char* user_key_for_xmax, const char* user_key_for_ymax, 
    const char* user_key_for_filename);

///
/// \param rali_context
/// \param source_path path to the coco json file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateCOCOReader(RaliContext rali_context, const char* source_path, bool is_output);

///
/// \param rali_context
/// \param source_path path to the file that contains the metadata file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTextFileBasedLabelReader(RaliContext rali_context, const char* source_path);

///
/// \param rali_context
/// \param source_path path to the Caffe LMDB records for Classification
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateCaffeLMDBLabelReader(RaliContext rali_context, const char* source_path);

///
/// \param rali_context
/// \param source_path path to the Caffe LMDB records for Object Detection
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateCaffeLMDBReaderDetection(RaliContext rali_context, const char* source_path);

///
/// \param rali_context
/// \param source_path path to the Caffe2LMDB records for Classification
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors

extern "C" RaliMetaData RALI_API_CALL raliCreateCaffe2LMDBLabelReader(RaliContext rali_context, const char* source_path, bool is_output);

///
/// \param rali_context
/// \param source_path path to the Caffe2LMDB records for Object Detection
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors

extern "C" RaliMetaData RALI_API_CALL raliCreateCaffe2LMDBReaderDetection(RaliContext rali_context, const char* source_path, bool is_output);

///
/// \param rali_context
/// \param buf user buffer provided to be filled with output image names for images in the output batch.
extern "C" void RALI_API_CALL raliGetImageName(RaliContext rali_context,  char* buf);

///
/// \param rali_context
/// \param buf userbuffer provided to be filled with the length of the image names in the output batch
/// \return The size of the buffer needs to be provided by user to get the image names of the output batch
extern "C" unsigned RALI_API_CALL raliGetImageNameLen(RaliContext rali_context, int* buf);

/// \param meta_data RaliMetaData object that contains info about the images and labels
/// \param buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
extern "C" void RALI_API_CALL raliGetImageLabels(RaliContext rali_context, int* buf);

///
/// \param rali_context
/// \param buf The user's buffer that will be filled with number of object in the images.
/// \return The size of the buffer needs to be provided by user to get bounding box info for all images in the output batch.
extern "C" unsigned RALI_API_CALL raliGetBoundingBoxCount(RaliContext rali_context, int* buf);

///
/// \param rali_context
/// \param buf The user's buffer that will be filled with bounding box label info for the images in the output batch. It needs to be of size returned by a call to the raliGetBoundingBoxCount
extern "C" void RALI_API_CALL raliGetBoundingBoxLabel(RaliContext rali_context, int* buf);
extern "C" void RALI_API_CALL raliGetBoundingBoxCords(RaliContext rali_context, float* buf);

extern "C" void RALI_API_CALL raliGetImageSizes(RaliContext rali_context, int* buf);

///
/// \param rali_context
/// \param source_path path to the file that contains the metadata file
/// \param filename_prefix: look only files with prefix ( needed for cifar10)
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTextCifar10LabelReader(RaliContext rali_context, const char* source_path, const char* file_prefix);

/// \param meta_data RaliMetaData object that contains info about the images and labels
/// \param numOfClasses the number of classes for a image dataset
/// \param buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
extern "C" void RALI_API_CALL raliGetOneHotImageLabels(RaliContext rali_context,int *buf, int numOfClasses);
extern "C" void RALI_API_CALL raliRandomBBoxCrop(RaliContext p_context, bool all_boxes_overlap, bool no_crop, RaliFloatParam aspect_ratio = NULL, bool has_shape = false, int crop_width = 0, int crop_height = 0, int num_attempts = 1, RaliFloatParam scaling = NULL, int total_num_attempts = 0);


#endif //MIVISIONX_RALI_API_META_DATA_H
