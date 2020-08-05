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
extern "C" RaliMetaData RALI_API_CALL raliCreateTFReader(RaliContext rali_context, const char* source_path, bool is_output);

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
/// \param buf user buffer provided to be filled with output image name
/// \param image_idx the imageIdx in the output batch
extern "C" void RALI_API_CALL raliGetImageName(RaliContext rali_context,  char* buf, unsigned image_idx);

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \return The length of the name of the image associated with image_idx in the output batch
extern "C" unsigned RALI_API_CALL raliGetImageNameLen(RaliContext rali_context,  unsigned image_idx);

/// \param meta_data RaliMetaData object that contains info about the images and labels
/// \param buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
extern "C" void RALI_API_CALL raliGetImageLabels(RaliContext rali_context, int* buf);

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \return The size of the buffer needs to be provided by user to get bounding box info associated with image_idx in the output batch.
extern "C" unsigned RALI_API_CALL raliGetBoundingBoxCount(RaliContext rali_context, unsigned image_idx );

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \param buf The user's buffer that will be filled with bounding box info. It needs to be of size bounding box len returned by a call to the raliGetBoundingBoxCount
extern "C" void RALI_API_CALL raliGetBoundingBoxLabel(RaliContext rali_context, int* buf, unsigned image_idx );
extern "C" void RALI_API_CALL raliGetBoundingBoxCords(RaliContext rali_context, float* buf, unsigned image_idx );

///
/// \param rali_context
/// \param source_path path to the file that contains the metadata file
/// \param filename_prefix: look only files with prefix ( needed for cifar10)
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTextCifar10LabelReader(RaliContext rali_context, const char* source_path, const char* file_prefix);


#endif //MIVISIONX_RALI_API_META_DATA_H
